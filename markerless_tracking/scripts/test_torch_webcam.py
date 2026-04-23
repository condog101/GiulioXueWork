#!/usr/bin/env python3
"""
PyTorch webcam debug harness for the AlexNet2 RGB detector.

Twin of test_coreml_webcam.py. Runs the *original PyTorch* model end-to-end on
the Mac's built-in webcam, with the exact preprocessing / NMS / extend factor
the RealSense pipeline (markerless_tracking_standalone.py) uses. Intended for
A/B comparison against the Core ML harness: if boxes look correct here but
wrong in the Core ML run, the problem is in the conversion (most likely
batch-norm running-stats vs. train-mode batch stats). If boxes look wrong
here too, the model/checkpoint itself is to blame.

Defaults match the standalone exactly:
    - rgb_ckpt_path = ckpt/rgb/alexnet2.pt
    - model.train()   (BN uses batch stats at inference — the author's choice)
    - extend = 1.5
    - conf = CONF_THRESH = 0.8
    - input: 512x512 RGB, raw 0-255 float32, NCHW

The depth-based masking in the original standalone (lines 98-99) is skipped —
we have no depth from a webcam. Geometry verification is unaffected.

Controls:
    q / ESC : quit
    s       : save 512 view + original view to debug_captures/*.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from utils.model_pytorch import AlexNet2, nms  # noqa: E402
from utils.settings import FM_SIZES, IMG_H, IMG_W, CONF_THRESH  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(_PARENT / "ckpt" / "rgb" / "alexnet2.pt"),
                   help="path to alexnet2.pt state dict")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--bn-mode", choices=["train", "eval"], default="train",
                   help="train (matches standalone, batch stats) or eval (running stats)")
    p.add_argument("--extend", type=float, default=1.5,
                   help="offset extend factor (matches markerless_tracking_standalone.py)")
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help=f"confidence threshold (default {CONF_THRESH})")
    p.add_argument("--show-all", action="store_true",
                   help="draw every above-threshold candidate (yellow) alongside the NMS winner (red)")
    p.add_argument("--save-dir", default=str(_PARENT / "debug_captures"),
                   help="where `s` key dumps PNGs")
    return p.parse_args()


def decode_all_candidates(probs: np.ndarray, preds_loc: np.ndarray,
                          conf: float, extend: float) -> list[tuple[int, int, int, int, float]]:
    """Every anchor whose prob > conf, in 512-space pixel coords."""
    out: list[tuple[int, int, int, int, float]] = []
    i = 0
    for fm_h, fm_w in FM_SIZES:
        sx = IMG_W / fm_w
        sy = IMG_H / fm_h
        for row in range(fm_h):
            for col in range(fm_w):
                score = float(probs[i])
                if score > conf:
                    off = i * 4
                    xc = col + 0.5
                    yc = row + 0.5
                    x_min = (xc + extend * float(preds_loc[off])) * sx
                    y_min = (yc + extend * float(preds_loc[off + 1])) * sy
                    x_max = (xc + extend * float(preds_loc[off + 2])) * sx
                    y_max = (yc + extend * float(preds_loc[off + 3])) * sy
                    out.append((int(x_min), int(y_min), int(x_max), int(y_max), score))
                i += 1
    return out


def per_fm_max(probs: np.ndarray) -> list[float]:
    peaks: list[float] = []
    i = 0
    for fm_h, fm_w in FM_SIZES:
        n = fm_h * fm_w
        peaks.append(float(probs[i:i + n].max()))
        i += n
    return peaks


def load_model(ckpt_path: str, bn_mode: str, device: torch.device) -> AlexNet2:
    print(f"Loading checkpoint: {ckpt_path}")
    model = AlexNet2()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    if bn_mode == "train":
        model.train()
        print("BN mode: train (batch statistics) — matches markerless_tracking_standalone.py")
    else:
        model.eval()
        print("BN mode: eval (running statistics from checkpoint) — matches --bn-mode eval in CoreML conversion")
    return model


@torch.no_grad()
def infer(model: AlexNet2, rgb512: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """rgb512: (512,512,3) uint8 RGB. Returns (probs[305], preds_loc[1220])."""
    # Match markerless_tracking_standalone.py:105-109 exactly.
    t = torch.from_numpy(rgb512.reshape(1, IMG_H, IMG_W, 3).astype(np.float32))
    t = t.permute(0, 3, 1, 2).to(device)  # NHWC -> NCHW
    probs_t, preds_loc_t = model(t)
    probs = probs_t.cpu().numpy()[0]          # (305,)
    preds_loc = preds_loc_t.cpu().numpy()[0]  # (1220,)
    return probs, preds_loc


def main() -> None:
    args = parse_args()
    if not Path(args.ckpt).exists():
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args.ckpt, args.bn_mode, device)

    print(f"Opening camera {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera}. "
                 "Grant camera access to Terminal in System Settings → Privacy & Security → Camera if needed.")

    save_dir = Path(args.save_dir)
    window_model = "torch model_view (512x512)"
    window_cam = "torch camera (orig)"
    cv2.namedWindow(window_model, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_cam, cv2.WINDOW_NORMAL)

    frame_idx = 0
    ema_ms = 0.0
    try:
        while True:
            ok, bgr_orig = cap.read()
            if not ok:
                print("camera read failed; exiting")
                break

            orig_h, orig_w = bgr_orig.shape[:2]
            rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)
            rgb512 = cv2.resize(rgb_orig, (IMG_W, IMG_H))

            t0 = time.perf_counter()
            probs, preds_loc = infer(model, rgb512, device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ema_ms = 0.9 * ema_ms + 0.1 * elapsed_ms if frame_idx > 0 else elapsed_ms

            y_pred_conf = (probs > args.conf).astype(np.float32)
            box = nms(y_pred_conf, preds_loc, probs, extend=args.extend)

            bgr512 = cv2.cvtColor(rgb512, cv2.COLOR_RGB2BGR).copy()

            if args.show_all:
                for (x1, y1, x2, y2, sc) in decode_all_candidates(probs, preds_loc, args.conf, args.extend):
                    cv2.rectangle(bgr512, (x1, y1), (x2, y2), (0, 255, 255), 1)

            bgr_cam = bgr_orig.copy()
            if box is not None and getattr(box, "shape", None) and box.shape != ():
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                score = float(box[5]) if box.size >= 6 else float(box[-1])

                cv2.rectangle(bgr512, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(bgr512, f"{score:.2f}", (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                fx = orig_w / IMG_W
                fy = orig_h / IMG_H
                ox1, oy1 = int(round(x1 * fx)), int(round(y1 * fy))
                ox2, oy2 = int(round(x2 * fx)), int(round(y2 * fy))
                cv2.rectangle(bgr_cam, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                cv2.putText(bgr_cam, f"{score:.2f}", (ox1, max(oy1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            n_above = int((probs > args.conf).sum())
            peaks = per_fm_max(probs)
            hud = (f"torch/{args.bn_mode}  "
                   f"ms {elapsed_ms:5.1f} ({ema_ms:5.1f} ema)  "
                   f"max {probs.max():.3f}  "
                   f"above>{args.conf:.2f}: {n_above:3d}  "
                   f"fm peaks [{peaks[0]:.2f} {peaks[1]:.2f} {peaks[2]:.2f}]")
            for img in (bgr512, bgr_cam):
                cv2.putText(img, hud, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, hud, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_model, bgr512)
            cv2.imshow(window_cam, bgr_cam)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                save_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(save_dir / f"{ts}_torch_{args.bn_mode}_model.png"), bgr512)
                cv2.imwrite(str(save_dir / f"{ts}_torch_{args.bn_mode}_cam.png"), bgr_cam)
                print(f"saved {ts}_torch_{args.bn_mode}_{{model,cam}}.png to {save_dir}")

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
