#!/usr/bin/env python3
"""
Mac webcam debug harness for the converted AlexNet2 Core ML model.

Isolates whether an on-device bounding-box sizing issue on Apple Vision Pro
is a model-side regression (conversion) or a display-side regression (Swift
overlay transform). This script exercises the exact same .mlpackage through
coremltools, decodes with the numpy `nms()` the RealSense pipeline uses, and
renders the result in two OpenCV windows so geometry can be compared against
an on-device screenshot.

Usage:
    python scripts/test_coreml_webcam.py                  # default model path
    python scripts/test_coreml_webcam.py --show-all       # draw every anchor above conf
    python scripts/test_coreml_webcam.py --conf 0.5       # relax threshold
    python scripts/test_coreml_webcam.py --camera 1       # pick a different cam

Controls:
    q / ESC : quit
    s       : save 512 view + original view to debug_captures/*.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import coremltools as ct  # noqa: E402
from utils.model_pytorch import nms  # noqa: E402
from utils.settings import FM_SIZES, IMG_H, IMG_W, CONF_THRESH  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(_PARENT / "AlexNet2.mlpackage"),
                   help="path to the Core ML model")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--extend", type=float, default=1.5,
                   help="offset extend factor (matches markerless_tracking_standalone.py)")
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help=f"confidence threshold (default {CONF_THRESH})")
    p.add_argument("--show-all", action="store_true",
                   help="draw every above-threshold candidate (yellow) in addition to the NMS winner (red)")
    p.add_argument("--save-dir", default=str(_PARENT / "debug_captures"),
                   help="where `s` key dumps PNGs")
    return p.parse_args()


def decode_all_candidates(probs: np.ndarray, preds_loc: np.ndarray,
                          conf: float, extend: float) -> list[tuple[int, int, int, int, float]]:
    """Return every anchor whose prob > conf, in 512-space pixel coords."""
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
    """Max prob value for each feature map, in declaration order."""
    peaks: list[float] = []
    i = 0
    for fm_h, fm_w in FM_SIZES:
        n = fm_h * fm_w
        peaks.append(float(probs[i:i + n].max()))
        i += n
    return peaks


def main() -> None:
    args = parse_args()
    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}")

    print(f"Loading {args.model}")
    ml_model = ct.models.MLModel(args.model)

    print(f"Opening camera {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera}. "
                 "If this is macOS, grant Terminal camera access in "
                 "System Settings → Privacy & Security → Camera.")

    save_dir = Path(args.save_dir)
    window_model = "model_view (512x512)"
    window_cam = "camera (orig)"
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
            pil = Image.fromarray(rgb512, mode="RGB")

            t0 = time.perf_counter()
            out = ml_model.predict({"image": pil})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ema_ms = 0.9 * ema_ms + 0.1 * elapsed_ms if frame_idx > 0 else elapsed_ms

            probs = np.asarray(out["probs"]).reshape(-1).astype(np.float32)
            preds_loc = np.asarray(out["preds_loc"]).reshape(-1).astype(np.float32)

            # Match the standalone pipeline: a 0/1 conf mask AND the raw prob mask.
            y_pred_conf = (probs > args.conf).astype(np.float32)
            box = nms(y_pred_conf, preds_loc, probs, extend=args.extend)

            # 512-space view
            bgr512 = cv2.cvtColor(rgb512, cv2.COLOR_RGB2BGR).copy()

            if args.show_all:
                for (x1, y1, x2, y2, sc) in decode_all_candidates(probs, preds_loc, args.conf, args.extend):
                    cv2.rectangle(bgr512, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # NMS winner in red on both windows
            bgr_cam = bgr_orig.copy()
            if box is not None and getattr(box, "shape", None) and box.shape != ():
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                score = float(box[5]) if box.size >= 6 else float(box[-1])

                cv2.rectangle(bgr512, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(bgr512, f"{score:.2f}", (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Project 512-space rect back to original camera frame coords (anisotropic).
                fx = orig_w / IMG_W
                fy = orig_h / IMG_H
                ox1, oy1 = int(round(x1 * fx)), int(round(y1 * fy))
                ox2, oy2 = int(round(x2 * fx)), int(round(y2 * fy))
                cv2.rectangle(bgr_cam, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                cv2.putText(bgr_cam, f"{score:.2f}", (ox1, max(oy1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # HUD
            n_above = int((probs > args.conf).sum())
            peaks = per_fm_max(probs)
            hud = (f"ms {elapsed_ms:5.1f} ({ema_ms:5.1f} ema)  "
                   f"max {probs.max():.3f}  "
                   f"above>{args.conf:.2f}: {n_above:3d}  "
                   f"fm peaks [{peaks[0]:.2f} {peaks[1]:.2f} {peaks[2]:.2f}]")
            for img in (bgr512, bgr_cam):
                cv2.putText(img, hud, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, hud, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow(window_model, bgr512)
            cv2.imshow(window_cam, bgr_cam)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                save_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(save_dir / f"{ts}_model.png"), bgr512)
                cv2.imwrite(str(save_dir / f"{ts}_cam.png"), bgr_cam)
                print(f"saved {ts}_{{model,cam}}.png to {save_dir}")

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
