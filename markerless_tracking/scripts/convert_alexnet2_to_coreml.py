#!/usr/bin/env python3
"""
Convert the AlexNet2 RGB SSD detector (PyTorch) to a Core ML .mlpackage
suitable for on-device inference on Apple Vision Pro (visionOS 2+).

The PyTorch model expects a (1, 3, 512, 512) NCHW float32 tensor in raw 0-255
range. The original standalone pipeline runs the model with `model.train()` so
that batch-norm uses *batch* statistics (not running stats). Tracing the model
in train mode would burn the dummy-batch stats into constants, which is
useless at runtime. We offer three BN strategies:

  --bn-mode instance  (DEFAULT, recommended)
                     Replace every BatchNorm2d with an explicit per-sample
                     normalization that is mathematically equivalent to
                     train-mode BN at batch=1: compute per-channel mean/var
                     across the spatial dims of the current sample, then apply
                     the original gamma/beta. No running stats. No calibration
                     needed. This exactly matches what markerless_tracking_
                     standalone.py does at inference.
  --bn-mode warmup   Run a directory of representative frames through the model
                     in train mode under no_grad, so BN's exponential moving
                     averages of running_mean / running_var converge, then trace
                     in eval mode. Preserves training-time BN behaviour as a
                     static affine op, but only approximately.
  --bn-mode eval     Trust the running stats already in the checkpoint (likely
                     stale / unrepresentative; produced the original "box too
                     large" regression at inference time).

Outputs:
    probs       (1, 305) float  -- sigmoid confidences
    preds_loc   (1, 1220) float -- raw box-offset regressions

NMS is intentionally NOT included in the graph; it is implemented in Swift on
the consumer side.

Example:
    python scripts/convert_alexnet2_to_coreml.py \
        --ckpt ckpt/rgb/alexnet2.pt \
        --bn-mode instance \
        --out AlexNet2.mlpackage --target iOS26 --parity
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow `from utils...` imports when running from anywhere.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from utils.model_pytorch import AlexNet2  # noqa: E402
from utils.settings import FM_SIZES, IMG_H, IMG_W  # noqa: E402

import coremltools as ct  # noqa: E402

NUM_ANCHORS = sum(h * w for h, w in FM_SIZES)
PROBS_SHAPE = (1, NUM_ANCHORS)
LOC_SHAPE = (1, NUM_ANCHORS * 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(_PARENT / "ckpt" / "rgb" / "alexnet2.pt"),
                   help="path to alexnet2.pt state dict")
    p.add_argument("--out", default=str(_PARENT / "AlexNet2.mlpackage"),
                   help="output .mlpackage path")
    p.add_argument("--bn-mode", choices=["instance", "warmup", "eval"], default="instance",
                   help="batch-norm handling (see module docstring)")
    p.add_argument("--calib-dir", default=None,
                   help="directory of calibration images (required for --bn-mode warmup)")
    p.add_argument("--calib-max", type=int, default=500,
                   help="cap on calibration images consumed during warmup")
    p.add_argument("--parity", action="store_true",
                   help="after conversion, compare PyTorch vs Core ML on random inputs")
    p.add_argument("--parity-n", type=int, default=8, help="number of random parity samples")
    p.add_argument("--probs-tol", type=float, default=None,
                   help="max-abs probs diff tolerated in --parity (post-FP16 quant). "
                        "Default picks a value appropriate for the --bn-mode: "
                        "0.01 for eval (folded BN, stable at FP16), "
                        "0.05 for instance/warmup (explicit mean/var ops).")
    p.add_argument("--target", default="iOS17",
                   help="ct.target enum name. coremltools has no visionOS-specific "
                        "enum; iOS17 loads on visionOS 1.0+, iOS18 on visionOS 2+, "
                        "iOS26 on visionOS 26.")
    return p.parse_args()


class _BatchOneNorm(nn.Module):
    """Drop-in replacement for nn.BatchNorm2d that reproduces train-mode BN at
    batch=1 as explicit ops (mean / var / rsqrt / affine). Traces cleanly and
    converts to Core ML without relying on BN-specific rewrite passes.

    Equivalence to `BN(training=True)` at batch=1: BN computes per-channel
    mean/var across `(batch, H, W)` — with a single sample, that collapses to
    per-channel mean/var over `(H, W)` of that sample, which is exactly what
    this module computes. Gamma/beta are copied from the source BN.
    """

    def __init__(self, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        self.eps = bn.eps
        gamma = bn.weight.detach().clone() if bn.affine else torch.ones(bn.num_features)
        beta = bn.bias.detach().clone() if bn.affine else torch.zeros(bn.num_features)
        self.weight = nn.Parameter(gamma)
        self.bias = nn.Parameter(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W). Assumes N == 1 at inference (matches visionOS use).
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_hat + self.bias.view(1, -1, 1, 1)


def swap_bn_to_instance(module: nn.Module) -> int:
    """In-place replace every nn.BatchNorm2d descendant with _BatchOneNorm.
    Returns the number of BN layers swapped."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _BatchOneNorm(child))
            n += 1
        else:
            n += swap_bn_to_instance(child)
    return n


def verify_instance_equivalence(model_before_swap_forward: torch.Tensor,
                                model_after: nn.Module,
                                dummy: torch.Tensor,
                                tol: float = 1e-4) -> None:
    """Forward `dummy` through the already-swapped model and compare against
    a tuple of reference tensors captured from the original model in train mode."""
    ref_probs, ref_loc = model_before_swap_forward  # type: ignore[misc]
    model_after.eval()
    with torch.no_grad():
        s_probs, s_loc = model_after(dummy)
    d_probs = (ref_probs - s_probs).abs().max().item()
    d_loc = (ref_loc - s_loc).abs().max().item()
    print(f"InstanceNorm equivalence: |Δprobs|={d_probs:.3e}  |Δloc|={d_loc:.3e}  (tol {tol:.0e})")
    if d_probs > tol or d_loc > tol:
        raise SystemExit(
            f"InstanceNorm swap diverges from train-mode BN: "
            f"|Δprobs|={d_probs:.3e}, |Δloc|={d_loc:.3e} (tol {tol:.0e})")


def load_calibration_tensors(calib_dir: str, max_n: int) -> list[torch.Tensor]:
    """Load up to max_n images from calib_dir, resize to 512x512 RGB float32 in 0-255."""
    import cv2  # local import: only needed for warmup
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in Path(calib_dir).iterdir() if p.suffix.lower() in exts)[:max_n]
    if not paths:
        raise FileNotFoundError(f"No images found in {calib_dir}")
    out: list[torch.Tensor] = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skipping unreadable: {p.name}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_W, IMG_H))
        t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        out.append(t)
    print(f"Loaded {len(out)} calibration images from {calib_dir}")
    return out


def warmup_batch_norm(model: AlexNet2, calib_dir: str, max_n: int) -> None:
    tensors = load_calibration_tensors(calib_dir, max_n)
    model.train()  # so BN updates running_mean / running_var via EMA
    with torch.no_grad():
        for i, t in enumerate(tensors):
            model(t)
            if (i + 1) % 50 == 0:
                print(f"  warmup {i + 1}/{len(tensors)}")
    model.eval()
    print("BN warmup complete; switched to eval mode.")


def load_model(args: argparse.Namespace) -> AlexNet2:
    print(f"Loading checkpoint: {args.ckpt}")
    model = AlexNet2()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    if args.bn_mode == "instance":
        # Capture train-mode BN reference outputs on a dummy input BEFORE the
        # swap, so we can verify equivalence afterwards.
        dummy = torch.rand(1, 3, IMG_H, IMG_W) * 255.0
        model.train()
        with torch.no_grad():
            ref = model(dummy)

        n_swapped = swap_bn_to_instance(model)
        print(f"BN mode: instance (swapped {n_swapped} BatchNorm2d → _BatchOneNorm)")
        verify_instance_equivalence(ref, model, dummy)
    elif args.bn_mode == "warmup":
        if not args.calib_dir:
            raise SystemExit("--bn-mode warmup requires --calib-dir")
        warmup_batch_norm(model, args.calib_dir, args.calib_max)
    else:
        model.eval()
        print("BN mode: eval (using running stats from checkpoint)")
    return model


def trace_model(model: AlexNet2) -> torch.jit.ScriptModule:
    dummy = torch.rand(1, 3, IMG_H, IMG_W) * 255.0
    with torch.no_grad():
        probs, preds_loc = model(dummy)
    if tuple(probs.shape) != PROBS_SHAPE or tuple(preds_loc.shape) != LOC_SHAPE:
        raise RuntimeError(
            f"Unexpected output shapes: probs={tuple(probs.shape)} (want {PROBS_SHAPE}), "
            f"preds_loc={tuple(preds_loc.shape)} (want {LOC_SHAPE})")
    print(f"Forward OK: probs {tuple(probs.shape)}, preds_loc {tuple(preds_loc.shape)}")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    return traced


def resolve_target(name: str):
    target = getattr(ct.target, name, None)
    if target is None:
        available = [a for a in dir(ct.target) if not a.startswith("_")]
        raise SystemExit(f"Unknown ct.target '{name}'. Available: {available}")
    return target


def convert_to_coreml(traced: torch.jit.ScriptModule, target_name: str) -> ct.models.MLModel:
    target = resolve_target(target_name)
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, IMG_H, IMG_W),
        color_layout=ct.colorlayout.RGB,
        scale=1.0,            # AlexNet2 wants raw 0-255; do NOT divide
        bias=[0.0, 0.0, 0.0],
    )
    outputs = [
        ct.TensorType(name="probs"),
        ct.TensorType(name="preds_loc"),
    ]
    print(f"Converting (target={target_name}, fp16, mlprogram) ...")
    ml_model = ct.convert(
        traced,
        inputs=[image_input],
        outputs=outputs,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target,
        convert_to="mlprogram",
    )
    return ml_model


def parity_check(torch_model: AlexNet2, mlpackage_path: str, n: int, probs_tol: float) -> None:
    """Run n random uint8 images through both PyTorch and Core ML; report max abs diffs."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping parity check (pip install pillow)")
        return
    print(f"Loading saved Core ML model for parity: {mlpackage_path}")
    ml_model = ct.models.MLModel(mlpackage_path)
    torch_model.eval()
    max_probs_diff = 0.0
    max_loc_diff = 0.0
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)

        # Core ML expects a PIL image input named "image".
        pil = Image.fromarray(arr, mode="RGB")
        ml_out = ml_model.predict({"image": pil})
        ml_probs = np.asarray(ml_out["probs"]).reshape(PROBS_SHAPE)
        ml_loc = np.asarray(ml_out["preds_loc"]).reshape(LOC_SHAPE)

        # PyTorch reference: same pixels, NCHW float32 0-255.
        t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            t_probs, t_loc = torch_model(t)
        t_probs = t_probs.numpy()
        t_loc = t_loc.numpy()

        d_probs = float(np.max(np.abs(ml_probs - t_probs)))
        d_loc = float(np.max(np.abs(ml_loc - t_loc)))
        max_probs_diff = max(max_probs_diff, d_probs)
        max_loc_diff = max(max_loc_diff, d_loc)
        print(f"  sample {i + 1}/{n}: |Δprobs|={d_probs:.4e}  |Δloc|={d_loc:.4e}")

    print(f"Parity max |Δprobs| = {max_probs_diff:.4e} (tol {probs_tol})")
    print(f"Parity max |Δloc|   = {max_loc_diff:.4e}")
    if max_probs_diff > probs_tol:
        raise SystemExit(f"Parity check FAILED: probs diff {max_probs_diff} > {probs_tol}")
    print("Parity check passed.")


def main() -> None:
    args = parse_args()
    model = load_model(args)
    traced = trace_model(model)
    ml_model = convert_to_coreml(traced, args.target)
    out_path = os.path.abspath(args.out)
    print(f"Saving: {out_path}")
    ml_model.save(out_path)
    print(f"Wrote {out_path}")
    if args.parity:
        probs_tol = args.probs_tol
        if probs_tol is None:
            probs_tol = 0.01 if args.bn_mode == "eval" else 0.05
        parity_check(model, out_path, args.parity_n, probs_tol)


if __name__ == "__main__":
    main()
