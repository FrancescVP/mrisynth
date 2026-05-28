"""Inference script for the Wavelet Flow Matching model.

Loads a trained WFMModel checkpoint, runs denoising on full volumes,
and saves predictions + ground truth + inputs as NIfTI files.

Usage
-----
  uv run python scripts/infer_wfm.py \\
      --data_dir   /path/to/preprocessed/val \\
      --checkpoint checkpoints/wfm_best/latest_net_WFM.pth \\
      --out_dir    predictions/wfm \\
      --n_cases    5 \\
      --device     cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib
import numpy as np
import torch

from mrisynth.model.dataset import Pix2Pix3dDataset
from mrisynth.model.wfm import WFMModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))
    print(f"  saved {path}")


def get_affine(data_dir: Path, case_id: str) -> np.ndarray:
    candidates = list(data_dir.glob(f"{case_id}*.json"))
    if candidates:
        meta = json.loads(candidates[0].read_text())
        return np.array(meta["affine"], dtype=np.float64)
    return np.eye(4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",        required=True, help="Val .npz directory.")
    p.add_argument("--checkpoint",      required=True, help="WFM checkpoint (.pth).")
    p.add_argument("--out_dir",         default="predictions/wfm")
    p.add_argument("--n_cases",         type=int, default=5)
    p.add_argument("--n_inference_steps", type=int, default=200)
    p.add_argument("--n_cond",          type=int, default=2)
    p.add_argument("--input_channels",  nargs="+", default=["T1n", "T2F"])
    p.add_argument("--target_channels", nargs="+", default=["T1CE"])
    p.add_argument("--unet_channels",   nargs="+", type=int, default=[64, 128, 256, 256])
    p.add_argument("--num_res_blocks",  type=int, default=2)
    p.add_argument("--n_attention_levels", type=int, default=2)
    p.add_argument("--device",          default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir  = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    opt = SimpleNamespace(
        isTrain=False,
        checkpoints_dir=str(Path(args.checkpoint).parents[1]),
        name=Path(args.checkpoint).parent.name,
        device=device,
        verbose=False,
        init_type="kaiming",
        init_gain=0.02,
        norm="instance",
        backbone="unet",
        n_cond=args.n_cond,
        unet_channels=args.unet_channels,
        num_res_blocks=args.num_res_blocks,
        n_attention_levels=args.n_attention_levels,
        n_timesteps=1000,
        n_inference_steps=args.n_inference_steps,
        inference_noise=0.0,
    )
    model = WFMModel(opt)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.netWFM.load_state_dict(state)
    model.netWFM.to(device).float().eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    ds = Pix2Pix3dDataset(
        data_dir,
        input_channels=args.input_channels,
        target_channels=args.target_channels,
        patch_size=None,
        augment=False,
    )
    cases = ds.case_ids[: args.n_cases]
    print(f"Running inference on {len(cases)} cases …\n")

    for i, cid in enumerate(cases):
        batch = ds[i]
        # Add batch dimension
        batch = {
            "A":       batch["A"].unsqueeze(0),
            "B":       batch["B"].unsqueeze(0),
            "seg":     batch["seg"].unsqueeze(0) if batch["seg"] is not None else None,
            "A_paths": batch["A_paths"],
        }
        print(f"[{cid}]")

        model.set_input(batch)
        model.forward()

        pred_vol = model.pred_image[0, 0].cpu().float().numpy()
        gt_vol   = batch["B"][0, 0].float().numpy()

        affine = get_affine(data_dir, cid)

        save_nifti(pred_vol, affine, out_dir / cid / "pred_target.nii.gz")
        save_nifti(gt_vol,   affine, out_dir / cid / "gt_target.nii.gz")

        A = batch["A"][0]  # (n_cond, D, H, W)
        for ch_i, ch_name in enumerate(args.input_channels):
            save_nifti(A[ch_i].numpy(), affine, out_dir / cid / f"input_{ch_name}.nii.gz")

        print(f"  pred  min={pred_vol.min():.3f}  max={pred_vol.max():.3f}")
        print()

    print(f"Done. Predictions saved to {out_dir}/")


if __name__ == "__main__":
    main()
