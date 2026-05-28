"""Inference script for RFlowControlNet (latent diffusion + ControlNet).

Loads a trained RFlowControlNetModel checkpoint, runs denoising on validation
latents, decodes with the MAISI VAE, and saves predictions as NIfTI files.

Usage
-----
  uv run python scripts/infer_rflow_controlnet.py \\
      --latent_dir     latents/dataset500/val \\
      --vae_ckpt       /path/to/autoencoder_epoch273.pt \\
      --checkpoint_unet checkpoints/rflow_cn/latest_net_UNet.pth \\
      --checkpoint_cn   checkpoints/rflow_cn/latest_net_ControlNet.pth \\
      --out_dir        predictions/rflow_controlnet \\
      --n_cases        5 \\
      --device         cuda:0
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

from mrisynth.model.vae import MaisiVAE
from mrisynth.model.rflow_controlnet import RFlowControlNetModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_latent(path: Path) -> torch.Tensor:
    return torch.load(path, weights_only=True)


def decode_latent(vae: MaisiVAE, z: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        img = vae.decode(z.float().to(device))
    return img[0, 0].cpu().float().numpy()


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
    p.add_argument("--latent_dir",       required=True, help="Val latent directory.")
    p.add_argument("--data_dir",         default=None,  help="Preprocessed .npz/.json dir (for affines).")
    p.add_argument("--vae_ckpt",         required=True, help="MAISI VAE checkpoint.")
    p.add_argument("--checkpoint_unet",  required=True, help="UNet checkpoint (.pth).")
    p.add_argument("--checkpoint_cn",    required=True, help="ControlNet checkpoint (.pth).")
    p.add_argument("--out_dir",          default="predictions/rflow_controlnet")
    p.add_argument("--unet_channels",    nargs="+", type=int, default=[32, 64, 128, 128])
    p.add_argument("--num_res_blocks",   type=int, default=2)
    p.add_argument("--n_attention_levels", type=int, default=2)
    p.add_argument("--n_cond",           type=int, default=2)
    p.add_argument("--n_cases",          type=int, default=5)
    p.add_argument("--n_inference_steps", type=int, default=200)
    p.add_argument("--cond_keys",        nargs="+", default=["t1n", "t2f"])
    p.add_argument("--target_key",       default="t1c")
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--save_gt",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir   = Path(args.out_dir)
    latent_dir = Path(args.latent_dir)
    data_dir   = Path(args.data_dir) if args.data_dir else None

    print(f"Loading VAE from {args.vae_ckpt} …")
    vae = MaisiVAE.from_checkpoint(args.vae_ckpt, device=device)
    vae.eval()

    # Build shared opt namespace
    ckpt_dir = Path(args.checkpoint_unet).parent
    opt = SimpleNamespace(
        isTrain=False,
        checkpoints_dir=str(ckpt_dir.parent),
        name=ckpt_dir.name,
        device=device,
        verbose=False,
        init_type="kaiming",
        init_gain=0.02,
        norm="instance",
        backbone="unet",
        latent_channels=4,
        n_cond=args.n_cond,
        unet_channels=args.unet_channels,
        num_res_blocks=args.num_res_blocks,
        n_attention_levels=args.n_attention_levels,
        n_timesteps=1000,
        n_inference_steps=args.n_inference_steps,
    )
    model = RFlowControlNetModel(opt)

    unet_state = torch.load(args.checkpoint_unet, map_location="cpu", weights_only=True)
    cn_state   = torch.load(args.checkpoint_cn,   map_location="cpu", weights_only=True)
    model.netUNet.load_state_dict(unet_state)
    model.netControlNet.load_state_dict(cn_state)
    model.netUNet.to(device).float().eval()
    model.netControlNet.to(device).float().eval()
    print(f"Loaded UNet: {args.checkpoint_unet}")
    print(f"Loaded ControlNet: {args.checkpoint_cn}")

    cases = sorted([d.name for d in latent_dir.iterdir() if d.is_dir()])
    cases = cases[: args.n_cases]
    print(f"Running inference on {len(cases)} cases …\n")

    for cid in cases:
        d = latent_dir / cid
        print(f"[{cid}]")

        cond_parts = [
            load_latent(d / f"{cid}-{k}_z_mu.pt")
            for k in args.cond_keys
        ]
        latent_cond = torch.cat(cond_parts, dim=0).unsqueeze(0).to(device).float()
        mu_tgt = load_latent(d / f"{cid}-{args.target_key}_z_mu.pt")
        latent_tgt = mu_tgt.unsqueeze(0).float()

        # Try to load seg for ControlNet conditioning
        seg_path = d / f"{cid}-seg.pt"
        seg = None
        if seg_path.exists():
            seg = torch.load(seg_path, map_location="cpu", weights_only=True).unsqueeze(0)

        dummy_batch = {
            "latent_tgt":  latent_tgt,
            "latent_cond": latent_cond,
            "seg":         seg,
            "case_id":     [cid],
        }
        model.set_input(dummy_batch)
        model.forward()
        pred_lat = model.pred_latent

        pred_vol = decode_latent(vae, pred_lat, device)
        affine = get_affine(data_dir, cid) if data_dir else np.eye(4)
        save_nifti(pred_vol, affine, out_dir / cid / "pred_target.nii.gz")

        if args.save_gt:
            gt_vol = decode_latent(vae, latent_tgt, device)
            save_nifti(gt_vol, affine, out_dir / cid / "gt_target.nii.gz")

        print(f"  pred  min={pred_vol.min():.3f}  max={pred_vol.max():.3f}")
        print()

    print(f"Done. Predictions saved to {out_dir}/")


if __name__ == "__main__":
    main()
