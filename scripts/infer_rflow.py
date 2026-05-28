"""Inference script for the RFlow latent diffusion model.

Loads a trained RFlowModel checkpoint, runs denoising on validation latents,
decodes with the MAISI VAE, and saves predictions + ground truth as NIfTI files.

Usage
-----
  uv run python scripts/infer_rflow.py \\
      --latent_dir     latents/dataset500/val \\
      --data_dir       /path/to/preprocessed_500 \\
      --vae_ckpt       /path/to/autoencoder_epoch273.pt \\
      --checkpoint     checkpoints/rflow_t1n_t2f_to_t1ce/latest_net_UNet.pth \\
      --out_dir        predictions/rflow \\
      --unet_channels  32 64 128 128 \\
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

from gan_data_processing.model.vae import MaisiVAE
from gan_data_processing.model.rflow import RFlowModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_latent(path: Path) -> torch.Tensor:
    return torch.load(path, weights_only=True)


def decode_latent(vae: MaisiVAE, z: torch.Tensor, device: torch.device) -> np.ndarray:
    """Decode a single latent (1, 4, D', H', W') → numpy (D, H, W) float32."""
    with torch.no_grad():
        img = vae.decode(z.float().to(device))   # (1, 1, D, H, W)
    return img[0, 0].cpu().float().numpy()


def save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))
    print(f"  saved {path}")


def get_affine(data_dir: Path, case_id: str) -> np.ndarray:
    """Load affine from the preprocessed .json metadata, or return identity."""
    candidates = list(data_dir.glob(f"{case_id}*.json"))
    if candidates:
        meta = json.loads(candidates[0].read_text())
        return np.array(meta["affine"], dtype=np.float64)
    return np.eye(4)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--latent_dir",    required=True, help="Val latent directory.")
    p.add_argument("--data_dir",      default=None,  help="Preprocessed .npz/.json dir (for affines).")
    p.add_argument("--vae_ckpt",      required=True, help="MAISI VAE checkpoint.")
    p.add_argument("--checkpoint",    required=True, help="RFlow UNet checkpoint (.pth).")
    p.add_argument("--out_dir",       default="predictions/rflow")
    p.add_argument("--unet_channels",     nargs="+", type=int, default=[32, 64, 128, 128])
    p.add_argument("--num_res_blocks",    type=int, default=2)
    p.add_argument("--n_attention_levels",type=int, default=2)
    p.add_argument("--n_cases",           type=int, default=5, help="Number of val cases to run.")
    p.add_argument("--n_inference_steps", type=int, default=200)
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--save_gt",       action="store_true", help="Also save GT T1CE decoded from mu_tgt.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    latent_dir = Path(args.latent_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None

    # ── VAE ──────────────────────────────────────────────────────────────────
    print(f"Loading VAE from {args.vae_ckpt} …")
    vae = MaisiVAE.from_checkpoint(args.vae_ckpt, device=device)
    vae.eval()

    # ── RFlow model ──────────────────────────────────────────────────────────
    opt = SimpleNamespace(
        isTrain=False,
        checkpoints_dir=str(Path(args.checkpoint).parents[1]),
        name=Path(args.checkpoint).parent.name,
        device=device,
        verbose=False,
        init_type="kaiming",
        init_gain=0.02,
        norm="instance",
        latent_channels=4,
        unet_channels=args.unet_channels,
        n_timesteps=1000,
        n_inference_steps=args.n_inference_steps,
        num_res_blocks=args.num_res_blocks,
        n_attention_levels=args.n_attention_levels,
    )
    model = RFlowModel(opt)

    # Load weights directly
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.netUNet.load_state_dict(state)
    model.netUNet.to(device).float().eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Discover val cases ───────────────────────────────────────────────────
    cases = sorted([d.name for d in latent_dir.iterdir() if d.is_dir()])
    cases = cases[: args.n_cases]
    print(f"Running inference on {len(cases)} cases …\n")

    for cid in cases:
        d = latent_dir / cid
        print(f"[{cid}]")

        mu_t1n = load_latent(d / f"{cid}-t1n_z_mu.pt")
        mu_t2f = load_latent(d / f"{cid}-t2f_z_mu.pt")
        mu_tgt = load_latent(d / f"{cid}-t1c_z_mu.pt")

        latent_cond = torch.cat([mu_t1n, mu_t2f], dim=0).unsqueeze(0).to(device).float()  # (1,8,D',H',W')
        latent_tgt  = mu_tgt.unsqueeze(0).float()

        # ── Denoising ──────────────────────────────────────────────────────
        dummy_batch = {
            "latent_tgt":  latent_tgt,
            "latent_cond": latent_cond,
        }
        model.set_input(dummy_batch)
        model.forward()                            # sets model.pred_latent
        pred_lat = model.pred_latent               # (1, 4, D', H', W')

        # ── Decode ────────────────────────────────────────────────────────
        pred_vol = decode_latent(vae, pred_lat, device)

        affine = get_affine(data_dir, cid) if data_dir else np.eye(4)

        save_nifti(pred_vol, affine, out_dir / cid / "pred_t1ce.nii.gz")

        if args.save_gt:
            gt_vol = decode_latent(vae, latent_tgt, device)
            save_nifti(gt_vol, affine, out_dir / cid / "gt_t1ce.nii.gz")

        # ── Save inputs from .npz (ch1=T1n, ch2=T2FLAIR) ─────────────────
        if data_dir is not None:
            npz_files = list(data_dir.glob(f"{cid}*.npz"))
            if npz_files:
                data = np.load(npz_files[0])["data"]  # (C, D, H, W)
                save_nifti(data[1], affine, out_dir / cid / "input_t1n.nii.gz")
                save_nifti(data[2], affine, out_dir / cid / "input_t2flair.nii.gz")

        # Quick stats
        print(f"  pred  min={pred_vol.min():.3f}  max={pred_vol.max():.3f}  "
              f"mean={pred_vol.mean():.3f}")
        if args.save_gt:
            print(f"  gt    min={gt_vol.min():.3f}  max={gt_vol.max():.3f}  "
                  f"mean={gt_vol.mean():.3f}")
        print()

    print(f"Done. Predictions saved to {out_dir}/")


if __name__ == "__main__":
    main()
