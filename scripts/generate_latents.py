#!/usr/bin/env python3
"""Pre-compute MAISI VAE latents from preprocessed .npz files.

Reads the gan_data_processing .npz dataset (Z-score normalised, shape (4,D,H,W))
and encodes each modality channel independently with the pretrained MAISI VAE.

Output layout (one subdirectory per case):
    <out_dir>/
        train/
            <case_id>/
                <case_id>-t1c_z_mu.pt      (4, D', H', W')
                <case_id>-t1c_z_sigma.pt
                <case_id>-t1n_z_mu.pt
                <case_id>-t1n_z_sigma.pt
                <case_id>-t2f_z_mu.pt
                <case_id>-t2f_z_sigma.pt
        val/
            ...

The MAISI VAE was trained on percentile-normalised data (0–99.5 → [0, 1]).
We apply the same normalisation per channel before encoding.

Usage (from repo root):
    uv run python scripts/generate_latents.py \\
        --data_dir /path/to/Dataset015_preprocessed_500 \\
        --vae_ckpt /path/to/autoencoder_epoch273.pt \\
        --out_dir  latents/dataset500 \\
        --val_fraction 0.1 \\
        --device cuda:0

Checkpoint location (git-lfs — pull first if not yet downloaded):
    /home/fran/Projects/mri_image_synthesis/
      An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation/
      checkpoints/autoencoder_epoch273.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Channel indices in our .npz files (nnUNet Dataset015 convention)
# ---------------------------------------------------------------------------
CH_T1CE = 0
CH_T1N  = 1
CH_T2F  = 2   # FLAIR
CH_T2W  = 3   # T2-weighted (non-FLAIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_norm(arr: np.ndarray, lo: float = 0.0, hi: float = 99.5) -> np.ndarray:
    """Clip to [lo, hi] percentile of non-zero voxels and scale to [0, 1]."""
    mask = arr != 0
    if mask.sum() == 0:
        return np.zeros_like(arr)
    p_lo = np.percentile(arr[mask], lo)
    p_hi = np.percentile(arr[mask], hi)
    if p_hi <= p_lo:
        return np.zeros_like(arr)
    arr = np.clip(arr, p_lo, p_hi)
    arr = (arr - p_lo) / (p_hi - p_lo)
    return arr.astype(np.float32)


def encode_channel(
    vae,
    data_chw: np.ndarray,   # (D, H, W) float32
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Percentile-normalise one channel and encode with the VAE."""
    vol = percentile_norm(data_chw)                      # (D, H, W) in [0, 1]
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    x = x.to(device)

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with autocast("cuda", dtype=torch.float16):
                mu, sigma = vae.encode(x)
        else:
            mu, sigma = vae.encode(x)

    mu    = mu.squeeze(0).cpu()
    sigma = sigma.squeeze(0).cpu()
    del x
    torch.cuda.empty_cache()
    return mu, sigma   # (4, D', H', W')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="Directory of .npz files from gan-preprocess.")
    ap.add_argument("--vae_ckpt", required=True,
                    help="Path to autoencoder_epoch273.pt (MAISI VAE checkpoint).")
    ap.add_argument("--out_dir",  default="latents",
                    help="Output directory for latent .pt files.")
    ap.add_argument("--val_fraction", type=float, default=0.1,
                    help="Fraction of cases to assign to the val split.")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",       default="cuda:0")
    ap.add_argument("--no_amp",       action="store_true",
                    help="Disable automatic mixed precision.")
    ap.add_argument("--max_cases",    type=int,   default=0,
                    help="Limit to N cases total (0 = all; useful for testing).")
    args = ap.parse_args()

    device   = torch.device(args.device)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    use_amp  = not args.no_amp

    # -----------------------------------------------------------------------
    # Load VAE
    # -----------------------------------------------------------------------
    print(f"Loading MAISI VAE from  {args.vae_ckpt}")
    from gan_data_processing.model.vae import MaisiVAE
    vae = MaisiVAE.from_checkpoint(args.vae_ckpt, device=device)
    vae.eval()
    print("VAE loaded.")

    # -----------------------------------------------------------------------
    # Discover cases and split
    # -----------------------------------------------------------------------
    all_ids = sorted(p.stem for p in data_dir.glob("*.npz"))
    if not all_ids:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    if args.max_cases > 0:
        all_ids = all_ids[: args.max_cases]

    rng = random.Random(args.seed)
    rng.shuffle(all_ids)
    n_val   = max(1, int(len(all_ids) * args.val_fraction))
    val_ids = set(all_ids[:n_val])
    splits  = {cid: ("val" if cid in val_ids else "train") for cid in all_ids}

    print(f"Cases: {len(all_ids)}  (train {len(all_ids)-n_val}, val {n_val})")

    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------
    done = skipped = 0

    for cid in tqdm(all_ids, desc="Encoding"):
        split   = splits[cid]
        case_out = out_dir / split / cid
        case_out.mkdir(parents=True, exist_ok=True)

        # Skip if all 8 latent files already exist
        expected = [
            case_out / f"{cid}-t1c_z_mu.pt", case_out / f"{cid}-t1c_z_sigma.pt",
            case_out / f"{cid}-t1n_z_mu.pt", case_out / f"{cid}-t1n_z_sigma.pt",
            case_out / f"{cid}-t2f_z_mu.pt", case_out / f"{cid}-t2f_z_sigma.pt",
            case_out / f"{cid}-t2w_z_mu.pt", case_out / f"{cid}-t2w_z_sigma.pt",
        ]
        if all(p.exists() for p in expected):
            skipped += 1
            continue

        with np.load(data_dir / f"{cid}.npz") as z:
            data = np.asarray(z["data"], dtype=np.float32)  # (4, D, H, W)
            # Save segmentation downscaled to latent resolution (4× spatial compression)
            if "seg" in z:
                seg = torch.from_numpy(np.asarray(z["seg"], dtype=np.int8))  # (D, H, W)
                seg_path = case_out / f"{cid}-seg.pt"
                if not seg_path.exists():
                    torch.save(seg, seg_path)

        for ch_idx, key in [
            (CH_T1CE, "t1c"), (CH_T1N, "t1n"), (CH_T2F, "t2f"), (CH_T2W, "t2w"),
        ]:
            mu_path    = case_out / f"{cid}-{key}_z_mu.pt"
            sigma_path = case_out / f"{cid}-{key}_z_sigma.pt"
            if mu_path.exists() and sigma_path.exists():
                continue  # already cached for this channel

            try:
                mu, sigma = encode_channel(vae, data[ch_idx], device, use_amp)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                # Fall back to CPU for oversized volumes
                torch.cuda.empty_cache()
                cpu = torch.device("cpu")
                vae.ae.to(cpu)
                mu, sigma = encode_channel(vae, data[ch_idx], cpu, use_amp=False)
                vae.ae.to(device)
                torch.cuda.empty_cache()
            torch.save(mu,    mu_path)
            torch.save(sigma, sigma_path)

        done += 1

    print(f"Done. Encoded {done} cases, skipped {skipped} (already cached).")
    print(f"Latents saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
