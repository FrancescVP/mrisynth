"""Create axial-slice GIFs comparing input / prediction / ground truth.

For each case in the predictions directory, produces a side-by-side animated
GIF: T1n input | predicted T1CE | ground-truth T1CE, scrolling through axial
slices.

Usage
-----
  uv run python scripts/make_gifs.py \
      --pred_dir predictions/pix2pix_full_2047 \
      --fps 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import nibabel as nib
import numpy as np


def pct_norm(vol: np.ndarray) -> np.ndarray:
    """Percentile-normalise a volume to [0, 255] uint8."""
    lo = np.percentile(vol, 0.5)
    hi = np.percentile(vol, 99.5)
    vol = np.clip(vol, lo, hi)
    if hi > lo:
        vol = (vol - lo) / (hi - lo)
    return (vol * 255).astype(np.uint8)


def make_gif(case_dir: Path, fps: int) -> Path:
    t1n  = nib.load(str(case_dir / "input_t1n.nii.gz")).get_fdata(dtype=np.float32)
    pred = nib.load(str(case_dir / "pred_t1ce.nii.gz")).get_fdata(dtype=np.float32)
    gt   = nib.load(str(case_dir / "gt_t1ce.nii.gz")).get_fdata(dtype=np.float32)

    t1n  = pct_norm(t1n)
    pred = pct_norm(pred)
    gt   = pct_norm(gt)

    D = t1n.shape[2]   # axial slices along last dim (RAS: D=axial)
    sep = np.zeros((t1n.shape[0], 4), dtype=np.uint8)  # 4-px vertical separator

    frames = []
    for i in range(D):
        sl_t1n  = t1n[..., i]
        sl_pred = pred[..., i]
        sl_gt   = gt[..., i]

        # Stack horizontally: T1n | pred | GT
        row = np.concatenate([sl_t1n, sep, sl_pred, sep, sl_gt], axis=1)

        # Convert to RGB and add label row
        rgb = np.stack([row, row, row], axis=-1)

        frames.append(rgb)

    out_path = case_dir / "comparison.gif"
    iio.imwrite(
        str(out_path),
        frames,
        plugin="pillow",
        format="GIF",
        duration=int(1000 / fps),
        loop=0,
    )
    return out_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pred_dir", default="predictions/pix2pix_full_2047",
                   help="Directory containing per-case prediction subdirectories.")
    p.add_argument("--fps", type=int, default=8, help="Frames per second.")
    args = p.parse_args()

    pred_root = Path(args.pred_dir)
    cases = sorted(d for d in pred_root.iterdir() if d.is_dir())

    for case_dir in cases:
        if not (case_dir / "pred_t1ce.nii.gz").exists():
            continue
        out = make_gif(case_dir, args.fps)
        print(f"  {out}")

    print(f"\nDone. {len(cases)} GIFs saved.")


if __name__ == "__main__":
    main()
