"""Evaluate predicted/GT NIfTI pairs in a directory using the project metrics.

Convention assumed (matches /home/pariente/Downloads/predictions/multimodal_140):
    {CASE}_gt_{modality}.nii.gz
    {CASE}_pred_{modality}.nii.gz

Computes MAE, MSE, NMSE, PSNR, SSIM per case. Treats the volume as a
single-channel 3D tensor (1, 1, D, H, W).

Both volumes are min-max normalized into [0, 1] using the GT range so PSNR
and SSIM use a consistent dynamic range across cases. Use --no-rescale to
score on raw intensities.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gan_data_processing.losses import (  # noqa: E402
    region_weighted_ssim_loss,
    seg_to_mask,
    tumor_ssim_loss,
)
from gan_data_processing.metrics import (  # noqa: E402
    et_metrics,
    mae,
    masked_ssim,
    mse,
    nmse,
    psnr,
    ssim,
)


PAIR_RE = re.compile(r"^(?P<case>.+)_(?P<kind>gt|pred)_(?P<mod>.+)\.nii(?:\.gz)?$")


def find_pairs(d: Path) -> list[tuple[str, str, Path, Path]]:
    """Return list of (case, modality, gt_path, pred_path)."""
    bucket: dict[tuple[str, str], dict[str, Path]] = {}
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        key = (m["case"], m["mod"])
        bucket.setdefault(key, {})[m["kind"]] = p

    pairs = []
    for (case, mod), parts in sorted(bucket.items()):
        if "gt" in parts and "pred" in parts:
            pairs.append((case, mod, parts["gt"], parts["pred"]))
    return pairs


def load_volume(p: Path) -> np.ndarray:
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def load_seg_aligned_to(seg_path: Path, ref_nifti: Path) -> np.ndarray:
    """Resample seg into the geometry of ref_nifti (nearest-neighbor)."""
    ref = nib.load(ref_nifti)
    seg_img = nib.load(seg_path)
    if seg_img.shape == ref.shape and np.allclose(seg_img.affine, ref.affine):
        return np.asanyarray(seg_img.dataobj).astype(np.int16)
    aligned = resample_from_to(seg_img, ref, order=0)
    return np.asanyarray(aligned.dataobj).astype(np.int16)


def rescale_to_unit(vol: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    span = max(ref_max - ref_min, 1e-8)
    return np.clip((vol - ref_min) / span, 0.0, 1.0)


def evaluate(
    gt: np.ndarray,
    pred: np.ndarray,
    rescale: bool,
    win_size: int,
    device: torch.device,
    seg: Optional[np.ndarray] = None,
    tumor_classes: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float]:
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    if rescale:
        lo, hi = float(gt.min()), float(gt.max())
        gt_n = rescale_to_unit(gt, lo, hi)
        pred_n = rescale_to_unit(pred, lo, hi)
        max_value = 1.0
    else:
        gt_n, pred_n = gt, pred
        max_value = float(max(gt.max(), pred.max()))

    g = torch.from_numpy(gt_n).to(device).unsqueeze(0).unsqueeze(0)
    p = torch.from_numpy(pred_n).to(device).unsqueeze(0).unsqueeze(0)

    out = {
        "mae": mae(p, g).item(),
        "mse": mse(p, g).item(),
        "nmse": nmse(p, g).item(),
        "psnr": psnr(p, g, max_value=max_value).item(),
        "ssim": ssim(p, g, max_value=max_value, win_size=win_size).item(),
    }

    if seg is not None:
        seg_t = torch.from_numpy(seg.astype(np.int64)).to(device).unsqueeze(0)  # (1,D,H,W)
        # Whole-tumor SSIM (1 - loss = SSIM-in-tumor).
        whole_loss = tumor_ssim_loss(
            p, g, seg_t,
            tumor_classes=tumor_classes,
            max_value=max_value,
            win_size=win_size,
        ).item()
        out["tumor_ssim"] = 1.0 - whole_loss
        out["tumor_ssim_loss"] = whole_loss

        # Per-region SSIM (NETC=1, SNFH=2, ET=3).
        for cls, name in [(1, "NETC"), (2, "SNFH"), (3, "ET")]:
            mask = seg_to_mask(seg_t, [cls])
            if mask.sum() == 0:
                out[f"ssim_{name}"] = float("nan")
                continue
            s = masked_ssim(p, g, mask, max_value=max_value, win_size=win_size).item()
            out[f"ssim_{name}"] = s

        # Region-weighted loss (equal weights 1/2/3).
        rw = region_weighted_ssim_loss(
            p, g, seg_t,
            weights={1: 1.0, 2: 1.0, 3: 1.0},
            max_value=max_value,
            win_size=win_size,
        ).item()
        out["rw_ssim_loss"] = rw

        # MAE inside tumor (sanity).
        tumor_mask = seg_to_mask(seg_t, list(tumor_classes))
        if tumor_mask.sum() > 0:
            tm_b = tumor_mask.unsqueeze(1).bool()  # (1,1,D,H,W)
            out["tumor_mae"] = (p - g).abs()[tm_b].mean().item()
        else:
            out["tumor_mae"] = float("nan")

        out["tumor_voxels"] = int(tumor_mask.sum().item())

        # ET-specific metrics (only meaningful when ET voxels present).
        out.update(
            et_metrics(p, g, seg_t, et_class=3, max_value=max_value, win_size=min(win_size, 7))
        )

    return out


def find_seg(seg_dir: Optional[Path], case: str) -> Optional[Path]:
    if seg_dir is None:
        return None
    for ext in (".nii.gz", ".nii"):
        cand = seg_dir / f"{case}{ext}"
        if cand.exists():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument(
        "--seg-dir",
        type=Path,
        default=None,
        help="Optional dir of NIfTI segmentations named {CASE}.nii.gz. "
             "Enables tumor-aware metrics. Resampled to pred geometry (NN).",
    )
    ap.add_argument("--no-rescale", action="store_true", help="Score on raw intensities.")
    ap.add_argument("--win-size", type=int, default=11, help="SSIM Gaussian window (odd).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    device = torch.device(args.device)
    pairs = find_pairs(args.dir)
    if not pairs:
        print("No pairs found.")
        return

    use_tumor = args.seg_dir is not None
    rows = []

    if use_tumor:
        header = (
            f"{'case':<28} {'mod':<6} "
            f"{'ssim':>5} | "
            f"{'tu_ssim':>7} {'NETC':>5} {'SNFH':>5} {'ET':>5} | "
            f"{'ET_mae':>6} {'ET_pear':>7} {'ET_edge':>7} {'ET_ctrR':>7} {'ET_loc':>6} {'ET_vox':>7}"
        )
    else:
        header = (
            f"{'case':<28} {'mod':<6} "
            f"{'mae':>8} {'mse':>10} {'nmse':>10} {'psnr':>7} {'ssim':>7}"
        )
    print(header)
    print("-" * len(header))

    for case, mod, gt_p, pred_p in pairs:
        gt = load_volume(gt_p)
        pred = load_volume(pred_p)
        seg = None
        if use_tumor:
            seg_path = find_seg(args.seg_dir, case)
            if seg_path is None:
                print(f"[warn] no seg for {case}, skipping tumor metrics")
            else:
                seg = load_seg_aligned_to(seg_path, gt_p)
                if seg.shape != gt.shape:
                    print(f"[warn] seg/gt shape mismatch after resample: {seg.shape} vs {gt.shape}")
                    seg = None

        m = evaluate(
            gt, pred,
            rescale=not args.no_rescale,
            win_size=args.win_size,
            device=device,
            seg=seg,
        )
        rows.append({"case": case, "modality": mod, **m})

        if use_tumor and "tumor_ssim" in m:
            print(
                f"{case:<28} {mod:<6} "
                f"{m['ssim']:>5.3f} | "
                f"{m['tumor_ssim']:>7.3f} "
                f"{_fmt3(m.get('ssim_NETC')):>5} "
                f"{_fmt3(m.get('ssim_SNFH')):>5} "
                f"{_fmt3(m.get('ssim_ET')):>5} | "
                f"{_fmt3(m.get('et_mae')):>6} "
                f"{_fmt3(m.get('et_pearson')):>7} "
                f"{_fmt3(m.get('et_edge_ssim')):>7} "
                f"{_fmt3(m.get('et_contrast_rel_err')):>7} "
                f"{_fmt3(m.get('et_loc_dice')):>6} "
                f"{m.get('et_voxels', 0):>7d}"
            )
        else:
            print(
                f"{case:<28} {mod:<6} "
                f"{m['mae']:>8.4f} {m['mse']:>10.4e} {m['nmse']:>10.4e} "
                f"{m['psnr']:>7.2f} {m['ssim']:>7.4f}"
            )

    if rows:
        keys = ["mae", "mse", "nmse", "psnr", "ssim"]
        if use_tumor:
            keys += ["tumor_ssim", "ssim_NETC", "ssim_SNFH", "ssim_ET", "rw_ssim_loss", "tumor_mae"]
        print("-" * len(header))
        for label, fn in [("MEAN", np.nanmean), ("STD", np.nanstd)]:
            print(f"{label:<28} {'':6} ", end="")
            parts = []
            for k in keys:
                vals = [r.get(k, float("nan")) for r in rows]
                vals = [v for v in vals if v is not None]
                if not vals:
                    parts.append("    -")
                    continue
                parts.append(f"{fn(np.array(vals, dtype=float)):>7.4f}")
            print("  ".join(parts))

    if args.csv:
        import csv
        all_keys = sorted({k for r in rows for k in r.keys() if k not in ("case", "modality")})
        fields = ["case", "modality"] + all_keys
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")


def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  -  "
    return f"{x:.4f}"


def _fmt3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  -  "
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
