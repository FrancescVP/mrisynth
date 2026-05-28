"""Inference script for the Pix2Pix3D model.

Loads a trained generator checkpoint, runs full-volume inference on validation
cases, and saves predictions + ground truth + inputs as NIfTI files.

Usage
-----
  uv run python scripts/infer_pix2pix.py \\
      --data_dir    /path/to/preprocessed \\
      --checkpoint  checkpoints/pix2pix_full_2047/100_net_G.pth \\
      --out_dir     predictions/pix2pix_full_2047 \\
      --n_cases     10 \\
      --device      cuda:0
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
import torch.nn.functional as F

from mrisynth.model.pix2pix3d import Pix2Pix3dModel
from mrisynth.model.dataset import CHANNEL_NAMES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_affine(data_dir: Path, case_id: str) -> np.ndarray:
    candidates = list(data_dir.glob(f"{case_id}*.json"))
    if candidates:
        meta = json.loads(candidates[0].read_text())
        return np.array(meta["affine"], dtype=np.float64)
    return np.eye(4)


def save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))
    print(f"  saved {path}")


def infer_volume(model: Pix2Pix3dModel, A: torch.Tensor, num_downs: int = 5) -> torch.Tensor:
    """Run generator on a full volume, padding spatial dims to multiples of 2**num_downs.

    With num_downs=5 (default), factor=32 is required. Using 16 (2**4) caused
    dimensions divisible by 16 but not 32 (e.g. 144) to lose slices via silent
    skip-connection trimming in UnetSkipConnectionBlock3D, producing predictions
    shorter than the input and missing the upper brain.
    """
    D, H, W = A.shape[-3:]
    factor = 2 ** num_downs
    pd = (factor - D % factor) % factor
    ph = (factor - H % factor) % factor
    pw = (factor - W % factor) % factor
    if pd or ph or pw:
        A = F.pad(A, (0, pw, 0, ph, 0, pd))

    with torch.no_grad():
        fake_B = model.netG(A)

    return fake_B[..., :D, :H, :W]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   required=True, help="Preprocessed .npz directory.")
    p.add_argument("--checkpoint", required=True, help="Generator .pth checkpoint.")
    p.add_argument("--out_dir",    default="predictions/pix2pix")
    p.add_argument("--n_cases",    type=int, default=10)
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Same split used during training (to pick val cases).")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--ngf",        type=int, default=64)
    p.add_argument("--num_downs",  type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # ── Discover val split (same logic as training) ──────────────────────────
    all_cases = sorted(p.stem for p in data_dir.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_cases)
    n_val = max(1, int(len(all_cases) * args.val_fraction))
    val_cases = all_cases[:n_val]
    cases = val_cases[: args.n_cases]
    print(f"Val split: {n_val} cases — running inference on {len(cases)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    opt = SimpleNamespace(
        isTrain=False,
        checkpoints_dir=str(Path(args.checkpoint).parents[1]),
        name=Path(args.checkpoint).parent.name,
        device=device,
        verbose=False,
        init_type="kaiming",
        init_gain=0.02,
        norm="instance",
        ngf=args.ngf,
        ndf=32,
        num_downs=args.num_downs,
        n_layers_D=3,
        input_nc=2,   # T1n + T2FLAIR
        output_nc=1,  # T1CE
        no_dropout=False,
        gan_mode="lsgan",
        direction="AtoB",
    )
    model = Pix2Pix3dModel(opt)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.netG.load_state_dict(state)
    model.netG.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    # Channel indices: T1n=1, T2FLAIR=2, T1CE=0
    CH_T1N   = 1
    CH_T2F   = 2
    CH_T1CE  = 0

    for cid in cases:
        npz_path = data_dir / f"{cid}.npz"
        with np.load(npz_path) as z:
            data = np.asarray(z["data"], dtype=np.float32)  # (4, D, H, W)

        affine = get_affine(data_dir, cid)

        t1n  = torch.from_numpy(data[CH_T1N]).unsqueeze(0).unsqueeze(0)   # (1,1,D,H,W)
        t2f  = torch.from_numpy(data[CH_T2F]).unsqueeze(0).unsqueeze(0)
        t1ce = data[CH_T1CE]                                               # (D,H,W) numpy

        A = torch.cat([t1n, t2f], dim=1).to(device)  # (1,2,D,H,W)

        pred = infer_volume(model, A, num_downs=args.num_downs)  # (1,1,D,H,W)
        pred_vol = pred[0, 0].cpu().float().numpy()

        print(f"[{cid}]")
        save_nifti(pred_vol, affine, out_dir / cid / "pred_t1ce.nii.gz")
        save_nifti(t1ce,    affine, out_dir / cid / "gt_t1ce.nii.gz")
        save_nifti(data[CH_T1N], affine, out_dir / cid / "input_t1n.nii.gz")
        save_nifti(data[CH_T2F], affine, out_dir / cid / "input_t2flair.nii.gz")

        print(f"  pred  min={pred_vol.min():.3f}  max={pred_vol.max():.3f}  mean={pred_vol.mean():.3f}")
        print(f"  gt    min={t1ce.min():.3f}  max={t1ce.max():.3f}  mean={t1ce.mean():.3f}")
        print()

    print(f"Done. Saved to {out_dir}/")


if __name__ == "__main__":
    main()
