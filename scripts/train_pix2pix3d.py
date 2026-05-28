"""Training loop for 3-D pix2pix MRI synthesis with TensorBoard logging.

Metrics tracked
---------------
  Training   : G_GAN, G_pixel, G_feat, D_real, D_fake, G_total  (per epoch)
  Validation : MAE, MSE, NMSE, SSIM
               ET-specific (when ET voxels present):
                 et_mae, et_pearson, et_edge_ssim, et_loc_dice,
                 et_contrast_abs_err, et_contrast_rel_err

Images saved to TensorBoard (middle axial slice, percentile-normalised)
  real_A   — input channel(s)
  fake_B   — generator prediction
  real_B   — ground-truth target
  abs_err  — |fake_B - real_B|
  ssim_map — per-voxel SSIM map

Usage
-----
  uv run python scripts/train_pix2pix3d.py \\
      --data_dir  /path/to/preprocessed \\
      --name      t1n_to_t1ce \\
      --input_channels T1n \\
      --target_channels T1CE \\
      --device cuda

  tensorboard --logdir runs/
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mrisynth.metrics import (
    mae,
    mse,
    nmse,
    ssim,
    ssim_map,
    et_metrics,
)
from mrisynth.model import Pix2Pix3dDataset, Pix2Pix3dModel, resolve_channels


# ---------------------------------------------------------------------------
# Image helpers for TensorBoard
# ---------------------------------------------------------------------------

def _midslice(vol: torch.Tensor) -> torch.Tensor:
    """(N, C, D, H, W) → middle axial slice (N, C, H, W)."""
    mid = vol.shape[2] // 2
    return vol[:, :, mid].contiguous()


def _pct_norm(img: torch.Tensor, plo: float = 0.5, phi: float = 99.5) -> torch.Tensor:
    """Percentile-clip and normalise to [0, 1] for display.

    Works on any shape — percentiles are computed over the whole tensor.
    """
    flat = img.float().flatten()
    lo = torch.quantile(flat, plo / 100.0)
    hi = torch.quantile(flat, phi / 100.0)
    return ((img.float() - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def log_images(
    writer: SummaryWriter,
    real_A: torch.Tensor,   # (N, C_in, D, H, W)
    fake_B: torch.Tensor,   # (N, 1, D, H, W)
    real_B: torch.Tensor,   # (N, 1, D, H, W)
    seg: torch.Tensor,      # (N, 1, D, H, W) int — may be None
    step: int,
    n_images: int = 4,
) -> None:
    """Log middle-slice comparison images to TensorBoard.

    Args:
        n_images: max number of batch samples to show per panel.
    """
    N = min(n_images, real_A.shape[0])
    fake_B = fake_B[:N].cpu().float()
    real_B = real_B[:N].cpu().float()
    real_A = real_A[:N].cpu().float()

    # Trim real_B to pred size (odd-dimension UNet mismatch)
    real_B = real_B[:, :, : fake_B.shape[2], : fake_B.shape[3], : fake_B.shape[4]]

    # Middle axial slices
    slc_A   = _midslice(real_A)               # (N, C_in, H, W)
    slc_pred = _midslice(fake_B)              # (N, 1, H, W)
    slc_gt   = _midslice(real_B)              # (N, 1, H, W)

    # Log each input channel
    for ch in range(slc_A.shape[1]):
        writer.add_images(
            f"images/real_A_ch{ch}",
            _pct_norm(slc_A[:, ch : ch + 1]),
            step,
        )

    writer.add_images("images/fake_B",  _pct_norm(slc_pred), step)
    writer.add_images("images/real_B",  _pct_norm(slc_gt),   step)

    # Absolute error map (normalised to [0, 1] by its own max)
    abs_err = (slc_pred - slc_gt).abs()
    amax = abs_err.amax().clamp(min=1e-6)
    writer.add_images("images/abs_error", (abs_err / amax), step)

    # Per-voxel SSIM map on the 2D slice (directly in [0, 1])
    try:
        dr = float(slc_gt.max() - slc_gt.min())
        sm = ssim_map(slc_pred, slc_gt, max_value=max(dr, 1e-6), same_size=True)
        writer.add_images("images/ssim_map", sm.clamp(0, 1), step)
    except Exception:
        pass  # skip if slice is too small for win_size

    # Segmentation overlay (label 3 = ET in red, label 1&2 = other tumor in gray)
    if seg is not None:
        seg_cpu = seg[:N].cpu()
        seg_cpu = seg_cpu[:, :, : fake_B.shape[2], : fake_B.shape[3], : fake_B.shape[4]]
        slc_seg = _midslice(seg_cpu.float())   # (N, 1, H, W)
        # Normalize label values to [0, 1] for grayscale display (0→0, 3→1)
        writer.add_images("images/seg_labels", (slc_seg / 3.0).clamp(0, 1), step)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model: Pix2Pix3dModel,
    val_loader: DataLoader,
    max_cases: int,
) -> tuple[dict[str, float], tuple | None]:
    """Run validation on up to `max_cases` full volumes.

    Returns
    -------
    metrics : averaged scalar metrics (NaN ET values are dropped)
    vis     : (real_A, fake_B, real_B, seg) tensors from the last batch
              — used for image logging. None if val_loader is empty.
    """
    model.eval()
    accum: dict[str, list[float]] = defaultdict(list)
    vis = None

    for i, batch in enumerate(val_loader):
        if i >= max_cases:
            break

        model.set_input(batch)
        model.forward()

        pred = model.fake_B.cpu().float()
        gt   = model.real_B.cpu().float()
        # Trim gt to pred size (odd-dimension UNet output mismatch)
        gt = gt[:, :, : pred.shape[2], : pred.shape[3], : pred.shape[4]]

        seg = batch.get("seg")
        if seg is not None:
            seg = seg.cpu()
            seg = seg[:, :, : pred.shape[2], : pred.shape[3], : pred.shape[4]]

        # ── Pixel-wise ──────────────────────────────────────────────────────
        # data_range: per-volume gt range, correct for Z-score normalised data
        # (max_value=1.0 would underestimate SSIM by ~0.25 on Z-score inputs)
        data_range = float(gt.max() - gt.min())
        accum["mae"].append(mae(pred, gt).item())
        accum["mse"].append(mse(pred, gt).item())
        accum["nmse"].append(nmse(pred, gt).item())
        accum["ssim"].append(ssim(pred, gt, max_value=data_range).item())

        # ── ET-specific ──────────────────────────────────────────────────────
        if seg is not None:
            # et_metrics expects (N, *spatial) or (N, 1, *spatial)
            seg_nd = seg.squeeze(1).long()
            et = et_metrics(pred, gt, seg_nd, max_value=data_range)
            for k, v in et.items():
                if k != "et_voxels" and not math.isnan(v):
                    accum[k].append(v)

        vis = (model.real_A.cpu().float(), pred, gt, seg)

    model.train_mode()

    avg = {k: sum(vs) / len(vs) for k, vs in accum.items() if vs}
    return avg, vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3-D pix2pix for MRI synthesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("data")
    g.add_argument("--data_dir", required=True,
                   help="Directory of preprocessed .npz files.")
    g.add_argument("--input_channels", nargs="+", default=["T1n"],
                   help="Input (A) channels — int indices or names: T1n, T1CE, FLAIR, T2w.")
    g.add_argument("--target_channels", nargs="+", default=["T1CE"],
                   help="Target (B) channels.")
    g.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128],
                   metavar=("D", "H", "W"), help="Training patch size.")
    g.add_argument("--patches_per_volume", type=int, default=4,
                   help="Random crops drawn per volume per epoch.")
    g.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of cases held out for validation.")

    # ── Model ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("model")
    g.add_argument("--ngf", type=int, default=64)
    g.add_argument("--ndf", type=int, default=32)
    g.add_argument("--num_downs", type=int, default=5,
                   help="U-Net depth (>=5).")
    g.add_argument("--norm", default="instance",
                   choices=["instance", "batch", "none"])
    g.add_argument("--no_dropout", action="store_true")
    g.add_argument("--n_layers_D", type=int, default=3)
    g.add_argument("--gan_mode", default="lsgan",
                   choices=["lsgan", "vanilla", "wgangp"])

    # ── Loss ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("loss")
    g.add_argument("--pixel_loss", default="l2", choices=["l1", "l2"],
                   help="Pixel reconstruction loss: l2 (MSE) or l1 (MAE).")
    g.add_argument("--lambda_pixel", type=float, default=100.0)
    g.add_argument("--lambda_feat", type=float, default=10.0,
                   help="Feature-matching weight. 0 = disabled.")
    g.add_argument("--d_update_freq", type=int, default=2,
                   help="Update D once every N generator steps.")
    g.add_argument("--use_tumor_loss", action="store_true",
                   help="Blend G loss with ET-aware tumor loss.")
    g.add_argument("--alpha_tumor", type=float, default=0.5,
                   help="Tumor loss blending weight (use_tumor_loss only).")

    # ── Optimiser / schedule ────────────────────────────────────────────────
    g = p.add_argument_group("optimiser")
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--beta1", type=float, default=0.5)
    g.add_argument("--n_epochs", type=int, default=100,
                   help="Epochs at constant LR.")
    g.add_argument("--n_epochs_decay", type=int, default=100,
                   help="Epochs over which LR linearly decays to 0.")
    g.add_argument("--lr_policy", default="linear",
                   choices=["linear", "step", "plateau", "cosine"])
    g.add_argument("--lr_decay_iters", type=int, default=50)

    # ── Checkpoints & logging ────────────────────────────────────────────────
    g = p.add_argument_group("checkpoints / logging")
    g.add_argument("--name", required=True,
                   help="Experiment name.")
    g.add_argument("--checkpoints_dir", default="./checkpoints",
                   help="Root for checkpoint files.")
    g.add_argument("--log_dir", default="./runs",
                   help="Root for TensorBoard event files.")
    g.add_argument("--save_every", type=int, default=10,
                   help="Save numbered checkpoint every N epochs.")
    g.add_argument("--val_every", type=int, default=5,
                   help="Validate + log images every N epochs (0 = off).")
    g.add_argument("--val_max_cases", type=int, default=20,
                   help="Max validation cases per validation run.")
    g.add_argument("--log_images_n", type=int, default=4,
                   help="Number of batch samples shown in image panels.")
    g.add_argument("--continue_train", action="store_true")
    g.add_argument("--epoch", default="latest",
                   help="Epoch label to resume from.")
    g.add_argument("--load_iter", type=int, default=0)

    # ── Misc ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("misc")
    g.add_argument("--device", default="cuda")
    g.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_opt(args: argparse.Namespace) -> SimpleNamespace:
    opt = SimpleNamespace(**vars(args))
    opt.isTrain   = True
    opt.verbose   = False
    opt.direction = "AtoB"
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.epoch_count = 1
    opt.pool_size   = 0

    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARN: CUDA not available, falling back to CPU.")
        dev = "cpu"
    opt.device = torch.device(dev)

    opt.input_nc  = len(resolve_channels(args.input_channels))
    opt.output_nc = len(resolve_channels(args.target_channels))
    return opt


def make_datasets(args: argparse.Namespace):
    all_cases = sorted(p.stem for p in Path(args.data_dir).glob("*.npz"))
    n_val = max(1, int(len(all_cases) * args.val_fraction)) if args.val_fraction > 0 else 0
    val_cases   = all_cases[:n_val]
    train_cases = all_cases[n_val:]

    train_ds = Pix2Pix3dDataset(
        args.data_dir,
        input_channels=args.input_channels,
        target_channels=args.target_channels,
        patch_size=tuple(args.patch_size),
        augment=True,
        patches_per_volume=args.patches_per_volume,
        case_ids=train_cases,
    )
    val_ds = Pix2Pix3dDataset(
        args.data_dir,
        input_channels=args.input_channels,
        target_channels=args.target_channels,
        patch_size=None,      # full volume
        augment=False,
        case_ids=val_cases,
    ) if n_val > 0 else None

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    opt = build_opt(args)

    train_ds, val_ds = make_datasets(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(opt.device.type == "cuda"),
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=1, num_workers=0)
        if val_ds is not None else None
    )

    model = Pix2Pix3dModel(opt)
    model.setup(opt)

    n_total = args.n_epochs + args.n_epochs_decay

    # TensorBoard
    log_path = Path(args.log_dir) / args.name
    writer = SummaryWriter(log_dir=str(log_path))

    # ── Print run summary ────────────────────────────────────────────────────
    sep = "─" * 66
    print(sep)
    print(f"  Experiment   : {args.name}")
    print(f"  Dataset      : {train_ds}")
    if val_ds:
        print(f"  Val          : {len(val_ds)} cases, every {args.val_every} epochs")
    print(f"  Steps/epoch  : {len(train_loader)}")
    print(f"  Pixel loss   : {args.pixel_loss.upper()}, λ={args.lambda_pixel}")
    print(f"  Feat-match   : λ={args.lambda_feat}")
    if args.use_tumor_loss:
        print(f"  Tumor loss   : ET-aware, α={args.alpha_tumor}")
    print(f"  Schedule     : {args.n_epochs} const + {args.n_epochs_decay} decay = {n_total}")
    print(f"  LR           : {args.lr}  ({args.lr_policy})")
    print(f"  Device       : {opt.device}")
    print(f"  TensorBoard  : {log_path}")
    print(sep)

    ckpt_dir = Path(args.checkpoints_dir) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_total + 1):
        t0 = time.time()
        accum: dict[str, float] = {}
        n_steps = 0

        # ── Training ────────────────────────────────────────────────────────
        model.train_mode()
        for batch in train_loader:
            model.set_input(batch)
            model.optimize_parameters()

            for k, v in model.get_current_losses().items():
                accum[k] = accum.get(k, 0.0) + v
            n_steps += 1

        avg = {k: v / n_steps for k, v in accum.items()}
        elapsed = time.time() - t0

        # ── LR schedule ─────────────────────────────────────────────────────
        model.update_learning_rate()
        lr_now = model.optimizers[0].param_groups[0]["lr"]

        # ── Log training scalars ────────────────────────────────────────────
        for k, v in avg.items():
            writer.add_scalar(f"train/loss_{k}", v, epoch)
        # Total G loss
        g_total = avg.get("G_GAN", 0) + avg.get("G_pixel", 0) + avg.get("G_feat", 0) + avg.get("G_tumor", 0)
        writer.add_scalar("train/loss_G_total", g_total, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        # ── Console ─────────────────────────────────────────────────────────
        loss_str = "  ".join(f"{k}:{v:.4f}" for k, v in avg.items())
        print(f"[{epoch:4d}/{n_total}]  {loss_str}  lr:{lr_now:.2e}  ({elapsed:.0f}s)")

        # ── Validation ──────────────────────────────────────────────────────
        if (
            val_loader is not None
            and args.val_every > 0
            and epoch % args.val_every == 0
        ):
            val_metrics, vis = run_validation(model, val_loader, args.val_max_cases)

            # Scalars
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            # Console
            core = {k: v for k in ("mae", "mse", "nmse", "ssim") if (v := val_metrics.get(k)) is not None}
            et_keys = [k for k in val_metrics if k.startswith("et_")]
            core_str = "  ".join(f"{k}:{v:.4f}" for k, v in core.items())
            et_str   = "  ".join(f"{k}:{val_metrics[k]:.4f}" for k in et_keys[:4])  # first 4
            print(f"           val  {core_str}")
            if et_str:
                print(f"           ET   {et_str}")

            # Images
            if vis is not None:
                real_A, fake_B, real_B, seg = vis
                log_images(writer, real_A, fake_B, real_B, seg, epoch, args.log_images_n)

        # ── Checkpoint ──────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            model.save_networks(epoch)
            model.save_networks("latest")
            print(f"           ckpt saved → {ckpt_dir}")

    # Final save
    model.save_networks("final")
    writer.close()
    print(sep)
    print(f"  Done. Checkpoints → {ckpt_dir}")
    print(f"        TensorBoard → tensorboard --logdir {Path(args.log_dir)}")
    print(sep)


if __name__ == "__main__":
    train(parse_args())
