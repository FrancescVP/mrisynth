"""Train a Conditional Wavelet Diffusion Model (cWDM) for MRI synthesis.

Same as train_wfm.py but uses DDPM/DDIM instead of Rectified Flow.
No VAE needed — operates directly on image-space wavelet coefficients.

Quick start:
  uv run python scripts/train_cwdm.py \\
      --task   t1n_t2f_to_t1c \\
      --data_dir /path/to/preprocessed \\
      --name   cwdm_best \\
      --device cuda:0
"""
from __future__ import annotations

import argparse
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

from mrisynth.metrics import mae, ssim
from mrisynth.model import Pix2Pix3dDataset, cWDMModel


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "t1n_to_t2w":     {"input": ["T1n"],        "target": ["T2w"],  "n_cond": 1, "label": "T1w → T2w"},
    "t1n_to_t2f":     {"input": ["T1n"],        "target": ["T2F"],  "n_cond": 1, "label": "T1w → FLAIR"},
    "t1n_t2f_to_t1c": {"input": ["T1n", "T2F"], "target": ["T1CE"], "n_cond": 2, "label": "T1w + T2FLAIR → T1CE"},
    "t1n_t2w_to_t1c": {"input": ["T1n", "T2w"], "target": ["T1CE"], "n_cond": 2, "label": "T1w + T2w → T1CE"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_norm(img: torch.Tensor) -> torch.Tensor:
    flat = img.float().flatten()
    lo = torch.quantile(flat, 0.005)
    hi = torch.quantile(flat, 0.995)
    return ((img.float() - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _midslice(vol: torch.Tensor) -> torch.Tensor:
    return vol[:, :, vol.shape[2] // 2].contiguous()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model: cWDMModel, val_loader: DataLoader, device: torch.device,
                   max_cases: int) -> tuple[dict[str, float], tuple | None]:
    model.eval()
    accum: dict[str, list[float]] = defaultdict(list)
    vis = None

    for i, batch in enumerate(val_loader):
        if i >= max_cases:
            break

        model.set_input(batch)
        model.forward()

        pred = model.pred_image.cpu().float()
        gt   = batch["B"].float()

        for b in range(pred.shape[0]):
            p = pred[b:b+1]
            g = gt[b:b+1]
            dr = float((g.max() - g.min()).clamp(min=1e-6))
            accum["mae"].append(mae(p, g).item())
            accum["ssim"].append(ssim(p, g, max_value=dr).item())

        vis = (pred.cpu(), gt.cpu())

    model.train_mode()
    avg = {k: sum(vs) / len(vs) for k, vs in accum.items() if vs}
    return avg, vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train cWDM for MRI synthesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("task")
    g.add_argument("--task", default=None, choices=list(TASKS))
    g.add_argument("--input_channels", nargs="+", default=["T1n", "T2F"])
    g.add_argument("--target_channels", nargs="+", default=["T1CE"])
    g.add_argument("--n_cond", type=int, default=2)

    g = p.add_argument_group("data")
    g.add_argument("--data_dir", default=None)
    g.add_argument("--train_dir", default=None)
    g.add_argument("--val_dir", default=None)
    g.add_argument("--modality_dropout", type=float, default=0.0,
                   help="Fraction of TRAIN samples with one input modality "
                        "zeroed (robustness to a missing sequence). Needs >1 "
                        "input channel; 0 disables. Val/inference unaffected.")

    g = p.add_argument_group("model")
    g.add_argument("--backbone", default="unet", choices=["unet", "dit"])
    g.add_argument("--unet_channels", nargs="+", type=int, default=[64, 128, 256, 256])
    g.add_argument("--num_res_blocks", type=int, default=2)
    g.add_argument("--n_attention_levels", type=int, default=2)
    g.add_argument("--dit_patch_size", type=int, default=2)
    g.add_argument("--dit_hidden_size", type=int, default=384)
    g.add_argument("--dit_depth", type=int, default=12)
    g.add_argument("--dit_num_heads", type=int, default=6)
    g.add_argument("--ema_decay", type=float, default=0.0)
    g.add_argument("--grad_clip", type=float, default=0.0)
    g.add_argument("--n_timesteps", type=int, default=1000)
    g.add_argument("--n_ddim_steps", type=int, default=50)
    g.add_argument("--beta_schedule", default="linear",
                   choices=["linear", "scaled_linear"])

    g = p.add_argument_group("optimiser")
    g.add_argument("--lr",             type=float, default=1e-4)
    g.add_argument("--beta1",          type=float, default=0.9)
    g.add_argument("--weight_decay",   type=float, default=1e-4)
    g.add_argument("--n_epochs",       type=int,   default=300)
    g.add_argument("--n_epochs_decay", type=int,   default=100)
    g.add_argument("--lr_policy",      default="linear", choices=["linear", "step", "cosine"])
    g.add_argument("--lr_decay_iters", type=int,   default=50)

    g = p.add_argument_group("checkpoints / logging")
    g.add_argument("--name",            default=None)
    g.add_argument("--checkpoints_dir", default="./checkpoints")
    g.add_argument("--log_dir",         default="./runs")
    g.add_argument("--save_every",      type=int, default=10)
    g.add_argument("--val_every",       type=int, default=5)
    g.add_argument("--val_max_cases",   type=int, default=20)
    g.add_argument("--continue_train",  action="store_true")
    g.add_argument("--epoch",           default="latest")
    g.add_argument("--load_iter",       type=int, default=0)

    g = p.add_argument_group("misc")
    g.add_argument("--device",           default="cuda:0")
    g.add_argument("--num_workers",      type=int, default=4)
    g.add_argument("--batch_size",       type=int, default=1)
    g.add_argument("--patch_size",       nargs=3, type=int, default=None,
                   metavar=("D", "H", "W"),
                   help="Crop patch for training. None = full volume (OOM risk). "
                        "Recommended: 64 64 64. DWT requires even dims.")
    g.add_argument("--patches_per_volume", type=int, default=4,
                   help="Random crops per volume per epoch when --patch_size is set.")

    return p.parse_args()


def _resolve_dirs(args):
    if args.train_dir:
        train_dir = Path(args.train_dir)
    elif args.data_dir:
        train_dir = Path(args.data_dir) / "train"
    else:
        raise ValueError("Provide --data_dir or --train_dir.")

    if args.val_dir:
        val_dir = Path(args.val_dir)
    elif args.data_dir:
        val_dir = Path(args.data_dir) / "val"
    else:
        val_dir = None

    return train_dir, val_dir


def _build_opt(args, n_cond):
    return SimpleNamespace(
        isTrain           = True,
        checkpoints_dir   = args.checkpoints_dir,
        name              = args.name,
        device            = torch.device(args.device),
        verbose           = False,
        init_type         = "kaiming",
        init_gain         = 0.02,
        continue_train    = args.continue_train,
        epoch             = args.epoch,
        load_iter         = args.load_iter,
        norm              = "instance",
        lr_policy         = args.lr_policy,
        n_epochs          = args.n_epochs,
        n_epochs_decay    = args.n_epochs_decay,
        epoch_count       = 1,
        lr_decay_iters    = args.lr_decay_iters,
        lr                = args.lr,
        beta1             = args.beta1,
        weight_decay      = args.weight_decay,
        n_cond            = n_cond,
        backbone          = args.backbone,
        unet_channels     = args.unet_channels,
        num_res_blocks    = args.num_res_blocks,
        n_attention_levels = args.n_attention_levels,
        dit_patch_size    = args.dit_patch_size,
        dit_hidden_size   = args.dit_hidden_size,
        dit_depth         = args.dit_depth,
        dit_num_heads     = args.dit_num_heads,
        n_timesteps       = args.n_timesteps,
        n_ddim_steps      = args.n_ddim_steps,
        beta_schedule     = args.beta_schedule,
        ema_decay         = args.ema_decay,
        grad_clip         = args.grad_clip,
        use_ot_coupling   = False,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.task is not None:
        cfg = TASKS[args.task]
        input_channels  = cfg["input"]
        target_channels = cfg["target"]
        n_cond          = cfg["n_cond"]
        task_label      = cfg["label"]
    else:
        input_channels  = args.input_channels
        target_channels = args.target_channels
        n_cond          = args.n_cond
        task_label      = f"{input_channels} → {target_channels}"

    train_dir, val_dir = _resolve_dirs(args)
    name = args.name or args.task or "cwdm"
    args.name = name
    device = torch.device(args.device)

    patch_size = tuple(args.patch_size) if args.patch_size else None
    train_ds = Pix2Pix3dDataset(
        train_dir, input_channels=input_channels,
        target_channels=target_channels,
        patch_size=patch_size,
        augment=patch_size is not None,
        patches_per_volume=args.patches_per_volume if patch_size else 1,
        modality_dropout=args.modality_dropout,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    val_loader = None
    val_ds = None
    if val_dir and Path(val_dir).exists():
        val_ds = Pix2Pix3dDataset(
            val_dir, input_channels=input_channels,
            target_channels=target_channels,
            patch_size=patch_size,
            augment=False,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    opt = _build_opt(args, n_cond)
    model = cWDMModel(opt)
    model.setup(opt)
    model.train_mode()

    writer = SummaryWriter(str(Path(args.log_dir) / name))
    n_total = args.n_epochs + args.n_epochs_decay

    print("=" * 60)
    print(f"  Task        : {task_label}")
    print(f"  Experiment  : {name}")
    print(f"  Train cases : {len(train_ds)}")
    print(f"  Val cases   : {len(val_ds) if val_ds else 0}")
    print(f"  Epochs      : {args.n_epochs} + {args.n_epochs_decay} decay = {n_total}")
    print(f"  Backbone    : {args.backbone}  channels={args.unet_channels}")
    print(f"  DDIM steps  : {args.n_ddim_steps}")
    print(f"  Device      : {device}")
    print("=" * 60)

    for epoch in range(1, n_total + 1):
        epoch_start = time.time()
        loss_accum = 0.0

        model.train_mode()
        for batch in train_loader:
            model.set_input(batch)
            model.optimize_parameters()
            loss_accum += model.get_current_losses()["cwdm"]

        epoch_loss = loss_accum / max(len(train_loader), 1)
        elapsed = time.time() - epoch_start

        writer.add_scalar("loss/cwdm_train", epoch_loss, epoch)
        model.schedulers[0].step()

        current_lr = model.optimizers[0].param_groups[0]["lr"]
        print(f"[{epoch:4d}/{n_total}]  cwdm: {epoch_loss:.4f}  lr: {current_lr:.2e}  ({elapsed:.0f}s)")

        do_val = args.val_every > 0 and epoch % args.val_every == 0 and val_loader is not None
        if do_val:
            avg, vis = run_validation(model, val_loader, device, args.val_max_cases)
            for k, v in avg.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print("  val  " + "  ".join(f"{k}:{v:.4f}" for k, v in avg.items()))

            if vis is not None:
                pred_img, gt_img = vis
                writer.add_images(
                    "image/pred", _pct_norm(_midslice(pred_img[:1, :1])), epoch
                )
                writer.add_images(
                    "image/gt", _pct_norm(_midslice(gt_img[:1, :1])), epoch
                )

        if epoch % args.save_every == 0:
            model.save_networks(epoch)
        model.save_networks("latest")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
