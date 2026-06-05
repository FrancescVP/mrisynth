"""Train RFlowControlNet (latent diffusion + ControlNet seg conditioning).

Same as train_rflow.py but instantiates RFlowControlNetModel and uses seg masks
from LatentDataset for ControlNet conditioning.

Quick start:
  uv run python scripts/train_rflow_controlnet.py \\
      --task   t1n_t2f_to_t1c \\
      --latent_root latents/dataset500 \\
      --name   rflow_controlnet_best \\
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
from mrisynth.model import LatentDataset, RFlowControlNetModel


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "t1n_to_t2w":     {"target": "t2w", "conds": ["t1n"],        "n_cond": 1, "label": "T1w → T2w"},
    "t1n_to_t2f":     {"target": "t2f", "conds": ["t1n"],        "n_cond": 1, "label": "T1w → FLAIR"},
    "t1n_t2f_to_t1c": {"target": "t1c", "conds": ["t1n", "t2f"], "n_cond": 2, "label": "T1w + T2FLAIR → T1CE"},
    "t1n_t2w_to_t1c": {"target": "t1c", "conds": ["t1n", "t2w"], "n_cond": 2, "label": "T1w + T2w → T1CE"},
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
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    from torch.utils.data.dataloader import default_collate
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = None if vals[0] is None else default_collate(vals)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model: RFlowControlNetModel, val_loader: DataLoader,
                   vae, device: torch.device, max_cases: int):
    model.eval()
    accum: dict[str, list[float]] = defaultdict(list)
    vis = None

    model.scheduler.set_timesteps(model._n_inf)

    for i, batch in enumerate(val_loader):
        if i >= max_cases:
            break

        model.set_input(batch)
        mu_tgt = batch["mu_tgt"].to(device)
        model.forward()
        pred_lat = model.pred_latent

        lat_l1 = (pred_lat - mu_tgt).abs().mean().item()
        accum["rflow_val"].append(lat_l1)

        if vae is not None:
            for b in range(pred_lat.shape[0]):
                p_lat = pred_lat[b:b+1].to(device)
                g_lat = mu_tgt[b:b+1].to(device)
                pred_img = vae.decode(p_lat.float())
                gt_img   = vae.decode(g_lat.float())
                D, H, W = [min(pred_img.shape[i+2], gt_img.shape[i+2]) for i in range(3)]
                pred_img = pred_img[:, :, :D, :H, :W].cpu().float()
                gt_img   = gt_img[:, :, :D, :H, :W].cpu().float()
                dr = float(gt_img.max() - gt_img.min())
                accum["mae"].append(mae(pred_img, gt_img).item())
                accum["ssim"].append(ssim(pred_img, gt_img, max_value=max(dr, 1e-6)).item())

        vis = (pred_lat.cpu(), mu_tgt.cpu())

    model.train_mode()
    avg = {k: sum(vs) / len(vs) for k, vs in accum.items() if vs}
    return avg, vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train RFlowControlNet for MRI synthesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("task")
    g.add_argument("--task", default=None, choices=list(TASKS))
    g.add_argument("--latent_root", default=None)
    g.add_argument("--latent_dir", default=None)
    g.add_argument("--val_latent_dir", default=None)
    g.add_argument("--vae_ckpt", default=None)
    g.add_argument("--target_key", default="t1c")
    g.add_argument("--cond_keys", nargs="+", default=["t1n", "t2f"])
    g.add_argument("--n_cond", type=int, default=2)
    g.add_argument("--modality_dropout", type=float, default=0.0,
                   help="Fraction of TRAIN samples with one conditioning "
                        "modality zeroed (robustness to a missing sequence). "
                        "Needs >1 conditioning modality; 0 disables. Validation "
                        "is deterministic and never affected.")

    g = p.add_argument_group("model")
    g.add_argument("--latent_channels", type=int, default=4)
    g.add_argument("--unet_channels", nargs="+", type=int, default=[64, 128, 256, 256])
    g.add_argument("--num_res_blocks", type=int, default=2)
    g.add_argument("--n_attention_levels", type=int, default=2)
    g.add_argument("--ema_decay", type=float, default=0.0)
    g.add_argument("--grad_clip", type=float, default=0.0)
    g.add_argument("--n_timesteps", type=int, default=1000)
    g.add_argument("--n_inference_steps", type=int, default=200)
    g.add_argument("--velocity_loss", default="l1",
                   choices=["l1", "l2", "ssim", "ncc", "l1+ssim"])
    g.add_argument("--loss_alpha", type=float, default=0.5)
    g.add_argument("--use_ot_coupling", action="store_true")

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
    g.add_argument("--device",      default="cuda:0")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--batch_size",  type=int, default=1)

    return p.parse_args()


def _resolve_task(args):
    if args.task is not None:
        cfg = TASKS[args.task]
        return cfg["target"], cfg["conds"], cfg["n_cond"]
    return args.target_key, args.cond_keys, args.n_cond


def _resolve_dirs(args):
    if args.latent_dir:
        train_dir = Path(args.latent_dir)
    elif args.latent_root:
        train_dir = Path(args.latent_root) / "train"
    else:
        raise ValueError("Provide --latent_dir or --latent_root.")

    if args.val_latent_dir:
        val_dir = Path(args.val_latent_dir)
    elif args.latent_root:
        val_dir = Path(args.latent_root) / "val"
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
        latent_channels   = args.latent_channels,
        n_cond            = n_cond,
        backbone          = "unet",
        unet_channels     = args.unet_channels,
        num_res_blocks    = args.num_res_blocks,
        n_attention_levels = args.n_attention_levels,
        n_timesteps       = args.n_timesteps,
        n_inference_steps = args.n_inference_steps,
        velocity_loss     = args.velocity_loss,
        loss_alpha        = args.loss_alpha,
        tumor_weight      = None,
        use_ot_coupling   = args.use_ot_coupling,
        ema_decay         = args.ema_decay,
        grad_clip         = args.grad_clip,
        contrastive_weight = 0.1,
        contrastive_temp   = 0.1,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    target_key, cond_keys, n_cond = _resolve_task(args)
    train_dir, val_dir = _resolve_dirs(args)

    name = args.name or args.task or "rflow_controlnet"
    args.name = name
    device = torch.device(args.device)

    train_ds = LatentDataset(train_dir, target_key=target_key, cond_keys=cond_keys,
                             modality_dropout=args.modality_dropout)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    val_loader = None
    val_ds = None
    if val_dir and Path(val_dir).exists():
        val_ds = LatentDataset(val_dir, target_key=target_key, cond_keys=cond_keys,
                               deterministic=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=2, collate_fn=collate_fn)

    vae = None
    if args.vae_ckpt and Path(args.vae_ckpt).exists():
        from mrisynth.model.vae import MaisiVAE
        print(f"Loading VAE from {args.vae_ckpt} …")
        vae = MaisiVAE.from_checkpoint(args.vae_ckpt, device=device)
        vae.eval()

    opt = _build_opt(args, n_cond)
    model = RFlowControlNetModel(opt)
    model.setup(opt)
    model.train_mode()

    writer = SummaryWriter(str(Path(args.log_dir) / name))
    task_label = TASKS[args.task]["label"] if args.task else f"{cond_keys} → {target_key}"
    n_total = args.n_epochs + args.n_epochs_decay

    print("=" * 60)
    print(f"  Task        : {task_label}")
    print(f"  Experiment  : {name}")
    print(f"  Train cases : {len(train_ds)}")
    print(f"  Val cases   : {len(val_ds) if val_ds else 0}")
    print(f"  Epochs      : {args.n_epochs} + {args.n_epochs_decay} decay = {n_total}")
    print(f"  UNet chans  : {args.unet_channels}")
    print(f"  Device      : {device}")
    print("=" * 60)

    for epoch in range(1, n_total + 1):
        epoch_start = time.time()
        loss_accum = 0.0

        model.train_mode()
        for batch in train_loader:
            model.set_input(batch)
            model.optimize_parameters()
            loss_accum += model.get_current_losses()["rflow"]

        epoch_loss = loss_accum / max(len(train_loader), 1)
        elapsed = time.time() - epoch_start

        writer.add_scalar("loss/rflow_train", epoch_loss, epoch)
        model.schedulers[0].step()

        current_lr = model.optimizers[0].param_groups[0]["lr"]
        print(f"[{epoch:4d}/{n_total}]  rflow: {epoch_loss:.4f}  lr: {current_lr:.2e}  ({elapsed:.0f}s)")

        do_val = args.val_every > 0 and epoch % args.val_every == 0 and val_loader is not None
        if do_val:
            avg, vis = run_validation(model, val_loader, vae, device, args.val_max_cases)
            for k, v in avg.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print("  val  " + "  ".join(f"{k}:{v:.4f}" for k, v in avg.items()))

            if vis is not None:
                pred_lat, gt_lat = vis
                writer.add_images("latent/pred_ch0", _pct_norm(_midslice(pred_lat[:1, :1])), epoch)
                writer.add_images("latent/gt_ch0",   _pct_norm(_midslice(gt_lat[:1, :1])),   epoch)

        if epoch % args.save_every == 0:
            model.save_networks(epoch)
        model.save_networks("latest")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
