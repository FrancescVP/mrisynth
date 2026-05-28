"""Smoke-test both RFlow backbones (UNet and DiT) with synthetic data.

Runs 10 training epochs each on randomly generated latent tensors — no real
data or VAE checkpoint needed.  Verifies that both architectures:
  - train without errors
  - produce finite loss at every step
  - can run a full denoising forward pass after training

Usage
-----
  uv run python scripts/test_architectures.py
  uv run python scripts/test_architectures.py --device cuda:0
  uv run python scripts/test_architectures.py --epochs 20 --latent_size 16
"""
from __future__ import annotations

import argparse
import sys
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_loader(n_cases: int, lat_ch: int, spatial: int,
                n_cond: int, batch_size: int) -> DataLoader:
    """Return a DataLoader that yields synthetic latent batches."""
    tgt  = torch.randn(n_cases, lat_ch, spatial, spatial, spatial)
    cond = torch.randn(n_cases, lat_ch * n_cond, spatial, spatial, spatial)
    mu   = tgt.clone()
    ds   = TensorDataset(tgt, cond, mu)

    def collate(batch):
        t, c, m = zip(*batch)
        return {
            "latent_tgt":  torch.stack(t),
            "latent_cond": torch.stack(c),
            "mu_tgt":      torch.stack(m),
            "case_id":     [f"case{i:04d}" for i in range(len(t))],
            "seg":         None,
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)


# ---------------------------------------------------------------------------
# opt builder
# ---------------------------------------------------------------------------

def build_opt(tmp_dir: str, backbone: str, device: torch.device,
              lat_ch: int, n_cond: int, spatial: int, epochs: int) -> SimpleNamespace:
    # Schedule: constant LR for first half, linear decay to 0 over second half.
    # n_epochs + n_epochs_decay must equal the actual number of training epochs
    # or the linear scheduler goes negative and flips gradient direction.
    half = epochs // 2
    return SimpleNamespace(
        isTrain           = True,
        checkpoints_dir   = tmp_dir,
        name              = f"smoke_{backbone}",
        device            = device,
        verbose           = False,
        init_type         = "kaiming",
        init_gain         = 0.02,
        continue_train    = False,
        epoch             = "latest",
        load_iter         = 0,
        norm              = "instance",
        lr_policy         = "linear",
        n_epochs          = half,
        n_epochs_decay    = epochs - half,
        epoch_count       = 1,
        lr_decay_iters    = 5,
        lr                = 1e-4,
        beta1             = 0.9,
        weight_decay      = 1e-4,
        latent_channels   = lat_ch,
        n_cond            = n_cond,
        backbone          = backbone,
        # UNet
        unet_channels     = [32, 64],
        num_res_blocks    = 1,
        n_attention_levels = 1,
        # DiT
        dit_patch_size    = 2,
        dit_hidden_size   = 64,
        dit_depth         = 2,
        dit_num_heads     = 4,
        # shared
        n_timesteps       = 100,
        n_inference_steps = 10,
        velocity_loss     = "l1",
        loss_alpha        = 0.5,
        tumor_weight      = None,
        ema_decay         = 0.0,
        use_checkpointing = False,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(backbone: str, loader: DataLoader, device: torch.device,
        epochs: int, lat_ch: int, n_cond: int, spatial: int) -> list[float]:
    from mrisynth.model.rflow import RFlowModel

    with tempfile.TemporaryDirectory() as tmp:
        opt   = build_opt(tmp, backbone, device, lat_ch, n_cond, spatial, epochs)
        model = RFlowModel(opt)
        model.setup(opt)
        model.train_mode()

        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            t0   = time.time()
            accum = 0.0
            steps = 0

            for batch in loader:
                model.set_input(batch)
                model.optimize_parameters()
                loss = model.get_current_losses()["rflow"]
                assert torch.isfinite(torch.tensor(loss)), \
                    f"Non-finite loss at epoch {epoch}: {loss}"
                accum += loss
                steps += 1

            model.schedulers[0].step()
            avg = accum / steps
            epoch_losses.append(avg)
            print(f"    [{epoch:2d}/{epochs}]  loss: {avg:.4f}  ({time.time()-t0:.1f}s)")

        # Forward pass (denoising) after training
        model.eval()
        batch = next(iter(loader))
        model.set_input(batch)
        model.forward()
        pred = model.pred_latent
        assert pred.shape[1:] == (lat_ch, spatial, spatial, spatial), \
            f"Unexpected pred shape: {pred.shape}"
        assert torch.isfinite(pred).all(), "Non-finite values in pred_latent"
        print(f"    forward OK — pred shape: {tuple(pred.shape)}")

    return epoch_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test UNet and DiT backbones.")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--n_cases",      type=int,   default=8,
                   help="Number of synthetic training cases.")
    p.add_argument("--latent_size",  type=int,   default=8,
                   help="Spatial size of the synthetic latent (D=H=W).")
    p.add_argument("--lat_ch",       type=int,   default=4,
                   help="Latent channels (must match VAE, default 4).")
    p.add_argument("--n_cond",       type=int,   default=2,
                   help="Number of conditioning modalities.")
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--device",       default="cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    loader = make_loader(
        n_cases=args.n_cases,
        lat_ch=args.lat_ch,
        spatial=args.latent_size,
        n_cond=args.n_cond,
        batch_size=args.batch_size,
    )

    results: dict[str, list[float]] = {}

    for backbone in ("unet", "dit"):
        print(f"\n{'='*55}")
        print(f"  Backbone : {backbone.upper()}")
        print(f"  Epochs   : {args.epochs}")
        print(f"  Data     : {args.n_cases} cases, "
              f"latent {args.lat_ch}×{args.latent_size}³, "
              f"{args.n_cond} cond modalities")
        print(f"  Device   : {device}")
        print(f"{'='*55}")

        t_start = time.time()
        losses  = run(backbone, loader, device,
                      args.epochs, args.lat_ch, args.n_cond, args.latent_size)
        elapsed = time.time() - t_start
        results[backbone] = losses
        print(f"  Done in {elapsed:.1f}s  —  "
              f"loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    print(f"\n{'='*55}")
    print("  Summary")
    print(f"{'='*55}")
    for backbone, losses in results.items():
        delta = losses[-1] - losses[0]
        print(f"  {backbone.upper():<6}  "
              f"start: {losses[0]:.4f}  "
              f"end: {losses[-1]:.4f}  "
              f"delta: {delta:+.4f}")
    print(f"{'='*55}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
