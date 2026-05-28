#!/usr/bin/env bash
# Sweep: RFlow with DiT3D backbone (Approach 2.3)
# Covers architecture size, learning rate, and gradient clipping.
# Medium UNet baseline (76 M) is Approach 2; DiT equivalents are benchmarked here.
#
# Usage:
#   LATENT_ROOT=/path/to/latents VAE_CKPT=/path/to/vae.pt bash scripts/sweeps/sweep_rflow_dit.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
LATENT_ROOT="${LATENT_ROOT:-/path/to/latents}"
VAE_CKPT="${VAE_CKPT:-pretrained/autoencoder_epoch273.pt}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./runs}"

TASK="t1n_t2f_to_t1c"
N_EPOCHS=300
N_EPOCHS_DECAY=100

BASE="uv run python scripts/train_rflow.py
  --task            $TASK
  --latent_root     $LATENT_ROOT
  --vae_ckpt        $VAE_CKPT
  --backbone        dit
  --velocity_loss   l1
  --n_timesteps     1000
  --n_inference_steps 200
  --n_epochs        $N_EPOCHS
  --n_epochs_decay  $N_EPOCHS_DECAY
  --checkpoints_dir $CKPT_DIR
  --log_dir         $LOG_DIR
  --device          $DEVICE"

echo "======================================================================"
echo " sweep_rflow_dit.sh — 6 experiments"
echo "======================================================================"

# ── Architecture sweep (lr=1e-4) ──────────────────────────────────────────────

echo "[1/6] dit_small — hidden=256, depth=8, heads=4, lr=1e-4"
$BASE \
  --name            dit_small \
  --dit_hidden_size 256 --dit_depth 8  --dit_num_heads 4 \
  --lr              1e-4 --grad_clip 1.0

echo "[2/6] dit_medium — hidden=384, depth=12, heads=6, lr=1e-4  (baseline)"
$BASE \
  --name            dit_medium \
  --dit_hidden_size 384 --dit_depth 12 --dit_num_heads 6 \
  --lr              1e-4 --grad_clip 1.0

echo "[3/6] dit_large — hidden=512, depth=16, heads=8, lr=1e-4"
$BASE \
  --name            dit_large \
  --dit_hidden_size 512 --dit_depth 16 --dit_num_heads 8 \
  --lr              1e-4 --grad_clip 1.0

# ── LR sweep (medium DiT) ─────────────────────────────────────────────────────

echo "[4/6] dit_medium_lr2e4 — lr=2e-4  (best for UNet in prior sweep)"
$BASE \
  --name            dit_medium_lr2e4 \
  --dit_hidden_size 384 --dit_depth 12 --dit_num_heads 6 \
  --lr              2e-4 --grad_clip 1.0

echo "[5/6] dit_medium_lr5e5 — lr=5e-5"
$BASE \
  --name            dit_medium_lr5e5 \
  --dit_hidden_size 384 --dit_depth 12 --dit_num_heads 6 \
  --lr              5e-5 --grad_clip 1.0

# ── Grad-clip sensitivity (medium, lr=1e-4) ───────────────────────────────────

echo "[6/6] dit_medium_gc05 — grad_clip=0.5"
$BASE \
  --name            dit_medium_gc05 \
  --dit_hidden_size 384 --dit_depth 12 --dit_num_heads 6 \
  --lr              1e-4 --grad_clip 0.5

echo "======================================================================"
echo " sweep_rflow_dit.sh done"
echo "======================================================================"
