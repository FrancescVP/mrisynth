#!/usr/bin/env bash
# Sweep: cWDM — Conditional Wavelet Diffusion Model (Approach 5)
# No VAE. Covers architecture size, learning rate, DDIM inference steps,
# and beta schedule. Pretrained BraTS weights available at
# https://github.com/pfriedri/cwdm for sanity-checking before training.
#
# Usage:
#   DATA_DIR=/path/to/preprocessed bash scripts/sweeps/sweep_cwdm.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
DATA_DIR="${DATA_DIR:-/path/to/preprocessed}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./runs}"

TASK="t1n_t2f_to_t1c"
N_EPOCHS=300
N_EPOCHS_DECAY=100

BASE="uv run python scripts/train_cwdm.py
  --task              $TASK
  --data_dir          $DATA_DIR
  --n_timesteps       1000
  --n_ddim_steps      50
  --beta_schedule     linear
  --n_epochs          $N_EPOCHS
  --n_epochs_decay    $N_EPOCHS_DECAY
  --checkpoints_dir   $CKPT_DIR
  --log_dir           $LOG_DIR
  --device            $DEVICE"

echo "======================================================================"
echo " sweep_cwdm.sh — 7 experiments"
echo "======================================================================"

# ── Architecture sweep (lr=1e-4, linear, 50 DDIM steps) ──────────────────────

echo "[1/7] cwdm_medium — [64,128,256,256] ~76 M, lr=1e-4  (baseline)"
$BASE \
  --name            cwdm_medium \
  --unet_channels   64 128 256 256 \
  --lr              1e-4

echo "[2/7] cwdm_large — [64,128,256,512], lr=1e-4"
$BASE \
  --name            cwdm_large \
  --unet_channels   64 128 256 512 \
  --lr              1e-4

# ── LR sweep (medium) ─────────────────────────────────────────────────────────

echo "[3/7] cwdm_medium_lr2e4 — lr=2e-4"
$BASE \
  --name            cwdm_medium_lr2e4 \
  --unet_channels   64 128 256 256 \
  --lr              2e-4

echo "[4/7] cwdm_medium_lr5e5 — lr=5e-5"
$BASE \
  --name            cwdm_medium_lr5e5 \
  --unet_channels   64 128 256 256 \
  --lr              5e-5

# ── DDIM inference steps sweep (medium, lr=1e-4, linear) ─────────────────────
# These only affect inference speed vs quality; training is identical.

echo "[5/7] cwdm_medium_20steps — 20 DDIM steps at inference"
$BASE \
  --name            cwdm_medium_20steps \
  --unet_channels   64 128 256 256 \
  --lr              1e-4 \
  --n_ddim_steps    20

echo "[6/7] cwdm_medium_100steps — 100 DDIM steps at inference"
$BASE \
  --name            cwdm_medium_100steps \
  --unet_channels   64 128 256 256 \
  --lr              1e-4 \
  --n_ddim_steps    100

# ── Beta schedule (medium, lr=1e-4, 50 DDIM steps) ───────────────────────────

echo "[7/7] cwdm_medium_scaled — scaled_linear beta schedule"
$BASE \
  --name            cwdm_medium_scaled \
  --unet_channels   64 128 256 256 \
  --lr              1e-4 \
  --beta_schedule   scaled_linear

echo "======================================================================"
echo " sweep_cwdm.sh done"
echo "======================================================================"
