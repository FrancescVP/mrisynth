#!/usr/bin/env bash
# Sweep: RFlow + ControlNet seg-guided synthesis (Approach 6)
# All runs use medium UNet [64,128,256,256] ~76 M + matching ControlNet encoder.
# Requires seg.pt files alongside the cached latents (generated automatically
# by generate_latents.py when seg channels are present in the .npz files).
#
# Usage:
#   LATENT_ROOT=/path/to/latents VAE_CKPT=/path/to/vae.pt bash scripts/sweeps/sweep_rflow_controlnet.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
LATENT_ROOT="${LATENT_ROOT:-/path/to/latents}"
VAE_CKPT="${VAE_CKPT:-pretrained/autoencoder_epoch273.pt}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./runs}"

TASK="t1n_t2f_to_t1c"
N_EPOCHS=300
N_EPOCHS_DECAY=100
UNET="64 128 256 256"

BASE="uv run python scripts/train_rflow_controlnet.py
  --task              $TASK
  --latent_root       $LATENT_ROOT
  --vae_ckpt          $VAE_CKPT
  --unet_channels     $UNET
  --velocity_loss     l1
  --n_timesteps       1000
  --n_inference_steps 200
  --n_epochs          $N_EPOCHS
  --n_epochs_decay    $N_EPOCHS_DECAY
  --checkpoints_dir   $CKPT_DIR
  --log_dir           $LOG_DIR
  --device            $DEVICE"

echo "======================================================================"
echo " sweep_rflow_controlnet.sh — 5 experiments"
echo "======================================================================"

# ── LR sweep ──────────────────────────────────────────────────────────────────

echo "[1/5] cn_l1_lr1e4 — L1, lr=1e-4  (baseline)"
$BASE \
  --name  cn_l1_lr1e4 \
  --lr    1e-4

echo "[2/5] cn_l1_lr2e4 — L1, lr=2e-4  (best for plain RFlow)"
$BASE \
  --name  cn_l1_lr2e4 \
  --lr    2e-4

echo "[3/5] cn_l1_lr5e5 — L1, lr=5e-5"
$BASE \
  --name  cn_l1_lr5e5 \
  --lr    5e-5

# ── Combined with other improvements ─────────────────────────────────────────

echo "[4/5] cn_contrastive_lr2e4 — ControlNet + MAISI-v2 contrastive loss, lr=2e-4"
uv run python scripts/train_rflow_controlnet.py \
  --task              $TASK \
  --latent_root       $LATENT_ROOT \
  --vae_ckpt          $VAE_CKPT \
  --unet_channels     $UNET \
  --velocity_loss     l1+contrastive \
  --contrastive_weight 0.1 \
  --contrastive_temp  0.1 \
  --lr                2e-4 \
  --n_timesteps       1000 \
  --n_inference_steps 200 \
  --n_epochs          $N_EPOCHS \
  --n_epochs_decay    $N_EPOCHS_DECAY \
  --name              cn_contrastive_lr2e4 \
  --checkpoints_dir   $CKPT_DIR \
  --log_dir           $LOG_DIR \
  --device            $DEVICE

echo "[5/5] cn_otfm_lr2e4 — ControlNet + OT-FM coupling, lr=2e-4"
$BASE \
  --name              cn_otfm_lr2e4 \
  --lr                2e-4 \
  --use_ot_coupling

echo "======================================================================"
echo " sweep_rflow_controlnet.sh done"
echo "======================================================================"
