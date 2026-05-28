#!/usr/bin/env bash
# Sweep: RFlow + OT-FM noise coupling (Approach 2.2)
# Ablates OT coupling on/off and tests batch_size > 1 (OT is a no-op at B=1).
# All runs use the proven medium UNet [64,128,256,256] ~76 M params.
# The no-OT baseline (rflow_best, SSIM 0.765, MAE 0.063) is from prior sweeps.
#
# Usage:
#   LATENT_ROOT=/path/to/latents VAE_CKPT=/path/to/vae.pt bash scripts/sweeps/sweep_rflow_otfm.sh
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

BASE="uv run python scripts/train_rflow.py
  --task            $TASK
  --latent_root     $LATENT_ROOT
  --vae_ckpt        $VAE_CKPT
  --unet_channels   $UNET
  --velocity_loss   l1
  --n_timesteps     1000
  --n_inference_steps 200
  --n_epochs        $N_EPOCHS
  --n_epochs_decay  $N_EPOCHS_DECAY
  --checkpoints_dir $CKPT_DIR
  --log_dir         $LOG_DIR
  --device          $DEVICE"

echo "======================================================================"
echo " sweep_rflow_otfm.sh — 4 experiments"
echo "======================================================================"

# ── OT coupling on, lr sweep ──────────────────────────────────────────────────

echo "[1/4] otfm_l1_lr1e4 — OT coupling, lr=1e-4"
$BASE \
  --name            otfm_l1_lr1e4 \
  --lr              1e-4 \
  --use_ot_coupling

echo "[2/4] otfm_l1_lr2e4 — OT coupling, lr=2e-4  (mirrors best no-OT config)"
$BASE \
  --name            otfm_l1_lr2e4 \
  --lr              2e-4 \
  --use_ot_coupling

# ── Loss function with OT ─────────────────────────────────────────────────────

echo "[3/4] otfm_l2_lr1e4 — OT coupling + L2 loss"
$BASE \
  --name            otfm_l2_lr1e4 \
  --velocity_loss   l2 \
  --lr              1e-4 \
  --use_ot_coupling

# ── Batch-size > 1: OT only does work when B > 1 ─────────────────────────────
# Requires enough VRAM — reduce unet_channels if OOM.

echo "[4/4] otfm_l1_lr2e4_bs2 — OT coupling, lr=2e-4, batch_size=2"
$BASE \
  --name            otfm_l1_lr2e4_bs2 \
  --lr              2e-4 \
  --batch_size      2 \
  --use_ot_coupling

echo "======================================================================"
echo " sweep_rflow_otfm.sh done"
echo "======================================================================"
