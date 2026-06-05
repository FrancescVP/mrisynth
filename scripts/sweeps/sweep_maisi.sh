#!/usr/bin/env bash
# Sweep: MAISI-v2 region-specific contrastive loss (Approach 3)
# Tests --velocity_loss l1+contrastive with different weights and temperatures.
# All runs: medium UNet [64,128,256,256] ~76 M, lr=2e-4 (best from prior sweep).
# Requires seg.pt files alongside the cached latents.
#
# Usage:
#   LATENT_ROOT=/path/to/latents VAE_CKPT=/path/to/vae.pt bash scripts/sweeps/sweep_maisi.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
LATENT_ROOT="${LATENT_ROOT:-/home/fran/Projects/mrisynth/latents/dataset500}"
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
  --velocity_loss   l1+contrastive
  --lr              2e-4
  --n_timesteps     1000
  --n_inference_steps 200
  --n_epochs        $N_EPOCHS
  --n_epochs_decay  $N_EPOCHS_DECAY
  --checkpoints_dir $CKPT_DIR
  --log_dir         $LOG_DIR
  --device          $DEVICE"

echo "======================================================================"
echo " sweep_maisi.sh — 5 experiments  (MAISI-v2 contrastive loss)"
echo "======================================================================"

# ── Contrastive weight sweep (temperature=0.1) ────────────────────────────────

echo "[1/5] maisi_w01_t01 — weight=0.1, temp=0.1  (default)"
$BASE \
  --name                maisi_w01_t01 \
  --contrastive_weight  0.1 \
  --contrastive_temp    0.1

echo "[2/5] maisi_w05_t01 — weight=0.5, temp=0.1"
$BASE \
  --name                maisi_w05_t01 \
  --contrastive_weight  0.5 \
  --contrastive_temp    0.1

echo "[3/5] maisi_w10_t01 — weight=1.0, temp=0.1  (strong contrastive)"
$BASE \
  --name                maisi_w10_t01 \
  --contrastive_weight  1.0 \
  --contrastive_temp    0.1

# ── Temperature sweep (weight=0.1) ────────────────────────────────────────────

echo "[4/5] maisi_w01_t005 — weight=0.1, temp=0.05  (sharper InfoNCE)"
$BASE \
  --name                maisi_w01_t005 \
  --contrastive_weight  0.1 \
  --contrastive_temp    0.05

echo "[5/5] maisi_w01_t02 — weight=0.1, temp=0.2  (softer InfoNCE)"
$BASE \
  --name                maisi_w01_t02 \
  --contrastive_weight  0.1 \
  --contrastive_temp    0.2

echo "======================================================================"
echo " sweep_maisi.sh done"
echo "======================================================================"
