#!/usr/bin/env bash
# Sweep: WFM — Wavelet Flow Matching (Approach 4)
# No VAE. Covers architecture size, learning rate, velocity loss, and
# inference noise (adds diversity at test time via the informed prior).
# Medium UNet [64,128,256,256] mirrors the proven 76 M RFlow backbone.
#
# Usage:
#   DATA_DIR=/path/to/preprocessed bash scripts/sweeps/sweep_wfm.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
DATA_DIR="${DATA_DIR:-/home/fran/DATA/Dataset015_GeneralBrainTumor_preprocessed_500}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./runs}"

TASK="t1n_t2f_to_t1c"
N_EPOCHS=300
N_EPOCHS_DECAY=100

BASE="uv run python scripts/train_wfm.py
  --task                $TASK
  --data_dir            $DATA_DIR
  --patch_size          64 64 64
  --patches_per_volume  4
  --n_timesteps         1000
  --n_inference_steps   10
  --n_epochs            $N_EPOCHS
  --n_epochs_decay      $N_EPOCHS_DECAY
  --checkpoints_dir     $CKPT_DIR
  --log_dir             $LOG_DIR
  --device              $DEVICE"

echo "======================================================================"
echo " sweep_wfm.sh — 8 experiments"
echo "======================================================================"

# ── Architecture sweep (lr=1e-4, L1) ─────────────────────────────────────────

echo "[1/8] wfm_tiny — [32,64,128,128], lr=1e-4"
$BASE \
  --name            wfm_tiny \
  --unet_channels   32 64 128 128 \
  --velocity_loss   l1 \
  --lr              1e-4

echo "[2/8] wfm_medium — [64,128,256,256] ~76 M, lr=1e-4  (baseline)"
$BASE \
  --name            wfm_medium \
  --unet_channels   64 128 256 256 \
  --velocity_loss   l1 \
  --lr              1e-4

echo "[3/8] wfm_large — [64,128,256,512], lr=1e-4"
$BASE \
  --name            wfm_large \
  --unet_channels   64 128 256 512 \
  --velocity_loss   l1 \
  --lr              1e-4

# ── LR sweep (medium) ─────────────────────────────────────────────────────────

echo "[4/8] wfm_medium_lr2e4 — lr=2e-4  (best for latent RFlow)"
$BASE \
  --name            wfm_medium_lr2e4 \
  --unet_channels   64 128 256 256 \
  --velocity_loss   l1 \
  --lr              2e-4

echo "[5/8] wfm_medium_lr5e5 — lr=5e-5"
$BASE \
  --name            wfm_medium_lr5e5 \
  --unet_channels   64 128 256 256 \
  --velocity_loss   l1 \
  --lr              5e-5

# ── Velocity loss sweep (medium, lr=2e-4) ─────────────────────────────────────

echo "[6/8] wfm_medium_l2 — L2 loss, lr=2e-4"
$BASE \
  --name            wfm_medium_l2 \
  --unet_channels   64 128 256 256 \
  --velocity_loss   l2 \
  --lr              2e-4

echo "[7/8] wfm_medium_l1ssim — L1+SSIM blend (alpha=0.5), lr=2e-4"
$BASE \
  --name            wfm_medium_l1ssim \
  --unet_channels   64 128 256 256 \
  --velocity_loss   l1+ssim \
  --loss_alpha      0.5 \
  --lr              2e-4

# ── Inference noise (medium, lr=2e-4, L1) ────────────────────────────────────
# Adds small Gaussian perturbation to the informed prior at inference,
# introducing stochasticity without changing the deterministic training.

echo "[8/8] wfm_medium_inoise01 — inference_noise=0.1, lr=2e-4"
$BASE \
  --name            wfm_medium_inoise01 \
  --unet_channels   64 128 256 256 \
  --velocity_loss   l1 \
  --lr              2e-4 \
  --inference_noise 0.1

echo "======================================================================"
echo " sweep_wfm.sh done"
echo "======================================================================"
