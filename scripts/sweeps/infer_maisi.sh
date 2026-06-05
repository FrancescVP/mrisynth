#!/usr/bin/env bash
# Inference for all 5 MAISI-v2 contrastive sweep experiments.
# Outputs go to predictions/maisi_<name>/.
#
# Usage:
#   bash scripts/sweeps/infer_maisi.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
LATENT_DIR="/home/fran/Projects/mrisynth/latents/dataset500/val"
VAE_CKPT="pretrained/autoencoder_epoch273.pt"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
OUT_ROOT="${OUT_ROOT:-./predictions}"
N_CASES="${N_CASES:-5}"

BASE="uv run python scripts/infer_rflow.py
  --latent_dir        $LATENT_DIR
  --vae_ckpt          $VAE_CKPT
  --unet_channels     64 128 256 256
  --n_cases           $N_CASES
  --save_gt
  --device            $DEVICE"

EXPERIMENTS=(
  maisi_w01_t01
  maisi_w05_t01
  maisi_w10_t01
  maisi_w01_t005
  maisi_w01_t02
)

echo "======================================================================"
echo " infer_maisi.sh — inference on ${#EXPERIMENTS[@]} experiments"
echo "======================================================================"

for name in "${EXPERIMENTS[@]}"; do
  echo ""
  echo ">>> $name"
  $BASE \
    --checkpoint "$CKPT_DIR/$name/latest_net_UNet.pth" \
    --out_dir    "$OUT_ROOT/$name"
done

echo ""
echo "======================================================================"
echo " infer_maisi.sh done"
echo "======================================================================"
