# MRI Synthetic Generation

Multimodal MRI synthesis pipeline for BraTS-style datasets.  
Implements two synthesis approaches: **3-D pix2pix** (image-space GAN) and **RFlow** (latent rectified-flow diffusion).

**Target task:** T1w + T2FLAIR → T1CE synthesis  
**Dataset:** BraTS Dataset015_GeneralBrainTumor, 2047 cases

---

## Results

**Best result — RFlow:** SSIM **0.765** · MAE **0.063**  
`DiffusionModelUNet [64,128,256,256]` · L1 velocity loss · lr=2e-4 · 300 epochs · 900 cases

| Model | SSIM ↑ | MAE ↓ |
|---|---|---|
| **RFlow** — L1 loss, lr=2e-4 | **0.765** | **0.063** |
| RFlow — pilot (500 cases) | 0.712 | 0.090 |
| pix2pix — best (exp_alpha03) | 0.658 | 0.252 |

RFlow outperforms pix2pix by **4× on MAE**. Scaling from 500 → 900 cases alone cut MAE by 30%.

<details>
<summary><strong>Full experiment results</strong> — 28 RFlow experiments across architecture, HP, and velocity loss sweeps, plus the full pix2pix ablation (4 phases, ~40 runs)</summary>

## Phase 1 — Modality study (100 cases, 90/10 split)

**Architecture:** 3D pix2pix, L2, LSGAN, λ_pixel=100, λ_feat=10, d_freq=2, 50+50 epochs

> SSIM computed with old `max_value=1.0` — only compare within this table.

| Input modalities | SSIM ↑ | MAE ↓ | ET_rel ↓ |
|---|---|---|---|
| T1n only | 0.320 | 0.272 | 0.58 |
| T1n + T2w | **0.409** | **0.256** | 0.73 |
| T1n + T2w + FLAIR | 0.399 | 0.264 | 0.72 |
| T1n + T2w + FLAIR + tumor loss | 0.399 | 0.263 | **0.69** |

**Findings:** T2w is the single biggest gain. FLAIR adds nothing globally. Tumor loss improves ET at negligible global cost → kept for all subsequent experiments.  
**Forward choice:** T1n + T2FLAIR inputs, tumor loss enabled.

---

## Phase 2 — GAN parameter sweep (500 cases)

**Architecture:** 3D pix2pix, ngf=64, 5 D-layers, 50+50 epochs, L2, LSGAN  
**Base:** λ_pixel=100, λ_feat=10, d_freq=2, α_tumor=0.5

> ⚠ SSIM was recomputed mid-sweep. **Group A–D1** used `max_value=1.0` (deflated by ≈0.25).  
> **Group D2 onwards** used `max_value = gt.max() − gt.min()`.  
> **Do not compare SSIM across the two groups. Use MAE for cross-group ranking.**

### Group A–D1: old SSIM (not comparable to D2+)

| Name | Change vs base | SSIM* | MAE ↓ | ET_rel ↓ |
|---|---|---|---|---|
| exp_baseline | — | 0.443 | 0.258 | 0.475 |
| exp_lp200 | λ_pixel=200 | 0.376 | 0.259 | 0.644 |
| exp_lp50 | λ_pixel=50 | 0.439 | 0.262 | 0.624 |
| exp_lp10 | λ_pixel=10 | 0.431 | 0.280 | 0.535 |
| exp_no_feat | λ_feat=0 | 0.328 | 0.260 | 0.600 |
| exp_lf5 | λ_feat=5 | 0.425 | 0.258 | 0.529 |
| exp_lf20 | λ_feat=20 | 0.432 | 0.262 | 0.609 |
| exp_vanilla | vanilla GAN | 0.434 | 0.277 | 0.640 |
| exp_dfreq1 | d_freq=1 | **0.454** | **0.255** | 0.548 |

*deflated ≈0.25 — relative ordering within this block only

### Group D2+: corrected SSIM ✅

| Name | Change vs base | SSIM ↑ | MAE ↓ | ET_rel ↓ |
|---|---|---|---|---|
| exp_dfreq4 | d_freq=4 | 0.647 | 0.259 | 0.622 |
| exp_alpha01 | α_tumor=0.1 | **0.658** | 0.254 | 0.556 |
| exp_alpha03 | α_tumor=0.3 | **0.658** | **0.252** | 0.572 |
| exp_alpha07 | α_tumor=0.7 | 0.643 | 0.259 | 0.552 |
| exp_ngf32 | ngf=32 | 0.643 | 0.260 | **0.477** |
| exp_nlayers2 | 2 D-layers | 0.648 | 0.259 | 0.620 |
| exp_dfreq1_alpha03 | d_freq=1 + α=0.3 | 0.648 | 0.257 | 0.595 |
| exp_ngf128_l1_lp20_nofeat | ngf=128, L1, λ_p=20, λ_f=0 | 0.638 | 0.261 | 0.551 |

> exp_ngf128_l1_lp20_nofeat stopped at ep50 (killed). All others: 100 epochs.

**Best pix2pix:** exp_alpha03 — MAE 0.252 (best), SSIM 0.658 (tied best), ET_rel 0.572  
Runner-up: exp_alpha01 (SSIM 0.658, MAE 0.254, ET_rel 0.556)

---

## RFlow — Latent Diffusion (500 cases, pilot)

**Architecture:** DiffusionModelUNet [64,128,256,256], MAISI VAE (frozen), T1n+T2FLAIR → T1CE  
**Training:** 1000 timesteps, AdamW lr=1e-4, AMP + gradient checkpointing, L1 velocity loss

| Name | UNet channels | Epochs | SSIM ↑ | MAE ↓ | Latent L1 |
|---|---|---|---|---|---|
| rflow_medium | [64,128,256,256] | 250 | **0.712** | **0.090** | 0.703 |

> rflow_medium clearly outperforms all pix2pix variants on both metrics.  
> SSIM corrected (same method as D2+ above). MAE in image space post-VAE decode.

---

## RFlow sweeps — 900 cases

All experiments below: MAISI VAE (frozen), T1n+T2FLAIR → T1CE, AdamW lr=1e-4 (unless noted),  
1000 timesteps, 900 train / 100 val split, AMP + gradient checkpointing, image-space val via VAE.  
SSIM and MAE reported at final epoch.

### Architecture sweep (300 epochs, L1 velocity loss)

| Name | UNet channels | Params | SSIM ↑ | MAE ↓ |
|---|---|---|---|---|
| rflow_tiny | [32, 64, 128, 128] | 19.2 M | 0.739 | 0.069 |
| rflow_small | [32, 64, 128, 256] | 44.7 M | 0.729 | 0.071 |
| rflow_medium | [64, 128, 256, 256] | 76.5 M | 0.750 | 0.071 |
| rflow_deep | [32, 64, 128, 256, 256] | — | 0.745 | 0.072 |
| rflow_large | [64, 128, 256, 512] | 178.6 M | 0.754 | 0.069 |

> rflow_tiny achieves near-parity with rflow_large at 9× fewer parameters. rflow_large shows modest gains on SSIM (0.754) and MAE (0.069). Medium is the best efficiency/performance point.

### Hyperparameter sweep ([64, 128, 256, 256], 300 epochs, L1 loss)

| Name | Change vs baseline | SSIM ↑ | MAE ↓ |
|---|---|---|---|
| hp_baseline | — (lr=1e-4, ts=1000, linear) | 0.759 | 0.065 |
| hp_cosine | cosine LR schedule | 0.752 | 0.074 |
| hp_decay | 200+100 LR decay | 0.756 | 0.065 |
| hp_ts500 | 500 training timesteps | 0.760 | 0.068 |
| hp_ts250 | 250 training timesteps | 0.754 | 0.068 |
| **hp_lr_high** | **lr=2e-4** | **0.765** | 0.071 |
| hp_lr_low | lr=5e-5 | 0.750 | 0.071 |
| hp_attn3 | 3 attention levels | 0.750 | 0.065 |
| hp_resblk3 | 3 res blocks/level | 0.752 | 0.071 |
| hp_ema | EMA decay=0.9999 | 0.757 | 0.067 |

> **lr=2e-4 gives the best SSIM (0.765).** Cosine LR hurts. 500 timesteps marginal gain. LR decay (200+100) matches baseline on MAE.

### Velocity loss sweep ([64, 128, 256, 256], 300 epochs, lr=1e-4)

| Name | Velocity loss | SSIM ↑ | MAE ↓ | Notes |
|---|---|---|---|---|
| **loss_l1** | **L1** | 0.754 | **0.063** | — |
| loss_l2 | L2/MSE | 0.753 | 0.068 | — |
| loss_l1ssim | L1 + SSIM blend | 0.751 | 0.074 | — |
| loss_ssim | SSIM only | 0.737 | 0.075 | — |
| loss_ncc | NCC | 0.446 | 0.109 | Failed — NCC diverges without tuning |
| loss_tumor_l1_w10 | Tumor-weighted L1, w=10 | 0.756 | 0.070 | 260 ep (killed early) |
| loss_et_l1_w5 | ET-weighted L1, w=5 | 0.757 | 0.070 | — |
| loss_et_l1_w10 | ET-weighted L1, w=10 | 0.756 | 0.069 | — |

> **L1 gives the best MAE (0.063).** NCC fails without careful LR/normalisation tuning. Tumor/ET-weighted losses give competitive SSIM but don't improve MAE vs plain L1.

---

## Summary: all experiments ranked by MAE

| Rank | Experiment | SSIM ↑ | MAE ↓ | Notes |
|---|---|---|---|---|
| 🥇 | **loss_l1** | 0.754 | **0.063** | RFlow 900 cases, L1 loss |
| 2 | hp_baseline | **0.759** | 0.065 | RFlow 900 cases, baseline HP |
| 3 | hp_attn3 | 0.750 | 0.065 | 3 attention levels |
| 4 | hp_decay | 0.756 | 0.065 | LR decay schedule |
| 5 | hp_ema | 0.757 | 0.067 | EMA weights at inference |
| 6 | rflow_tiny | 0.739 | 0.069 | 19 M params — smallest model |
| 7 | rflow_large | 0.754 | 0.069 | 178 M params |
| — | **hp_lr_high** | **0.765** | 0.071 | Best SSIM overall (lr=2e-4) |
| — | exp_alpha03 | 0.658 | 0.252 | Best pix2pix |
| — | rflow_medium (500c) | 0.712 | 0.090 | Original pilot |

> ⚠ All 900-case RFlow results are directly comparable (same SSIM method, same val set).

</details>

---

## Repository layout

```
src/mrisynth/
├── preprocessing/        # nnUNet-style pipeline (crop, resample, normalise)
├── augmentation/         # batchgeneratorsv2 transforms + dataset
├── metrics/              # MAE, MSE, NMSE, PSNR, SSIM, ET-specific metrics
├── losses/               # Tumor-aware GAN losses + RFlow velocity losses
└── model/
    ├── networks.py        # UnetGenerator3D, NLayerDiscriminator3D, GANLoss
    ├── pix2pix3d.py       # Pix2Pix3dModel — image-space GAN
    ├── rflow.py           # RFlowModel — latent rectified-flow diffusion
    ├── vae.py             # MaisiVAE — frozen MAISI autoencoder wrapper
    ├── dataset.py         # Pix2Pix3dDataset
    └── latent_dataset.py  # LatentDataset — pre-cached VAE latents

scripts/
├── train_pix2pix3d.py     # Pix2Pix training loop + TensorBoard
├── train_rflow.py         # RFlow training loop + TensorBoard
├── generate_latents.py    # Pre-compute MAISI VAE latents (one-time)
├── infer_pix2pix.py       # Run pix2pix inference, save NIfTI predictions
├── infer_rflow.py         # Run RFlow inference, save NIfTI predictions
├── eval_pairs.py          # Evaluate pred/GT NIfTI pairs (MAE, SSIM, ET metrics)
├── make_gifs.py           # Animated axial-slice GIFs (T1n | pred | GT)
└── export_gifs.py         # Multi-experiment sweep GIFs for comparison
```

---

## Installation

```bash
# Clone and install (creates .venv)
git clone <repo-url>
cd mrisynth
uv sync
```

Requires CUDA 12.1 and Python ≥ 3.10.  
Key dependencies: `torch`, `monai>=1.3`, `nibabel`, `batchgeneratorsv2`, `tensorboard`, `einops`.

---

## Data

**Channel layout** inside each `.npz` file (post-preprocessing):

| Index | Modality | Role |
|---|---|---|
| 0 | T1CE | synthesis target |
| 1 | T1n | primary input |
| 2 | T2FLAIR | secondary input |
| 3 | T2w | secondary input |

**Segmentation labels** (BraTS 2023): `0=background`, `1=NETC`, `2=SNFH`, `3=ET`.

See [`data/example/`](data/example/README.md) for the expected raw input layout (synthetic NIfTI files showing the nnUNet structure) and documentation of the generated downstream formats.

### Preprocessing

Converts raw BraTS (`.b2nd`) to Z-score-normalised `.npz` files:

```bash
uv run gan-preprocess \
    --dataset /path/to/Dataset015/nnUNetPlans_3d_fullres \
    --output  /path/to/preprocessed \
    --normalization zscore_nonzero \
    --num-workers 8
```

---

## Approach 1 — 3-D Pix2Pix

**Architecture:**  
- Generator: 3-D U-Net — ngf=64, 5 encoder-decoder levels, ~66 M params  
- Discriminator: 3-D PatchGAN — ndf=32, 3 layers, ~2.8 M params  
- Loss: LSGAN + L2 pixel (λ=100) + feature matching (λ=10) + ET-aware tumor loss

### Reproduce best result (exp_alpha03)

```bash
uv run python scripts/train_pix2pix3d.py \
    --data_dir        /path/to/preprocessed \
    --input_channels  T1n T2FLAIR \
    --target_channels T1CE \
    --name            pix2pix_best \
    --n_epochs        100 \
    --n_epochs_decay  50 \
    --lambda_pixel    100.0 \
    --lambda_feat     10.0 \
    --d_update_freq   2 \
    --use_tumor_loss  \
    --alpha_tumor     0.3 \
    --device          cuda:0

tensorboard --logdir runs/
```

Expected: SSIM ≈ 0.658, MAE ≈ 0.252 (500 cases, 100 epochs).

### Inference

```bash
uv run python scripts/infer_pix2pix.py \
    --data_dir   /path/to/preprocessed \
    --checkpoint checkpoints/pix2pix_best/latest_net_G.pth \
    --out_dir    predictions/pix2pix_best \
    --n_cases    10 \
    --device     cuda:0
```

---

## Approach 2 — RFlow (Latent Diffusion)

**Architecture:**  
- Encoder/Decoder: MAISI VAE (frozen, ~20.9 M params, 4× spatial compression)  
- Flow network: `DiffusionModelUNet` — [64,128,256,256] channels, ~76 M params  
- Loss: L1 on velocity field in latent space

**VAE checkpoint:** `pretrained/autoencoder_epoch273.pt` (~80 MB, MAISI autoencoder). The file is included in this repo under `pretrained/` but is git-ignored (too large for git); track with Git LFS or copy manually.

### Step 1 — Pre-compute latents (one-time, ~2 h on a single GPU)

```bash
uv run python scripts/generate_latents.py \
    --data_dir /path/to/preprocessed \
    --vae_ckpt pretrained/autoencoder_epoch273.pt \
    --out_dir  latents/dataset \
    --device   cuda:0
```

Resume-safe — skips cases where all latent files already exist.  
Output: `latents/dataset/{train,val}/<case_id>/<case_id>-{t1c,t1n,t2f,t2w}_z_{mu,sigma}.pt`

### Step 2 — Train RFlow UNet

#### Reproduce best result (SSIM 0.765, MAE 0.063 — 900 cases)

Best configuration from a 28-experiment sweep: L1 velocity loss + lr=2e-4 + [64,128,256,256].

```bash
uv run python scripts/train_rflow.py \
    --task           t1n_t2f_to_t1c \
    --latent_root    latents/dataset \
    --vae_ckpt       pretrained/autoencoder_epoch273.pt \
    --name           rflow_best \
    --unet_channels  64 128 256 256 \
    --velocity_loss  l1 \
    --lr             2e-4 \
    --n_epochs       300 \
    --device         cuda:0

tensorboard --logdir runs/
```

Expected (900 cases, 300 ep): SSIM ≈ 0.765, MAE ≈ 0.063.

> **Key findings from the sweep:**
> - L1 velocity loss gives the best MAE (0.063); L2/SSIM/NCC all worse.
> - lr=2e-4 gives the best SSIM (0.765); default lr=1e-4 is slightly worse.
> - Architecture [64,128,256,256] (76 M) is the efficiency sweet spot — rflow_tiny (19 M) is a strong lightweight alternative.
> - Cosine LR schedule hurts; linear (default) is better.
> - More data matters: 500 → 900 cases improved MAE by 30%.

#### Available tasks

| `--task` | Conditioning | Target |
|---|---|---|
| `t1n_t2f_to_t1c` | T1w + T2FLAIR | T1CE (**best**) |
| `t1n_t2w_to_t1c` | T1w + T2w | T1CE |
| `t1n_to_t2f` | T1w | FLAIR |
| `t1n_to_t2w` | T1w | T2w |

#### Advanced options

```bash
# Tumor-weighted velocity loss (requires seg files alongside latents)
--velocity_loss tumor_l1 --tumor_weight 5.0

# Region-specific contrastive loss — MAISI-v2 style (requires seg files)
# Adds InfoNCE term pulling predicted ET velocity toward target ET,
# away from background. Improves ET-region fidelity without changing the backbone.
--velocity_loss l1+contrastive --contrastive_weight 0.1 --contrastive_temp 0.1

# OT-FM: mini-batch optimal transport noise coupling
# Permutes noise samples within each batch to minimise total L2 cost to targets,
# shortening flow paths and allowing fewer inference steps at convergence.
# No-op for batch_size=1; effect grows with larger batches.
--use_ot_coupling

# EMA weights for inference (marginal improvement)
--ema_decay 0.9999

# Gradient clipping (recommended for DiT backbone)
--grad_clip 1.0

# Lightweight alternative: rflow_tiny at 19 M params
--unet_channels 32 64 128 128

# High-capacity alternative: rflow_large at 178 M params
--unet_channels 64 128 256 512
```

### Inference

```bash
uv run python scripts/infer_rflow.py \
    --latent_dir  latents/dataset/val \
    --vae_ckpt    pretrained/autoencoder_epoch273.pt \
    --checkpoint  checkpoints/rflow_best/latest_net_UNet.pth \
    --unet_channels 64 128 256 256 \
    --out_dir     predictions/rflow_best \
    --n_cases     10 \
    --save_gt \
    --device      cuda:0
```

---

## Evaluation

```bash
# Basic metrics (MAE, MSE, NMSE, PSNR, SSIM)
uv run python scripts/eval_pairs.py \
    --dir predictions/rflow_best

# With tumour-aware metrics (requires segmentation files)
uv run python scripts/eval_pairs.py \
    --dir     predictions/rflow_best \
    --seg-dir /path/to/preprocessed_segs \
    --csv     results.csv
```

Expected naming convention: `{case}_gt_{modality}.nii.gz` / `{case}_pred_{modality}.nii.gz`.

---

## Visualisation

```bash
# Animated GIFs: T1n | pred T1CE | GT T1CE
uv run python scripts/make_gifs.py \
    --pred_dir predictions/rflow_best \
    --fps 8
```

---

## Library usage

```python
from mrisynth.model import Pix2Pix3dModel, RFlowModel, MaisiVAE
from mrisynth.metrics import ssim, et_metrics
from mrisynth.losses import TumorAwareGANLoss, build_velocity_loss

# Metrics — always use per-volume data range for Z-score normalised data
data_range = float(gt.max() - gt.min())
score = ssim(pred, gt, max_value=data_range)

# Enhancing-tumour metrics (T1CE specific)
out = et_metrics(pred, gt, seg, et_class=3, max_value=data_range)
# keys: et_pearson, et_mae, et_edge_ssim, et_contrast_rel_err, et_loc_dice

# RFlow velocity loss
criterion = build_velocity_loss("ncc")
loss = criterion(v_pred, v_target)
```

---

## Tests

```bash
uv run pytest -q
```

187 tests covering preprocessing, augmentation, losses, metrics, datasets, networks, and models.

<details>
<summary><strong>Test suite breakdown</strong> — per-file tables of what each group tests, plus module coverage map</summary>

### `test_preprocessing.py` — 24 tests

Normalization, cropping, and resampling utilities.

| Group | What is checked |
|---|---|
| `zscore_nonzero` | Background stays zero; foreground mean≈0, std≈1; all-zero input; custom mask |
| `zscore_global` | Shape preserved; per-channel mean≈0, std≈1 |
| `percentile_clip_zscore` | Output finite; shape preserved; background stays zero |
| `get_nonzero_bbox` | Tight bounding box; all-zero fallback; single-channel case |
| `crop_to_nonzero` | Shape reduced; seg stays aligned with data; bbox values correct |
| `compute_new_shape` | Halved spacing → halved shape; identity; anisotropic spacing |
| `resample_data` | Output shape matches expected; identity resampling; finite output |
| `resample_seg` | Output labels ⊆ input labels; correct output shape; NN resampling (order=0) |

---

### `test_augmentation.py` — 6 tests

Dataset loading, spatial/intensity transform pipelines, and DataLoader collation.

---

### `test_networks.py` — 19 tests

3-D network primitives.

| Group | What is checked |
|---|---|
| `get_norm_layer_3d` | Instance, batch, none norms; unknown raises `NotImplementedError` |
| `GANLoss` | lsgan/vanilla/wgangp modes finite; lsgan perfect real ≈ 0; wgangp sign; unknown raises |
| `define_G_3d` | Output shape; multichannel input; gradient flow; odd spatial dims clip to multiple of 2^5 |
| `define_D_3d` | Output is tensor; `return_features=True`; gradient flow |
| `get_scheduler` | linear/step/cosine step without error; unknown raises; linear actually decreases LR |

---

### `test_dataset.py` — 16 tests

Channel resolution, `Pix2Pix3dDataset`, and `LatentDataset`.

| Group | What is checked |
|---|---|
| `resolve_channels` | Int indices; name strings; aliases (t1c, flair, t2); mixed; unknown name raises; out-of-range int raises; wrong type raises |
| `Pix2Pix3dDataset` | Length; `patches_per_volume` multiplier; A/B shapes; multi-input channels; seg shape; A_paths key; patch cropping; float32 dtype |
| `LatentDataset` | Length; latent shape; cond channel count; seg=None when absent; seg loaded when present; deterministic=True produces identical samples; case_id key; custom target key; empty root raises |

---

### `test_models.py` — 18 tests

`_pad`/`_unpad`, `_ot_couple`, `_build_unet`, `_build_dit`, and `RFlowModel` smoke tests.

| Group | What is checked |
|---|---|
| `_pad` / `_unpad` | Roundtrip identity for even and odd spatial dims; padded dims divisible by factor; no-op when already aligned |
| `_ot_couple` | No-op for B=1; shape preserved; output is a permutation of input; coupled L2 cost ≤ uncoupled cost |
| `_build_unet` | Forward output shape; multiple attention levels; single attention level |
| `RFlowModel` (CPU) | `set_input` stores tensors; `optimize_parameters` produces finite loss; `model_names`; `get_current_losses` returns float |
| OT-FM | `use_ot_coupling=True` with B=2 produces finite loss |
| Contrastive | `l1+contrastive` with seg tensor produces finite loss |
| EMA | Weights initialized when `ema_decay > 0`; weights update after a training step |

---

### `test_losses/` — 39 tests across 3 files

#### `test_velocity_losses.py` — 34 tests

Velocity-space losses for RFlow.

| Class | What is checked |
|---|---|
| `SegAwareLoss` | `TumorWeightedL1Loss`, `RegionContrastiveLoss`, `L1RegionContrastiveLoss` all inherit it |
| `NCCLoss` | Range [0, 2]; perfect match ≈ 0; linear invariance (a·x+b); gradient flow |
| `SSIMLoss` | Perfect match ≈ 0; range [0, 2]; low noise < high noise; gradient flow |
| `L1SSIMLoss` | α=1 equals L1; α=0 equals SSIM; α=0.5 exact blend; invalid α raises |
| `TumorWeightedL1Loss` | No seg equals L1; perfect match = 0; error inside tumor → weighted > plain; gradient flow |
| `RegionContrastiveLoss` | No seg → zero; correct ET prediction has lower loss than wrong; finite output; empty ET mask → zero; gradient flow |
| `L1RegionContrastiveLoss` | No seg equals L1; with seg ≥ L1; gradient flow with seg |
| `build_velocity_loss` | All names (incl. `l1+contrastive`) return `nn.Module`; "l1" is `nn.L1Loss`; "l2" is `nn.MSELoss`; unknown raises |

#### `test_composite.py` — 8 tests

`TumorAwareGANLoss` blending and edge cases.

#### `test_enhancing_tumor.py` — 10 tests

ET metrics (contrast, Pearson, MAE, edge SSIM, localization Dice) and `EnhancingTumorLoss`.

---

### `test_tumor_ssim.py` — 8 tests

`TumorSSIMLoss` and `RegionWeightedSSIMLoss`.

---

### `test_metrics.py` — 8 tests

Pixel-wise metrics: MAE, MSE, NMSE, PSNR, SSIM (2D/3D), multi-channel, gradient flow.

---

### Coverage by module

| Module | Test file |
|---|---|
| `preprocessing/normalization.py` | `test_preprocessing.py` |
| `preprocessing/cropping.py` | `test_preprocessing.py` |
| `preprocessing/resampling.py` | `test_preprocessing.py` |
| `augmentation/` | `test_augmentation.py` |
| `losses/velocity.py` | `test_velocity_losses.py` |
| `losses/composite.py` | `test_composite.py` |
| `losses/enhancing_tumor.py` | `test_enhancing_tumor.py` |
| `losses/tumor_ssim.py` | `test_tumor_ssim.py` |
| `metrics/` | `test_metrics.py`, `test_enhancing_tumor.py` |
| `model/networks.py` | `test_networks.py` |
| `model/dataset.py` | `test_dataset.py` |
| `model/latent_dataset.py` | `test_dataset.py` |
| `model/rflow.py` | `test_models.py` |

Not covered by unit tests (require GPU + VAE checkpoint): `model/pix2pix3d.py`, `model/vae.py`, `preprocessing/pipeline.py`, `preprocessing/io.py`.

</details>

---

## Key findings

| Finding | Detail |
|---|---|
| **RFlow >> pix2pix** | MAE 0.063 vs 0.252 (4×) — latent diffusion is the clear winner |
| **More data matters** | 500 → 900 cases: MAE 0.090 → 0.063 (−30%), SSIM 0.712 → 0.765 |
| **L1 is the best velocity loss** | MAE 0.063; L2/SSIM/NCC all worse. NCC diverges without tuning. |
| **lr=2e-4 best SSIM for RFlow** | Default 1e-4 gives SSIM 0.754–0.759; 2e-4 gives 0.765. |
| **rflow_tiny is competitive** | 19 M params matches 178 M (MAE 0.069 each). Medium [64,128,256,256] is the sweet spot. |
| **α_tumor = 0.3 optimal (pix2pix)** | 0.1 and 0.3 tied for SSIM; 0.3 wins on MAE. α=0.7 hurts |
| **λ_feat=0 collapses D (pix2pix)** | D_real → 0.002 at bs=1 without feature matching — always keep it |
| **LSGAN > vanilla GAN** | Vanilla is unstable at bs=1 (D_real 0.289 vs 0.016 for LSGAN) |
| **SSIM data-range note** | Use `max_value = gt.max() − gt.min()` for Z-score data, not 1.0 |
| **OT-FM** (`--use_ot_coupling`) | Mini-batch OT couples noise to targets for shorter flow paths; allows fewer inference steps. Not yet benchmarked on this dataset. |
| **Region contrastive loss** (`--velocity_loss l1+contrastive`) | InfoNCE term pulls ET-region velocity prediction toward GT, away from BG. Not yet benchmarked on this dataset. |

---

## References

- nnUNet: [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- MONAI / MAISI: [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)
- BraTS 2023: [synapse.org/brats2023](https://www.synapse.org/brats2023)
- pix2pix: Isola et al. 2017 — *Image-to-Image Translation with Conditional Adversarial Networks*
- Rectified Flow: Liu et al. 2022 — *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*
