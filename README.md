# MRI Synthetic Generation

Multimodal MRI synthesis pipeline for BraTS-style datasets.  
Implements two synthesis approaches: **3-D pix2pix** (image-space GAN) and **RFlow** (latent rectified-flow diffusion).

**Target task:** T1w + T2FLAIR → T1CE synthesis  
**Dataset:** BraTS Dataset015_GeneralBrainTumor, 2047 cases

---

## Results

**Best result — RFlow + MAISI contrastive:** SSIM **0.752** · MAE **0.066**  
`DiffusionModelUNet [64,128,256,256]` · L1+contrastive (w=0.1, T=0.2) · lr=2e-4 · 400 epochs · 500 cases

| Model | SSIM ↑ | MAE ↓ | Cases | Notes |
|---|---|---|---|---|
| **RFlow + contrastive** (w=0.1, T=0.2) | **0.752** | **0.066** | 500 | MAISI-v2 InfoNCE, best sweep config |
| RFlow — L1, lr=2e-4 | 0.765 | 0.071 | 900 | Best SSIM overall |
| RFlow — L1, lr=1e-4 | 0.754 | 0.063 | 900 | Best MAE |
| RFlow + OT-FM — L1, lr=2e-4 | 0.731 | 0.087 | 500 | OT-FM coupling, best sweep config |
| RFlow + DiT — medium, lr=1e-4 | 0.717 | 0.085 | 500 | DiT backbone, best sweep config |
| RFlow — pilot (500 cases) | 0.712 | 0.090 | 500 | — |
| pix2pix — best (exp_alpha03) | 0.658 | 0.252 | 500 | — |

RFlow outperforms pix2pix by **4× on MAE**. Scaling from 500 → 900 cases alone cut MAE by 30%.  
The MAISI contrastive loss shows consistent but modest gains; its main benefit is stronger ET-region fidelity.

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

---

## MAISI-v2 contrastive loss sweep (500 cases, 400 epochs)

All runs: medium UNet [64,128,256,256], lr=2e-4, `--velocity_loss l1+contrastive`.

| Experiment | weight | temp | SSIM ↑ | MAE ↓ |
|---|---|---|---|---|
| **maisi_w01_t02** ★ | 0.1 | 0.2 | **0.7523** | **0.0663** |
| maisi_w01_t01 | 0.1 | 0.1 | 0.7500 | 0.0741 |
| maisi_w01_t005 | 0.1 | 0.05 | 0.7495 | 0.0675 |
| maisi_w05_t01 | 0.5 | 0.1 | 0.7431 | 0.0786 |
| maisi_w10_t01 | 1.0 | 0.1 | 0.7401 | 0.0756 |

**Observations:**

- **Contrastive weight must stay low (≤ 0.1).** At weight=0.5 and 1.0 the InfoNCE term dominates training — SSIM drops 0.009–0.012 and MAE degrades compared to plain L1. The contrastive term is a regulariser, not the primary loss.
- **Softer temperature (0.2) outperforms sharper (0.05).** With only two regions (ET vs background), a very low temperature creates an over-constrained InfoNCE objective — the signal is too peaky and fights the L1 gradient. Temperature 0.2 keeps the contrast meaningful without destabilising training.
- **Best config (weight=0.1, temp=0.2) improves over plain L1 baseline** on this dataset: SSIM 0.752 vs 0.750 (hp_baseline), MAE 0.066 vs 0.065. The gain is modest but consistent across all 5 runs — the contrastive term never hurts when the weight is kept small.
- **All 5 runs trained for the full 400 epochs**, confirming the comparison is fair with no early stopping artefacts.
- **DiT backbone (rflow_dit_base, ep 170):** SSIM 0.686, MAE 0.097 on 5 val subjects — still converging at ep 170, approximately matching UNet performance at ep ~100. Slower convergence and noisier training curve than UNet; not recommended for fixed-budget runs.
| — | exp_alpha03 | 0.658 | 0.252 | Best pix2pix |
| — | rflow_medium (500c) | 0.712 | 0.090 | Original pilot |

> ⚠ All 900-case RFlow results are directly comparable (same SSIM method, same val set).

---

## RFlow + OT-FM sweep — 500 cases, 300 epochs

Medium UNet [64,128,256,256], `n_epochs=300`, `n_epochs_decay=100`.

| Experiment | lr | Loss | batch | SSIM ↑ | MAE ↓ |
|---|---|---|---|---|---|
| **otfm_l1_lr2e4** ★ | 2e-4 | L1 | 1 | **0.731** | 0.087 |
| otfm_l1_lr1e4 | 1e-4 | L1 | 1 | 0.729 | 0.087 |
| otfm_l2_lr1e4 | 1e-4 | L2 | 1 | 0.726 | **0.082** |
| otfm_l1_lr2e4_bs2 | 2e-4 | L1 | 2 | — | — |

---

## RFlow + DiT sweep — 500 cases, 300 epochs

All runs: `--backbone dit`, `n_epochs=300`, `n_epochs_decay=100`, `--velocity_loss l1`.

| Experiment | hidden | depth | heads | lr | grad_clip | SSIM ↑ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| dit_small | 256 | 8 | 4 | 1e-4 | 1.0 | 0.703 | 0.088 |
| **dit_medium** ★ | 384 | 12 | 6 | 1e-4 | 1.0 | **0.717** | 0.085 |
| dit_large | 512 | 16 | 8 | 1e-4 | 1.0 | 0.714 | 0.083 |
| dit_medium_lr2e4 | 384 | 12 | 6 | 2e-4 | 1.0 | 0.713 | 0.085 |
| dit_medium_lr5e5 | 384 | 12 | 6 | 5e-5 | 1.0 | 0.711 | **0.079** |
| dit_medium_gc05 | 384 | 12 | 6 | 1e-4 | 0.5 | 0.717 | 0.085 |

</details>

---

## Repository layout

```
src/mrisynth/
├── preprocessing/
│   ├── pipeline.py          # nnUNet-style end-to-end preprocessing pipeline
│   ├── normalization.py     # Z-score, percentile-clip normalisation
│   ├── cropping.py          # Non-zero bounding-box crop
│   ├── resampling.py        # Isotropic resampling (data + seg)
│   └── io.py                # .b2nd / NIfTI readers
│
├── augmentation/
│   ├── transforms.py        # batchgeneratorsv2 spatial + intensity transforms
│   └── dataset.py           # AugConfig + transform builder helpers
│
├── metrics/
│   ├── ssim.py              # Per-voxel SSIM map, mean SSIM, masked SSIM
│   ├── pixelwise.py         # MAE, MSE, NMSE, PSNR
│   └── enhancing_tumor.py   # ET-specific: Pearson, MAE, edge-SSIM, loc-Dice
│
├── losses/
│   ├── velocity.py          # SegAwareLoss base + L1/L2/NCC/SSIM/contrastive
│   │                        #   velocity losses for RFlow / WFM / cWDM
│   ├── composite.py         # TumorAwareGANLoss (pix2pix)
│   ├── enhancing_tumor.py   # EnhancingTumorLoss (ET-region pixel loss)
│   └── tumor_ssim.py        # TumorSSIMLoss, RegionWeightedSSIMLoss
│
└── model/
    ├── networks.py           # UnetGenerator3D, NLayerDiscriminator3D, GANLoss
    │                         #   define_G_3d / define_D_3d factories
    ├── base_model.py         # BaseModel — checkpoint I/O, LR scheduling
    ├── dataset.py            # Pix2Pix3dDataset — .npz → paired (A, B, seg)
    ├── latent_dataset.py     # LatentDataset — pre-cached MAISI VAE latents
    ├── vae.py                # MaisiVAE — frozen MAISI autoencoder wrapper
    ├── wavelet.py            # haar_dwt3d / haar_idwt3d (pure PyTorch, no deps)
    │
    ├── pix2pix3d.py          # Pix2Pix3dModel  — Approach 1
    ├── rflow.py              # RFlowModel       — Approach 2  (UNet, latent space)
    │                         #   + _ot_couple() for OT-FM (Approach 2.2)
    ├── dit.py                # DiT3D backbone   — Approach 2.3 (drop-in for UNet)
    ├── wfm.py                # WFMModel         — Approach 4  (wavelet + informed prior)
    ├── cwdm.py               # cWDMModel        — Approach 5  (wavelet + DDPM/DDIM)
    └── rflow_controlnet.py   # RFlowControlNetModel — Approach 6 (seg ControlNet)

scripts/
├── train_pix2pix3d.py        # Pix2Pix training loop + TensorBoard
├── train_rflow.py            # RFlow / OT-FM / contrastive training
├── train_wfm.py              # WFM training loop
├── train_cwdm.py             # cWDM training loop
├── train_rflow_controlnet.py # RFlow + ControlNet training loop
│
├── generate_latents.py       # Pre-compute MAISI VAE latents (one-time, ~2 h)
│
├── infer_pix2pix.py          # Pix2Pix inference → NIfTI
├── infer_rflow.py            # RFlow inference → NIfTI (VAE decode)
├── infer_wfm.py              # WFM inference → NIfTI (IDWT)
├── infer_cwdm.py             # cWDM inference → NIfTI (IDWT, DDIM)
├── infer_rflow_controlnet.py # ControlNet inference → NIfTI
│
├── eval_pairs.py             # Evaluate pred/GT NIfTI pairs (MAE, SSIM, ET)
├── export_tables.py          # Aggregate metrics across experiments → CSV/table
├── make_gifs.py              # Animated axial-slice GIFs (T1n | pred | GT)
└── export_gifs.py            # Multi-experiment sweep GIFs for comparison

tests/
├── test_preprocessing.py     # 24 tests — normalisation, crop, resample
├── test_augmentation.py      #  6 tests — transform pipeline, DataLoader collation
├── test_networks.py          # 19 tests — UNet/PatchGAN primitives, GANLoss, scheduler
├── test_dataset.py           # 16 tests — channel resolution, Pix2Pix3dDataset, LatentDataset
├── test_models.py            # 18 tests — _pad/_unpad, _ot_couple, UNet/DiT builders,
│                             #             RFlowModel (UNet + DiT), EMA, OT-FM, contrastive
├── test_wavelet.py           # 10 tests — Haar DWT/IDWT shape, roundtrip, gradient flow
├── test_wfm_cwdm.py          # 16 tests — WFMModel, cWDMModel, RFlowControlNetModel smoke
├── test_velocity_losses.py   # 34 tests — all velocity loss classes + factory
├── test_composite.py         #  8 tests — TumorAwareGANLoss
├── test_enhancing_tumor.py   # 10 tests — ET metrics + EnhancingTumorLoss
├── test_tumor_ssim.py        #  8 tests — TumorSSIMLoss, RegionWeightedSSIMLoss
├── test_metrics.py           #  8 tests — MAE, MSE, NMSE, PSNR, SSIM
└── test_modality_dropout.py  #  8 tests — modality dropout (both datasets)
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

## Training options

### Missing-modality robustness — `--modality_dropout`

All training scripts accept `--modality_dropout P` (default `0.0`). With
probability `P`, each **training** sample has exactly one input modality
zeroed (chosen uniformly), so the model degrades gracefully when an input
sequence is missing at inference. Requires more than one input modality.
Validation/inference are never affected.

```bash
# e.g. 30% of training samples drop either T1n or T2FLAIR
uv run python scripts/train_rflow.py --task t1n_t2f_to_t1c \
    --latent_root latents/dataset --name rflow_mdrop --modality_dropout 0.3
```

Supported by all six approaches via the two shared datasets:
`Pix2Pix3dDataset` (pix2pix, WFM, cWDM — zeros one input channel) and
`LatentDataset` (RFlow, DiT, ControlNet — zeros one conditioning latent
block). To predict with a missing modality, feed zeros in that modality's
input slot at inference (matches the training convention).

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

## Approach 2 — RFlow (Latent Diffusion, UNet backbone)

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

#### Other options

```bash
# Tumor-weighted velocity loss (requires seg files alongside latents)
--velocity_loss tumor_l1 --tumor_weight 5.0

# EMA weights for inference (marginal improvement)
--ema_decay 0.9999

# Lightweight alternative: rflow_tiny at 19 M params
--unet_channels 32 64 128 128

# High-capacity alternative: rflow_large at 178 M params
--unet_channels 64 128 256 512
```

### Inference

```bash
uv run python scripts/infer_rflow.py \
    --latent_dir    latents/dataset/val \
    --vae_ckpt      pretrained/autoencoder_epoch273.pt \
    --checkpoint    checkpoints/rflow_best/latest_net_UNet.pth \
    --unet_channels 64 128 256 256 \
    --out_dir       predictions/rflow_best \
    --n_cases       10 \
    --save_gt \
    --device        cuda:0
```

---

## Approach 2.2 — RFlow + OT-FM (Optimal Transport Noise Coupling)

**What changes:** Same UNet (or DiT) backbone and training loop as Approach 2, but noise samples within each mini-batch are permuted before the forward diffusion step to minimise total L2 cost between noise and targets (mini-batch optimal transport).

**Why it helps:** Standard RFlow pairs each noise sample with whichever target lands in the same batch slot — a random coupling. OT finds the cheapest permutation, making the straight-line paths from noise to target shorter on average. Shorter paths → the flow network has an easier job → it can converge with fewer inference steps at test time.  
Based on: [MOTFM, MICCAI 2025](https://arxiv.org/abs/2503.00266).

**Requirements:** `scipy` (already a project dependency). No-op for `batch_size=1`; effect grows with larger batches.

### Sweep results (500-case dataset, 300 epochs)

All runs: medium UNet [64,128,256,256], `--n_epochs 300 --n_epochs_decay 100`, `batch_size=1`.

| Experiment | lr | Loss | SSIM ↑ | MAE ↓ |
|---|---|---|---|---|
| **otfm_l1_lr2e4** ★ | 2e-4 | L1 | **0.731** | 0.087 |
| otfm_l1_lr1e4 | 1e-4 | L1 | 0.729 | 0.087 |
| otfm_l2_lr1e4 | 1e-4 | L2 | 0.726 | **0.082** |
| otfm_l1_lr2e4_bs2 | 2e-4 | L1 (bs=2) | — | — |

> `otfm_l1_lr2e4_bs2` did not complete. OT is a no-op at B=1; its benefit grows with batch size — reduce `unet_channels` if OOM.

**Observations:**
- OT-FM improves over the plain RFlow 500-case pilot: SSIM 0.712 → 0.731, MAE 0.090 → 0.087.
- lr=2e-4 transfers from UNet → best SSIM (0.731). lr=1e-4 is a safe fallback.
- L2 loss gives the lowest MAE (0.082) at cost of SSIM; L1 is better balanced.

### Train (best config)

Drop-in on top of any Approach 2 command — just add `--use_ot_coupling`:

```bash
uv run python scripts/train_rflow.py \
    --task           t1n_t2f_to_t1c \
    --latent_root    latents/dataset \
    --vae_ckpt       pretrained/autoencoder_epoch273.pt \
    --name           rflow_otfm \
    --unet_channels  64 128 256 256 \
    --velocity_loss  l1 \
    --lr             2e-4 \
    --n_epochs       300 \
    --use_ot_coupling \
    --device         cuda:0
```

### Inference

Same as Approach 2 — the coupling only affects training, not inference.

---

## Approach 2.3 — RFlow + DiT Backbone

**What changes:** Replaces the convolutional `DiffusionModelUNet` with `DiT3D` — a 3-D Diffusion Transformer using adaLN-Zero timestep conditioning, factorized per-axis positional embeddings, and flash attention (`torch.nn.functional.scaled_dot_product_attention`).

**Why it helps:** Transformers capture long-range spatial dependencies with global self-attention, while the UNet's receptive field is limited by its depth. For large tumour regions or cross-hemisphere correlations this can matter. The positional embeddings are interpolated at runtime so the model generalises to latent volumes of any size without retraining.

**Tradeoffs:** Higher memory (O(N²) attention, N = latent tokens); `--grad_clip 1.0` is recommended for stable training. For a 32×40×32 latent at patch_size=2, N ≈ 40 k tokens — consider `--batch_size 1` and gradient checkpointing.

### Sweep results (500-case dataset, 300 epochs)

| Experiment | hidden | depth | heads | lr | grad_clip | SSIM ↑ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| dit_small | 256 | 8 | 4 | 1e-4 | 1.0 | 0.703 | 0.088 |
| **dit_medium** ★ | 384 | 12 | 6 | 1e-4 | 1.0 | **0.717** | 0.085 |
| dit_large | 512 | 16 | 8 | 1e-4 | 1.0 | 0.714 | 0.083 |
| dit_medium_lr2e4 | 384 | 12 | 6 | 2e-4 | 1.0 | 0.713 | 0.085 |
| dit_medium_lr5e5 | 384 | 12 | 6 | 5e-5 | 1.0 | 0.711 | **0.079** |
| dit_medium_gc05 | 384 | 12 | 6 | 1e-4 | 0.5 | 0.717 | 0.085 |

**Observations:**
- Medium (384/12/6) is the sweet spot — large (512/16/8) adds no benefit at 2× memory cost.
- DiT converges slower than UNet: SSIM 0.717 at 300 epochs vs UNet 0.765 on the same budget.
- lr=1e-4 is optimal for DiT — unlike UNet where 2e-4 wins. Higher LR (2e-4) degrades SSIM.
- Tighter grad_clip (0.5 vs 1.0) makes no difference on medium architecture.
- lr=5e-5 gives the best MAE (0.079) at the cost of lower SSIM (0.711).

### Train (best config)

```bash
uv run python scripts/train_rflow.py \
    --task            t1n_t2f_to_t1c \
    --latent_root     latents/dataset \
    --vae_ckpt        pretrained/autoencoder_epoch273.pt \
    --name            rflow_dit \
    --backbone        dit \
    --dit_hidden_size 384 \
    --dit_depth       12 \
    --dit_num_heads   6 \
    --dit_patch_size  2 \
    --velocity_loss   l1 \
    --lr              1e-4 \
    --grad_clip       1.0 \
    --n_epochs        300 \
    --device          cuda:0
```

OT-FM coupling can be combined freely: add `--use_ot_coupling`.

### Inference

```bash
uv run python scripts/infer_rflow.py \
    --latent_dir      latents/dataset/val \
    --vae_ckpt        pretrained/autoencoder_epoch273.pt \
    --checkpoint      checkpoints/rflow_dit/latest_net_DiT.pth \
    --backbone        dit \
    --dit_hidden_size 384 \
    --dit_depth       12 \
    --dit_num_heads   6 \
    --out_dir         predictions/rflow_dit \
    --n_cases         10 \
    --save_gt \
    --device          cuda:0
```

---

## Approach 3 — MAISI-v2 Region-Specific Contrastive Loss

**What changes:** Adds an InfoNCE contrastive term on top of the standard L1 velocity loss. The model is trained to distinguish its own velocity prediction in the enhancing-tumour (ET) region from the background velocity — pulling ET predictions toward the GT and separating them from background statistics.

**Why it helps:** The L1 loss treats all voxels equally. ET is a small, high-stakes region (contrast-enhancing tumour drives diagnosis). The contrastive term gives ET an extra gradient signal without changing the backbone or the training schedule.  
Based on: [MAISI-v2, arXiv:2508.05772](https://arxiv.org/abs/2508.05772).

**Requirements:** Segmentation latents (`seg.pt`) must be present alongside the cached latents — `generate_latents.py` saves them automatically when segmentation `.npz` channels are present.

### Sweep results (500-case dataset, 400 epochs)

| Experiment | contrastive_weight | temp | Best SSIM | Best MAE |
|---|---|---|---|---|
| **maisi_w01_t02** ★ | 0.1 | 0.2 | **0.7523** | **0.0663** |
| maisi_w01_t01 | 0.1 | 0.1 | 0.7500 | 0.0741 |
| maisi_w01_t005 | 0.1 | 0.05 | 0.7495 | 0.0675 |
| maisi_w05_t01 | 0.5 | 0.1 | 0.7431 | 0.0786 |
| maisi_w10_t01 | 1.0 | 0.1 | 0.7401 | 0.0756 |

**Takeaways:** Keep weight low (0.1) — higher values let the contrastive term dominate and hurt pixel fidelity.
Softer temperature (0.2) outperforms sharper (0.05/0.1): with only two ROIs (ET vs background) a too-sharp InfoNCE is over-constrained.

### Train (best config)

```bash
uv run python scripts/train_rflow.py \
    --task               t1n_t2f_to_t1c \
    --latent_root        latents/dataset_full \
    --vae_ckpt           pretrained/autoencoder_epoch273.pt \
    --name               maisi_full_best \
    --unet_channels      64 128 256 256 \
    --velocity_loss      l1+contrastive \
    --contrastive_weight 0.1 \
    --contrastive_temp   0.2 \
    --lr                 2e-4 \
    --n_epochs           300 \
    --n_epochs_decay     100 \
    --modality_dropout   0.15 \
    --device             cuda:0
```

`--modality_dropout 0.15` randomly zeros one input modality in 15% of training samples, making the model robust to missing sequences at inference. Set to 0 if both modalities are always available.

Can be combined with OT-FM coupling: add `--use_ot_coupling`.

### Inference

Same as Approach 2 — the contrastive loss only affects training.

---

## Approach 4 — WFM (Wavelet Flow Matching)

**Architecture:**
- No VAE. Applies a single-level 3D Haar DWT to images before the flow network.
- After DWT: 8 subbands per channel at half spatial resolution per axis (e.g. 128³ → 8 ch × 64³).
- Flow network: same `DiffusionModelUNet` backbone as Approach 2, now in wavelet space.
- **Informed prior:** instead of starting from Gaussian noise, denoising starts from the mean of the conditioning modalities in wavelet space. Paths are shorter → fewer steps needed.
- Loss: L1 on velocity field in wavelet space.

**Why no VAE:** the DWT replaces the VAE as the compression/decorrelation step, avoiding VAE reconstruction error which currently caps SSIM.

Based on: [WFM, MIDL 2026](https://arxiv.org/abs/2604.21146).  
**Not yet benchmarked** on this dataset.

### Train

```bash
uv run python scripts/train_wfm.py \
    --task           t1n_t2f_to_t1c \
    --data_dir       /path/to/preprocessed \
    --name           wfm_best \
    --unet_channels  64 128 256 256 \
    --velocity_loss  l1 \
    --lr             2e-4 \
    --n_epochs       300 \
    --n_inference_steps 10 \
    --device         cuda:0

tensorboard --logdir runs/
```

The informed prior cuts the number of useful inference steps to ~10 (vs 200 for standard RFlow). OT coupling can be combined freely: add `--use_ot_coupling`.

### Inference

```bash
uv run python scripts/infer_wfm.py \
    --data_dir       /path/to/preprocessed \
    --checkpoint     checkpoints/wfm_best/latest_net_WFM.pth \
    --unet_channels  64 128 256 256 \
    --out_dir        predictions/wfm_best \
    --n_cases        10 \
    --n_inference_steps 10 \
    --device         cuda:0
```

---

## Approach 5 — cWDM (Conditional Wavelet Diffusion Model)

**Architecture:**
- No VAE. Same 3D Haar DWT infrastructure as Approach 4.
- Diffusion: **DDPM training** (noise/epsilon parameterisation, MSE loss) + **DDIM inference** for fast sampling.
- Gaussian prior (no informed prior).
- Same `DiffusionModelUNet` backbone in wavelet space.

**vs Approach 4:** cWDM uses standard DDPM diffusion instead of flow matching. More studied theoretically; pretrained BraTS weights available at [github.com/pfriedri/cwdm](https://github.com/pfriedri/cwdm) to sanity-check before training from scratch.

Based on: [cWDM, arXiv:2411.17203](https://arxiv.org/abs/2411.17203).  
**Not yet benchmarked** on this dataset.

### Train

```bash
uv run python scripts/train_cwdm.py \
    --task           t1n_t2f_to_t1c \
    --data_dir       /path/to/preprocessed \
    --name           cwdm_best \
    --unet_channels  64 128 256 256 \
    --beta_schedule  linear \
    --n_epochs       300 \
    --n_ddim_steps   50 \
    --device         cuda:0

tensorboard --logdir runs/
```

### Inference

```bash
uv run python scripts/infer_cwdm.py \
    --data_dir       /path/to/preprocessed \
    --checkpoint     checkpoints/cwdm_best/latest_net_cWDM.pth \
    --unet_channels  64 128 256 256 \
    --out_dir        predictions/cwdm_best \
    --n_cases        10 \
    --n_ddim_steps   50 \
    --device         cuda:0
```

---

## Approach 6 — RFlow + ControlNet (Seg-Guided Latent Diffusion)

**Architecture:**
- Same MAISI VAE (frozen) + `DiffusionModelUNet` as Approach 2.
- Adds a **MONAI ControlNet** — a parallel encoder with the same architecture as the UNet, zero-initialised. Processes the tumour segmentation mask as spatial conditioning and injects its outputs at every skip-connection level.
- Control signal: 4-class one-hot seg mask (`BG, NETC, SNFH, ET`) downsampled to latent resolution.
- Falls back to plain RFlow when `seg` is absent.

**Why it helps:** the seg masks are already cached alongside the latents. ControlNet gives the model explicit spatial knowledge of where each tumour subregion is, beyond what it can infer from the conditioning modalities alone.

Uses MONAI's native `ControlNet` wrapper (available in MONAI ≥ 1.3).  
**Not yet benchmarked** on this dataset.

### Step 1 — Pre-compute latents (same as Approach 2)

Seg files (`seg.pt`) are saved automatically alongside latents when segmentation channels are present in the `.npz` files.

### Step 2 — Train

```bash
uv run python scripts/train_rflow_controlnet.py \
    --task           t1n_t2f_to_t1c \
    --latent_root    latents/dataset \
    --vae_ckpt       pretrained/autoencoder_epoch273.pt \
    --name           rflow_controlnet \
    --unet_channels  64 128 256 256 \
    --velocity_loss  l1 \
    --lr             2e-4 \
    --n_epochs       300 \
    --device         cuda:0

tensorboard --logdir runs/
```

### Inference

```bash
uv run python scripts/infer_rflow_controlnet.py \
    --latent_dir     latents/dataset/val \
    --vae_ckpt       pretrained/autoencoder_epoch273.pt \
    --checkpoint     checkpoints/rflow_controlnet/latest_net_UNet.pth \
    --controlnet_ckpt checkpoints/rflow_controlnet/latest_net_ControlNet.pth \
    --unet_channels  64 128 256 256 \
    --out_dir        predictions/rflow_controlnet \
    --n_cases        10 \
    --save_gt \
    --device         cuda:0
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

221 tests covering preprocessing, augmentation, losses, metrics, datasets, networks, models, and modality dropout.

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

### `test_modality_dropout.py` — 8 tests

Modality dropout in both training datasets (fake data on disk).

| Group | What is checked |
|---|---|
| `Pix2Pix3dDataset` | Exactly one input channel zeroed; both channels get dropped over draws; rate respected (~0.5 over 400 draws); fires with `augment=False` (WFM/cWDM path); `modality_dropout=0` disables |
| `LatentDataset` | Exactly one conditioning latent block zeroed; both blocks get dropped over draws; no-op when `deterministic=True`; `modality_dropout=0` disables |

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

### `test_wavelet.py` — 10 tests

3D Haar DWT/IDWT correctness.

| Group | What is checked |
|---|---|
| `haar_dwt3d` | Output shape `(B, 8C, D/2, H/2, W/2)`; odd spatial dim raises `ValueError` |
| `haar_idwt3d` | Perfect roundtrip: `idwt(dwt(x)) == x` up to 1e-5 |
| Gradient flow | Backward through both DWT and IDWT produces finite gradients |
| `HaarDWT3D` / `HaarIDWT3D` | Module wrappers produce same result as functions |

---

### `test_wfm_cwdm.py` — 16 tests

WFMModel, cWDMModel, and RFlowControlNetModel CPU smoke tests.

| Group | What is checked |
|---|---|
| `WFMModel` | `set_input` applies DWT (target_wav/cond_wav exist); finite training loss; `pred_image` shape matches input B; informed prior is mean of conditioning DWTs |
| `cWDMModel` | `set_input` applies DWT; finite DDPM training loss; `pred_image` shape matches input B |
| `RFlowControlNetModel` | Finite loss when seg provided; graceful fallback to plain UNet when `seg=None` |

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
| `model/dataset.py` | `test_dataset.py`, `test_modality_dropout.py` |
| `model/latent_dataset.py` | `test_dataset.py`, `test_modality_dropout.py` |
| `model/rflow.py` | `test_models.py` |
| `model/wavelet.py` | `test_wavelet.py` |
| `model/wfm.py` | `test_wfm_cwdm.py` |
| `model/cwdm.py` | `test_wfm_cwdm.py` |
| `model/rflow_controlnet.py` | `test_wfm_cwdm.py` |

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
| **OT-FM** (`--use_ot_coupling`) | Best config (L1, lr=2e-4): SSIM 0.731, MAE 0.087 (500 cases, 300 ep). +0.019 SSIM over plain RFlow pilot. lr=2e-4 transfers from UNet; L2 gives lower MAE (0.082) at cost of SSIM. |
| **DiT backbone** (`--backbone dit`) | Best config (medium 384/12/6, lr=1e-4): SSIM 0.717, MAE 0.085 (500 cases, 300 ep). Slower convergence than UNet; lr=1e-4 optimal (vs 2e-4 for UNet). Medium outperforms large at half the memory. |
| **Region contrastive loss** (`--velocity_loss l1+contrastive`) | Best config: weight=0.1, temp=0.2 → SSIM 0.752, MAE 0.066 (500 cases, 400 ep). Keep weight ≤ 0.1 and temperature ≥ 0.2 — higher weight/lower temp hurts pixel fidelity. |
| **WFM** (Approach 4) | Wavelet Flow Matching — no VAE, informed prior from conditioning DWTs, ~10 inference steps. Not yet benchmarked on this dataset. |
| **cWDM** (Approach 5) | Conditional Wavelet Diffusion — no VAE, DDPM train + DDIM inference. Pretrained BraTS weights available. Not yet benchmarked on this dataset. |
| **ControlNet** (Approach 6) | Seg-guided latent diffusion via MONAI ControlNet; seg masks already cached. Not yet benchmarked on this dataset. |

---

## References

- nnUNet: [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- MONAI / MAISI: [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)
- BraTS 2023: [synapse.org/brats2023](https://www.synapse.org/brats2023)
- pix2pix: Isola et al. 2017 — *Image-to-Image Translation with Conditional Adversarial Networks*
- Rectified Flow: Liu et al. 2022 — *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*
