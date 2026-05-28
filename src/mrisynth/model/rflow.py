"""Rectified Flow latent diffusion model for MRI synthesis.

Supports two backbone architectures selected via opt.backbone:
  "unet"  — DiffusionModelUNet (3-D conv U-Net, default, proven best)
  "dit"   — DiT3D (3-D Diffusion Transformer, adaLN-Zero)

Input to backbone: cat([noisy_T1c (4ch), T1n_cond (4ch), T2F_cond (4ch)]) = 12 ch
Output:            velocity prediction (4 ch)
Loss:              configurable velocity loss (default L1)

The VAE is NOT part of this model — it is used externally in generate_latents.py
and in train_rflow.py for optional image-space validation.

Expected opt fields (in addition to BaseModel fields)
------------------------------------------------------
  backbone         str    — "unet" or "dit" (default "unet")
  latent_channels  int    — VAE latent channels (default 4)
  unet_channels    list   — UNet channel widths per level (backbone=unet only)
  num_res_blocks   int    — ResNet blocks per UNet level (backbone=unet only)
  n_attention_levels int  — deepest UNet levels with self-attention (unet only)
  dit_patch_size   int    — spatial patch size (backbone=dit only, default 2)
  dit_hidden_size  int    — transformer hidden dim (default 384)
  dit_depth        int    — number of DiT blocks (default 12)
  dit_num_heads    int    — attention heads (default 6)
  n_timesteps      int    — training timesteps for RFlow (default 1000)
  n_inference_steps int   — denoising steps at val/inference (default 200)
  lr               float
  beta1            float
  weight_decay     float  — AdamW weight decay (default 1e-4)
  velocity_loss    str    — loss name (default "l1")
  ema_decay        float  — EMA decay for inference weights (default 0.0 = off)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import RFlowScheduler

from .base_model import BaseModel
from .dit import DiT3D
from ..losses.velocity import build_velocity_loss, SegAwareLoss


def _ot_couple(noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Permute a batch of noise samples to minimise total L2 cost to targets (mini-batch OT).

    Straight-line paths from noise to data are shorter on average, which lets the
    flow network learn with fewer integration steps at inference time.

    For B == 1 the function is a no-op (no pairing to do).
    Uses scipy.optimize.linear_sum_assignment (already a project dependency).
    """
    B = noise.shape[0]
    if B < 2:
        return noise
    from scipy.optimize import linear_sum_assignment
    n = noise.detach().float().flatten(1).cpu().numpy()   # (B, N)
    t = target.detach().float().flatten(1).cpu().numpy()  # (B, N)
    # ||t_i - n_j||^2 = ||t_i||^2 + ||n_j||^2 - 2 * (t @ n.T)  — memory-efficient
    cost = (t * t).sum(1, keepdims=True) + (n * n).sum(1, keepdims=True).T - 2.0 * (t @ n.T)
    _, col_ind = linear_sum_assignment(cost)   # col_ind[i] = best noise index for target i
    return noise[col_ind]


def _pad(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Pad spatial dims to be divisible by factor. Returns (padded, original_DHW)."""
    D, H, W = x.shape[-3:]
    pd = (factor - D % factor) % factor
    ph = (factor - H % factor) % factor
    pw = (factor - W % factor) % factor
    if pd or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pd))
    return x, (D, H, W)


def _unpad(x: torch.Tensor, original: tuple[int, int, int]) -> torch.Tensor:
    D, H, W = original
    return x[..., :D, :H, :W]


# ---------------------------------------------------------------------------
# UNet factory
# ---------------------------------------------------------------------------

def _build_unet(
    in_channels: int,
    out_channels: int,
    num_channels: list[int],
    n_attention_levels: int = 2,
    num_res_blocks: int = 2,
) -> nn.Module:
    """3-D timestep-conditioned diffusion UNet."""
    n_levels = len(num_channels)
    attention_levels = [False] * n_levels
    for i in range(max(0, n_levels - n_attention_levels), n_levels):
        attention_levels[i] = True

    num_head_channels = [
        (num_channels[i] // 4) if attention_levels[i] else 0
        for i in range(n_levels)
    ]

    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in num_channels):
        norm_num_groups //= 2

    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=num_channels,
        attention_levels=attention_levels,
        num_head_channels=num_head_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        with_conditioning=False,
        resblock_updown=True,
    )


# ---------------------------------------------------------------------------
# DiT factory
# ---------------------------------------------------------------------------

def _build_dit(
    in_channels: int,
    out_channels: int,
    patch_size: int = 2,
    hidden_size: int = 384,
    depth: int = 12,
    num_heads: int = 6,
) -> nn.Module:
    """3-D timestep-conditioned Diffusion Transformer."""
    return DiT3D(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RFlowModel(BaseModel):
    """Rectified Flow latent diffusion — trained on pre-cached VAE latents.

    Follows the same BaseModel interface as Pix2Pix3dModel.
    Call  set_input(batch)  then  optimize_parameters()  each training step.
    For validation, call  set_input(batch)  then  forward()  to get
    self.pred_latent (the denoised T1CE latent).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train: bool = True):
        parser.add_argument(
            "--backbone", type=str, default="unet", choices=["unet", "dit"],
            help="Velocity backbone: 'unet' (DiffusionModelUNet) or 'dit' (DiT3D).",
        )
        parser.add_argument(
            "--latent_channels", type=int, default=4,
            help="VAE latent channels (must match the checkpoint, default 4).",
        )
        parser.add_argument(
            "--n_cond", type=int, default=2,
            help="Number of conditioning modalities (sets backbone in_channels).",
        )
        # UNet-specific
        parser.add_argument(
            "--unet_channels", nargs="+", type=int, default=[128, 256, 512, 512],
            help="[unet] DiffusionUNet channel widths per level.",
        )
        parser.add_argument(
            "--num_res_blocks", type=int, default=2,
            help="[unet] ResNet blocks per UNet level.",
        )
        parser.add_argument(
            "--n_attention_levels", type=int, default=2,
            help="[unet] Number of deepest UNet levels that get self-attention.",
        )
        # DiT-specific
        parser.add_argument(
            "--dit_patch_size", type=int, default=2,
            help="[dit] Spatial patch size (D, H, W). Default 2.",
        )
        parser.add_argument(
            "--dit_hidden_size", type=int, default=384,
            help="[dit] Transformer hidden dimension. Presets: 256=small, 384=base, 512=large.",
        )
        parser.add_argument(
            "--dit_depth", type=int, default=12,
            help="[dit] Number of DiT blocks.",
        )
        parser.add_argument(
            "--dit_num_heads", type=int, default=6,
            help="[dit] Number of attention heads (hidden_size must be divisible by this).",
        )
        # Shared
        parser.add_argument(
            "--n_timesteps", type=int, default=1000,
            help="Number of RFlow training timesteps.",
        )
        parser.add_argument(
            "--n_inference_steps", type=int, default=200,
            help="Denoising steps used during validation / inference.",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4,
            help="AdamW weight decay.",
        )
        parser.add_argument(
            "--velocity_loss", type=str, default="l1",
            choices=["l1", "l2", "ssim", "ncc", "l1+ssim", "tumor_l1", "et_l1", "l1+contrastive"],
            help="Velocity loss. ncc/ssim are robust to cross-scanner intensity shifts.",
        )
        parser.add_argument(
            "--loss_alpha", type=float, default=0.5,
            help="Blend weight for l1+ssim (alpha*L1 + (1-alpha)*SSIM).",
        )
        parser.add_argument(
            "--contrastive_weight", type=float, default=0.1,
            help="[l1+contrastive] Weight λ for the ET contrastive term.",
        )
        parser.add_argument(
            "--contrastive_temp", type=float, default=0.1,
            help="[l1+contrastive] InfoNCE temperature.",
        )
        parser.add_argument(
            "--use_ot_coupling", action="store_true",
            help="Use mini-batch OT to couple noise to targets (OT-FM). Requires scipy.",
        )
        parser.add_argument(
            "--ema_decay", type=float, default=0.0,
            help="EMA decay for inference weights (0 = disabled, 0.9999 recommended).",
        )
        parser.add_argument(
            "--grad_clip", type=float, default=0.0,
            help="Max gradient norm for clipping (0 = disabled). Recommended 1.0 for DiT.",
        )
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names   = ["rflow"]
        self.visual_names = []

        lat_ch   = getattr(opt, "latent_channels", 4)
        n_cond   = getattr(opt, "n_cond", 2)
        backbone = getattr(opt, "backbone", "unet")
        in_ch    = lat_ch * (1 + n_cond)

        if backbone == "dit":
            patch_size = getattr(opt, "dit_patch_size", 2)
            self.netDiT = _build_dit(
                in_channels=in_ch,
                out_channels=lat_ch,
                patch_size=patch_size,
                hidden_size=getattr(opt, "dit_hidden_size", 384),
                depth=getattr(opt, "dit_depth", 12),
                num_heads=getattr(opt, "dit_num_heads", 6),
            )
            self.model_names = ["DiT"]
            self._backbone   = self.netDiT
            self._pad_factor = patch_size
        else:
            ch = getattr(opt, "unet_channels", [128, 256, 512, 512])
            self.netUNet = _build_unet(
                in_channels=in_ch,
                out_channels=lat_ch,
                num_channels=list(ch),
                n_attention_levels=getattr(opt, "n_attention_levels", 2),
                num_res_blocks=getattr(opt, "num_res_blocks", 2),
            )
            self.model_names = ["UNet"]
            self._backbone   = self.netUNet
            self._pad_factor = 2 ** (len(list(ch)) - 1)

        n_ts = getattr(opt, "n_timesteps", 1000)
        self.scheduler = RFlowScheduler(
            num_train_timesteps=n_ts,
            use_discrete_timesteps=True,
        )

        self._lat_ch         = lat_ch
        self._n_inf          = getattr(opt, "n_inference_steps", 200)
        self._use_ckpt       = getattr(opt, "use_checkpointing", True)
        self._ema_decay      = getattr(opt, "ema_decay", 0.0)
        self._grad_clip      = getattr(opt, "grad_clip", 0.0)
        self._use_ot_coupling = getattr(opt, "use_ot_coupling", False)
        self._ema_weights: dict | None = None

        if self.isTrain:
            vel_loss = getattr(opt, "velocity_loss", "l1")
            loss_kwargs = {
                "alpha":              getattr(opt, "loss_alpha", 0.5),
                "contrastive_weight": getattr(opt, "contrastive_weight", 0.1),
                "contrastive_temp":   getattr(opt, "contrastive_temp", 0.1),
            }
            tumor_weight = getattr(opt, "tumor_weight", None)
            if tumor_weight is not None:
                loss_kwargs["tumor_weight"] = tumor_weight
            self.criterion = build_velocity_loss(vel_loss, **loss_kwargs)
            self.optimizer_UNet = torch.optim.AdamW(
                self._backbone.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=getattr(opt, "weight_decay", 1e-4),
            )
            self.optimizers = [self.optimizer_UNet]
            self.loss_rflow = torch.tensor(0.0)
            self._scaler = torch.amp.GradScaler("cuda")

            if self._ema_decay > 0:
                self._ema_weights = {
                    k: v.clone().float().to(self.device)
                    for k, v in self._backbone.state_dict().items()
                }

    # ------------------------------------------------------------------
    def set_input(self, batch: dict):
        """Accept a batch from LatentDataset."""
        self.latent_tgt  = batch["latent_tgt"].to(self.device)
        self.latent_cond = batch["latent_cond"].to(self.device)
        self.case_ids    = batch.get("case_id", [])
        self.seg         = batch.get("seg", None)  # (B, D, H, W) or None

    def forward(self):
        """Full denoising inference (no grad).  Sets self.pred_latent.

        Uses EMA weights if available, falling back to live weights.
        """
        B = self.latent_cond.shape[0]
        lat_shape = (B, self._lat_ch) + tuple(self.latent_cond.shape[2:])

        z = torch.randn(lat_shape, device=self.device, dtype=self.latent_cond.dtype)
        self.scheduler.set_timesteps(self._n_inf, device=self.device)

        z_padded, orig = _pad(z, self._pad_factor)
        cond_padded, _ = _pad(self.latent_cond, self._pad_factor)

        # Swap in EMA weights for inference
        live_state = None
        if self._ema_weights is not None:
            live_state = {k: v.clone() for k, v in self._backbone.state_dict().items()}
            self._backbone.load_state_dict(
                {k: v.to(self.device) for k, v in self._ema_weights.items()}
            )

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_batch = torch.full(
                    (B,), t.item(), device=self.device, dtype=torch.long
                )
                x_in = torch.cat([z_padded, cond_padded], dim=1)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    v_pred = self._backbone(x_in, timesteps=t_batch)
                z_padded = self.scheduler.step(v_pred.float(), t.item(), z_padded.float())[0]

        if live_state is not None:
            self._backbone.load_state_dict(live_state)

        self.pred_latent = _unpad(z_padded, orig)  # (B, 4, D', H', W')

    def optimize_parameters(self):
        """One RFlow training step.

        RFlow velocity target: v = x_0 - noise  (direction from noise → clean).
        add_noise() in monai returns only the noisy sample; velocity computed here.
        """
        B = self.latent_tgt.shape[0]

        t = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,), device=self.device
        )
        noise = torch.randn_like(self.latent_tgt)
        if self._use_ot_coupling:
            noise = _ot_couple(noise, self.latent_tgt)
        noisy_tgt = self.scheduler.add_noise(self.latent_tgt, noise, t)
        target_v  = self.latent_tgt - noise           # RFlow velocity target

        noisy_padded, orig = _pad(noisy_tgt, self._pad_factor)
        cond_padded,  _    = _pad(self.latent_cond, self._pad_factor)
        x_in = torch.cat([noisy_padded, cond_padded], dim=1)

        self.optimizer_UNet.zero_grad()
        with torch.amp.autocast("cuda"):
            if self._use_ckpt:
                v_padded = grad_checkpoint(
                    self._backbone, x_in, t, use_reentrant=False
                )
            else:
                v_padded = self._backbone(x_in, timesteps=t)
            v_pred = _unpad(v_padded, orig)
            seg = getattr(self, "seg", None)
            if isinstance(self.criterion, SegAwareLoss):
                self.loss_rflow = self.criterion(v_pred, target_v, seg)
            else:
                self.loss_rflow = self.criterion(v_pred, target_v)
        self._scaler.scale(self.loss_rflow).backward()
        if self._grad_clip > 0:
            self._scaler.unscale_(self.optimizer_UNet)
            torch.nn.utils.clip_grad_norm_(self._backbone.parameters(), self._grad_clip)
        self._scaler.step(self.optimizer_UNet)
        self._scaler.update()

        if self._ema_decay > 0 and self._ema_weights is not None:
            with torch.no_grad():
                for k, v in self._backbone.state_dict().items():
                    self._ema_weights[k].mul_(self._ema_decay).add_(
                        v.float(), alpha=1.0 - self._ema_decay
                    )
