"""RFlow latent diffusion with MONAI ControlNet for seg-conditioned synthesis.

Architecture
------------
  Backbone : same DiffusionModelUNet as RFlowModel
  ControlNet: MONAI ControlNet with matching architecture, conditions on 4-class
              one-hot segmentation at latent spatial resolution.

When seg is None in set_input, falls back to plain RFlow (ControlNet skipped).

Expected opt fields: identical to RFlowModel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from monai.networks.nets import ControlNet
from monai.networks.schedulers import RFlowScheduler

from .base_model import BaseModel
from .rflow import _pad, _unpad, _build_unet, _build_dit, _ot_couple
from ..losses.velocity import build_velocity_loss, SegAwareLoss


def _build_controlnet(
    in_channels: int,
    num_channels: list[int],
    n_attention_levels: int = 2,
    num_res_blocks: int = 2,
) -> ControlNet:
    """Build a ControlNet with the same architecture as the companion UNet."""
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

    # conditioning_embedding_num_channels with a single element → no spatial downsampling
    # (each additional element adds a stride-2 conv, which would shrink latent dims)
    return ControlNet(
        spatial_dims=3,
        in_channels=in_channels,
        channels=num_channels,
        attention_levels=attention_levels,
        num_head_channels=num_head_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        resblock_updown=True,
        conditioning_embedding_in_channels=4,
        conditioning_embedding_num_channels=(num_channels[0],),
    )


class RFlowControlNetModel(BaseModel):
    """Rectified Flow with ControlNet seg conditioning — trained on pre-cached VAE latents."""

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names   = ["rflow"]
        self.visual_names = []

        lat_ch   = getattr(opt, "latent_channels", 4)
        n_cond   = getattr(opt, "n_cond", 2)
        backbone = getattr(opt, "backbone", "unet")
        in_ch    = lat_ch * (1 + n_cond)

        # Only UNet backbone supports ControlNet pairing
        if backbone == "dit":
            raise ValueError("ControlNet is only supported with backbone='unet'.")

        ch = getattr(opt, "unet_channels", [128, 256, 512, 512])
        n_attn = getattr(opt, "n_attention_levels", 2)
        n_res  = getattr(opt, "num_res_blocks", 2)

        self.netUNet = _build_unet(
            in_channels=in_ch,
            out_channels=lat_ch,
            num_channels=list(ch),
            n_attention_levels=n_attn,
            num_res_blocks=n_res,
        )
        self.netControlNet = _build_controlnet(
            in_channels=in_ch,
            num_channels=list(ch),
            n_attention_levels=n_attn,
            num_res_blocks=n_res,
        )

        self.model_names = ["UNet", "ControlNet"]
        self._backbone   = self.netUNet
        self._pad_factor = 2 ** (len(list(ch)) - 1)

        n_ts = getattr(opt, "n_timesteps", 1000)
        self.scheduler = RFlowScheduler(
            num_train_timesteps=n_ts,
            use_discrete_timesteps=True,
        )

        self._lat_ch          = lat_ch
        self._n_inf           = getattr(opt, "n_inference_steps", 200)
        self._use_ckpt        = getattr(opt, "use_checkpointing", True)
        self._ema_decay       = getattr(opt, "ema_decay", 0.0)
        self._grad_clip       = getattr(opt, "grad_clip", 0.0)
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
            all_params = (
                list(self.netUNet.parameters()) +
                list(self.netControlNet.parameters())
            )
            self.optimizer_UNet = torch.optim.AdamW(
                all_params,
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
        seg = batch.get("seg", None)
        self.seg = seg.to(self.device) if seg is not None else None

    def _get_controlnet_residuals(
        self,
        x_in_padded: torch.Tensor,
        t_batch: torch.Tensor,
        latent_spatial: tuple,
    ):
        """Compute ControlNet down/mid residuals, or None if seg is absent."""
        if self.seg is None:
            return None, None

        # Build 4-class one-hot seg at latent spatial dims
        seg = self.seg
        if seg.ndim == 5:
            # (B, 1, D, H, W) → (B, D, H, W)
            seg = seg[:, 0]
        # seg is now (B, D, H, W)
        seg_lat = F.interpolate(
            seg.float().unsqueeze(1), size=latent_spatial, mode="nearest"
        ).squeeze(1).long()
        seg_oh = F.one_hot(seg_lat, num_classes=4).permute(0, 4, 1, 2, 3).float()

        seg_oh_padded, _ = _pad(seg_oh, self._pad_factor)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            down_res, mid_res = self.netControlNet(
                x=x_in_padded,
                timesteps=t_batch,
                controlnet_cond=seg_oh_padded,
            )
        return down_res, mid_res

    def _forward_backbone(
        self,
        x_in_padded: torch.Tensor,
        t_batch: torch.Tensor,
        latent_spatial: tuple,
        use_ckpt: bool = False,
    ) -> torch.Tensor:
        """Run backbone with optional ControlNet residuals."""
        down_res, mid_res = self._get_controlnet_residuals(
            x_in_padded, t_batch, latent_spatial
        )

        if down_res is None:
            # No seg — plain RFlow
            if use_ckpt:
                return grad_checkpoint(
                    self._backbone, x_in_padded, t_batch, use_reentrant=False
                )
            return self._backbone(x_in_padded, timesteps=t_batch)

        if use_ckpt:
            def _fn(x, t):
                return self._backbone(
                    x, timesteps=t,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                )
            return grad_checkpoint(_fn, x_in_padded, t_batch, use_reentrant=False)

        return self._backbone(
            x_in_padded, timesteps=t_batch,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )

    def forward(self):
        """Full denoising inference (no grad).  Sets self.pred_latent."""
        B = self.latent_cond.shape[0]
        lat_shape = (B, self._lat_ch) + tuple(self.latent_cond.shape[2:])

        z = torch.randn(lat_shape, device=self.device, dtype=self.latent_cond.dtype)
        self.scheduler.set_timesteps(self._n_inf, device=self.device)

        z_padded, orig = _pad(z, self._pad_factor)
        cond_padded, _ = _pad(self.latent_cond, self._pad_factor)
        latent_spatial = tuple(self.latent_cond.shape[2:])

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
                    v_pred = self._forward_backbone(x_in, t_batch, latent_spatial)
                z_padded = self.scheduler.step(v_pred.float(), t.item(), z_padded.float())[0]

        if live_state is not None:
            self._backbone.load_state_dict(live_state)

        self.pred_latent = _unpad(z_padded, orig)

    def optimize_parameters(self):
        """One RFlow + ControlNet training step."""
        B = self.latent_tgt.shape[0]

        t = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,), device=self.device
        )
        noise = torch.randn_like(self.latent_tgt)
        if self._use_ot_coupling:
            noise = _ot_couple(noise, self.latent_tgt)
        noisy_tgt = self.scheduler.add_noise(self.latent_tgt, noise, t)
        target_v  = self.latent_tgt - noise

        noisy_padded, orig = _pad(noisy_tgt, self._pad_factor)
        cond_padded,  _    = _pad(self.latent_cond, self._pad_factor)
        x_in = torch.cat([noisy_padded, cond_padded], dim=1)
        latent_spatial = tuple(self.latent_tgt.shape[2:])

        self.optimizer_UNet.zero_grad()
        with torch.amp.autocast("cuda"):
            v_padded = self._forward_backbone(
                x_in, t, latent_spatial, use_ckpt=self._use_ckpt
            )
            v_pred = _unpad(v_padded, orig)
            seg = getattr(self, "seg", None)
            if isinstance(self.criterion, SegAwareLoss):
                self.loss_rflow = self.criterion(v_pred, target_v, seg)
            else:
                self.loss_rflow = self.criterion(v_pred, target_v)
        self._scaler.scale(self.loss_rflow).backward()
        if self._grad_clip > 0:
            self._scaler.unscale_(self.optimizer_UNet)
            all_params = (
                list(self.netUNet.parameters()) +
                list(self.netControlNet.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, self._grad_clip)
        self._scaler.step(self.optimizer_UNet)
        self._scaler.update()

        if self._ema_decay > 0 and self._ema_weights is not None:
            with torch.no_grad():
                for k, v in self._backbone.state_dict().items():
                    self._ema_weights[k].mul_(self._ema_decay).add_(
                        v.float(), alpha=1.0 - self._ema_decay
                    )
