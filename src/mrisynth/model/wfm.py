"""Wavelet Flow Matching model for MRI synthesis.

Same as RFlowModel but operates in 3D Haar wavelet space instead of VAE latent space.
Uses an informed prior (mean of conditioning modality DWTs) instead of Gaussian noise.

Expected opt fields
-------------------
  backbone         str    — "unet" or "dit"
  n_cond           int    — number of conditioning modalities
  unet_channels    list   — UNet channel widths per level
  num_res_blocks   int
  n_attention_levels int
  dit_patch_size   int
  dit_hidden_size  int
  dit_depth        int
  dit_num_heads    int
  n_timesteps      int
  n_inference_steps int
  velocity_loss    str
  loss_alpha       float
  lr               float
  beta1            float
  weight_decay     float
  ema_decay        float
  grad_clip        float
  use_ot_coupling  bool
  inference_noise  float  — noise added to prior at inference (default 0.0)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from monai.networks.schedulers import RFlowScheduler

from .base_model import BaseModel
from .rflow import _pad, _unpad, _build_unet, _build_dit, _ot_couple
from .wavelet import haar_dwt3d, haar_idwt3d
from ..losses.velocity import build_velocity_loss, SegAwareLoss


class WFMModel(BaseModel):
    """Wavelet Flow Matching — trained directly on image-space Pix2Pix3dDataset volumes."""

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names   = ["wfm"]
        self.visual_names = []

        n_cond   = getattr(opt, "n_cond", 2)
        backbone = getattr(opt, "backbone", "unet")
        # After DWT: 8 subbands per channel, 1 target channel → 8 output channels
        self._wav_ch = 8
        in_ch  = self._wav_ch * (1 + n_cond)
        out_ch = self._wav_ch

        if backbone == "dit":
            patch_size = getattr(opt, "dit_patch_size", 2)
            self.netWFM  = _build_dit(
                in_channels=in_ch,
                out_channels=out_ch,
                patch_size=patch_size,
                hidden_size=getattr(opt, "dit_hidden_size", 384),
                depth=getattr(opt, "dit_depth", 12),
                num_heads=getattr(opt, "dit_num_heads", 6),
            )
            self._pad_factor = patch_size
        else:
            ch = getattr(opt, "unet_channels", [128, 256, 512, 512])
            self.netWFM = _build_unet(
                in_channels=in_ch,
                out_channels=out_ch,
                num_channels=list(ch),
                n_attention_levels=getattr(opt, "n_attention_levels", 2),
                num_res_blocks=getattr(opt, "num_res_blocks", 2),
            )
            self._pad_factor = 2 ** (len(list(ch)) - 1)

        self.model_names = ["WFM"]
        self._backbone   = self.netWFM

        n_ts = getattr(opt, "n_timesteps", 1000)
        self.scheduler = RFlowScheduler(
            num_train_timesteps=n_ts,
            use_discrete_timesteps=True,
        )

        self._n_inf          = getattr(opt, "n_inference_steps", 200)
        self._use_ckpt       = getattr(opt, "use_checkpointing", True)
        self._ema_decay      = getattr(opt, "ema_decay", 0.0)
        self._grad_clip      = getattr(opt, "grad_clip", 0.0)
        self._use_ot_coupling = getattr(opt, "use_ot_coupling", False)
        self._inference_noise = getattr(opt, "inference_noise", 0.0)
        self._ema_weights: dict | None = None

        if self.isTrain:
            vel_loss = getattr(opt, "velocity_loss", "l1")
            loss_kwargs = {
                "alpha":              getattr(opt, "loss_alpha", 0.5),
                "contrastive_weight": getattr(opt, "contrastive_weight", 0.1),
                "contrastive_temp":   getattr(opt, "contrastive_temp", 0.1),
            }
            self.criterion = build_velocity_loss(vel_loss, **loss_kwargs)
            self.optimizer_WFM = torch.optim.AdamW(
                self._backbone.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=getattr(opt, "weight_decay", 1e-4),
            )
            self.optimizers = [self.optimizer_WFM]
            self.loss_wfm = torch.tensor(0.0)
            self._scaler = torch.amp.GradScaler("cuda")

            if self._ema_decay > 0:
                self._ema_weights = {
                    k: v.clone().float().to(self.device)
                    for k, v in self._backbone.state_dict().items()
                }

    # ------------------------------------------------------------------
    def set_input(self, batch: dict):
        """Accept a batch from Pix2Pix3dDataset; applies DWT to A and B."""
        A = batch["A"].to(self.device)   # (B, n_cond, D, H, W)
        B = batch["B"].to(self.device)   # (B, 1, D, H, W)

        # Pad to even dims before DWT
        B, orig = _pad(B, factor=2)
        A, _    = _pad(A, factor=2)

        self._orig_spatial = orig
        self.target_wav = haar_dwt3d(B)  # (B, 8, D/2, H/2, W/2)

        # DWT per conditioning modality, then concatenate
        n_cond = A.shape[1]
        cond_wavs = [haar_dwt3d(A[:, i:i+1]) for i in range(n_cond)]
        self.cond_wav = torch.cat(cond_wavs, dim=1)  # (B, 8*n_cond, D/2, H/2, W/2)

        self.seg      = batch.get("seg", None)
        self.case_ids = batch.get("A_paths", [])

    def _build_prior(self) -> torch.Tensor:
        """Informed prior: mean of conditioning wavelet subbands."""
        n_cond = self.cond_wav.shape[1] // self._wav_ch
        # Average corresponding subbands across conditioning modalities
        chunks = self.cond_wav.chunk(n_cond, dim=1)  # n_cond × (B, 8, ...)
        prior = torch.stack(chunks, dim=0).mean(dim=0)  # (B, 8, ...)
        return prior

    def forward(self):
        """Full denoising inference (no grad).  Sets self.pred_image."""
        B = self.cond_wav.shape[0]
        wav_shape = (B, self._wav_ch) + tuple(self.cond_wav.shape[2:])

        prior = self._build_prior()
        z = prior.clone()
        if self._inference_noise > 0.0:
            z = z + self._inference_noise * torch.randn_like(z)

        self.scheduler.set_timesteps(self._n_inf, device=self.device)

        z_padded, orig = _pad(z, self._pad_factor)
        cond_padded, _ = _pad(self.cond_wav, self._pad_factor)

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

        pred_wav = _unpad(z_padded, orig)                      # (B, 8, D/2, H/2, W/2)
        pred_img = haar_idwt3d(pred_wav)                       # (B, 1, D, H, W)
        self.pred_image = _unpad(pred_img, self._orig_spatial)  # crop to original size

    def optimize_parameters(self):
        """One WFM training step."""
        B = self.target_wav.shape[0]

        prior = self._build_prior()
        noise = prior  # informed prior as noise source
        if self._use_ot_coupling:
            noise = _ot_couple(noise, self.target_wav)

        t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device)
        noisy_tgt = self.scheduler.add_noise(self.target_wav, noise, t)
        target_v  = self.target_wav - noise  # RFlow velocity target

        noisy_padded, orig = _pad(noisy_tgt, self._pad_factor)
        cond_padded,  _    = _pad(self.cond_wav, self._pad_factor)
        x_in = torch.cat([noisy_padded, cond_padded], dim=1)

        self.optimizer_WFM.zero_grad()
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
                self.loss_wfm = self.criterion(v_pred, target_v, seg)
            else:
                self.loss_wfm = self.criterion(v_pred, target_v)
        self._scaler.scale(self.loss_wfm).backward()
        if self._grad_clip > 0:
            self._scaler.unscale_(self.optimizer_WFM)
            torch.nn.utils.clip_grad_norm_(self._backbone.parameters(), self._grad_clip)
        self._scaler.step(self.optimizer_WFM)
        self._scaler.update()

        if self._ema_decay > 0 and self._ema_weights is not None:
            with torch.no_grad():
                for k, v in self._backbone.state_dict().items():
                    self._ema_weights[k].mul_(self._ema_decay).add_(
                        v.float(), alpha=1.0 - self._ema_decay
                    )
