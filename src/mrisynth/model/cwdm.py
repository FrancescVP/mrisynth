"""Conditional Wavelet Diffusion Model (cWDM) for MRI synthesis.

Same wavelet infrastructure as WFMModel but uses DDPM training / DDIM inference
with epsilon (noise) parameterization and a Gaussian prior.

Expected opt fields
-------------------
  backbone         str    — "unet" or "dit"
  n_cond           int    — number of conditioning modalities
  unet_channels    list
  num_res_blocks   int
  n_attention_levels int
  dit_patch_size   int
  dit_hidden_size  int
  dit_depth        int
  dit_num_heads    int
  n_timesteps      int
  n_ddim_steps     int    — DDIM inference steps (default 50)
  beta_schedule    str    — "linear" or "scaled_linear" (default "linear")
  lr               float
  beta1            float
  weight_decay     float
  ema_decay        float
  grad_clip        float
  use_ot_coupling  bool
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

from .base_model import BaseModel
from .rflow import _pad, _unpad, _build_unet, _build_dit, _ot_couple
from .wavelet import haar_dwt3d, haar_idwt3d


class cWDMModel(BaseModel):
    """Conditional Wavelet Diffusion Model — DDPM train, DDIM inference."""

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names   = ["cwdm"]
        self.visual_names = []

        n_cond   = getattr(opt, "n_cond", 2)
        backbone = getattr(opt, "backbone", "unet")
        self._wav_ch = 8
        in_ch  = self._wav_ch * (1 + n_cond)
        out_ch = self._wav_ch

        if backbone == "dit":
            patch_size = getattr(opt, "dit_patch_size", 2)
            self.netcWDM = _build_dit(
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
            self.netcWDM = _build_unet(
                in_channels=in_ch,
                out_channels=out_ch,
                num_channels=list(ch),
                n_attention_levels=getattr(opt, "n_attention_levels", 2),
                num_res_blocks=getattr(opt, "num_res_blocks", 2),
            )
            self._pad_factor = 2 ** (len(list(ch)) - 1)

        self.model_names = ["cWDM"]
        self._backbone   = self.netcWDM

        n_ts  = getattr(opt, "n_timesteps", 1000)
        # MONAI schedulers use 'schedule' param with names like "linear_beta"
        beta_schedule = getattr(opt, "beta_schedule", "linear_beta")
        # Accept shorthand names for convenience
        _schedule_map = {"linear": "linear_beta", "scaled_linear": "scaled_linear_beta"}
        monai_schedule = _schedule_map.get(beta_schedule, beta_schedule)
        self._n_ddim  = getattr(opt, "n_ddim_steps", 50)

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=n_ts,
            schedule=monai_schedule,
        )
        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=n_ts,
            schedule=monai_schedule,
        )

        self._n_inf       = getattr(opt, "n_inference_steps", self._n_ddim)
        self._use_ckpt    = getattr(opt, "use_checkpointing", True)
        self._ema_decay   = getattr(opt, "ema_decay", 0.0)
        self._grad_clip   = getattr(opt, "grad_clip", 0.0)
        self._use_ot_coupling = getattr(opt, "use_ot_coupling", False)
        self._ema_weights: dict | None = None

        if self.isTrain:
            self.criterion = nn.MSELoss()
            self.optimizer_cWDM = torch.optim.AdamW(
                self._backbone.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=getattr(opt, "weight_decay", 1e-4),
            )
            self.optimizers = [self.optimizer_cWDM]
            self.loss_cwdm = torch.tensor(0.0)
            self._scaler = torch.amp.GradScaler("cuda")

            if self._ema_decay > 0:
                self._ema_weights = {
                    k: v.clone().float().to(self.device)
                    for k, v in self._backbone.state_dict().items()
                }

    # ------------------------------------------------------------------
    def set_input(self, batch: dict):
        """Accept a batch from Pix2Pix3dDataset; applies DWT to A and B."""
        A = batch["A"].to(self.device)
        B = batch["B"].to(self.device)

        B, orig = _pad(B, factor=2)
        A, _    = _pad(A, factor=2)

        self._orig_spatial = orig
        self.target_wav = haar_dwt3d(B)

        n_cond = A.shape[1]
        cond_wavs = [haar_dwt3d(A[:, i:i+1]) for i in range(n_cond)]
        self.cond_wav = torch.cat(cond_wavs, dim=1)

        self.seg      = batch.get("seg", None)
        self.case_ids = batch.get("A_paths", [])

    def forward(self):
        """DDIM denoising inference. Sets self.pred_image."""
        B = self.cond_wav.shape[0]

        z = torch.randn(
            (B, self._wav_ch) + tuple(self.cond_wav.shape[2:]),
            device=self.device, dtype=self.cond_wav.dtype,
        )
        self.infer_scheduler.set_timesteps(self._n_ddim, device=self.device)

        z_padded, orig = _pad(z, self._pad_factor)
        cond_padded, _ = _pad(self.cond_wav, self._pad_factor)

        live_state = None
        if self._ema_weights is not None:
            live_state = {k: v.clone() for k, v in self._backbone.state_dict().items()}
            self._backbone.load_state_dict(
                {k: v.to(self.device) for k, v in self._ema_weights.items()}
            )

        with torch.no_grad():
            for t in self.infer_scheduler.timesteps:
                t_batch = torch.full(
                    (B,), t.item(), device=self.device, dtype=torch.long
                )
                x_in = torch.cat([z_padded, cond_padded], dim=1)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    noise_pred = self._backbone(x_in, timesteps=t_batch)
                z_padded = self.infer_scheduler.step(noise_pred.float(), t.item(), z_padded.float())[0]

        if live_state is not None:
            self._backbone.load_state_dict(live_state)

        pred_wav = _unpad(z_padded, orig)
        pred_img = haar_idwt3d(pred_wav)
        self.pred_image = _unpad(pred_img, self._orig_spatial)

    def optimize_parameters(self):
        """One DDPM training step (epsilon parameterization)."""
        Bs = self.target_wav.shape[0]

        t = torch.randint(
            0, self.train_scheduler.num_train_timesteps, (Bs,), device=self.device
        )
        noise = torch.randn_like(self.target_wav)
        if self._use_ot_coupling:
            noise = _ot_couple(noise, self.target_wav)

        noisy_tgt = self.train_scheduler.add_noise(self.target_wav, noise, t)

        noisy_padded, orig = _pad(noisy_tgt, self._pad_factor)
        cond_padded,  _    = _pad(self.cond_wav, self._pad_factor)
        x_in = torch.cat([noisy_padded, cond_padded], dim=1)
        noise_padded, _ = _pad(noise, self._pad_factor)

        self.optimizer_cWDM.zero_grad()
        with torch.amp.autocast("cuda"):
            if self._use_ckpt:
                noise_pred_padded = grad_checkpoint(
                    self._backbone, x_in, t, use_reentrant=False
                )
            else:
                noise_pred_padded = self._backbone(x_in, timesteps=t)
            noise_pred = _unpad(noise_pred_padded, orig)
            noise_tgt  = _unpad(noise_padded, orig)
            self.loss_cwdm = self.criterion(noise_pred, noise_tgt)
        self._scaler.scale(self.loss_cwdm).backward()
        if self._grad_clip > 0:
            self._scaler.unscale_(self.optimizer_cWDM)
            torch.nn.utils.clip_grad_norm_(self._backbone.parameters(), self._grad_clip)
        self._scaler.step(self.optimizer_cWDM)
        self._scaler.update()

        if self._ema_decay > 0 and self._ema_weights is not None:
            with torch.no_grad():
                for k, v in self._backbone.state_dict().items():
                    self._ema_weights[k].mul_(self._ema_decay).add_(
                        v.float(), alpha=1.0 - self._ema_decay
                    )
