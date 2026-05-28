"""Smoke tests for WFMModel, cWDMModel, and RFlowControlNetModel (CPU, tiny nets)."""
from __future__ import annotations

import types
import tempfile
from pathlib import Path

import pytest
import torch

from mrisynth.model.wfm import WFMModel
from mrisynth.model.cwdm import cWDMModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wfm_opt(tmp_path, n_cond=2, channels=(16, 32), n_attention_levels=1,
             num_res_blocks=1, velocity_loss="l1", ema_decay=0.0,
             use_ot_coupling=False):
    return types.SimpleNamespace(
        isTrain=True,
        checkpoints_dir=str(tmp_path),
        name="test_wfm",
        device=torch.device("cpu"),
        verbose=False,
        init_type="normal",
        init_gain=0.02,
        continue_train=False,
        epoch="latest",
        load_iter=0,
        norm="instance",
        lr_policy="linear",
        n_epochs=10,
        n_epochs_decay=5,
        epoch_count=1,
        lr_decay_iters=5,
        lr=1e-4,
        beta1=0.5,
        backbone="unet",
        n_cond=n_cond,
        unet_channels=list(channels),
        num_res_blocks=num_res_blocks,
        n_attention_levels=n_attention_levels,
        dit_patch_size=2,
        dit_hidden_size=64,
        dit_depth=2,
        dit_num_heads=4,
        n_timesteps=10,
        n_inference_steps=2,
        weight_decay=1e-4,
        velocity_loss=velocity_loss,
        loss_alpha=0.5,
        ema_decay=ema_decay,
        use_checkpointing=False,
        use_ot_coupling=use_ot_coupling,
        contrastive_weight=0.1,
        contrastive_temp=0.1,
        inference_noise=0.0,
    )


def _cwdm_opt(tmp_path, n_cond=2, channels=(16, 32), n_attention_levels=1,
              num_res_blocks=1, ema_decay=0.0):
    return types.SimpleNamespace(
        isTrain=True,
        checkpoints_dir=str(tmp_path),
        name="test_cwdm",
        device=torch.device("cpu"),
        verbose=False,
        init_type="normal",
        init_gain=0.02,
        continue_train=False,
        epoch="latest",
        load_iter=0,
        norm="instance",
        lr_policy="linear",
        n_epochs=10,
        n_epochs_decay=5,
        epoch_count=1,
        lr_decay_iters=5,
        lr=1e-4,
        beta1=0.5,
        backbone="unet",
        n_cond=n_cond,
        unet_channels=list(channels),
        num_res_blocks=num_res_blocks,
        n_attention_levels=n_attention_levels,
        dit_patch_size=2,
        dit_hidden_size=64,
        dit_depth=2,
        dit_num_heads=4,
        n_timesteps=10,
        n_inference_steps=2,
        n_ddim_steps=2,
        beta_schedule="linear",
        weight_decay=1e-4,
        ema_decay=ema_decay,
        use_checkpointing=False,
        use_ot_coupling=False,
    )


def _batch(B=1, n_cond=2, D=8, H=8, W=8):
    """Pix2Pix3dDataset-style batch with even dims."""
    return {
        "A":       torch.randn(B, n_cond, D, H, W),
        "B":       torch.randn(B, 1, D, H, W),
        "seg":     torch.zeros(B, 1, D, H, W, dtype=torch.int16),
        "A_paths": [["case0000"]] * B,
    }


# ---------------------------------------------------------------------------
# WFM tests
# ---------------------------------------------------------------------------

class TestWFMModel:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        opt = _wfm_opt(tmp_path)
        self.model = WFMModel(opt)
        self.model.netWFM = self.model.netWFM.cpu().float()

    def test_wfm_set_input_applies_dwt(self):
        self.model.set_input(_batch())
        assert hasattr(self.model, "target_wav")
        assert hasattr(self.model, "cond_wav")
        # After DWT: target_wav should have 8 channels, cond_wav 8*n_cond
        assert self.model.target_wav.shape[1] == 8
        assert self.model.cond_wav.shape[1] == 16  # 8 * 2

    def test_wfm_optimize_parameters_finite(self):
        self.model.set_input(_batch())
        self.model.optimize_parameters()
        assert torch.isfinite(self.model.loss_wfm)

    def test_wfm_forward_pred_image_shape(self):
        self.model.set_input(_batch())
        self.model.forward()
        assert hasattr(self.model, "pred_image")
        assert self.model.pred_image.shape == (1, 1, 8, 8, 8)

    def test_wfm_informed_prior_uses_cond(self):
        # With non-zero conditioning, prior should differ from zeros
        batch = _batch()
        batch["A"] = torch.ones_like(batch["A"])  # all-ones conditioning
        self.model.set_input(batch)
        prior = self.model._build_prior()
        assert not torch.allclose(prior, torch.zeros_like(prior))

    def test_wfm_loss_names(self):
        assert "wfm" in self.model.loss_names

    def test_wfm_model_names(self):
        assert "WFM" in self.model.model_names

    def test_wfm_get_current_losses(self):
        self.model.set_input(_batch())
        self.model.optimize_parameters()
        losses = self.model.get_current_losses()
        assert "wfm" in losses
        assert isinstance(losses["wfm"], float)


# ---------------------------------------------------------------------------
# cWDM tests
# ---------------------------------------------------------------------------

class TestcWDMModel:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        opt = _cwdm_opt(tmp_path)
        self.model = cWDMModel(opt)
        self.model.netcWDM = self.model.netcWDM.cpu().float()

    def test_cwdm_set_input_applies_dwt(self):
        self.model.set_input(_batch())
        assert hasattr(self.model, "target_wav")
        assert hasattr(self.model, "cond_wav")
        assert self.model.target_wav.shape[1] == 8
        assert self.model.cond_wav.shape[1] == 16

    def test_cwdm_optimize_parameters_finite(self):
        self.model.set_input(_batch())
        self.model.optimize_parameters()
        assert torch.isfinite(self.model.loss_cwdm)

    def test_cwdm_forward_pred_image_shape(self):
        self.model.set_input(_batch())
        self.model.forward()
        assert hasattr(self.model, "pred_image")
        assert self.model.pred_image.shape == (1, 1, 8, 8, 8)

    def test_cwdm_loss_names(self):
        assert "cwdm" in self.model.loss_names

    def test_cwdm_model_names(self):
        assert "cWDM" in self.model.model_names

    def test_cwdm_get_current_losses(self):
        self.model.set_input(_batch())
        self.model.optimize_parameters()
        losses = self.model.get_current_losses()
        assert "cwdm" in losses
        assert isinstance(losses["cwdm"], float)


# ---------------------------------------------------------------------------
# RFlowControlNet tests
# ---------------------------------------------------------------------------

def _controlnet_opt(tmp_path, n_cond=1, channels=(16, 32), n_attention_levels=1,
                    num_res_blocks=1):
    return types.SimpleNamespace(
        isTrain=True,
        checkpoints_dir=str(tmp_path),
        name="test_controlnet",
        device=torch.device("cpu"),
        verbose=False,
        init_type="normal",
        init_gain=0.02,
        continue_train=False,
        epoch="latest",
        load_iter=0,
        norm="instance",
        lr_policy="linear",
        n_epochs=10,
        n_epochs_decay=5,
        epoch_count=1,
        lr_decay_iters=5,
        lr=1e-4,
        beta1=0.5,
        backbone="unet",
        latent_channels=4,
        n_cond=n_cond,
        unet_channels=list(channels),
        num_res_blocks=num_res_blocks,
        n_attention_levels=n_attention_levels,
        n_timesteps=10,
        n_inference_steps=2,
        weight_decay=1e-4,
        velocity_loss="l1",
        loss_alpha=0.5,
        tumor_weight=None,
        ema_decay=0.0,
        use_checkpointing=False,
        use_ot_coupling=False,
        contrastive_weight=0.1,
        contrastive_temp=0.1,
    )


def _latent_batch(B=1, lat_ch=4, D=8, H=8, W=8, n_cond=1, with_seg=True):
    seg = torch.zeros(B, D, H, W, dtype=torch.long) if with_seg else None
    return {
        "latent_tgt":  torch.randn(B, lat_ch, D, H, W),
        "latent_cond": torch.randn(B, lat_ch * n_cond, D, H, W),
        "mu_tgt":      torch.randn(B, lat_ch, D, H, W),
        "case_id":     ["case0000"] * B,
        "seg":         seg,
    }


def test_rflow_controlnet_optimize_finite(tmp_path):
    """With seg provided, ControlNet should produce a finite loss."""
    from mrisynth.model.rflow_controlnet import RFlowControlNetModel
    opt = _controlnet_opt(tmp_path)
    model = RFlowControlNetModel(opt)
    model.netUNet = model.netUNet.cpu().float()
    model.netControlNet = model.netControlNet.cpu().float()

    batch = _latent_batch(with_seg=True)
    model.set_input(batch)
    model.optimize_parameters()
    assert torch.isfinite(model.loss_rflow)


def test_rflow_controlnet_no_seg_fallback(tmp_path):
    """With seg=None, should fall back to plain RFlow and still produce finite loss."""
    from mrisynth.model.rflow_controlnet import RFlowControlNetModel
    opt = _controlnet_opt(tmp_path)
    model = RFlowControlNetModel(opt)
    model.netUNet = model.netUNet.cpu().float()
    model.netControlNet = model.netControlNet.cpu().float()

    batch = _latent_batch(with_seg=False)
    model.set_input(batch)
    model.optimize_parameters()
    assert torch.isfinite(model.loss_rflow)


def test_rflow_controlnet_model_names(tmp_path):
    from mrisynth.model.rflow_controlnet import RFlowControlNetModel
    opt = _controlnet_opt(tmp_path)
    model = RFlowControlNetModel(opt)
    assert "UNet" in model.model_names
    assert "ControlNet" in model.model_names
