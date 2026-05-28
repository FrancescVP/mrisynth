"""Model components: _pad/_unpad roundtrip, _build_unet, RFlowModel smoke test."""
from __future__ import annotations

import types
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mrisynth.model.rflow import _pad, _unpad, _build_unet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rflow_opt(tmp_path, isTrain=True, channels=(32, 64), n_cond=1,
               velocity_loss="l1", ema_decay=0.0, n_attention_levels=1,
               num_res_blocks=1):
    return types.SimpleNamespace(
        isTrain=isTrain,
        checkpoints_dir=str(tmp_path),
        name="test_rflow",
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
        # RFlow-specific
        latent_channels=4,
        n_cond=n_cond,
        unet_channels=list(channels),
        n_timesteps=10,
        n_inference_steps=2,
        weight_decay=1e-4,
        velocity_loss=velocity_loss,
        loss_alpha=0.5,
        tumor_weight=None,
        num_res_blocks=num_res_blocks,
        n_attention_levels=n_attention_levels,
        ema_decay=ema_decay,
        use_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# _pad / _unpad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 4, 8, 8, 8),
    (1, 4, 7, 9, 11),  # odd dims
    (2, 4, 16, 16, 16),
])
def test_pad_unpad_roundtrip(shape):
    x = torch.randn(*shape)
    padded, orig = _pad(x, factor=4)
    restored = _unpad(padded, orig)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


def test_pad_makes_divisible():
    x = torch.randn(1, 4, 7, 9, 11)
    padded, _ = _pad(x, factor=4)
    D, H, W = padded.shape[-3:]
    assert D % 4 == 0 and H % 4 == 0 and W % 4 == 0


def test_pad_already_aligned_no_change():
    x = torch.randn(1, 4, 8, 8, 8)
    padded, orig = _pad(x, factor=4)
    assert padded.shape == x.shape
    assert orig == (8, 8, 8)


# ---------------------------------------------------------------------------
# _build_unet
# ---------------------------------------------------------------------------

def test_build_unet_forward_shape():
    net = _build_unet(
        in_channels=12, out_channels=4,
        num_channels=[32, 64],
        n_attention_levels=1, num_res_blocks=1,
    )
    x = torch.randn(1, 12, 8, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = net(x, timesteps=t)
    assert y.shape == (1, 4, 8, 8, 8)


def test_build_unet_attention_levels():
    net = _build_unet(
        in_channels=8, out_channels=4,
        num_channels=[32, 64, 64],
        n_attention_levels=2, num_res_blocks=1,
    )
    assert isinstance(net, nn.Module)


def test_build_unet_single_attention_level():
    # n_attention_levels=1 means only the deepest level has attention
    net = _build_unet(
        in_channels=8, out_channels=4,
        num_channels=[32, 64],
        n_attention_levels=1, num_res_blocks=1,
    )
    x = torch.randn(1, 8, 8, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = net(x, timesteps=t)
    assert y.shape == (1, 4, 8, 8, 8)


# ---------------------------------------------------------------------------
# RFlowModel smoke tests (CPU, tiny UNet, no VAE)
# ---------------------------------------------------------------------------

class TestRFlowModel:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from mrisynth.model.rflow import RFlowModel
        self.opt = _rflow_opt(tmp_path)
        self.model = RFlowModel(self.opt)
        self.model.netUNet = self.model.netUNet.cpu().float()

    def _batch(self, B=1, C=4, D=8, H=8, W=8):
        return {
            "latent_tgt":  torch.randn(B, C, D, H, W),
            "latent_cond": torch.randn(B, C, D, H, W),  # n_cond=1 → 4ch
            "mu_tgt":      torch.randn(B, C, D, H, W),
            "case_id":     ["case0000"] * B,
            "seg":         None,
        }

    def test_set_input_stores_tensors(self):
        batch = self._batch()
        self.model.set_input(batch)
        assert hasattr(self.model, "latent_tgt")
        assert hasattr(self.model, "latent_cond")

    def test_optimize_parameters_loss_finite(self):
        self.model.set_input(self._batch())
        self.model.optimize_parameters()
        assert torch.isfinite(self.model.loss_rflow)

    def test_model_names(self):
        assert "UNet" in self.model.model_names

    def test_get_current_losses_key(self):
        self.model.set_input(self._batch())
        self.model.optimize_parameters()
        losses = self.model.get_current_losses()
        assert "rflow" in losses
        assert isinstance(losses["rflow"], float)


# ---------------------------------------------------------------------------
# RFlowModel with EMA enabled
# ---------------------------------------------------------------------------

def test_rflow_ema_weights_initialized(tmp_path):
    from mrisynth.model.rflow import RFlowModel
    opt = _rflow_opt(tmp_path, ema_decay=0.999)
    model = RFlowModel(opt)
    assert model._ema_weights is not None
    assert len(model._ema_weights) > 0


def test_rflow_ema_updates_after_step(tmp_path):
    from mrisynth.model.rflow import RFlowModel
    opt = _rflow_opt(tmp_path, ema_decay=0.9)
    model = RFlowModel(opt)
    model.netUNet = model.netUNet.cpu().float()

    first_key = next(iter(model._ema_weights))
    initial = model._ema_weights[first_key].clone()

    batch = {
        "latent_tgt":  torch.randn(1, 4, 8, 8, 8),
        "latent_cond": torch.randn(1, 4, 8, 8, 8),
        "mu_tgt":      torch.randn(1, 4, 8, 8, 8),
        "case_id":     ["case0000"],
        "seg":         None,
    }
    model.set_input(batch)
    model.optimize_parameters()

    assert not torch.equal(model._ema_weights[first_key], initial)
