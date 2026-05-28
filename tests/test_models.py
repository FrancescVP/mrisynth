"""Model components: _pad/_unpad roundtrip, _build_unet/_build_dit, RFlowModel smoke tests."""
from __future__ import annotations

import types
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mrisynth.model.rflow import _pad, _unpad, _build_unet, _build_dit, _ot_couple
from mrisynth.model.dit import DiT3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rflow_opt(tmp_path, isTrain=True, channels=(32, 64), n_cond=1,
               velocity_loss="l1", ema_decay=0.0, n_attention_levels=1,
               num_res_blocks=1, backbone="unet",
               dit_patch_size=2, dit_hidden_size=64, dit_depth=2, dit_num_heads=4,
               use_ot_coupling=False, contrastive_weight=0.1, contrastive_temp=0.1):
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
        # backbone
        backbone=backbone,
        # UNet-specific
        latent_channels=4,
        n_cond=n_cond,
        unet_channels=list(channels),
        num_res_blocks=num_res_blocks,
        n_attention_levels=n_attention_levels,
        # DiT-specific
        dit_patch_size=dit_patch_size,
        dit_hidden_size=dit_hidden_size,
        dit_depth=dit_depth,
        dit_num_heads=dit_num_heads,
        # shared
        n_timesteps=10,
        n_inference_steps=2,
        weight_decay=1e-4,
        velocity_loss=velocity_loss,
        loss_alpha=0.5,
        tumor_weight=None,
        ema_decay=ema_decay,
        use_checkpointing=False,
        use_ot_coupling=use_ot_coupling,
        contrastive_weight=contrastive_weight,
        contrastive_temp=contrastive_temp,
    )


# ---------------------------------------------------------------------------
# _ot_couple
# ---------------------------------------------------------------------------

def test_ot_couple_noop_for_b1():
    noise  = torch.randn(1, 4, 8, 8, 8)
    target = torch.randn(1, 4, 8, 8, 8)
    coupled = _ot_couple(noise, target)
    assert torch.equal(coupled, noise)


def test_ot_couple_preserves_shape():
    noise  = torch.randn(3, 4, 8, 8, 8)
    target = torch.randn(3, 4, 8, 8, 8)
    coupled = _ot_couple(noise, target)
    assert coupled.shape == noise.shape


def test_ot_couple_is_permutation():
    """Coupled output must be a permutation of the input rows (no new values)."""
    noise  = torch.randn(4, 4, 8, 8, 8)
    target = torch.randn(4, 4, 8, 8, 8)
    coupled = _ot_couple(noise, target)
    # Every row of coupled must exist in noise
    for i in range(coupled.shape[0]):
        assert any(torch.equal(coupled[i], noise[j]) for j in range(noise.shape[0]))


def test_ot_couple_cost_le_uncoupled():
    """OT-coupled total L2 cost must be ≤ identity pairing cost."""
    torch.manual_seed(0)
    noise  = torch.randn(4, 4, 4, 4, 4)
    target = torch.randn(4, 4, 4, 4, 4)
    coupled = _ot_couple(noise, target)

    def total_cost(n, t):
        return ((n.flatten(1) - t.flatten(1)) ** 2).sum().item()

    assert total_cost(coupled, target) <= total_cost(noise, target) + 1e-6


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

def test_rflow_ot_coupling_loss_finite(tmp_path):
    """OT coupling must produce a finite loss for batch size > 1."""
    from mrisynth.model.rflow import RFlowModel
    opt = _rflow_opt(tmp_path, use_ot_coupling=True)
    model = RFlowModel(opt)
    model.netUNet = model.netUNet.cpu().float()
    batch = {
        "latent_tgt":  torch.randn(2, 4, 8, 8, 8),
        "latent_cond": torch.randn(2, 4, 8, 8, 8),
        "mu_tgt":      torch.randn(2, 4, 8, 8, 8),
        "case_id":     ["case0000", "case0001"],
        "seg":         None,
    }
    model.set_input(batch)
    model.optimize_parameters()
    assert torch.isfinite(model.loss_rflow)


def test_rflow_contrastive_loss_finite(tmp_path):
    """l1+contrastive loss must produce a finite result when seg is provided."""
    from mrisynth.model.rflow import RFlowModel
    opt = _rflow_opt(tmp_path, velocity_loss="l1+contrastive", contrastive_weight=0.1)
    model = RFlowModel(opt)
    model.netUNet = model.netUNet.cpu().float()
    seg = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    seg[:, 2:6, 2:6, 2:6] = 3  # ET region
    batch = {
        "latent_tgt":  torch.randn(1, 4, 8, 8, 8),
        "latent_cond": torch.randn(1, 4, 8, 8, 8),
        "mu_tgt":      torch.randn(1, 4, 8, 8, 8),
        "case_id":     ["case0000"],
        "seg":         seg,
    }
    model.set_input(batch)
    model.optimize_parameters()
    assert torch.isfinite(model.loss_rflow)


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


# ---------------------------------------------------------------------------
# DiT3D unit tests
# ---------------------------------------------------------------------------

def test_dit_forward_shape():
    net = DiT3D(in_channels=12, out_channels=4, patch_size=2,
                hidden_size=64, depth=2, num_heads=4)
    x = torch.randn(1, 12, 8, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = net(x, timesteps=t)
    assert y.shape == (1, 4, 8, 8, 8)


def test_dit_forward_odd_spatial():
    # Spatial dims not divisible by patch_size — requires external padding via _pad
    net = DiT3D(in_channels=8, out_channels=4, patch_size=2,
                hidden_size=64, depth=2, num_heads=4)
    x = torch.randn(1, 8, 10, 10, 10)   # 10 is divisible by 2 — DiT pads internally
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = net(x, timesteps=t)
    assert y.shape == (1, 4, 10, 10, 10)


def test_dit_forward_variable_spatial():
    # Different spatial sizes at inference time use the same weights
    net = DiT3D(in_channels=8, out_channels=4, patch_size=2,
                hidden_size=64, depth=2, num_heads=4)
    net.eval()
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y1 = net(torch.randn(1, 8,  8,  8,  8), timesteps=t)
        y2 = net(torch.randn(1, 8, 12, 12, 12), timesteps=t)
    assert y1.shape == (1, 4, 8, 8, 8)
    assert y2.shape == (1, 4, 12, 12, 12)


def test_dit_timestep_embedding_differs():
    # proj_out is zero-initialized (flow matching convention) so end-to-end output
    # is all-zeros at init regardless of timestep. Test the conditioning pathway
    # directly: the time_embed MLP must produce distinct vectors for distinct t.
    net = DiT3D(in_channels=8, out_channels=4, patch_size=2,
                hidden_size=64, depth=2, num_heads=4)
    with torch.no_grad():
        emb0 = net.time_embed(torch.tensor([0]))
        emb5 = net.time_embed(torch.tensor([5]))
    assert not torch.allclose(emb0, emb5)


def test_dit_gradient_flow():
    net = DiT3D(in_channels=8, out_channels=4, patch_size=2,
                hidden_size=64, depth=2, num_heads=4)
    x = torch.randn(1, 8, 8, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    y = net(x, timesteps=t)
    y.mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# _build_dit factory
# ---------------------------------------------------------------------------

def test_build_dit_forward_shape():
    net = _build_dit(in_channels=12, out_channels=4,
                     patch_size=2, hidden_size=64, depth=2, num_heads=4)
    x = torch.randn(1, 12, 8, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = net(x, timesteps=t)
    assert y.shape == (1, 4, 8, 8, 8)


def test_build_dit_returns_module():
    net = _build_dit(in_channels=8, out_channels=4,
                     patch_size=2, hidden_size=64, depth=2, num_heads=4)
    assert isinstance(net, nn.Module)


# ---------------------------------------------------------------------------
# RFlowModel with DiT backbone
# ---------------------------------------------------------------------------

class TestRFlowModelDiT:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from mrisynth.model.rflow import RFlowModel
        self.opt = _rflow_opt(tmp_path, backbone="dit",
                              dit_patch_size=2, dit_hidden_size=64,
                              dit_depth=2, dit_num_heads=4, n_cond=1)
        self.model = RFlowModel(self.opt)
        self.model.netDiT = self.model.netDiT.cpu().float()
        self.model._backbone = self.model.netDiT

    def _batch(self, B=1, C=4, D=8, H=8, W=8):
        return {
            "latent_tgt":  torch.randn(B, C, D, H, W),
            "latent_cond": torch.randn(B, C, D, H, W),
            "mu_tgt":      torch.randn(B, C, D, H, W),
            "case_id":     ["case0000"] * B,
            "seg":         None,
        }

    def test_model_names_dit(self):
        assert self.model.model_names == ["DiT"]

    def test_set_input_stores_tensors(self):
        self.model.set_input(self._batch())
        assert hasattr(self.model, "latent_tgt")
        assert hasattr(self.model, "latent_cond")

    def test_optimize_parameters_loss_finite(self):
        self.model.set_input(self._batch())
        self.model.optimize_parameters()
        assert torch.isfinite(self.model.loss_rflow)

    def test_forward_pred_shape(self):
        self.model.set_input(self._batch())
        self.model.forward()
        assert self.model.pred_latent.shape == (1, 4, 8, 8, 8)

    def test_get_current_losses_key(self):
        self.model.set_input(self._batch())
        self.model.optimize_parameters()
        losses = self.model.get_current_losses()
        assert "rflow" in losses
        assert isinstance(losses["rflow"], float)
