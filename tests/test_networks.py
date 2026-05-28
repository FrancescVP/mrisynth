"""3-D network primitives: norms, GANLoss, generators, discriminators, schedulers."""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from gan_data_processing.model.networks import (
    GANLoss,
    NLayerDiscriminator3D,
    UnetGenerator3D,
    define_D_3d,
    define_G_3d,
    get_norm_layer_3d,
    get_scheduler,
)


# ---------------------------------------------------------------------------
# get_norm_layer_3d
# ---------------------------------------------------------------------------

def test_norm_instance():
    layer = get_norm_layer_3d("instance")(32)
    assert isinstance(layer, nn.InstanceNorm3d)


def test_norm_batch():
    layer = get_norm_layer_3d("batch")(32)
    assert isinstance(layer, nn.BatchNorm3d)


def test_norm_none():
    layer = get_norm_layer_3d("none")(32)
    assert isinstance(layer, nn.Module)


def test_norm_unknown_raises():
    with pytest.raises(NotImplementedError):
        get_norm_layer_3d("unknown_norm")


# ---------------------------------------------------------------------------
# GANLoss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["lsgan", "vanilla", "wgangp"])
def test_ganloss_real_finite(mode):
    loss_fn = GANLoss(mode)
    pred = torch.randn(2, 1, 4, 4, 4)
    loss = loss_fn(pred, target_is_real=True)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mode", ["lsgan", "vanilla", "wgangp"])
def test_ganloss_fake_finite(mode):
    loss_fn = GANLoss(mode)
    pred = torch.randn(2, 1, 4, 4, 4)
    loss = loss_fn(pred, target_is_real=False)
    assert torch.isfinite(loss)


def test_ganloss_lsgan_perfect_real_near_zero():
    loss_fn = GANLoss("lsgan", target_real_label=1.0)
    pred = torch.ones(1, 1, 4, 4, 4)
    assert loss_fn(pred, target_is_real=True).item() == pytest.approx(0.0, abs=1e-6)


def test_ganloss_wgangp_sign():
    loss_fn = GANLoss("wgangp")
    pred = torch.ones(1, 1, 4, 4, 4)
    real_loss = loss_fn(pred, target_is_real=True)
    fake_loss = loss_fn(pred, target_is_real=False)
    assert real_loss < 0 and fake_loss > 0


def test_ganloss_unknown_mode_raises():
    with pytest.raises(NotImplementedError):
        GANLoss("unknown_mode")


# ---------------------------------------------------------------------------
# define_G_3d / UnetGenerator3D
# ---------------------------------------------------------------------------

def test_define_G_output_shape():
    G = define_G_3d(input_nc=2, output_nc=1, ngf=8, num_downs=5)
    x = torch.randn(1, 2, 32, 32, 32)
    with torch.no_grad():
        y = G(x)
    assert y.shape == (1, 1, 32, 32, 32)


def test_define_G_multichannel_input():
    G = define_G_3d(input_nc=3, output_nc=1, ngf=8, num_downs=5)
    x = torch.randn(1, 3, 32, 32, 32)
    with torch.no_grad():
        y = G(x)
    assert y.shape == (1, 1, 32, 32, 32)


def test_define_G_gradient_flows():
    G = define_G_3d(input_nc=1, output_nc=1, ngf=8, num_downs=5)
    x = torch.randn(1, 1, 32, 32, 32)
    loss = G(x).sum()
    loss.backward()
    grads = [p.grad for p in G.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_define_G_odd_spatial():
    """Generator clips output to the nearest multiple of 2^num_downs for odd inputs."""
    G = define_G_3d(input_nc=1, output_nc=1, ngf=8, num_downs=5)
    x = torch.randn(1, 1, 33, 33, 33)
    with torch.no_grad():
        y = G(x)
    # 33 clipped to 32 (nearest multiple of 2^5=32) — expected, not a bug
    assert y.shape == (1, 1, 32, 32, 32)


# ---------------------------------------------------------------------------
# define_D_3d / NLayerDiscriminator3D
# ---------------------------------------------------------------------------

def test_define_D_output_is_tensor():
    D = define_D_3d(input_nc=2, ndf=8, n_layers_D=2)
    x = torch.randn(1, 2, 32, 32, 32)
    with torch.no_grad():
        out = D(x)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 5


def test_define_D_return_features():
    D = define_D_3d(input_nc=2, ndf=8, n_layers_D=2)
    x = torch.randn(1, 2, 32, 32, 32)
    with torch.no_grad():
        feats, out = D(x, return_features=True)
    assert isinstance(feats, list)
    assert len(feats) > 0


def test_define_D_gradient_flows():
    D = define_D_3d(input_nc=1, ndf=8, n_layers_D=2)
    x = torch.randn(1, 1, 32, 32, 32)
    out = D(x)
    out.sum().backward()
    grads = [p.grad for p in D.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# get_scheduler
# ---------------------------------------------------------------------------

def _mock_opt(lr_policy, n_epochs=100, n_epochs_decay=50,
              epoch_count=1, lr_decay_iters=50):
    opt = types.SimpleNamespace(
        lr_policy=lr_policy,
        n_epochs=n_epochs,
        n_epochs_decay=n_epochs_decay,
        epoch_count=epoch_count,
        lr_decay_iters=lr_decay_iters,
    )
    return opt


@pytest.mark.parametrize("policy", ["linear", "step", "cosine"])
def test_scheduler_steps_without_error(policy):
    net = nn.Linear(4, 4)
    optimizer = Adam(net.parameters(), lr=1e-3)
    scheduler = get_scheduler(optimizer, _mock_opt(policy))
    for _ in range(3):
        optimizer.step()
        scheduler.step()


def test_scheduler_unknown_policy_raises():
    net = nn.Linear(4, 4)
    optimizer = Adam(net.parameters(), lr=1e-3)
    with pytest.raises(NotImplementedError):
        get_scheduler(optimizer, _mock_opt("unknown_policy"))


def test_scheduler_linear_decreases_lr():
    net = nn.Linear(4, 4)
    optimizer = Adam(net.parameters(), lr=1e-3)
    scheduler = get_scheduler(optimizer, _mock_opt("linear", n_epochs=5, n_epochs_decay=5, epoch_count=1))
    initial_lr = optimizer.param_groups[0]["lr"]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr <= initial_lr
