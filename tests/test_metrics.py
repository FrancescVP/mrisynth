"""Sanity tests: 2D reproduces canonical values, 3D works, gradient flows."""
from __future__ import annotations

import pytest
import torch

from mrisynth.metrics import mae, mse, nmse, psnr, ssim


def test_perfect_match_2d():
    x = torch.rand(2, 1, 64, 64)
    assert mae(x, x).item() == 0.0
    assert mse(x, x).item() == 0.0
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)


def test_perfect_match_3d():
    x = torch.rand(2, 1, 32, 64, 64)
    assert mae(x, x).item() == 0.0
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)


def test_psnr_decreases_with_noise():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    y_low = x + 0.01 * torch.randn_like(x)
    y_high = x + 0.5 * torch.randn_like(x)
    assert psnr(x, y_low).item() > psnr(x, y_high).item()


def test_ssim_decreases_with_noise_3d():
    torch.manual_seed(0)
    x = torch.rand(1, 4, 32, 64, 64)
    y_low = x + 0.01 * torch.randn_like(x)
    y_high = x + 0.5 * torch.randn_like(x)
    assert ssim(x, y_low).item() > ssim(x, y_high).item()


def test_ssim_no_size_average_3d():
    x = torch.rand(3, 1, 16, 32, 32)
    y = torch.rand(3, 1, 16, 32, 32)
    out = ssim(x, y, size_average=False)
    assert out.shape == (3,)


def test_multichannel_3d_matches_glioma_layout():
    """4 channels × 3D volume — mimics processed BraTS-PED case."""
    x = torch.rand(2, 4, 32, 64, 64)
    y = x + 0.1 * torch.randn_like(x)
    s = ssim(x, y)
    assert 0.0 < s.item() < 1.0


def test_gradient_flows():
    x = torch.rand(1, 1, 16, 32, 32, requires_grad=True)
    y = torch.rand(1, 1, 16, 32, 32)
    loss = 1.0 - ssim(x, y)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_nmse():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    y = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    assert nmse(x, y).item() == pytest.approx(0.0)
