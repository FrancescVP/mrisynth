"""Tests for 3D Haar wavelet transform and inverse."""
from __future__ import annotations

import pytest
import torch

from mrisynth.model.wavelet import HaarDWT3D, HaarIDWT3D, haar_dwt3d, haar_idwt3d


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_haar_dwt3d_output_shape():
    x = torch.randn(2, 3, 8, 8, 8)
    y = haar_dwt3d(x)
    assert y.shape == (2, 24, 4, 4, 4)


def test_haar_dwt3d_multichannel_shape():
    x = torch.randn(1, 4, 16, 12, 8)
    y = haar_dwt3d(x)
    assert y.shape == (1, 32, 8, 6, 4)


def test_haar_dwt3d_odd_spatial_raises():
    x = torch.randn(1, 1, 7, 8, 8)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        haar_dwt3d(x)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

def test_haar_idwt3d_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(2, 3, 8, 8, 8)
    assert torch.allclose(haar_idwt3d(haar_dwt3d(x)), x, atol=1e-5)


def test_haar_idwt3d_roundtrip_larger():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16, 16)
    assert torch.allclose(haar_idwt3d(haar_dwt3d(x)), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_haar_dwt3d_gradient_flows():
    x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    y = haar_dwt3d(x)
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_haar_idwt3d_gradient_flows():
    x = torch.randn(1, 16, 4, 4, 4, requires_grad=True)
    y = haar_idwt3d(x)
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

def test_dwt_module_forward():
    mod = HaarDWT3D()
    x = torch.randn(1, 2, 8, 8, 8)
    y = mod(x)
    assert y.shape == (1, 16, 4, 4, 4)


def test_idwt_module_forward():
    mod = HaarIDWT3D()
    x = torch.randn(1, 16, 4, 4, 4)
    y = mod(x)
    assert y.shape == (1, 2, 8, 8, 8)


def test_dwt_idwt_modules_roundtrip():
    dwt  = HaarDWT3D()
    idwt = HaarIDWT3D()
    x = torch.randn(1, 1, 8, 8, 8)
    assert torch.allclose(idwt(dwt(x)), x, atol=1e-5)
