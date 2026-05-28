"""Composite GAN+tumor loss: 50/50 blend, fallback to GAN-only when no tumor."""
from __future__ import annotations

import pytest
import torch

from gan_data_processing.losses import (
    EnhancingTumorLoss,
    TumorAwareGANLoss,
    TumorSSIMLoss,
    make_default_et_aware_loss,
)


def _case(B=1, C=1, D=24, H=64, W=64, with_tumor=True, et_only=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(B, C, D, H, W, generator=g)
    pred = gt + 0.05 * torch.randn(B, C, D, H, W, generator=g)
    seg = torch.zeros(B, D, H, W, dtype=torch.int64)
    if with_tumor:
        seg[:, D // 2 - 4 : D // 2 + 4, H // 2 - 8 : H // 2 + 8, W // 2 - 8 : W // 2 + 8] = 2
        if not et_only:
            seg[:, D // 2 - 2 : D // 2 + 2, H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 3
        else:
            seg[:, D // 2 - 2 : D // 2 + 2, H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 3
    return pred, gt, seg


def test_blend_50_50():
    pred, gt, seg = _case()
    pred = pred.requires_grad_(True)
    gan_loss = torch.tensor(0.4, requires_grad=False)

    loss_fn = TumorAwareGANLoss(tumor_loss=TumorSSIMLoss(), alpha=0.5)
    total, info = loss_fn(gan_loss, pred, gt, seg)

    assert info["used_tumor"] is True
    assert info["tumor_loss"] is not None
    expected = 0.5 * gan_loss.item() + 0.5 * info["tumor_loss"]
    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_no_tumor_returns_gan_only():
    pred, gt, seg = _case(with_tumor=False)
    gan_loss = torch.tensor(0.3, requires_grad=False)
    loss_fn = TumorAwareGANLoss(alpha=0.5)
    total, info = loss_fn(gan_loss, pred, gt, seg)
    assert info["used_tumor"] is False
    assert info["tumor_loss"] is None
    assert total.item() == pytest.approx(gan_loss.item())


def test_alpha_zero_returns_gan_only():
    pred, gt, seg = _case()
    gan_loss = torch.tensor(0.7, requires_grad=False)
    loss_fn = TumorAwareGANLoss(alpha=0.0)
    total, info = loss_fn(gan_loss, pred, gt, seg)
    assert total.item() == pytest.approx(gan_loss.item(), rel=1e-5)


def test_alpha_one_returns_tumor_only():
    pred, gt, seg = _case()
    pred = pred.requires_grad_(True)
    gan_loss = torch.tensor(0.7)
    loss_fn = TumorAwareGANLoss(alpha=1.0)
    total, info = loss_fn(gan_loss, pred, gt, seg)
    assert info["tumor_loss"] is not None
    assert total.item() == pytest.approx(info["tumor_loss"], rel=1e-5)


def test_gradient_flows_through_blend():
    pred, gt, seg = _case()
    pred = pred.clone().requires_grad_(True)
    # gan_loss can also be a real graph node — simulate that.
    gan_loss = (pred - gt).pow(2).mean()
    loss_fn = TumorAwareGANLoss(alpha=0.5)
    total, _ = loss_fn(gan_loss, pred, gt, seg)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_alpha_validation():
    with pytest.raises(ValueError):
        TumorAwareGANLoss(alpha=1.5)
    with pytest.raises(ValueError):
        TumorAwareGANLoss(alpha=-0.1)


def test_default_et_aware_helper():
    """Default helper uses ET-only gate + EnhancingTumorLoss."""
    pred, gt, seg = _case(et_only=True)
    pred = pred.requires_grad_(True)
    gan_loss = torch.tensor(0.5)
    loss_fn = make_default_et_aware_loss(alpha=0.5)
    total, info = loss_fn(gan_loss, pred, gt, seg)
    assert info["used_tumor"] is True
    assert isinstance(loss_fn.tumor_loss, EnhancingTumorLoss)


def test_default_et_aware_no_et_returns_gan_only():
    """seg has only label 2 (SNFH) -> no ET -> GAN only."""
    pred, gt, seg = _case()
    seg[seg == 3] = 2  # remove ET
    gan_loss = torch.tensor(0.5)
    loss_fn = make_default_et_aware_loss(alpha=0.5)
    total, info = loss_fn(gan_loss, pred, gt, seg)
    assert info["used_tumor"] is False
    assert total.item() == pytest.approx(gan_loss.item())
