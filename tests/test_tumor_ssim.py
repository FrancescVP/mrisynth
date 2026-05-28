"""Tumor-aware loss tests: shape handling, gradient flow, sanity bounds."""
from __future__ import annotations

import pytest
import torch

from gan_data_processing.losses import (
    RegionWeightedSSIMLoss,
    TumorSSIMLoss,
    region_weighted_ssim_loss,
    seg_to_mask,
    tumor_ssim_loss,
)
from gan_data_processing.metrics import masked_ssim, ssim


def _fake_case(B=2, C=1, D=24, H=64, W=64, with_tumor=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(B, C, D, H, W, generator=g)
    pred = gt.clone()
    seg = torch.zeros(B, D, H, W, dtype=torch.int64)
    if with_tumor:
        seg[:, D // 2 - 4 : D // 2 + 4, H // 2 - 8 : H // 2 + 8, W // 2 - 8 : W // 2 + 8] = 1
        seg[:, D // 2 - 2 : D // 2 + 2, H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 3
    return pred, gt, seg


def test_seg_to_mask_basic():
    seg = torch.tensor([[[0, 1], [2, 3]]])
    m = seg_to_mask(seg, [1, 3])
    assert m.dtype == torch.float32
    assert torch.equal(m, torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))


def test_perfect_pred_gives_zero_loss():
    pred, gt, seg = _fake_case()
    loss = tumor_ssim_loss(pred, gt, seg)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_no_tumor_returns_zero():
    pred, gt, seg = _fake_case(with_tumor=False)
    loss = tumor_ssim_loss(pred, gt, seg)
    assert loss.item() == 0.0


def test_loss_increases_with_perturbation_inside_tumor():
    torch.manual_seed(1)
    pred, gt, seg = _fake_case()
    perturbed = gt.clone()
    tumor_mask = (seg > 0).unsqueeze(1).float()
    perturbed = perturbed + 0.5 * tumor_mask * torch.randn_like(perturbed)
    loss_clean = tumor_ssim_loss(gt, gt, seg)
    loss_dirty = tumor_ssim_loss(perturbed, gt, seg)
    assert loss_dirty > loss_clean


def test_gradient_flows():
    pred, gt, seg = _fake_case()
    pred = pred.clone().requires_grad_(True)
    perturbed = pred + 0.1 * torch.randn_like(pred)
    loss = TumorSSIMLoss()(perturbed, gt, seg)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_region_weighted_loss_module():
    pred, gt, seg = _fake_case()
    pred = pred + 0.1 * torch.randn_like(pred)
    loss_fn = RegionWeightedSSIMLoss(weights={1: 1.0, 3: 2.0})
    out = loss_fn(pred, gt, seg)
    assert out.ndim == 0
    assert out.item() >= 0.0


def test_region_weighted_matches_single_when_one_class():
    """Single-class region-weighted SSIM should equal direct masked SSIM (loss form)."""
    torch.manual_seed(2)
    pred, gt, seg = _fake_case()
    pred = pred + 0.05 * torch.randn_like(pred)
    direct = 1.0 - masked_ssim(pred, gt, seg_to_mask(seg, [1]))
    weighted = region_weighted_ssim_loss(pred, gt, seg, weights={1: 1.0})
    assert weighted.item() == pytest.approx(direct.item(), rel=1e-5)


def test_2d_input_supported():
    """Loss works on 2D batches (N,C,H,W) too."""
    torch.manual_seed(3)
    gt = torch.rand(2, 1, 64, 64)
    pred = gt + 0.05 * torch.randn_like(gt)
    seg = torch.zeros(2, 64, 64, dtype=torch.int64)
    seg[:, 20:40, 20:40] = 1
    loss = tumor_ssim_loss(pred, gt, seg)
    assert loss.item() > 0


def test_masked_ssim_aligns_with_full_ssim_when_mask_full():
    """Mask = all ones -> masked SSIM == full SSIM (after replicate padding)."""
    torch.manual_seed(4)
    x = torch.rand(1, 1, 16, 32, 32)
    y = x + 0.05 * torch.randn_like(x)
    mask = torch.ones(1, 16, 32, 32)
    s_masked = masked_ssim(x, y, mask)
    s_full = ssim(x, y)
    # Different padding regimes -> values close but not identical.
    assert abs(s_masked.item() - s_full.item()) < 0.05
