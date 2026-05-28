"""Velocity-space losses used by RFlow: behavior, invariances, gradient flow."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mrisynth.losses import (
    SegAwareLoss,
    L1SSIMLoss,
    NCCLoss,
    SSIMLoss,
    TumorWeightedL1Loss,
    RegionContrastiveLoss,
    L1RegionContrastiveLoss,
    build_velocity_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vel(B=1, C=4, D=16, H=16, W=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred   = torch.rand(B, C, D, H, W, generator=g)
    target = torch.rand(B, C, D, H, W, generator=g)
    return pred, target


def _seg(B=1, D=16, H=16, W=16):
    seg = torch.zeros(B, D, H, W, dtype=torch.int64)
    seg[:, 4:12, 4:12, 4:12] = 1  # whole-tumour region
    return seg


# ---------------------------------------------------------------------------
# NCCLoss
# ---------------------------------------------------------------------------

def test_ncc_range():
    pred, target = _vel()
    loss = NCCLoss()(pred, target)
    assert 0.0 <= loss.item() <= 2.0


def test_ncc_perfect_match_near_zero():
    pred, _ = _vel()
    loss = NCCLoss()(pred, pred)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_ncc_linear_invariance():
    """NCC is invariant to a * x + b: loss(2*pred+3, pred) should be ~0."""
    pred, _ = _vel()
    shifted = 2.0 * pred + 3.0
    loss = NCCLoss()(shifted, pred)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_ncc_gradient_flows():
    pred, target = _vel()
    pred = pred.requires_grad_(True)
    NCCLoss()(pred, target).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# SSIMLoss
# ---------------------------------------------------------------------------

def test_ssim_loss_perfect_match():
    pred, _ = _vel()
    assert SSIMLoss(win_size=5)(pred, pred).item() == pytest.approx(0.0, abs=1e-4)


def test_ssim_loss_range():
    pred, target = _vel()
    loss = SSIMLoss(win_size=5)(pred, target)
    assert 0.0 <= loss.item() <= 2.0


def test_ssim_loss_increases_with_noise():
    _, target = _vel()
    low_noise  = target + 0.01 * torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
    high_noise = target + 1.0  * torch.randn(target.shape, generator=torch.Generator().manual_seed(2))
    assert SSIMLoss(win_size=5)(low_noise, target) < SSIMLoss(win_size=5)(high_noise, target)


def test_ssim_loss_gradient_flows():
    pred, target = _vel()
    pred = pred.requires_grad_(True)
    SSIMLoss(win_size=5)(pred, target).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# L1SSIMLoss
# ---------------------------------------------------------------------------

def test_l1ssim_alpha_one_equals_l1():
    pred, target = _vel()
    expected = nn.L1Loss()(pred, target).item()
    actual   = L1SSIMLoss(alpha=1.0, win_size=5)(pred, target).item()
    assert actual == pytest.approx(expected, rel=1e-5)


def test_l1ssim_alpha_zero_equals_ssim():
    pred, target = _vel()
    expected = SSIMLoss(win_size=5)(pred, target).item()
    actual   = L1SSIMLoss(alpha=0.0, win_size=5)(pred, target).item()
    assert actual == pytest.approx(expected, rel=1e-5)


def test_l1ssim_blend_between_extremes():
    pred, target = _vel()
    l1   = nn.L1Loss()(pred, target).item()
    ssim = SSIMLoss(win_size=5)(pred, target).item()
    blend = L1SSIMLoss(alpha=0.5, win_size=5)(pred, target).item()
    # alpha=0.5 → exactly 0.5*L1 + 0.5*SSIM
    assert blend == pytest.approx(0.5 * l1 + 0.5 * ssim, rel=1e-5)


def test_l1ssim_invalid_alpha():
    with pytest.raises(ValueError):
        L1SSIMLoss(alpha=1.5)
    with pytest.raises(ValueError):
        L1SSIMLoss(alpha=-0.1)


# ---------------------------------------------------------------------------
# TumorWeightedL1Loss
# ---------------------------------------------------------------------------

def test_tumor_weighted_no_seg_equals_l1():
    pred, target = _vel()
    expected = nn.L1Loss()(pred, target).item()
    actual   = TumorWeightedL1Loss()(pred, target, seg=None).item()
    assert actual == pytest.approx(expected, rel=1e-5)


def test_tumor_weighted_perfect_match_is_zero():
    pred, _ = _vel()
    seg = _seg()
    assert TumorWeightedL1Loss()(pred, pred, seg).item() == pytest.approx(0.0, abs=1e-6)


def test_tumor_weighted_higher_loss_inside_tumor():
    """Error concentrated inside tumor → weighted loss > plain L1."""
    pred, target = _vel()
    # Put all error inside the tumor mask region
    tumor_pred = target.clone()
    seg = _seg()
    mask = seg.bool().unsqueeze(1).expand_as(tumor_pred)
    tumor_pred[mask] += 1.0  # large error only inside tumor

    plain_l1 = nn.L1Loss()(tumor_pred, target).item()
    weighted = TumorWeightedL1Loss(tumor_class=None, tumor_weight=5.0)(tumor_pred, target, seg).item()
    assert weighted > plain_l1


def test_tumor_weighted_gradient_flows():
    pred, target = _vel()
    pred = pred.requires_grad_(True)
    seg = _seg()
    TumorWeightedL1Loss()(pred, target, seg).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# SegAwareLoss base class
# ---------------------------------------------------------------------------

def test_tumor_weighted_is_seg_aware():
    assert isinstance(TumorWeightedL1Loss(), SegAwareLoss)


def test_region_contrastive_is_seg_aware():
    assert isinstance(RegionContrastiveLoss(), SegAwareLoss)


def test_l1_region_contrastive_is_seg_aware():
    assert isinstance(L1RegionContrastiveLoss(), SegAwareLoss)


# ---------------------------------------------------------------------------
# RegionContrastiveLoss
# ---------------------------------------------------------------------------

def _et_seg(B=1, D=16, H=16, W=16):
    seg = torch.zeros(B, D, H, W, dtype=torch.int64)
    seg[:, 4:12, 4:12, 4:12] = 3   # ET region
    return seg


def test_region_contrastive_no_seg_returns_zero():
    pred, target = _vel()
    loss = RegionContrastiveLoss()(pred, target, seg=None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_region_contrastive_correct_vs_wrong():
    """Correct ET prediction has lower loss than an inverted (wrong) ET prediction.

    We need ET and BG to have clearly different velocity directions — random
    uniform data gives near-identical pooled vectors, making the test ambiguous.
    """
    C, D, H, W = 4, 16, 16, 16
    seg = _et_seg()

    # Target: ET region has opposite direction to BG
    target = torch.ones(1, C, D, H, W) * -1.0   # BG: all -1
    target[:, :, 4:12, 4:12, 4:12] = 1.0        # ET: all +1

    pred_correct = target.clone()                # ET prediction matches ET target
    pred_wrong   = target.clone()
    pred_wrong[:, :, 4:12, 4:12, 4:12] = -1.0  # ET prediction matches BG (wrong)

    loss_correct = RegionContrastiveLoss()(pred_correct, target, seg)
    loss_wrong   = RegionContrastiveLoss()(pred_wrong,   target, seg)
    assert loss_correct.item() < loss_wrong.item()


def test_region_contrastive_finite():
    pred, target = _vel()
    seg = _et_seg()
    loss = RegionContrastiveLoss()(pred, target, seg)
    assert torch.isfinite(loss)


def test_region_contrastive_gradient_flows():
    pred, target = _vel()
    seg = _et_seg()
    pred = pred.requires_grad_(True)
    RegionContrastiveLoss()(pred, target, seg).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_region_contrastive_empty_et_returns_zero():
    """When the ET mask is empty the loss must be zero (no valid region)."""
    pred, target = _vel()
    seg = torch.zeros(1, 16, 16, 16, dtype=torch.int64)  # no ET
    loss = RegionContrastiveLoss()(pred, target, seg)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_l1_contrastive_no_seg_equals_l1():
    pred, target = _vel()
    expected = nn.L1Loss()(pred, target).item()
    actual = L1RegionContrastiveLoss()(pred, target, seg=None).item()
    assert actual == pytest.approx(expected, rel=1e-5)


def test_l1_contrastive_with_seg_larger_than_l1():
    """When there is an ET region the contrastive term adds a positive cost."""
    pred, target = _vel()
    seg = _et_seg()
    l1_only = nn.L1Loss()(pred, target).item()
    combined = L1RegionContrastiveLoss(contrastive_weight=1.0)(pred, target, seg).item()
    assert combined >= l1_only


def test_l1_contrastive_gradient_flows():
    pred, target = _vel()
    seg = _et_seg()
    pred = pred.requires_grad_(True)
    L1RegionContrastiveLoss(contrastive_weight=0.5)(pred, target, seg).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# build_velocity_loss factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["l1", "l2", "ssim", "ncc", "l1+ssim", "tumor_l1", "et_l1", "l1+contrastive"])
def test_factory_returns_module(name):
    loss_fn = build_velocity_loss(name)
    assert isinstance(loss_fn, nn.Module)


def test_factory_l1_is_l1loss():
    assert isinstance(build_velocity_loss("l1"), nn.L1Loss)


def test_factory_l2_is_mseloss():
    assert isinstance(build_velocity_loss("l2"), nn.MSELoss)


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown velocity loss"):
        build_velocity_loss("garbage_loss_name")


def test_factory_l1_correct_result():
    pred, target = _vel()
    expected = nn.L1Loss()(pred, target).item()
    actual   = build_velocity_loss("l1")(pred, target).item()
    assert actual == pytest.approx(expected)
