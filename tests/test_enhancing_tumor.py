"""Enhancing-tumor metrics + loss: per-region behavior and gradient flow."""
from __future__ import annotations

import pytest
import torch

from gan_data_processing.losses import EnhancingTumorLoss
from gan_data_processing.metrics import (
    et_contrast,
    et_contrast_error,
    et_edge_ssim,
    et_localization_dice,
    et_mae,
    et_metrics,
    et_pearson,
)


def _et_case(B=1, D=24, H=64, W=64, et_intensity=0.9, seed=0):
    """Volume with a deliberate enhancing region: brain ~ 0.3, ET ~ 0.9."""
    g = torch.Generator().manual_seed(seed)
    gt = 0.3 * torch.ones(B, 1, D, H, W) + 0.05 * torch.randn(B, 1, D, H, W, generator=g)
    seg = torch.zeros(B, D, H, W, dtype=torch.long)
    seg[:, D // 2 - 4 : D // 2 + 4, H // 2 - 8 : H // 2 + 8, W // 2 - 8 : W // 2 + 8] = 2
    seg[:, D // 2 - 2 : D // 2 + 2, H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 3
    et_mask = (seg == 3).unsqueeze(1).float()
    gt = gt + (et_intensity - 0.3) * et_mask
    return gt.clamp(0, 1), seg


def test_et_contrast_positive_when_brighter():
    gt, seg = _et_case(et_intensity=0.9)
    c = et_contrast(gt, seg, et_class=3)
    assert c.shape == (1, 1)
    assert c.item() > 0  # ET is brighter than brain


def test_et_contrast_zero_when_same():
    gt, seg = _et_case(et_intensity=0.3)
    c = et_contrast(gt, seg, et_class=3)
    assert abs(c.item()) < 1.0  # close to 0


def test_et_contrast_error_increases_with_misprediction():
    gt, seg = _et_case()
    pred_good = gt.clone()
    pred_bad = gt.clone()
    pred_bad[seg.unsqueeze(1) == 3] *= 0.5  # destroy enhancement
    err_good = et_contrast_error(pred_good, gt, seg).item()
    err_bad = et_contrast_error(pred_bad, gt, seg).item()
    assert err_bad > err_good


def test_et_pearson_perfect():
    gt, seg = _et_case()
    r = et_pearson(gt, gt, seg).item()
    assert r == pytest.approx(1.0, abs=1e-3)


def test_et_mae_zero_for_perfect():
    gt, seg = _et_case()
    assert et_mae(gt, gt, seg).item() == pytest.approx(0.0, abs=1e-6)


def test_et_edge_ssim_runs_3d():
    gt, seg = _et_case()
    pred = gt + 0.05 * torch.randn_like(gt)
    s = et_edge_ssim(pred, gt, seg, et_class=3, win_size=5).item()
    assert -1.0 <= s <= 1.0


def test_et_localization_dice_perfect_match():
    """If pred has exactly the GT pattern, top-percentile voxels should overlap ET."""
    gt, seg = _et_case(et_intensity=0.95)
    d = et_localization_dice(gt, gt, seg, et_class=3, percentile=99.0).item()
    assert d > 0.0  # at minimum some overlap


def test_et_metrics_dict_no_et_returns_nans():
    gt = torch.rand(1, 1, 24, 64, 64)
    seg = torch.zeros(1, 24, 64, 64, dtype=torch.long)  # no ET
    out = et_metrics(gt, gt, seg)
    assert out["et_voxels"] == 0
    import math
    assert math.isnan(out["et_pearson"])


def test_enhancing_tumor_loss_gradient_flow():
    gt, seg = _et_case()
    pred = (gt + 0.1 * torch.randn_like(gt)).requires_grad_(True)
    loss_fn = EnhancingTumorLoss(w_l1=1.0, w_contrast=1.0, w_edge=0.5)
    loss = loss_fn(pred, gt, seg)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_enhancing_tumor_loss_zero_when_no_et():
    gt = torch.rand(1, 1, 24, 64, 64)
    pred = torch.rand(1, 1, 24, 64, 64)
    seg = torch.zeros(1, 24, 64, 64, dtype=torch.long)
    loss = EnhancingTumorLoss()(pred, gt, seg)
    assert loss.item() == 0.0
