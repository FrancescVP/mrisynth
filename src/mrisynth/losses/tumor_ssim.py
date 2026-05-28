"""Tumor-aware GAN losses built on the SSIM family.

Starting point: SSIM evaluated only inside the tumor region. The same
machinery scales to per-class weighted SSIM losses (BraTS-PED labels:
NETC=1, SNFH=2, ET=3) so non-tumor and tumor regions can be balanced
independently.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from ..metrics.ssim import masked_ssim
from ..utils import seg_to_mask


def tumor_ssim_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    tumor_classes: Iterable[int] = (1, 2, 3),
    max_value: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """1 - masked_ssim(pred, gt) over voxels in `tumor_classes`.

    Cases without tumor voxels return a zero loss (do not penalize).
    """
    mask = seg_to_mask(seg, tumor_classes).to(pred.device)
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    s = masked_ssim(
        pred,
        gt,
        mask,
        max_value=max_value,
        win_size=win_size,
        win_sigma=win_sigma,
    )
    return 1.0 - s


def region_weighted_ssim_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    weights: dict[int, float],
    max_value: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    background_weight: float = 0.0,
) -> torch.Tensor:
    """Weighted sum of (1 - masked_ssim) per class id.

    Args:
        weights: mapping class_id -> weight, e.g. {1: 1.0, 2: 0.5, 3: 2.0}.
        background_weight: weight applied to class 0 (background). 0.0 by
            default (ignore background SSIM in this loss).
    """
    full_weights = dict(weights)
    if background_weight > 0:
        full_weights.setdefault(0, background_weight)

    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    norm = 0.0
    for cls, w in full_weights.items():
        if w == 0:
            continue
        mask = seg_to_mask(seg, [cls]).to(pred.device)
        if mask.sum() == 0:
            continue
        s = masked_ssim(
            pred,
            gt,
            mask,
            max_value=max_value,
            win_size=win_size,
            win_sigma=win_sigma,
        )
        total = total + w * (1.0 - s)
        norm += w
    return total / max(norm, 1e-8)


class TumorSSIMLoss(nn.Module):
    """`nn.Module` wrapper around `tumor_ssim_loss`."""

    def __init__(
        self,
        tumor_classes: Iterable[int] = (1, 2, 3),
        max_value: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
    ):
        super().__init__()
        self.tumor_classes = tuple(tumor_classes)
        self.max_value = max_value
        self.win_size = win_size
        self.win_sigma = win_sigma

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, seg: torch.Tensor
    ) -> torch.Tensor:
        return tumor_ssim_loss(
            pred,
            gt,
            seg,
            tumor_classes=self.tumor_classes,
            max_value=self.max_value,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
        )


class RegionWeightedSSIMLoss(nn.Module):
    """`nn.Module` wrapper around `region_weighted_ssim_loss`."""

    def __init__(
        self,
        weights: Optional[dict[int, float]] = None,
        background_weight: float = 0.0,
        max_value: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
    ):
        super().__init__()
        # BraTS-PED default: equal weight on the 3 tumor subregions.
        self.weights = weights if weights is not None else {1: 1.0, 2: 1.0, 3: 1.0}
        self.background_weight = background_weight
        self.max_value = max_value
        self.win_size = win_size
        self.win_sigma = win_sigma

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, seg: torch.Tensor
    ) -> torch.Tensor:
        return region_weighted_ssim_loss(
            pred,
            gt,
            seg,
            weights=self.weights,
            background_weight=self.background_weight,
            max_value=self.max_value,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
        )
