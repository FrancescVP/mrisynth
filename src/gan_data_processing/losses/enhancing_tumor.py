"""Differentiable Enhancing-Tumor loss for T1CE synthesis.

Combines three terms with configurable weights:
  - L1 inside ET voxels (intensity fidelity)
  - |contrast(pred) - contrast(gt)| (preserves the enhancement signal,
    scale-normalized to non-tumor brain)
  - 1 - SSIM on gradient magnitude inside the dilated ET rim
    (preserves rim morphology / sharpness)

When the ET region is empty in a batch the loss returns 0 so the loss can
be safely added to a recon term without spurious gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..metrics.enhancing_tumor import et_contrast_error, et_edge_ssim, et_mae
from ..utils import seg_to_mask


def enhancing_tumor_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    et_class: int = 3,
    w_l1: float = 1.0,
    w_contrast: float = 1.0,
    w_edge: float = 0.5,
    max_value: float = 1.0,
    win_size: int = 7,
    rim_dilation: int = 2,
) -> torch.Tensor:
    """Weighted ET loss. Returns 0 when no ET voxels in the batch."""
    et_mask = seg_to_mask(seg, [et_class])
    if et_mask.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    norm = max(w_l1 + w_contrast + w_edge, 1e-8)
    parts = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if w_l1 > 0:
        parts = parts + w_l1 * et_mae(pred, gt, seg, et_class=et_class)
    if w_contrast > 0:
        parts = parts + w_contrast * et_contrast_error(
            pred, gt, seg, et_class=et_class, relative=False
        )
    if w_edge > 0:
        edge = et_edge_ssim(
            pred, gt, seg,
            et_class=et_class,
            rim_dilation=rim_dilation,
            max_value=max_value,
            win_size=win_size,
        )
        parts = parts + w_edge * (1.0 - edge)
    return parts / norm


class EnhancingTumorLoss(nn.Module):
    """`nn.Module` wrapper around `enhancing_tumor_loss`."""

    def __init__(
        self,
        et_class: int = 3,
        w_l1: float = 1.0,
        w_contrast: float = 1.0,
        w_edge: float = 0.5,
        max_value: float = 1.0,
        win_size: int = 7,
        rim_dilation: int = 2,
    ):
        super().__init__()
        self.et_class = et_class
        self.w_l1 = w_l1
        self.w_contrast = w_contrast
        self.w_edge = w_edge
        self.max_value = max_value
        self.win_size = win_size
        self.rim_dilation = rim_dilation

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, seg: torch.Tensor
    ) -> torch.Tensor:
        return enhancing_tumor_loss(
            pred, gt, seg,
            et_class=self.et_class,
            w_l1=self.w_l1,
            w_contrast=self.w_contrast,
            w_edge=self.w_edge,
            max_value=self.max_value,
            win_size=self.win_size,
            rim_dilation=self.rim_dilation,
        )
