"""Composite training loss: blend a GAN/recon loss with a tumor-aware loss.

`TumorAwareGANLoss` takes a pre-computed scalar loss (typically your
generator's adversarial + reconstruction term) and adds a tumor-region
term weighted by `alpha` so the total is

    loss = (1 - alpha) * gan_loss + alpha * tumor_loss

When the batch contains no tumor voxels (empty seg union over the
configured classes) the tumor term is skipped and `gan_loss` is returned
unchanged — no normalization rebalancing — so training is stable on
healthy slabs.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from ..utils import seg_to_mask
from .enhancing_tumor import EnhancingTumorLoss
from .tumor_ssim import TumorSSIMLoss


class TumorAwareGANLoss(nn.Module):
    """Blend a scalar GAN/recon loss with a tumor-aware loss.

    Args:
        tumor_loss: any module with signature `(pred, gt, seg) -> scalar`.
            Defaults to `TumorSSIMLoss(tumor_classes=tumor_classes)`.
        alpha: weight for the tumor term. 0.5 = 50/50.
        tumor_classes: class ids that count as "tumor present" for the
            fallback decision. Defaults to (1, 2, 3) (whole tumor in BraTS).
    """

    def __init__(
        self,
        tumor_loss: Optional[nn.Module] = None,
        alpha: float = 0.5,
        tumor_classes: Iterable[int] = (1, 2, 3),
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.tumor_loss = tumor_loss if tumor_loss is not None else TumorSSIMLoss(
            tumor_classes=tuple(tumor_classes)
        )
        self.alpha = alpha
        self.tumor_classes = tuple(tumor_classes)

    def has_tumor(self, seg: torch.Tensor) -> bool:
        return seg_to_mask(seg, self.tumor_classes).sum().item() > 0

    def forward(
        self,
        gan_loss: torch.Tensor,
        pred: torch.Tensor,
        gt: torch.Tensor,
        seg: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, info_dict).

        `info_dict` keys: gan_loss, tumor_loss (None if skipped),
        used_tumor (bool).
        """
        info = {
            "gan_loss": gan_loss.detach().item() if gan_loss.requires_grad else float(gan_loss),
            "tumor_loss": None,
            "used_tumor": False,
        }
        if not self.has_tumor(seg):
            return gan_loss, info

        t = self.tumor_loss(pred, gt, seg)
        info["tumor_loss"] = t.detach().item()
        info["used_tumor"] = True
        total = (1.0 - self.alpha) * gan_loss + self.alpha * t
        return total, info


def make_default_et_aware_loss(alpha: float = 0.5, max_value: float = 1.0) -> TumorAwareGANLoss:
    """Convenience: 50/50 GAN-vs-EnhancingTumor loss."""
    return TumorAwareGANLoss(
        tumor_loss=EnhancingTumorLoss(max_value=max_value),
        alpha=alpha,
        tumor_classes=(3,),  # ET only for the "tumor present" gate
    )
