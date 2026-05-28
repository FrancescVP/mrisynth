"""Velocity-space losses for RFlow training.

Supports L1, L2, SSIM (1-SSIM), NCC, and L1+SSIM blends.
All losses operate on arbitrary tensors (latent velocity fields).

NCC is the most robust to cross-scanner intensity drift:
it is invariant to any linear transform a*x + b applied to both
the target and the prediction, which matches the global intensity
shifts seen across MRI machines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.ssim import ssim as _ssim


# ---------------------------------------------------------------------------
# NCC
# ---------------------------------------------------------------------------

class NCCLoss(nn.Module):
    """Global Normalized Cross-Correlation loss.

    Invariant to linear intensity scaling (a*x + b).
    Returns 1 - NCC, so 0 is perfect.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten all but batch dim → (B, N)
        p = pred.flatten(1)
        t = target.flatten(1)
        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)
        ncc = (p * t).sum(dim=1) / (
            p.norm(dim=1) * t.norm(dim=1) + self.eps
        )
        return 1.0 - ncc.mean()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """1 - SSIM loss.

    Treats each latent channel as a 3-D volume and averages over channels.
    Uses per-batch data range so it adapts to the magnitude of the velocity.
    """

    def __init__(self, win_size: int = 7):
        super().__init__()
        self.win_size = win_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dr = float((target.max() - target.min()).clamp(min=1e-6))
        # ssim expects (N, C, ...) — shape is already (B, 4, D, H, W)
        return 1.0 - _ssim(pred, target, max_value=dr, win_size=self.win_size)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class L1SSIMLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM).

    Balances absolute accuracy with structural quality.
    """

    def __init__(self, alpha: float = 0.5, win_size: int = 7):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self._l1   = nn.L1Loss()
        self._ssim = SSIMLoss(win_size=win_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self._l1(pred, target) + (1.0 - self.alpha) * self._ssim(pred, target)


# ---------------------------------------------------------------------------
# Tumor-weighted velocity losses
# ---------------------------------------------------------------------------

def _seg_to_latent_mask(
    seg: torch.Tensor,
    target_shape: tuple,
    tumor_class: int | None,
) -> torch.Tensor:
    """Downsample (D,H,W) seg to latent spatial dims and return a float mask.

    Args:
        seg: (D, H, W) int tensor at image resolution.
        target_shape: (D', H', W') latent spatial dims.
        tumor_class: specific class to mask (e.g. 3=ET), or None for whole tumour (>0).
    """
    seg_5d = seg.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    seg_down = F.interpolate(seg_5d, size=target_shape, mode="nearest").squeeze()  # (D', H', W')
    if tumor_class is None:
        return (seg_down > 0).float()
    return (seg_down == tumor_class).float()


class TumorWeightedL1Loss(nn.Module):
    """L1 velocity loss with higher weight inside the tumour region.

    Operates in latent space: seg is downscaled to match latent resolution.
    Falls back to plain L1 when seg is None (no seg available).

    Args:
        tumor_class: BraTS class to upweight (1=NETC, 2=SNFH, 3=ET, None=whole tumour).
        tumor_weight: multiplier applied inside tumour voxels (default 5.0).
    """

    def __init__(self, tumor_class: int | None = None, tumor_weight: float = 5.0):
        super().__init__()
        self.tumor_class = tumor_class
        self.tumor_weight = tumor_weight
        self._l1 = nn.L1Loss(reduction="none")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        err = self._l1(pred, target)  # (B, C, D', H', W')
        if seg is None:
            return err.mean()

        mask = _seg_to_latent_mask(seg[0], pred.shape[2:], self.tumor_class)
        # (D', H', W') → (1, 1, D', H', W')
        mask = mask.to(pred.device).view(1, 1, *mask.shape)
        weight = 1.0 + (self.tumor_weight - 1.0) * mask
        return (err * weight).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_velocity_loss(name: str, **kwargs) -> nn.Module:
    """Return a velocity loss by name.

    Supported names
    ---------------
    l1          nn.L1Loss (default, current behaviour)
    l2          nn.MSELoss
    ssim        1 - SSIM
    ncc         1 - NCC  (invariant to linear intensity shifts)
    l1+ssim     alpha * L1 + (1-alpha) * (1-SSIM); pass alpha=<float>
    """
    name = name.lower().strip()
    if name == "l1":
        return nn.L1Loss()
    if name == "l2":
        return nn.MSELoss()
    if name == "ssim":
        return SSIMLoss(win_size=kwargs.get("win_size", 7))
    if name == "ncc":
        return NCCLoss()
    if name in ("l1+ssim", "l1_ssim"):
        return L1SSIMLoss(
            alpha=kwargs.get("alpha", 0.5),
            win_size=kwargs.get("win_size", 7),
        )
    if name == "tumor_l1":
        return TumorWeightedL1Loss(tumor_class=None, tumor_weight=kwargs.get("tumor_weight", 5.0))
    if name == "et_l1":
        return TumorWeightedL1Loss(tumor_class=3, tumor_weight=kwargs.get("tumor_weight", 10.0))
    raise ValueError(
        f"Unknown velocity loss '{name}'. "
        "Choose: l1, l2, ssim, ncc, l1+ssim, tumor_l1, et_l1"
    )
