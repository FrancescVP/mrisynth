"""Enhancing-Tumor (ET) specific metrics for T1CE synthesis.

ET (BraTS label 3) is the contrast-enhancing region — clinically the most
important to recover. Standard pixel metrics (MAE/SSIM) on the whole volume
under-weight ET because it is small. These metrics target the properties
that make ET clinically meaningful:

1. Enhancement contrast vs surrounding brain (z-scored) — is the "this is
   enhancing" signal preserved?
2. Intensity fidelity inside ET (MAE, Pearson r).
3. Edge preservation along the ET rim (gradient SSIM in a dilated band).
4. Spatial localization: soft Dice between top-percentile pred and GT ET.

All routines accept (N, C, *spatial) tensors. Seg can be (N, *spatial) or
(N, 1, *spatial) integer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils import broadcast_to_data, seg_to_mask
from .ssim import masked_ssim


def _et_mask(seg: torch.Tensor, classes) -> torch.Tensor:
    """Bool mask of (N, *spatial) for the union of `classes`."""
    return seg_to_mask(seg, classes).bool()


def _reference_mask(seg: torch.Tensor, data: torch.Tensor, et_class: int) -> torch.Tensor:
    """Non-tumor brain reference: foreground (data!=0) AND seg==0.

    Uses the GT volume to define foreground. Returns bool (N, C, *spatial).
    """
    bg_seg = _et_mask(seg, [0])  # (N, *spatial)
    bg_seg = broadcast_to_data(bg_seg, data)
    fg = data != 0
    return bg_seg & fg


# ---------------------------------------------------------------------------
# Core ET metrics.
# ---------------------------------------------------------------------------

def et_contrast(
    x: torch.Tensor, seg: torch.Tensor, et_class: int = 3, eps: float = 1e-6
) -> torch.Tensor:
    """Per-sample enhancement contrast: (mean(x in ET) - mean(x in brain)) / std(brain).

    Z-scored against the non-tumor brain so the score is invariant to
    intensity rescaling. Returns (N, C). NaN where ET or reference empty.
    """
    et = broadcast_to_data(_et_mask(seg, [et_class]), x)
    ref = _reference_mask(seg, x, et_class)

    spatial_axes = tuple(range(2, x.ndim))
    et_n = et.float().sum(dim=spatial_axes).clamp(min=eps)
    ref_n = ref.float().sum(dim=spatial_axes).clamp(min=eps)

    et_sum = (x * et).sum(dim=spatial_axes)
    et_mean = et_sum / et_n

    ref_sum = (x * ref).sum(dim=spatial_axes)
    ref_mean = ref_sum / ref_n
    ref_var = ((x - ref_mean.view(*ref_mean.shape, *([1] * len(spatial_axes)))) ** 2 * ref).sum(
        dim=spatial_axes
    ) / ref_n
    ref_std = ref_var.clamp(min=eps).sqrt()

    out = (et_mean - ref_mean) / ref_std
    # Mark empty-region samples as NaN so the caller can skip them.
    valid = (et.float().sum(dim=spatial_axes) > 0) & (ref.float().sum(dim=spatial_axes) > 0)
    out = torch.where(valid, out, torch.full_like(out, float("nan")))
    return out


def et_contrast_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    et_class: int = 3,
    relative: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """|contrast(pred) - contrast(gt)|, optionally normalized by |contrast(gt)|.

    Returns scalar mean over valid samples/channels.
    """
    cp = et_contrast(pred, seg, et_class=et_class, eps=eps)
    cg = et_contrast(gt, seg, et_class=et_class, eps=eps)
    valid = ~(torch.isnan(cp) | torch.isnan(cg))
    if not valid.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff = (cp - cg).abs()
    if relative:
        diff = diff / cg.abs().clamp(min=eps)
    return diff[valid].mean()


def et_pearson(
    pred: torch.Tensor, gt: torch.Tensor, seg: torch.Tensor, et_class: int = 3, eps: float = 1e-6
) -> torch.Tensor:
    """Pearson correlation between pred and gt, restricted to ET voxels.

    Aggregates voxels across the whole batch (single scalar). Returns 0 when
    no ET voxels in the batch.
    """
    et = broadcast_to_data(_et_mask(seg, [et_class]), pred)
    if et.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    p = pred[et]
    g = gt[et]
    p = p - p.mean()
    g = g - g.mean()
    denom = (p.pow(2).sum().sqrt() * g.pow(2).sum().sqrt()).clamp(min=eps)
    return (p * g).sum() / denom


def et_mae(
    pred: torch.Tensor, gt: torch.Tensor, seg: torch.Tensor, et_class: int = 3
) -> torch.Tensor:
    et = broadcast_to_data(_et_mask(seg, [et_class]), pred)
    if et.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (pred - gt).abs()[et].mean()


# ---------------------------------------------------------------------------
# Edge preservation along ET rim.
# ---------------------------------------------------------------------------

def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Central-difference gradient magnitude. Works for 2D and 3D inputs."""
    grads = []
    for axis in range(2, x.ndim):
        g = torch.zeros_like(x)
        slc_p = [slice(None)] * x.ndim
        slc_n = [slice(None)] * x.ndim
        slc_o = [slice(None)] * x.ndim
        slc_p[axis] = slice(2, None)
        slc_n[axis] = slice(None, -2)
        slc_o[axis] = slice(1, -1)
        g[tuple(slc_o)] = 0.5 * (x[tuple(slc_p)] - x[tuple(slc_n)])
        grads.append(g)
    return torch.sqrt(sum(g.pow(2) for g in grads) + 1e-12)


def _dilate_mask(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Binary dilation via max-pool (3x3 / 3x3x3 kernel).

    `mask` shape (N, *spatial); returns same shape (float, 0/1).
    """
    if iterations <= 0:
        return mask.float()
    spatial_dims = mask.ndim - 1
    pool = F.max_pool2d if spatial_dims == 2 else F.max_pool3d
    m = mask.float().unsqueeze(1)
    for _ in range(iterations):
        m = pool(m, kernel_size=3, stride=1, padding=1)
    return m.squeeze(1)


def et_edge_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    et_class: int = 3,
    rim_dilation: int = 2,
    max_value: float = 1.0,
    win_size: int = 7,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """SSIM on gradient-magnitude maps inside a dilated ET rim.

    Captures whether the pred reproduces the sharpness/morphology of the
    enhancing rim. Returns scalar (mean over valid samples).
    """
    et_mask = _et_mask(seg, [et_class]).float()  # (N, *spatial)
    rim = _dilate_mask(et_mask > 0, rim_dilation)
    if rim.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    grad_p = _gradient_magnitude(pred)
    grad_g = _gradient_magnitude(gt)
    # Normalize gradient maps to roughly [0, max_value] using gt range.
    gmax = grad_g.amax().clamp(min=1e-6)
    grad_p = (grad_p / gmax).clamp(0, 1) * max_value
    grad_g = (grad_g / gmax).clamp(0, 1) * max_value

    return masked_ssim(
        grad_p,
        grad_g,
        rim,
        max_value=max_value,
        win_size=win_size,
        win_sigma=win_sigma,
    )


# ---------------------------------------------------------------------------
# Soft localization Dice (top-percentile pred vs ET).
# ---------------------------------------------------------------------------

def et_localization_dice(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    et_class: int = 3,
    percentile: float = 99.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Hard Dice between {voxels of pred above gt's pth-percentile in foreground}
    and the GT ET mask.

    Tests whether pred's brightest voxels actually land inside the true ET.
    """
    et_mask = _et_mask(seg, [et_class])  # (N, *spatial)
    et_mask = broadcast_to_data(et_mask, pred)
    fg = gt != 0
    if et_mask.sum() == 0 or fg.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    # Threshold from GT foreground intensity distribution (per batch).
    thr = torch.quantile(gt[fg], percentile / 100.0)
    pred_hot = (pred >= thr) & fg

    inter = (pred_hot & et_mask).float().sum()
    denom = pred_hot.float().sum() + et_mask.float().sum()
    return (2 * inter + eps) / (denom + eps)


# ---------------------------------------------------------------------------
# Aggregated dict for evaluation pipelines.
# ---------------------------------------------------------------------------

def et_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg: torch.Tensor,
    et_class: int = 3,
    max_value: float = 1.0,
    win_size: int = 7,
    rim_dilation: int = 2,
) -> dict[str, float]:
    """All ET-specific scalar metrics, packaged for logging."""
    et_mask = broadcast_to_data(_et_mask(seg, [et_class]), pred)
    if et_mask.sum() == 0:
        return {
            "et_contrast_pred": float("nan"),
            "et_contrast_gt": float("nan"),
            "et_contrast_abs_err": float("nan"),
            "et_contrast_rel_err": float("nan"),
            "et_pearson": float("nan"),
            "et_mae": float("nan"),
            "et_edge_ssim": float("nan"),
            "et_loc_dice": float("nan"),
            "et_voxels": 0,
        }

    cp = et_contrast(pred, seg, et_class=et_class)
    cg = et_contrast(gt, seg, et_class=et_class)
    valid = ~(torch.isnan(cp) | torch.isnan(cg))
    cp_s = cp[valid].mean() if valid.any() else torch.tensor(float("nan"))
    cg_s = cg[valid].mean() if valid.any() else torch.tensor(float("nan"))

    return {
        "et_contrast_pred": cp_s.item(),
        "et_contrast_gt": cg_s.item(),
        "et_contrast_abs_err": et_contrast_error(pred, gt, seg, et_class, relative=False).item(),
        "et_contrast_rel_err": et_contrast_error(pred, gt, seg, et_class, relative=True).item(),
        "et_pearson": et_pearson(pred, gt, seg, et_class).item(),
        "et_mae": et_mae(pred, gt, seg, et_class).item(),
        "et_edge_ssim": et_edge_ssim(
            pred, gt, seg, et_class=et_class,
            rim_dilation=rim_dilation, max_value=max_value, win_size=win_size,
        ).item(),
        "et_loc_dice": et_localization_dice(pred, gt, seg, et_class=et_class).item(),
        "et_voxels": int(et_mask.float().sum().item()),
    }
