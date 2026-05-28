"""Normalization. Default: z-score within nonzero (brain MRI scheme)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def zscore_nonzero(
    data: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-8
) -> np.ndarray:
    """Z-score per channel using nonzero (or supplied) mask voxels.

    Voxels outside the mask are set to 0 (background stays 0). Mirrors
    nnUNet's `ZScoreNormalization` with `use_nonzero_mask=True`.

    Args:
        data: (C, X, Y, Z) float32.
        mask: optional (X, Y, Z) bool mask. If None, foreground = any channel != 0.
    """
    assert data.ndim == 4
    out = np.zeros_like(data, dtype=np.float32)
    if mask is None:
        mask = (data != 0).any(axis=0)
    if not mask.any():
        return out
    for c in range(data.shape[0]):
        vals = data[c][mask]
        mean = float(vals.mean())
        std = float(vals.std())
        out[c][mask] = (data[c][mask] - mean) / max(std, eps)
    return out


def zscore_global(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score per channel over whole volume."""
    out = np.empty_like(data, dtype=np.float32)
    for c in range(data.shape[0]):
        mean = float(data[c].mean())
        std = float(data[c].std())
        out[c] = (data[c] - mean) / max(std, eps)
    return out


def percentile_clip_zscore(
    data: np.ndarray, mask: Optional[np.ndarray] = None, low: float = 0.5, high: float = 99.5
) -> np.ndarray:
    """Clip to [low, high] percentiles within mask, then z-score (CT-like scheme)."""
    assert data.ndim == 4
    out = np.zeros_like(data, dtype=np.float32)
    if mask is None:
        mask = (data != 0).any(axis=0)
    for c in range(data.shape[0]):
        vals = data[c][mask] if mask.any() else data[c].ravel()
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [low, high])
        clipped = np.clip(data[c], lo, hi)
        kept = clipped[mask] if mask.any() else clipped
        mean = float(kept.mean())
        std = float(kept.std())
        out[c] = ((clipped - mean) / max(std, 1e-8)).astype(np.float32)
        if mask.any():
            out[c][~mask] = 0.0
    return out
