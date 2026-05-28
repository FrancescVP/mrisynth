"""3D single-level Haar wavelet transform and inverse (pure PyTorch)."""
from __future__ import annotations

import torch
import torch.nn as nn


def haar_dwt3d(x: torch.Tensor) -> torch.Tensor:
    """Single-level 3D Haar DWT.
    Input:  (B, C, D, H, W) — spatial dims must be even
    Output: (B, 8*C, D//2, H//2, W//2)
    Subband order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    Convention: lo=(a+b)*0.5, hi=(a-b)*0.5
    """
    D, H, W = x.shape[-3:]
    if D % 2 != 0 or H % 2 != 0 or W % 2 != 0:
        raise ValueError(
            f"Spatial dims must be even for Haar DWT, got ({D}, {H}, {W})"
        )

    # Apply 1D Haar along D (dim 2)
    a, b = x[..., 0::2, :, :], x[..., 1::2, :, :]
    xL = (a + b) * 0.5
    xH = (a - b) * 0.5

    # Apply along H (dim 3)
    a, b = xL[..., 0::2, :], xL[..., 1::2, :]
    xLL = (a + b) * 0.5
    xLH = (a - b) * 0.5

    a, b = xH[..., 0::2, :], xH[..., 1::2, :]
    xHL = (a + b) * 0.5
    xHH = (a - b) * 0.5

    # Apply along W (dim 4)
    def _split_w(t):
        a, b = t[..., 0::2], t[..., 1::2]
        return (a + b) * 0.5, (a - b) * 0.5

    LLL, LLH = _split_w(xLL)
    LHL, LHH = _split_w(xLH)
    HLL, HLH = _split_w(xHL)
    HHL, HHH = _split_w(xHH)

    return torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)


def haar_idwt3d(x: torch.Tensor) -> torch.Tensor:
    """Single-level 3D Haar IDWT.
    Input:  (B, 8*C, D//2, H//2, W//2)
    Output: (B, C, D, H, W)
    """
    assert x.shape[1] % 8 == 0, f"Channel dim must be divisible by 8, got {x.shape[1]}"
    C = x.shape[1] // 8
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = x.split(C, dim=1)

    # Invert W: a = lo + hi, b = lo - hi
    def _merge_w(lo, hi):
        B, C, D, H, W2 = lo.shape
        out = lo.new_empty(B, C, D, H, W2 * 2)
        out[..., 0::2] = lo + hi
        out[..., 1::2] = lo - hi
        return out

    xLL = _merge_w(LLL, LLH)
    xLH = _merge_w(LHL, LHH)
    xHL = _merge_w(HLL, HLH)
    xHH = _merge_w(HHL, HHH)

    # Invert H
    def _merge_h(lo, hi):
        B, C, D, H2, W = lo.shape
        out = lo.new_empty(B, C, D, H2 * 2, W)
        out[..., 0::2, :] = lo + hi
        out[..., 1::2, :] = lo - hi
        return out

    xL = _merge_h(xLL, xLH)
    xH = _merge_h(xHL, xHH)

    # Invert D
    B, C, D2, H, W = xL.shape
    out = xL.new_empty(B, C, D2 * 2, H, W)
    out[..., 0::2, :, :] = xL + xH
    out[..., 1::2, :, :] = xL - xH
    return out


class HaarDWT3D(nn.Module):
    """nn.Module wrapper for haar_dwt3d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return haar_dwt3d(x)


class HaarIDWT3D(nn.Module):
    """nn.Module wrapper for haar_idwt3d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return haar_idwt3d(x)
