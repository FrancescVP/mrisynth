"""Pixel-wise image-quality metrics: MAE, MSE, NMSE, PSNR.

Dimensionality-agnostic — work on any tensor shape since they reduce
globally.
"""
from __future__ import annotations

import torch


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def nmse(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized MSE: ||y_pred - y_true||^2 / ||y_true||^2."""
    return mse(y_pred, y_true) / (y_true.norm() ** 2 + eps)


def psnr(
    x: torch.Tensor, y: torch.Tensor, max_value: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio. Uses global MSE → any tensor rank."""
    return 10 * torch.log10(max_value**2 / (mse(x, y) + eps))
