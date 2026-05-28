"""Shared helpers used by both metrics/ and losses/.

Kept tiny and dependency-light to avoid cycles between subpackages.
"""
from __future__ import annotations

from typing import Iterable

import torch


def seg_to_mask(seg: torch.Tensor, classes: Iterable[int]) -> torch.Tensor:
    """Build a binary float mask for the union of `classes` from an integer seg.

    Args:
        seg: integer tensor of shape (N, *spatial) or (N, 1, *spatial).
        classes: class ids to include.

    Returns:
        Float (0/1) mask of shape (N, *spatial).
    """
    if seg.ndim >= 2 and seg.shape[1] == 1:
        seg = seg.squeeze(1)
    classes = list(classes)
    if not classes:
        return torch.zeros_like(seg, dtype=torch.float32)
    mask = torch.zeros_like(seg, dtype=torch.bool)
    for c in classes:
        mask |= seg == c
    return mask.to(dtype=torch.float32)


def broadcast_to_data(mask_nd: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """(N, *spatial) -> (N, C, *spatial), expanded to match `data`."""
    return mask_nd.unsqueeze(1).expand_as(data)
