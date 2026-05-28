"""Foreground (nonzero) cropping. Mirrors nnUNet's crop_to_nonzero."""
from __future__ import annotations

from typing import Optional

import numpy as np


def get_nonzero_bbox(data: np.ndarray) -> list[tuple[int, int]]:
    """Bounding box of nonzero voxels across channels.

    Args:
        data: (C, X, Y, Z) array.

    Returns:
        list of (start, end_exclusive) per spatial axis.
    """
    assert data.ndim == 4, "expect (C,X,Y,Z)"
    nonzero = data != 0
    fg = nonzero.any(axis=0)
    bbox = []
    for axis in range(fg.ndim):
        reduce_axes = tuple(a for a in range(fg.ndim) if a != axis)
        proj = fg.any(axis=reduce_axes)
        idx = np.where(proj)[0]
        if idx.size == 0:
            bbox.append((0, fg.shape[axis]))
        else:
            bbox.append((int(idx[0]), int(idx[-1]) + 1))
    return bbox


def crop_to_nonzero(
    data: np.ndarray, seg: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Optional[np.ndarray], list[tuple[int, int]]]:
    """Crop data and seg to nonzero bbox of data.

    Returns cropped data, cropped seg (or None), bbox.
    """
    bbox = get_nonzero_bbox(data)
    sl = (slice(None),) + tuple(slice(s, e) for s, e in bbox)
    cropped = data[sl]
    cropped_seg = None
    if seg is not None:
        cropped_seg = seg[tuple(slice(s, e) for s, e in bbox)]
    return cropped, cropped_seg, bbox
