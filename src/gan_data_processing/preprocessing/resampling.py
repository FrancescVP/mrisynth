"""Resampling (mirrors nnUNet semantics: cubic spline for data, NN for seg).

For anisotropic spacings nnUNet does separate-z resampling. BraTS data is
isotropic 1mm so default 3D zoom is sufficient; the separate-z path is
still implemented for general use.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import map_coordinates


ANISOTROPY_THRESHOLD = 3.0


def compute_new_shape(
    old_shape: Sequence[int],
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
) -> tuple[int, ...]:
    return tuple(
        int(round(o * s_old / s_new))
        for o, s_old, s_new in zip(old_shape, old_spacing, new_spacing)
    )


def _is_anisotropic(spacing: Sequence[float]) -> tuple[bool, Optional[int]]:
    spacing = np.asarray(spacing, dtype=float)
    lo_axis = int(spacing.argmax())
    if spacing[lo_axis] / spacing.min() > ANISOTROPY_THRESHOLD:
        return True, lo_axis
    return False, None


def _zoom_3d(volume: np.ndarray, new_shape: Sequence[int], order: int) -> np.ndarray:
    """3D resample via map_coordinates (handles non-integer factors)."""
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(volume.shape) == new_shape:
        return volume.copy()
    coords = np.meshgrid(
        *[np.linspace(0, s - 1, ns) for s, ns in zip(volume.shape, new_shape)],
        indexing="ij",
    )
    coords = np.stack(coords, axis=0)
    return map_coordinates(volume, coords, order=order, mode="nearest").astype(
        volume.dtype, copy=False
    )


def _resample_separate_z(
    volume: np.ndarray, new_shape: Sequence[int], lo_axis: int, order: int, order_z: int
) -> np.ndarray:
    """Resample in-plane with `order`, along low-res axis with `order_z`."""
    new_shape = tuple(int(s) for s in new_shape)
    if volume.shape[lo_axis] == new_shape[lo_axis]:
        in_plane_shape = list(volume.shape)
        in_plane_shape[lo_axis] = new_shape[lo_axis]
        target = list(new_shape)
        # Slice-wise resample of in-plane axes.
        out = np.empty(target, dtype=volume.dtype)
        for i in range(volume.shape[lo_axis]):
            sl = [slice(None)] * 3
            sl[lo_axis] = i
            slc = volume[tuple(sl)]
            new_slice_shape = [new_shape[a] for a in range(3) if a != lo_axis]
            out_sl = [slice(None)] * 3
            out_sl[lo_axis] = i
            out[tuple(out_sl)] = _zoom_3d(
                slc[None], (1, *new_slice_shape), order
            )[0] if False else _zoom_2d(slc, new_slice_shape, order)
        return out

    # Two stages: in-plane with `order`, then along z with `order_z`.
    intermediate_shape = list(new_shape)
    intermediate_shape[lo_axis] = volume.shape[lo_axis]
    stage1 = np.empty(intermediate_shape, dtype=volume.dtype)
    for i in range(volume.shape[lo_axis]):
        sl = [slice(None)] * 3
        sl[lo_axis] = i
        slc = volume[tuple(sl)]
        new_slice_shape = [new_shape[a] for a in range(3) if a != lo_axis]
        out_sl = [slice(None)] * 3
        out_sl[lo_axis] = i
        stage1[tuple(out_sl)] = _zoom_2d(slc, new_slice_shape, order)
    return _zoom_3d(stage1, new_shape, order_z)


def _zoom_2d(slc: np.ndarray, new_shape: Sequence[int], order: int) -> np.ndarray:
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(slc.shape) == new_shape:
        return slc.copy()
    coords = np.meshgrid(
        *[np.linspace(0, s - 1, ns) for s, ns in zip(slc.shape, new_shape)],
        indexing="ij",
    )
    coords = np.stack(coords, axis=0)
    return map_coordinates(slc, coords, order=order, mode="nearest").astype(
        slc.dtype, copy=False
    )


def resample_data(
    data: np.ndarray,
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
    order: int = 3,
    order_z_when_anisotropic: int = 0,
) -> np.ndarray:
    """Resample (C, X, Y, Z) image data."""
    assert data.ndim == 4
    new_shape = compute_new_shape(data.shape[1:], old_spacing, new_spacing)
    aniso, lo_axis = _is_anisotropic(old_spacing)
    out = np.empty((data.shape[0], *new_shape), dtype=np.float32)
    for c in range(data.shape[0]):
        vol = data[c].astype(np.float32, copy=False)
        if aniso:
            out[c] = _resample_separate_z(
                vol, new_shape, lo_axis, order, order_z_when_anisotropic
            )
        else:
            out[c] = _zoom_3d(vol, new_shape, order)
    return out


def resample_seg(
    seg: np.ndarray,
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
    order: int = 1,
    order_z_when_anisotropic: int = 0,
) -> np.ndarray:
    """Resample (X, Y, Z) integer seg via per-class one-hot + linear (nnUNet style).

    For order=0 falls back to nearest-neighbor on labels directly.
    """
    new_shape = compute_new_shape(seg.shape, old_spacing, new_spacing)
    aniso, lo_axis = _is_anisotropic(old_spacing)
    if order == 0:
        if aniso:
            return _resample_separate_z(
                seg.astype(np.float32), new_shape, lo_axis, 0, 0
            ).astype(seg.dtype)
        return _zoom_3d(seg.astype(np.float32), new_shape, 0).astype(seg.dtype)

    classes = np.unique(seg)
    out_shape = tuple(int(s) for s in new_shape)
    accum = np.zeros((len(classes), *out_shape), dtype=np.float32)
    for i, cls in enumerate(classes):
        mask = (seg == cls).astype(np.float32)
        if aniso:
            accum[i] = _resample_separate_z(
                mask, new_shape, lo_axis, order, order_z_when_anisotropic
            )
        else:
            accum[i] = _zoom_3d(mask, new_shape, order)
    chosen = accum.argmax(axis=0).astype(np.int64)
    out = np.zeros(out_shape, dtype=seg.dtype)
    for i, cls in enumerate(classes):
        out[chosen == i] = cls
    return out
