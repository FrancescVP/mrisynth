"""Preprocessing utilities: normalization, cropping, resampling."""
from __future__ import annotations

import numpy as np
import pytest

from mrisynth.preprocessing.normalization import (
    zscore_nonzero,
    zscore_global,
    percentile_clip_zscore,
)
from mrisynth.preprocessing.cropping import (
    get_nonzero_bbox,
    crop_to_nonzero,
)
from mrisynth.preprocessing.resampling import (
    compute_new_shape,
    resample_data,
    resample_seg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(C=2, X=20, Y=20, Z=20, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((C, X, Y, Z)).astype(np.float32)
    d[:, :5, :, :] = 0.0  # background band
    return d


def _seg(X=20, Y=20, Z=20):
    s = np.zeros((X, Y, Z), dtype=np.int16)
    s[5:15, 5:15, 5:15] = 1
    s[8:12, 8:12, 8:12] = 3
    return s


# ---------------------------------------------------------------------------
# zscore_nonzero
# ---------------------------------------------------------------------------

def test_zscore_nonzero_background_stays_zero():
    data = _data()
    out = zscore_nonzero(data)
    bg = (data == 0).all(axis=0)
    assert (out[:, bg] == 0.0).all()


def test_zscore_nonzero_mean_near_zero():
    data = _data()
    out = zscore_nonzero(data)
    mask = (data != 0).any(axis=0)
    for c in range(data.shape[0]):
        vals = out[c][mask]
        assert abs(float(vals.mean())) < 0.05


def test_zscore_nonzero_std_near_one():
    data = _data()
    out = zscore_nonzero(data)
    mask = (data != 0).any(axis=0)
    for c in range(data.shape[0]):
        vals = out[c][mask]
        assert abs(float(vals.std()) - 1.0) < 0.05


def test_zscore_nonzero_all_zero_input():
    data = np.zeros((2, 8, 8, 8), dtype=np.float32)
    out = zscore_nonzero(data)
    assert (out == 0.0).all()


def test_zscore_nonzero_custom_mask():
    data = _data()
    mask = np.ones(data.shape[1:], dtype=bool)
    out = zscore_nonzero(data, mask=mask)
    assert out.shape == data.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# zscore_global
# ---------------------------------------------------------------------------

def test_zscore_global_output_shape():
    data = _data()
    out = zscore_global(data)
    assert out.shape == data.shape


def test_zscore_global_mean_near_zero():
    data = _data()
    out = zscore_global(data)
    for c in range(data.shape[0]):
        assert abs(float(out[c].mean())) < 0.05


def test_zscore_global_std_near_one():
    data = _data()
    out = zscore_global(data)
    for c in range(data.shape[0]):
        assert abs(float(out[c].std()) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# percentile_clip_zscore
# ---------------------------------------------------------------------------

def test_percentile_clip_zscore_finite():
    data = _data()
    out = percentile_clip_zscore(data)
    assert np.isfinite(out).all()


def test_percentile_clip_zscore_shape():
    data = _data()
    out = percentile_clip_zscore(data)
    assert out.shape == data.shape


def test_percentile_clip_zscore_background_zero():
    data = _data()
    out = percentile_clip_zscore(data)
    bg = (data == 0).all(axis=0)
    assert (out[:, bg] == 0.0).all()


# ---------------------------------------------------------------------------
# get_nonzero_bbox
# ---------------------------------------------------------------------------

def test_get_nonzero_bbox_basic():
    data = np.zeros((2, 20, 20, 20), dtype=np.float32)
    data[:, 3:15, 4:16, 5:17] = 1.0
    bbox = get_nonzero_bbox(data)
    assert len(bbox) == 3
    assert bbox[0] == (3, 15)
    assert bbox[1] == (4, 16)
    assert bbox[2] == (5, 17)


def test_get_nonzero_bbox_all_zero():
    data = np.zeros((1, 10, 10, 10), dtype=np.float32)
    bbox = get_nonzero_bbox(data)
    assert bbox[0] == (0, 10)
    assert bbox[1] == (0, 10)
    assert bbox[2] == (0, 10)


def test_get_nonzero_bbox_single_channel():
    data = np.zeros((1, 16, 16, 16), dtype=np.float32)
    data[0, 6:10, 6:10, 6:10] = 1.0
    bbox = get_nonzero_bbox(data)
    assert bbox[0] == (6, 10)


# ---------------------------------------------------------------------------
# crop_to_nonzero
# ---------------------------------------------------------------------------

def test_crop_to_nonzero_reduces_shape():
    data = np.zeros((2, 20, 20, 20), dtype=np.float32)
    data[:, 5:15, 5:15, 5:15] = 1.0
    cropped, cropped_seg, bbox = crop_to_nonzero(data)
    assert cropped.shape[1] < data.shape[1]
    assert cropped_seg is None


def test_crop_to_nonzero_seg_aligned():
    data = np.zeros((2, 20, 20, 20), dtype=np.float32)
    data[:, 5:15, 5:15, 5:15] = 1.0
    seg = _seg()
    cropped, cropped_seg, bbox = crop_to_nonzero(data, seg)
    assert cropped.shape[1:] == cropped_seg.shape


def test_crop_to_nonzero_bbox_correct():
    data = np.zeros((1, 20, 20, 20), dtype=np.float32)
    data[:, 4:14, 3:13, 2:12] = 1.0
    _, _, bbox = crop_to_nonzero(data)
    assert bbox[0] == (4, 14)
    assert bbox[1] == (3, 13)
    assert bbox[2] == (2, 12)


# ---------------------------------------------------------------------------
# compute_new_shape
# ---------------------------------------------------------------------------

def test_compute_new_shape_halved():
    new = compute_new_shape((100, 80, 60), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    assert new == (50, 40, 30)


def test_compute_new_shape_identity():
    new = compute_new_shape((64, 64, 64), (1.5, 1.5, 1.5), (1.5, 1.5, 1.5))
    assert new == (64, 64, 64)


def test_compute_new_shape_anisotropic():
    new = compute_new_shape((100, 100, 50), (1.0, 1.0, 2.0), (1.0, 1.0, 1.0))
    assert new == (100, 100, 100)


# ---------------------------------------------------------------------------
# resample_data
# ---------------------------------------------------------------------------

def test_resample_data_output_shape():
    data = np.ones((2, 32, 32, 32), dtype=np.float32)
    out = resample_data(data, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    assert out.shape == (2, 64, 64, 64)


def test_resample_data_identity():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 16, 16, 16)).astype(np.float32)
    out = resample_data(data, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    np.testing.assert_allclose(out, data, atol=1e-5)


def test_resample_data_finite():
    data = _data(C=1, X=16, Y=16, Z=16)
    out = resample_data(data, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# resample_seg
# ---------------------------------------------------------------------------

def test_resample_seg_labels_subset():
    seg = _seg(X=20, Y=20, Z=20)
    out = resample_seg(seg, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    assert set(np.unique(out)).issubset(set(np.unique(seg)))


def test_resample_seg_output_shape():
    seg = _seg(X=32, Y=32, Z=32)
    out = resample_seg(seg, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    assert out.shape == (64, 64, 64)


def test_resample_seg_nearest_integer_labels():
    seg = _seg()
    out = resample_seg(seg, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), order=0)
    assert set(np.unique(out)).issubset({0, 1, 3})
