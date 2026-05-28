"""Augmentation pipeline: shapes preserved, seg/image stay registered."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mrisynth.augmentation import (
    AugConfig,
    ProcessedNiftiDataset,
    build_train_transforms,
    build_val_transforms,
)


def _fake_case_npz(tmp_path, case_id="case_0", shape=(4, 64, 64, 64), with_seg=True):
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape).astype(np.float32)
    arrays = {"data": data}
    if with_seg:
        seg = np.zeros(shape[1:], dtype=np.int8)
        seg[24:40, 24:40, 24:40] = 2
        seg[28:36, 28:36, 28:36] = 3
        arrays["seg"] = seg
    np.savez(tmp_path / f"{case_id}.npz", **arrays)
    (tmp_path / f"{case_id}.json").write_text("{}")
    return case_id


def test_dataset_loads_without_transforms(tmp_path):
    cid = _fake_case_npz(tmp_path)
    ds = ProcessedNiftiDataset(tmp_path)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["case_id"] == cid
    assert sample["image"].shape == (4, 64, 64, 64)
    assert sample["segmentation"].shape == (1, 64, 64, 64)
    assert sample["image"].dtype == torch.float32


def test_train_transforms_preserves_shape_and_keys(tmp_path):
    _fake_case_npz(tmp_path)
    cfg = AugConfig(mirror_axes=(0,))
    tfm = build_train_transforms(cfg, patch_size=(64, 64, 64))
    ds = ProcessedNiftiDataset(tmp_path, transforms=tfm)

    torch.manual_seed(0)
    sample = ds[0]
    assert sample["image"].shape == (4, 64, 64, 64)
    assert sample["segmentation"].shape == (1, 64, 64, 64)
    # Segmentation labels should still be in the original set.
    assert set(sample["segmentation"].unique().tolist()).issubset({0, 2, 3, -1})


def test_val_transforms_identity(tmp_path):
    _fake_case_npz(tmp_path)
    tfm = build_val_transforms()
    ds = ProcessedNiftiDataset(tmp_path, transforms=tfm)
    sample = ds[0]
    # Val pipeline should not modify shape.
    assert sample["image"].shape == (4, 64, 64, 64)


def test_train_transforms_keep_image_seg_registered(tmp_path):
    """Spatial transforms must apply identically to image and seg.

    Construct an image whose values encode position, run aug, check that
    where the seg has label > 0, the image values are still consistent
    with the seg locations after transform (proxy: nonzero seg pixels
    correspond to the elevated-image-value region).
    """
    rng = np.random.default_rng(1)
    shape = (1, 48, 48, 48)
    data = rng.standard_normal(shape).astype(np.float32) * 0.01
    seg = np.zeros(shape[1:], dtype=np.int8)
    seg[16:32, 16:32, 16:32] = 1
    data[0][seg == 1] += 5.0  # large signal where seg > 0

    np.savez(tmp_path / "c.npz", data=data, seg=seg)
    (tmp_path / "c.json").write_text("{}")

    cfg = AugConfig(p_rotation=1.0, p_scaling=1.0, mirror_axes=(0, 1, 2))
    tfm = build_train_transforms(cfg, patch_size=(48, 48, 48))
    ds = ProcessedNiftiDataset(tmp_path, transforms=tfm)

    torch.manual_seed(0)
    sample = ds[0]
    img = sample["image"][0]
    seg_t = sample["segmentation"][0]
    inside_mean = img[seg_t == 1].mean().item()
    outside_mean = img[seg_t == 0].mean().item()
    assert inside_mean > outside_mean + 1.0, (
        f"image and seg desynchronized after spatial aug: "
        f"inside={inside_mean:.3f} outside={outside_mean:.3f}"
    )


def test_no_seg_path(tmp_path):
    _fake_case_npz(tmp_path, with_seg=False)
    ds = ProcessedNiftiDataset(tmp_path, return_seg=False)
    sample = ds[0]
    assert "segmentation" not in sample
    assert sample["image"].shape == (4, 64, 64, 64)


def test_dataloader_collation(tmp_path):
    """Sanity: batching with default collate works on returned dicts."""
    for i in range(3):
        _fake_case_npz(tmp_path, case_id=f"c{i}")
    cfg = AugConfig()
    tfm = build_train_transforms(cfg, patch_size=(64, 64, 64))
    ds = ProcessedNiftiDataset(tmp_path, transforms=tfm)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 4, 64, 64, 64)
    assert batch["segmentation"].shape == (2, 1, 64, 64, 64)
    assert isinstance(batch["case_id"], list) and len(batch["case_id"]) == 2
