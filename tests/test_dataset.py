"""Dataset classes: resolve_channels, Pix2Pix3dDataset, LatentDataset."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mrisynth.model.dataset import (
    Pix2Pix3dDataset,
    resolve_channels,
)
from mrisynth.model.latent_dataset import LatentDataset


# ---------------------------------------------------------------------------
# resolve_channels
# ---------------------------------------------------------------------------

def test_resolve_channels_int():
    assert resolve_channels([0, 1, 2, 3]) == [0, 1, 2, 3]


def test_resolve_channels_name_strings():
    assert resolve_channels(["T1CE", "T1n", "T2FLAIR", "T2w"]) == [0, 1, 2, 3]


def test_resolve_channels_aliases():
    assert resolve_channels(["t1c"]) == [0]
    assert resolve_channels(["flair"]) == [2]
    assert resolve_channels(["t2"]) == [3]


def test_resolve_channels_mixed():
    assert resolve_channels([1, "T2FLAIR"]) == [1, 2]


def test_resolve_channels_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown channel name"):
        resolve_channels(["NotAChannel"])


def test_resolve_channels_out_of_range_int_raises():
    with pytest.raises(ValueError, match="out of range"):
        resolve_channels([99])


def test_resolve_channels_wrong_type_raises():
    with pytest.raises(TypeError):
        resolve_channels([3.14])


# ---------------------------------------------------------------------------
# Helpers: build minimal fake .npz files on disk
# ---------------------------------------------------------------------------

def _make_npz_dataset(tmp_dir: Path, n_cases: int = 3,
                       D: int = 32, H: int = 32, W: int = 32) -> list[str]:
    rng = np.random.default_rng(0)
    case_ids = [f"case{i:04d}" for i in range(n_cases)]
    for cid in case_ids:
        data = rng.standard_normal((4, D, H, W)).astype(np.float32)
        seg  = rng.integers(0, 4, (D, H, W), dtype=np.int16)
        np.savez_compressed(tmp_dir / f"{cid}.npz", data=data, seg=seg)
    return case_ids


# ---------------------------------------------------------------------------
# Pix2Pix3dDataset
# ---------------------------------------------------------------------------

class TestPix2Pix3dDataset:
    @pytest.fixture(autouse=True)
    def tmp_root(self, tmp_path):
        self.case_ids = _make_npz_dataset(tmp_path)
        self.root = tmp_path

    def test_len(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=None, augment=False,
        )
        assert len(ds) == len(self.case_ids)

    def test_patches_per_volume_multiplies_len(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=None, augment=False, patches_per_volume=4,
        )
        assert len(ds) == len(self.case_ids) * 4

    def test_sample_A_B_shapes(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=None, augment=False,
        )
        s = ds[0]
        assert s["A"].shape[0] == 1
        assert s["B"].shape[0] == 1
        assert s["A"].shape[1:] == s["B"].shape[1:]

    def test_sample_multi_input(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n", "T2FLAIR"], target_channels=["T1CE"],
            patch_size=None, augment=False,
        )
        s = ds[0]
        assert s["A"].shape[0] == 2
        assert s["B"].shape[0] == 1

    def test_sample_seg_shape(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=None, augment=False,
        )
        s = ds[0]
        assert s["seg"].shape[0] == 1  # (1, D, H, W)
        assert s["A"].shape[1:] == s["seg"].shape[1:]

    def test_sample_a_paths(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=None, augment=False,
        )
        s = ds[0]
        assert isinstance(s["A_paths"], list)
        assert s["A_paths"][0] in self.case_ids

    def test_patch_size_crops_correctly(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=(16, 16, 16), augment=False,
        )
        s = ds[0]
        assert s["A"].shape[1:] == (16, 16, 16)

    def test_dtype_float32(self):
        ds = Pix2Pix3dDataset(
            self.root, input_channels=["T1n"], target_channels=["T1CE"],
            patch_size=None, augment=False,
        )
        s = ds[0]
        assert s["A"].dtype == torch.float32


# ---------------------------------------------------------------------------
# LatentDataset
# ---------------------------------------------------------------------------

def _make_latent_dataset(tmp_dir: Path, n_cases: int = 2,
                          C: int = 4, D: int = 8, H: int = 8, W: int = 8,
                          with_seg: bool = False) -> list[str]:
    case_ids = [f"case{i:04d}" for i in range(n_cases)]
    for cid in case_ids:
        case_dir = tmp_dir / cid
        case_dir.mkdir()
        for mod in ("t1c", "t1n", "t2f"):
            for suffix in ("z_mu", "z_sigma"):
                t = torch.randn(C, D, H, W)
                torch.save(t, case_dir / f"{cid}-{mod}_{suffix}.pt")
        if with_seg:
            seg = torch.zeros(D * 4, H * 4, W * 4, dtype=torch.int8)
            torch.save(seg, case_dir / f"{cid}-seg.pt")
    return case_ids


class TestLatentDataset:
    @pytest.fixture(autouse=True)
    def tmp_root(self, tmp_path):
        self.case_ids = _make_latent_dataset(tmp_path)
        self.root = tmp_path

    def test_len(self):
        ds = LatentDataset(self.root, case_ids=self.case_ids)
        assert len(ds) == len(self.case_ids)

    def test_latent_tgt_shape(self):
        ds = LatentDataset(self.root, case_ids=self.case_ids, deterministic=True)
        s = ds[0]
        assert s["latent_tgt"].shape == (4, 8, 8, 8)

    def test_latent_cond_channels(self):
        ds = LatentDataset(self.root, case_ids=self.case_ids,
                           deterministic=True, cond_keys=["t1n", "t2f"])
        s = ds[0]
        assert s["latent_cond"].shape[0] == 8  # 2 × 4 channels

    def test_seg_none_when_not_present(self):
        ds = LatentDataset(self.root, case_ids=self.case_ids)
        s = ds[0]
        assert s["seg"] is None

    def test_seg_loaded_when_present(self, tmp_path):
        seg_root = tmp_path / "with_seg"
        seg_root.mkdir()
        case_ids = _make_latent_dataset(seg_root, with_seg=True)
        ds = LatentDataset(seg_root, case_ids=case_ids)
        s = ds[0]
        assert s["seg"] is not None
        assert isinstance(s["seg"], torch.Tensor)

    def test_deterministic_uses_mu(self):
        ds = LatentDataset(self.root, case_ids=self.case_ids, deterministic=True)
        s1 = ds[0]
        s2 = ds[0]
        assert torch.equal(s1["latent_tgt"], s2["latent_tgt"])

    def test_case_id_key(self):
        ds = LatentDataset(self.root, case_ids=self.case_ids)
        s = ds[0]
        assert s["case_id"] in self.case_ids

    def test_custom_target_key(self, tmp_path):
        custom_root = tmp_path / "custom"
        custom_root.mkdir()
        case_ids = _make_latent_dataset(custom_root)
        ds = LatentDataset(custom_root, case_ids=case_ids,
                           target_key="t2f", cond_keys=["t1n"])
        s = ds[0]
        assert s["latent_tgt"].shape == (4, 8, 8, 8)

    def test_empty_root_raises(self, tmp_path):
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        with pytest.raises(ValueError, match="No latent cases found"):
            LatentDataset(empty_root)
