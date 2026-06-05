"""Functional tests for modality dropout in both training datasets.

Builds tiny fake data on disk and checks that exactly one input modality is
zeroed at the requested rate, and that validation/inference paths are never
affected.
"""
from __future__ import annotations

import numpy as np
import torch

from mrisynth.model.dataset import Pix2Pix3dDataset
from mrisynth.model.latent_dataset import LatentDataset


# ---------------------------------------------------------------------------
# Pix2Pix3dDataset (image space: pix2pix, WFM, cWDM)
# ---------------------------------------------------------------------------

def _make_npz_dataset(root, n_cases=3, shape=(8, 8, 8)):
    """4-channel volumes with a distinct nonzero constant per channel."""
    for i in range(n_cases):
        data = np.stack(
            [np.full(shape, c + 1, dtype=np.float32) for c in range(4)], axis=0
        )
        seg = np.zeros(shape, dtype=np.int16)
        np.savez(root / f"case{i:03d}.npz", data=data, seg=seg)


def _channel_is_zero(t: torch.Tensor) -> list[bool]:
    return [bool((t[c] == 0).all()) for c in range(t.shape[0])]


def test_pix2pix_dropout_zeros_exactly_one(tmp_path):
    _make_npz_dataset(tmp_path)
    ds = Pix2Pix3dDataset(
        root=tmp_path,
        input_channels=["T1n", "T2FLAIR"],   # n_A = 2
        target_channels=["T1CE"],
        patch_size=None,
        augment=True,
        modality_dropout=1.0,                 # always drop
    )
    seen_dropped = set()
    for _ in range(40):
        A = ds[0]["A"]
        zeros = _channel_is_zero(A)
        assert sum(zeros) == 1, f"expected exactly one zeroed channel, got {zeros}"
        seen_dropped.add(zeros.index(True))
    # Over many draws both input channels should get dropped (random choice).
    assert seen_dropped == {0, 1}


def test_pix2pix_dropout_zero_rate_disables(tmp_path):
    _make_npz_dataset(tmp_path)
    ds = Pix2Pix3dDataset(
        root=tmp_path,
        input_channels=["T1n", "T2FLAIR"],
        target_channels=["T1CE"],
        patch_size=None,
        augment=True,
        modality_dropout=0.0,
    )
    for _ in range(10):
        A = ds[0]["A"]
        assert sum(_channel_is_zero(A)) == 0


def test_pix2pix_dropout_independent_of_augment(tmp_path):
    """Dropout must work with augment=False — WFM/cWDM train that way."""
    _make_npz_dataset(tmp_path)
    ds = Pix2Pix3dDataset(
        root=tmp_path,
        input_channels=["T1n", "T2FLAIR"],
        target_channels=["T1CE"],
        patch_size=None,
        augment=False,           # wavelet-model train mode
        modality_dropout=1.0,
    )
    for _ in range(10):
        A = ds[0]["A"]
        assert sum(_channel_is_zero(A)) == 1


def test_pix2pix_dropout_off_when_val_passes_zero(tmp_path):
    """Val safety is the caller's contract: val datasets pass modality_dropout=0."""
    _make_npz_dataset(tmp_path)
    ds = Pix2Pix3dDataset(
        root=tmp_path,
        input_channels=["T1n", "T2FLAIR"],
        target_channels=["T1CE"],
        patch_size=None,
        augment=False,           # val/inference mode
        modality_dropout=0.0,    # caller passes 0 for val
    )
    for _ in range(10):
        A = ds[0]["A"]
        assert sum(_channel_is_zero(A)) == 0


def test_pix2pix_dropout_rate_is_respected(tmp_path):
    _make_npz_dataset(tmp_path)
    ds = Pix2Pix3dDataset(
        root=tmp_path,
        input_channels=["T1n", "T2FLAIR"],
        target_channels=["T1CE"],
        patch_size=None,
        augment=True,
        modality_dropout=0.5,
    )
    n = 400
    dropped = sum(1 for _ in range(n) if sum(_channel_is_zero(ds[0]["A"])) == 1)
    assert 0.4 < dropped / n < 0.6, f"rate {dropped / n:.2f} off target 0.5"


# ---------------------------------------------------------------------------
# LatentDataset (latent space: RFlow, DiT, ControlNet)
# ---------------------------------------------------------------------------

def _make_latent_dataset(root, n_cases=3, shape=(4, 4, 4, 4)):
    """Per-case latent .pt files; each modality block a distinct constant."""
    keys = {"t1c": 1.0, "t1n": 2.0, "t2f": 3.0}
    for i in range(n_cases):
        cid = f"case{i:03d}"
        d = root / cid
        d.mkdir(parents=True)
        for k, v in keys.items():
            torch.save(torch.full(shape, v), d / f"{cid}-{k}_z_mu.pt")
            torch.save(torch.full(shape, 0.1), d / f"{cid}-{k}_z_sigma.pt")


def _block_is_zero(latent_cond: torch.Tensor, n_blocks: int) -> list[bool]:
    blocks = torch.chunk(latent_cond, n_blocks, dim=0)
    return [bool((b == 0).all()) for b in blocks]


def test_latent_dropout_zeros_exactly_one_block(tmp_path):
    _make_latent_dataset(tmp_path)
    ds = LatentDataset(
        root=tmp_path,
        target_key="t1c",
        cond_keys=["t1n", "t2f"],   # 2 blocks of 4 channels
        deterministic=False,
        modality_dropout=1.0,
    )
    seen = set()
    for _ in range(40):
        lc = ds[0]["latent_cond"]
        assert lc.shape[0] == 8
        zeros = _block_is_zero(lc, n_blocks=2)
        assert sum(zeros) == 1, f"expected one zeroed block, got {zeros}"
        seen.add(zeros.index(True))
    assert seen == {0, 1}


def test_latent_dropout_off_when_deterministic(tmp_path):
    _make_latent_dataset(tmp_path)
    ds = LatentDataset(
        root=tmp_path,
        target_key="t1c",
        cond_keys=["t1n", "t2f"],
        deterministic=True,        # val/inference
        modality_dropout=1.0,      # should be ignored
    )
    for _ in range(10):
        lc = ds[0]["latent_cond"]
        assert sum(_block_is_zero(lc, n_blocks=2)) == 0


def test_latent_dropout_zero_rate_disables(tmp_path):
    _make_latent_dataset(tmp_path)
    ds = LatentDataset(
        root=tmp_path,
        target_key="t1c",
        cond_keys=["t1n", "t2f"],
        deterministic=False,
        modality_dropout=0.0,
    )
    for _ in range(10):
        lc = ds[0]["latent_cond"]
        assert sum(_block_is_zero(lc, n_blocks=2)) == 0
