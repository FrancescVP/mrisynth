"""PyTorch Dataset over preprocessed `.npz` files.

Reads the per-case `.npz` produced by `gan_data_processing.preprocessing`
and applies a `batchgeneratorsv2` `ComposeTransforms` per sample.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class ProcessedNiftiDataset(Dataset):
    """Per-case dataset of preprocessed multimodal MRI volumes.

    Each `.npz` is expected to contain:
        - ``data``: float32 array of shape ``(C, X, Y, Z)``
        - ``seg``:  int8 array of shape ``(X, Y, Z)`` (optional)

    Returns dict with keys matching the bgv2 contract:
        - ``image``: torch.float32, shape ``(C, X, Y, Z)``
        - ``segmentation``: torch.int8, shape ``(1, X, Y, Z)`` (if available)
        - ``case_id``: str

    Args:
        root: directory of `.npz` files (output of preprocess pipeline).
        case_ids: optional subset; else all .npz files are used.
        transforms: optional bgv2 `ComposeTransforms`. Applied per sample.
        return_seg: if False, the seg key is omitted from the output even
            when present on disk.
        dtype: torch dtype to cast image to (default float32).
    """

    def __init__(
        self,
        root: Path,
        case_ids: Optional[Sequence[str]] = None,
        transforms: Optional[BasicTransform] = None,
        return_seg: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.root = Path(root)
        if case_ids is None:
            case_ids = sorted(p.stem for p in self.root.glob("*.npz"))
        self.case_ids = list(case_ids)
        self.transforms = transforms
        self.return_seg = return_seg
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]
        path = self.root / f"{case_id}.npz"
        with np.load(path) as z:
            data = torch.from_numpy(np.asarray(z["data"])).to(dtype=self.dtype)
            seg = (
                torch.from_numpy(np.asarray(z["seg"]))
                if (self.return_seg and "seg" in z.files)
                else None
            )

        sample: dict = {"image": data, "case_id": case_id}
        if seg is not None:
            # bgv2 expects a leading channel dim on segmentation as well.
            sample["segmentation"] = seg.unsqueeze(0).to(dtype=torch.int16)

        if self.transforms is not None:
            sample = self.transforms(**sample)
        return sample

    def load_props(self, case_id: str) -> dict:
        """Read the sidecar JSON written by the preprocess pipeline."""
        with open(self.root / f"{case_id}.json") as f:
            return json.load(f)
