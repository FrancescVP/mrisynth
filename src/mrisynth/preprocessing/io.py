"""Nifti IO for nnUNet-format datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


@dataclass
class DatasetInfo:
    name: str
    channel_names: dict[int, str]
    labels: dict[str, int]
    file_ending: str
    num_training: int
    raw: dict


def load_dataset_json(dataset_dir: Path) -> DatasetInfo:
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "dataset.json") as f:
        d = json.load(f)
    channel_names = {int(k): v for k, v in d["channel_names"].items()}
    return DatasetInfo(
        name=d.get("name", dataset_dir.name),
        channel_names=channel_names,
        labels=d["labels"],
        file_ending=d["file_ending"],
        num_training=d["numTraining"],
        raw=d,
    )


def list_case_ids(dataset_dir: Path, ending: str) -> list[str]:
    images_dir = Path(dataset_dir) / "imagesTr"
    ids = set()
    for p in images_dir.glob(f"*_0000{ending}"):
        ids.add(p.name[: -len(f"_0000{ending}")])
    return sorted(ids)


def load_case(
    dataset_dir: Path,
    case_id: str,
    num_channels: int,
    file_ending: str,
    load_seg: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Load all channels and optional seg for one case.

    Returns:
        data: (C, X, Y, Z) float32
        seg:  (X, Y, Z) int8 or None
        props: dict with spacing, affine, original_shape
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagesTr"

    channels = []
    spacing = None
    affine = None
    for c in range(num_channels):
        path = images_dir / f"{case_id}_{c:04d}{file_ending}"
        img = nib.load(path)
        arr = np.asanyarray(img.dataobj).astype(np.float32)
        channels.append(arr)
        if c == 0:
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
            affine = np.asarray(img.affine, dtype=np.float64)
    data = np.stack(channels, axis=0)

    seg = None
    if load_seg:
        seg_path = dataset_dir / "labelsTr" / f"{case_id}{file_ending}"
        if seg_path.exists():
            seg_img = nib.load(seg_path)
            seg = np.asanyarray(seg_img.dataobj).astype(np.int8)

    props = {
        "case_id": case_id,
        "spacing": spacing,
        "affine": affine.tolist() if affine is not None else None,
        "original_shape": data.shape[1:],
    }
    return data, seg, props


def save_processed_case(
    output_dir: Path,
    case_id: str,
    data: np.ndarray,
    seg: Optional[np.ndarray],
    props: dict,
    compress: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {"data": data.astype(np.float32, copy=False)}
    if seg is not None:
        payload["seg"] = seg.astype(np.int8, copy=False)

    saver = np.savez_compressed if compress else np.savez
    saver(output_dir / f"{case_id}.npz", **payload)

    with open(output_dir / f"{case_id}.json", "w") as f:
        json.dump(props, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not serializable: {type(o)}")
