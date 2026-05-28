"""End-to-end preprocessing pipeline: load -> crop -> resample -> normalize -> save."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .cropping import crop_to_nonzero
from .io import (
    DatasetInfo,
    list_case_ids,
    load_case,
    load_dataset_json,
    save_processed_case,
)
from .normalization import percentile_clip_zscore, zscore_global, zscore_nonzero
from .resampling import resample_data, resample_seg


NORMALIZATION_FNS = {
    "zscore_nonzero": zscore_nonzero,
    "zscore_global": zscore_global,
    "percentile_clip_zscore": percentile_clip_zscore,
}


@dataclass
class PreprocessConfig:
    target_spacing: Optional[tuple[float, float, float]] = None  # None = keep original
    normalization: str = "zscore_nonzero"
    crop_to_nonzero: bool = True
    image_resample_order: int = 3
    seg_resample_order: int = 1
    save_compressed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess_case(
    dataset_dir: Path,
    case_id: str,
    info: DatasetInfo,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    num_channels = len(info.channel_names)
    data, seg, props = load_case(
        dataset_dir, case_id, num_channels, info.file_ending, load_seg=True
    )
    props["channel_names"] = info.channel_names
    props["labels"] = info.labels

    if cfg.crop_to_nonzero:
        data, seg, bbox = crop_to_nonzero(data, seg)
        props["crop_bbox"] = bbox
        props["shape_after_crop"] = list(data.shape[1:])

    old_spacing = props["spacing"]
    if cfg.target_spacing is not None and tuple(old_spacing) != tuple(cfg.target_spacing):
        data_rs = resample_data(
            data, old_spacing, cfg.target_spacing, order=cfg.image_resample_order
        )
        seg_rs = (
            resample_seg(
                seg, old_spacing, cfg.target_spacing, order=cfg.seg_resample_order
            )
            if seg is not None
            else None
        )
        data, seg = data_rs, seg_rs
        props["spacing_after_resample"] = list(cfg.target_spacing)
        props["shape_after_resample"] = list(data.shape[1:])
    else:
        props["spacing_after_resample"] = list(old_spacing)
        props["shape_after_resample"] = list(data.shape[1:])

    norm_fn = NORMALIZATION_FNS[cfg.normalization]
    fg_mask = (data != 0).any(axis=0)
    data = norm_fn(data, fg_mask) if cfg.normalization != "zscore_global" else norm_fn(data)
    props["foreground_voxels"] = int(fg_mask.sum())
    props["preprocess_config"] = cfg.to_dict()

    return data, seg, props


def preprocess_dataset(
    dataset_dir: Path,
    output_dir: Path,
    cfg: Optional[PreprocessConfig] = None,
    case_ids: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    verbose: bool = True,
) -> None:
    cfg = cfg or PreprocessConfig()
    info = load_dataset_json(dataset_dir)
    if case_ids is None:
        case_ids = list_case_ids(dataset_dir, info.file_ending)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save dataset-level config + channel/label info.
    import json as _json

    with open(output_dir / "preprocess_info.json", "w") as f:
        _json.dump(
            {
                "dataset_name": info.name,
                "channel_names": info.channel_names,
                "labels": info.labels,
                "config": cfg.to_dict(),
                "num_cases": len(case_ids),
            },
            f,
            indent=2,
        )

    if num_workers <= 1:
        for cid in case_ids:
            _process_one(dataset_dir, cid, info, cfg, output_dir, verbose)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_process_one, dataset_dir, cid, info, cfg, output_dir, verbose): cid
            for cid in case_ids
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERR] {cid}: {e}")


def _process_one(
    dataset_dir: Path,
    case_id: str,
    info: DatasetInfo,
    cfg: PreprocessConfig,
    output_dir: Path,
    verbose: bool,
) -> None:
    if verbose:
        print(f"[preprocess] {case_id}", flush=True)
    data, seg, props = preprocess_case(dataset_dir, case_id, info, cfg)
    save_processed_case(output_dir, case_id, data, seg, props, compress=cfg.save_compressed)
