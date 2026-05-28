"""CLI entrypoint installed as `gan-preprocess`."""
from __future__ import annotations

import argparse
from pathlib import Path

from .preprocessing import PreprocessConfig, preprocess_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gan-preprocess",
        description="Preprocess an nnUNet-format multimodal MRI dataset for GAN training.",
    )
    p.add_argument("--dataset", required=True, type=Path, help="nnUNet raw dataset dir.")
    p.add_argument("--output", required=True, type=Path, help="Output dir for .npz/.json.")
    p.add_argument(
        "--target-spacing",
        nargs=3,
        type=float,
        default=None,
        help="Target voxel spacing (X Y Z) in mm. Omit to keep original.",
    )
    p.add_argument(
        "--normalization",
        choices=["zscore_nonzero", "zscore_global", "percentile_clip_zscore"],
        default="zscore_nonzero",
    )
    p.add_argument("--no-crop", action="store_true", help="Skip nonzero crop.")
    p.add_argument("--image-order", type=int, default=3)
    p.add_argument("--seg-order", type=int, default=1)
    p.add_argument("--no-compress", action="store_true", help="Save uncompressed .npz.")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--cases", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
        normalization=args.normalization,
        crop_to_nonzero=not args.no_crop,
        image_resample_order=args.image_order,
        seg_resample_order=args.seg_order,
        save_compressed=not args.no_compress,
    )
    preprocess_dataset(
        dataset_dir=args.dataset,
        output_dir=args.output,
        cfg=cfg,
        case_ids=args.cases,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
