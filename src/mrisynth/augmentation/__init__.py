"""Augmentation built on `batchgeneratorsv2`.

Thin adapter — we do not reimplement transforms. The recipe mirrors
nnUNet's default training augmentation, with the segmentation-specific
transforms (cascade, region conversion, deep-supervision downsampling)
removed so it fits an image-synthesis (GAN) workflow.
"""
from .dataset import ProcessedNiftiDataset
from .transforms import (
    AugConfig,
    build_train_transforms,
    build_val_transforms,
    build_spatial_transforms,
    build_intensity_transforms,
)

__all__ = [
    "AugConfig",
    "ProcessedNiftiDataset",
    "build_train_transforms",
    "build_val_transforms",
    "build_spatial_transforms",
    "build_intensity_transforms",
]
