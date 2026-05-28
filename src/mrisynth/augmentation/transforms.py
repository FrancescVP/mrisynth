"""nnUNet-style augmentation recipe for GAN-based MRI synthesis.

Mirrors `nnUNetTrainer.get_training_transforms` from nnUNetv2 with two
adjustments for image synthesis:

1. Segmentation-specific transforms (`MoveSegAsOneHotToData`,
   `ConvertSegmentationToRegions`, `DownsampleSegForDS`) are dropped — we
   do not train a segmentation head.
2. Spatial transforms (rotation, scale, mirroring, low-resolution
   simulation) are synchronized across input channels (T1CE / T1 /
   T2FLAIR / T2 must stay registered). Intensity transforms are
   *desynchronized* per channel — different scanners give different gain
   per modality, so per-channel intensity perturbation is realistic.

The recipe is exposed as a dataclass `AugConfig` so probabilities and
ranges can be tuned without rewriting the factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform


@dataclass
class AugConfig:
    """Tunable knobs for the training-aug recipe.

    Defaults mirror nnUNet's `nnUNetTrainer.get_training_transforms`
    probabilities and ranges.
    """
    # Spatial.
    rotation_range_deg: Tuple[float, float] = (-30.0, 30.0)
    p_rotation: float = 0.2
    scaling_range: Tuple[float, float] = (0.7, 1.4)
    p_scaling: float = 0.2

    # Intensity.
    p_gaussian_noise: float = 0.1
    gaussian_noise_variance: Tuple[float, float] = (0.0, 0.1)

    p_gaussian_blur: float = 0.2
    gaussian_blur_sigma: Tuple[float, float] = (0.5, 1.0)

    p_brightness: float = 0.15
    brightness_multiplier_range: Tuple[float, float] = (0.75, 1.25)

    p_contrast: float = 0.15
    contrast_range: Tuple[float, float] = (0.75, 1.25)

    p_lowres: float = 0.25
    lowres_scale_range: Tuple[float, float] = (0.5, 1.0)

    p_gamma_inverted: float = 0.1
    p_gamma: float = 0.3
    gamma_range: Tuple[float, float] = (0.7, 1.5)

    # Mirroring.
    mirror_axes: Optional[Sequence[int]] = field(default=None)
    """Axes (0=X, 1=Y, 2=Z) along which random flipping is allowed.

    For brain MRI, only sagittal flip (left-right) is anatomically benign;
    set to `(0,)` for typical RAI orientation. Use `None` to disable.
    """


def build_train_transforms(
    cfg: AugConfig,
    patch_size: Sequence[int],
) -> BasicTransform:
    """Build the training-augmentation pipeline.

    Args:
        cfg: tunable config.
        patch_size: spatial patch size (X, Y, Z) for SpatialTransform's
            crop. SpatialTransform applies rotation/scaling and crops to
            this size, so the input volume must be at least this big.

    Returns:
        A `ComposeTransforms` callable: `compose(image=tensor, segmentation=tensor)`
        returning a dict with the same keys.
    """
    transforms: list[BasicTransform] = []

    # 1. Spatial: rotation, scale, deterministic crop (no random crop —
    # patches are sampled by the dataset). Channels stay synchronized.
    rotation: RandomScalar = (
        cfg.rotation_range_deg[0] * 3.14159265 / 180.0,
        cfg.rotation_range_deg[1] * 3.14159265 / 180.0,
    )
    transforms.append(
        SpatialTransform(
            patch_size=tuple(patch_size),
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0.0,
            p_rotation=cfg.p_rotation,
            rotation=rotation,
            p_scaling=cfg.p_scaling,
            scaling=cfg.scaling_range,
            p_synchronize_scaling_across_axes=1.0,
            bg_style_seg_sampling=False,
            border_mode_seg="constant",
            padding_value_seg=-1,
        )
    )

    # 2. Intensity: per-channel desynchronized.
    transforms.append(
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=cfg.gaussian_noise_variance,
                p_per_channel=1.0,
                synchronize_channels=True,  # noise: keep paired modalities consistent
            ),
            apply_probability=cfg.p_gaussian_noise,
        )
    )
    transforms.append(
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=cfg.gaussian_blur_sigma,
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            ),
            apply_probability=cfg.p_gaussian_blur,
        )
    )
    transforms.append(
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast(cfg.brightness_multiplier_range),
                synchronize_channels=False,
                p_per_channel=1.0,
            ),
            apply_probability=cfg.p_brightness,
        )
    )
    transforms.append(
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast(cfg.contrast_range),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1.0,
            ),
            apply_probability=cfg.p_contrast,
        )
    )
    transforms.append(
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=cfg.lowres_scale_range,
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=cfg.p_lowres,
        )
    )
    # Inverted gamma (negative-image style).
    transforms.append(
        RandomTransform(
            GammaTransform(
                gamma=BGContrast(cfg.gamma_range),
                p_invert_image=1.0,
                synchronize_channels=False,
                p_per_channel=1.0,
                p_retain_stats=1.0,
            ),
            apply_probability=cfg.p_gamma_inverted,
        )
    )
    # Standard gamma.
    transforms.append(
        RandomTransform(
            GammaTransform(
                gamma=BGContrast(cfg.gamma_range),
                p_invert_image=0.0,
                synchronize_channels=False,
                p_per_channel=1.0,
                p_retain_stats=1.0,
            ),
            apply_probability=cfg.p_gamma,
        )
    )

    # 3. Mirroring (after intensity).
    if cfg.mirror_axes is not None and len(cfg.mirror_axes) > 0:
        transforms.append(MirrorTransform(allowed_axes=tuple(cfg.mirror_axes)))

    return ComposeTransforms(transforms)


def build_val_transforms() -> BasicTransform:
    """Identity-ish val pipeline. Kept for symmetry with the train builder."""
    return ComposeTransforms([])


def build_spatial_transforms(
    cfg: AugConfig,
    patch_size: Sequence[int],
    random_crop: bool = True,
) -> BasicTransform:
    """Spatial-only pipeline: rotation, scaling, mirroring, and patch crop.

    Apply to **all** channels stacked together (A + B + seg) so that the
    spatial transform is perfectly synchronised across input and target.

    Args:
        cfg: augmentation config.
        patch_size: output spatial size after crop.
        random_crop: if True (default) the crop centre is sampled randomly
            inside the volume; if False the centre is always the image centre.
    """
    rotation: RandomScalar = (
        cfg.rotation_range_deg[0] * 3.14159265 / 180.0,
        cfg.rotation_range_deg[1] * 3.14159265 / 180.0,
    )
    transforms: list[BasicTransform] = [
        SpatialTransform(
            patch_size=tuple(patch_size),
            patch_center_dist_from_border=0,
            random_crop=random_crop,
            p_elastic_deform=0.0,
            p_rotation=cfg.p_rotation,
            rotation=rotation,
            p_scaling=cfg.p_scaling,
            scaling=cfg.scaling_range,
            p_synchronize_scaling_across_axes=1.0,
            bg_style_seg_sampling=False,
            border_mode_seg="constant",
            padding_value_seg=-1,
        )
    ]
    if cfg.mirror_axes is not None and len(cfg.mirror_axes) > 0:
        transforms.append(MirrorTransform(allowed_axes=tuple(cfg.mirror_axes)))
    return ComposeTransforms(transforms)


def build_intensity_transforms(cfg: AugConfig) -> BasicTransform:
    """Intensity-only pipeline: noise, blur, brightness, contrast, gamma.

    Apply to **input (A) channels only** — the synthesis target (B) should
    stay at ground-truth intensity so that the L1 loss is not corrupted.
    Each channel is augmented independently (simulates scanner variability).
    """
    transforms: list[BasicTransform] = [
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=cfg.gaussian_noise_variance,
                p_per_channel=1.0,
                synchronize_channels=False,
            ),
            apply_probability=cfg.p_gaussian_noise,
        ),
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=cfg.gaussian_blur_sigma,
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            ),
            apply_probability=cfg.p_gaussian_blur,
        ),
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast(cfg.brightness_multiplier_range),
                synchronize_channels=False,
                p_per_channel=1.0,
            ),
            apply_probability=cfg.p_brightness,
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast(cfg.contrast_range),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1.0,
            ),
            apply_probability=cfg.p_contrast,
        ),
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=cfg.lowres_scale_range,
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=cfg.p_lowres,
        ),
        RandomTransform(
            GammaTransform(
                gamma=BGContrast(cfg.gamma_range),
                p_invert_image=1.0,
                synchronize_channels=False,
                p_per_channel=1.0,
                p_retain_stats=1.0,
            ),
            apply_probability=cfg.p_gamma_inverted,
        ),
        RandomTransform(
            GammaTransform(
                gamma=BGContrast(cfg.gamma_range),
                p_invert_image=0.0,
                synchronize_channels=False,
                p_per_channel=1.0,
                p_retain_stats=1.0,
            ),
            apply_probability=cfg.p_gamma,
        ),
    ]
    return ComposeTransforms(transforms)
