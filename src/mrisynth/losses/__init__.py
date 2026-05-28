"""Losses for GAN and latent-diffusion MRI synthesis."""
from ..utils import seg_to_mask
from .composite import TumorAwareGANLoss, make_default_et_aware_loss
from .enhancing_tumor import EnhancingTumorLoss, enhancing_tumor_loss
from .tumor_ssim import (
    RegionWeightedSSIMLoss,
    TumorSSIMLoss,
    region_weighted_ssim_loss,
    tumor_ssim_loss,
)
from .velocity import (
    SegAwareLoss,
    NCCLoss,
    SSIMLoss,
    L1SSIMLoss,
    TumorWeightedL1Loss,
    RegionContrastiveLoss,
    L1RegionContrastiveLoss,
    build_velocity_loss,
)

__all__ = [
    "seg_to_mask",
    # Pix2pix tumor-aware losses.
    "TumorSSIMLoss",
    "RegionWeightedSSIMLoss",
    "tumor_ssim_loss",
    "region_weighted_ssim_loss",
    "EnhancingTumorLoss",
    "enhancing_tumor_loss",
    "TumorAwareGANLoss",
    "make_default_et_aware_loss",
    # RFlow velocity losses.
    "SegAwareLoss",
    "NCCLoss",
    "SSIMLoss",
    "L1SSIMLoss",
    "TumorWeightedL1Loss",
    "RegionContrastiveLoss",
    "L1RegionContrastiveLoss",
    "build_velocity_loss",
]
