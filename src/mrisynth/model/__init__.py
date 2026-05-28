"""3-D model package for volumetric MRI synthesis (pix2pix + latent diffusion)."""
from .pix2pix3d import Pix2Pix3dModel
from .rflow import RFlowModel
from .vae import MaisiVAE, LATENT_CHANNELS, SPATIAL_DOWNSAMPLE
from .latent_dataset import LatentDataset
from .dataset import Pix2Pix3dDataset, CHANNEL_NAMES, resolve_channels
from .networks import (
    GANLoss,
    define_G_3d,
    define_D_3d,
    UnetGenerator3D,
    NLayerDiscriminator3D,
    get_norm_layer_3d,
    get_scheduler,
    init_weights,
    init_net,
)

__all__ = [
    "Pix2Pix3dModel",
    "RFlowModel",
    "MaisiVAE",
    "LATENT_CHANNELS",
    "SPATIAL_DOWNSAMPLE",
    "LatentDataset",
    "Pix2Pix3dDataset",
    "CHANNEL_NAMES",
    "resolve_channels",
    "GANLoss",
    "define_G_3d",
    "define_D_3d",
    "UnetGenerator3D",
    "NLayerDiscriminator3D",
    "get_norm_layer_3d",
    "get_scheduler",
    "init_weights",
    "init_net",
]
