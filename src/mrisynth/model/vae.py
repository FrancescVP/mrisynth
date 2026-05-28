"""Frozen MAISI VAE wrapper for latent-space diffusion.

Architecture matches autoencoder_epoch273.pt:
  AutoencoderKlMaisi(channels=[64,128,256], latent_channels=4, in_channels=1)
  Spatial compression: 4× per axis  →  (1,D,H,W) → (4, D//4, H//4, W//4)

The checkpoint is stored in git-lfs in the companion repo.  To download it:
    cd /home/fran/Projects/mri_image_synthesis/\\
        An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation
    git lfs pull
Then pass the path via --vae_ckpt to generate_latents.py / train_rflow.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Architecture config matching the pretrained checkpoint
# ---------------------------------------------------------------------------
_VAE_CFG = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    num_channels=[64, 128, 256],
    num_res_blocks=[2, 2, 2],
    norm_num_groups=32,
    norm_eps=1e-6,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    use_checkpointing=False,
    use_convtranspose=False,
    norm_float16=False,
    num_splits=4,
    dim_split=1,
)

LATENT_CHANNELS = 4
SPATIAL_DOWNSAMPLE = 4  # input / 4 per spatial axis in the latent


def _build_ae() -> nn.Module:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
    return AutoencoderKlMaisi(**_VAE_CFG)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, tuple]:
    """Pad spatial dims to be divisible by *multiple*.  Returns (padded, pad_amounts)."""
    _, _, D, H, W = x.shape
    pd = (multiple - D % multiple) % multiple
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    if pd == 0 and ph == 0 and pw == 0:
        return x, (0, 0, 0, 0, 0, 0)
    # F.pad order is (W_before, W_after, H_before, H_after, D_before, D_after)
    pad = (0, pw, 0, ph, 0, pd)
    return F.pad(x, pad, mode="constant", value=0.0), pad


def _unpad(x: torch.Tensor, pad: tuple) -> torch.Tensor:
    """Remove padding added by _pad_to_multiple (acts on spatial dims)."""
    _, pw_after, _, ph_after, _, pd_after = pad
    D, H, W = x.shape[-3:]
    x = x[..., : D - pd_after if pd_after else D,
               : H - ph_after if ph_after else H,
               : W - pw_after if pw_after else W]
    return x


class MaisiVAE(nn.Module):
    """Frozen pretrained MAISI VAE — single-channel MRI → 4-channel latents.

    Usage
    -----
    vae = MaisiVAE.from_checkpoint(ckpt_path, device)
    mu, sigma = vae.encode(x)        # x: (B,1,D,H,W) in [0,1]; returns (B,4,D',H',W')
    z = MaisiVAE.sample(mu, sigma)
    x_hat = vae.decode(z)            # (B,1,D,H,W)
    """

    def __init__(self, ae: nn.Module):
        super().__init__()
        self.ae = ae
        for p in self.ae.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "MaisiVAE":
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"VAE checkpoint not found: {ckpt_path}\n"
                "It is stored in git-lfs.  Run:\n"
                "  cd /home/fran/Projects/mri_image_synthesis/"
                "An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation\n"
                "  git lfs pull\n"
                "Then re-run with --vae_ckpt <path>."
            )
        ae = _build_ae()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        ae.load_state_dict(state, strict=True)
        ae.float()  # norm_float16=True leaves some buffers as fp16; force fp32
        ae.eval()
        ae.to(device)
        return cls(ae)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single-channel volume.

        Parameters
        ----------
        x : (B, 1, D, H, W)  float32, values normalised to [0, 1]

        Returns
        -------
        mu, sigma : each (B, 4, D//4, H//4, W//4)
        """
        x_padded, _ = _pad_to_multiple(x)
        z_mu, z_sigma = self.ae.encode(x_padded)
        return z_mu, z_sigma

    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu + sigma * torch.randn_like(sigma)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, original_spatial: Optional[tuple] = None) -> torch.Tensor:
        """Decode a latent to image space.

        Parameters
        ----------
        z                : (B, 4, D', H', W')
        original_spatial : (D, H, W) of the un-padded input — used to crop away
                           any padding artefacts.  If None, no cropping is done.
        """
        x = self.ae.decode(z)
        if original_spatial is not None:
            D, H, W = original_spatial
            x = x[:, :, :D, :H, :W]
        return x

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Call encode() / decode() directly.")
