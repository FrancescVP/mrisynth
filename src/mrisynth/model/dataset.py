"""Flexible paired 3-D MRI dataset for pix2pix training.

Loads nnUNet-preprocessed `.npz` files (output of `mrisynth.preprocessing`)
and exposes them as paired (A → B) samples suitable for Pix2Pix3dModel.

Features
--------
- **One-to-one** translation: e.g. ``input_channels=["T1n"]`` → ``target_channels=["T1CE"]``
- **Multimodal** input: e.g. ``input_channels=["T1n", "FLAIR", "T2w"]`` → ``target_channels=["T1CE"]``
- **Flexible channel spec**: accept integer indices or name strings (see CHANNEL_NAMES).
- **Correct augmentation split**:
    - Spatial transforms (rotation, flip, random crop) → A + B + seg together (stay aligned).
    - Intensity transforms (noise, blur, brightness, gamma) → A only (B is ground-truth).
- **Segmentation masks** for tumor-aware losses (labels 0–3: bg, NETC, SNFH, ET).
- **patches_per_volume**: multiply dataset length without extra IO — each call
  produces a fresh random crop from the same volume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..augmentation.transforms import (
    AugConfig,
    build_intensity_transforms,
    build_spatial_transforms,
)


# ---------------------------------------------------------------------------
# Channel name registry
# Dataset015_GeneralBrainTumor convention (nnUNet channel order)
# ---------------------------------------------------------------------------

CHANNEL_NAMES: dict[int, str] = {
    0: "T1CE",
    1: "T1n",
    2: "T2FLAIR",
    3: "T2w",
}

_ALIASES: dict[str, int] = {
    # T1CE — contrast-enhanced T1 (channel 0)
    "T1CE": 0, "T1C": 0, "T1c": 0, "t1ce": 0, "t1c": 0,
    # T1n — non-contrast T1 (channel 1)
    "T1": 1, "T1n": 1, "T1N": 1, "t1": 1, "t1n": 1,
    # T2FLAIR (channel 2)
    "T2FLAIR": 2, "FLAIR": 2, "flair": 2, "T2F": 2, "t2f": 2, "t2flair": 2,
    # T2w (channel 3)
    "T2": 3, "T2w": 3, "T2W": 3, "t2": 3, "t2w": 3,
}


def resolve_channels(spec: Sequence[Union[int, str]]) -> list[int]:
    """Convert a channel specification to a list of integer indices.

    Each element can be an int (0–3) or a channel name string.

    >>> resolve_channels(["T1n", "FLAIR", "T2w"])
    [1, 2, 3]
    >>> resolve_channels([0])
    [0]
    """
    out: list[int] = []
    for s in spec:
        if isinstance(s, int):
            if s not in CHANNEL_NAMES:
                raise ValueError(f"Channel index {s} out of range; valid: {list(CHANNEL_NAMES)}")
            out.append(s)
        elif isinstance(s, str):
            if s not in _ALIASES:
                raise ValueError(
                    f"Unknown channel name {s!r}. "
                    f"Valid names: {sorted(_ALIASES)}"
                )
            out.append(_ALIASES[s])
        else:
            raise TypeError(f"Channel spec must be int or str, got {type(s)!r}")
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Pix2Pix3dDataset(Dataset):
    """Paired 3-D MRI dataset for pix2pix training.

    Each sample is a dict with keys expected by ``Pix2Pix3dModel.set_input``:

    - ``"A"``:       float32 tensor ``(C_in, D, H, W)``  — input channels
    - ``"B"``:       float32 tensor ``(C_out, D, H, W)`` — target channels
    - ``"seg"``:     int16 tensor  ``(1, D, H, W)``      — segmentation mask
    - ``"A_paths"``: list of str                          — case ID(s)

    Args:
        root: directory of ``.npz`` files produced by the preprocessing pipeline.
        input_channels: channels used as model input (A). Each element can be
            an int index or a name string — see ``CHANNEL_NAMES`` / ``_ALIASES``.
            Examples: ``["T1n"]``, ``["T1n", "FLAIR", "T2w"]``, ``[1, 2, 3]``.
        target_channels: channels used as synthesis target (B). Typically
            ``["T1CE"]`` or ``[0]``.
        patch_size: spatial size of the extracted patch ``(D, H, W)``.
            If ``None``, returns the full volume (use for validation / inference).
        augment: if ``True``, apply spatial + intensity augmentation (train mode).
            Spatial aug is always applied to A, B and seg jointly.
            Intensity aug is applied to A only.
        aug_cfg: ``AugConfig`` instance; defaults to ``AugConfig()`` if ``None``.
        patches_per_volume: multiply the effective dataset length.  Because the
            spatial transform samples a fresh random crop each call, repeating
            the same case index gives a different patch — this is a zero-cost
            way to see more of each volume per epoch.
        modality_dropout: fraction of samples in which exactly one input (A)
            channel is zeroed (chosen uniformly at random), to train robustness
            to a missing input sequence. Requires >1 input channel. Applied only
            when ``augment=True`` (train mode), so validation/inference always
            see the full input. ``0.0`` disables it entirely.
        case_ids: optional explicit list of case IDs; if ``None`` all ``.npz``
            files in *root* are used.
        dtype: floating-point dtype for image tensors (default ``float32``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        input_channels: Sequence[Union[int, str]] = ("T1n",),
        target_channels: Sequence[Union[int, str]] = ("T1CE",),
        patch_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
        augment: bool = True,
        aug_cfg: Optional[AugConfig] = None,
        patches_per_volume: int = 1,
        modality_dropout: float = 0.0,
        case_ids: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.root = Path(root)
        self.channels_A = resolve_channels(input_channels)
        self.channels_B = resolve_channels(target_channels)
        self.n_A = len(self.channels_A)
        self.n_B = len(self.channels_B)
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_volume = max(1, patches_per_volume)
        self.modality_dropout = float(modality_dropout)
        self.dtype = dtype

        if case_ids is None:
            case_ids = sorted(p.stem for p in self.root.glob("*.npz"))
        self.case_ids = list(case_ids)

        cfg = aug_cfg if aug_cfg is not None else AugConfig()

        # Build transforms — only when a patch size is specified
        if patch_size is not None and augment:
            self._spatial_tf  = build_spatial_transforms(cfg, patch_size, random_crop=True)
            self._intensity_tf = build_intensity_transforms(cfg)
        elif patch_size is not None:
            # Center-crop only, no augmentation
            self._spatial_tf  = build_spatial_transforms(
                AugConfig(p_rotation=0.0, p_scaling=0.0, mirror_axes=None),
                patch_size, random_crop=False,
            )
            self._intensity_tf = None
        else:
            # Full-volume mode (validation / inference) — no transforms
            self._spatial_tf  = None
            self._intensity_tf = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.case_ids) * self.patches_per_volume

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx % len(self.case_ids)]

        with np.load(self.root / f"{case_id}.npz") as z:
            data = np.asarray(z["data"], dtype=np.float32)   # (4, D, H, W)
            seg  = np.asarray(z["seg"],  dtype=np.int16)     # (D, H, W)

        # Stack A and B channels in one array so spatial transforms are
        # applied identically to both (required for registration).
        all_ch = self.channels_A + self.channels_B
        combined = data[all_ch]                               # (n_A+n_B, D, H, W)

        image = torch.from_numpy(combined).to(self.dtype)
        segmentation = torch.from_numpy(seg).unsqueeze(0)    # (1, D, H, W)

        # 1. Spatial: rotation + scaling + crop — applied to ALL channels + seg
        if self._spatial_tf is not None:
            sample = self._spatial_tf(image=image, segmentation=segmentation)
            image = sample["image"]
            segmentation = sample["segmentation"]

        # 2. Intensity: noise / blur / brightness / gamma — applied to A only
        if self._intensity_tf is not None:
            aug = self._intensity_tf(image=image[: self.n_A])
            image = torch.cat([aug["image"], image[self.n_A :]], dim=0)

        A   = image[: self.n_A]    # (n_A, D, H, W)
        B   = image[self.n_A :]    # (n_B, D, H, W)

        # 3. Modality dropout (train only): with prob `modality_dropout`, zero
        # one input channel so the model learns to cope with a missing input.
        if (
            self.augment
            and self.modality_dropout > 0.0
            and self.n_A > 1
            and torch.rand(1).item() < self.modality_dropout
        ):
            drop = int(torch.randint(self.n_A, (1,)).item())
            A = A.clone()
            A[drop] = 0.0

        return {
            "A":       A,
            "B":       B,
            "seg":     segmentation,   # (1, D, H, W), int16, labels 0-3
            "A_paths": [case_id],
        }

    # ------------------------------------------------------------------
    def load_props(self, case_id: str) -> dict:
        """Return the sidecar JSON with per-case preprocessing metadata."""
        with open(self.root / f"{case_id}.json") as f:
            return json.load(f)

    def input_channel_names(self) -> list[str]:
        """Human-readable names of the input (A) channels."""
        return [CHANNEL_NAMES[c] for c in self.channels_A]

    def target_channel_names(self) -> list[str]:
        """Human-readable names of the target (B) channels."""
        return [CHANNEL_NAMES[c] for c in self.channels_B]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={len(self.case_ids)}, "
            f"A={self.input_channel_names()} → B={self.target_channel_names()}, "
            f"patch={self.patch_size}, augment={self.augment}, "
            f"patches_per_volume={self.patches_per_volume})"
        )
