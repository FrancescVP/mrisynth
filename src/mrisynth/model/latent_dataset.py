"""Dataset for pre-cached VAE latent files.

Expected layout (produced by scripts/generate_latents.py):

    <root>/
        train/
            <case_id>/
                <case_id>-t1c_z_mu.pt      (4, D', H', W')
                <case_id>-t1c_z_sigma.pt
                <case_id>-t1n_z_mu.pt
                <case_id>-t1n_z_sigma.pt
                <case_id>-t2f_z_mu.pt
                <case_id>-t2f_z_sigma.pt
                <case_id>-t2w_z_mu.pt
                <case_id>-t2w_z_sigma.pt
        val/
            ...

The 'split' subdirectory is optional — if the root contains case folders directly,
that works too.

Returned dict keys
------------------
  "latent_tgt"   (4, D', H', W')  target latent, sampled from q(z|x)
  "latent_cond"  (4*n_cond, D', H', W')  conditioning latent means concatenated
  "mu_tgt"       (4, D', H', W')  target mu (used for val loss without randomness)
  "case_id"      str
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """Load pre-cached VAE latent pairs for RFlow training.

    Args:
        root: directory that contains per-case subdirectories of .pt files.
        case_ids: explicit case ID list; auto-discovered if None.
        deterministic: if True, use mu directly instead of sampling z.
            Use this for validation to get reproducible metrics.
        target_key: modality key for the synthesis target (e.g. "t1c", "t2w", "t2f").
        cond_keys: modality keys for conditioning inputs, concatenated in order.
        modality_dropout: fraction of samples in which exactly one conditioning
            modality is zeroed (chosen uniformly at random), to train robustness
            to a missing input sequence. Requires >1 conditioning modality.
            Disabled (no-op) when ``deterministic=True`` so validation/inference
            always see the full input. ``0.0`` disables it entirely.
    """

    def __init__(
        self,
        root: Union[str, Path],
        case_ids: Optional[Sequence[str]] = None,
        deterministic: bool = False,
        target_key: str = "t1c",
        cond_keys: Optional[Sequence[str]] = None,
        modality_dropout: float = 0.0,
    ):
        self.root = Path(root)
        self.deterministic = deterministic
        self.target_key = target_key
        self.cond_keys = list(cond_keys) if cond_keys is not None else ["t1n", "t2f"]
        self.modality_dropout = float(modality_dropout)

        if case_ids is None:
            case_ids = sorted(
                d.name
                for d in sorted(self.root.iterdir())
                if d.is_dir() and list(d.glob(f"*{self.target_key}_z_mu.pt"))
            )
        self.case_ids = list(case_ids)

        if not self.case_ids:
            raise ValueError(
                f"No latent cases found under {self.root} for target '{self.target_key}'.\n"
                "Run  uv run python scripts/generate_latents.py  first."
            )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        cid = self.case_ids[idx]
        d = self.root / cid

        mu_tgt    = torch.load(d / f"{cid}-{self.target_key}_z_mu.pt",    map_location="cpu", weights_only=True)
        sigma_tgt = torch.load(d / f"{cid}-{self.target_key}_z_sigma.pt", map_location="cpu", weights_only=True)

        if self.deterministic:
            latent_tgt = mu_tgt
        else:
            latent_tgt = mu_tgt + sigma_tgt * torch.randn_like(sigma_tgt)

        seg_path = d / f"{cid}-seg.pt"
        seg = torch.load(seg_path, map_location="cpu", weights_only=True) if seg_path.exists() else None

        cond_parts = [
            torch.load(d / f"{cid}-{k}_z_mu.pt", map_location="cpu", weights_only=True)
            for k in self.cond_keys
        ]
        # Modality dropout (train only): with prob `modality_dropout`, zero one
        # conditioning modality so the model learns to cope with a missing input.
        if (
            not self.deterministic
            and self.modality_dropout > 0.0
            and len(cond_parts) > 1
            and random.random() < self.modality_dropout
        ):
            drop = random.randrange(len(cond_parts))
            cond_parts[drop] = torch.zeros_like(cond_parts[drop])
        latent_cond = torch.cat(cond_parts, dim=0)  # (4 * n_cond, D', H', W')

        return {
            "latent_tgt":  latent_tgt,
            "latent_cond": latent_cond,
            "mu_tgt":      mu_tgt,
            "case_id":     cid,
            "seg":         seg,   # (D, H, W) int8 or None if not available
        }

    def __repr__(self) -> str:
        return (
            f"LatentDataset(root={self.root}, target={self.target_key}, "
            f"conds={self.cond_keys}, n={len(self.case_ids)}, deterministic={self.deterministic})"
        )
