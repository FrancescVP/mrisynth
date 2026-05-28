"""3-D Diffusion Transformer (DiT) for latent velocity prediction.

Implements the adaLN-Zero DiT architecture (Peebles & Xie, 2022) adapted for
3-D latent volumes.  Drop-in replacement for DiffusionModelUNet in RFlowModel.

Interface
---------
  model = DiT3D(in_channels=12, out_channels=4, ...)
  v = model(x, timesteps)   # same signature as DiffusionModelUNet

Architecture
------------
  1. Patchify : Conv3d(in_ch, hidden, ks=patch_size, stride=patch_size)
  2. Add factorized per-axis positional embeddings (D + H + W, interpolated for
     variable spatial size — correct for any latent shape, no cube-root assumption)
  3. N × DiTBlock (LayerNorm + SDPA + adaLN-Zero gating on timestep emb)
  4. Unpatchify : Linear(hidden, patch_size^3 * out_ch) + reshape

Timestep conditioning via sinusoidal embedding → 2-layer MLP → adaLN-Zero.
Self-attention uses torch.nn.functional.scaled_dot_product_attention (flash
attention when CUDA is available).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class _SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding, output shape (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([x.cos(), x.sin()], dim=-1)       # (B, dim)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

class _DiTBlock(nn.Module):
    """Single DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv       = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj      = nn.Linear(hidden_size, hidden_size)

        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

        # adaLN-Zero: 6 modulation params (shift/scale/gate × 2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: (B, hidden_size)
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)

        # Attention
        x_norm = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        B, N, _ = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                        # each (B, N, H, D)
        q = q.transpose(1, 2)                              # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)     # flash attn when available
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        x = x + g1.unsqueeze(1) * self.proj(attn)

        # FFN
        x_norm = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(x_norm)
        return x


# ---------------------------------------------------------------------------
# DiT3D
# ---------------------------------------------------------------------------

class DiT3D(nn.Module):
    """3-D Diffusion Transformer for latent velocity prediction.

    Args:
        in_channels:     Input channels (noisy_target + conditioning latents).
        out_channels:    Output channels (velocity prediction = latent channels).
        patch_size:      Spatial patch size (applied equally in D, H, W). Default 2.
        hidden_size:     Transformer hidden dimension.
        depth:           Number of DiTBlock layers.
        num_heads:       Attention heads (hidden_size must be divisible by num_heads).
        mlp_ratio:       FFN expansion ratio.
        max_spatial_dim: Stored size per axis for factorized pos embeddings.
                         Interpolated at runtime for larger volumes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        max_spatial_dim: int = 64,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.patch_size   = patch_size
        self.out_channels = out_channels

        # Patchify
        self.patch_embed = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            _SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Factorized per-axis positional embeddings — correct for any spatial shape.
        # Each axis stores max_spatial_dim entries; 1D-interpolated at runtime.
        self._max_sdim  = max_spatial_dim
        self.pos_embed_d = nn.Parameter(torch.zeros(1, max_spatial_dim, hidden_size))
        self.pos_embed_h = nn.Parameter(torch.zeros(1, max_spatial_dim, hidden_size))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, max_spatial_dim, hidden_size))
        for emb in (self.pos_embed_d, self.pos_embed_h, self.pos_embed_w):
            nn.init.normal_(emb, std=0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            _DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Output head — zero-init so model starts as identity
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out   = nn.Linear(hidden_size, patch_size ** 3 * out_channels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    # ------------------------------------------------------------------
    def _interp_1d(self, emb: torch.Tensor, size: int) -> torch.Tensor:
        """1-D linear interpolation of a (1, S, C) embedding to length `size`."""
        if emb.shape[1] == size:
            return emb
        return F.interpolate(
            emb.permute(0, 2, 1).float(), size=size, mode="linear", align_corners=False
        ).permute(0, 2, 1).to(emb.dtype)

    def _get_pos_embed(self, D: int, H: int, W: int) -> torch.Tensor:
        """Factorized positional embedding for a D×H×W token grid, shape (1, N, C).

        Each axis embedding is independently 1-D interpolated then summed.
        Works for any spatial size — no cube-root assumption required.
        """
        pe_d = self._interp_1d(self.pos_embed_d, D)   # (1, D, C)
        pe_h = self._interp_1d(self.pos_embed_h, H)   # (1, H, C)
        pe_w = self._interp_1d(self.pos_embed_w, W)   # (1, W, C)
        # Broadcast-sum over D×H×W grid
        pe = (pe_d.unsqueeze(2).unsqueeze(3) +         # (1, D, 1, 1, C)
              pe_h.unsqueeze(1).unsqueeze(3) +         # (1, 1, H, 1, C)
              pe_w.unsqueeze(1).unsqueeze(2))           # (1, 1, 1, W, C)
        return pe.reshape(1, D * H * W, pe.shape[-1])  # (1, N, C)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size

        # Patchify → tokens
        tok = self.patch_embed(x)                    # (B, C, D', H', W')
        D_, H_, W_ = tok.shape[2:]
        tok = tok.flatten(2).transpose(1, 2)         # (B, N, C)
        tok = tok + self._get_pos_embed(D_, H_, W_).to(tok.dtype)

        # Timestep conditioning
        c = self.time_embed(timesteps)               # (B, hidden)

        # Transformer blocks
        for block in self.blocks:
            tok = block(tok, c)

        tok = self.norm_final(tok)

        # Unpatchify
        tok = self.proj_out(tok)                     # (B, N, p^3 * out_ch)
        out = rearrange(
            tok,
            "b (dp hp wp) (p1 p2 p3 c) -> b c (dp p1) (hp p2) (wp p3)",
            dp=D_, hp=H_, wp=W_, p1=p, p2=p, p3=p, c=self.out_channels,
        )
        return out
