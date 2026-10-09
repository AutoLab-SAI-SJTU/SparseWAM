"""Temporal cross-attention fusion of multi-frame DINO tokens into current.

All condition frames (history + current) form the KV memory; current tokens
are the queries. Each KV/Q token gets:

- sinusoidal **time** embed from relative frame offset (e.g. ``-4..0``)
- learned **slot** embed from register-slot index ``0..K-1``

so the attention can tell *when* and *which query slot* a token belongs to.

Layout::

    frames:     [B, T, K, D]   # T = T_h + 1, last frame is current
    Q:          [B, K, D]      # current + slot embed
    KV:         [B, T*K, D]    # all frames + time embed + slot embed
    output:     [B, K, D]      # current + CrossAttn + FFN (near-identity init)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dit import sinusoidal_embedding_1d


class HistoryTemporalFusion(nn.Module):
    """Fuse multi-frame DINO tokens into current via temporal cross-attention.

    Args:
        dim: Token channel dim ``D`` (DrivoR neck / ``tf_d_model``).
        num_slots: Register slots per frame ``K`` (``num_cams * num_scene_tokens``).
        num_heads: Cross-attention heads.
        mlp_ratio: FFN hidden = ``dim * mlp_ratio``.
        dropout: Attention / FFN dropout.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}.")
        self.dim = dim
        self.num_slots = num_slots

        # Map sinusoidal time features → token dim.
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Slot identity so flattened T*K tokens remain slot-aligned.
        self.slot_embed = nn.Embedding(num_slots, dim)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        # Near-identity start: residual path dominates until fusion learns.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-2].weight)
        if self.ffn[-2].bias is not None:
            nn.init.zeros_(self.ffn[-2].bias)

    def _slot_emb(self, batch: int, num_frames: int, device: torch.device) -> torch.Tensor:
        """Build slot embeddings broadcast over frames.

        Args:
            batch: Batch size ``B``.
            num_frames: Number of frames ``T`` (1 for Q, T_cond for KV).
            device: Target device.

        Returns:
            Slot encodings ``[B, T, K, D]`` e.g. ``[2, 5, 16, 256]``.
        """
        # slot ids [K] → [1, 1, K, D] → [B, T, K, D]
        slot_ids = torch.arange(self.num_slots, device=device, dtype=torch.long)
        emb = self.slot_embed(slot_ids)  # [K, D]
        return (
            emb.view(1, 1, self.num_slots, self.dim)
            .expand(batch, num_frames, self.num_slots, self.dim)
        )

    def _time_emb(
        self,
        time_ids: torch.Tensor,
        batch: int,
        num_slots: int,
    ) -> torch.Tensor:
        """Build time embeddings broadcast over slots.

        Args:
            time_ids: Relative frame offsets ``[T]`` e.g. ``[-4,-3,-2,-1,0]``.
            batch: Batch size ``B``.
            num_slots: ``K``.

        Returns:
            Time encodings ``[B, T, K, D]``.
        """
        # sinusoidal [T, D] → MLP [T, D] → [1, T, 1, D] → [B, T, K, D]
        t_emb = self.time_mlp(
            sinusoidal_embedding_1d(self.dim, time_ids.float())
        )  # [T, D]
        num_frames = time_ids.numel()
        return (
            t_emb.view(1, num_frames, 1, self.dim)
            .expand(batch, num_frames, num_slots, self.dim)
        )

    def forward(
        self,
        frames: torch.Tensor,
        frame_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse all condition frames into the current-frame tokens.

        Args:
            frames: Condition DINO tokens ``[B, T, K, D]``
                e.g. ``[2, 5, 16, 256]`` for offsets ``[-4,-3,-2,-1,0]``.
                Last frame (``T-1``) is the current observation.
            frame_time_ids: Relative offsets ``[T]`` matching ``frames``
                e.g. ``[-4., -3., -2., -1., 0.]``.

        Returns:
            Fused current tokens ``[B, K, D]`` e.g. ``[2, 16, 256]``.
        """
        if frames.dim() != 4:
            raise ValueError(
                f"frames must be [B,T,K,D], got {tuple(frames.shape)}"
            )
        batch, num_frames, num_slots, dim = frames.shape
        if num_slots != self.num_slots:
            raise ValueError(
                f"frames K={num_slots} != num_slots={self.num_slots}"
            )
        if dim != self.dim:
            raise ValueError(f"frames D={dim} != dim={self.dim}")
        if frame_time_ids.numel() != num_frames:
            raise ValueError(
                f"frame_time_ids length {frame_time_ids.numel()} != T={num_frames}"
            )

        device = frames.device
        dtype = frames.dtype
        time_ids = frame_time_ids.to(device=device)

        # Current frame as residual base / query content: [B, K, D]
        current = frames[:, -1]

        # ---- Q: current + slot embed ----
        # [B, 1, K, D] → squeeze → [B, K, D]
        q_slot = self._slot_emb(batch, 1, device)[:, 0]  # [B, K, D]
        query = self.norm_q(current + q_slot.to(dtype=dtype))

        # ---- KV: all frames (history + current) + time + slot ----
        # [B, T, K, D]
        kv_content = (
            frames
            + self._time_emb(time_ids, batch, num_slots).to(dtype=dtype)
            + self._slot_emb(batch, num_frames, device).to(dtype=dtype)
        )
        # Flatten to [B, T*K, D]
        kv = self.norm_kv(kv_content.reshape(batch, num_frames * num_slots, dim))

        # Cross-attn: Q=current [B,K,D], K/V=all frames [B,T*K,D].
        attn_out, _ = self.cross_attn(query, kv, kv, need_weights=False)
        # Residual on raw current (not slot-augmented) keeps DiT interface clean.
        fused = current + attn_out
        fused = fused + self.ffn(self.norm_ffn(fused))
        return fused
