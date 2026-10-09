"""Mixture-of-Transformers mixer for DinoDiT video / action experts.

Each layer: experts emit Q/K/V in a shared joint-attn width, tokens are
concatenated as ``[video | action]``, mixed attention is applied with a
boolean allow-mask (True = attend), then each expert runs its own out-proj
+ FFN. Hidden sizes may differ; ``num_heads`` and ``attn_head_dim`` must
match.

Not SimWAM's isolated mask — callers pass the DinoDiT ``[H|F|A]`` mask.
Position is shared 1D frame-time RoPE on Q/K. Video tokens may carry a
fixed sinusoidal slot PE added before MoT; action has no type embedding.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import Expert


def allow_mask_to_bias(allow: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert True=attend bool mask to additive SDPA bias.

    PyTorch SDPA bool masks treat True as *blocked*; this helper uses
    ``0`` / ``-inf`` instead so the public API can stay True=allow.

    Args:
        allow: Square mask ``[S, S]`` e.g. ``[72, 72]``.
        dtype: Compute dtype for the bias.

    Returns:
        Additive mask ``[S, S]`` with ``0`` (attend) / ``-inf`` (block).
    """
    bias = torch.zeros(allow.shape, device=allow.device, dtype=dtype)
    return bias.masked_fill(~allow, torch.finfo(dtype).min)


class MoT(nn.Module):
    """Layer-wise mixed attention over a video expert and an action expert."""

    def __init__(self, video_expert: Expert, action_expert: Expert) -> None:
        super().__init__()
        if len(video_expert.blocks) != len(action_expert.blocks):
            raise ValueError(
                "Video/action experts must have the same depth: "
                f"{len(video_expert.blocks)} vs {len(action_expert.blocks)}"
            )
        if video_expert.num_heads != action_expert.num_heads:
            raise ValueError(
                "Video/action experts must share num_heads: "
                f"{video_expert.num_heads} vs {action_expert.num_heads}"
            )
        if video_expert.attn_head_dim != action_expert.attn_head_dim:
            raise ValueError(
                "Video/action experts must share attn_head_dim: "
                f"{video_expert.attn_head_dim} vs {action_expert.attn_head_dim}"
            )
        if video_expert.attn_dim != action_expert.attn_dim:
            raise ValueError(
                "Joint attention dim mismatch: "
                f"{video_expert.attn_dim} vs {action_expert.attn_dim}"
            )
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.num_layers = len(video_expert.blocks)
        self.num_heads = video_expert.num_heads
        self.attn_head_dim = video_expert.attn_head_dim
        self.attn_dim = video_expert.attn_dim

    def _mixed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allow_mask: torch.Tensor,
        return_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Joint attention over concatenated expert tokens.

        Args:
            q/k/v: ``[B, S, D_attn]`` e.g. ``[2, 72, 512]``.
            allow_mask: True=attend ``[S, S]`` e.g. ``[72, 72]``.
            return_weights: If True, also return head-averaged softmax
                ``[B, S, S]``.

        Returns:
            Context ``[B, S, D_attn]`` and optional weights.
        """
        batch, seq_len, attn_dim = q.shape
        if attn_dim != self.attn_dim:
            raise ValueError(
                f"Q last dim {attn_dim} != joint attn dim {self.attn_dim}"
            )
        heads = self.num_heads
        head_dim = self.attn_head_dim
        # [B, S, H*Dh] → [B, H, S, Dh]
        qh = q.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        kh = k.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        vh = v.view(batch, seq_len, heads, head_dim).transpose(1, 2)
        bias = allow_mask_to_bias(allow_mask, dtype=q.dtype)
        # SDPA wants ``[S,S]`` or ``[B, H, S, S]``.
        if return_weights:
            scale = head_dim**-0.5
            logits = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            logits = logits + bias.view(1, 1, seq_len, seq_len)
            weights = torch.softmax(logits, dim=-1)
            ctx = torch.matmul(weights, vh)
            avg_w = weights.mean(dim=1)
        else:
            ctx = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=bias, dropout_p=0.0
            )
            avg_w = None
        # [B, H, S, Dh] → [B, S, H*Dh]
        mixed = ctx.transpose(1, 2).contiguous().view(batch, seq_len, attn_dim)
        return mixed, avg_w

    def forward(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_t_emb: torch.Tensor,
        action_t_emb: torch.Tensor,
        allow_mask: torch.Tensor,
        video_rope_freqs: torch.Tensor,
        action_rope_freqs: torch.Tensor,
        return_attn: bool = False,
    ):
        """Run all MoT layers.

        Args:
            video_tokens: ``[B, Sv, Dv]`` e.g. ``[2, 64, 512]`` (H+F).
            action_tokens: ``[B, Sa, Da]`` e.g. ``[2, 8, 256]``.
            video_t_emb: ``[B, Dv]`` or ``[B, Sv, Dv]``.
            action_t_emb: ``[B, Da]`` or ``[B, Sa, Da]``.
            allow_mask: True=attend ``[Sv+Sa, Sv+Sa]`` e.g. ``[72, 72]``.
            video_rope_freqs: 1D frame-time RoPE cis ``[Sv, 1, Dh/2, 2]``
                e.g. ``[64, 1, 32, 2]``.
            action_rope_freqs: 1D frame-time RoPE cis ``[Sa, 1, Dh/2, 2]``
                e.g. ``[8, 1, 32, 2]``.
            return_attn: If True, collect per-layer joint maps.

        Returns:
            ``video_out`` ``[B, Sv, Dv]``, ``action_out`` ``[B, Sa, Da]``.
            When ``return_attn``, also a list of ``[B, S, S]`` maps.
        """
        if allow_mask.ndim != 2 or allow_mask.shape[0] != allow_mask.shape[1]:
            raise ValueError(
                f"`allow_mask` must be square 2D, got {tuple(allow_mask.shape)}"
            )
        video_seq = int(video_tokens.shape[1])
        action_seq = int(action_tokens.shape[1])
        total = video_seq + action_seq
        if allow_mask.shape[0] != total:
            raise ValueError(
                f"Mask length {allow_mask.shape[0]} != video+action {total}"
            )

        x_v = video_tokens
        x_a = action_tokens
        attn_maps: List[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            v_block = self.video_expert.blocks[layer_idx]
            a_block = self.action_expert.blocks[layer_idx]
            q_v, k_v, v_v, res_v, g_v, sh_v, sc_v, gf_v = v_block.build_attn_io(
                x_v, video_t_emb, rope_freqs=video_rope_freqs
            )
            q_a, k_a, v_a, res_a, g_a, sh_a, sc_a, gf_a = a_block.build_attn_io(
                x_a, action_t_emb, rope_freqs=action_rope_freqs
            )
            q_cat = torch.cat([q_v, q_a], dim=1)
            k_cat = torch.cat([k_v, k_a], dim=1)
            v_cat = torch.cat([v_v, v_a], dim=1)
            mixed, weights = self._mixed_attention(
                q_cat, k_cat, v_cat, allow_mask, return_weights=return_attn
            )
            mixed_v = mixed[:, :video_seq]
            mixed_a = mixed[:, video_seq:]
            x_v = v_block.apply_post(res_v, mixed_v, g_v, sh_v, sc_v, gf_v)
            x_a = a_block.apply_post(res_a, mixed_a, g_a, sh_a, sc_a, gf_a)
            if return_attn:
                if weights is None:
                    raise RuntimeError("return_attn=True but weights is None.")
                attn_maps.append(weights)

        x_v = self.video_expert.final_norm(x_v)
        x_a = self.action_expert.final_norm(x_a)
        if return_attn:
            return x_v, x_a, attn_maps
        return x_v, x_a
