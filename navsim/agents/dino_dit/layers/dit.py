"""DiT blocks for DinoDiT.

``DiT`` / ``DiTBlock`` keep the older CDiT layout (self-attn + visual
cross-attn) used by ``dino_dit_action``. MoT DinoDiT uses ``ExpertBlock``
/ ``Expert``: AdaLN → QKV (joint-attn width) → RoPE on Q/K → out-proj +
FFN, with no cross-attention. Video and action experts may have
different ``dim`` but must share ``num_heads`` and ``attn_head_dim``.
AdaLN may be a low-rank bottleneck (``adaln_rank``) so QKV can take a
larger share of the expert parameter budget.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1.0 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding for a 1D scalar index (diffusion AdaLN step).

    Args:
        dim: Embedding dimension (even).
        position: Scalar indices ``[B]`` or ``[N]`` (diffusion t).

    Returns:
        Embeddings ``[B|N, dim]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=position.device, dtype=torch.float32)
        / half
    )
    args = position.float().unsqueeze(-1) * freqs.unsqueeze(0)
    # Always float embeddings; never cast to Long index dtype.
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _rope_cis_1d(
    positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Complex cis for one RoPE axis.

    Args:
        positions: Token coordinates ``[S]`` e.g. ``[8]`` pose ids or
            ``[64]`` frame offsets.
        dim: Even real width of this axis (2 × number of 2D rotation
            planes), e.g. ``64`` (1D) or ``32`` (one half of 2D).
        theta: RoPE base.

    Returns:
        Complex cis ``[S, dim/2]`` e.g. ``[8, 32]`` / ``[64, 16]``.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE axis dim must be positive even, got {dim}.")
    # inv_freq: [dim/2] e.g. [32]
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
            / float(dim)
        )
    )
    # angles: [S, dim/2] e.g. [8, 32]
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.polar(torch.ones_like(angles), angles)


def build_rope_freqs_1d(
    positions: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """1D RoPE frequencies on a shared 2Hz frame-time axis.

    Used for both video tokens (frame offsets, may be negative) and
    action poses (``t = k+1`` so pose 0 is +0.5s).

    Args:
        positions: Time indices ``[S]`` e.g. video ``[64]`` with
            ``{-2,0,4,8}`` or action ``[8]`` with ``1..8``.
        head_dim: Attention head dim, even, e.g. ``64``.
        theta: RoPE base.

    Returns:
        Real cis ``[S, 1, Dh/2, 2]`` e.g. ``[64, 1, 32, 2]`` / ``[8, 1, 32, 2]``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"1D RoPE requires even head_dim, got {head_dim}.")
    cis = _rope_cis_1d(positions, head_dim, theta)  # [Sa, Dh/2]
    # Keep real storage so DDP / checkpoints never see complex buffers.
    return torch.view_as_real(cis).unsqueeze(1)


def build_rope_freqs_2d(
    time_pos: torch.Tensor,
    slot_pos: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """2D RoPE frequencies for DINO tokens: (horizon time, register slot).

    Splits each head: first ``Dh/2`` real dims rotate by time, the rest
    by slot. Camera is not an axis; multi-cam tokens that share
    ``(time, slot)`` get the same rotation.

    Args:
        time_pos: Frame offsets ``[Sv]`` e.g. ``[64]`` with
            ``-3, 0, 4, 8`` (2Hz ticks).
        slot_pos: Register-slot ids ``[Sv]`` e.g. ``[64]`` with
            ``0..15`` repeating per camera / frame.
        head_dim: Attention head dim, divisible by 4, e.g. ``64``.
        theta: RoPE base.

    Returns:
        Real cis ``[Sv, 1, Dh/2, 2]`` e.g. ``[64, 1, 32, 2]``.
    """
    if time_pos.shape != slot_pos.shape:
        raise ValueError(
            "2D RoPE time/slot length mismatch: "
            f"{tuple(time_pos.shape)} vs {tuple(slot_pos.shape)}"
        )
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D RoPE requires head_dim % 4 == 0, got {head_dim}."
        )
    dim_time = head_dim // 2
    dim_slot = head_dim - dim_time
    cis_time = _rope_cis_1d(time_pos, dim_time, theta)  # [Sv, Dh/4]
    cis_slot = _rope_cis_1d(slot_pos, dim_slot, theta)  # [Sv, Dh/4]
    cis = torch.cat([cis_time, cis_slot], dim=-1)  # [Sv, Dh/2]
    return torch.view_as_real(cis).unsqueeze(1)


def rope_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Apply RoPE to Q or K.

    Args:
        x: Projected Q/K ``[B, S, H*Dh]`` e.g. ``[2, 64, 512]``.
        freqs: Real cis ``[S, 1, Dh/2, 2]`` e.g. ``[64, 1, 32, 2]``.
        num_heads: ``H`` e.g. ``8``.

    Returns:
        Rotated tensor ``[B, S, H*Dh]`` e.g. ``[2, 64, 512]``.
    """
    batch, seq_len, _ = x.shape
    if freqs.shape[0] != seq_len:
        raise ValueError(
            f"RoPE seq mismatch: x S={seq_len} vs freqs S={freqs.shape[0]}"
        )
    head_dim = x.shape[-1] // num_heads
    expected = (seq_len, 1, head_dim // 2, 2)
    if tuple(freqs.shape) != expected:
        raise ValueError(
            f"RoPE freqs shape {tuple(freqs.shape)} != {expected}"
        )
    # xh: [B, S, H, Dh] e.g. [2, 64, 8, 64]
    xh = x.view(batch, seq_len, num_heads, head_dim)
    x_complex = torch.view_as_complex(
        xh.float().reshape(batch, seq_len, num_heads, head_dim // 2, 2)
    )
    cis = torch.view_as_complex(freqs.float())  # [S, 1, Dh/2]
    rotated = torch.view_as_real(x_complex * cis).flatten(-2)
    return rotated.reshape(batch, seq_len, -1).to(dtype=x.dtype)


class DiTBlock(nn.Module):
    """One CDiT block: AdaLN → Self-Attn → AdaLN → Cross-Attn → AdaLN → FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        cross_heads = num_heads if cross_num_heads is None else int(cross_num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        if dim % cross_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by cross_num_heads ({cross_heads})."
            )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, cross_heads, dropout=dropout, batch_first=True
        )
        # Near-identity start: residual path dominates until context is useful.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        # 9 vectors: shift/scale/gate for self-attn, cross-attn, and FFN.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """Forward one CDiT block.

        Args:
            x: RF token sequence ``[B, N, D]`` e.g. ``[2, 40, 512]``.
            t_emb: AdaLN condition ``[B, D]`` e.g. ``[2, 512]``
                (diffusion timestep + optional ego embedding).
            context: Visual K/V ``[B, M, D]`` e.g. ``[2, 80, 512]``.
                If None, the cross-attention residual is skipped.
            return_attn: If True, also return head-averaged self-attn
                ``[B, N, N]`` and cross-attn ``[B, N, M]`` (or None).

        Returns:
            Updated tokens ``[B, N, D]``. When ``return_attn=True``, also
            ``(self_attn, cross_attn)``.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t_emb).chunk(9, dim=-1)
        # Broadcast ``[B, D]`` → ``[B, 1, D]``.
        h = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        # need_weights=True forces the math attention path (not fused SDPA).
        h, self_w = self.attn(
            h,
            h,
            h,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        x = x + gate_msa.unsqueeze(1) * h

        cross_w = None
        if context is not None:
            h = modulate(
                self.norm3(x), shift_mca.unsqueeze(1), scale_mca.unsqueeze(1)
            )
            h, cross_w = self.cross_attn(
                h,
                context,
                context,
                need_weights=return_attn,
                average_attn_weights=True,
            )
            x = x + gate_mca.unsqueeze(1) * h

        h = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.ffn(h)
        if return_attn:
            # self_w: [B, N, N]; cross_w: [B, N, M] or None.
            return x, self_w, cross_w
        return x


class DiT(nn.Module):
    """Stack of CDiT blocks.

    Two different "time" signals (do not confuse them):

    1. ``timestep`` / ``t_emb``: **diffusion** noise level for AdaLN
       (how noisy the RF sample is). Shared across all tokens.
    2. ``token_time_pos``: **calendar / horizon** time of each token
       (when a scene query or action pose lives), added onto tokens.

    Optional ``cond_emb`` (e.g. ego status) is added onto ``t_emb`` before
    AdaLN. Visual context is consumed only via per-block cross-attention.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_tokens: int = 128,
        time_freq_dim: int = 256,
        dropout: float = 0.0,
        cross_num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_tokens = max_tokens
        # Diffusion-step embedding → AdaLN (not sequence position).
        self.time_mlp = nn.Sequential(
            nn.Linear(time_freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    cross_num_heads=cross_num_heads,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.time_freq_dim = time_freq_dim

    def forward(
        self,
        tokens: torch.Tensor,
        timestep: torch.Tensor,
        token_time_pos: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor], List[Optional[torch.Tensor]]],
    ]:
        """Run the CDiT stack.

        Args:
            tokens: RF tokens ``[B, N, D]`` e.g. ``[2, 40, 512]``.
            timestep: Diffusion step ``[B]`` e.g. ``[2]`` → AdaLN.
            token_time_pos: Additive horizon-time / role / slot pos
                ``[1, N, D]`` or ``[B, N, D]``. If None, no additive pos.
            context: Visual K/V ``[B, M, D]`` e.g. ``[2, 80, 512]``.
            cond_emb: Extra AdaLN condition ``[B, D]`` (ego status).
            return_attn: If True, also return per-layer self-attn and
                cross-attn maps.

        Returns:
            Denoised tokens ``[B, N, D]``. When ``return_attn=True``, also
            a list of self-attn maps ``[B, N, N]`` and a list of cross-attn
            maps ``[B, N, M]`` (or None per layer if ``context`` is None).
        """
        batch, num_tokens, dim = tokens.shape
        if num_tokens > self.max_tokens:
            raise ValueError(
                f"Sequence length {num_tokens} exceeds max_tokens "
                f"{self.max_tokens}"
            )
        if context is not None:
            if context.dim() != 3:
                raise ValueError(
                    f"context must be [B, M, D], got {tuple(context.shape)}"
                )
            if context.shape[0] != batch or context.shape[2] != dim:
                raise ValueError(
                    "context batch/dim mismatch: "
                    f"tokens {tuple(tokens.shape)} vs context {tuple(context.shape)}"
                )

        # Diffusion noise-level embedding (AdaLN), NOT query/action time.
        t_emb = self.time_mlp(
            sinusoidal_embedding_1d(self.time_freq_dim, timestep)
        )  # [B, D]
        if cond_emb is not None:
            if cond_emb.shape != t_emb.shape:
                raise ValueError(
                    f"cond_emb must be {tuple(t_emb.shape)}, got {tuple(cond_emb.shape)}"
                )
            t_emb = t_emb + cond_emb
        x = tokens if token_time_pos is None else tokens + token_time_pos
        self_maps: List[torch.Tensor] = []
        cross_maps: List[Optional[torch.Tensor]] = []
        for block in self.blocks:
            if return_attn:
                x, self_w, cross_w = block(
                    x, t_emb, context=context, return_attn=True
                )
                self_maps.append(self_w)
                cross_maps.append(cross_w)
            else:
                x = block(x, t_emb, context=context, return_attn=False)
        out = self.final_norm(x)
        if return_attn:
            return out, self_maps, cross_maps
        return out


class ExpertBlock(nn.Module):
    """One MoT expert block: AdaLN self-attn (external mixed QKV) + FFN.

    Q/K/V project ``hidden → n_heads * head_dim`` so video/action can mix
    in a shared joint-attention space. RoPE is applied to Q/K (not V),
    then ``o`` maps back to ``dim``.

    AdaLN produces 6 vectors: shift/scale/gate for attention and FFN.
    ``t_emb`` may be ``[B, D]`` (shared) or ``[B, S, D]`` (per-token).
    When ``adaln_rank > 0`` the modulation MLP is a low-rank bottleneck
    ``Linear(D, r) → Linear(r, 6D)`` instead of a full ``Linear(D, 6D)``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        adaln_rank: int = 0,
    ) -> None:
        super().__init__()
        attn_dim = int(num_heads) * int(attn_head_dim)
        if attn_dim <= 0:
            raise ValueError(
                f"attn_dim must be > 0, got num_heads={num_heads} "
                f"attn_head_dim={attn_head_dim}"
            )
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = attn_dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.q = nn.Linear(dim, attn_dim)
        self.k = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(dim, attn_dim)
        self.o = nn.Linear(attn_dim, dim)
        self.attn_drop = float(dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        rank = int(adaln_rank)
        self.adaln_rank = rank
        # Full AdaLN is Linear(D, 6D) e.g. [512, 3072] = 1.58M.
        # Low-rank r=64: Linear(D, r) then Linear(r, 6D)
        # e.g. [512, 64] + [64, 3072] = 0.23M, no extra nonlinearity.
        if rank <= 0 or rank >= dim:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, rank),
                nn.Linear(rank, 6 * dim),
            )

    def _split_modulation(
        self, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split AdaLN vectors and broadcast to token dim.

        Args:
            t_emb: ``[B, D]`` or ``[B, S, D]`` e.g. ``[2, 512]`` / ``[2, 64, 512]``.

        Returns:
            shift/scale/gate for MSA and MLP, each ``[B, 1, D]`` or ``[B, S, D]``.
        """
        mods = self.adaLN_modulation(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(
            6, dim=-1
        )
        if t_emb.ndim == 2:
            # Shared timestep: ``[B, D]`` → ``[B, 1, D]``.
            shift_msa = shift_msa.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def build_attn_io(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build Q/K/V in joint-attn space plus post-attn AdaLN state.

        Args:
            x: Expert tokens ``[B, S, D]`` e.g. video ``[2, 64, 512]``
                or action ``[2, 8, 256]``.
            t_emb: AdaLN condition ``[B, D]`` or ``[B, S, D]``.
            rope_freqs: Optional RoPE cis ``[S, 1, Dh/2, 2]``. Applied to
                Q and K only (not V).

        Returns:
            q/k/v ``[B, S, D_attn]`` e.g. ``[2, 64, 512]``, residual ``x``,
            and MSA/MLP gates/shifts/scales.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._split_modulation(t_emb)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        if rope_freqs is not None:
            q = rope_apply(q, rope_freqs, self.num_heads)
            k = rope_apply(k, rope_freqs, self.num_heads)
        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def apply_post(
        self,
        residual_x: torch.Tensor,
        mixed_attn: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """Map mixed-attn output back to expert dim and run FFN.

        Args:
            residual_x: Pre-attn tokens ``[B, S, D]``.
            mixed_attn: Joint-attn output ``[B, S, D_attn]``.
            gate_msa / shift_mlp / scale_mlp / gate_mlp: AdaLN tensors.

        Returns:
            Updated tokens ``[B, S, D]``.
        """
        x = residual_x + gate_msa * self.o(mixed_attn)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.ffn(h)


class Expert(nn.Module):
    """Stack of ``ExpertBlock`` plus diffusion-step MLP.

    ``embed_timestep`` maps scheduler ``t`` into AdaLN space. Per-token
    timesteps (clean H vs noisy F) use ``timestep`` of shape ``[B, S]``.
    ``adaln_rank`` is forwarded to every block (0 = full ``Linear(D, 6D)``).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        attn_head_dim: int,
        mlp_ratio: float = 4.0,
        time_freq_dim: int = 256,
        dropout: float = 0.0,
        adaln_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.time_freq_dim = int(time_freq_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [
                ExpertBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn_head_dim=attn_head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    adaln_rank=adaln_rank,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed diffusion timesteps for AdaLN.

        Args:
            timestep: ``[B]`` shared or ``[B, S]`` per-token
                e.g. ``[2]`` / ``[2, 64]``.

        Returns:
            ``[B, D]`` or ``[B, S, D]`` e.g. ``[2, 512]`` / ``[2, 64, 512]``.
        """
        if timestep.ndim == 1:
            return self.time_mlp(
                sinusoidal_embedding_1d(self.time_freq_dim, timestep)
            )
        if timestep.ndim != 2:
            raise ValueError(
                f"`timestep` must be [B] or [B, S], got {tuple(timestep.shape)}"
            )
        batch, seq_len = timestep.shape
        flat = sinusoidal_embedding_1d(
            self.time_freq_dim, timestep.reshape(-1)
        )
        return self.time_mlp(flat).reshape(batch, seq_len, self.dim)
