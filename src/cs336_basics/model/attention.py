"""Attention modules for the decoder-only Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .positional import RotaryPositionalEmbedding
from .primitives import Linear, scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional rotary embeddings."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Return causal attention outputs for ``(..., sequence, d_model)``."""
        sequence_length = x.shape[-2]
        q, k, v = (
            projection(x)
            .unflatten(-1, (self.num_heads, self.head_dim))
            .transpose(-3, -2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            device=x.device,
            dtype=torch.bool,
        ).tril()
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        attended = attended.transpose(-3, -2).flatten(-2)
        return self.output_proj(attended)
