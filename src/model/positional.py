"""Rotary positional embeddings."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Apply cached rotary rotations to query or key vectors.

    The cache stores one cosine and sine value for every position and every
    pair of channels.  Position tensors may omit batch-like axes; those axes
    are inserted in the natural (left-to-right) location so a position tensor
    shaped ``(batch, sequence)`` broadcasts correctly over attention heads.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        half_dimension = d_k // 2
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        frequencies = torch.arange(half_dimension, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-2.0 * frequencies / d_k)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        self.max_seq_len = max_seq_len
        self.d_k = d_k

    def _align_positions(
        self, token_positions: Tensor, leading_shape: tuple[int, ...]
    ) -> Tensor:
        """Broadcast position axes to the leading axes of an input tensor."""
        position_leading = token_positions.shape[:-1]
        if not position_leading:
            return token_positions.reshape(
                (1,) * len(leading_shape) + token_positions.shape[-1:]
            )

        prefix_fits = len(position_leading) <= len(leading_shape) and all(
            position_size in (1, input_size)
            for position_size, input_size in zip(
                position_leading, leading_shape, strict=False
            )
        )
        if prefix_fits:
            shape = (
                position_leading
                + (1,) * (len(leading_shape) - len(position_leading))
                + token_positions.shape[-1:]
            )
            return token_positions.reshape(shape)

        suffix_fits = len(position_leading) <= len(leading_shape) and all(
            position_size in (1, input_size)
            for position_size, input_size in zip(
                position_leading, leading_shape[-len(position_leading) :], strict=False
            )
        )
        if suffix_fits:
            shape = (
                (1,) * (len(leading_shape) - len(position_leading))
                + position_leading
                + token_positions.shape[-1:]
            )
            return token_positions.reshape(shape)

        shape = (1,) * (
            len(leading_shape) - len(position_leading)
        ) + token_positions.shape
        return token_positions.reshape(shape)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Rotate ``x`` at the positions in ``token_positions``."""
        leading_shape = tuple(x.shape[:-2])
        aligned_positions = self._align_positions(token_positions, leading_shape)
        cosine_cache = self.get_buffer("cos")
        sine_cache = self.get_buffer("sin")
        aligned_positions = aligned_positions.to(device=cosine_cache.device)
        cos = cosine_cache[aligned_positions].to(dtype=x.dtype, device=x.device)
        sin = sine_cache[aligned_positions].to(dtype=x.dtype, device=x.device)

        pairs = rearrange(
            x,
            "... (pairs components) -> ... pairs components",
            pairs=self.d_k // 2,
            components=2,
        )
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rearrange(rotated, "... pairs components -> ... (pairs components)")
