"""Neural-network primitives used by the decoder-only language model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """A bias-free affine transformation.

    The weight is stored in the same orientation as :class:`torch.nn.Linear`:
    ``(out_features, in_features)``.  Keeping the parameter in that
    orientation makes checkpoints interchangeable with the assignment's
    reference implementation while the forward pass remains a plain matrix
    multiplication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        standard_deviation = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=standard_deviation,
            a=-3.0 * standard_deviation,
            b=3.0 * standard_deviation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the transformation to the final dimension of ``x``."""
        return x @ self.weight.T


class Embedding(nn.Module):
    """A token embedding lookup with assignment-specified initialization."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return the embedding vector for every token id."""
        return self.weight[input_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` in float32 and restore its original dtype."""
        input_dtype = x.dtype
        normalized = x.to(torch.float32)
        mean_square = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight).to(input_dtype)


def silu(x: Tensor) -> Tensor:
    """Apply the numerically stable SiLU/Swish activation."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """The bias-free SwiGLU position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``w2(silu(w1(x)) * w3(x))``."""
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: Tensor, dim: int) -> Tensor:
    """Compute a numerically stable softmax along ``dim``."""
    input_dtype = x.dtype
    working = (
        x.to(torch.float32) if input_dtype in (torch.float16, torch.bfloat16) else x
    )
    shifted = working - working.amax(dim=dim, keepdim=True)
    probabilities = shifted.exp()
    probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True)
    return probabilities.to(input_dtype)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute masked scaled dot-product attention for arbitrary batch axes."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v
