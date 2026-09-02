"""Compact Transformer components for CS336 assignment 1."""

from .attention import MultiHeadSelfAttention
from .positional import RotaryPositionalEmbedding
from .primitives import (
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from .transformer import TransformerBlock, TransformerLanguageModel

__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLanguageModel",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]
