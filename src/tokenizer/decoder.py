"""Decode token IDs from a byte-pair vocabulary."""

from collections.abc import Iterable

from .types import Token
from .vocabulary import Vocabulary


def decode(vocabulary: Vocabulary, tokens: Iterable[Token]) -> str:
    """Decode *tokens* into UTF-8 text without modifying *vocabulary*."""
    data = b"".join(vocabulary.get_token_data(token) for token in tokens)
    return data.decode("utf-8")
