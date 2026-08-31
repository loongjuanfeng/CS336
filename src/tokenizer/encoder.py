"""Encode text with a trained byte-pair vocabulary."""

from itertools import pairwise

from .pre_tokenize import pre_tokenize_text
from .types import Token
from .vocabulary import Vocabulary


def _encode_pre_token(vocabulary: Vocabulary, pre_token: bytes) -> list[Token]:
    """Apply the vocabulary's merges to one pre-tokenized byte string."""
    tokens: list[Token] = [vocabulary.get_token(bytes([byte])) for byte in pre_token]

    while len(tokens) > 1:
        best_index: int | None = None
        best_token: Token | None = None

        for index, (left, right) in enumerate(pairwise(tokens)):
            merged_data = vocabulary.get_token_data(left) + vocabulary.get_token_data(
                right
            )
            merged_token = vocabulary.data_to_token.get(merged_data)
            if merged_token is not None and (
                best_token is None or merged_token < best_token
            ):
                best_index = index
                best_token = merged_token

        if best_index is None or best_token is None:
            break

        tokens[best_index : best_index + 2] = [best_token]

    return tokens


def encode(vocabulary: Vocabulary, text: str) -> list[Token]:
    """Encode *text* into token IDs without modifying *vocabulary*.

    Text is split using the same pre-tokenization pattern used during
    training. Each pre-token is then reduced by applying available merges in
    vocabulary-ID order, which preserves the original BPE merge order.
    """
    return [
        token
        for pre_token in pre_tokenize_text(text)
        for token in _encode_pre_token(vocabulary, pre_token)
    ]
