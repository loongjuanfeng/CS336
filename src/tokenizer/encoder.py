"""Encode text with a trained byte-pair vocabulary."""

from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise

from .pre_tokenize import (
    Pattern,
    compile_special_pattern,
    pre_tokenize_text,
    split_special_tokens,
)
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


def _encode_segments(
    vocabulary: Vocabulary,
    text: str,
    special_pattern: Pattern | None,
) -> Iterator[Token]:
    """Encode *text*, mapping each matched special token to a single id."""
    for segment, is_special in split_special_tokens(text, special_pattern):
        if is_special:
            yield vocabulary.get_token(segment.encode("utf-8"))
        else:
            for pre_token in pre_tokenize_text(segment):
                yield from _encode_pre_token(vocabulary, pre_token)


def _proper_prefixes(special_tokens: Sequence[str]) -> frozenset[str]:
    """Return every non-empty proper prefix of *special_tokens*."""
    return frozenset(
        token[:length] for token in special_tokens for length in range(1, len(token))
    )


def _split_pending(
    text: str, prefixes: frozenset[str], longest: int
) -> tuple[str, str]:
    """Split *text* into an encodable head and a held-back tail.

    The tail is the longest suffix that could still grow into a special token
    once the next chunk arrives, and is empty when no suffix can.
    """
    for length in range(min(longest, len(text)), 0, -1):
        if text[-length:] in prefixes:
            return text[:-length], text[-length:]
    return text, ""


def encode(
    vocabulary: Vocabulary, text: str, special_tokens: Sequence[str] = ()
) -> list[Token]:
    """Encode *text* into token IDs without modifying *vocabulary*.

    *text* is split on *special_tokens* first: each match becomes exactly one
    id and never merges with its neighbours. Remaining segments are split with
    the same pre-tokenization pattern used during training, then reduced by
    applying available merges in vocabulary-ID order, which preserves the
    original BPE merge order.
    """
    return list(
        _encode_segments(vocabulary, text, compile_special_pattern(special_tokens))
    )


def encode_iterable(
    vocabulary: Vocabulary,
    chunks: Iterable[str],
    special_tokens: Sequence[str] = (),
) -> Iterator[Token]:
    """Lazily encode a stream of text *chunks*, such as an open file handle.

    Memory stays proportional to one chunk. Chunks are encoded independently,
    matching the per-line behaviour of :func:`pre_tokenize_lines`, except that
    a trailing partial special token is held back until a later chunk can
    complete it.
    """
    pattern = compile_special_pattern(special_tokens)
    prefixes = _proper_prefixes(special_tokens)
    longest = max((len(prefix) for prefix in prefixes), default=0)

    pending = ""
    for chunk in chunks:
        head, pending = _split_pending(pending + chunk, prefixes, longest)
        yield from _encode_segments(vocabulary, head, pattern)

    if pending:
        yield from _encode_segments(vocabulary, pending, pattern)
