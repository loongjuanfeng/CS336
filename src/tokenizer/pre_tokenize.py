"""Regex-based pre-tokenization shared by training and encoding."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

import regex


class _Match(Protocol):
    def group(self) -> str: ...

    def start(self) -> int: ...

    def end(self) -> int: ...


class Pattern(Protocol):
    def finditer(self, text: str) -> Iterator[_Match]: ...


_PRE_TOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize_lines(lines: Iterable[str]) -> Iterator[bytes]:
    """Yield UTF-8 pre-tokens from an iterable of text lines."""
    for line in lines:
        for match in _PRE_TOKEN_PATTERN.finditer(line):
            yield match.group().encode("utf-8")


def pre_tokenize_text(text: str) -> Iterator[bytes]:
    """Yield UTF-8 pre-tokens from a text string."""
    return pre_tokenize_lines((text,))


def compile_special_pattern(
    special_tokens: Sequence[str],
) -> Pattern | None:
    """Compile an alternation matching *special_tokens*, longest first.

    Alternation picks the leftmost match and, at equal start, the branch listed
    first; ordering by descending length therefore resolves overlaps to the
    longest token, so ``<|eot|><|eot|>`` wins over ``<|eot|>``. Returns ``None``
    when there is nothing to match.
    """
    ordered = sorted(
        {token for token in special_tokens if token}, key=len, reverse=True
    )
    if not ordered:
        return None
    return regex.compile("|".join(regex.escape(token) for token in ordered))


def split_special_tokens(
    text: str, pattern: Pattern | None
) -> Iterator[tuple[str, bool]]:
    """Yield ``(segment, is_special)`` pairs covering *text* in order.

    Special segments must bypass :func:`pre_tokenize_text`, which would
    otherwise shatter them into ordinary pre-tokens.
    """
    if pattern is None:
        if text:
            yield text, False
        return

    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            yield text[position : match.start()], False
        yield match.group(), True
        position = match.end()

    if position < len(text):
        yield text[position:], False
