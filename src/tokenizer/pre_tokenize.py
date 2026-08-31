"""Regex-based pre-tokenization shared by training and encoding."""

from collections.abc import Iterable, Iterator

import regex

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
