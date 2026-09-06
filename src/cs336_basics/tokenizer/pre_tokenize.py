"""Regex-based pre-tokenization shared by training and encoding."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

import regex


# regex exposes no static Pattern type in this environment.
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


def iter_pre_tokens(
    chunks: Iterable[str], special_tokens: Sequence[str] = ()
) -> Iterator[tuple[bytes, bool]]:
    """Yield (pre-token bytes, is_special) without rescanning stable text.

    Retain regex lookahead and partial special tokens across input chunks.
    Memory scales with the largest chunk or unfinished pre-token.
    """
    special_pattern = compile_special_pattern(special_tokens)
    special_prefixes = {
        token[:i] for token in special_tokens for i in range(1, len(token))
    }
    max_special_length = max(map(len, special_tokens), default=0)
    pending = ""
    for chunk in chunks:
        pending += chunk
        # A suffix may become a special token, including a longer token
        # overlapping an already-complete shorter special token.
        prefix_start = len(pending)
        for length in range(min(len(pending), max_special_length), 0, -1):
            if pending[-length:] in special_prefixes:
                prefix_start = len(pending) - length
                break
        tail_start = 0
        if special_pattern is not None:
            for match in special_pattern.finditer(pending):
                if match.end() > prefix_start:
                    prefix_start = match.start()
                    break
                for token in pre_tokenize_text(pending[tail_start : match.start()]):
                    yield token, False
                yield match.group().encode("utf-8"), True
                tail_start = match.end()
        # Keep the last two regex matches: contractions and the trailing
        # whitespace lookahead can change when the next chunk arrives.
        previous_matches: list[_Match] = []
        for match in _PRE_TOKEN_PATTERN.finditer(pending[tail_start:prefix_start]):
            previous_matches.append(match)
            if len(previous_matches) > 2:
                yield previous_matches.pop(0).group().encode("utf-8"), False
        safe_end = tail_start
        if previous_matches:
            safe_end += previous_matches[0].start()
        pending = pending[safe_end:]
    for segment, is_special in split_special_tokens(pending, special_pattern):
        if is_special:
            yield segment.encode("utf-8"), True
        else:
            for token in pre_tokenize_text(segment):
                yield token, False
