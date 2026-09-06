"""The assignment tokenizer interface, using explicit ranked byte-pair merges."""

import base64
import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike
from pathlib import Path

from .pre_tokenize import (
    _PRE_TOKEN_PATTERN,
    compile_special_pattern,
    iter_pre_tokens,
    split_special_tokens,
)


class Tokenizer:
    """Byte-level BPE with special tokens and lazy, chunk-safe encoding.

    Streaming retains the unfinished pre-token plus regex lookahead context.
    Memory is bounded by the largest input chunk/unfinished pre-token and a
    fixed-size encoding cache, rather than the total corpus size.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or ()))
        self._ids = {value: key for key, value in self.vocab.items()}
        for token in self.special_tokens:
            if not token:
                raise ValueError("special tokens must be non-empty")
            data = token.encode("utf-8")
            if data not in self._ids:
                index = max(self.vocab, default=-1) + 1
                self.vocab[index] = data
                self._ids[data] = index
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_pattern = compile_special_pattern(self.special_tokens)
        self._encode_word = lru_cache(maxsize=4096)(self._merge_word)

    def _merge_word(self, word: bytes) -> tuple[int, ...]:
        pieces = [bytes([b]) for b in word]
        while len(pieces) > 1:
            best = min(
                range(len(pieces) - 1),
                key=lambda i: self._ranks.get((pieces[i], pieces[i + 1]), float("inf")),
            )
            if (pieces[best], pieces[best + 1]) not in self._ranks:
                break
            pieces[best : best + 2] = [pieces[best] + pieces[best + 1]]
        return tuple(self._ids[piece] for piece in pieces)

    def _encode(self, text: str) -> Iterator[int]:
        for segment, is_special in split_special_tokens(text, self._special_pattern):
            if is_special:
                yield self._ids[segment.encode("utf-8")]
            else:
                for match in _PRE_TOKEN_PATTERN.finditer(segment):
                    yield from self._encode_word(match.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[index] for index in ids).decode(
            "utf-8", errors="replace"
        )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for token, is_special in iter_pre_tokens(iterable, self.special_tokens):
            if is_special:
                yield self._ids[token]
            else:
                yield from self._encode_word(token)

    def save(
        self, vocab_filepath: str | PathLike[str], merges_filepath: str | PathLike[str]
    ) -> None:
        """Save lossless Base64 JSON files, readable by :meth:`from_files`."""

        def encoded(data: bytes) -> str:
            return base64.b64encode(data).decode("ascii")

        Path(vocab_filepath).write_text(
            json.dumps(
                {
                    "format": "cs336-bpe-vocab",
                    "version": 1,
                    "vocab": {str(k): encoded(v) for k, v in self.vocab.items()},
                    "special_tokens": self.special_tokens,
                }
            ),
            encoding="utf-8",
        )
        Path(merges_filepath).write_text(
            json.dumps(
                {
                    "format": "cs336-bpe-merges",
                    "version": 1,
                    "merges": [[encoded(a), encoded(b)] for a, b in self.merges],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Read the Base64 JSON pair written by :meth:`save`."""
        vocab = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        if (vocab.get("format"), vocab.get("version")) != ("cs336-bpe-vocab", 1):
            raise ValueError("unsupported tokenizer vocabulary format")
        if (merges.get("format"), merges.get("version")) != ("cs336-bpe-merges", 1):
            raise ValueError("unsupported tokenizer merges format")
        return cls(
            {
                int(k): base64.b64decode(v, validate=True)
                for k, v in vocab["vocab"].items()
            },
            [
                (base64.b64decode(a, validate=True), base64.b64decode(b, validate=True))
                for a, b in merges["merges"]
            ],
            vocab.get("special_tokens", [])
            if special_tokens is None
            else special_tokens,
        )
