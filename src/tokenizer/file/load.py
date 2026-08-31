"""Utilities for loading tokenizer text and vocabularies."""

import base64
import json
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

from ..vocabulary import Vocabulary


def load_text(
    path: str | PathLike[str],
    *,
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Lazily yield lines from *path* without loading the file at once."""
    with open(path, "r", encoding=encoding) as file:
        yield from file


def load_vocabulary(path: str | PathLike[str]) -> Vocabulary:
    """Load a vocabulary saved by :func:`save_vocabulary`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if payload.get("format") != "cs336-vocabulary":
        raise ValueError("unsupported vocabulary format")
    if payload.get("version") != 1:
        raise ValueError("unsupported vocabulary version")

    try:
        tokens = [base64.b64decode(token, validate=True) for token in payload["tokens"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid vocabulary tokens") from error

    return Vocabulary(tokens)
