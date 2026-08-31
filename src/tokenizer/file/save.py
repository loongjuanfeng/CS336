"""Utilities for serializing tokenizer vocabularies."""

import base64
import json
from os import PathLike
from pathlib import Path

from ..vocabulary import Vocabulary

_FORMAT = "cs336-vocabulary"
_VERSION = 1


def save_vocabulary(vocabulary: Vocabulary, path: str | PathLike[str]) -> None:
    """Save *vocabulary* as a versioned JSON file.

    Tokens are stored in ID order and encoded with Base64 because JSON has no
    native bytes type. The list index is the token ID.
    """
    tokens = [
        base64.b64encode(vocabulary.token_to_data[token]).decode("ascii")
        for token in range(len(vocabulary))
    ]
    payload = {
        "format": _FORMAT,
        "version": _VERSION,
        "tokens": tokens,
    }
    Path(path).write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
