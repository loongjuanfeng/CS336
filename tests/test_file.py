import json

import pytest

from tokenizer import (
    Vocabulary,
    load_text,
    load_vocabulary,
    save_vocabulary,
)


def test_save_and_load_vocabulary_roundtrip(tmp_path):
    vocabulary = Vocabulary()
    vocabulary.add_token(b"ab")
    vocabulary.add_token(b"abab")
    path = tmp_path / "vocabulary.json"

    save_vocabulary(vocabulary, path)
    loaded = load_vocabulary(path)

    assert loaded.token_to_data == vocabulary.token_to_data
    assert loaded.data_to_token == vocabulary.data_to_token


def test_save_vocabulary_writes_versioned_json(tmp_path):
    vocabulary = Vocabulary([b"a", b"b"])
    path = tmp_path / "vocabulary.json"

    save_vocabulary(vocabulary, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format"] == "cs336-vocabulary"
    assert payload["version"] == 1
    assert len(payload["tokens"]) == 2


def test_load_vocabulary_rejects_unsupported_format(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps({"format": "other", "version": 1, "tokens": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported vocabulary format"):
        load_vocabulary(path)


def test_load_vocabulary_rejects_unsupported_version(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps({"format": "cs336-vocabulary", "version": 2, "tokens": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported vocabulary version"):
        load_vocabulary(path)


def test_load_vocabulary_rejects_invalid_tokens(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps(
            {"format": "cs336-vocabulary", "version": 1, "tokens": ["!!!"]},
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid vocabulary tokens"):
        load_vocabulary(path)


def test_load_vocabulary_rejects_missing_tokens(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps({"format": "cs336-vocabulary", "version": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid vocabulary tokens"):
        load_vocabulary(path)


def test_load_text_yields_lines(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")

    assert list(load_text(path)) == ["hello\n", "world\n"]
