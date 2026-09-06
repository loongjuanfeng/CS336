import json

import pytest

from cs336_basics.tokenizer.core import Tokenizer


def test_tokenizer_file_roundtrip(tmp_path):
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [], ["<end>"])
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.json"
    tokenizer.save(vocab_path, merges_path)
    restored = Tokenizer.from_files(vocab_path, merges_path)
    assert restored.vocab == tokenizer.vocab
    assert restored.special_tokens == tokenizer.special_tokens
    assert restored.decode(restored.encode("Héllò<end>")) == "Héllò<end>"
    assert json.loads(vocab_path.read_text())["version"] == 1
    assert Tokenizer.from_files(vocab_path, merges_path, []).special_tokens == []


@pytest.mark.parametrize(
    "filename, key, value",
    [
        ("vocab.json", "format", "other"),
        ("vocab.json", "version", 2),
        ("merges.json", "format", "other"),
        ("merges.json", "version", 2),
        ("vocab.json", "vocab", {"0": "!!!"}),
        ("merges.json", "merges", [["!!!", "YQ=="]]),
    ],
)
def test_tokenizer_rejects_invalid_files(tmp_path, filename, key, value):
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.json"
    Tokenizer({0: b"a"}, []).save(vocab_path, merges_path)
    path = tmp_path / filename
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        Tokenizer.from_files(vocab_path, merges_path)
