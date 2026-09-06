from pathlib import Path

import pytest

from cs336_basics.tokenizer.core import Tokenizer
from cs336_basics.tokenizer.training import train_bpe


@pytest.mark.parametrize(
    "text, vocab_size, expected_merges",
    [
        ("abababababab\n", 257, [(b"a", b"b")]),
        ("aaa\n", 257, [(b"a", b"a")]),
        (
            "abcde\n",
            1000,
            [(b"d", b"e"), (b"c", b"de"), (b"b", b"cde"), (b"a", b"bcde")],
        ),
        ("abababababab\n", 258, [(b"a", b"b"), (b"ab", b"ab")]),
        ("abab cdcd abab cdcd abab cdcd\n", 257, [(b"c", b"d")]),
        ("", 300, []),
        ("banana", 256, []),
    ],
)
def test_training_merge_order(tmp_path: Path, text, vocab_size, expected_merges):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    vocab, merges = train_bpe(path, vocab_size, [])
    assert merges == expected_merges
    assert len(vocab) == 256 + len(merges)
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    if text == "aaa\n":
        assert tokenizer.encode("aaa") == [256, ord("a")]


def test_training_preserves_special_tokens(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text("ab<end>ab\n", encoding="utf-8")
    vocab, merges = train_bpe(path, 258, ["<end>", "<end>"])
    assert vocab[256] == b"<end>"
    assert merges == [(b"a", b"b")]
    tokenizer = Tokenizer(vocab, merges, ["<end>"])
    assert tokenizer.encode("ab<end>ab") == [257, 256, 257]


def test_training_compresses_unicode(tmp_path: Path):
    text = "Héllò héllò héllò\n"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    tokenizer = Tokenizer(*train_bpe(path, 259, []))
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


@pytest.mark.parametrize(
    "vocab_size, special_tokens", [(255, []), (256, ["<end>"]), (257, [""])]
)
def test_training_rejects_invalid_configuration(tmp_path, vocab_size, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_text("banana", encoding="utf-8")
    with pytest.raises(ValueError):
        train_bpe(path, vocab_size, special_tokens)
