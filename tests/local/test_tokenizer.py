import pytest

from cs336_basics.tokenizer.core import Tokenizer
from cs336_basics.tokenizer.pre_tokenize import (
    compile_special_pattern,
    iter_pre_tokens,
    pre_tokenize_lines,
    pre_tokenize_text,
    split_special_tokens,
)


def _tokenizer(merges=(), special_tokens=()):
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for pair in merges:
        vocab[len(vocab)] = b"".join(pair)
    return Tokenizer(vocab, list(merges), list(special_tokens))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "s",
        "Hello, how are you?",
        "Héllò hôw are ü? 🙃",
        "I'm a tester.\n\nNew paragraph.",
        "hello  world",
        "hello \nworld",
    ],
)
def test_encode_decode_roundtrip(text):
    tokenizer = _tokenizer()
    assert tokenizer.encode(text) == list(text.encode("utf-8"))
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize(
    "merges, text, expected",
    [
        ([(b"a", b"b"), (b"b", b"c")], "abc", [256, ord("c")]),
        ([(b"b", b"c"), (b"a", b"b")], "abc", [ord("a"), 256]),
        ([(b"a", b"b"), (b"ab", b"ab")], "abab", [257]),
        ([(b"a", b"b"), (b"ab", b" ")], "ab ab", [256, ord(" "), 256]),
    ],
)
def test_encode_applies_ranked_merges_within_pre_tokens(merges, text, expected):
    assert _tokenizer(merges).encode(text) == expected


def test_encoding_does_not_modify_vocabulary():
    tokenizer = _tokenizer([(b"a", b"b")], ["<end>"])
    before = dict(tokenizer.vocab)
    text = "abab<end>ab"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert list(tokenizer.encode_iterable([text])) == tokenizer.encode(text)
    assert tokenizer.vocab == before


def test_missing_tokens_raise_key_error():
    with pytest.raises(KeyError):
        Tokenizer({0: b"a"}, []).encode("b")
    with pytest.raises(KeyError):
        _tokenizer().decode([256])


def test_decode_invalid_utf8_uses_replacement_character():
    assert _tokenizer().decode([255]) == "\ufffd"


def test_special_tokens_are_atomic_and_use_longest_match():
    tokenizer = _tokenizer([(b"a", b"b")], ["<end>", "<end><end>"])
    assert tokenizer.encode("ab<end>ab") == [256, 257, 256]
    assert tokenizer.encode("<end><end>") == [258]
    assert tokenizer.decode(tokenizer.encode("hello<end>world")) == "hello<end>world"
    undeclared = Tokenizer(tokenizer.vocab, tokenizer.merges)
    assert undeclared.encode("<end>") != [257]
    assert undeclared.decode(undeclared.encode("<end>")) == "<end>"


def test_special_tokens_are_added_without_mutating_input_vocab():
    vocab = {0: b"a"}
    tokenizer = Tokenizer(vocab, [], ["<end>", "<end>"])
    assert tokenizer.encode("a<end>") == [0, 1]
    assert vocab == {0: b"a"}
    assert tokenizer.special_tokens == ["<end>"]
    with pytest.raises(ValueError, match="non-empty"):
        Tokenizer(vocab, [], [""])


def test_streaming_handles_partial_special_tokens():
    tokenizer = _tokenizer(special_tokens=["<end>"])
    assert list(tokenizer.encode_iterable(["a<en", "d", ">b"])) == [
        ord("a"),
        256,
        ord("b"),
    ]
    assert tokenizer.decode(tokenizer.encode_iterable(["a<en"])) == "a<en"


def test_pre_tokenize_text_uses_gpt2_pattern():
    assert list(pre_tokenize_text("Hello, how are you?")) == [
        b"Hello",
        b",",
        b" how",
        b" are",
        b" you",
        b"?",
    ]
    assert list(pre_tokenize_text("I'm")) == [b"I", b"'m"]
    assert list(pre_tokenize_text("don't")) == [b"don", b"'t"]
    assert list(pre_tokenize_text("hello  world")) == [b"hello", b" ", b" world"]
    assert list(pre_tokenize_text("hello 123")) == [b"hello", b" 123"]
    assert list(pre_tokenize_text("hello  ")) == [b"hello", b"  "]


def test_pre_tokenize_lines_keeps_trailing_line_whitespace_together():
    text = "hello \nworld"

    assert list(pre_tokenize_text(text)) == [b"hello", b" ", b"\n", b"world"]
    assert list(pre_tokenize_lines(["hello \n", "world"])) == [
        b"hello",
        b" \n",
        b"world",
    ]


@pytest.mark.parametrize("specials", [[], ["<end>", "<end><end>", "<x\ny>"]])
@pytest.mark.parametrize(
    "text",
    [
        "",
        "I'm don't we'll they're I'd hello  \n\n world\t ",
        "Héllò 世界 🙃 123!!!\n",
        "a<end><end>b<x\ny>c<en",
        "<end><end><end>",
    ],
)
def test_streaming_pre_tokens_match_whole_text_at_every_boundary(text, specials):
    expected = []
    for segment, is_special in split_special_tokens(
        text, compile_special_pattern(specials)
    ):
        if is_special:
            expected.append((segment.encode("utf-8"), True))
        else:
            expected.extend((token, False) for token in pre_tokenize_text(segment))
    for boundary in range(len(text) + 1):
        assert (
            list(iter_pre_tokens([text[:boundary], "", text[boundary:]], specials))
            == expected
        )
    assert list(iter_pre_tokens(iter(text), specials)) == expected
