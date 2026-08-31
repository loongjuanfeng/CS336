import pytest

from tokenizer import Vocabulary, decode, encode
from tokenizer.pre_tokenize import pre_tokenize_lines, pre_tokenize_text


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
def test_encode_decode_roundtrip(text: str):
    vocabulary = Vocabulary()

    assert decode(vocabulary, encode(vocabulary, text)) == text


def test_encode_empty_string_returns_no_tokens():
    assert encode(Vocabulary(), "") == []


def test_decode_empty_tokens_returns_empty_string():
    assert decode(Vocabulary(), []) == ""


def test_encode_with_byte_vocabulary_returns_utf8_bytes():
    vocabulary = Vocabulary()
    text = "Héllò 🙃"

    assert encode(vocabulary, text) == list(text.encode("utf-8"))


def test_encode_applies_earliest_merge():
    vocabulary = Vocabulary()
    ab = vocabulary.add_token(b"ab")
    vocabulary.add_token(b"bc")

    assert encode(vocabulary, "abc") == [ab, ord("c")]


def test_encode_prefers_smaller_token_id_when_merges_overlap():
    vocabulary = Vocabulary()
    bc = vocabulary.add_token(b"bc")
    vocabulary.add_token(b"ab")

    assert encode(vocabulary, "abc") == [ord("a"), bc]


def test_encode_applies_merges_until_none_remain():
    vocabulary = Vocabulary()
    ab = vocabulary.add_token(b"ab")
    abab = vocabulary.add_token(b"abab")

    assert encode(vocabulary, "abab") == [abab]
    assert encode(vocabulary, "ab") == [ab]


def test_encode_does_not_merge_across_pre_tokens():
    vocabulary = Vocabulary()
    ab = vocabulary.add_token(b"ab")

    assert encode(vocabulary, "ab ab") == [ab, ord(" "), ab]


def test_encode_does_not_modify_vocabulary():
    vocabulary = Vocabulary()
    vocabulary.add_token(b"ab")
    before = dict(vocabulary.token_to_data)

    encode(vocabulary, "abababab")

    assert vocabulary.token_to_data == before


def test_decode_does_not_modify_vocabulary():
    vocabulary = Vocabulary()
    before = dict(vocabulary.token_to_data)

    decode(vocabulary, [ord("h"), ord("i")])

    assert vocabulary.token_to_data == before


def test_encode_missing_byte_raises_key_error():
    vocabulary = Vocabulary([b"a"])

    with pytest.raises(KeyError):
        encode(vocabulary, "b")


def test_decode_unknown_token_raises_key_error():
    with pytest.raises(KeyError):
        decode(Vocabulary(), [256])


def test_decode_invalid_utf8_raises_unicode_decode_error():
    with pytest.raises(UnicodeDecodeError):
        decode(Vocabulary(), [0xFF])


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
    assert list(pre_tokenize_lines(["hello \n", "world"])) == [b"hello", b" \n", b"world"]
