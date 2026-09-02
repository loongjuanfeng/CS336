import pytest

from tokenizer import Vocabulary, decode, encode, encode_iterable
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
    assert list(pre_tokenize_lines(["hello \n", "world"])) == [
        b"hello",
        b" \n",
        b"world",
    ]


def _vocabulary_with_special(*special_tokens: str) -> Vocabulary:
    vocabulary = Vocabulary()
    for special_token in special_tokens:
        vocabulary.add_token(special_token.encode("utf-8"))
    return vocabulary


def test_encode_special_token_becomes_single_token():
    vocabulary = _vocabulary_with_special("<|endoftext|>")
    end_of_text = vocabulary.get_token(b"<|endoftext|>")

    assert encode(vocabulary, "<|endoftext|>", ["<|endoftext|>"]) == [end_of_text]


def test_encode_special_token_is_split_by_pre_tokenizer_when_not_declared():
    vocabulary = _vocabulary_with_special("<|endoftext|>")

    tokens = encode(vocabulary, "<|endoftext|>")

    assert tokens != [vocabulary.get_token(b"<|endoftext|>")]
    assert decode(vocabulary, tokens) == "<|endoftext|>"


def test_encode_special_token_does_not_merge_with_neighbours():
    vocabulary = _vocabulary_with_special("<|endoftext|>")
    end_of_text = vocabulary.get_token(b"<|endoftext|>")
    ab = vocabulary.add_token(b"ab")

    tokens = encode(vocabulary, "ab<|endoftext|>ab", ["<|endoftext|>"])

    assert tokens == [ab, end_of_text, ab]


def test_encode_prefers_longest_overlapping_special_token():
    vocabulary = _vocabulary_with_special("<|eot|>", "<|eot|><|eot|>")
    doubled = vocabulary.get_token(b"<|eot|><|eot|>")

    tokens = encode(vocabulary, "<|eot|><|eot|>", ["<|eot|>", "<|eot|><|eot|>"])

    assert tokens == [doubled]


def test_encode_special_tokens_roundtrip():
    vocabulary = _vocabulary_with_special("<|endoftext|>")
    text = "hello<|endoftext|>world<|endoftext|>"

    tokens = encode(vocabulary, text, ["<|endoftext|>"])

    assert decode(vocabulary, tokens) == text


def test_encode_special_token_missing_from_vocabulary_raises_key_error():
    with pytest.raises(KeyError):
        encode(Vocabulary(), "<|endoftext|>", ["<|endoftext|>"])


def test_encode_iterable_matches_encode_per_line():
    vocabulary = _vocabulary_with_special("<|endoftext|>")
    vocabulary.add_token(b"he")
    lines = ["hello \n", "world<|endoftext|>", "hello"]

    streamed = list(encode_iterable(vocabulary, lines, ["<|endoftext|>"]))
    expected = [
        token for line in lines for token in encode(vocabulary, line, ["<|endoftext|>"])
    ]

    assert streamed == expected


def test_encode_iterable_joins_special_token_split_across_chunks():
    vocabulary = _vocabulary_with_special("<|endoftext|>")
    end_of_text = vocabulary.get_token(b"<|endoftext|>")

    tokens = list(
        encode_iterable(vocabulary, ["a<|end", "oftext", "|>b"], ["<|endoftext|>"])
    )

    assert tokens == [ord("a"), end_of_text, ord("b")]


def test_encode_iterable_flushes_incomplete_special_token_suffix():
    vocabulary = _vocabulary_with_special("<|endoftext|>")

    tokens = list(encode_iterable(vocabulary, ["a<|end"], ["<|endoftext|>"]))

    assert decode(vocabulary, tokens) == "a<|end"


def test_encode_iterable_is_lazy():
    vocabulary = Vocabulary()
    consumed = 0

    def chunks():
        nonlocal consumed
        for chunk in ["ab", "cd", "ef"]:
            consumed += 1
            yield chunk

    stream = encode_iterable(vocabulary, chunks())

    assert next(stream) == ord("a")
    assert consumed == 1


def test_encode_iterable_does_not_modify_vocabulary():
    vocabulary = _vocabulary_with_special("<|endoftext|>")
    before = dict(vocabulary.token_to_data)

    list(encode_iterable(vocabulary, ["ab<|endoftext|>ab"], ["<|endoftext|>"]))

    assert vocabulary.token_to_data == before
