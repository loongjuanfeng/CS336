import pytest

from tokenizer import Vocabulary


def test_vocabulary_default_maps_each_byte_to_its_id():
    vocabulary = Vocabulary()

    assert len(vocabulary) == 256
    for byte in range(256):
        token_data = bytes([byte])
        assert token_data in vocabulary
        assert vocabulary[token_data] == byte
        assert vocabulary.get_token(token_data) == byte
        assert vocabulary.get_token_data(byte) == token_data


def test_vocabulary_empty_iterable_has_no_tokens():
    vocabulary = Vocabulary([])

    assert len(vocabulary) == 0
    assert list(vocabulary) == []


def test_vocabulary_iterable_assigns_contiguous_ids():
    vocabulary = Vocabulary([b"a", b"b", b"ab"])

    assert list(vocabulary) == [b"a", b"b", b"ab"]
    assert vocabulary.get_token(b"a") == 0
    assert vocabulary.get_token(b"b") == 1
    assert vocabulary.get_token(b"ab") == 2


def test_vocabulary_mapping_preserves_ids():
    vocabulary = Vocabulary({b"a": 5, b"b": 9})

    assert len(vocabulary) == 2
    assert vocabulary.get_token(b"a") == 5
    assert vocabulary.get_token(b"b") == 9
    assert vocabulary.get_token_data(5) == b"a"
    assert vocabulary.add_token(b"c") == 10


def test_vocabulary_add_token_is_idempotent():
    vocabulary = Vocabulary()

    assert vocabulary.add_token(b"a") == 97
    assert len(vocabulary) == 256

    token = vocabulary.add_token(b"ab")
    assert token == 256
    assert vocabulary.add_token(b"ab") == token
    assert len(vocabulary) == 257
    assert vocabulary.get_token_data(token) == b"ab"


def test_vocabulary_add_pair_concatenates_token_data():
    vocabulary = Vocabulary()

    token = vocabulary.add_pair((ord("a"), ord("b")))

    assert token == 256
    assert vocabulary.get_token_data(token) == b"ab"
    assert vocabulary.add_pair((ord("a"), ord("b"))) == token


def test_vocabulary_add_pair_requires_existing_tokens():
    vocabulary = Vocabulary([])

    with pytest.raises(KeyError):
        vocabulary.add_pair((0, 1))


def test_vocabulary_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="token_data id already exists"):
        Vocabulary({b"a": 0, b"b": 0})


def test_vocabulary_rejects_negative_ids():
    with pytest.raises(ValueError, match="non-negative integers"):
        Vocabulary({b"a": -1})


def test_vocabulary_rejects_non_integer_ids():
    with pytest.raises(ValueError, match="non-negative integers"):
        Vocabulary({b"a": 1.5})


def test_vocabulary_get_token_missing_raises_key_error():
    vocabulary = Vocabulary([b"a"])

    with pytest.raises(KeyError):
        vocabulary.get_token(b"missing")
    with pytest.raises(KeyError):
        vocabulary[b"missing"]
    with pytest.raises(KeyError):
        vocabulary.get_token_data(1)


def test_vocabulary_contains_token_data_not_ids():
    vocabulary = Vocabulary()

    assert b"a" in vocabulary
    assert 97 not in vocabulary
    assert b"ab" not in vocabulary
