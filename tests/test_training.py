from collections.abc import Callable
from pathlib import Path

from tokenizer import (
    Corpus,
    Vocabulary,
    decode,
    encode,
    training_loop_baseline,
    training_loop_optimized,
)


def _write_corpus(path: Path, text: str) -> Corpus:
    path.write_text(text, encoding="utf-8")
    return Corpus(path)


def _train_both(
    tmp_path: Path,
    text: str,
    stop_when: Callable[[int, int], bool],
) -> tuple[Vocabulary, Vocabulary]:
    baseline = training_loop_baseline(
        [_write_corpus(tmp_path / "baseline.txt", text)],
        stop_when,
    )
    optimized = training_loop_optimized(
        [_write_corpus(tmp_path / "optimized.txt", text)],
        stop_when,
    )
    return baseline, optimized


def test_corpus_yields_lines_once(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")
    corpus = Corpus(path)

    assert list(corpus) == ["hello\n", "world\n"]
    assert list(corpus) == []


def test_training_loop_adds_one_merge(tmp_path: Path):
    text = "abababababab\n"
    baseline, optimized = _train_both(tmp_path, text, lambda _count, term: term >= 1)

    for vocabulary in (baseline, optimized):
        assert len(vocabulary) == 257
        assert vocabulary.get_token_data(256) == b"ab"
        assert encode(vocabulary, "ab") == [256]


def test_training_loop_applies_merges_left_to_right(tmp_path: Path):
    text = "aaa\n"
    baseline, optimized = _train_both(tmp_path, text, lambda _count, _term: False)

    for vocabulary in (baseline, optimized):
        assert vocabulary.get_token_data(256) == b"aa"
        assert encode(vocabulary, "aaa") == [256, ord("a")]


def test_training_loop_stops_when_no_pair_repeats(tmp_path: Path):
    text = "abcde\n"
    baseline, optimized = _train_both(tmp_path, text, lambda _count, _term: False)

    assert len(baseline) == 256
    assert len(optimized) == 256
    assert baseline.token_to_data == optimized.token_to_data


def test_training_loops_agree_on_repeated_pairs(tmp_path: Path):
    text = "abababababab\n"
    baseline, optimized = _train_both(tmp_path, text, lambda _count, term: term >= 2)

    assert baseline.token_to_data == optimized.token_to_data
    assert baseline.get_token_data(256) == b"ab"
    assert baseline.get_token_data(257) == b"abab"
    assert encode(baseline, "abab") == [257]


def test_training_loop_tie_breaks_by_smaller_pair(tmp_path: Path):
    text = "abab cdcd abab cdcd abab cdcd\n"
    baseline, optimized = _train_both(tmp_path, text, lambda _count, term: term >= 1)

    assert baseline.get_token_data(256) == b"ab"
    assert optimized.get_token_data(256) == b"ab"


def test_training_loop_stop_when_receives_count_and_term(tmp_path: Path):
    calls: list[tuple[int, int]] = []

    def stop_when(count: int, term: int) -> bool:
        calls.append((count, term))
        return term >= 1

    training_loop_baseline(
        [_write_corpus(tmp_path / "corpus.txt", "abababababab\n")],
        stop_when,
    )

    assert calls == [(6, 1)]


def test_training_loop_accepts_multiple_corpora(tmp_path: Path):
    corpora = [
        _write_corpus(tmp_path / "a.txt", "abababab\n"),
        _write_corpus(tmp_path / "b.txt", "abababab\n"),
    ]

    vocabulary = training_loop_optimized(corpora, lambda _count, term: term >= 1)

    assert vocabulary.get_token_data(256) == b"ab"
    assert len(vocabulary) == 257


def test_trained_vocabulary_roundtrip(tmp_path: Path):
    text = "Héllò héllò héllò\n"
    vocabulary, _ = _train_both(tmp_path, text, lambda _count, term: term >= 3)
    sample = "Héllò héllò"

    assert decode(vocabulary, encode(vocabulary, sample)) == sample
    assert len(encode(vocabulary, sample)) < len(sample.encode("utf-8"))

def test_training_excludes_special_tokens_from_bpe_merges(tmp_path: Path):
    special_token = "<|endoftext|>"
    corpus = _write_corpus(
        tmp_path / "corpus.txt",
        f"ab{special_token}ab\n",
    )

    vocabulary = training_loop_baseline(
        [corpus],
        lambda _count, term: term >= 1,
        [special_token],
    )

    special_id = vocabulary.get_token(special_token.encode("utf-8"))
    ab_id = vocabulary.get_token(b"ab")
    assert special_id >= 256
    assert ab_id >= 256
    assert encode(vocabulary, f"ab{special_token}ab", [special_token]) == [
        ab_id,
        special_id,
        ab_id,
    ]
def test_optimized_training_preserves_special_tokens(tmp_path: Path):
    special_token = "<|endoftext|>"
    corpus = _write_corpus(
        tmp_path / "corpus.txt",
        f"hello{special_token}world\n",
    )

    vocabulary = training_loop_optimized([corpus], special_tokens=[special_token])

    assert vocabulary.get_token(special_token.encode("utf-8")) >= 256
    assert decode(
        vocabulary,
        encode(vocabulary, f"hello{special_token}world", [special_token]),
    ) == f"hello{special_token}world"
