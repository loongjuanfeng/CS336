"""Lazy corpus input for tokenizer training."""

import heapq
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from itertools import chain, pairwise
from os import PathLike

from .file.load import load_text
from .pre_tokenize import pre_tokenize_lines
from .types import *
from .vocabulary import Vocabulary


class Corpus:
    """Thin wrapper around the lazy line loader."""

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        encoding: str = "utf-8",
    ) -> None:
        self.text: Iterator[str] = load_text(path, encoding=encoding)

    def __iter__(self) -> Iterator[str]:
        return self.text


def pre_tokenize(text: Corpus) -> Iterator[bytes]:
    return pre_tokenize_lines(iter(text))


def aggregate_tokens(
    tokens: Iterator[Tokens],
) -> Counter[Tokens]:
    return Counter(tokens)


def merge_pair(original_pair: Pair, new_token: Token, source_tokens: Tokens) -> Tokens:
    merged_tokens: list[Token] = []

    i = 0
    while i < len(source_tokens):
        if i + 1 < len(source_tokens) and source_tokens[i : i + 2] == original_pair:
            merged_tokens.append(new_token)
            i += 2
        else:
            merged_tokens.append(source_tokens[i])
            i += 1

    return tuple(merged_tokens)


def construct_heap(aggregated_tokens: Counter[Tokens]) -> list[tuple[int, Pair]]:
    pair_counts: Counter[Pair] = Counter()
    for tokens, frequency in aggregated_tokens.items():
        for pair in pairwise(tokens):
            pair_counts[pair] += frequency

    heap = [(-count, pair) for (pair, count) in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def get_original_pair(heap: list[tuple[int, Pair]]) -> tuple[Pair, int]:
    _count, original_pair = heapq.heappop(heap)
    count = -_count
    return original_pair, count


def update_aggregated_tokens(
    aggregated_tokens: Counter[Tokens], original_pair: Pair, new_token: Token
) -> Counter[Tokens]:
    updated_aggregated_tokens: Counter[Tokens] = Counter()
    for tokens, frequency in aggregated_tokens.items():
        updated_aggregated_tokens[merge_pair(original_pair, new_token, tokens)] += (
            frequency
        )
    return updated_aggregated_tokens


def training_loop_baseline(
    corpora: list[Corpus],
    stop_when: Callable[[int, int], bool] = lambda count, term: term >= 100,
) -> Vocabulary:
    logger = logging.getLogger("BASELINE")
    logger.setLevel(logging.DEBUG)

    pre_tokens = chain.from_iterable([pre_tokenize(corpus) for corpus in corpora])

    vocabulary = Vocabulary()

    aggregated_tokens = aggregate_tokens(map(tuple, pre_tokens))
    term = 0
    while True:
        term += 1
        heap = construct_heap(aggregated_tokens)

        if not heap:
            break

        pair, count = get_original_pair(heap)

        if count <= 1:
            break

        token = vocabulary.add_pair(pair)
        aggregated_tokens = update_aggregated_tokens(aggregated_tokens, pair, token)

        logger.debug(f"pair = {pair}, new token = {token}, count = {count}")

        if stop_when(count, term):
            break

    return vocabulary


def heap_update(
    heap: list[tuple[int, Pair]], pair_counts: Counter[Pair]
) -> tuple[int, Pair] | None:
    while heap:
        negative_count, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)

        if -negative_count == count:
            return count, pair

    return None


def training_loop_optimized(
    corpora: list[Corpus],
    stop_when: Callable[[int, int], bool] = lambda count, term: term >= 100,
) -> Vocabulary:
    logger = logging.getLogger("OPTIMIZED(1)")
    logger.setLevel(logging.DEBUG)

    pre_tokens = chain.from_iterable([pre_tokenize(corpus) for corpus in corpora])

    vocabulary = Vocabulary()
    pair_counts: Counter[Pair] = Counter()
    pair_to_tokens: dict[Pair, set[Tokens]] = defaultdict(set)

    aggregated_tokens = aggregate_tokens(map(tuple, pre_tokens))

    for tokens, frequency in aggregated_tokens.items():
        for pair in pairwise(tokens):
            pair_counts[pair] += frequency
            pair_to_tokens[pair].add(tokens)

    heap = construct_heap(aggregated_tokens)

    term = 0
    while True:
        term += 1

        result = heap_update(heap, pair_counts)

        if result is None:
            break

        max_count, pair = result
        if max_count <= 1:
            break

        new_token = vocabulary.add_pair(pair)

        for old_tokens in list(pair_to_tokens[pair]):
            frequency = aggregated_tokens.pop(old_tokens)

            old_pair_counts: Counter[Pair] = Counter(pairwise(old_tokens))
            new_tokens = merge_pair(pair, new_token, old_tokens)
            new_pair_counts: Counter[Pair] = Counter(pairwise(new_tokens))

            changed_pairs: set[Pair] = old_pair_counts.keys() | new_pair_counts.keys()

            for changed_pair in changed_pairs:
                pair_counts[changed_pair] -= frequency * old_pair_counts.get(
                    changed_pair, 0
                )
                pair_counts[changed_pair] += frequency * new_pair_counts.get(
                    changed_pair, 0
                )

                pair_to_tokens[changed_pair].discard(old_tokens)

                if new_pair_counts.get(changed_pair, 0):
                    pair_to_tokens[changed_pair].add(new_tokens)

                new_count = pair_counts.get(changed_pair, 0)
                if new_count > 0:
                    heapq.heappush(
                        heap,
                        (-new_count, changed_pair),
                    )

            aggregated_tokens[new_tokens] += frequency

        logger.debug(f"pair = {pair}, new token = {new_token}, count = {max_count}")

        if stop_when(max_count, term):
            break

    return vocabulary
