"""Byte-pair training with incremental pair counts and explicit merge order."""

import heapq
import logging
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from itertools import pairwise
from os import PathLike, cpu_count

from .pre_tokenize import iter_pre_tokens

Pair = tuple[bytes, bytes]
Word = tuple[bytes, ...]


def _count_chunk(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    path, start, end, special_tokens = args
    with open(path, "rb") as corpus:
        corpus.seek(start)
        text = corpus.read(end - start).decode("utf-8")
    return Counter(
        token
        for token, is_special in iter_pre_tokens((text,), special_tokens)
        if not is_special
    )


def _count_pre_tokens_parallel(
    path: str, special_tokens: list[str], workers: int | None = None
) -> Counter[bytes]:
    with open(path, "rb") as corpus:
        corpus.seek(0, 2)
        size = corpus.tell()
    workers = workers or cpu_count() or 1
    step = max(size // workers, 1)
    boundaries = [0]
    with open(path, "rb") as corpus:
        for offset in range(step, size, step):
            corpus.seek(offset)
            corpus.readline()
            boundaries.append(corpus.tell())
    boundaries.append(size)
    chunks = [
        (path, start, end, tuple(special_tokens))
        for start, end in pairwise(boundaries)
        if start < end
    ]
    if not chunks:
        return Counter()
    result: Counter[bytes] = Counter()
    with mp.Pool(min(workers, len(chunks))) as pool:
        for partial in pool.imap_unordered(_count_chunk, chunks):
            result.update(partial)
    return result


def _merge(word: Word, pair: Pair) -> Word:
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    workers: int | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train until the target vocabulary size or exhaustion of adjacent pairs."""
    token_ids = {bytes([token_id]): token_id for token_id in range(256)}
    for special in special_tokens:
        if not special:
            raise ValueError("special tokens must be non-empty")
        token_ids.setdefault(special.encode("utf-8"), len(token_ids))
    if vocab_size < len(token_ids):
        raise ValueError("vocab_size must cover all bytes and special tokens")
    merges: list[Pair] = []
    started = time.perf_counter()
    frequencies = _count_pre_tokens_parallel(str(input_path), special_tokens, workers)
    logging.getLogger(__name__).info(
        "pretokenization seconds=%.3f unique=%d",
        time.perf_counter() - started,
        len(frequencies),
    )
    started = time.perf_counter()
    words = [tuple(bytes([b]) for b in word) for word in frequencies]
    counts = list(frequencies.values())
    pair_counts: Counter[Pair] = Counter()
    occurrences: dict[Pair, set[int]] = defaultdict(set)
    for index, word in enumerate(words):
        for pair, count in Counter(pairwise(word)).items():
            pair_counts[pair] += count * counts[index]
            occurrences[pair].add(index)
    heap = [(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify_max(heap)
    logging.getLogger(__name__).info(
        "pair initialization seconds=%.3f", time.perf_counter() - started
    )
    started = time.perf_counter()
    while heap and len(token_ids) < vocab_size:
        count, pair = heapq.heappop_max(heap)
        if count == 0 or count != pair_counts[pair]:
            continue
        token_ids.setdefault(pair[0] + pair[1], len(token_ids))
        merges.append(pair)
        affected = list(occurrences[pair])
        changed = set()
        for index in affected:
            old = words[index]
            new = _merge(old, pair)
            if old == new:
                continue
            before = Counter(pairwise(old))
            after = Counter(pairwise(new))
            for changed_pair in before.keys() | after.keys():
                pair_counts[changed_pair] += (
                    after[changed_pair] - before[changed_pair]
                ) * counts[index]
                occurrences[changed_pair].discard(index)
                if after[changed_pair]:
                    occurrences[changed_pair].add(index)
                changed.add(changed_pair)
            words[index] = new
        for changed_pair in changed:
            if pair_counts[changed_pair] > 0:
                heapq.heappush_max(heap, (pair_counts[changed_pair], changed_pair))
    logging.getLogger(__name__).info(
        "merges seconds=%.3f count=%d", time.perf_counter() - started, len(merges)
    )
    return {token_id: data for data, token_id in token_ids.items()}, merges
