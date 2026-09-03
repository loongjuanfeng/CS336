"""Adapters between the official CS336 tests and this implementation.

The assignment's tests intentionally describe a small, implementation-agnostic
interface. This module keeps that interface in the test package while routing
the calls to the modules in ``src``. The tokenizer adapter is implemented here
as well because this repository's public tokenizer API is function-based rather
than class-based.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from model.attention import MultiHeadSelfAttention
from model.positional import RotaryPositionalEmbedding
from model.primitives import (
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from model.transformer import TransformerBlock, TransformerLanguageModel
from optimization import AdamW, clip_gradient, cosine_learning_rate
from tokenizer.pre_tokenize import (
    compile_special_pattern,
    pre_tokenize_text,
    split_special_tokens,
)
from training import cross_entropy, get_batch, load_checkpoint, save_checkpoint


def _run_module(
    module: torch.nn.Module, weights: dict[str, Tensor], inputs: Tensor
) -> Tensor:
    """Load a reference state dict and run a module without changing its state."""
    with torch.no_grad():
        module.load_state_dict(weights)
        return module(inputs)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    module = Linear(
        d_in,
        d_out,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    return _run_module(module, {"weight": weights}, in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    module = Embedding(
        vocab_size,
        d_model,
        device=weights.device,
        dtype=weights.dtype,
    )
    with torch.no_grad():
        module.load_state_dict({"weight": weights})
        return module(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    module = SwiGLU(
        d_model,
        d_ff,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    return _run_module(
        module,
        {
            "w1.weight": w1_weight,
            "w2.weight": w2_weight,
            "w3.weight": w3_weight,
        },
        in_features,
    )


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    module = MultiHeadSelfAttention(
        d_model,
        num_heads,
        rope=None,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    return _run_module(
        module,
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        },
        in_features,
    )


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    rope = RotaryPositionalEmbedding(
        theta,
        d_model // num_heads,
        max_seq_len,
        device=in_features.device,
    )
    module = MultiHeadSelfAttention(
        d_model,
        num_heads,
        rope=rope,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    with torch.no_grad():
        module.load_state_dict(
            {
                "q_proj.weight": q_proj_weight,
                "k_proj.weight": k_proj_weight,
                "v_proj.weight": v_proj_weight,
                "output_proj.weight": o_proj_weight,
            }
        )
        return module(in_features, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    module = RotaryPositionalEmbedding(
        theta,
        d_k,
        max_seq_len,
        device=in_query_or_key.device,
    )
    with torch.no_grad():
        return module(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    module = TransformerBlock(
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    return _run_module(module, weights, in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    module = TransformerLanguageModel(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=in_indices.device,
        dtype=weights["token_embeddings.weight"].dtype,
    )
    return _run_module(module, weights, in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    module = RMSNorm(
        d_model,
        eps=eps,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    return _run_module(module, {"weight": weights}, in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy(inputs, targets)


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    clip_gradient(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return cosine_learning_rate(
        it,
        maximum_learning_rate=max_learning_rate,
        minimum_learning_rate=min_learning_rate,
        warmup_steps=warmup_iters,
        decay_steps=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src, model, optimizer)


def _proper_special_prefixes(special_tokens: Sequence[str]) -> frozenset[str]:
    return frozenset(
        token[:length]
        for token in special_tokens
        for length in range(1, len(token))
    )


def _split_pending_special_suffix(
    text: str, prefixes: frozenset[str], longest_prefix: int
) -> tuple[str, str]:
    for length in range(min(longest_prefix, len(text)), 0, -1):
        if text[-length:] in prefixes:
            return text[:-length], text[-length:]
    return text, ""


class _TokenizerAdapter:
    """A small byte-level BPE tokenizer with the official test contract."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None,
    ) -> None:
        self._id_to_bytes = dict(vocab)
        self._bytes_to_id: dict[bytes, int] = {}
        for token_id, token_data in sorted(self._id_to_bytes.items()):
            self._bytes_to_id.setdefault(token_data, token_id)

        # The official helper normally appends specials before calling us. Do
        # the same defensively for callers that provide a bare GPT-2 vocab.
        next_id = max(self._id_to_bytes, default=-1) + 1
        self._special_tokens = tuple(special_tokens or ())
        for special_token in self._special_tokens:
            token_data = special_token.encode("utf-8")
            if token_data not in self._bytes_to_id:
                self._id_to_bytes[next_id] = token_data
                self._bytes_to_id[token_data] = next_id
                next_id += 1

        self._merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(merges):
            self._merge_ranks.setdefault(pair, rank)

        self._special_pattern = compile_special_pattern(self._special_tokens)
        self._special_ids = {
            token: self._bytes_to_id[token.encode("utf-8")]
            for token in self._special_tokens
        }

    def _encode_pre_token(self, pre_token: bytes) -> list[int]:
        pieces = [bytes([byte]) for byte in pre_token]
        while len(pieces) > 1:
            best_index: int | None = None
            best_rank: int | None = None
            for index, pair in enumerate(pairwise(pieces)):
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_index = index
                    best_rank = rank
            if best_index is None:
                break
            pieces[best_index : best_index + 2] = [
                pieces[best_index] + pieces[best_index + 1]
            ]
        return [self._bytes_to_id[piece] for piece in pieces]

    def _encode_segments(self, text: str) -> Iterator[int]:
        for segment, is_special in split_special_tokens(text, self._special_pattern):
            if is_special:
                yield self._special_ids[segment]
                continue
            for pre_token in pre_tokenize_text(segment):
                yield from self._encode_pre_token(pre_token)

    def encode(self, text: str) -> list[int]:
        return list(self._encode_segments(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        prefixes = _proper_special_prefixes(self._special_tokens)
        longest_prefix = max((len(prefix) for prefix in prefixes), default=0)
        pending = ""
        for chunk in iterable:
            head, pending = _split_pending_special_suffix(
                pending + chunk, prefixes, longest_prefix
            )
            yield from self._encode_segments(head)
        if pending:
            yield from self._encode_segments(pending)

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self._id_to_bytes[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return _TokenizerAdapter(vocab, merges, special_tokens)


def _merge_token_sequence(
    sequence: tuple[int, ...], pair: tuple[int, int], new_token: int
) -> tuple[int, ...]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index : index + 2] == pair:
            merged.append(new_token)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return tuple(merged)


def _iter_training_pretokens(
    input_path: str | os.PathLike, special_tokens: Sequence[str]
) -> Iterator[bytes]:
    pattern = compile_special_pattern(special_tokens)
    with open(input_path, encoding="utf-8") as input_file:
        # Pretokenize the corpus as one text stream.  In particular, the GPT-2
        # pattern is allowed to group adjacent newlines (``\n\n``), which is
        # observable in the official TinyStories reference merges.
        text = input_file.read()
    for segment, is_special in split_special_tokens(text, pattern):
        if not is_special:
            yield from pre_tokenize_text(segment)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE with the official lexicographically-greatest tie break.

    This is deliberately separate from the production training loops. The
    repository's public training API retains its existing tie-break rule, while
    the official assignment contract requires the greater byte pair on ties.
    """
    del kwargs

    vocab: dict[int, bytes] = {}
    bytes_to_id: dict[bytes, int] = {}
    next_id = 0
    for special_token in special_tokens:
        token_data = special_token.encode("utf-8")
        if token_data not in bytes_to_id:
            vocab[next_id] = token_data
            bytes_to_id[token_data] = next_id
            next_id += 1
    for byte in range(256):
        token_data = bytes([byte])
        if token_data not in bytes_to_id:
            vocab[next_id] = token_data
            bytes_to_id[token_data] = next_id
            next_id += 1

    if vocab_size <= len(vocab):
        return dict(list(vocab.items())[:vocab_size]), []

    aggregated: Counter[tuple[int, ...]] = Counter()
    for pre_token in _iter_training_pretokens(input_path, special_tokens):
        sequence = tuple(bytes_to_id[bytes([byte])] for byte in pre_token)
        if sequence:
            aggregated[sequence] += 1

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_sequences: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for sequence, frequency in aggregated.items():
        for pair in pairwise(sequence):
            pair_counts[pair] += frequency
            pair_to_sequences[pair].add(sequence)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        active_pairs = [
            (pair, count) for pair, count in pair_counts.items() if count > 0
        ]
        if not active_pairs:
            break

        # Tuple comparison is part of the official contract: on equal counts,
        # compare the original token bytes, not their remapped printable forms.
        pair, _count = max(
            active_pairs,
            key=lambda item: (
                item[1],
                vocab[item[0][0]],
                vocab[item[0][1]],
            ),
        )
        left_data = vocab[pair[0]]
        right_data = vocab[pair[1]]
        merged_data = left_data + right_data
        new_token = len(vocab)
        vocab[new_token] = merged_data
        merges.append((left_data, right_data))

        affected_sequences = list(pair_to_sequences.get(pair, ()))
        for old_sequence in affected_sequences:
            frequency = aggregated.pop(old_sequence, 0)
            if not frequency:
                continue

            old_pair_counts = Counter(pairwise(old_sequence))
            new_sequence = _merge_token_sequence(old_sequence, pair, new_token)
            new_pair_counts = Counter(pairwise(new_sequence))
            changed_pairs = old_pair_counts.keys() | new_pair_counts.keys()

            for changed_pair in changed_pairs:
                pair_counts[changed_pair] += frequency * (
                    new_pair_counts.get(changed_pair, 0)
                    - old_pair_counts.get(changed_pair, 0)
                )
                pair_to_sequences[changed_pair].discard(old_sequence)
                if new_pair_counts.get(changed_pair, 0):
                    pair_to_sequences[changed_pair].add(new_sequence)
                if pair_counts[changed_pair] <= 0:
                    del pair_counts[changed_pair]

            aggregated[new_sequence] += frequency

    return vocab, merges
