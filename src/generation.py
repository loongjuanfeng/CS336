"""Batched autoregressive token generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model import softmax


def _top_p_filter(probabilities: Tensor, top_p: float) -> Tensor:
    """Keep the smallest descending-probability nucleus for each row."""
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, dim=-1, descending=True
    )
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    filtered = sorted_probabilities.masked_fill(remove, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter(-1, sorted_indices, filtered)


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: Tensor,
    maximum_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    """Sample up to ``maximum_new_tokens`` continuations for a token batch."""
    was_training = model.training
    model.eval()
    try:
        generated = input_ids.clone()
        context_length = getattr(model, "context_length", generated.shape[-1])
        finished = (
            (generated == eos_token_id).any(dim=-1)
            if eos_token_id is not None
            else torch.zeros(
                generated.shape[0], dtype=torch.bool, device=generated.device
            )
        )

        for _ in range(maximum_new_tokens):
            if eos_token_id is not None and bool(finished.all()):
                break

            context = generated[:, -context_length:]
            logits = model(context)
            next_logits = logits[:, -1, :] / temperature
            probabilities = softmax(next_logits, dim=-1)
            if top_p < 1.0:
                probabilities = _top_p_filter(probabilities, top_p)
            next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

            if eos_token_id is not None:
                eos_tokens = torch.full_like(next_tokens, eos_token_id)
                next_tokens = torch.where(finished, eos_tokens, next_tokens)
                finished = finished | (next_tokens == eos_token_id)

            generated = torch.cat((generated, next_tokens.unsqueeze(-1)), dim=-1)

        return generated
    finally:
        model.train(was_training)


__all__ = ["generate"]
