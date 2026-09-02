"""Losses, data loading, checkpointing, and a small library training loop."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from optimization import clip_gradient


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute mean next-token cross-entropy over arbitrary batch axes."""
    input_dtype = logits.dtype
    working = (
        logits.to(torch.float32)
        if input_dtype in (torch.float16, torch.bfloat16)
        else logits
    )
    shifted = working - working.amax(dim=-1, keepdim=True)
    log_partition = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_partition - target_logits).mean().to(input_dtype)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Sample random contiguous input/target windows from a token array."""
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    targets = np.stack(
        [dataset[start + 1 : start + context_length + 1] for start in starts]
    )
    input_tensor = torch.as_tensor(inputs, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    return input_tensor, target_tensor


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO | IO[bytes],
) -> None:
    """Serialize model state, optimizer state, and completed iteration."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """Restore a checkpoint and return its completed iteration."""
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    num_batches: int = 10,
) -> Tensor:
    """Return mean loss over sampled validation batches.

    Evaluation temporarily switches the model to evaluation mode and restores
    the caller's original mode before returning.
    """
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(num_batches):
            input_ids, targets = get_batch(dataset, batch_size, context_length, device)
            losses.append(cross_entropy(model(input_ids), targets))
        return torch.stack(losses).mean()
    finally:
        model.train(was_training)


def train(
    model: nn.Module,
    optimizer: Optimizer,
    training_data: np.ndarray,
    final_iteration: int,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    *,
    start_iteration: int = 0,
    learning_rate_schedule: Callable[[int], float] | None = None,
    maximum_gradient_norm: float | None = None,
    validation_data: np.ndarray | None = None,
    validation_batches: int = 10,
    report_every: int = 100,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_every: int = 1000,
) -> int:
    """Train ``model`` until ``final_iteration`` and return that iteration.

    Iterations are counted as completed optimizer updates, so resuming with
    ``start_iteration`` continues at the same schedule index and checkpoint
    numbering.  A supplied checkpoint path is a single rolling checkpoint.
    """
    model.to(device)
    model.train()
    last_loss: Tensor | None = None
    started_at = time.perf_counter()

    for iteration in range(start_iteration, final_iteration):
        if learning_rate_schedule is not None:
            learning_rate = learning_rate_schedule(iteration)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

        input_ids, targets = get_batch(
            training_data, batch_size, context_length, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        last_loss = cross_entropy(logits, targets)
        last_loss.backward()
        if maximum_gradient_norm is not None:
            clip_gradient(model.parameters(), maximum_gradient_norm)
        optimizer.step()

        completed_iteration = iteration + 1
        should_report = report_every and (
            completed_iteration % report_every == 0
            or completed_iteration == final_iteration
        )
        if should_report:
            elapsed = time.perf_counter() - started_at
            message = f"iteration {completed_iteration}: train_loss={last_loss.item():.6f} elapsed={elapsed:.1f}s"
            if validation_data is not None:
                validation_loss = evaluate(
                    model,
                    validation_data,
                    batch_size,
                    context_length,
                    device,
                    validation_batches,
                )
                message += f" validation_loss={validation_loss.item():.6f}"
            print(message)

        if checkpoint_path is not None and (
            completed_iteration % checkpoint_every == 0
            or completed_iteration == final_iteration
        ):
            save_checkpoint(model, optimizer, completed_iteration, checkpoint_path)

    return final_iteration


__all__ = [
    "cross_entropy",
    "evaluate",
    "get_batch",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
