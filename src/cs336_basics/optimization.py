"""Optimization utilities used by the language-model training loop."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Adam with decoupled weight decay.

    The update follows the assignment handout's epsilon placement, while
    keeping the implementation small enough to make the assignment's update
    rule explicit.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one update and return a closure's recomputed loss, if any."""
        with torch.enable_grad():
            loss = closure() if closure is not None else None

        with torch.no_grad():
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                learning_rate = group["lr"]
                weight_decay = group["weight_decay"]
                epsilon = group["eps"]

                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue

                    state: dict[str, Any] = self.state[parameter]
                    if not state:
                        state["step"] = torch.zeros(
                            (), dtype=torch.float32, device=parameter.device
                        )
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)

                    step = state["step"]
                    if torch.is_tensor(step):
                        step.add_(1)
                        step_number = step.item()
                    else:
                        step_number = state["step"] = step + 1

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        gradient, gradient, value=1.0 - beta2
                    )

                    parameter.mul_(1.0 - learning_rate * weight_decay)
                    bias_correction1 = 1.0 - beta1**step_number
                    bias_correction2 = 1.0 - beta2**step_number
                    step_size = (
                        learning_rate * math.sqrt(bias_correction2) / bias_correction1
                    )
                    denominator = exp_avg_sq.sqrt().add_(epsilon)
                    parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss


def cosine_learning_rate(
    step: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Return a linear-warmup, cosine-decay learning rate.

    Iteration zero is the first warmup point and therefore has learning rate
    zero.  At and beyond ``decay_steps`` the schedule stays at the minimum.
    """
    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps
    if step >= decay_steps:
        return min_learning_rate
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
        1.0 + math.cos(math.pi * progress)
    )


def clip_gradient(parameters: Iterable[Tensor], max_norm: float) -> None:
    """Clip all present gradients in place to a combined L2 norm."""
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return

    total_norm = torch.linalg.vector_norm(
        torch.stack([gradient.norm(2) for gradient in gradients])
    )
    clip_coefficient = max_norm / (total_norm + 1e-6)
    clip_coefficient = clip_coefficient.clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient)


__all__ = [
    "AdamW",
    "clip_gradient",
    "cosine_learning_rate",
]
