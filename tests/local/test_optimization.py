import math

import pytest
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from cs336_basics.optimization import AdamW, clip_gradient, cosine_learning_rate


def test_adamw_agrees_with_pytorch_at_small_epsilon():
    torch.manual_seed(0)
    initial = torch.randn(4, 5)
    actual_parameter = nn.Parameter(initial.clone())
    expected_parameter = nn.Parameter(initial.clone())
    actual_optimizer = AdamW(
        [actual_parameter],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    expected_optimizer = torch.optim.AdamW(
        [expected_parameter],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
    )

    for step in range(50):
        gradient = torch.sin(initial + step / 10)
        actual_parameter.grad = gradient.clone()
        expected_parameter.grad = gradient.clone()
        actual_optimizer.step()
        expected_optimizer.step()

    torch.testing.assert_close(
        actual_parameter, expected_parameter, atol=2e-7, rtol=2e-6
    )


def test_adamw_runs_closure_with_gradients_enabled():
    parameter = nn.Parameter(torch.tensor([2.0]))
    optimizer = AdamW([parameter], lr=0.1, weight_decay=0.0)
    calls = 0

    def closure() -> float:
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss.item()

    loss = optimizer.step(closure)

    assert calls == 1
    assert loss == 4.0
    assert parameter.item() < 2.0


def test_gradient_clipping_matches_pytorch_and_returns_none():
    torch.manual_seed(1)
    actual_parameters = [nn.Parameter(torch.randn(3, 4)) for _ in range(3)]
    expected_parameters = [
        nn.Parameter(parameter.detach().clone()) for parameter in actual_parameters
    ]
    for index, (actual, expected) in enumerate(
        zip(actual_parameters, expected_parameters, strict=True)
    ):
        if index < 2:
            gradient = torch.randn_like(actual)
            actual.grad = gradient.clone()
            expected.grad = gradient.clone()

    clip_grad_norm_(expected_parameters, max_norm=0.2)
    result = clip_gradient(actual_parameters, max_norm=0.2)

    assert result is None
    for actual, expected in zip(actual_parameters, expected_parameters, strict=True):
        if actual.grad is not None and expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)


def test_cosine_schedule_has_exact_boundaries():
    schedule = lambda step: cosine_learning_rate(
        step,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_steps=4,
        decay_steps=12,
    )

    assert schedule(0) == 0.0
    assert schedule(2) == 0.5
    assert schedule(4) == 1.0
    assert schedule(12) == 0.1
    assert schedule(20) == 0.1
    assert schedule(8) == pytest.approx(0.55)
    assert schedule(6) == pytest.approx(0.1 + 0.45 * (1.0 + math.cos(math.pi / 4)))


def test_adamw_uses_handout_epsilon_placement():
    # A large epsilon distinguishes sqrt(v) + eps from sqrt(v_hat) + eps.
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    parameter.grad = torch.tensor([0.5], dtype=torch.float64)
    optimizer = AdamW([parameter], lr=0.1, betas=(0.8, 0.9), eps=0.2, weight_decay=0.3)
    optimizer.step()
    expected = 2.0 * (1 - 0.1 * 0.3) - (0.1 * math.sqrt(0.1) / 0.2) * 0.1 / (
        math.sqrt(0.025) + 0.2
    )
    assert parameter.item() == pytest.approx(expected)
