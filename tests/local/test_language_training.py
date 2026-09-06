from io import BytesIO

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cs336_basics.model.transformer import TransformerLanguageModel
from cs336_basics.optimization import AdamW
from cs336_basics.training import (
    cross_entropy,
    evaluate,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    train,
)


class _LookupLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 3) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.logits[input_ids]


def _update(
    model: nn.Module, optimizer: AdamW, input_ids: Tensor, targets: Tensor
) -> None:
    optimizer.zero_grad()
    cross_entropy(model(input_ids), targets).backward()
    optimizer.step()


def test_cross_entropy_is_shift_stable_and_preserves_dtype():
    logits = torch.tensor(
        [[1000.0, 999.0, -1000.0], [-1000.0, -999.0, 1000.0]],
    )
    targets = torch.tensor([1, 2])
    shifted_logits = logits - 10_000.0

    output = cross_entropy(logits, targets)
    half_output = cross_entropy(logits.to(torch.float16), targets)

    assert torch.isfinite(output)
    assert half_output.dtype == torch.float16
    torch.testing.assert_close(
        output, torch.nn.functional.cross_entropy(logits, targets)
    )
    torch.testing.assert_close(output, cross_entropy(shifted_logits, targets))
    torch.testing.assert_close(
        half_output,
        torch.nn.functional.cross_entropy(logits.to(torch.float16), targets),
    )


def test_get_batch_supports_arrays_and_memory_maps(tmp_path):
    tokens = np.arange(64, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    tokens.tofile(path)
    memory_map = np.memmap(path, mode="r", dtype=np.uint16, shape=tokens.shape)

    np.random.seed(10)
    array_inputs, array_targets = get_batch(
        tokens, batch_size=6, context_length=7, device="cpu"
    )
    np.random.seed(10)
    mapped_inputs, mapped_targets = get_batch(
        memory_map, batch_size=6, context_length=7, device="cpu"
    )

    assert array_inputs.dtype == torch.long
    assert array_inputs.shape == (6, 7)
    torch.testing.assert_close(array_inputs, mapped_inputs)
    torch.testing.assert_close(array_targets, mapped_targets)
    torch.testing.assert_close(array_inputs + 1, array_targets)


def test_checkpoint_round_trip_restores_optimizer_for_the_next_update():
    model = _LookupLanguageModel()
    optimizer = AdamW(model.parameters(), lr=0.03)
    input_ids = torch.tensor([[0, 1, 2], [2, 1, 0]])
    targets = torch.tensor([[1, 2, 0], [0, 2, 1]])
    _update(model, optimizer, input_ids, targets)
    checkpoint = BytesIO()

    save_checkpoint(model, optimizer, iteration=7, out=checkpoint)
    restored_model = _LookupLanguageModel()
    restored_optimizer = AdamW(restored_model.parameters(), lr=0.03)
    checkpoint.seek(0)
    iteration = load_checkpoint(checkpoint, restored_model, restored_optimizer)

    assert iteration == 7
    _update(model, optimizer, input_ids, targets)
    _update(restored_model, restored_optimizer, input_ids, targets)
    torch.testing.assert_close(model.logits, restored_model.logits)


def test_train_overwrites_one_checkpoint_and_resumes_schedule(tmp_path):
    data = np.zeros(32, dtype=np.int64)
    checkpoint_path = tmp_path / "checkpoint.pt"
    schedule_steps: list[int] = []

    def schedule(step: int) -> float:
        schedule_steps.append(step)
        return 0.05

    model = _LookupLanguageModel()
    optimizer = AdamW(model.parameters(), lr=0.01)
    completed = train(
        model,
        optimizer,
        data,
        final_iteration=3,
        batch_size=2,
        context_length=4,
        device="cpu",
        learning_rate_schedule=schedule,
        report_every=0,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )

    restored_model = _LookupLanguageModel()
    restored_optimizer = AdamW(restored_model.parameters(), lr=0.01)
    start_iteration = load_checkpoint(
        checkpoint_path, restored_model, restored_optimizer
    )
    resumed = train(
        restored_model,
        restored_optimizer,
        data,
        final_iteration=5,
        batch_size=2,
        context_length=4,
        device="cpu",
        start_iteration=start_iteration,
        learning_rate_schedule=schedule,
        report_every=0,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert completed == 3
    assert resumed == 5
    assert schedule_steps == [0, 1, 2, 3, 4]
    assert checkpoint["iteration"] == 5
    assert set(checkpoint) == {"model", "optimizer", "iteration"}
    assert list(tmp_path.iterdir()) == [checkpoint_path]


def test_evaluate_restores_model_mode():
    model = _LookupLanguageModel()
    data = np.zeros(32, dtype=np.int64)

    model.eval()
    evaluation_loss = evaluate(
        model, data, batch_size=2, context_length=4, device="cpu", num_batches=2
    )
    assert not model.training
    assert torch.isfinite(evaluation_loss)

    model.train()
    evaluate(model, data, batch_size=2, context_length=4, device="cpu", num_batches=1)
    assert model.training


def test_train_reports_training_and_validation_loss(capsys: pytest.CaptureFixture[str]):
    model = _LookupLanguageModel()
    optimizer = AdamW(model.parameters(), lr=0.05)
    data = np.zeros(24, dtype=np.int64)

    train(
        model,
        optimizer,
        data,
        final_iteration=1,
        batch_size=2,
        context_length=3,
        device="cpu",
        validation_data=data,
        validation_batches=1,
        report_every=1,
    )
    output = capsys.readouterr().out

    assert "iteration 1:" in output
    assert "train_loss=" in output
    assert "validation_loss=" in output
    assert "elapsed=" in output
    assert model.training


def test_train_deterministically_overfits_a_tiny_transformer():
    torch.manual_seed(1)
    np.random.seed(1)
    model = TransformerLanguageModel(
        vocab_size=2,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10_000.0,
    )
    optimizer = AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    data = np.zeros(64, dtype=np.int64)
    input_ids = torch.zeros((4, 4), dtype=torch.long)
    initial_loss = cross_entropy(model(input_ids), input_ids)

    train(
        model,
        optimizer,
        data,
        final_iteration=15,
        batch_size=4,
        context_length=4,
        device="cpu",
        report_every=0,
    )
    final_loss = cross_entropy(model(input_ids), input_ids)

    assert final_loss < 1e-3
    assert final_loss < initial_loss / 100
