"""Exercise the actual A1 training path used by the A2 baseline."""

import pytest
import torch

from cs336_basics.optimization import AdamW
from cs336_systems.baseline.cli import parse_args
from cs336_systems.baseline.transformer import benchmark


@pytest.mark.parametrize("mode", ["forward", "backward", "train"])
def test_baseline_training_path(mode, monkeypatch, tmp_path):
    updates = []
    original = AdamW.step

    def checked_step(self, closure=None):
        parameters = self.param_groups[0]["params"]
        before = parameters[0].detach().clone()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
        )
        result = original(self, closure)
        updates.append(not torch.equal(before, parameters[0]))
        return result

    monkeypatch.setattr(AdamW, "step", checked_step)
    args = parse_args(
        [
            "--device",
            "cpu",
            "--mode",
            mode,
            "--d-model",
            "16",
            "--d-ff",
            "32",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--context-length",
            "8",
            "--vocab-size",
            "32",
            "--batch-size",
            "2",
            "--warmup",
            "1",
            "--steps",
            "2",
        ]
    )
    report = benchmark(args, tmp_path)
    assert len(report["step_ms"]) == 2
    assert report["mean_ms"] > 0
    assert updates == ([True] * 3 if mode == "train" else [])


def test_checkpoint_preserves_outputs_and_gradients():
    from cs336_systems.baseline.transformer import forward, make_model

    args = parse_args(
        [
            "--device",
            "cpu",
            "--d-model",
            "16",
            "--d-ff",
            "32",
            "--num-heads",
            "2",
            "--num-layers",
            "3",
            "--vocab-size",
            "32",
            "--context-length",
            "8",
        ]
    )
    model = make_model(args, "cpu")
    inputs = torch.randint(32, (2, 8))
    reference = forward(model, inputs)
    reference.square().mean().backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    actual = forward(model, inputs, group_size=2)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, reference)
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_bf16_autocast_keeps_fp32_parameters(tmp_path, monkeypatch):
    from cs336_basics.model.transformer import TransformerLanguageModel

    dtypes = []
    original = TransformerLanguageModel.forward

    def checked_forward(self, inputs):
        assert all(p.dtype == torch.float32 for p in self.parameters())
        result = original(self, inputs)
        dtypes.append(result.dtype)
        return result

    monkeypatch.setattr(TransformerLanguageModel, "forward", checked_forward)
    args = parse_args(
        [
            "--device",
            "cpu",
            "--precision",
            "bf16",
            "--d-model",
            "16",
            "--d-ff",
            "32",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
            "--context-length",
            "8",
            "--vocab-size",
            "32",
            "--steps",
            "2",
            "--warmup",
            "1",
        ]
    )
    benchmark(args, tmp_path)
    assert dtypes == [torch.bfloat16] * 3


def _check_reduction(rank, path):
    import torch.distributed as dist

    from cs336_systems.baseline.distributed import reduce_gradients

    dist.init_process_group("gloo", init_method=path, rank=rank, world_size=2)
    try:
        parameters = [
            torch.nn.Parameter(torch.zeros(2)),
            torch.nn.Parameter(torch.zeros(2, 3)),
        ]
        for flat in (False, True):
            for i, p in enumerate(parameters):
                p.grad = torch.full_like(p, rank + i + 1.0)
            reduce_gradients(parameters, flat)
            for i, p in enumerate(parameters):
                torch.testing.assert_close(p.grad, torch.full_like(p, i + 1.5))
    finally:
        dist.destroy_process_group()


def test_distributed_gradient_average(tmp_path):
    torch.multiprocessing.spawn(
        _check_reduction,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_all_suite_cases_parse():
    from cs336_systems.baseline.runner import command
    from cs336_systems.baseline.suites import cases

    for case in cases("all"):
        parse_args(command(case)[3:])


def test_cuda_oom_is_a_recorded_result(tmp_path, monkeypatch):
    import json

    from cs336_systems.baseline import transformer
    from cs336_systems.baseline.runner import run_case

    def out_of_memory(*args):
        raise torch.OutOfMemoryError("test allocation limit")

    monkeypatch.setattr(transformer, "benchmark", out_of_memory)
    args = parse_args(["--device", "cpu"])
    run_case(args, tmp_path)
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["status"] == "oom"
    assert report["config"]["d_model"] == 768
