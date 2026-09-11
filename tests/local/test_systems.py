"""A2 edge cases and actual multi-GPU correctness (no performance sweeps)."""

import os
import tempfile
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model.primitives import Embedding, Linear, RMSNorm
from cs336_systems.checkpointing import memory_optimal_forward
from cs336_systems.ddp import FlatDDP, NaiveDDP, OverlapDDP
from cs336_systems.flash_attention import FlashAttentionPytorch, FlashAttentionTriton
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


def _attention_case(implementation, device, dtype, causal, nq, nk, d):
    torch.manual_seed(7)
    inputs = [
        torch.randn(2, d, n, device=device, dtype=dtype)
        .transpose(-1, -2)
        .requires_grad_()
        for n in (nq, nk, nk)
    ]
    q, k, v = inputs
    scores = q.float() @ k.float().transpose(-1, -2) / d**0.5
    if causal:
        mask = torch.arange(nq, device=device)[:, None] < torch.arange(
            nk, device=device
        )
        scores = scores.masked_fill(mask, -torch.inf)
    expected = (scores.softmax(-1) @ v.float()).to(dtype)
    grad = torch.randn(2, d, nq, device=device, dtype=dtype).transpose(-1, -2)
    reference_grads = torch.autograd.grad(expected, inputs, grad)
    actual = implementation.apply(q, k, v, causal)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    tolerance = (
        3e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 2e-4
    )
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for result, reference in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_edges(causal):
    _attention_case(FlashAttentionPytorch, "cpu", torch.float32, causal, 65, 97, 24)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_attention_edges(causal):
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_bf16_supported(including_emulation=False):
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        _attention_case(FlashAttentionTriton, "cuda", dtype, causal, 65, 97, 24)


def test_recursive_checkpoint():
    torch.manual_seed(9)
    blocks = torch.nn.Sequential(
        *[torch.nn.Sequential(Linear(8, 8), torch.nn.SiLU()) for _ in range(4)]
    )
    x = torch.randn(2, 8, requires_grad=True)
    actual = memory_optimal_forward(blocks, x)
    expected = blocks(x)
    torch.testing.assert_close(actual, expected)
    params = (x, *blocks.parameters())
    for a, b in zip(
        torch.autograd.grad(actual.sum(), params),
        torch.autograd.grad(expected.sum(), params),
        strict=True,
    ):
        torch.testing.assert_close(a, b)


def _setup(rank, rendezvous, backend):
    torch.set_num_threads(2)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    return torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")


def _ddp_worker(rank, rendezvous, backend):
    device = _setup(rank, rendezvous, backend)
    try:
        for cls in (NaiveDDP, FlatDDP, OverlapDDP):
            torch.manual_seed(11)
            reference = torch.nn.Sequential(
                Linear(5, 7),
                torch.nn.Tanh(),
                Linear(7, 7),
                torch.nn.Tanh(),
                Linear(7, 7),
                torch.nn.Tanh(),
                Linear(7, 5),
            ).to(device)
            reference[4].weight = reference[2].weight
            reference[6].weight.requires_grad_(False)
            distributed = cls(deepcopy(reference))
            plain = torch.optim.SGD(reference.parameters(), lr=0.01)
            sharded = ShardedOptimizer(
                distributed.parameters(), torch.optim.SGD, lr=0.01
            )
            x = torch.randn(8, 5, device=device)
            for _ in range(3):
                plain.zero_grad(set_to_none=True)
                sharded.zero_grad(set_to_none=True)
                reference(x).square().mean().backward()
                distributed(x.chunk(2)[rank]).square().mean().backward()
                if cls is OverlapDDP:
                    assert distributed._pending, (
                        "No collective launched during backward"
                    )
                distributed.finish_gradient_synchronization()
                plain.step()
                sharded.step()
                for a, b in zip(
                    reference.parameters(), distributed.parameters(), strict=True
                ):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def _optimizer_worker(rank, rendezvous, backend):
    device = _setup(rank, rendezvous, backend)
    try:
        a = torch.nn.Parameter(torch.ones(3, device=device))
        b = torch.nn.Parameter(torch.ones(5, device=device))
        ra, rb = [torch.nn.Parameter(p.detach().clone()) for p in (a, b)]
        opt = ShardedOptimizer([a], torch.optim.AdamW, lr=0.01)
        ref = torch.optim.AdamW([ra], lr=0.01)
        # Rank 1 initially owns nothing, including no optimizer state.
        for p in (a, ra):
            p.grad = torch.ones_like(p)
        opt.step()
        ref.step()
        assert len(opt.state) == (1 if rank == 0 else 0)
        opt.add_param_group({"params": [b], "lr": 0.02, "weight_decay": 0.1})
        ref.add_param_group({"params": [rb], "lr": 0.02, "weight_decay": 0.1})
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            assert a.grad is None and b.grad is None
            for p in (a, b, ra, rb):
                p.grad = torch.full_like(p, step + 1)
            opt.param_groups[0]["lr"] = ref.param_groups[0]["lr"] = 0.005
            opt.step()
            ref.step()
            torch.testing.assert_close(a, ra)
            torch.testing.assert_close(b, rb)
        assert set(opt.state) == {p for p, owner in opt.owners.items() if owner == rank}
        saved = deepcopy(opt.state_dict())
        opt.load_state_dict(saved)
        opt.step()
        ref.step()
        torch.testing.assert_close(a, ra)
        torch.testing.assert_close(b, rb)
    finally:
        dist.destroy_process_group()


class _TinyFSDP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(13, 7)
        self.norm = RMSNorm(7)
        self.linear = Linear(7, 7)
        self.frozen = Linear(7, 7)
        self.frozen.weight.requires_grad_(False)
        self.head = Linear(7, 13)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        x = self.norm(self.embedding(x))
        x = torch.tanh(self.linear(x))
        return self.head(self.frozen(torch.tanh(self.linear(x))))


def _fsdp_worker(rank, rendezvous, backend):
    device = _setup(rank, rendezvous, backend)
    try:
        for dtype in (
            (None, torch.float16, torch.bfloat16) if backend == "nccl" else (None,)
        ):
            torch.manual_seed(12)
            reference = _TinyFSDP().to(device)
            model = FSDP(deepcopy(reference), compute_dtype=dtype)

            # A functional oracle casts weights without mutating master parameters.
            def oracle(x, reference=reference, dtype=dtype):
                def cast(parameter):
                    return parameter.to(dtype or parameter.dtype)

                x = torch.nn.functional.embedding(x, cast(reference.embedding.weight))
                x = reference.norm(x)
                x = torch.tanh(x @ cast(reference.linear.weight).T)
                x = torch.tanh(x @ cast(reference.linear.weight).T)
                x = x @ cast(reference.frozen.weight).T
                return x @ cast(reference.head.weight).T

            plain = torch.optim.SGD(reference.parameters(), lr=0.01)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            shard_ids = [id(p) for p in model.parameters()]
            x = torch.randint(0, 13, (8, 4), device=device)
            for _ in range(3):
                plain.zero_grad(set_to_none=True)
                opt.zero_grad(set_to_none=True)
                oracle(x).float().square().mean().backward()
                out = model(x.chunk(2)[rank])
                assert all(info.full is None for info in model._weights.values())
                out.float().square().mean().backward()
                model.finish_gradient_synchronization()
                for p in model.parameters():
                    if not p.requires_grad:
                        assert p.grad is None
                        continue
                    assert p.grad is not None
                    assert (
                        p.grad.shape == p.shape
                        and p.grad.dtype == p.dtype == torch.float32
                    )
                plain.step()
                opt.step()
                full = model.gather_full_params()
                tolerance = 5e-4 if dtype else 1e-6
                for name, p in reference.named_parameters():
                    torch.testing.assert_close(
                        full[name], p, atol=tolerance, rtol=tolerance
                    )
                assert shard_ids == [id(p) for p in model.parameters()]
                assert all(
                    info.parameter.numel() == (info.numel + 1) // 2
                    for info in model._weights.values()
                )
                assert all(info.full is None for info in model._weights.values())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "worker",
    [_ddp_worker, _optimizer_worker, _fsdp_worker],
    ids=["ddp", "optimizer", "fsdp"],
)
def test_distributed_edges(worker):
    backend = os.environ.get("CS336_TEST_BACKEND", "gloo")
    if backend == "nccl":
        assert torch.cuda.device_count() >= 2, "NCCL validation requires two real GPUs"
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            worker,
            args=(f"file://{directory}/rendezvous", backend),
            nprocs=2,
            join=True,
        )
