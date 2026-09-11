"""Explicit quadratic PyTorch attention; reference for compile and FlashAttention."""

import torch

from cs336_basics.model.primitives import scaled_dot_product_attention

from .common import measure, memory, setup, summary, sync, timed


def benchmark(args, output):
    if args.compile and args.timing == "triton":
        from torch._functorch import config

        # do_bench repeats backward on a retained graph; donated buffers cannot be reused.
        with config.patch(donated_buffer=False):
            return _benchmark(args, output)
    return _benchmark(args, output)


def _benchmark(args, output):
    setup(args)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    shape = (args.batch_size, args.context_length, args.d_model)
    q, k, v = [
        torch.randn(shape, device=args.device, dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    grad = torch.randn_like(q)
    mask = (
        torch.ones(
            args.context_length,
            args.context_length,
            device=args.device,
            dtype=torch.bool,
        ).tril()
        if args.causal
        else None
    )

    def attention():
        return scaled_dot_product_attention(q, k, v, mask)

    run = torch.compile(attention) if args.compile else attention
    forward_times, backward_times = [], []
    before_backward = {}
    collect = False

    def clear():
        q.grad = k.grad = v.grad = None

    def step():
        nonlocal before_backward
        clear()
        # Separate synchronized phase timings are required by pytorch_attention.
        sync(args.device)
        import time

        start = time.perf_counter()
        result = run()
        sync(args.device)
        fwd = (time.perf_counter() - start) * 1000
        before_backward = memory(args.device)
        bwd = timed(lambda: result.backward(grad), args.device)
        if collect:
            forward_times.append(fwd)
            backward_times.append(bwd)

    for _ in range(args.warmup):
        step()
    collect = True
    from argparse import Namespace

    report = measure(step, Namespace(**(vars(args) | {"warmup": 0})), output)
    report.update(
        forward=summary(forward_times),
        backward=summary(backward_times),
        memory_before_backward=before_backward,
        input_dtype=str(dtype),
        timing_method="synchronized_wall_clock",
    )
    if args.timing == "triton":
        from triton.testing import do_bench

        # Retaining a fixed graph keeps forward construction outside backward timing.
        clear()
        result = run()

        def backward():
            clear()
            result.backward(grad, retain_graph=True)

        def combined():
            clear()
            run().backward(grad)

        report["triton_ms"] = {
            "forward": do_bench(run, warmup=25, rep=100, return_mode="mean"),
            "backward": do_bench(backward, warmup=25, rep=100, return_mode="mean"),
            "forward_backward": do_bench(
                combined, warmup=25, rep=100, return_mode="mean"
            ),
        }
    return report
