"""A small, self-contained Triton vector-add example.

Run it with a CUDA-capable GPU, for example::

    uv run python examples/triton/010-vector_add.py --n 1000003 --dtype fp32

The non-multiple-of-``BLOCK_SIZE`` default is intentional: it exercises the
mask on the last program launched by the kernel.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - depends on the local install.
    raise SystemExit(
        "This example requires Triton. Install a PyTorch build that includes "
        "Triton, or install Triton separately."
    ) from exc


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Add one contiguous 1-D block of ``x`` and ``y``.

    Triton launches one *program* for each block.  The mask keeps the final
    program from reading or writing past the end when ``n_elements`` is not a
    multiple of ``BLOCK_SIZE``.
    """

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def triton_add(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    block_size: int = 1024,
) -> torch.Tensor:
    """Return ``x + y`` using :func:`add_kernel`.

    The wrapper deliberately performs the inexpensive checks that are useful
    when learning to connect a PyTorch tensor to a custom Triton kernel.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must both be one-dimensional")
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )
    if x.dtype != y.dtype:
        raise ValueError(
            f"x and y must have the same dtype, got {x.dtype} and {y.dtype}"
        )
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("x and y must be CUDA tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = ((n_elements + block_size - 1) // block_size,)
    # Triton's launcher accepts an int for a constexpr meta-parameter, but its
    # current type information requires an instance of tl.constexpr here.
    # ty: ignore[invalid-argument-type]
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output


def benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    """Return mean, standard deviation, and median latency in milliseconds."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings: list[float] = []
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        # CUDA launches are asynchronous; synchronize before reading elapsed time.
        end.synchronize()
        timings.append(start.elapsed_time(end))

    mean = statistics.fmean(timings)
    stddev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return mean, stddev, statistics.median(timings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=1_000_003,
        help="number of elements (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp32",
        help="element dtype (default: %(default)s)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Triton block size; use a positive power of two (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warm-up iterations before timing (default: %(default)s)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="timed iterations (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        parser.error("--block-size must be a positive power of two")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU.")

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(args.n, device=device, dtype=dtype)
    y = torch.randn(args.n, device=device, dtype=dtype)

    # The first call also compiles the Triton kernel, so compile time is not
    # included in the benchmark below.
    triton_result = triton_add(x, y, block_size=args.block_size)
    reference = torch.add(x, y)
    tolerance = {torch.float16: (1e-3, 1e-3), torch.float32: (1e-5, 1e-5)}[dtype]
    torch.testing.assert_close(
        triton_result,
        reference,
        rtol=tolerance[0],
        atol=tolerance[1],
    )
    torch.cuda.synchronize()

    triton_stats = benchmark(
        lambda: triton_add(x, y, block_size=args.block_size),
        warmup=args.warmup,
        iterations=args.iters,
    )
    torch_stats = benchmark(
        lambda: torch.add(x, y),
        warmup=args.warmup,
        iterations=args.iters,
    )

    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"n={args.n}, dtype={args.dtype}, block_size={args.block_size}")
    print("correctness: passed")
    print(
        "triton.add: "
        f"mean={triton_stats[0]:.4f} ms, "
        f"std={triton_stats[1]:.4f} ms, "
        f"median={triton_stats[2]:.4f} ms"
    )
    print(
        "torch.add:   "
        f"mean={torch_stats[0]:.4f} ms, "
        f"std={torch_stats[1]:.4f} ms, "
        f"median={torch_stats[2]:.4f} ms"
    )
    print(f"mean speedup (torch / triton): {torch_stats[0] / triton_stats[0]:.2f}x")


if __name__ == "__main__":
    main()
