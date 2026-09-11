"""Timing and memory measurements shared by the baseline experiments."""

import statistics
import time
from contextlib import nullcontext

import torch


def setup(args):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu for small checks")


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def region(name, device):
    return (
        torch.cuda.nvtx.range(name) if str(device).startswith("cuda") else nullcontext()
    )


def autocast(args):
    return torch.autocast(
        args.device, dtype=torch.bfloat16, enabled=args.precision == "bf16"
    )


def memory(device):
    if not str(device).startswith("cuda"):
        return {}
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def summary(times):
    return {
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "step_ms": times,
    }


def timed(fn, device):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return (time.perf_counter() - start) * 1000


def measure(step, args, output, device=None):
    device = device or args.device
    for i in range(args.warmup):
        with region(f"warmup {i}", device):
            step()
        sync(device)
    sync(device)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(stacks="python", max_entries=100_000)
    if args.capture:
        torch.cuda.profiler.start()
    times = []
    try:
        for i in range(args.steps):
            start = time.perf_counter()
            with region(f"step {i}", device):
                step()
                with region("sync", device):
                    sync(device)
            times.append((time.perf_counter() - start) * 1000)
    finally:
        if args.capture:
            torch.cuda.profiler.stop()
        if args.memory_snapshot:
            try:
                torch.cuda.memory._dump_snapshot(str(output / "memory.pickle"))
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
    report = summary(times)
    report.update({k: v for k, v in memory(device).items() if k.startswith("peak_")})
    return report


def environment(args):
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "precision": args.precision,
    }
