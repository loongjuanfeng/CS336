"""Minimal end-to-end benchmark for the Assignment 2 profiling exercises."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from cs336_basics.model.transformer import TransformerLanguageModel


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("forward", "backward", "train"), default="train")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1_024)
    args = p.parse_args()

    device = torch.device(args.device)
    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10_000.0,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=device
    )

    def step() -> None:
        if args.mode == "train":
            optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        if args.mode != "forward":
            logits.float().mean().backward()
        if args.mode == "train":
            optimizer.step()
        sync(device)

    model.train()
    for _ in range(args.warmup):
        step()
    timings = []
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        timings.append(time.perf_counter() - start)
    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"mode={args.mode} device={device} steps={args.steps}")
    print(f"mean={mean * 1e3:.3f} ms  stdev={stdev * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
