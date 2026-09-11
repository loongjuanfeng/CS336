"""Command-line entry point and handout model presets."""

import argparse
import json
import tempfile
from pathlib import Path

from .suites import MODELS, SUITES, cases, model_config


def parser():
    p = argparse.ArgumentParser(
        description="A2 reference experiments (eager FP32 by default)"
    )
    p.add_argument(
        "--experiment",
        choices=(
            "transformer",
            "attention",
            "communication",
            "ddp",
            "accumulation",
            "precision",
        ),
        default="transformer",
    )
    p.add_argument("--model-size", choices=MODELS, default="small")
    p.add_argument("--mode", choices=("forward", "backward", "train"), default="train")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    for key in ("compile", "causal", "memory-snapshot", "modal", "nsys", "list-cases"):
        p.add_argument("--" + key, action="store_true")
    for key in (
        "d-model",
        "d-ff",
        "num-layers",
        "num-heads",
        "batch-size",
        "context-length",
        "steps",
    ):
        p.add_argument("--" + key, type=int)
    for key, default in {
        "warmup": 5,
        "vocab-size": 10000,
        "seed": 0,
        "checkpoint-layers": 0,
        "world-size": 2,
        "message-mb": 1,
    }.items():
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--timing", choices=("wall", "triton"), default="wall")
    p.add_argument(
        "--gradient-reduce", choices=("individual", "flat"), default="individual"
    )
    p.add_argument("--backend", choices=("gloo", "nccl"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--gpu", default="A100-80GB", help="Modal GPU type, without :count")
    p.add_argument("--suite", choices=(*SUITES, "all"))
    p.add_argument("--case", type=int, help="Run only this zero-based suite case")
    p.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    p.add_argument("--capture", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    defaults = model_config(args.model_size) | {
        "batch_size": 4,
        "context_length": 512,
        "steps": 10,
    }
    if args.experiment == "attention":
        defaults.update(batch_size=8, d_model=64, context_length=256, steps=100)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.backend = args.backend or ("nccl" if args.device == "cuda" else "gloo")
    for key in (
        "steps",
        "batch_size",
        "context_length",
        "vocab_size",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "world_size",
        "message_mb",
    ):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.warmup < 0 or args.checkpoint_layers < 0:
        p.error("warmup and checkpoint-layers must be nonnegative")
    if args.experiment in ("transformer", "ddp") and (
        args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2
    ):
        p.error("head dimension must be integral and even")
    if (
        args.nsys
        or args.capture
        or args.memory_snapshot
        or args.modal
        or args.timing == "triton"
    ) and args.device != "cuda":
        p.error("Modal, nsys, memory snapshots and Triton timing require CUDA")
    if args.experiment == "communication" and args.precision != "fp32":
        p.error("the communication baseline uses FP32 tensors")
    if args.experiment != "attention" and args.timing == "triton":
        p.error("--timing triton is for standalone attention")
    if args.experiment != "transformer" and args.checkpoint_layers:
        p.error("--checkpoint-layers is for the Transformer experiment")
    if args.experiment in ("accumulation", "precision") and (
        args.nsys or args.memory_snapshot or args.compile
    ):
        p.error("dtype toy experiments do not use profiling or compile flags")
    if args.device == "cpu" and args.backend == "nccl":
        p.error("NCCL requires CUDA")
    if args.experiment in ("communication", "ddp") and (
        args.compile or args.checkpoint_layers
    ):
        p.error("distributed references do not support compile/checkpointing yet")
    if args.experiment == "ddp" and args.mode != "train":
        p.error("DDP reference measures full training steps")
    if (args.list_cases or args.case is not None) and not args.suite:
        p.error("--list-cases/--case require --suite")
    return args


def main(argv=None):
    args = parse_args(argv)
    configurations = list(enumerate(cases(args.suite))) if args.suite else []
    if args.case is not None:
        if not 0 <= args.case < len(configurations):
            raise SystemExit("--case is outside the suite")
        configurations = [configurations[args.case]]
    if args.list_cases:
        print(
            json.dumps(
                [{"index": i, **config} for i, config in configurations], indent=2
            )
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        args.output_dir.resolve()
        if args.worker
        else Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    )
    from .runner import launch, run_suite

    if args.suite:
        run_suite(args, configurations, output)
    else:
        launch(args, output)
    print(f"Results: {output}")
