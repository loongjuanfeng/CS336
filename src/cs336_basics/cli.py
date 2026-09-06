"""A small, reproducible A1 training entrypoint; see ``cs336-train --help``."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .generation import generate
from .model.transformer import TransformerLanguageModel
from .optimization import AdamW, cosine_learning_rate
from .training import load_checkpoint, train


def _load_tokens(path: Path, dtype: str) -> np.ndarray:
    data = (
        np.load(path, mmap_mode="r")
        if path.suffix == ".npy"
        else np.memmap(path, mode="r", dtype=dtype)
    )
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("token data must be a one-dimensional integer array")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--train-data", type=Path, help="1D .npy or raw binary token array"
    )
    source.add_argument(
        "--smoke",
        action="store_true",
        help="use a tiny repeating synthetic corpus; no download",
    )
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument(
        "--data-dtype", choices=("uint16", "uint32", "int32", "int64"), default="uint16"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/a1"))
    parser.add_argument(
        "--resume", type=Path, help="checkpoint.pt with adjacent config.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="total completed steps, including resumed steps",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    for name in (
        "vocab-size",
        "context-length",
        "d-model",
        "num-layers",
        "num-heads",
        "d-ff",
    ):
        parser.add_argument(
            f"--{name}",
            type=int,
            default=None,
            help="defaults to a small A1 model; restored on resume",
        )
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("steps, batch-size and threads must be positive")
    if args.checkpoint_every < 1 or args.report_every < 0:
        parser.error(
            "checkpoint-every must be positive; report-every must be nonnegative"
        )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_defaults = {
        "vocab_size": 256,
        "context_length": 64,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 344,
        "rope_theta": 10000.0,
    }
    if args.smoke:
        model_defaults.update(
            vocab_size=16,
            context_length=8,
            d_model=32,
            num_layers=1,
            num_heads=4,
            d_ff=64,
        )
    training_defaults = {
        "learning_rate": 0.003,
        "min_learning_rate": 0.0003,
        "warmup_steps": 1,
        "decay_steps": 1000,
        "weight_decay": 0.01,
    }
    saved = (
        json.loads((args.resume.parent / "config.json").read_text())
        if args.resume
        else None
    )

    def resolve_config(section, defaults):
        result = {}
        for key, default in defaults.items():
            requested = getattr(args, key)
            value = saved[section][key] if saved else default
            if saved and requested is not None and requested != value:
                parser.error(
                    f"--{key.replace('_', '-')} differs from saved {section} configuration"
                )
            result[key] = requested if requested is not None else value
        return result

    model_config = resolve_config("model", model_defaults)
    training_config = resolve_config("training", training_defaults)
    if min(model_config[k] for k in model_config if k != "rope_theta") < 1:
        parser.error("model dimensions must be positive")
    if (
        model_config["d_model"] % model_config["num_heads"]
        or (model_config["d_model"] // model_config["num_heads"]) % 2
    ):
        parser.error(
            "d-model must be divisible by num-heads with an even RoPE head dimension"
        )
    if model_config["rope_theta"] <= 0:
        parser.error("rope-theta must be positive")
    if not 0 <= training_config["warmup_steps"] < training_config["decay_steps"]:
        parser.error("require 0 <= warmup-steps < decay-steps")
    if args.smoke:
        training_data = np.tile(
            np.arange(min(8, model_config["vocab_size"]), dtype=np.uint16), 1024
        )
        validation_data = training_data
    else:
        training_data = _load_tokens(args.train_data, args.data_dtype)
        validation_data = (
            _load_tokens(args.validation_data, args.data_dtype)
            if args.validation_data
            else None
        )
    for data in (training_data, validation_data):
        if data is not None and len(data) <= model_config["context_length"]:
            parser.error("each dataset must contain more tokens than context-length")
    model = TransformerLanguageModel(**model_config, device=args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    start = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    if start >= args.steps:
        parser.error("steps must exceed the checkpoint's completed iteration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({"model": model_config, "training": training_config}, indent=2)
        + "\n"
    )
    schedule = partial(
        cosine_learning_rate,
        max_learning_rate=training_config["learning_rate"],
        min_learning_rate=training_config["min_learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        decay_steps=training_config["decay_steps"],
    )
    train(
        model,
        optimizer,
        training_data,
        args.steps,
        args.batch_size,
        model.context_length,
        args.device,
        start_iteration=start,
        learning_rate_schedule=schedule,
        max_gradient_norm=args.max_gradient_norm,
        validation_data=validation_data,
        validation_batches=2,
        report_every=args.report_every,
        checkpoint_path=args.output_dir / "checkpoint.pt",
        checkpoint_every=args.checkpoint_every,
    )
    if args.smoke:
        prompt = (
            torch.tensor([[0, 1, 2]], device=args.device) % model_config["vocab_size"]
        )
        print("generated token IDs:", generate(model, prompt, 8).tolist())


if __name__ == "__main__":
    main()
