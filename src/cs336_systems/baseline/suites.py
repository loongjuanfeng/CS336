"""Handout configurations. These generate experiments, not completed answers."""

from itertools import product

MODELS = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}
SUITES = (
    "benchmarking",
    "nsys",
    "mixed-precision",
    "memory",
    "checkpointing",
    "attention",
    "compile",
    "flash",
    "communication",
    "ddp",
    "sharding",
    "fsdp",
    "leaderboard",
    "accumulation",
)


def model_config(name):
    return dict(
        zip(("d_model", "d_ff", "num_layers", "num_heads"), MODELS[name], strict=True)
    ) | {"model_size": name}


def cases(suite):
    if suite == "all":
        return [
            dict(case, case_name=f"{name}-{i:03}")
            for name in SUITES
            for i, case in enumerate(cases(name))
        ]
    model = {
        "experiment": "transformer",
        "batch_size": 4,
        "context_length": 512,
        "vocab_size": 10000,
        "precision": "fp32",
        "compile": False,
        "checkpoint_layers": 0,
        "warmup": 5,
        "steps": 10,
    }
    if suite == "benchmarking":
        return [
            model | model_config(size) | {"mode": mode, "warmup": warmup}
            for size, mode, warmup in product(
                MODELS, ("forward", "backward", "train"), (0, 1, 2, 5)
            )
        ]
    if suite == "mixed-precision":
        return [{"experiment": "precision"}] + [
            model | model_config(size) | {"mode": mode, "precision": precision}
            for size, mode, precision in product(
                MODELS, ("forward", "backward", "train"), ("fp32", "bf16")
            )
        ]
    if suite == "nsys":
        return [
            model
            | model_config(size)
            | {"context_length": length, "mode": mode, "nsys": True}
            for size, length, mode in product(
                ("small", "medium"), (256, 512, 1024), ("forward", "train")
            )
        ]
    if suite == "memory":
        return [
            model
            | model_config("xl")
            | {
                "context_length": length,
                "mode": mode,
                "precision": precision,
                "memory_snapshot": True,
            }
            for length, mode, precision in product(
                (128, 2048), ("forward", "train"), ("fp32", "bf16")
            )
        ]
    if suite == "checkpointing":
        return [
            model
            | model_config("xl")
            | {"context_length": 2048, "mode": "train", "checkpoint_layers": group}
            for group in (0, 1, 2, 4, 5, 6, 8, 16, 32)
        ]
    attention = {
        "experiment": "attention",
        "batch_size": 8,
        "warmup": 5,
        "steps": 100,
        "precision": "fp32",
        "causal": False,
        "compile": False,
        "timing": "wall",
    }
    if suite in ("attention", "compile"):
        rows = [
            attention | {"d_model": dim, "context_length": length, "compile": compiled}
            for dim, length, compiled in product(
                (16, 32, 64, 128),
                (256, 1024, 4096, 8192, 16384),
                (False, True) if suite == "compile" else (False,),
            )
        ]
        if suite == "compile":
            rows += [
                model | model_config(size) | {"mode": mode, "compile": compiled}
                for size, mode, compiled in product(
                    MODELS, ("forward", "backward", "train"), (False, True)
                )
            ]
        return rows
    if suite == "flash":
        return [
            attention
            | {
                "batch_size": 1,
                "causal": True,
                "timing": "triton",
                "gpu": "B200",
                "d_model": dim,
                "context_length": 2**power,
                "precision": precision,
            }
            for dim, power, precision in product(
                (16, 32, 64, 128), range(7, 17), ("fp32", "bf16")
            )
        ]
    if suite == "communication":
        return [
            {
                "experiment": "communication",
                "world_size": n,
                "message_mb": mb,
                "precision": "fp32",
            }
            for n, mb in product((2, 4, 6), (1, 10, 100, 1000))
        ]
    if suite in ("ddp", "sharding", "fsdp"):
        return [
            model
            | model_config("xl")
            | {
                "experiment": "ddp",
                "mode": "train",
                "world_size": 2,
                "gradient_reduce": reduction,
            }
            for reduction in (
                ("individual", "flat") if suite == "ddp" else ("individual",)
            )
        ]
    if suite == "leaderboard":
        return [
            model
            | {
                "experiment": "ddp",
                "mode": "train",
                "world_size": 2,
                "gpu": "B200",
                "d_model": 4096,
                "d_ff": 11008,
                "num_layers": 34,
                "num_heads": 32,
                "batch_size": 2,
                "context_length": 32768,
                "vocab_size": 151936,
                "precision": "bf16",
            }
        ]
    if suite == "accumulation":
        return [{"experiment": "accumulation"}]
    raise ValueError(suite)
