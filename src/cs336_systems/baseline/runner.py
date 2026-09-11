"""Local/Modal execution and Nsight capture for every baseline experiment."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def command(config):
    argv = [sys.executable, "-m", "cs336_systems.baseline"]
    for key, value in config.items():
        if value is None or value is False or key in {"case_name"}:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is not True:
            argv.append(str(value))
    return argv


def run_case(args, output):
    import torch

    from .common import environment

    config = {
        k: v
        for k, v in vars(args).items()
        if k not in {"output_dir", "worker", "capture"}
    }
    report: dict
    try:
        if args.experiment == "transformer":
            from .transformer import benchmark
        elif args.experiment == "attention":
            from .attention import benchmark
        elif args.experiment in ("communication", "ddp"):
            from .distributed import benchmark
        else:
            from .precision import benchmark
        report = benchmark(args, output) | environment(args) | {"status": "ok"}
    except torch.OutOfMemoryError as error:
        report = {"status": "oom", "error": str(error)}
    report["config"] = config
    report["profiled"] = args.capture
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def execute(args, output):
    if not args.nsys:
        run_case(args, output)
        return
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("nsys is not installed or not on PATH")
    child = vars(args) | {
        "nsys": False,
        "modal": False,
        "worker": True,
        "capture": True,
        "output_dir": str(output),
    }
    cmd = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-memory-usage=true",
        "--pytorch=autograd-shapes-nvtx,functions-trace",
        "--output=" + str(output / "trace"),
    ]
    subprocess.run(cmd + command(child), check=True)
    if (output / "trace.nsys-rep").exists():
        with (output / "trace.stats.log").open("w") as log:
            subprocess.run(
                [nsys, "stats", str(output / "trace.nsys-rep")],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def remote_run(config):
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(
            command(config | {"modal": False, "worker": True, "output_dir": folder}),
            check=True,
        )
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix in {".json", ".nsys-rep", ".log", ".pickle"}:
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        yield str(path.relative_to(folder)), chunk


def modal_run(args, output):
    import modal

    image = (
        modal.Image.from_registry("ubuntu:24.04", add_python="3.14")
        .apt_install("ca-certificates", "build-essential")
        .uv_pip_install(
            "torch==2.14.0.dev20260811+cu130",
            "numpy",
            "regex",
            "einops",
            extra_index_url="https://download.pytorch.org/whl/nightly/cu130",
        )
    )
    if args.nsys:
        image = image.apt_install("wget").run_commands(
            "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb",
            "dpkg -i /tmp/cuda-keyring.deb",
            "apt-get update && apt-get install -y nsight-systems-2026.1.3",
        )
    image = image.env({"PYTHONPATH": "/root/project/src"}).add_local_dir(
        Path(__file__).resolve().parents[2], "/root/project/src"
    )
    gpu = args.gpu
    if args.experiment in ("ddp", "communication"):
        gpu += f":{args.world_size}"
    app = modal.App("cs336-a2-baseline")
    run = app.function(image=image, gpu=gpu, cpu=4, timeout=1800)(remote_run)
    with modal.enable_output(), app.run():
        for name, chunk in run.remote_gen(vars(args) | {"output_dir": None}):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("ab") as destination:
                destination.write(chunk)


def launch(args, output):
    output.mkdir(parents=True, exist_ok=True)
    if args.modal:
        modal_run(args, output)
    else:
        execute(args, output)
    path = output / "metrics.json"
    report = json.loads(path.read_text())
    report["config"] = {
        k: v
        for k, v in vars(args).items()
        if k not in {"output_dir", "worker", "capture"}
    }
    report["profiled"] = args.nsys or args.capture
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_suite(args, configurations, output):
    import csv

    rows: list[dict] = []
    for index, overrides in configurations:
        name = overrides.get("case_name", f"{args.suite}-{index:03}")
        config = (
            vars(args)
            | overrides
            | {
                "suite": None,
                "case": None,
                "list_cases": False,
                "worker": True,
                "output_dir": str(output / name),
            }
        )
        # Each case starts in a fresh process, so allocator/compile state cannot leak.
        child = subprocess.run(command(config), env=os.environ.copy(), check=False)
        path = output / name / "metrics.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else {"status": "error", "exit_code": child.returncode}
        )
        result["case"] = name
        result.setdefault("config", config)
        rows.append(result)
        (output / "suite.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = (
        "case",
        "status",
        "experiment",
        "model_size",
        "mode",
        "precision",
        "compile",
        "batch_size",
        "context_length",
        "d_model",
        "checkpoint_layers",
        "world_size",
        "mean_ms",
        "stdev_ms",
        "forward_ms",
        "backward_ms",
        "peak_allocated_mib",
    )
    with (output / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in rows:
            row = result.get("config", {}) | result
            for phase in ("forward", "backward"):
                row[phase + "_ms"] = result.get(phase, {}).get("mean_ms")
            writer.writerow(row)
