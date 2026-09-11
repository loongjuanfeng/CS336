"""Reproducible local/Modal correctness validation, with logs on every outcome."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    digest = hashlib.sha256()
    paths = [
        p
        for folder in ("src", "tests")
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc", ".pyo"}
    ]
    paths += [ROOT / name for name in ("pyproject.toml", "uv.lock")]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def run_checks(output, full=False):
    import torch
    import triton

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "OMP_NUM_THREADS": "2",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "USE_LIBUV": "0",
        "GLOO_SOCKET_IFNAME": "lo",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    distributed = [
        f"tests/assignment2/test_{name}.py"
        for name in ("ddp", "sharded_optimizer", "fsdp")
    ]
    jobs = [("attention", ["tests/assignment2/test_attention.py"], {})]
    jobs += [("edges", ["tests/local/test_systems.py", "-k", "not distributed"], {})]
    for iteration in range(5 if full else 1):
        if full:
            jobs.append(
                (f"gloo-{iteration}", distributed, {"CUDA_VISIBLE_DEVICES": ""})
            )
        jobs.append(
            (
                f"edges-gloo-{iteration}",
                ["tests/local/test_systems.py", "-k", "distributed"],
                {"CUDA_VISIBLE_DEVICES": "", "CS336_TEST_BACKEND": "gloo"},
            )
        )
        if full:
            jobs.append(
                (
                    f"nccl-{iteration}",
                    ["tests/local/test_systems.py", "-k", "distributed"],
                    {"CS336_TEST_BACKEND": "nccl"},
                )
            )
    report = {
        "source_sha256": source_hash(),
        "full": full,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "validation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "jobs": [],
        "status": "running",
    }
    if not torch.cuda.is_available() or (full and torch.cuda.device_count() < 2):
        raise RuntimeError("Validation requires one local GPU or two Modal GPUs")
    destination = output / "validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for name, selection, overrides in jobs:
        command = [
            sys.executable,
            "-m",
            "pytest",
            *selection,
            "-q",
            "--tb=short",
            "--timeout=180",
            "--timeout-method=thread",
            f"--junitxml={output / (name + '.xml')}",
        ]
        started = time.monotonic()
        with (output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env | overrides,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                # A timed-out pytest can leave multiprocessing children alive.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
        xml = output / (name + ".xml")
        skipped = (
            sum(
                int(s.get("skipped", 0))
                for s in ET.parse(xml).getroot().iter("testsuite")
            )
            if xml.exists()
            else 0
        )
        # CPU selections omit CUDA-only tests; required validation has no skips.
        passed = code == 0 and xml.exists() and skipped == 0
        report["jobs"].append(
            {
                "name": name,
                "command": command,
                "environment": overrides,
                "exit_code": code,
                "skipped": skipped,
                "seconds": time.monotonic() - started,
                "passed": passed,
            }
        )
        report["status"] = "running" if passed else "failed"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{name}: {'passed' if passed else 'FAILED'} ({time.monotonic() - started:.1f}s)",
            flush=True,
        )
        if not passed:
            print((output / (name + ".log")).read_text(), flush=True)
            return 1
    report["status"] = "passed"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def remote_checks():
    with tempfile.TemporaryDirectory() as directory:
        process = subprocess.run(
            [
                sys.executable,
                "/root/project/scripts/verify_a2.py",
                "--worker",
                "--output-dir",
                directory,
            ],
            check=False,
        )
        for path in sorted(Path(directory).glob("*")):
            yield path.name, path.read_bytes()
        yield "remote-exit.json", json.dumps({"exit_code": process.returncode}).encode()


def modal_checks(output):
    import modal

    image = (
        modal.Image.from_registry("ubuntu:24.04", add_python="3.14")
        .apt_install("ca-certificates", "build-essential")
        .uv_pip_install(
            "torch==2.14.0.dev20260811+cu130",
            "numpy",
            "regex",
            "einops",
            "pytest",
            "pytest-timeout",
            "jaxtyping",
            "psutil",
            extra_index_url="https://download.pytorch.org/whl/nightly/cu130",
        )
        .env({"PYTHONPATH": "/root/project/src"})
    )
    for folder in ("src", "tests", "scripts"):
        image = image.add_local_dir(
            ROOT / folder, f"/root/project/{folder}", ignore=["**/__pycache__/**"]
        )
    for name in (
        "pyproject.toml",
        "uv.lock",
        "justfile",
        "test_and_make_submission.sh",
    ):
        image = image.add_local_file(ROOT / name, f"/root/project/{name}")
    app = modal.App("cs336-a2-validation")
    remote = app.function(image=image, gpu="A100-80GB:2", cpu=4, timeout=1800)(
        remote_checks
    )
    with modal.enable_output(), app.run():
        for name, content in remote.remote_gen():
            (output / name).write_bytes(content)
    status = json.loads((output / "remote-exit.json").read_text())
    report = json.loads((output / "validation.json").read_text())
    if report["source_sha256"] != source_hash():
        raise RuntimeError(
            "Source changed during validation; this run cannot certify the current tree"
        )
    return status["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        parent = ROOT / "results/assignment2"
        parent.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(
            tempfile.mkdtemp(prefix="modal-" if args.modal else "local-", dir=parent)
        )
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validation: {args.output_dir}", flush=True)
    code = (
        modal_checks(args.output_dir)
        if args.modal
        else run_checks(args.output_dir, full=args.worker)
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
