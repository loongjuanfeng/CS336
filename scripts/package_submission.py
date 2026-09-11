"""Pack the source and test fixtures without local environments or training data."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist/cs336-2026-assignment2.zip"
# An allowlist avoids accidentally including .env, wheels or large experiment outputs.
INCLUDE = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "justfile",
    ".gitignore",
    ".envrc",
    "test_and_make_submission.sh",
    "src",
    "tests",
    "scripts",
    "examples",
    "docs",
    "results/assignment2.xml",
    "results/assignment2/regression.xml",
    "results/assignment2/regression.log",
    "results/assignment2/clean-install.xml",
)
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".ruff_cache"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation", type=Path, help="Require a matching full validation report"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.validation:
        from verify_a2 import source_hash

        report = json.loads((args.validation / "validation.json").read_text())
        if not (
            report["status"] == "passed"
            and report["full"]
            and report["source_sha256"] == source_hash()
        ):
            raise SystemExit(
                "Full validation is missing, failed, or belongs to different source"
            )
        if len(report["jobs"]) != 17 or not all(
            job["passed"] for job in report["jobs"]
        ):
            raise SystemExit(
                "Expected all 17 validation jobs, including five distributed repetitions"
            )
        expected = {"attention", "edges"} | {
            f"{kind}-{iteration}"
            for kind in ("gloo", "edges-gloo", "nccl")
            for iteration in range(5)
        }
        if {job["name"] for job in report["jobs"]} != expected:
            raise SystemExit("Required validation jobs are missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        for relative in INCLUDE:
            path = ROOT / relative
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in paths:
                if not file.is_file() or file.is_symlink():
                    continue
                if SKIP.intersection(file.parts) or file.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(file, file.relative_to(ROOT))
        if args.validation:
            for file in sorted(args.validation.iterdir()):
                if file.is_file() and file.suffix in {".json", ".log", ".xml"}:
                    archive.write(
                        file, Path("results/assignment2/verified") / file.name
                    )
    print(args.output)
    print("sha256=" + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
