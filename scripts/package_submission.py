"""Pack the source and test fixtures without local environments or training data."""

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
)
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".ruff_cache"}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for relative in INCLUDE:
            path = ROOT / relative
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in paths:
                if not file.is_file() or file.is_symlink():
                    continue
                if SKIP.intersection(file.parts) or file.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(file, file.relative_to(ROOT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
