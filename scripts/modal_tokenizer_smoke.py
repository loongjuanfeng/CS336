from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cs336-tokenizer-smoke")
volume = modal.Volume.from_name("cs336-owt")
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install("regex>=2026.7.19")
    .add_local_dir(ROOT / "src", remote_path="/root/src")
)


@app.function(
    image=image,
    volumes={"/data": volume.with_mount_options(read_only=True)},
    cpu=2,
    memory=4096,
    timeout=600,
)
def smoke_test() -> None:
    import sys
    import time

    sys.path.insert(0, "/root/src")

    from tokenizer import Corpus, training_loop_optimized

    sample_path = Path("/tmp/owt-smoke.txt")
    with open("/data/data/owt.txt", "rb") as source, sample_path.open("wb") as target:
        target.write(source.read(1_000_000))

    started = time.perf_counter()
    vocabulary = training_loop_optimized(
        [Corpus(sample_path)], lambda _count, term: term >= 10
    )
    elapsed = time.perf_counter() - started
    print(f"smoke OK: vocabulary={len(vocabulary)} elapsed={elapsed:.3f}s")


@app.local_entrypoint()
def main() -> None:
    smoke_test.remote()
