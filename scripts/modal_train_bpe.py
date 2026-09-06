"""Run the Assignment 1 BPE trainer on Modal (not invoked by this file)."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cs336-train-bpe")
data = modal.Volume.from_name("cs336-owt")
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install("regex>=2026.7.19")
    .add_local_dir(ROOT / "src", remote_path="/root/src")
)


@app.function(
    image=image,
    volumes={"/data": data.with_mount_options(read_only=True)},
    cpu=8,
    memory=12288,
    timeout=3600,
)
def train(
    input_path: str = "/data/data/TinyStoriesV2-GPT4-train.txt", vocab_size: int = 10_000
) -> None:
    import resource
    import sys
    import time

    sys.path.insert(0, "/root/src")
    from cs336_basics.tokenizer.training import train_bpe

    print(
        f"training input={input_path} bytes={Path(input_path).stat().st_size} vocab_size={vocab_size}",
        flush=True,
    )
    started = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"])
    elapsed = time.perf_counter() - started
    assert len(vocab) == vocab_size
    print(f"trained vocab={len(vocab)} merges={len(merges)} elapsed={elapsed:.3f}s")
    print(
        f"peak_process_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB"
    )


@app.local_entrypoint()
def main() -> None:
    train.remote()

