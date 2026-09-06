"""Stream a 1 GB Pile sample on Modal and train a 100K BPE vocabulary."""

from datetime import UTC
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cs336-pile-bpe")
results = modal.Volume.from_name("cs336-bpe-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install("regex>=2026.7.19", "zstandard", "psutil")
    .add_local_dir(ROOT / "src", remote_path="/root/src")
)


@app.function(
    image=image, volumes={"/results": results}, cpu=8, memory=12288, timeout=3600
)
def run() -> dict:
    import io
    import json
    import logging
    import sys
    import threading
    import time
    import urllib.request
    from collections import Counter
    from datetime import datetime

    import psutil
    import zstandard

    sys.path.insert(0, "/root/src")
    from cs336_basics.tokenizer.core import Tokenizer
    from cs336_basics.tokenizer.training import train_bpe

    logging.basicConfig(level=logging.INFO)
    url = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst"
    output = Path("/results") / datetime.now(UTC).strftime(
        "pile-1gb-100k-%Y%m%dT%H%M%SZ"
    )
    output.mkdir()
    corpus = Path("/tmp/pile-1gb.txt")
    sources = Counter()
    text_bytes = 0
    started = time.perf_counter()
    print(f"Preparing sample from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: SIM117
        with zstandard.ZstdDecompressor().stream_reader(response) as stream:
            with (
                io.TextIOWrapper(stream, encoding="utf-8") as lines,
                corpus.open("wb") as target,
            ):
                for line in lines:
                    document = json.loads(line)
                    text = document["text"].encode("utf-8")
                    target.write(text + b"<|endoftext|>\n")
                    text_bytes += len(text)
                    sources[
                        document.get("meta", {}).get("pile_set_name", "unknown")
                    ] += 1
                    if text_bytes >= 1_000_000_000:
                        break
    preparation_seconds = time.perf_counter() - started
    print(
        f"Prepared text_bytes={text_bytes} documents={sum(sources.values())} seconds={preparation_seconds:.2f}",
        flush=True,
    )

    peak = {"sampled_process_tree_rss_bytes": 0, "sampled_cgroup_memory_bytes": 0}
    stop = threading.Event()
    process = psutil.Process()
    memory_path = Path("/sys/fs/cgroup/memory.current")

    def monitor() -> None:
        while not stop.is_set():
            rss = 0
            for child in [process, *process.children(recursive=True)]:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak["sampled_process_tree_rss_bytes"] = max(
                peak["sampled_process_tree_rss_bytes"], rss
            )
            if memory_path.exists():
                peak["sampled_cgroup_memory_bytes"] = max(
                    peak["sampled_cgroup_memory_bytes"], int(memory_path.read_text())
                )
            print(
                f"training elapsed={time.perf_counter() - started:.1f}s tree_rss={rss / 2**20:.1f}MiB peak={peak}",
                flush=True,
            )
            stop.wait(10)

    started = time.perf_counter()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        vocab, merges = train_bpe(corpus, 100_000, ["<|endoftext|>"], workers=8)
    finally:
        stop.set()
        watcher.join()
    training_seconds = time.perf_counter() - started
    assert len(vocab) == 100_000, f"Only reached {len(vocab)} vocabulary entries"
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    tokenizer.save(output / "vocab.json", output / "merges.json")
    restored = Tokenizer.from_files(output / "vocab.json", output / "merges.json")
    example = "Hello, 世界! Testing BPE 123.\n<|endoftext|>"
    assert restored.decode(restored.encode(example)) == example
    report = dict(
        source=url,
        text_bytes=text_bytes,
        file_bytes=corpus.stat().st_size,
        documents=sum(sources.values()),
        sources=dict(sources),
        vocab_size=len(vocab),
        merges=len(merges),
        preparation_seconds=preparation_seconds,
        training_seconds=training_seconds,
        **peak,
        output=str(output),
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    results.commit()
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    run.remote()
