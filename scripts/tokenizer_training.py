from pathlib import Path

from tokenizer import *

ROOT = Path(__file__).resolve().parents[1]
corpus = Corpus(ROOT / "data" / "owt.txt")

save_vocabulary(
    training_loop_optimized([corpus], lambda count, term: term >= 10_000),
    ROOT / "data" / "owt.json",
)
