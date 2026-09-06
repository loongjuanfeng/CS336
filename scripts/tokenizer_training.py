from pathlib import Path

from cs336_basics.tokenizer.core import Tokenizer
from cs336_basics.tokenizer.training import train_bpe

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    vocab, merges = train_bpe(ROOT / "data" / "owt.txt", 256 + 10_000, [])
    Tokenizer(vocab, merges).save(
        ROOT / "data" / "owt-vocab.json", ROOT / "data" / "owt-merges.json"
    )


if __name__ == "__main__":
    main()
