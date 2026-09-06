"""Regressions at the A1/A2 boundary, beyond upstream coverage."""

import numpy as np
import pytest
import torch

from cs336_basics.model.primitives import Linear
from cs336_basics.model.transformer import TransformerBlock, TransformerLanguageModel
from cs336_basics.tokenizer.core import Tokenizer
from cs336_basics.tokenizer.training import train_bpe
from cs336_basics.training import cross_entropy


def test_model_uses_canonical_layer_types():
    model = TransformerLanguageModel(16, 8, 16, 1, 2, 32, 10000)
    for layer in model.layers:
        assert isinstance(layer, TransformerBlock)
        assert isinstance(layer.attn.q_proj, Linear)


def test_merge_order_is_independent_of_token_ids(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"bc"})
    tokenizer = Tokenizer(vocab, [(b"b", b"c"), (b"a", b"b")], ["<end>"])
    assert tokenizer.encode("abc") == [ord("a"), 257]
    tokenizer.save(tmp_path / "vocab.json", tmp_path / "merges.json")
    restored = Tokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.json")
    assert restored.merges == tokenizer.merges
    assert restored.encode("abc<end>") == tokenizer.encode("abc<end>")


def test_learned_vocabulary_keeps_merges_when_saved(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("banana banana banana")
    vocab, merges = train_bpe(corpus, 260, [])
    tokenizer = Tokenizer(vocab, merges)
    tokenizer.save(tmp_path / "vocab.json", tmp_path / "merges.json")
    restored = Tokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.json")
    assert restored.vocab == vocab
    assert restored.merges == merges
    vocab, merges = train_bpe(corpus, 256, [])
    assert len(vocab) == 256 and merges == []


@pytest.mark.parametrize(
    "text",
    [
        "hello world! I'm testing contractions.\n\n Next line",
        "<end><end>hello<end> world\n\n!",
        "   hello\n \t\nthere! é😃 <end",
    ],
)
def test_streaming_matches_whole_text_at_every_split(text):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"he", 257: b"hel", 258: b"hell", 259: b"hello", 260: b"\n\n"})
    tokenizer = Tokenizer(
        vocab,
        [(b"h", b"e"), (b"he", b"l"), (b"hel", b"l"), (b"hell", b"o"), (b"\n", b"\n")],
        ["<end>", "<end><end>"],
    )
    expected = tokenizer.encode(text)
    for split in range(len(text) + 1):
        assert list(tokenizer.encode_iterable([text[:split], text[split:]])) == expected
    assert list(tokenizer.encode_iterable(iter(text))) == expected


def test_streaming_does_not_consume_whole_input():
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [])
    consumed = 0

    def chunks():
        nonlocal consumed
        for _ in range(1000):
            consumed += 1
            yield "hello world! "

    assert next(tokenizer.encode_iterable(chunks())) == ord("h")
    assert consumed == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_step_and_checkpoint(tmp_path):
    from cs336_basics.optimization import AdamW
    from cs336_basics.training import get_batch, load_checkpoint, save_checkpoint

    torch.manual_seed(4)
    model = TransformerLanguageModel(16, 8, 16, 1, 2, 32, 10000, device="cuda")
    optimizer = AdamW(model.parameters())
    x, y = get_batch(np.arange(100) % 16, 2, 8, "cuda")
    loss = cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    save_checkpoint(model, optimizer, 1, tmp_path / "checkpoint.pt")
    restored = TransformerLanguageModel(16, 8, 16, 1, 2, 32, 10000, device="cuda")
    restored_optimizer = AdamW(restored.parameters())
    assert (
        load_checkpoint(tmp_path / "checkpoint.pt", restored, restored_optimizer) == 1
    )
    for current, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad()
        cross_entropy(current(x), y).backward()
        opt.step()
    torch.testing.assert_close(restored(x), model(x))
