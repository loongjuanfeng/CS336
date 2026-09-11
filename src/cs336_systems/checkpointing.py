"""Recursive recomputation using PyTorch's native non-reentrant checkpoint."""

from torch.utils.checkpoint import checkpoint


def memory_optimal_forward(blocks, inputs):
    """Checkpoint nested prefixes, prioritizing saved activations over compute."""
    blocks = tuple(blocks)
    if not blocks:
        return inputs
    if len(blocks) == 1:
        return checkpoint(blocks[0], inputs, use_reentrant=False)
    prefix = checkpoint(
        lambda x: memory_optimal_forward(blocks[:-1], x),
        inputs,
        use_reentrant=False,
    )
    return checkpoint(blocks[-1], prefix, use_reentrant=False)
