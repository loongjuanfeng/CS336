import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLanguageModel,
    scaled_dot_product_attention,
    silu,
    softmax,
)


def _rotate_reference(x: Tensor, angles: Tensor) -> Tensor:
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    cosine = angles.cos().to(x.dtype)
    sine = angles.sin().to(x.dtype)
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).reshape_as(x)


def _rope_angles(token_positions: Tensor, d_k: int, theta: float) -> Tensor:
    frequencies = theta ** (-2.0 * torch.arange(d_k // 2) / d_k)
    return token_positions.to(torch.float32).unsqueeze(-1) * frequencies


def test_linear_matches_pytorch_and_uses_assignment_initialization():
    torch.manual_seed(0)
    layer = Linear(7, 11)
    inputs = torch.randn(2, 3, 7)
    standard_deviation = math.sqrt(2.0 / (7 + 11))

    assert not isinstance(layer, nn.Linear)
    assert layer.weight.shape == (11, 7)
    assert layer.weight.abs().max() <= 3.0 * standard_deviation
    torch.testing.assert_close(layer(inputs), F.linear(inputs, layer.weight))


def test_embedding_matches_indexing_and_uses_assignment_initialization():
    torch.manual_seed(0)
    layer = Embedding(13, 5)
    token_ids = torch.tensor([[0, 4, 12], [3, 3, 1]])

    assert not isinstance(layer, nn.Embedding)
    assert layer.weight.shape == (13, 5)
    assert layer.weight.abs().max() <= 3.0
    torch.testing.assert_close(layer(token_ids), layer.weight[token_ids])


def test_rms_norm_computes_in_float32_and_restores_dtype():
    layer = RMSNorm(4, eps=1e-5, dtype=torch.bfloat16)
    layer.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    inputs = torch.tensor(
        [[1000.0, 1.0, -1000.0, 0.25], [0.5, -2.0, 3.0, -4.0]],
        dtype=torch.bfloat16,
    )
    working = inputs.to(torch.float32)
    expected = working * torch.rsqrt(
        working.square().mean(dim=-1, keepdim=True) + layer.eps
    )
    expected = (expected * layer.weight.to(torch.float32)).to(inputs.dtype)

    output = layer(inputs)

    assert output.dtype == inputs.dtype
    torch.testing.assert_close(output, expected)


def test_silu_and_swiglu_match_pytorch_references():
    inputs = torch.tensor([-1000.0, -30.0, -1.0, 0.0, 1.0, 30.0, 1000.0])
    torch.testing.assert_close(silu(inputs), F.silu(inputs))
    assert torch.isfinite(silu(inputs)).all()

    torch.manual_seed(1)
    layer = SwiGLU(6, 10)
    batched_inputs = torch.randn(2, 3, 4, 6)
    expected = F.linear(
        F.silu(F.linear(batched_inputs, layer.w1.weight))
        * F.linear(batched_inputs, layer.w3.weight),
        layer.w2.weight,
    )
    torch.testing.assert_close(layer(batched_inputs), expected)


def test_softmax_is_stable_and_preserves_dtype():
    inputs = torch.tensor(
        [[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]],
        dtype=torch.float16,
    )

    output = softmax(inputs, dim=-1)

    assert output.dtype == inputs.dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, F.softmax(inputs, dim=-1))


def test_scaled_dot_product_attention_matches_reference_with_batch_axes():
    torch.manual_seed(2)
    queries = torch.randn(2, 3, 4, 5)
    keys = torch.randn(2, 3, 6, 5)
    values = torch.randn(2, 3, 6, 7)
    mask = torch.rand(2, 1, 4, 6) > 0.3
    mask[..., 0] = True
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
    expected = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1) @ values

    torch.testing.assert_close(
        scaled_dot_product_attention(queries, keys, values, mask),
        expected,
    )


def test_rope_uses_cached_buffers_and_supports_arbitrary_batch_dimensions():
    torch.manual_seed(3)
    theta = 10_000.0
    d_k = 8
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len=16)
    inputs = torch.randn(2, 3, 4, 5, d_k)
    positions = torch.randint(0, 16, (2, 3, 4, 5))
    expected = _rotate_reference(inputs, _rope_angles(positions, d_k, theta))
    buffers = dict(rope.named_buffers())

    assert buffers["cos"].shape == (16, d_k // 2)
    assert buffers["sin"].shape == (16, d_k // 2)
    assert rope.state_dict() == {}
    torch.testing.assert_close(rope(inputs, positions), expected)


def test_attention_rope_broadcasts_when_batch_size_differs_from_head_count():
    torch.manual_seed(4)
    batch_size = 3
    num_heads = 2
    sequence_length = 5
    d_model = 8
    head_dimension = d_model // num_heads
    theta = 10_000.0
    rope = RotaryPositionalEmbedding(theta, head_dimension, max_seq_len=12)
    attention = MultiHeadSelfAttention(d_model, num_heads, rope)
    inputs = torch.randn(batch_size, sequence_length, d_model)
    positions = torch.tensor(
        [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 3, 5, 7, 9]],
    )

    queries = F.linear(inputs, attention.q_proj.weight)
    keys = F.linear(inputs, attention.k_proj.weight)
    values = F.linear(inputs, attention.v_proj.weight)
    queries = queries.reshape(
        batch_size, sequence_length, num_heads, head_dimension
    ).transpose(1, 2)
    keys = keys.reshape(
        batch_size, sequence_length, num_heads, head_dimension
    ).transpose(1, 2)
    values = values.reshape(
        batch_size, sequence_length, num_heads, head_dimension
    ).transpose(1, 2)
    angles = _rope_angles(positions, head_dimension, theta).unsqueeze(1)
    queries = _rotate_reference(queries, angles)
    keys = _rotate_reference(keys, angles)
    causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool).tril()
    attended = (
        torch.softmax(
            (queries @ keys.transpose(-2, -1) / math.sqrt(head_dimension)).masked_fill(
                ~causal_mask,
                -torch.inf,
            ),
            dim=-1,
        )
        @ values
    )
    attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, d_model)
    expected = F.linear(attended, attention.output_proj.weight)

    torch.testing.assert_close(attention(inputs, positions), expected)


def test_attention_is_causal_under_prefix_extension():
    torch.manual_seed(5)
    rope = RotaryPositionalEmbedding(10_000.0, d_k=4, max_seq_len=8)
    attention = MultiHeadSelfAttention(d_model=12, num_heads=3, rope=rope)
    inputs = torch.randn(2, 7, 12)

    full_output = attention(inputs)
    prefix_output = attention(inputs[:, :4])

    torch.testing.assert_close(full_output[:, :4], prefix_output, atol=1e-6, rtol=1e-6)


def test_transformer_block_matches_its_pre_norm_residual_definition():
    torch.manual_seed(6)
    block = TransformerBlock(8, 2, 16, max_seq_len=6, theta=10_000.0)
    inputs = torch.randn(2, 6, 8)
    after_attention = inputs + block.attn(block.ln1(inputs))
    expected = after_attention + block.ffn(block.ln2(after_attention))

    torch.testing.assert_close(block(inputs), expected)


def test_language_model_has_independent_embeddings_and_causal_logits():
    torch.manual_seed(7)
    model = TransformerLanguageModel(
        vocab_size=17,
        context_length=8,
        d_model=12,
        num_layers=2,
        num_heads=3,
        d_ff=20,
        rope_theta=10_000.0,
    )
    input_ids = torch.randint(0, 17, (2, 7))

    full_logits = model(input_ids)
    prefix_logits = model(input_ids[:, :4])

    assert full_logits.shape == (2, 7, 17)
    assert model.token_embeddings.weight.data_ptr() != model.lm_head.weight.data_ptr()
    torch.testing.assert_close(full_logits[:, :4], prefix_logits, atol=1e-5, rtol=1e-5)
