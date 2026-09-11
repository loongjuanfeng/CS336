"""Tiled online-softmax attention and recomputed, compiled backward tiles."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _validate(q, k, v):
    if q.ndim < 2 or k.ndim != q.ndim or k.shape != v.shape:
        raise ValueError("Expected Q (..., Nq, D), K/V (..., Nk, D)")
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Batch dimensions and head dimensions must match")
    if min(*q.shape, *k.shape) < 1:
        raise ValueError("Attention dimensions must be positive")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("Q/K/V must have the same device and dtype")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Supported dtypes: float32, float16, bfloat16")


def _pytorch_forward(q, k, v, causal):
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5
    for i in range(0, q.shape[-2], 64):
        qi = q[..., i : i + 64, :].float()
        m = torch.full(qi.shape[:-1], -torch.inf, device=q.device)
        z = torch.zeros_like(m)
        acc = torch.zeros_like(qi)
        for j in range(0, k.shape[-2], 64):
            kj = k[..., j : j + 64, :].float()
            scores = (qi @ kj.transpose(-1, -2)) * scale
            if causal:
                rows = torch.arange(i, i + qi.shape[-2], device=q.device)
                cols = torch.arange(j, j + kj.shape[-2], device=q.device)
                scores = scores.masked_fill(rows[:, None] < cols, -torch.inf)
            new_m = torch.maximum(m, scores.amax(-1))
            alpha = (m - new_m).exp()
            p = (scores - new_m.unsqueeze(-1)).exp()
            acc = acc * alpha.unsqueeze(-1) + p @ v[..., j : j + 64, :].float()
            z = z * alpha + p.sum(-1)
            m = new_m
        output[..., i : i + 64, :] = acc / z.unsqueeze(-1)
        lse[..., i : i + 64] = m + z.log()
    return output, lse


@triton.jit
def _forward_kernel(
    Q,
    K,
    V,
    O,
    L,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    QB: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KN: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VN: tl.constexpr,
    VD: tl.constexpr,
    CAUSAL: tl.constexpr,
    BD: tl.constexpr,
    B: tl.constexpr,
):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * B + tl.arange(0, B)
    dims = tl.arange(0, BD)
    q = tl.load(
        Q + batch * QB + rows[:, None] * QN + dims[None, :] * QD,
        (rows[:, None] < NQ) & (dims[None, :] < D),
        0,
    )
    m = tl.full((B,), -float("inf"), tl.float32)
    z = tl.full((B,), 0, tl.float32)
    acc = tl.full((B, BD), 0, tl.float32)
    end = NK
    if CAUSAL:
        end = tl.minimum(NK, (tl.program_id(0) + 1) * B)
    for start in range(0, end, B):
        cols = start + tl.arange(0, B)
        k = tl.load(
            K + batch * KB + dims[:, None] * KD + cols[None, :] * KN,
            (dims[:, None] < D) & (cols[None, :] < NK),
            0,
        )
        scores = tl.dot(q, k, input_precision="ieee") * (D**-0.5)
        valid = cols[None, :] < NK
        if CAUSAL:
            valid = valid & (rows[:, None] >= cols[None, :])
        scores = tl.where(valid, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))  # ty: ignore[invalid-argument-type]
        p = tl.exp(scores - new_m[:, None])
        alpha = tl.exp(m - new_m)
        v = tl.load(
            V + batch * VB + cols[:, None] * VN + dims[None, :] * VD,
            (cols[:, None] < NK) & (dims[None, :] < D),
            0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        z = z * alpha + tl.sum(p, 1)  # ty: ignore[invalid-argument-type]
        m = new_m
    tl.store(
        O + batch * NQ * D + rows[:, None] * D + dims[None, :],
        acc / z[:, None],
        (rows[:, None] < NQ) & (dims[None, :] < D),
    )
    tl.store(L + batch * NQ + rows, m + tl.log(z), rows < NQ)


def launch_flash_forward(q, k, v, is_causal=False):
    _validate(q, k, v)
    if q.device.type != "cuda":
        raise ValueError("Triton attention requires CUDA")
    shape = q.shape
    q, k, v = [t.reshape(-1, t.shape[-2], t.shape[-1]) for t in (q, k, v)]
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    _forward_kernel[((q.shape[1] + 31) // 32, q.shape[0])](
        q,
        k,
        v,
        output,
        lse,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        is_causal,  # ty: ignore[invalid-argument-type]
        max(16, 1 << (q.shape[2] - 1).bit_length()),  # ty: ignore[invalid-argument-type]
        32,  # ty: ignore[invalid-argument-type]
    )
    return output.reshape(shape), lse.reshape(shape[:-1])


def _backward_tile(q, k, v, do, lse, delta, mask):
    scale = q.shape[-1] ** -0.5
    q, k, v, do = (t.float() for t in (q, k, v, do))
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    p = (scores - lse.unsqueeze(-1)).exp()
    ds = p * (do @ v.transpose(-1, -2) - delta.unsqueeze(-1)) * scale
    return ds @ k, ds.transpose(-1, -2) @ q, p.transpose(-1, -2) @ do


# Compile bounded tiles, so long sequences do not create a huge unrolled graph.
_compiled_backward_tile = torch.compile(_backward_tile, fullgraph=True, dynamic=True)


def _backward(q, k, v, output, grad_output, logsumexp, causal, tile_fn):
    dq, dk, dv = [torch.zeros_like(t, dtype=torch.float32) for t in (q, k, v)]
    delta = (output.float() * grad_output.float()).sum(-1)
    # Normalize dtype/strides once, avoiding a compiler specialization per input
    # dtype and layout. Accumulation is FP32 for all supported precisions.
    qf, kf, vf, do = [
        F.pad(t.float(), (0, 0, 0, -t.shape[-2] % 64)) for t in (q, k, v, grad_output)
    ]
    logsumexp = F.pad(logsumexp, (0, -q.shape[-2] % 64))
    delta = F.pad(delta, (0, -q.shape[-2] % 64))
    for i in range(0, q.shape[-2], 64):
        rows = torch.arange(i, i + 64, device=q.device)
        for j in range(0, k.shape[-2], 64):
            if causal and j >= i + 64:
                break
            cols = torch.arange(j, j + 64, device=q.device)
            mask = (cols[None, :] < k.shape[-2]).expand(64, 64)
            if causal:
                mask = mask & (rows[:, None] >= cols[None, :])
            a, b, c = tile_fn(
                qf[..., i : i + 64, :].contiguous(),
                kf[..., j : j + 64, :].contiguous(),
                vf[..., j : j + 64, :].contiguous(),
                do[..., i : i + 64, :].contiguous(),
                logsumexp[..., i : i + 64].contiguous(),
                delta[..., i : i + 64].contiguous(),
                mask.contiguous(),
            )
            dq[..., i : i + 64, :] += a[..., : min(64, q.shape[-2] - i), :]
            dk[..., j : j + 64, :] += b[..., : min(64, k.shape[-2] - j), :]
            dv[..., j : j + 64, :] += c[..., : min(64, k.shape[-2] - j), :]
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def flash_backward_reference(q, k, v, output, grad_output, logsumexp, is_causal=False):
    return _backward(q, k, v, output, grad_output, logsumexp, is_causal, _backward_tile)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        _validate(q, k, v)
        output, lse = _pytorch_forward(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        q, k, v, output, lse = ctx.saved_tensors
        grads = _backward(
            q, k, v, output, grad_output, lse, ctx.is_causal, _compiled_backward_tile
        )
        return (*grads, None)[: len(ctx.needs_input_grad)]


class FlashAttentionTriton(FlashAttentionPytorch):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        output, lse = launch_flash_forward(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output
