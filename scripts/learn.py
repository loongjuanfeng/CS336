"""A1 decoder-only Transformer, using only PyTorch and the standard library.

Read bottom-up: Transformer -> Block -> Attention / SwiGLU -> rope.
B=batch, T=tokens, D=model width, H=heads, K=D/H, F=FFN width, V=vocab.

    python scripts/learn.py --check
    just trace scripts/learn.py --preset medium
    python -m pdb scripts/learn.py --preset small --warmup 0 --steps 1
    # At (Pdb): b train_step, then c; use n / s / p q.shape inside forward.

Random tokens are a fixed timing workload, not a language-learning dataset.
Modes: forward; backward = forward + loss + backward; train adds AdamW.
This is an educational A2 optimization target, not the official A1 baseline.
Presets target a 4 GiB GTX 1650 in FP32, not the official A2 model sizes.
CUDA emits nested NVTX ranges automatically; CPU uses no-op contexts.
Use eager mode for PDB / detailed NVTX; --compile keeps only outer step ranges.
"""

import argparse
import math
import statistics
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

#                         V     T    D  L  H     F  B
PRESETS = {
    "small":  (256,  128, 128, 2, 4,  352, 4),
    "medium": (4096, 256, 256, 4, 4,  704, 4),
    "large":  (8192, 512, 512, 6, 8, 1408, 2),
}


def region(name, tensor):
    # Keep Python annotations outside compiled graphs; never synchronize here.
    if not torch.compiler.is_compiling() and tensor.is_cuda:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def rope(x, cos, sin):
    """Rotate adjacent channel pairs; rotating Q and K encodes relative position."""
    cos, sin = cos[: x.size(-2)].to(x.dtype), sin[: x.size(-2)].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, rope_theta, sdpa):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.sdpa = sdpa
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        angles = torch.outer(
            torch.arange(context_length, dtype=torch.float32),
            rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
        )
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        with region("attention/qkv", x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            # [B,T,D] -> [B,H,T,K]: each head mixes tokens independently.
            q = q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
            k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
            v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        with region("attention/rope", x):
            q = rope(q, self.cos, self.sin)
            k = rope(k, self.cos, self.sin)
        if self.sdpa:
            with region("attention/sdpa", x):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # ponytail: O(T^2) scores expose the A2 bottleneck; compare with --sdpa.
            with region("attention/qk", x):
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            with region("attention/mask_softmax", x):
                future = torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool).triu(1)
                weights = F.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
            with region("attention/pv", x):
                y = weights @ v  # [B,H,T,T] @ [B,H,T,K] -> [B,H,T,K]
        with region("attention/output", x):
            y = y.transpose(1, 2).flatten(-2)
            return self.output_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        # Unlike attention, this mixes channels, independently at each token.
        with region("ffn/gate_up", x):
            gate = F.silu(self.w1(x))
            up = self.w3(x)
            gated = gate * up
        with region("ffn/down", x):
            return self.w2(gated)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, rope_theta, sdpa):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = Attention(d_model, num_heads, context_length, rope_theta, sdpa)
        self.ln2 = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        # Pre-norm: normalize the branch input, preserve the residual stream.
        with region("block/attention_norm", x):
            normalized = self.ln1(x)
        with region("block/attention", x):
            attended = self.attn(normalized)
        with region("block/attention_residual", x):
            x = x + attended
        with region("block/ffn_norm", x):
            normalized = self.ln2(x)
        with region("block/ffn", x):
            transformed = self.ffn(normalized)
        with region("block/ffn_residual", x):
            return x + transformed


class Transformer(nn.Module):
    def __init__(self, vocab_size=256, context_length=128, d_model=128,
                 num_layers=2, num_heads=4, d_ff=352, rope_theta=10_000.0, sdpa=False):
        super().__init__()
        if min(vocab_size, context_length, d_model, num_layers, num_heads, d_ff) <= 0:
            raise ValueError("Model dimensions must be positive")
        if d_model % num_heads or (d_model // num_heads) % 2:
            raise ValueError("d_model must divide into heads with even width for RoPE")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            Block(d_model, num_heads, d_ff, context_length, rope_theta, sdpa)
            for _ in range(num_layers)
        )
        self.ln_final = nn.RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # Untied, as in A1.
        # Preserve A1 initialization as well as parameter names / shapes.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = math.sqrt(2 / sum(module.weight.shape)) if isinstance(module, nn.Linear) else 1.0
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.size(1) <= self.context_length:
            raise ValueError("Expected [batch, tokens] with 1 <= tokens <= context_length")
        with region("model/embedding", tokens):
            x = self.token_embeddings(tokens)  # [B,T] -> [B,T,D]
        for index, layer in enumerate(self.layers):
            with region(f"model/layer_{index}", x):
                x = layer(x)
        with region("model/final_norm", x):
            x = self.ln_final(x)
        with region("model/lm_head", x):
            logits = self.lm_head(x)  # [B,T,V], raw logits, not probabilities.
        return logits


def train_step(model, inputs, targets, mode="train", optimizer=None):
    """PDB entry point: inspect logits, loss, and parameter.grad step by step."""
    with region("step/zero_grad", inputs):
        model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(mode != "forward"):
        with region("step/forward", inputs):
            logits = model(inputs)
        if mode != "forward":
            with region("step/loss", inputs):
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            with region("step/backward", inputs):
                loss.backward()
    if optimizer is not None:
        with region("step/optimizer", inputs):
            optimizer.step()


def check():
    """Small standalone check: causality, SDPA equivalence, gradients, update."""
    torch.manual_seed(0)
    model = Transformer(17, 8, 16, 2, 2, 32)
    fast = Transformer(17, 8, 16, 2, 2, 32, sdpa=True)
    fast.load_state_dict(model.state_dict())
    tokens = torch.randint(17, (2, 9))
    logits = model(tokens[:, :-1])
    assert logits.shape == (2, 8, 17)
    torch.testing.assert_close(logits[:, :4], model(tokens[:, :4]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits, fast(tokens[:, :-1]), atol=1e-5, rtol=1e-5)
    for net in (model, fast):
        F.cross_entropy(net(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten()).backward()
    for p, q in zip(model.parameters(), fast.parameters(), strict=True):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
    before = model.lm_head.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, model.lm_head.weight)
    print("check: causality, SDPA outputs/gradients, optimizer update OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--mode", choices=("forward", "backward", "train"), default="train")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--compile", action="store_true")
    dimensions = ("vocab_size", "context_length", "d_model", "num_layers",
                  "num_heads", "d_ff", "batch_size")
    for name in dimensions:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int,
                            help="Override the preset value")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    for name, value in zip(dimensions, PRESETS[args.preset], strict=True):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.check:
        check()
        return
    if args.batch_size <= 0 or args.steps <= 0 or args.warmup < 0:
        parser.error("batch-size and steps must be positive; warmup must be nonnegative")
    torch.manual_seed(0)
    model = Transformer(args.vocab_size, args.context_length, args.d_model,
                        args.num_layers, args.num_heads, args.d_ff, sdpa=args.sdpa).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if args.mode == "train" else None
    run = torch.compile(model) if args.compile else model
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length + 1), device=args.device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]  # Position t predicts token t+1.

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    with region("learn/warmup", inputs):
        for index in range(args.warmup):
            with region(f"warmup/{index}", inputs):
                train_step(run, inputs, targets, args.mode, optimizer)
    synchronize()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples = []
    with region("learn/measure", inputs):
        for index in range(args.steps):
            start = time.perf_counter()
            with region(f"measure/{index}", inputs):
                train_step(run, inputs, targets, args.mode, optimizer)
                synchronize()  # Include completed GPU work, not just CPU kernel launches.
            samples.append((time.perf_counter() - start) * 1000)
    print(f"torch={torch.__version__} seed=0 dtype=float32 {vars(args)}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"step_ms={statistics.mean(samples):.3f} +/- {statistics.pstdev(samples):.3f}")
    print(f"samples_ms={[round(t, 3) for t in samples]}")
    if args.device == "cuda":
        print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.1f}")


if __name__ == "__main__":
    main()
