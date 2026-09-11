"""A1 Transformer reference for timing, AMP, compilation and memory experiments."""

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model.transformer import TransformerLanguageModel
from cs336_basics.optimization import AdamW
from cs336_basics.training import cross_entropy

from .common import autocast, measure, memory, region, setup


def make_model(args, device):
    return TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rope_theta=10_000.0,
        device=device,
        dtype=torch.float32,
    )


def forward(model, inputs, group_size=0):
    if not group_size or not torch.is_grad_enabled():
        return model(inputs)
    hidden = model.token_embeddings(inputs)
    for start in range(0, len(model.layers), group_size):
        layers = model.layers[start : start + group_size]

        def group(x, layers=layers):
            for layer in layers:
                x = layer(x)
            return x

        hidden = checkpoint(group, hidden, use_reentrant=False)
    return model.lm_head(model.ln_final(hidden))


def benchmark(args, output):
    setup(args)
    model = make_model(args, args.device)
    initialization = memory(args.device)
    optimizer = AdamW(model.parameters()) if args.mode == "train" else None
    tokens = torch.randint(
        args.vocab_size, (args.batch_size, args.context_length + 1), device=args.device
    )
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def model_forward(inputs):
        return forward(model, inputs, args.checkpoint_layers)

    run_forward = torch.compile(model_forward) if args.compile else model_forward
    phases = {}

    def step():
        if args.mode != "forward":
            with region("zero_grad", args.device):
                model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(args.mode != "forward"):
            with autocast(args):
                with region("forward", args.device):
                    logits = run_forward(inputs)
                if args.mode != "forward":
                    with region("loss", args.device):
                        loss = cross_entropy(logits, targets)
            if args.mode != "forward":
                with region("backward", args.device):
                    loss.backward()
        if optimizer:
            phases["before_optimizer"] = memory(args.device)
            with region("optimizer", args.device):
                optimizer.step()
            phases["after_optimizer"] = memory(args.device)

    report = measure(step, args, output)
    report.update(
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_dtype="float32",
        memory_phases={"initialization": initialization, **phases},
    )
    return report
