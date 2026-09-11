"""The handout's accumulation and autocast dtype experiments."""

import torch


def accumulation(device):
    results = {}
    cases = (
        (torch.float32, torch.float32, False),
        (torch.float16, torch.float16, False),
        (torch.float16, torch.float32, False),
        (torch.float16, torch.float32, True),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, torch.float32, True),
    )
    for dtype, accumulator_dtype, promote in cases:
        total = torch.zeros((), dtype=accumulator_dtype, device=device)
        value = torch.tensor(0.01, dtype=dtype, device=device)
        for _ in range(1000):
            total += value.float() if promote else value
        key = f"input={dtype},accumulator={accumulator_dtype},explicit_cast={promote}"
        results[key] = total.item()
    return {"expected": 10.0, "sums": results}


def dtype_probe(device):
    probes = {}
    for dtype in (torch.float16, torch.bfloat16):
        fc1 = torch.nn.Linear(8, 10, bias=False, device=device)
        norm = torch.nn.LayerNorm(10, device=device)
        fc2 = torch.nn.Linear(10, 4, bias=False, device=device)
        parameters = [*fc1.parameters(), *norm.parameters(), *fc2.parameters()]
        inputs = torch.randn(2, 8, device=device)
        targets = torch.randint(4, (2,), device=device)
        with torch.autocast(device, dtype=dtype):
            hidden = fc1(inputs)
            normalized = norm(torch.relu(hidden))
            logits = fc2(normalized)
            # Use the standard autocast-aware CE for the toy exercise.
            loss = torch.nn.functional.cross_entropy(logits, targets)
            parameter_dtypes = sorted({str(p.dtype) for p in parameters})
        loss.backward()
        probes[str(dtype)] = {
            "parameters": parameter_dtypes,
            "fc1": str(hidden.dtype),
            "layer_norm": str(normalized.dtype),
            "logits": str(logits.dtype),
            "loss": str(loss.dtype),
            "gradients": sorted(
                {
                    str(p.grad.dtype) if p.grad is not None else "None"
                    for p in parameters
                }
            ),
        }
    return {"autocast_dtypes": probes}


def benchmark(args, output):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    return (
        dtype_probe(args.device)
        if args.experiment == "precision"
        else accumulation(args.device)
    )
