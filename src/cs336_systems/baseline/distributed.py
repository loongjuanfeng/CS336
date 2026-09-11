"""Blocking all-reduce and replicated training references (no sharding/overlap)."""

import tempfile
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.optimization import AdamW
from cs336_basics.training import cross_entropy

from .common import autocast, measure, memory, region, setup, summary, timed
from .transformer import make_model


def reduce_gradients(parameters, flat=False):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if flat:
        buffer = torch.cat([g.flatten() for g in gradients])
        dist.all_reduce(buffer)
        buffer.div_(dist.get_world_size())
        offset = 0
        for gradient in gradients:
            size = gradient.numel()
            gradient.copy_(buffer[offset : offset + size].view_as(gradient))
            offset += size
    else:
        for gradient in gradients:
            dist.all_reduce(gradient)
            gradient.div_(dist.get_world_size())


def worker(rank, config, rendezvous, folder):
    args = Namespace(**config)
    setup(args)
    device = f"cuda:{rank}" if args.device == "cuda" else "cpu"
    if args.device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        args.backend, init_method=rendezvous, rank=rank, world_size=args.world_size
    )
    try:
        phases = {}
        comm_times = []
        if args.experiment == "communication":
            # Zeros avoid repeated in-place sums growing to infinity.
            data = torch.zeros(args.message_mb * 1_000_000 // 4, device=device)

            def step():
                with region("all_reduce", device):
                    dist.all_reduce(data)
        else:
            if args.batch_size % args.world_size:
                raise ValueError(
                    "DDP global batch size must be divisible by world_size"
                )
            model = make_model(args, device)
            phases["initialization"] = memory(device)
            for parameter in model.parameters():
                dist.broadcast(parameter.data, src=0)
            optimizer = AdamW(model.parameters())
            torch.manual_seed(args.seed + rank + 1)
            data = torch.randint(
                args.vocab_size,
                (args.batch_size // args.world_size, args.context_length + 1),
                device=device,
            )

            def step():
                model.zero_grad(set_to_none=True)
                with autocast(args):
                    with region("forward", device):
                        logits = model(data[:, :-1])
                    with region("loss", device):
                        loss = cross_entropy(logits, data[:, 1:])
                with region("backward", device):
                    loss.backward()
                with region("communication", device):
                    comm_times.append(
                        timed(
                            lambda: reduce_gradients(
                                model.parameters(), args.gradient_reduce == "flat"
                            ),
                            device,
                        )
                    )
                phases["before_optimizer"] = memory(device)
                with region("optimizer", device):
                    optimizer.step()
                phases["after_optimizer"] = memory(device)

        dist.barrier()
        rank_output = Path(folder) / f"rank-{rank}"
        rank_output.mkdir(exist_ok=True)
        report = measure(step, args, rank_output, device)
        report["memory_phases"] = phases
        if comm_times:
            report["communication"] = summary(comm_times[-args.steps :])
        ranks = [None] * args.world_size
        dist.all_gather_object(ranks, report)
        if rank == 0:
            import json

            slowest = [max(r["step_ms"][i] for r in ranks) for i in range(args.steps)]
            result = summary(slowest) | {
                "ranks": ranks,
                "batch_semantics": "global",
                "world_size": args.world_size,
            }
            if comm_times:
                communication = [
                    max(r["communication"]["step_ms"][i] for r in ranks)
                    for i in range(args.steps)
                ]
                result["communication"] = summary(communication)
                result["communication_fraction"] = sum(communication) / sum(slowest)
            (Path(folder) / "distributed.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def benchmark(args, output):
    import json

    if args.device == "cuda" and torch.cuda.device_count() < args.world_size:
        raise ValueError(
            f"Need {args.world_size} GPUs; found {torch.cuda.device_count()}"
        )
    with tempfile.TemporaryDirectory() as folder:
        rendezvous = Path(folder) / "rendezvous"
        mp.spawn(
            worker,
            args=(vars(args), rendezvous.as_uri(), str(output)),
            nprocs=args.world_size,
            join=True,
        )
    path = output / "distributed.json"
    result = json.loads(path.read_text())
    path.unlink()
    return result
