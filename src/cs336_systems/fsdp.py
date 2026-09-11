"""Layer-wise FSDP with stable master shards and ephemeral gathered weights."""

from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cs336_basics.model.primitives import Embedding, Linear


class _Weight:
    def __init__(self, parameter, dtype):
        self.parameter = parameter
        self.shape = parameter.shape
        self.numel = parameter.numel()
        self.world = dist.get_world_size()
        self.size = (self.numel + self.world - 1) // self.world
        self.dtype = dtype or parameter.dtype
        self.pending = None
        self.full = None
        self.send = None
        flat = F.pad(
            parameter.detach().flatten().float(),
            (0, self.size * self.world - self.numel),
        )
        parameter.data = flat.chunk(self.world)[dist.get_rank()].clone()

    def prefetch(self, dtype=None):
        if self.full is not None:
            return
        self.send = self.parameter.detach().to(dtype or self.dtype).contiguous()
        self.full = torch.empty(
            self.size * self.world, device=self.send.device, dtype=self.send.dtype
        )
        self.pending = dist.all_gather_into_tensor(self.full, self.send, async_op=True)

    def gather(self):
        self.prefetch()
        assert self.pending is not None and self.full is not None
        self.pending.wait()
        result = self.full[: self.numel].view(self.shape)
        self.full = self.pending = self.send = None
        return result

    def release(self):
        if self.pending is not None:
            self.pending.wait()
        self.full = self.pending = self.send = None

    def reduce(self, gradient):
        flat = F.pad(
            gradient.contiguous().flatten(), (0, self.size * self.world - self.numel)
        )
        if dist.get_backend() == "nccl":
            shard = torch.empty(self.size, device=flat.device, dtype=flat.dtype)
            dist.reduce_scatter_tensor(shard, flat)
        else:
            dist.all_reduce(flat)
            shard = flat.chunk(self.world)[dist.get_rank()].clone()
        return shard.float().div_(self.world)


def _compute(x, weight, embedding):
    return F.embedding(x, weight) if embedding else x.to(weight.dtype) @ weight.T


class _ShardedLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, info, owner, index, embedding):
        weight = info.gather()
        owner._prefetch(index + 1)
        ctx.save_for_backward(x, shard)
        ctx.info, ctx.owner, ctx.index, ctx.embedding = info, owner, index, embedding
        return _compute(x, weight, embedding)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        x, _shard = ctx.saved_tensors
        weight = ctx.info.gather()
        ctx.owner._prefetch(ctx.index - 1)
        # Rebuild only this layer's graph; no full weight survives between layers.
        with torch.enable_grad():
            x = x.detach().requires_grad_(ctx.needs_input_grad[0])
            weight = weight.detach().requires_grad_(True)
            output = _compute(x, weight, ctx.embedding)
            inputs = (x, weight) if x.requires_grad else (weight,)
            grads = torch.autograd.grad(output, inputs, grad_output)
        dx = grads[0] if x.requires_grad else None
        dw = ctx.info.reduce(grads[-1]) if ctx.needs_input_grad[1] else None
        return dx, dw, None, None, None, None


def _layer_forward(layer, x):
    return _ShardedLayer.apply(
        x,
        layer.weight,
        layer._fsdp_weight,
        layer._fsdp_owner,
        layer._fsdp_index,
        isinstance(layer, Embedding),
    )


class FSDP(torch.nn.Module):
    """Shard A1 Linear/Embedding weights; keep other parameters replicated.

    Ranks must execute the same layer sequence. Prefetch follows registration
    order, at most one adjacent layer; repeated/shared layers remain correct.
    Construct the optimizer after wrapping. State dicts contain local shards;
    gather_full_params provides a full-precision export on every rank.
    """

    def __init__(self, module, compute_dtype=None):
        super().__init__()
        if compute_dtype not in (None, torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Unsupported compute dtype")
        self.module = module
        self._weights = {}
        self._layers = []
        self.world_size = dist.get_world_size()
        for tensor in (*module.parameters(), *module.buffers()):
            dist.broadcast(tensor.detach(), src=0)
        for layer in module.modules():
            if not isinstance(layer, (Linear, Embedding)):
                continue
            parameter = layer.weight
            if parameter not in self._weights:
                self._weights[parameter] = _Weight(parameter, compute_dtype)
            # Plain attributes avoid registering the owner as a child module.
            object.__setattr__(layer, "_fsdp_owner", self)
            layer._fsdp_weight = self._weights[parameter]
            layer._fsdp_index = len(self._layers)
            self._layers.append(layer)
            layer.forward = MethodType(_layer_forward, layer)

    def _prefetch(self, index):
        # Flush a speculative neighbor before starting another: bounded storage
        # also holds for tied weights and non-registration execution order.
        target = (
            self._layers[index]._fsdp_weight if 0 <= index < len(self._layers) else None
        )
        for info in self._weights.values():
            if info is not target:
                info.release()
        if target is not None:
            target.prefetch()

    def forward(self, *inputs, **kwargs):
        try:
            return self.module(*inputs, **kwargs)
        finally:
            for info in self._weights.values():
                info.release()

    def finish_gradient_synchronization(self):
        for info in self._weights.values():
            info.release()
        for p in self.module.parameters():
            if p not in self._weights and p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(self.world_size)

    @torch.no_grad()
    def gather_full_params(self):
        result = {}
        for name, p in self.module.named_parameters():
            if p in self._weights:
                info = self._weights[p]
                info.release()
                info.prefetch(torch.float32)
                result[name] = info.gather().clone()
            else:
                result[name] = p.detach().clone()
        return result
