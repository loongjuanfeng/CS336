"""Assignment DDP implementations; one process group, one backward per step."""

import torch
import torch.distributed as dist


class NaiveDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for tensor in (*module.parameters(), *module.buffers()):
            dist.broadcast(tensor.detach(), src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(self.world_size)


class FlatDDP(NaiveDDP):
    def finish_gradient_synchronization(self):
        groups = {}
        for p in self.module.parameters():
            if p.grad is not None:
                groups.setdefault((p.device, p.dtype), []).append(p.grad)
        for gradients in groups.values():
            flat = torch.cat([g.reshape(-1) for g in gradients])
            dist.all_reduce(flat)
            flat.div_(self.world_size)
            offset = 0
            for grad in gradients:
                grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
                offset += grad.numel()


class OverlapDDP(NaiveDDP):
    """Launch async reductions in stable reverse parameter order.

    All ranks must execute the same graph. Unused parameters are supported when
    unused on every rank; rank-dependent control flow is outside this wrapper.
    """

    def __init__(self, module):
        super().__init__(module)
        self._ordered = [
            p for p in reversed(list(module.parameters())) if p.requires_grad
        ]
        self._ready = set()
        self._pending = []
        self._next = 0
        self._hooks = [
            p.register_post_accumulate_grad_hook(self._on_grad) for p in self._ordered
        ]

    def _on_grad(self, parameter):
        self._ready.add(parameter)
        while self._next < len(self._ordered):
            p = self._ordered[self._next]
            if p not in self._ready:
                break
            self._launch(p)
            self._next += 1

    def _launch(self, p):
        if p.grad is not None:
            self._pending.append((dist.all_reduce(p.grad, async_op=True), p.grad))

    def finish_gradient_synchronization(self):
        for p in self._ordered[self._next :]:
            if p in self._ready:
                self._launch(p)
        for work, gradient in self._pending:
            work.wait()
            gradient.div_(self.world_size)
        self._pending.clear()
        self._ready.clear()
        self._next = 0
