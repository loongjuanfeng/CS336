"""ZeRO-1 style optimizer state ownership with replicated parameters/gradients."""

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.owners = {}
        self.local_optimizer = None
        super().__init__(params, kwargs)
        # Empty groups keep indices aligned even on ranks with no owners.
        self.local_optimizer = optimizer_cls(
            [self._local_group(g) for g in self.param_groups], **kwargs
        )
        for group, local in zip(
            self.param_groups, self.local_optimizer.param_groups, strict=True
        ):
            group.update({k: v for k, v in local.items() if k != "params"})
        self.defaults = self.local_optimizer.defaults.copy()
        self.state = self.local_optimizer.state

    def _local_group(self, group):
        return dict(
            group, params=[p for p in group["params"] if self.owners[p] == self.rank]
        )

    def add_param_group(self, param_group):
        group = dict(param_group)
        params = group["params"]
        params = [params] if isinstance(params, torch.Tensor) else list(params)
        group["params"] = list(dict.fromkeys(params))
        super().add_param_group(group)
        for p in group["params"]:
            self.owners[p] = len(self.owners) % self.world_size
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(
                self._local_group(self.param_groups[-1])
            )

    def step(self, closure=None, **kwargs):
        assert self.local_optimizer is not None
        for group, local in zip(
            self.param_groups, self.local_optimizer.param_groups, strict=True
        ):
            local.update({k: v for k, v in group.items() if k != "params"})
        result = self.local_optimizer.step(closure=closure, **kwargs)
        for p, owner in self.owners.items():
            dist.broadcast(p.detach(), src=owner)
        return result

    def load_state_dict(self, state_dict):
        assert self.local_optimizer is not None
        # Checkpoints are per rank; standard state dict indices refer to full groups.
        super().load_state_dict(state_dict)
        self.local_optimizer.state = self.state
        self.local_optimizer.param_groups = [
            self._local_group(g) for g in self.param_groups
        ]
