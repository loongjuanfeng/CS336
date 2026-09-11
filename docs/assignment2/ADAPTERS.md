# 官方测试接线

`tests/assignment2/adapters.py` 已接通 FlashAttentionPytorch、FlashAttentionTriton、OverlapDDP、ShardedOptimizer 和 FSDP。
接口签名与官方一致，验证记录见 [CODE_DELIVERY.md](CODE_DELIVERY.md)。下文保留原始逐步实现指南供学习使用。

| Adapter | 将来接到 | 操作 TODO |
|---|---|---|
| `get_flashattention_autograd_function_pytorch` | `FlashAttentionPytorch` | 从 `cs336_systems.flash_attention` 导入并返回类；不能返回 `.apply`、实例或计算结果 |
| `get_flashattention_autograd_function_triton` | `FlashAttentionTriton` | 返回类；本地名字以实际 adapters.py 为准，讲义有简写拼写 |
| `get_ddp` | `NaiveDDP`，最终 `OverlapDDP` | 构造包装给定 module 的实例；内部必须能通过 `.module` 访问原模型 |
| `ddp_on_after_backward` | DDP 的 `finish_gradient_synchronization()` | 在 optimizer.step 前等待通信并完成平均；不是再跑一次 backward |
| `get_sharded_optimizer` | `ShardedOptimizer(params, optimizer_cls, **kwargs)` | 转发参数组和所有超参数；不要固定成某个 AdamW 类型 |
| `get_fsdp` | `FSDP(module, compute_dtype=compute_dtype)` | 保留默认 None，支持低精度 compute 与 FP32 master |
| `fsdp_on_after_backward` | FSDP 的 `finish_gradient_synchronization()` | 完成 shard 和 replicated 梯度通信后才允许普通 optimizer 更新 |
| `fsdp_gather_full_params` | FSDP 的 `gather_full_params()` | 返回原始参数名字对应的完整 Tensor 字典，便于单进程 oracle 比较 |

## 手把手顺序

1. 一次只接你正在实现的入口；其他函数保留显式未实现，防止把错误归到不相干模块。
2. 先运行最小目标测试，再扩大到该文件。前向先检查输出及保存的L，再做反向。
3. DDP 初期可接NaiveDDP验证初始化/梯度平均；进入overlap题后再切换到OverlapDDP。
4. 分布式测试先CPU/gloo验证，不要为了性能而跳过CPU路径；真实GPU性能另测。
5. 检查 tied weights、冻结参数、参数组与动态组；保持 Parameter 对象身份和注册语义。
6. 保存测试退出码、完整日志与commit，分布式至少重复5次排查race。
7. 不修改官方误差阈值、断言或fixtures；不能把失败改成skip/xfail完成验收。

## 验证入口

```bash
uv run pytest tests/assignment2/test_attention.py -q
uv run pytest tests/assignment2/test_ddp.py -q
uv run pytest tests/assignment2/test_sharded_optimizer.py -q
uv run pytest tests/assignment2/test_fsdp.py -q
```

完整验证使用 `just verify-a2-modal`。`--collect-only` 仅代表发现测试，不代表正确性通过。
