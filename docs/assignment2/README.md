# A2 题目工作台

本目录对应 **Spring 2026 v26.1.3** 讲义，共27个计分题。
本次交付仅包含常规代码与正确性验证，详见 [CODE_DELIVERY.md](CODE_DELIVERY.md)。
下面的 TODO 指原始完整作业（含实验和报告）状态，不等同于代码状态；报告、图表和 leaderboard 不在本次范围。

## 从这里开始

1. 打开下表的一题，按每个小问的复选框执行。代码题同时读相应Python stub的docstring。
2. 写代码时先跑该题的小尺寸正确性测试，再跑完整配置；不要改官方断言或把失败改成skip。
3. 原始实验输出留在 `results/baseline/` 等运行目录；把经核对的数据填入本目录的表头模板。
4. 截图说明文件以 `.png.todo.md` 结尾；按说明生成真正的PNG后再插入报告。
5. 答案统一填写 [writeup.md](writeup.md)，操作说明留在 problems/，避免两处答案失去同步。
6. 用 [ADAPTERS.md](ADAPTERS.md) 接测试；最后按 [SUBMISSION.md](SUBMISSION.md) 准备PDF/ZIP。

机器可读清单：[manifest.json](manifest.json)。基线用法：[baseline README](../../src/cs336_systems/baseline/README.md)。

| 状态 | 题目 | 分值 | 小问 |
|---|---|---:|---|
| TODO | [benchmarking_script](problems/benchmarking_script.md) | 4 | a, b, c |
| TODO | [nsys_profile](problems/nsys_profile.md) | 5 | a, b, c, d, e |
| TODO | [mixed_precision_accumulation](problems/mixed_precision_accumulation.md) | 1 | answer |
| TODO | [benchmarking_mixed_precision](problems/benchmarking_mixed_precision.md) | 2 | a, b, c |
| TODO | [memory_profiling](problems/memory_profiling.md) | 4 | a, b, c, d, e, f |
| TODO | [gradient_checkpointing](problems/gradient_checkpointing.md) | 4 | a, b |
| TODO | [pytorch_attention](problems/pytorch_attention.md) | 2 | a |
| TODO | [torch_compile](problems/torch_compile.md) | 2 | a, b |
| TODO | [flash_forward](problems/flash_forward.md) | 15 | a, b, c |
| TODO | [flash_backward](problems/flash_backward.md) | 5 | answer |
| TODO | [flash_benchmarking](problems/flash_benchmarking.md) | 5 | a |
| TODO | [distributed_communication_single_node](problems/distributed_communication_single_node.md) | 5 | answer |
| TODO | [naive_ddp](problems/naive_ddp.md) | 5 | answer |
| TODO | [naive_ddp_benchmarking](problems/naive_ddp_benchmarking.md) | 3 | answer |
| TODO | [minimal_ddp_flat_benchmarking](problems/minimal_ddp_flat_benchmarking.md) | 2 | answer |
| TODO | [ddp_overlap_individual_parameters](problems/ddp_overlap_individual_parameters.md) | 5 | answer |
| TODO | [ddp_overlap_individual_parameters_benchmarking](problems/ddp_overlap_individual_parameters_benchmarking.md) | 1 | a, b |
| TODO | [optimizer_state_sharding](problems/optimizer_state_sharding.md) | 15 | answer |
| TODO | [optimizer_state_sharding_accounting](problems/optimizer_state_sharding_accounting.md) | 5 | a, b, c |
| TODO | [fsdp](problems/fsdp.md) | 15 | answer |
| TODO | [fsdp_accounting](problems/fsdp_accounting.md) | 5 | a, b |
| TODO | [alternate_ring_all_reduce](problems/alternate_ring_all_reduce.md) | 1 | answer |
| TODO | [data_parallel_calcs](problems/data_parallel_calcs.md) | 3 | a, b, c |
| TODO | [fsdp_calcs](problems/fsdp_calcs.md) | 3 | a, b, c |
| TODO | [tp_calcs](problems/tp_calcs.md) | 4 | a, b, c, d |
| TODO | [fsdp_tp_calcs](problems/fsdp_tp_calcs.md) | 6 | a, b, c, d |
| TODO | [leaderboard](problems/leaderboard.md) | 10 | answer |

## 推荐顺序

计时/精度/显存 → attention参照与compile → PyTorch分块forward → Triton forward/causal → 编译backward → Flash性能对比 → naive/flat/overlap DDP → optimizer分片 → FSDP → 理论题与leaderboard。

理论题可以与编码并行学习；optional全Triton backward单独见 [OPTIONAL.md](OPTIONAL.md)。
