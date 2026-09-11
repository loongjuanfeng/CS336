# memory_profiling

**Memory Profiling · 4 分 · 讲义第 9 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第9页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite memory --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：xl推理/训练的Active memory timeline各一张和2–3句。

- [ ] 1. 预览 --suite memory；用 --memory-snapshot 保存真实pickle，先验证短context可运行。
- [ ] 2. 在memory_viz打开对应pickle；对forward与train分别定位正式step和峰值。
- [ ] 3. 保存两张截图，标出模型、context、精度、phase和MiB单位；按波峰辨认阶段。

### (b)

交付要求：context=128/2048各自forward/full train峰值表。

- [ ] 1. 固定xl、batch4，测两个长度与两种mode；记录allocated和reserved，不混用。
- [ ] 2. 明确是否包含模型、梯度、Adam状态以及warmup之后的存量；写清重置峰值计数器的位置。
- [ ] 3. 填四个核心数字；OOM如实写出，不用其他小模型数据顶替。

### (c)

交付要求：xl mixed-precision的forward/train峰值与FP32比较，2–3句。

- [ ] 1. 同样context和mode切换 --precision bf16，保持主参数/优化器状态口径一致。
- [ ] 2. 记录哪些部分变小、哪些仍为FP32；用实测差异而不是“显存必减半”假设解释。

### (d)

交付要求：FP32 residual-stream张量大小推导，1–2句。

- [ ] 1. 从默认batch、context和xl d_model列出张量shape。
- [ ] 2. 按元素数×每元素字节数计算并转换MiB；在报告写出代入步骤和单位。

### (e)

交付要求：低Detail视图中的最大分配大小及来源，1–2句。

- [ ] 1. 在同一pickle降低Detail，点选最大的allocation。
- [ ] 2. 根据Python调用栈追到具体算子/张量形状，填写截图文件名及分配大小。

### (f)

交付要求：单个TransformerBlock保存的residual、前五操作占比、产生的梯度估计。

- [ ] 1. 在nsys启用PyTorch NVTX和CUDA memory tracing；选同一步的一层。
- [ ] 2. 记录为backward保留的分配生命周期，排除短暂workspace；按操作汇总前五项。
- [ ] 3. 观察backward释放residual同时创建梯度的变化；另按参数shape计算梯度字节作交叉验证。
- [ ] 4. 保存带层名/时间范围的截图，写1–2段，说明无法区分的allocator效应。

## 产物占位

- [ ] 填写 [tables/memory_profiling.csv](../tables/memory_profiling.csv)（当前只有表头，无任何结果）。
- [ ] 填写 [tables/memory_profiling_residuals.csv](../tables/memory_profiling_residuals.csv)（当前只有表头，无任何结果）。
- [ ] 按 [figures/memory_xl_forward.png.todo.md](../figures/memory_xl_forward.png.todo.md) 生成真实截图。
- [ ] 按 [figures/memory_xl_train.png.todo.md](../figures/memory_xl_train.png.todo.md) 生成真实截图。
- [ ] 按 [figures/memory_largest_allocations.png.todo.md](../figures/memory_largest_allocations.png.todo.md) 生成真实截图。
- [ ] 按 [figures/memory_block_residuals.png.todo.md](../figures/memory_block_residuals.png.todo.md) 生成真实截图。
- [ ] 按 [figures/memory_block_backward.png.todo.md](../figures/memory_block_backward.png.todo.md) 生成真实截图。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
