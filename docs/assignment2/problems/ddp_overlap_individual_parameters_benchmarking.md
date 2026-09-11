# ddp_overlap_individual_parameters_benchmarking

**DDP Overlapping Individual Parameters Benchmarking · 1 分 · 讲义第 36 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第36页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/experiments.py](../../../src/cs336_systems/experiments.py)。
- 代码：[src/cs336_systems/ddp.py](../../../src/cs336_systems/ddp.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite ddp --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

**边界：此suite只有已有参考实现/配置，不能替代本题要求的自写优化实现或正式leaderboard计时。**

## 手把手 TODO

### (a)

交付要求：xl/2GPU中overlap、naive和flat的step对比，1–2句。

- [ ] 1. 补齐experiments.benchmark_ddp_variants的overlap分支，保持其他设置一致。
- [ ] 2. 验证数值后测慢rank step；不要直接把有重叠的通信耗时从wall-clock里减掉。
- [ ] 3. 填三种实现的表，描述实际结果而不是预设overlap必胜。

### (b)

交付要求：naive与overlap各一张显示backward/通信关系的nsys截图。

- [ ] 1. 选择相同模型配置、同类正式step；展开CPU NVTX、CUDA kernel和通信stream。
- [ ] 2. 截图标注时间标尺、rank、backward和NCCL活动，证明是否真的交叠。
- [ ] 3. 把原始trace路径写入图注；只展示CPU异步API返回不足以证明GPU重叠。

## 产物占位

- [ ] 填写 [tables/ddp_overlap_individual_parameters_benchmarking.csv](../tables/ddp_overlap_individual_parameters_benchmarking.csv)（当前只有表头，无任何结果）。
- [ ] 按 [figures/ddp_naive.png.todo.md](../figures/ddp_naive.png.todo.md) 生成真实截图。
- [ ] 按 [figures/ddp_overlap.png.todo.md](../figures/ddp_overlap.png.todo.md) 生成真实截图。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
