# gradient_checkpointing

**Memory-Optimal Gradient Checkpointing · 4 分 · 讲义第 15 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第15页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/checkpointing.py](../../../src/cs336_systems/checkpointing.py)。
- 代码：[src/cs336_systems/baseline/transformer.py](../../../src/cs336_systems/baseline/transformer.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite checkpointing --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：允许嵌套、忽略计算代价时的策略/渐近内存/计算量，3–5句及代码草图。

- [ ] 1. 先列出checkpoint边界激活、块内residual和递归栈的存活规则。
- [ ] 2. 手画N=2/4/8的执行和重计算过程，设计递归划分与终止条件。
- [ ] 3. 给内存与计算量建立递推式，再求渐近形式；不要直接把非嵌套分组结论搬过来。
- [ ] 4. 在 checkpointing.memory_optimal_forward 填代码草图并验证小模型输出/梯度。

### (b)

交付要求：xl/batch4/context2048，仅一层重计算预算的最佳分组及邻居实测。

- [ ] 1. 预览 --suite checkpointing；0是无checkpoint对照，其余使用已有非递归分组。
- [ ] 2. 根据层数和边界激活/residual成本选择候选组大小，再测相邻更小/更大的组。
- [ ] 3. 比较peak allocated与step时间，确认没有嵌套checkpoint；解释3–5句并填实际数字。

## 产物占位

- [ ] 填写 [tables/gradient_checkpointing.csv](../tables/gradient_checkpointing.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
