# torch_compile

**Torch Compile · 2 分 · 讲义第 16 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第16页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite compile --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：相同attention网格的eager/compiled forward与backward对比表。

- [ ] 1. 使用 --suite compile 中的attention案例，和pytorch_attention保持相同配置。
- [ ] 2. 确保首次编译与重新编译不进正式测量；必要时增加warmup并记录理由。
- [ ] 3. 先检查编译前后输出/梯度，再计算对应phase的speedup；OOM单列。

### (b)

交付要求：完整Transformer各mode的vanilla/compiled对比表。

- [ ] 1. 使用compile suite的整模型案例；记录模型、精度、模式和warmup。
- [ ] 2. 保持A1损失与optimizer不变，核实本基线编译的是模型forward及AOT backward，不是整个optimizer循环。
- [ ] 3. 填写每种mode的速度变化，不用attention单算子的收益替代端到端收益。

## 产物占位

- [ ] 填写 [tables/torch_compile.csv](../tables/torch_compile.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
