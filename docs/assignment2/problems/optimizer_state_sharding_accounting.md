# optimizer_state_sharding_accounting

**Optimizer State Sharding Accounting · 5 分 · 讲义第 38 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第38页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/experiments.py](../../../src/cs336_systems/experiments.py)。
- 代码：[src/cs336_systems/sharded_optimizer.py](../../../src/cs336_systems/sharded_optimizer.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite sharding --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

**边界：此suite只有已有参考实现/配置，不能替代本题要求的自写优化实现或正式leaderboard计时。**

## 手把手 TODO

### (a)

交付要求：xl/2GPU各阶段峰值和参数/梯度/状态分解，2–3句。

- [ ] 1. 实现experiments.profile_optimizer_sharding：分别测未分片与分片，固定数据和模型。
- [ ] 2. 模型初始化后、optimizer前、optimizer后分别记录；把首次Adam状态分配与稳态分开。
- [ ] 3. 用numel×element_size独立核对各rank状态占用，报告峰值计数器重置语义。

### (b)

交付要求：有无状态分片的iteration时间，2–3句。

- [ ] 1. 同一通信路径只替换optimizer，预热后记录慢rank端到端时间和同步成本。
- [ ] 2. 解释内存收益与通信开销的权衡；不能用每rank局部optimizer耗时充当整个step。

### (c)

交付要求：与ZeRO stage 1的区别，2–3句。

- [ ] 1. 先写出自己的参数/梯度/optimizer state实际如何分布及每步collective。
- [ ] 2. 再读讲义引用的ZeRO-DP Pos部分，逐项比较内存与通信；给出处，不凭名字推断等价。

## 产物占位

- [ ] 填写 [tables/optimizer_state_sharding_accounting.csv](../tables/optimizer_state_sharding_accounting.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
