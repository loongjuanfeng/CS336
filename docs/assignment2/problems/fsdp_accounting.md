# fsdp_accounting

**FSDP Accounting · 5 分 · 讲义第 39 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第39页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/experiments.py](../../../src/cs336_systems/experiments.py)。
- 代码：[src/cs336_systems/fsdp.py](../../../src/cs336_systems/fsdp.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite fsdp --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

**边界：此suite只有已有参考实现/配置，不能替代本题要求的自写优化实现或正式leaderboard计时。**

## 手把手 TODO

### (a)

交付要求：预期峰值显存节省，2–3句。

- [ ] 1. 从上一题的未分片参数/梯度/状态字节出发，区分每rank存储与瞬时全权重。
- [ ] 2. 按题意单独说明是否忽略预分配gather buffers；列出估算，不把整卡显存除以world_size。

### (b)

交付要求：xl/2GPU各层all-gather是否及时，2–3句和nsys截图。

- [ ] 1. 完成experiments.profile_fsdp，用实现自己的gather/wait/释放标注。
- [ ] 2. 记录实际前向依赖是否等待，标出预取窗口与峰值full-weight存活数量。
- [ ] 3. 保存有层名与通信stream的截图，用时间线支撑结论。

## 产物占位

- [ ] 填写 [tables/fsdp_accounting.csv](../tables/fsdp_accounting.csv)（当前只有表头，无任何结果）。
- [ ] 按 [figures/fsdp_all_gather.png.todo.md](../figures/fsdp_all_gather.png.todo.md) 生成真实截图。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
