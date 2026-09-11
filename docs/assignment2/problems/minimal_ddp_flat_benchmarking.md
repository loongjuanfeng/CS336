# minimal_ddp_flat_benchmarking

**Minimal DDP with Flat Gradients Benchmarking · 2 分 · 讲义第 34 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第34页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/ddp.py](../../../src/cs336_systems/ddp.py)。
- 代码：[src/cs336_systems/experiments.py](../../../src/cs336_systems/experiments.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite ddp --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### 主交付

交付要求：单次flattened all-reduce与逐参数版本的时间对比，1–2句。

- [ ] 1. 实现 FlatDDP.finish_gradient_synchronization，稳定offset恢复每个grad。
- [ ] 2. 复用与naive完全相同的xl/2GPU配置；先检查权重更新一致，再跑性能。
- [ ] 3. 端到端计时包含flatten/unflatten的数据移动；填总时间、通信时间与speedup。

## 产物占位

- [ ] 填写 [tables/minimal_ddp_flat_benchmarking.csv](../tables/minimal_ddp_flat_benchmarking.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
