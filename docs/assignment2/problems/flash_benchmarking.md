# flash_benchmarking

**FlashAttention-2 Benchmarking · 5 分 · 讲义第 28 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第28页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/experiments.py](../../../src/cs336_systems/experiments.py)。
- 代码：[src/cs336_systems/flash_attention.py](../../../src/cs336_systems/flash_attention.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite flash --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

**边界：此suite只有已有参考实现/配置，不能替代本题要求的自写优化实现或正式leaderboard计时。**

## 手把手 TODO

### (a)

交付要求：B200上自写FlashAttention与普通PyTorch的forward/backward/联合时间表。

- [ ] 1. 先完成 experiments.benchmark_flash_attention；原有flash suite只测参考侧。
- [ ] 2. 核对batch1、causal、S=128…65536的2次幂、D=16…128、FP32/BF16。
- [ ] 3. 同一输入先比数值，再do_bench；单独backward排除建图，清理梯度。
- [ ] 4. 记录tile/版本/三类时间/OOM；在同精度同phase内计算speedup，保留真实输出。

## 产物占位

- [ ] 填写 [tables/flash_benchmarking.csv](../tables/flash_benchmarking.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
