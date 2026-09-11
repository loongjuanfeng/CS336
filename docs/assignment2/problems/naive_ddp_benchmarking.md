# naive_ddp_benchmarking

**Naïve DDP Benchmarking · 3 分 · 讲义第 33 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第33页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/experiments.py](../../../src/cs336_systems/experiments.py)。
- 代码：[src/cs336_systems/ddp.py](../../../src/cs336_systems/ddp.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite ddp --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### 主交付

交付要求：1节点2GPU、xl的step时间与通信比例，附benchmark设置。

- [ ] 1. 从 --suite ddp 的individual案例获取参照，再使用自己的NaiveDDP复测。
- [ ] 2. 定义global/per-rank batch与通信计时边界；输入、损失、optimizer均相同。
- [ ] 3. 记录慢rank总时间及通信时间，计算比例时注明是否含拼接或同步等待。

## 产物占位

- [ ] 填写 [tables/naive_ddp_benchmarking.csv](../tables/naive_ddp_benchmarking.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
