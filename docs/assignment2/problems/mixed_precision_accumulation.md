# mixed_precision_accumulation

**Mixed-Precision Accumulation · 1 分 · 讲义第 7 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第7页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite accumulation --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### 主交付

交付要求：运行讲义四段累加代码，提交2–3句观察。

- [ ] 1. 运行 --experiment accumulation --device cpu；识别FP32累加、FP16累加、FP16值加到FP32、显式先转FP32四种情况。
- [ ] 2. 填入实际和、相对数学目标的误差；BF16扩展结果另标“补充”。
- [ ] 3. 分别考虑0.01的表示误差与每次加法的舍入误差；写解释而不是只粘数字。

## 产物占位

- [ ] 填写 [tables/mixed_precision_accumulation.csv](../tables/mixed_precision_accumulation.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
