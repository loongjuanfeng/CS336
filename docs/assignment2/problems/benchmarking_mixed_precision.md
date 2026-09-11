# benchmarking_mixed_precision

**Benchmarking Mixed Precision · 2 分 · 讲义第 8 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第8页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite mixed-precision --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：列出参数、fc1、LayerNorm、logits、loss和梯度的dtype。

- [ ] 1. 读 baseline/precision.py 的与讲义一致的toy网络；在GPU跑 --experiment precision。
- [ ] 2. 记录FP16 autocast上下文内每个指定张量dtype；不要用CPU观测代替GPU答案。
- [ ] 3. 检查参数仍FP32且backward位于autocast之外；在表格列明实际PyTorch版本。

### (b)

交付要求：LayerNorm敏感部分，以及BF16是否需要类似处理，2–3句。

- [ ] 1. 把norm中的统计、归约、除法与输出转换拆开讨论。
- [ ] 2. 比较FP16/BF16范围与有效精度；区分输出dtype和内部累加dtype。
- [ ] 3. 用(a)观测支撑论述；不要仅凭输出dtype推断所有内部运算精度。

### (c)

交付要求：各模型FP32与BF16 mixed precision计时，2–3句趋势。

- [ ] 1. 预览 --suite mixed-precision；toy case之后才是整模型案例。
- [ ] 2. 使用相同设备、模式、batch/context与软件版本，保持FP32 master weights；不要直接 model.bfloat16() 偷换实验。
- [ ] 3. 填写forward/backward/train各路径均值标准差及OOM；计算成对speedup，说明规模变化趋势。

## 产物占位

- [ ] 填写 [tables/benchmarking_mixed_precision.csv](../tables/benchmarking_mixed_precision.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
