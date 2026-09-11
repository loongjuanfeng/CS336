# pytorch_attention

**PyTorch Attention Benchmarking · 2 分 · 讲义第 16 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第16页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite attention --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：20组普通attention的forward/backward时间、backward前显存与OOM/内存推导。

- [ ] 1. 用 --suite attention --list-cases 核对batch8、无head维度、D=16/32/64/128、S=256/1024/4096/8192/16384。
- [ ] 2. 先跑一组小输入确认forward/backward分别同步；正式每组预热后测100次。
- [ ] 3. 填表记录实际dtype、memory_before_backward与OOM；保存各次时间，不只截图终端。
- [ ] 4. 选择最小发生OOM的配置按Q/K/V、分数/概率/梯度等张量做字节核算；解释随S的增长与可消除的存储。

## 产物占位

- [ ] 填写 [tables/pytorch_attention.csv](../tables/pytorch_attention.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
