# flash_backward

**FlashAttention-2 Backward Pass · 5 分 · 讲义第 28 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第28页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/flash_attention.py](../../../src/cs336_systems/flash_attention.py)。

## 手把手 TODO

### 主交付

交付要求：使用PyTorch+torch.compile重计算反向，正确给出dQ/dK/dV。

- [ ] 1. 在 flash_backward_reference 中按Eq.13–19列出形状与D向量；从保存的L重建概率。
- [ ] 2. 先用普通attention autograd作oracle逐个比较梯度，再让两个自定义Function调用该实现。
- [ ] 3. 处理causal路径并返回布尔输入对应的None；跑所有test_flash_backward。
- [ ] 4. 必做版本通过后再考虑全Triton反向；可选优化不得阻塞必做题验收。

## 产物占位

- [ ] 填写 [tables/flash_backward.csv](../tables/flash_backward.csv)（当前只有表头，无任何结果）。

实现后运行（当前stubs预期未通过）：

```bash
uv run pytest tests/assignment2/test_attention.py -k flash_backward -q
```

- [ ] 保存完整输出、退出码、commit；分布式题重复5次。测试通过不等于性能报告已完成。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
