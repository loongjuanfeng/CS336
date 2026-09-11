# fsdp_calcs

**Fully sharded data parallel calculations · 3 分 · 讲义第 43 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第43页该题开始读，必要时回看前文公式。

## 手把手 TODO

### (a)

交付要求：FSDP forward/backward各自FLOPs及两句说明。

- [ ] 1. 沿用8.2计算图，在每rank写出实际batch与临时gather后的权重shape。
- [ ] 2. 分别列forward与backward matmul计数，说明为何不能再随意除一次world_size。

### (b)

交付要求：FSDP forward/backward通信时间各一式及两句说明。

- [ ] 1. 列每个阶段的权重all-gather、梯度reduce-scatter与payload字节。
- [ ] 2. 在指定带宽模型下合计，区分forward释放全权重后backward是否必须再次gather。

### (c)

交付要求：两阶段各自的N_FSDP瓶颈阈值及两句说明。

- [ ] 1. 分别比较forward计算/通信时间与backward计算/通信时间。
- [ ] 2. 保留两个不等式，做N=1和W趋大检查；不要用总step公式替代两问。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
