# tp_calcs

**Tensor parallel calculations · 4 分 · 讲义第 44 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第44页该题开始读，必要时回看前文公式。

## 手把手 TODO

### (a)

交付要求：TP反向的完整方程序列，得到各shard dW和dX。

- [ ] 1. 按8.4给出的列/行切分方式标记W1/W2/W3与保存激活的shard维度。
- [ ] 2. 从dY逆序应用链式法则，逐行写shape与是否replicated。
- [ ] 3. 在局部partial dX需要合并的位置明确写collective；不要把共享激活重复平均。

### (b)

交付要求：TP forward/backward FLOPs及两句说明。

- [ ] 1. 逐个用(a)和前向图的matmul shape计数；忽略非matmul但注明。
- [ ] 2. 检查按N_TP缩放的维度，不要把batch也当成DP切分。

### (c)

交付要求：TP forward/backward通信时间及两句说明。

- [ ] 1. 只统计实际需要的activation/gradient collective，列tensor shape与2bytes/元素。
- [ ] 2. 代入egress W通信模型，区分激活通信与权重梯度all-reduce。

### (d)

交付要求：两阶段的N_TP通信阈值及两句说明。

- [ ] 1. 分别用(b)(c)计算比较并整理不等式。
- [ ] 2. 检查batch与参数维度的趋势，用物理解释交叉验证符号结果。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
