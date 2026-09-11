# data_parallel_calcs

**Data parallel calculations · 3 分 · 讲义第 42 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第42页该题开始读，必要时回看前文公式。

## 手把手 TODO

### (a)

交付要求：DP backward FLOPs及一句说明。

- [ ] 1. 读8.2的具体SwiGLU计算图，列出每个matmul的输入/输出shape。
- [ ] 2. 将全局batch换成每rank batch，按2mnk只数matmul，分别统计权重梯度和输入梯度。

### (b)

交付要求：DP backward通信时间及一句说明。

- [ ] 1. 列出需要同步的权重梯度shape，按FP16每元素2bytes换算S。
- [ ] 2. 代入讲义的collective通信模型，保留N_DP与每设备egress W，写清是否计往返。

### (c)

交付要求：N_DP的通信瓶颈阈值不等式及一句说明。

- [ ] 1. 用(a)的FLOPs/C和(b)的时间建立计算与通信比较。
- [ ] 2. 求解N_DP边界并检查N_DP=1与带宽趋大等极限；不要遗漏C或单位。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
