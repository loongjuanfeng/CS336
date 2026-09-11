# alternate_ring_all_reduce

**Alternate ring all-reduce · 1 分 · 讲义第 41 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第41页该题开始读，必要时回看前文公式。

## 手把手 TODO

### 主交付

交付要求：以S,N,W表示的总时间及一句理由。

- [ ] 1. 手画N=3/4的每一轮：每个设备发送的是整向量还是分块、共几轮。
- [ ] 2. 按每轮egress bytes/W写时间；明确各设备可并行发送以及忽略延迟的假设。
- [ ] 3. 再和前文reduce-scatter+all-gather比较；只填推导，不运行GPU实验来代替符号答案。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
