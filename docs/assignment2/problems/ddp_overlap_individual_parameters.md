# ddp_overlap_individual_parameters

**DDP with Overlapping Individual Parameters · 5 分 · 讲义第 35 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第35页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/ddp.py](../../../src/cs336_systems/ddp.py)。

## 手把手 TODO

### 主交付

交付要求：按参数梯度就绪异步通信的自写DDP容器。

- [ ] 1. 先阅读ddp.OverlapDDP的hook生命周期TODO；保持初始广播与参数注册正确。
- [ ] 2. 注册post-accumulate hook，保留Work对象，处理tied weights去重和无梯度参数。
- [ ] 3. after_backward等待并完成平均；每步清理状态，避免重复通信或用错grad引用。
- [ ] 4. 将get_ddp最终接到OverlapDDP；官方测试重复5次，再用nsys检查重叠。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

实现后运行（当前stubs预期未通过）：

```bash
uv run pytest tests/assignment2/test_ddp.py -q
```

- [ ] 保存完整输出、退出码、commit；分布式题重复5次。测试通过不等于性能报告已完成。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
