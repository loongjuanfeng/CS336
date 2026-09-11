# optimizer_state_sharding

**Optimizer State Sharding · 15 分 · 讲义第 37 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第37页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/sharded_optimizer.py](../../../src/cs336_systems/sharded_optimizer.py)。

## 手把手 TODO

### 主交付

交付要求：支持参数组/动态add_param_group的自写分片optimizer。

- [ ] 1. 先读sharded_optimizer.ShardedOptimizer各方法TODO，设计稳定owner映射。
- [ ] 2. 正确调用Optimizer父类初始化，处理构造时add_param_group重入及空本地分片。
- [ ] 3. 本rank只保留owner状态；step后同步更新参数；所有参数的梯度都要被zero_grad清理。
- [ ] 4. 接get_sharded_optimizer；在torch AdamW oracle上跑10步比较，官方测试重复5次。
- [ ] 5. 额外检查动态组、不同lr与共享权重；不要把测试没覆盖当成接口不需要支持。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

实现后运行（当前stubs预期未通过）：

```bash
uv run pytest tests/assignment2/test_sharded_optimizer.py -q
```

- [ ] 保存完整输出、退出码、commit；分布式题重复5次。测试通过不等于性能报告已完成。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
