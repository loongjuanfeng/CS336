# naive_ddp

**Naïve DDP · 5 分 · 讲义第 33 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第33页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/ddp.py](../../../src/cs336_systems/ddp.py)。

## 手把手 TODO

### 主交付

交付要求：封装逐参数阻塞梯度平均的DDP，接adapter并通过正确性测试。

- [ ] 1. 将 baseline 的参考算法整理到 ddp.NaiveDDP 的接口，保留 .module 和参数注册。
- [ ] 2. 先测初始化广播：不同rank种子后必须与rank0一致，包括共享/冻结参数语义。
- [ ] 3. backward后平均梯度再更新；接get_ddp和after_backward，比较全局batch单进程SGD结果。
- [ ] 4. 不要把baseline已有reduce函数当作容器接口已通过测试。

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
