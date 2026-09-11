# flash_forward

**FlashAttention-2 Forward Pass · 15 分 · 讲义第 26 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第26页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/flash_attention.py](../../../src/cs336_systems/flash_attention.py)。

## 手把手 TODO

### (a)

交付要求：纯PyTorch分块FlashAttention autograd.Function，接adapter并通过前向测试。

- [ ] 1. 按 src/cs336_systems/flash_attention.py 的逐步TODO先实现非causal PyTorch类。
- [ ] 2. 小tile手动检查在线softmax状态，再比较O和L；forward只返回O，L通过ctx保存。
- [ ] 3. 把adapter返回值改为类对象；只跑前向测试，反向仍可显式未实现。

### (b)

交付要求：自写Triton前向及autograd包装，通过前向测试。

- [ ] 1. 逐步把(a)的tile运算搬到kernel；明确grid、strides和指针的推进。
- [ ] 2. 用FP32累加器，检查P与V的乘法dtype；先固定tile验证再调参。
- [ ] 3. 接本地实际名 get_flashattention_autograd_function_triton；讲义简写拼写不等于本地hook名。

### (c)

交付要求：默认False的causal开关，保存到ctx，覆盖True/False。

- [ ] 1. 用全局query/key索引构造mask；跨tile边界作手算检查。
- [ ] 2. 将flag作为kernel constexpr并保存给backward；测试默认省略参数仍能运行。
- [ ] 3. 分别跑两种causal测试；不得只修改测试容差让错误通过。

## 产物占位

- [ ] 填写 [tables/flash_forward.csv](../tables/flash_forward.csv)（当前只有表头，无任何结果）。

实现后运行（当前stubs预期未通过）：

```bash
uv run pytest tests/assignment2/test_attention.py -k flash_forward -q
```

- [ ] 保存完整输出、退出码、commit；分布式题重复5次。测试通过不等于性能报告已完成。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
