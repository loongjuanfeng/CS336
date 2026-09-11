# fsdp

**Fully-Sharded Data Parallel · 15 分 · 讲义第 38 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第38页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/fsdp.py](../../../src/cs336_systems/fsdp.py)。

## 手把手 TODO

### 主交付

交付要求：自写分片权重容器，可用于普通optimizer，支持可选低精度compute。

- [ ] 1. 从 fsdp.FSDP 的元数据/生命周期TODO开始，先同步实现再加入预取。
- [ ] 2. 识别A1 Linear/Embedding；FP32主权重不丢失，gather只短暂存在，norm等replicated梯度同步。
- [ ] 3. compute_dtype必须在通信前转换；backward后shard grad形状/dtype匹配主Parameter。
- [ ] 4. 接三个FSDP adapters，先CPU/gloo None与FP16正确性，测试重复5次。
- [ ] 5. 再增加满足讲义预取窗口的异步gather；不能直接使用torch FSDP作为自己的实现。

## 产物占位

- [ ] 在writeup填写本题要求的公式/文字或实现文件与验证记录。

实现后运行（当前stubs预期未通过）：

```bash
uv run pytest tests/assignment2/test_fsdp.py -q
```

- [ ] 保存完整输出、退出码、commit；分布式题重复5次。测试通过不等于性能报告已完成。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
