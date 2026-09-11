# distributed_communication_single_node

**Distributed Communication (Single Node) · 5 分 · 讲义第 32 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第32页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite communication --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### 主交付

交付要求：2/4/6进程、1/10/100/1000MB FP32 all-reduce图表及2–3句。

- [ ] 1. 先用CPU/gloo小消息验证通信，再在真正多GPU单节点运行 --suite communication。
- [ ] 2. 明确MB使用十进制bytes；记录backend、拓扑、world_size和同步方式。
- [ ] 3. 预热后测量，按慢rank报告时间；保存各rank原始记录，解释规模和消息大小趋势。
- [ ] 4. 检查单次benchmark小于5分钟；不能把一张卡上的多进程当成多GPU性能。

## 产物占位

- [ ] 填写 [tables/distributed_communication_single_node.csv](../tables/distributed_communication_single_node.csv)（当前只有表头，无任何结果）。
- [ ] 按 [figures/communication_scaling.png.todo.md](../figures/communication_scaling.png.todo.md) 生成真实截图。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
