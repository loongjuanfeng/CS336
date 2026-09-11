# nsys_profile

**Nsight Systems Profiling · 5 分 · 讲义第 6 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第6页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite nsys --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：前向实际耗时与Python计时的比较，1–2句。

- [ ] 1. 选讲义表1的两个模型和三个大于128的2次幂context；探索可容纳的最大长度。现有suite仅给256/512/1024起点。
- [ ] 2. 同一配置先无profiler计时，再 --nsys；选择正式step，排除warmup与初始化。
- [ ] 3. 分别辨认CPU NVTX范围和关联GPU执行区间，记录两种定义及额外采样开销。

### (b)

交付要求：前向最高累计GPU时间的kernel及调用次数，并与forward+backward比较。

- [ ] 1. 按前向launch关联筛选CUDA kernel，不只用时间轴垂直对齐判断归属。
- [ ] 2. 按kernel名称累计时长，记录名称、调用次数、总时长、所选step/rank。
- [ ] 3. 同样处理forward+backward；比较第一名是否变化并写1–2句。

### (c)

交付要求：非matmul的重要kernel，1–2句。

- [ ] 1. 从kernel统计表找elementwise/reduction/softmax等项，追到对应PyTorch操作。
- [ ] 2. 计算各项占同一个分母的比例；明确按GPU活动时间还是wall-clock。
- [ ] 3. 截图或表格保留证据；不要把所有GPU非活跃时间都归因于launch。

### (d)

交付要求：完整A1 AdamW训练步的算子时间占比与推理比较。

- [ ] 1. 使用train而不是logits.mean代理损失；核对optimizer确实是A1 AdamW。
- [ ] 2. 对相同模型/精度分别采forward和train，按一致规则分类kernel。
- [ ] 3. 检查optimizer内的同步是否在等待早先工作；不要直接把CPU optimizer范围当成独立GPU计算时间。

### (e)

交付要求：self-attention内softmax与matmul的耗时/FLOPs对照，1–2句。

- [ ] 1. 定位同一层同一步的QK、softmax、PV操作并记录实际形状。
- [ ] 2. 独立计算该形状的matmul FLOPs和softmax主要操作数，写清计数约定。
- [ ] 3. 把时间比和FLOPs比放在一起解释，不把低FLOPs自动当作低耗时。

## 产物占位

- [ ] 填写 [tables/nsys_profile.csv](../tables/nsys_profile.csv)（当前只有表头，无任何结果）。
- [ ] 按 [figures/nsys_profile_timeline.png.todo.md](../figures/nsys_profile_timeline.png.todo.md) 生成真实截图。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
