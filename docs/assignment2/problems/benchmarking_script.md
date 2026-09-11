# benchmarking_script

**Benchmarking Script · 4 分 · 讲义第 3 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第3页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/baseline/cli.py](../../../src/cs336_systems/baseline/cli.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite benchmarking --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

## 手把手 TODO

### (a)

交付要求：参数化计时脚本，覆盖 forward / forward+backward / 完整训练。

- [ ] 1. 先读 baseline/cli.py、transformer.py、common.py；确认采用 A1 model/loss/AdamW，不另写一套模型。
- [ ] 2. 用 --device cpu 和很小的尺寸跑三种 mode；检查 train 会更新参数、forward 不建 autograd 图。
- [ ] 3. 检查随机输入/初始化不在计时内，warmup 在计时前，每个正式 step 完成后同步；保存命令、seed 和软件版本。
- [ ] 4. 已有 baseline 是可复用起点；把本题交付所用 commit 与接口说明记录到 writeup，不因代码存在就勾选整个实验完成。

### (b)

交付要求：5 次 warmup、10 次测量，各模型尺寸的均值/标准差及1–2句说明。

- [ ] 1. 预览 --suite benchmarking --list-cases，选 warmup=5 的 small/medium/large/xl/10B 及三种mode。
- [ ] 2. 按实际硬件逐项运行，不能拿当前tiny配置充当small；逐行记录OOM和环境。
- [ ] 3. 先记录每种mode的累计时间；若给出纯backward/optimizer时间，明确是差分估算还是单独同步测量，不混淆。
- [ ] 4. 把 mean/stdev/10次原始记录的路径填入表格，再用1–2句回答波动大小。

### (c)

交付要求：无预热与1/2次预热的对比，2–3句解释。

- [ ] 1. 在独立进程分别跑 warmup=0/1/2/5，其他参数完全相同。
- [ ] 2. 检查首轮库加载、缓存/内存分配及编译影响；不要把不同进程的状态假定为完全相同。
- [ ] 3. 将每种设置填表，比较均值与标准差后再填写原因；不预填“必定更慢”的结论。

## 产物占位

- [ ] 填写 [tables/benchmarking_script.csv](../tables/benchmarking_script.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
