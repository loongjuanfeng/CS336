# leaderboard

**Leaderboard: fastest training step · 10 分 · 讲义第 48 页起**

> 状态：TODO。本文是操作指南；答案写入 [writeup.md](../writeup.md) 的同名章节。

## 先看哪里

- 讲义：[本地PDF](../../upstream/assignment2-systems/cs336_assignment2_systems.pdf)，从第48页该题开始读，必要时回看前文公式。
- 代码：[src/cs336_systems/leaderboard.py](../../../src/cs336_systems/leaderboard.py)。

已有基线的配置预览（不运行GPU）：

```bash
uv run python -m cs336_systems.baseline --suite leaderboard --list-cases
```

TODO：看清配置后再选择 `--case N --modal` 或自定义小尺寸。完整suite可能花费较多资源。

**边界：此suite只有已有参考实现/配置，不能替代本题要求的自写优化实现或正式leaderboard计时。**

## 手把手 TODO

### 主交付

交付要求：精确8B配置、两张B200的完整step最好墙钟时间，冷缓存总运行限制。

- [ ] 1. 先完成leaderboard.py的构建/训练/正式计时三个stub；baseline leaderboard suite只是配置与内存预演。
- [ ] 2. 核对34层、4096维、FFN11008、32heads、vocab151936、context32768、global batch2、BF16、causal。
- [ ] 3. 保持模型/loss/update语义正确；不得省略loss或AdamW，先进行小输入等价检查。
- [ ] 4. 按官方rep=30000ms,warmup=10000ms测完整step；多rank循环次数必须协调。
- [ ] 5. 在独立空缓存启动，从启动至结束限制10分钟；报告真实step时间及是否优于10秒。
- [ ] 6. 保存commit/env/命令/日志和正确性证据，最后按SUBMISSION.md手动上传。

## 产物占位

- [ ] 填写 [tables/leaderboard.csv](../tables/leaderboard.csv)（当前只有表头，无任何结果）。

## 本题完成前核对

- [ ] 所有小问都有对应答案/代码/真实产物；单位、配置与讲义匹配。
- [ ] OOM、硬件替代和未运行项已经明确说明；没有将TODO或空表当成测量值。
- [ ] writeup中引用的图表可追溯到原始记录；代码仍通过原始官方测试要求。
