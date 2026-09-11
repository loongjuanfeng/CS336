# TODO：填入真实实验数据

所有CSV当前只有表头，没有任何测量或答案。

1. 先读同名 `../problems/*.md`，核对配置与计时定义；参考侧可复用baseline，优化侧须自行完成。
2. 从真实metrics/日志填入值，保留source_run、commit和software_versions。
3. `status`只在真的运行后填 `ok`、`oom` 或 `error`；未运行不填0，也不填虚构的“正常结果”。
4. latency统一使用列名指示的ms；MiB按2^20 bytes，通信message_bytes记录真实字节。
5. 区分global/per-rank batch、FP32/BF16/内部累加精度和profiler/无profiler。
6. matched configurations才计算speedup；无法配对的记录不要强行除。
7. `experiments.assemble_writeup_tables`是待实现的汇总入口；完成前可手工核对填表。
8. 最终把表排版进writeup；CSV本身不是PDF中的表格。
