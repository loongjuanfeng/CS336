# 官方来源与本地调整

固定时间：2026-09-06。后续学习期间不自动跟随 upstream main 变化。

| 作业 | 官方仓库 | 固定提交 |
| --- | --- | --- |
| A1 | https://github.com/stanford-cs336/assignment1-basics | `a158843b20107949f1a8d7df1b05cd33b9166712` |
| A2 | https://github.com/stanford-cs336/assignment2-systems | `ca8bc81a59b70516f7ebb2da4808daade877c736` |

两者均为 Spring 2026。`upstream/assignment1-basics/` 和 `upstream/assignment2-systems/`
保留对应提交的原始讲义、README、CHANGELOG、LICENSE。原始 README 中的相对链接仍描述官方目录；
使用本仓库时以根 README 为准。

官方测试分别复制到 `tests/assignment1/` 和 `tests/assignment2/`，包括全部 fixtures 和 snapshots。
仅对 A1 做以下接入调整：

- 实现 `adapters.py`，调用本仓库的 A1 基础库；保留官方函数签名。
- `conftest.py` 的两个 snapshot 默认目录从 `tests/_snapshots` 改为相对于该文件的 `_snapshots`。

官方测试断言、数值容差、BPE 速度阈值、snapshots 和 A2 adapters 均保持原样。
上游的 AGENTS / CLAUDE 文件未导入，本地项目也不引入其额外代理工作流。

A1 代码来源是本仓库原实现，不是替换为助教代码。主要语义校正：

- 显式 merge 顺序、同频 pair 的字节序、单次出现 pair 的处理、特殊 token 和流式边界。
- UTF-8 解码按讲义使用替换字符。
- AdamW 按讲义将 epsilon 加在未偏差校正的二阶矩平方根之后，权重衰减先于梯度更新。
  官方测试也接受原有 PyTorch 形式；这里选择讲义形式便于阅读和推导。

打包脚本为适配本仓库重新编写，不沿用上游会误排除测试 fixtures 的扩展名过滤规则。
本地环境版本差异见根 README。
