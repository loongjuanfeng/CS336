# 验证记录

日期：2026-09-06。验收使用保留的 `local` 环境。

## 环境

- Python 3.14.7
- PyTorch `2.14.0a0+git2b3ec34`（原有本地 wheel）
- Triton 3.8.0（原有 nightly 来源）
- CUDA 可用，1 张 NVIDIA GeForce GTX 1650 with Max-Q Design

## A1 KISS 重构后的验证

同日基于现有工作区完成重构，使用同一个 local 环境：

- `.venv/bin/python -m pytest tests/local tests/assignment1 -q`：**124 passed, 1 xfailed**。
  删除仅验证旧接口的测试，将 tokenizer 行为回归迁移到 `Tokenizer` / `train_bpe`；
  官方测试断言与 fixtures 未改动，BPE 速度阈值保留。
- ruff、ty、`git diff --check`、`uv lock --check` 通过。
- CPU smoke 从 3 步恢复到 5 步；CUDA smoke 完成训练、验证与生成；
  现有 CUDA checkpoint 恢复后继续更新的回归通过。
- A2 仍完整收集 14 项测试，本次不执行未实现的 A2 测试。
- 清除旧 `build/lib` 缓存后构建 wheel，确认恰好包含当前 15 个 Python 源文件，
  无旧包或转发模块；console entrypoint 仅保留 `cs336-train`。
- 直接从 wheel 导入 tokenizer 并编码成功，未加载 torch。
- 搜索活动代码、README 与 examples，未发现已删除的导入路径。
- 未运行完整语料训练或远程 Modal 作业。

以下保留重构前的历史验证记录；旧接口和 wheel 内容描述不再代表当前 API。

## 重构前已执行

| 检查 | 结果 |
| --- | --- |
| `uv run pytest -q` | **148 passed, 1 xfailed**；约 12 秒 |
| 官方 A1 | 47 passed，1 xfailed；未放宽 BPE 的 1.5 秒速度阈值 |
| 自有回归 | 101 passed，包含跨块编码、新旧类型一致性、CUDA 训练与 checkpoint 恢复 |
| `just check` | ruff 与 ty 均通过 |
| `just collect-a2` | 14 项完整收集 |
| `uv lock --check` | 当前 local 锁文件有效 |
| `uv build --wheel` | 构建成功，新包、旧兼容模块及 tokenizer 文件 I/O 模块均在 wheel 内 |
| tokenizer-only import | 不加载 torch，保留仅安装 regex 的 Modal tokenizer 作业兼容性 |
| 官方文件比对 | A1 的 39 个测试/fixture 文件中仅 adapters、conftest 有预定修改；A2 的 10 个文件全部原样 |

唯一的 xfail 是官方 `test_encode_memory_usage`：官方本来就预期一次性 `encode`
无法在所设内存限额内处理整个语料。没有为本地实现新增跳过或 xfail。

训练 CLI 已实际验证：

- 合成语料 CPU 训练 3 步，再恢复到第 5 步；验证 loss 下降，生成 token 正常。
- 合成语料 CUDA 训练、验证与生成。
- `.npy` mmap 训练及验证输入；原始 `uint16` memmap 训练输入。
- 模型参数与优化器 checkpoint 在 CUDA 恢复后，再次更新与原模型结果一致。

## A2 的预期状态

实际执行 `./test_and_make_submission.sh`：**14 failed**，返回状态 **1**。
逐项检查 JUnit 报告，所有失败均源于 A2 adapters 中的 `NotImplementedError`，
其中多进程测试由 `ProcessRaisedException` 包装。没有缺失导入或 fixture 导致的失败。
这些失败表示题目尚未作答，不属于 A1 基线验收失败。

生成的 ZIP 已检查：完整保留测试与 fixtures、测试失败报告和 shell 可执行权限，
排除 `.env`、本地 wheel、虚拟环境、训练数据和 checkpoint。
报告与 ZIP 位于被忽略的 `results/` 和 `dist/` 下。

## 验证边界

- default 环境仅执行 `uv sync --no-sources --dry-run`，解析成功（125 个包）；
  未在该环境安装和运行完整测试。local 的解析结果为 107 个包。
- 未切换到官方 Python 3.13 / PyTorch 2.11 环境。
- 未运行完整语料训练、消融、A2 性能实验或多卡性能验证。
- Modal 脚本保留并通过静态检查，未启动远程作业。
- Triton 示例仅将主机端 grid 的向上取整写为等价整数算式，以通过当前 ty 的类型检查；
  未改动示例 kernel，也未把示例视为 A2 解答。
