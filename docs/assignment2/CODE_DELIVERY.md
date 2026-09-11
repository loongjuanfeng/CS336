# A2 代码交付

范围：Spring 2026 v26.1.3 常规代码与正确性验证。性能 sweep、通信性能实验、leaderboard、理论答案、图表和报告 PDF 不在本次交付范围。`experiments.py` 和 `leaderboard.py` 保留的实验占位不参与验收。

## 实现与接口

| 实现 | 行为 |
| --- | --- |
| FlashAttentionPytorch | PyTorch 分块在线 softmax，保存 Q/K/V/O/L |
| FlashAttentionTriton | 自写 Triton 前向，支持 causal、非连续输入和尾块 |
| 两种 attention 反向 | 自写概率重计算公式，FP32 累加，缓存的 torch.compile tile |
| NaiveDDP / FlatDDP / OverlapDDP | 初始化广播、梯度平均；官方 adapter 使用逐参数异步 OverlapDDP |
| ShardedOptimizer | 参数 owner 保存优化器状态、更新后广播；支持动态参数组和各 rank 独立 checkpoint |
| FSDP | A1 Linear/Embedding 的 FP32 master shard；低精度 gather、权重释放、NCCL reduce-scatter、相邻层预取 |
| checkpointing | 原生非重入 checkpoint 实现嵌套前缀重计算；非嵌套分组复用 baseline |

FSDP 对 norm 等参数保持复制并同步梯度，普通优化器须在包装模型之后创建。共享权重保持同一个 Parameter，重复调用同一层可累加梯度。`gather_full_params()` 在所有 rank 重建原始名字的完整 FP32 参数；常规 `state_dict()` 保存本地 shard。自写分布式容器要求所有 rank 执行相同计算图，不实现动态 rank 分支、DDP `no_sync` 或生产级弹性恢复。

FlashAttention 的 causal 约定是全局 query 索引大于等于 key 索引；所有 batch 维一致，Q/K/V 使用相同 device/dtype/head dimension。前向不生成完整注意力矩阵。反向使用固定 64×64 tile 和 FP32 临时张量，优先验证正确性，不作已达性能指标的声明。全 Triton backward 不在本次范围。

## 运行与环境

```bash
uv sync --no-sources                 # 干净机器使用公开 wheel
just verify-a2-local                # GTX 1650 + CPU 小量检查
just verify-a2-modal                # 两张 A100-80GB，完整正确性验证
UV_NO_SOURCES=1 just test           # A1 + 本地回归
UV_NO_SOURCES=1 just check          # lint 与类型检查
```

本地为 GTX 1650、4GB、SM7.5；FP32 和 FP16 做小尺寸检查，不要求原生 BF16。Modal 使用 Python 3.14、`torch==2.14.0.dev20260811+cu130`；完整版本信息记录在每次运行的 `validation.json`。

原始官方分布式测试固定使用 Gloo 和 TCP rendezvous，并自动选择可见 GPU。完整验证显式隐藏 CUDA，让这些用例使用 CPU/Gloo；另以 `CS336_TEST_BACKEND=nccl` 执行每 rank 一张真实 GPU 的补充检查。所有断言、数值阈值、fixtures 均未修改。

本机系统反向 DNS 查询会阻塞 PyTorch TCPStore 初始化，最小 TCPStore 程序也可复现；调用栈停在 `getnameinfo` / `_nss_resolve_gethostbyaddr_r`。本地小量检查采用 FileStore rendezvous，不修改系统 DNS。官方原样测试由 Modal 完成。

## 验收与证据

本地入口执行 6 项官方 attention、5 项 attention/checkpoint 边界检查、3 项 CPU 分布式补充检查。
Modal 执行 17 个任务：官方 attention 6 项、边界 5 项，以及连续 5 轮的官方 CPU/Gloo 8 项、CPU 补充 3 项、双卡 NCCL 补充 3 项。总计 81 次测试执行，必需用例不能以 skip/xfail 计作通过。

补充检查涵盖非连续输入与上游梯度、不同 query/key 长度及尾块、FP16/BF16、共享及冻结参数、重复调用层、动态参数组、空 owner、学习率变化、优化器状态恢复、FSDP padding/梯度形状/master dtype 和完整权重释放。

原始日志与 JUnit 留在 `results/assignment2/local-*` / `modal-*`。`validation.json` 保存源码/依赖 SHA-256、验证脚本 SHA-256、设备、精度相关环境变量、命令、退出码和耗时。失败任务保留日志并返回非零；不会将失败标记成通过。

2026-09-08 最终验收：

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 本地 GTX 1650 / CPU | 14 passed，0 skipped | `results/assignment2/local-10pbk7jz/` |
| Modal 双 A100-SXM4-80GB | 81 passed，0 skipped；17 个任务全部成功 | `results/assignment2/modal-t1eku9zu/` |
| A1 + 全部本地回归 | 151 passed，1 个官方原有 xpass | `results/assignment2/regression.xml` |
| 独立目录公开 wheel 安装 | 14 passed，导入路径指向解压目录 | `results/assignment2/clean-install.xml` |
| Ruff / ty / git diff --check | 全部通过 | 工作区最终检查 |

Modal 的最终源码与依赖清单哈希为 `a2cf509b575b70ce1e5f155ab78ac3e9be30cb2d5c29d022a411b3ea67d9cd02`。
远端 Python 3.14.2、PyTorch 2.14.0.dev20260811+cu130、Triton 3.8.0；本地为 Python 3.14.7、PyTorch 2.14.0+cu130。
早期 attention 编译缓存失败的日志也保留在 `modal-70ws4foh`；已修复后完整重跑，不用于最终验收。

## 代码包

```bash
./test_and_make_submission.sh
# 或复用与当前源码匹配的完整验证记录：
./test_and_make_submission.sh results/assignment2/modal-XXXX
```

输出 `dist/assignment2/code.zip`，包括源码、官方测试和 fixtures、配置、入口脚本及 `results/assignment2/verified/` 中的完整验证记录。脚本严格传播测试失败，打包器拒绝失败、缺失或源码哈希不匹配的完整验证。不打包本地 wheel、虚拟环境、训练数据或凭据，不要求报告 PDF，不执行上传。

在解压目录执行 `uv sync --no-sources` 后可直接运行上述入口。打包验证只覆盖本次代码范围，不表示原工作台中的 27 题报告均已完成。

若 shell 继承了另一仓库的 `UV_PROJECT_ENVIRONMENT` 或 `VIRTUAL_ENV`，使用
`env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --no-sources`
确保安装到解压目录。提交脚本首次安装时已显式使用这个隔离方式。
