# CS336 2026：Assignment 2 学习起点

本仓库提供**完成的 Assignment 1 编程基线 + 尚未作答的 Assignment 2 骨架**。
A1 基于原有实现整理，通过官方编程测试；不包含完整语料训练、消融实验或实验报告。
A2 的 FlashAttention、DDP、FSDP 和优化器状态分片留给你实现。

## 先运行这三条命令

```bash
just test-a1       # 官方 A1 编程测试
just smoke         # CPU 小模型训练、验证、保存 checkpoint、生成 token
just collect-a2    # 查看 A2 的 14 个测试，不执行未完成的题目
```

依赖尚未同步时先运行 `just environment local`；新机器没有本地 wheel 时使用
`just environment default`。命令使用 `uv`，并需要安装 `just`。
也可以直接运行 `uv run pytest tests/assignment1 -q` 和 `uv run cs336-train --smoke`。

## 目录与接口

| 路径 | 用途 |
| --- | --- |
| `src/cs336_basics/` | 已完成的 A1：tokenizer、Transformer、优化器、训练与生成 |
| `src/cs336_systems/` | 你编写 A2 代码的位置，当前只有包说明 |
| `tests/assignment1/` | 固定版本的官方 A1 测试、fixtures、snapshots 和已接好的 adapters |
| `tests/assignment2/` | 官方 A2 测试及待实现的 `adapters.py` |
| `tests/local/` | 原有回归测试及补充的接口、流式编码、CUDA 回归测试 |
| `examples/` | 保留的 PyTorch notebook 和 Triton 示例 |
| `docs/upstream/` | 官方作业 PDF、README、变更记录和许可证 |

A2 实验可以直接使用基础库：

```python
from cs336_basics.model.transformer import TransformerLanguageModel
from cs336_basics.model.primitives import Linear, Embedding, RMSNorm
from cs336_basics.optimization import AdamW
from cs336_basics.training import cross_entropy, get_batch

model = TransformerLanguageModel(
    vocab_size=10000, context_length=256, d_model=256,
    num_layers=4, num_heads=4, d_ff=688, rope_theta=10000.0,
)
```

统一使用 `cs336_basics`，直接从定义模块导入；包的 `__init__.py` 不转发接口。
旧的顶层导入、分组转发模块和 `BasicsTransformerLM` 别名已移除。

基础模型保留朴素 attention、手写损失和优化器，适合拿来测量、分析和优化。

## A1 训练与恢复

无数据下载的 CPU / CUDA smoke：

```bash
just smoke
uv run cs336-train --smoke --device cuda --steps 10 --output-dir checkpoints/cuda-smoke
```

真实 token 数据支持一维整数 `.npy`（以 mmap 加载）或原始二进制 memmap。
`--data-dtype` 只控制原始二进制文件的 dtype；数据中 token ID 必须落在模型词表范围内。

```bash
uv run cs336-train \
  --train-data data/train.npy --validation-data data/valid.npy \
  --vocab-size 10000 --context-length 256 --d-model 256 \
  --num-layers 4 --num-heads 4 --d-ff 688 \
  --steps 1000 --warmup-steps 100 --decay-steps 1000 \
  --batch-size 8 --device cuda --report-every 50 \
  --output-dir checkpoints/a1

uv run cs336-train \
  --train-data data/train.npy --validation-data data/valid.npy \
  --resume checkpoints/a1/checkpoint.pt --steps 1500 \
  --batch-size 8 --device cuda --output-dir checkpoints/a1
```

输出目录包含 `config.json` 和滚动更新的 `checkpoint.pt`。
恢复时自动读取模型和学习率配置，恢复权重、优化器状态、已完成迭代数；
`--steps` 表示最终总步数。原来的学习率计划继续生效，超过 `decay-steps` 后保持最小学习率。
不保存数据采样位置或随机数状态，恢复不保证与未中断训练逐位相同。
可用 `uv run cs336-train --help` 查看全部参数。

## Tokenizer

```python
from cs336_basics.tokenizer.core import Tokenizer
from cs336_basics.tokenizer.training import train_bpe

vocab, merges = train_bpe("data/corpus.txt", 10000, ["<|endoftext|>"])
tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
tokenizer.save("data/vocab.json", "data/merges.json")
tokenizer = Tokenizer.from_files("data/vocab.json", "data/merges.json")
ids = tokenizer.encode("hello<|endoftext|>")

with open("data/corpus.txt", encoding="utf-8") as source:
    for token_id in tokenizer.encode_iterable(source):
        pass  # 可在这里分批写入训练数据文件
```

文件采用本仓库的 Base64 JSON 格式，完整保留字节词表、merge 顺序和特殊 token。
训练按字节序取同频 pair 中较大者，直到目标词表大小或 pair 耗尽；解码非法 UTF-8
使用替换字符。流式编码保留跨输入块的 pre-token 和特殊 token 边界。
BPE 增量计数只更新受合并影响的 pre-token；语料分块读取，计数表仍随不同 pre-token 数量增长。

Tokenizer 统一使用显式 `vocab` 和 `merges`；旧 `Vocabulary` 接口、旧 JSON 格式、
`Corpus` 和 `training_loop_*` 已移除。使用 `train_bpe(input_path, vocab_size, special_tokens)`
训练，并用 `Tokenizer.save` / `Tokenizer.from_files` 保存和读取当前双文件格式。
参数统一采用 `max_*` / `min_*`，模型采用 `context_length`、`head_dim` 和 `rope_theta`；
CLI 参数、checkpoint 字段和模型权重键保持不变。

## 从哪里开始 Assignment 2

阅读 [A2 官方讲义](docs/upstream/assignment2-systems/cs336_assignment2_systems.pdf)。
当前固定的是 **Spring 2026**，其分布式部分包含 FSDP，和 2025 版有所不同。

1. 从 `benchmarking_script` 开始，在 `cs336_systems` 编写自己的测量脚本。
2. 继续计算与内存 profiling、混合精度和编译相关实验；Nsight Systems 的 `nsys` 需另行安装。
3. 实现 PyTorch / Triton FlashAttention，并连接 `tests/assignment2/adapters.py`。
4. 按讲义完成 DDP、优化器状态分片与 FSDP。

测试命令：

```bash
just test             # 自有回归 + 官方 A1；默认 pytest 也是这两组
just test-local
just test-a1
just collect-a2
just test-a2          # 尚未作答时应失败，不是环境验收命令
uv run pytest tests/assignment2/test_attention.py -q
just check            # lint + 类型检查；不改写官方测试
```

A2 adapters 保留官方 `NotImplementedError`，没有添加跳过或 xfail 来掩盖未实现项。
本机只有一张 GPU；多进程测试即使能共享单卡，也不能替代多卡通信性能实验。

## 环境

保留两个独立 `uv` 环境：

- `local`：使用 `wheels/` 中你编译的 PyTorch，以及现有 Triton nightly 来源。
- `default`：通过 `uv sync --no-sources` 使用 PyPI 的预编译 PyTorch / Triton。

```bash
just environment local
just environment default
```

选择保存在被忽略的 `.env` 中，`.venv` 指向 `.venvs/local` 或 `.venvs/default`。
`.envrc` 为支持 direnv 的 shell 设置环境；两种环境切换逻辑沿用原仓库配置。

本仓库保留 **Python 3.14、PyTorch >=2.13** 的现有依赖策略。
官方 2026 使用 Python `>=3.12,<3.14`、PyTorch `~=2.11.0`，所以本仓库是接口和作业内容对齐，
并非官方运行环境的复制。两种来源共享一个 `uv.lock`，切换来源时 uv 会重新解析相应依赖。
验证详情见 [验证记录](docs/VALIDATION.md)，固定提交与本地调整见 [来源记录](docs/UPSTREAM.md)。

## 打包

```bash
./test_and_make_submission.sh
```

脚本执行 A2 测试、写入测试报告，再生成 `dist/cs336-2026-assignment2.zip`；
若测试失败仍保留结果包，但返回失败状态。包内保留测试所需的 JSON、文本、pickle、PT fixtures，
不包含本地 wheel、环境、实际训练数据、checkpoint 或密钥环境文件。
解包后没有本地 wheel，脚本自动选用独立的 default 环境与 PyPI 来源。
这只是本地材料打包，不会上传或提交到任何平台。
