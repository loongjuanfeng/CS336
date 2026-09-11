# CS336：Assignment 1 & 2

## 目录与接口

| 路径 | 用途 |
| --- | --- |
| `src/cs336_basics/` | A1：tokenizer、Transformer、Optim、训练与生成 |
| `src/cs336_systems/` | A2:baseline、checkpoint、DDP、Flash Attention |
| `docs/upstream/` | 官方 repo，放在本地参考用 |

A1 为 A2 的基建：

```python
from cs336_basics.model.transformer import TransformerLanguageModel
from cs336_basics.model.primitives import Linear, Embedding, RMSNorm
from cs336_basics.optimization import AdamW
from cs336_basics.training import cross_entropy, get_batch

model = TransformerLanguageModel(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=688,
    rope_theta=10000.0,
)
```

## Assignment 2 实现

测试：

```bash
just test               # test all
just test-a1
just test-a2            # CUDA required
just check              # lint & type check
```

## 环境

使用 uv 管理的项目。遵循 uv 的环境配置规范。

保留两个独立 `uv` 环境：

- `local`：`wheels/` 中自编译的 PyTorch，以及 Triton nightly。
- `default`：正常 python hosted 的 wheels。

```bash
just environment local
just environment default
```

在 `.venvs` 做的软链接。需要配有 direnv。