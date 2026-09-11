# Assignment 2 reference experiments

Run from the repository root:

```bash
uv run python -m cs336_systems.baseline
uv run python -m cs336_systems.baseline --precision bf16
uv run python -m cs336_systems.baseline --compile --precision bf16
uv run python -m cs336_systems.baseline --modal --nsys
```

The default is A2 **small**, global batch 4, context 512, eager FP32, 5 warmups and
10 measurements. The Transformer is the A1 implementation, including explicit
quadratic causal attention, cross-entropy and A1 AdamW. TF32 is disabled. BF16 on
the Transformer means **autocast**, with FP32 master parameters and optimizer
states; BF16 for standalone attention means BF16 Q/K/V inputs. This is not an
optimized FlashAttention, overlapped DDP, sharded optimizer or FSDP implementation.
The original `scripts/baseline.py` remains a compatibility launcher only.

## Individual runs

| Experiment | Options | Reference measured |
|---|---|---|
| `transformer` (default) | `--model-size small\|medium\|large\|xl\|10B`, dimension overrides | A1 model |
| `transformer` | `--mode forward\|backward\|train` | Inference without autograd; forward+loss+backward; full training |
| `transformer` | `--precision fp32\|bf16`, `--compile` | Eager/compiled FP32/autocast BF16 |
| `transformer` | `--memory-snapshot` | PyTorch allocator snapshot for memory_viz |
| `transformer` | `--checkpoint-layers N` | Native non-reentrant checkpoint over groups of N layers; 0 disables it |
| `attention` | `--d-model`, `--context-length`, `--causal`, `--compile` | Explicit PyTorch attention without a head dimension |
| `attention` | `--precision fp32\|bf16`, `--timing triton` | FP32/BF16 reference for FlashAttention comparison |
| `communication` | `--world-size 2`, `--message-mb 100`, `--backend nccl\|gloo` | Blocking FP32 all-reduce, decimal MB |
| `ddp` | `--world-size 2`, `--gradient-reduce individual\|flat` | Replicated model/optimizer, blocking averaged gradients after backward |
| `precision` | `--device cuda` | Toy Linear/LayerNorm network: actual FP16/BF16 autocast output, loss, parameter and gradient dtypes |
| `accumulation` | `--device cpu` | The handout's accumulation accuracy examples, plus BF16 |

Standalone attention defaults to batch 8, dimension 64, context 256, and 100
measurements. It reports separately synchronized forward/backward times, memory
allocated immediately before backward, and combined step time. `--timing triton`
also reports `triton.testing.do_bench` forward/backward/combined latencies (25 ms
warmup, 100 ms repetition); backward reuses a retained graph and clears leaf grads.
When combined with `--compile`, buffer donation is disabled for this retained-graph
measurement. This option measures only the PyTorch side of the FlashAttention comparison.

DDP batch size is **global**, split evenly across ranks. Each rank uses a different
random batch, but starts from identical parameters. Step latency uses the slowest
rank at each iteration. Communication timing explicitly synchronizes after
backward before timing all-reduce; these extra boundaries are part of this blocking
reference. Rank reports retain initialization, before-optimizer and after-optimizer
allocated/reserved/peak memory. Peak counters are reset after warmup, so optimizer
states already exist in the steady-state measurements. These are not independent
per-phase peak counters or a cold first-optimizer-step memory measurement.

## Handout suites

```bash
# Preview without allocating a GPU or running any experiment.
uv run python -m cs336_systems.baseline --suite mixed-precision --list-cases
# Execute one listed case on Modal.
uv run python -m cs336_systems.baseline --suite mixed-precision --case 1 --modal
# Execute the entire selected suite (can be expensive).
uv run python -m cs336_systems.baseline --suite attention --modal
```

Suite values override individual model/timing flags. `--device`, `--modal`, and
backend apply to all cases; the flash/leaderboard presets request B200 hardware.
Each case runs in a fresh subprocess and produces its own output folder. CUDA OOM
is recorded as `status: oom`; other process failures are recorded as errors in the
suite report. A suite can finish with failing/OOM cases: inspect `suite.json`.
`--suite all --list-cases` inventories all cases; it does not mean the assignment's
written answers or optimized implementations are complete.

| Suite | Handout coverage and limits |
|---|---|
| `benchmarking` | All five model sizes, all three modes, warmups 0/1/2/5; 10 measurements |
| `nsys` | small/medium, contexts 256/512/1024, forward/train; choose longer individual contexts if memory allows (the handout asks for the longest fitting context) |
| `mixed-precision` | Toy-network dtype probe, then all five model sizes × three modes × FP32/BF16 |
| `memory` | xl, contexts 128/2048, forward/train, FP32/BF16; allocator snapshots |
| `checkpointing` | xl/context 2048; baseline plus group sizes 1/2/4/5/6/8/16/32, including neighbors around sqrt(32); no claim of implementing the theoretical recursively optimal strategy |
| `attention` | batch 8; dimensions 16/32/64/128 × contexts 256/1024/4096/8192/16384; 100 measurements |
| `compile` | Eager/compiled attention grid and all model sizes/modes |
| `flash` | **PyTorch reference only**, B200, batch 1, causal, lengths 128…65536, dimensions 16…128, FP32/BF16, do_bench |
| `communication` | 2/4/6 processes × 1/10/100/1000 MB FP32 all-reduce |
| `ddp` | xl on 2 GPUs; individual vs flat blocking gradient all-reduce |
| `sharding`, `fsdp` | **Unsharded reference only**, xl on 2 GPUs; compare against your future assessed implementations |
| `leaderboard` | Specified 34-layer/4096-dim model, vocab 151936, context 32768, global batch 2, BF16, 2 B200; naive replicated reference may OOM. Uses the common baseline timer, not the official 30 s rep/10 s warmup do_bench submission harness |
| `accumulation` | Numerical accumulation example results |

Written derivations, profiler screenshots/interpretations, interpretation of dtype
results, and all custom optimized implementation comparisons remain separate
assignment deliverables. In particular, passing these baseline checks does not
implement `tests/assignment2/adapters.py`.

## Execution and artifacts

`--modal` uses a fetched official prebuilt
`torch==2.14.0.dev20260811+cu130` on Python 3.14; local runs use the active Python
environment. Default remote GPU is A100-80GB. Use `--gpu H100` or `--gpu B200`
without a count; each remote container requests 4 CPU cores, and distributed experiments append `--world-size` automatically.
CPU/gloo runs are useful for small correctness checks, not CUDA performance data.

Outputs are under `results/baseline/run-*/`:

- `metrics.json`: exact configuration, runtime version, timings and memory.
- `trace.nsys-rep`, `trace.stats.log`: with `--nsys`, only measurement steps are
  captured; PyTorch NVTX labels and CUDA allocation tracking are enabled. NVTX
  describes CPU regions, not exact GPU execution intervals. Profiling affects timing.
- `memory.pickle`: with `--memory-snapshot`, open in PyTorch memory_viz; distributed
  snapshots are stored in `rank-N/`. Compilation and warmup are excluded.
- `suite.json`, `summary.csv`: combined results for a suite.

Remote artifacts are streamed back to the same local structure. Runs never replace
previous output directories. For throughput comparisons, use unprofiled runs and
hold configuration, precision and software versions constant. Memory snapshots
include live tensors at measurement start plus the bounded allocation history with
Python stacks. Native stack symbolization is omitted to keep snapshot export practical.
