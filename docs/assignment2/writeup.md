# CS336 Assignment 2 — writeup stub

> 状态：未完成。以下只有答案占位，不是作业答案或已完成实验。详细操作在 problems/。

作者：TODO  
日期：TODO  
代码 commit：TODO  
讲义：Spring 2026 v26.1.3

## 实验环境与计时口径

- TODO：GPU型号/数量、CPU分配、CUDA/PyTorch/Triton版本、seed、完整命令与原始结果路径。
- TODO：FP32/TF32/BF16含义、master/activation/state dtype、global/per-rank batch、warmup、同步边界。
- TODO：标出profiler数据、OOM、未运行项；不要把baseline已有记录当成所有正式配置已运行。

## benchmarking_script

[Benchmarking Script：详细TODO](problems/benchmarking_script.md)

### (a)

<!-- 要求：参数化计时脚本，覆盖 forward / forward+backward / 完整训练。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：5 次 warmup、10 次测量，各模型尺寸的均值/标准差及1–2句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：无预热与1/2次预热的对比，2–3句解释。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/benchmarking_script.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## nsys_profile

[Nsight Systems Profiling：详细TODO](problems/nsys_profile.md)

### (a)

<!-- 要求：前向实际耗时与Python计时的比较，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：前向最高累计GPU时间的kernel及调用次数，并与forward+backward比较。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：非matmul的重要kernel，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (d)

<!-- 要求：完整A1 AdamW训练步的算子时间占比与推理比较。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (e)

<!-- 要求：self-attention内softmax与matmul的耗时/FLOPs对照，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/nsys_profile.csv 的真实数据排版为表格，不要只放CSV链接。 -->
<!-- TODO：按 figures/nsys_profile_timeline.png.todo.md 拍摄并插入真实图片及图注。 -->
## mixed_precision_accumulation

[Mixed-Precision Accumulation：详细TODO](problems/mixed_precision_accumulation.md)

### 主交付

<!-- 要求：运行讲义四段累加代码，提交2–3句观察。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/mixed_precision_accumulation.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## benchmarking_mixed_precision

[Benchmarking Mixed Precision：详细TODO](problems/benchmarking_mixed_precision.md)

### (a)

<!-- 要求：列出参数、fc1、LayerNorm、logits、loss和梯度的dtype。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：LayerNorm敏感部分，以及BF16是否需要类似处理，2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：各模型FP32与BF16 mixed precision计时，2–3句趋势。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/benchmarking_mixed_precision.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## memory_profiling

[Memory Profiling：详细TODO](problems/memory_profiling.md)

### (a)

<!-- 要求：xl推理/训练的Active memory timeline各一张和2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：context=128/2048各自forward/full train峰值表。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：xl mixed-precision的forward/train峰值与FP32比较，2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (d)

<!-- 要求：FP32 residual-stream张量大小推导，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (e)

<!-- 要求：低Detail视图中的最大分配大小及来源，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (f)

<!-- 要求：单个TransformerBlock保存的residual、前五操作占比、产生的梯度估计。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/memory_profiling.csv 的真实数据排版为表格，不要只放CSV链接。 -->
<!-- TODO：将 tables/memory_profiling_residuals.csv 的真实数据排版为表格，不要只放CSV链接。 -->
<!-- TODO：按 figures/memory_xl_forward.png.todo.md 拍摄并插入真实图片及图注。 -->
<!-- TODO：按 figures/memory_xl_train.png.todo.md 拍摄并插入真实图片及图注。 -->
<!-- TODO：按 figures/memory_largest_allocations.png.todo.md 拍摄并插入真实图片及图注。 -->
<!-- TODO：按 figures/memory_block_residuals.png.todo.md 拍摄并插入真实图片及图注。 -->
<!-- TODO：按 figures/memory_block_backward.png.todo.md 拍摄并插入真实图片及图注。 -->
## gradient_checkpointing

[Memory-Optimal Gradient Checkpointing：详细TODO](problems/gradient_checkpointing.md)

### (a)

<!-- 要求：允许嵌套、忽略计算代价时的策略/渐近内存/计算量，3–5句及代码草图。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：xl/batch4/context2048，仅一层重计算预算的最佳分组及邻居实测。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/gradient_checkpointing.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## pytorch_attention

[PyTorch Attention Benchmarking：详细TODO](problems/pytorch_attention.md)

### (a)

<!-- 要求：20组普通attention的forward/backward时间、backward前显存与OOM/内存推导。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/pytorch_attention.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## torch_compile

[Torch Compile：详细TODO](problems/torch_compile.md)

### (a)

<!-- 要求：相同attention网格的eager/compiled forward与backward对比表。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：完整Transformer各mode的vanilla/compiled对比表。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/torch_compile.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## flash_forward

[FlashAttention-2 Forward Pass：详细TODO](problems/flash_forward.md)

### (a)

<!-- 要求：纯PyTorch分块FlashAttention autograd.Function，接adapter并通过前向测试。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：自写Triton前向及autograd包装，通过前向测试。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：默认False的causal开关，保存到ctx，覆盖True/False。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/flash_forward.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## flash_backward

[FlashAttention-2 Backward Pass：详细TODO](problems/flash_backward.md)

### 主交付

<!-- 要求：使用PyTorch+torch.compile重计算反向，正确给出dQ/dK/dV。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/flash_backward.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## flash_benchmarking

[FlashAttention-2 Benchmarking：详细TODO](problems/flash_benchmarking.md)

### (a)

<!-- 要求：B200上自写FlashAttention与普通PyTorch的forward/backward/联合时间表。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/flash_benchmarking.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## distributed_communication_single_node

[Distributed Communication (Single Node)：详细TODO](problems/distributed_communication_single_node.md)

### 主交付

<!-- 要求：2/4/6进程、1/10/100/1000MB FP32 all-reduce图表及2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/distributed_communication_single_node.csv 的真实数据排版为表格，不要只放CSV链接。 -->
<!-- TODO：按 figures/communication_scaling.png.todo.md 拍摄并插入真实图片及图注。 -->
## naive_ddp

[Naïve DDP：详细TODO](problems/naive_ddp.md)

### 主交付

<!-- 要求：封装逐参数阻塞梯度平均的DDP，接adapter并通过正确性测试。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## naive_ddp_benchmarking

[Naïve DDP Benchmarking：详细TODO](problems/naive_ddp_benchmarking.md)

### 主交付

<!-- 要求：1节点2GPU、xl的step时间与通信比例，附benchmark设置。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/naive_ddp_benchmarking.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## minimal_ddp_flat_benchmarking

[Minimal DDP with Flat Gradients Benchmarking：详细TODO](problems/minimal_ddp_flat_benchmarking.md)

### 主交付

<!-- 要求：单次flattened all-reduce与逐参数版本的时间对比，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/minimal_ddp_flat_benchmarking.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## ddp_overlap_individual_parameters

[DDP with Overlapping Individual Parameters：详细TODO](problems/ddp_overlap_individual_parameters.md)

### 主交付

<!-- 要求：按参数梯度就绪异步通信的自写DDP容器。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## ddp_overlap_individual_parameters_benchmarking

[DDP Overlapping Individual Parameters Benchmarking：详细TODO](problems/ddp_overlap_individual_parameters_benchmarking.md)

### (a)

<!-- 要求：xl/2GPU中overlap、naive和flat的step对比，1–2句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：naive与overlap各一张显示backward/通信关系的nsys截图。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/ddp_overlap_individual_parameters_benchmarking.csv 的真实数据排版为表格，不要只放CSV链接。 -->
<!-- TODO：按 figures/ddp_naive.png.todo.md 拍摄并插入真实图片及图注。 -->
<!-- TODO：按 figures/ddp_overlap.png.todo.md 拍摄并插入真实图片及图注。 -->
## optimizer_state_sharding

[Optimizer State Sharding：详细TODO](problems/optimizer_state_sharding.md)

### 主交付

<!-- 要求：支持参数组/动态add_param_group的自写分片optimizer。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## optimizer_state_sharding_accounting

[Optimizer State Sharding Accounting：详细TODO](problems/optimizer_state_sharding_accounting.md)

### (a)

<!-- 要求：xl/2GPU各阶段峰值和参数/梯度/状态分解，2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：有无状态分片的iteration时间，2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：与ZeRO stage 1的区别，2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/optimizer_state_sharding_accounting.csv 的真实数据排版为表格，不要只放CSV链接。 -->
## fsdp

[Fully-Sharded Data Parallel：详细TODO](problems/fsdp.md)

### 主交付

<!-- 要求：自写分片权重容器，可用于普通optimizer，支持可选低精度compute。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## fsdp_accounting

[FSDP Accounting：详细TODO](problems/fsdp_accounting.md)

### (a)

<!-- 要求：预期峰值显存节省，2–3句。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：xl/2GPU各层all-gather是否及时，2–3句和nsys截图。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/fsdp_accounting.csv 的真实数据排版为表格，不要只放CSV链接。 -->
<!-- TODO：按 figures/fsdp_all_gather.png.todo.md 拍摄并插入真实图片及图注。 -->
## alternate_ring_all_reduce

[Alternate ring all-reduce：详细TODO](problems/alternate_ring_all_reduce.md)

### 主交付

<!-- 要求：以S,N,W表示的总时间及一句理由。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## data_parallel_calcs

[Data parallel calculations：详细TODO](problems/data_parallel_calcs.md)

### (a)

<!-- 要求：DP backward FLOPs及一句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：DP backward通信时间及一句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：N_DP的通信瓶颈阈值不等式及一句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## fsdp_calcs

[Fully sharded data parallel calculations：详细TODO](problems/fsdp_calcs.md)

### (a)

<!-- 要求：FSDP forward/backward各自FLOPs及两句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：FSDP forward/backward通信时间各一式及两句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：两阶段各自的N_FSDP瓶颈阈值及两句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## tp_calcs

[Tensor parallel calculations：详细TODO](problems/tp_calcs.md)

### (a)

<!-- 要求：TP反向的完整方程序列，得到各shard dW和dX。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：TP forward/backward FLOPs及两句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：TP forward/backward通信时间及两句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (d)

<!-- 要求：两阶段的N_TP通信阈值及两句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## fsdp_tp_calcs

[2D parallelism calculations：详细TODO](problems/fsdp_tp_calcs.md)

### (a)

<!-- 要求：2D forward FLOPs及一句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (b)

<!-- 要求：两轴collective可重叠时的forward通信时间及一句说明。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (c)

<!-- 要求：可重叠时最优N_TP/N_FSDP选择与总N阈值，附推导。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

### (d)

<!-- 要求：两轴共享网络不能重叠时的最优总N阈值，附推导。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

## leaderboard

[Leaderboard: fastest training step：详细TODO](problems/leaderboard.md)

### 主交付

<!-- 要求：精确8B配置、两张B200的完整step最好墙钟时间，冷缓存总运行限制。 -->
TODO：在此填写经过验证的答案/实现与测试证据。

<!-- TODO：将 tables/leaderboard.csv 的真实数据排版为表格，不要只放CSV链接。 -->
