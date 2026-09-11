# OPTIONAL TODO：全 Triton FlashAttention backward

这不是 `flash_backward` 必做题的前置条件。必做方案是自写PyTorch反向并使用torch.compile。

- [ ] 先让两种forward、causal和编译反向全部通过官方测试。
- [ ] 阅读 `flash_attention.launch_flash_backward` 的stub，规划dQ与dK/dV输出tile归属。
- [ ] 明确原子累加、两遍遍历或临时buffer的取舍；检查哪些跨program共享写会产生race。
- [ ] 保持必要的FP32累加，检查cast位置，逐梯度与参考实现比较。
- [ ] 再考虑causal跳过整块、对角tile和非对角tile分离、autotune与硬件专用加载。
- [ ] 用相同B200网格测forward/backward/combined，确认优化了端到端而非只某一小算子。
- [ ] 在报告中单独标为可选工作，不能掩盖仍未完成的必做产物。
