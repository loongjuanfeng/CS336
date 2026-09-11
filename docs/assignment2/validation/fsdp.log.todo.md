# TODO：fsdp 验证日志

命令：

```bash
uv run pytest tests/assignment2/test_fsdp.py -q
```

- [ ] 实现并接adapter后执行；保留完整输出与exit code。
- [ ] 记录设备/backend、软件版本和commit；分布式测试重复5次。
- [ ] 保存真实日志，不能把“收集成功”当作“测试通过”。
- [ ] 当前所有未实现入口应继续显式失败；不要写虚构的passed结果。
