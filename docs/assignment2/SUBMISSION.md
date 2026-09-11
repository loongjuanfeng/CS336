# TODO：最终交付 writeup.pdf 与 code.zip

**本次仅交付代码与验证，操作见 [CODE_DELIVERY.md](CODE_DELIVERY.md)。**
`test_and_make_submission.sh` 已实现回归、验证哈希核对和代码打包，不要求 PDF，也不自动上传。
以下是原始完整作业的报告交付清单，保留供未来完成报告时使用。
讲义要求最终上传排版好的 `writeup.pdf` 与包含实现的 `code.zip`。

## 1. 填完报告

- [ ] 对照manifest的27题、58个小问，将答案写进 `writeup.md`。
- [ ] 将CSV数据真正排版成表格，而不是仅贴路径；按题目要求控制回答句数/段数。
- [ ] 用真实截图替换图片任务stub，保留模型/设备/版本/原始run路径和可读图注。
- [ ] 理论题写定义、公式、单位与推导；不拿性能测量代替符号答案。
- [ ] 标清未运行项、OOM和硬件替代；不要把同为2.14的不同build当成完全相同环境。
- [ ] 对所有TODO逐一核对。源代码中的解释性TODO可留，但答案占位和未实现必做入口不能充当成品。

## 2. 生成真正的PDF

- [ ] 选用LaTeX/Typst/Markdown排版工作流，把 `writeup.md` 转为正式报告源。
- [ ] 如使用Markdown导出，确认支持中文字体、数学公式、跨页表格及本地图片。
- [ ] **先填答案再导出**；将最终PDF命名为 `writeup.pdf`，放到 `dist/assignment2/`。
- [ ] 逐页渲染检查：无截断公式/表格、图片清晰、题号完整、没有剩余TODO或失效链接。
- [ ] 文件能在独立PDF阅读器打开；不能把 `.md` 重命名为 `.pdf`。

## 3. 完成实现与原始测试

- [ ] 按 ADAPTERS.md 接线，运行全部 `tests/assignment2`；保存真实JUnit与终端日志。
- [ ] 分布式目标测试重复5次；记录CPU/gloo与GPU/NCCL各自验证范围。
- [ ] 验证A1与baseline回归，避免为性能破坏正确性。
- [ ] 完成根目录 `test_and_make_submission.sh` 的TODO：严格传播测试失败退出码后再打包。

## 4. 打包代码

- [ ] 复用已有 `scripts/package_submission.py` 的allowlist，不重新实现ZIP打包器。
- [ ] 当前打包器输出 `dist/cs336-2026-assignment2.zip`；核对内容后按官方要求命名/复制为 `code.zip`。
- [ ] 包含 src、测试/fixtures、运行配置、必要脚本和报告源；排除虚拟环境、大数据、缓存和凭据。
- [ ] 在干净环境解压后确认导入、目标测试及入口可运行，不依赖本机未打包绝对路径。
- [ ] 注意本仓库使用Python3.14/PyTorch2.14；与官方环境不完全一致，提交前明确目标环境的安装方式。
- [ ] `code.zip`不能只是当前stub的归档。报告与实现的commit必须能对应。

## 5. Leaderboard / Gradescope

- [ ] leaderboard遵循讲义精确模型、两张B200、完整训练步与空缓存预算；不能提交baseline预演成绩冒充正式结果。
- [ ] 在使用的讲义版本对应的官方入口核对提交说明，再手动上传；不在stub中硬编码账户或自动发消息。
- [ ] 保留提交文件的哈希、时间和平台回执。现在不填写“已提交”。
