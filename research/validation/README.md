# 导出验证

`cpu-tests.log` 是从本次导出目录运行的定向 CPU 测试原始输出。

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  /root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python -m pytest -q -rs \
  experiments/ms_swift_aqa/tests experiments/ms_swift_librispeech/tests experiments/ms_swift_iemocap/tests
```

使用原服务器已有实验环境；未安装依赖、未运行 GPU 训练。该日志不能证明在新机器安装成功。

## 本次实际状态

- 合并运行三个测试目录：退出码 137，进程被系统 SIGKILL，未完成；原始部分输出保留在 `cpu-tests.log`。
- 终止后检查到执行容器 cgroup `memory.max=2147483648`（2 GiB）、`memory.events max=14484`，但 `oom=0`、`oom_kill=0`。资源限制是线索，尚无证据将 SIGKILL 明确归因为 OOM，更不能认定为代码断言失败。
- 随后独立运行 CoTA 条件门定向测试：**13 passed**，原始日志 `conditional-gate-tests.log`。命令将上方线程数设为1，测试路径为 `experiments/ms_swift_librispeech/tests/test_conditional_gate.py`。
- 历史 186 passed / 4 skipped 属于先前服务器运行，只保留为历史证据，不写成本次复跑成绩。
- 694份实验导出及其原来源哈希复核通过；导航、28条论文索引和总导出清单另由文件校验核对。
