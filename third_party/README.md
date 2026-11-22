# 外部依赖历史源码快照

这里保存一次历史实验源码采集中的依赖代码，供 Claude 和代码审阅者直接阅读。快照于 **2026-11-21 08:08:16–08:08:18 UTC** 采集；本次交接只按原始字节提取并核验，没有执行这些代码。

**该快照不证明所有训练、检查点或更早的运行都使用同一批源码。** 原始 `capture.json` 也明确限定了这一点。ms-swift 和两个 PEFT 版本没有记录 Git 提交，版本号不能代替提交证明；逐文件 SHA-256 才标识本目录实际保存的字节。

本目录用于浏览和来源核对。普通项目启动和 `import swift` / `import peft` 不会自动选用这里的副本：本次导出没有安装这些包、修改启动配置或向 `PYTHONPATH` 添加这些路径。两个 PEFT 目录代表不同历史环境，不应同时当作运行时依赖启用。

| 目录 | 内容与版本 | 文件数 | 字节数 | 许可依据 |
|---|---|---:|---:|---|
| [`snapshot/ms-swift`](snapshot/ms-swift) | 本地 ms-swift `4.7.0.dev0` 源码、测试、配置；可能含本地修改 | 864 | 7,152,202 | [随附 Apache-2.0](snapshot/ms-swift/LICENSE) |
| [`snapshot/peft-default`](snapshot/peft-default) | 默认 PEFT `0.15.2` 源码与包元数据 | 135 | 1,478,212 | [随附 Apache-2.0](snapshot/peft-default/peft-0.15.2.dist-info/LICENSE) |
| [`snapshot/peft-randlora`](snapshot/peft-randlora) | RandLoRA 环境使用的 PEFT `0.17.1` | 164 | 1,815,466 | [随附 Apache-2.0](snapshot/peft-randlora/peft-0.17.1.dist-info/licenses/LICENSE) |
| [`snapshot/official-mmau`](snapshot/official-mmau) | MMAU 评测脚本、上游 README/许可，以及本项目下载辅助脚本 | 4 | 24,580 | [上游 Apache-2.0](snapshot/official-mmau/LICENSE)；本项目脚本遵循项目许可 |
| [`snapshot/cpu-diagnostics`](snapshot/cpu-diagnostics) | 本项目原创的三个 CPU 验证脚本，并非论文作者代码 | 3 | 14,474 | [项目 Apache-2.0](licenses/PROJECT-APACHE-2.0.txt) |
| 合计 | 全部为 UTF-8 文本 | **1,170** | **10,484,934** | 原有 LICENSE 和源码声明保持原样 |

上游仓库为 [ms-swift](https://github.com/modelscope/ms-swift)、[PEFT](https://github.com/huggingface/peft) 和 [MMAU 固定提交](https://github.com/Sakshi113/MMAU/tree/110127f54c0dfba3faa5ec9feee4a7e4148679c5)。MMAU 的 `download.py` 是本项目固定版本下载辅助，不标作上游作者文件；运行它会下载数据，本次没有运行。`cpu-diagnostics` 也只保留源码，其中本机绝对路径仍是历史记录，迁移后需要自行配置。

源码文件相对归档未改动，版权头、许可证、内嵌来源声明与原始顺序保持不变。这里没有新增整体授权，也不覆盖任何原有文件许可。本项目自编辅助脚本的许可副本另放在 `licenses/`，避免给历史快照增加并不存在的文件。

## 授权尚未确认的论文 01 作者源码

归档中的 `paper01-reference` 含五个文件，但采集材料和已检查的本地作者缓存未附明确再分发许可，因此这里只记录来源、路径与哈希，**不复制作者源码、修改副本或 diff**。

- 作者仓库：[Expressive_Power_of_LoRA](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA/tree/eaeb7d3f265853343b234172ed44cd7623c5212a)。记录的提交是 `eaeb7d3f265853343b234172ed44cd7623c5212a`。
- 本机缓存：`/root/autodl-tmp/MMAR/runs/papers28-transfer-01/theory-01/`，包含 `official/helper.py`、`official/load_model.py`、`official/run_exp.py`、`run_tfn.py` 和 `transformer.diff`。
- 逐文件哈希见 [`provenance/origins.json`](provenance/origins.json) 与 [`provenance/excluded-files.json`](provenance/excluded-files.json)。线上页面抓取失败且 API 限流，因此此结论仅为“现有证据不足”，不声称作者仓库一定没有许可。

归档本来就没有打包 28 个方法各自的完整作者仓库。其他作者实现的出处由项目方法索引及 `experiments/ms_swift_aqa/papers28/sources-*.md` 说明；缺少作者源码副本不能解释为方法未实现。DoRA 等这里保留的是 PEFT 原生实现，不额外混入 NVlabs 作者仓库。若后续需要增加作者源码，应先记录固定提交并检查其对应授权。

## 来源与核验

原始采集目录：

```text
/root/autodl-tmp/MMAR/runs/papers28-transfer-01/sources/83075b150b88cbe603e29d7aee7aeab976fc280b9482606f906d69eb80014b5a/
```

原始 `sources.tar.gz` 的 SHA-256：

```text
8274738a6bf2a0c3fba122627b3f0cd23693025b531496b8745e9aa65f71370c
```

该压缩包未随交接仓库复制。它含未纳入本目录的主项目源码和上述论文 01 材料。

- [`provenance/manifest.json`](provenance/manifest.json)、[`environment.json`](provenance/environment.json)、[`capture.json`](provenance/capture.json) 是原始元数据，字节未改动。完整 manifest 列出的 1,474 个文件包括 299 个 `west/` 主项目文件与 5 个未复制的论文 01 文件；它不是本目录实际文件清单。
- [`provenance/extracted-files.json`](provenance/extracted-files.json) 是本次 1,170 个实际提取文件的路径、大小、权限和 SHA-256；[`extraction-receipt.json`](provenance/extraction-receipt.json) 记录核验范围、数量和来源。
- [`provenance/origins.json`](provenance/origins.json) 区分上游仓库、版本、已知提交、原始本机路径和未确认信息。
- [`provenance/SNAPSHOT_SHA256SUMS`](provenance/SNAPSHOT_SHA256SUMS) 可核验提取文件；[`METADATA_SHA256SUMS`](provenance/METADATA_SHA256SUMS) 可核验原始元数据。

在本目录执行：

```bash
sha256sum --check provenance/SNAPSHOT_SHA256SUMS
sha256sum --check provenance/METADATA_SHA256SUMS
```

提取前逐项检查了归档路径规范化、路径越界、重复成员、文件类型与全部原始哈希；只写入普通文件，没有解包链接、设备或任意路径。原归档的 1,474 项全部通过原始 manifest 校验。这里没有模型权重、实验原始数据、音频、检查点、虚拟环境、凭证或日志。ms-swift 的少量 CSV/JSONL 是上游 `11111` / `AAAAA` 等合成测试样例；JSON 配置和数据集注册元信息也属于源码包，并非实验数据。依赖源码含远程模型/数据集下载逻辑和本机路径，阅读不等于可以直接离线复现。
