# CoTA-LoRA 接手审查（2026-11-22）

对交接快照 `043b6fc` 的独立审查，以及为下一轮实验准备的对照、协议和代码。**本次没有运行任何模型实验**：审查环境没有 PyTorch、权重、数据和 GPU。

| 文件 | 内容 |
|---|---|
| [DIAGNOSIS.md](DIAGNOSIS.md) | 带文件行号的诊断，按"已确认 / 有日志支持 / 待检验"分级 |
| [PROTOCOL.md](PROTOCOL.md) | 预先声明的数据角色、实验臂、统计方法、判定规则、预算和命令 |
| [ACL_ASSESSMENT.md](ACL_ASSESSMENT.md) | 相关工作核对、对四个研究问题的回答、是否具备投稿贡献的判断 |
| [CHANGES.md](CHANGES.md) | 代码改动、哪些已验证哪些没有、与导出清单的差异 |
| [checks/](checks/) | 只依赖 NumPy 的复算脚本、结果和测试（已运行通过） |
| [../../experiments/cota_review/](../../experiments/cota_review/) | 配置生成器、实验矩阵和生成的配置 |

最短结论：现有实验既没有证明也没有否定 CoTA。条件门在数值上没有起作用，选模指标和目标集都没有足够的分辨力，而决定方法是否成立的那个对照（同位置、同参数、不同输入）还没有做过。按 PROTOCOL 约一天单卡可以得到明确答案。
