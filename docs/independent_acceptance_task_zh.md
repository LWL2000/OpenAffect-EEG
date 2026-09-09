# OpenAffect-EEG 独立电脑完整验收任务

## 验收目的

这项任务验证匿名代码包能否在另一台电脑上仅凭公开文档完成安装、运行和证据重建。它不下载原始 EEG，不重新训练模型，也不要求 GPU。

如果执行者本人参与过本项目开发，结果只能称为“独立机器验收”；如果执行者没有参与开发，并且只阅读 `docs/REVIEWER_QUICKSTART.md`，才可以称为“独立用户验收”。

## 执行前准备

- 使用一台没有本项目开发环境的新电脑、虚拟机或新系统账户。
- 安装 Git，以及 Python 3.11 或 3.12。
- 至少预留 2 GB 磁盘空间和稳定网络。
- 不需要 CUDA、GPU、原始 EEG、刺激视频、特征或模型权重。
- 验收者不要提前阅读内部开发文档，也不要接受作者口头操作指导。

## Windows PowerShell 步骤

将匿名仓库下载或克隆到新目录后，在仓库根目录执行：

```powershell
Start-Transcript -Path ..\openaffect_acceptance_terminal.txt
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[audit]"
.\.venv\Scripts\python.exe scripts\verify_review_artifact.py `
  --project-root . `
  --output acceptance-run
Stop-Transcript
```

如果电脑只安装了 Python 3.11，将第一条命令改为 `py -3.11 -m venv .venv`。

## Linux 或 macOS 步骤

```bash
script ../openaffect_acceptance_terminal.txt
python3.12 -m venv .venv
.venv/bin/python -m pip install ".[audit]"
.venv/bin/python scripts/verify_review_artifact.py \
  --project-root . \
  --output acceptance-run
exit
```

如果只安装了 Python 3.11，将 `python3.12` 改为 `python3.11`。

## 成功标准

终端命令退出码为 0，且 `acceptance-run/independent_acceptance_report.json` 中满足：

- 顶层 `status` 为 `pass`；
- `supported_python` 为 `pass`；
- `release_manifest` 为 `pass`；
- `dependency_consistency` 为 `pass`；
- `synthetic_toy_audit` 为 `pass`；
- `source_linked_evidence_rebuild` 为 `pass`。

## 需要记录并交回的材料

请把以下三个文件私下交给项目负责人：

1. `acceptance-run/independent_acceptance_report.json`：机器自动生成的核心验收结果。
2. `openaffect_acceptance_terminal.txt`：完整终端记录，包括首次失败，不要只保留最后成功部分。
3. `docs/independent_acceptance_form.md` 的填写副本：记录是否参与开发、系统与 Python 版本、耗时、文档不清楚之处、失败及所需帮助。

额外需要告诉项目负责人：

- 使用的匿名 URL；
- 下载日期和时区；
- 从打开文档到第一次得到结果的大致分钟数；
- 是否在不搜索作者信息的情况下从包内看到了姓名、单位、账号、邮箱或服务器信息；
- 如果失败，原始错误文本、发生在哪条命令、是否请求过帮助以及帮助内容；
- 修复后重新运行时，新旧两个报告对应的 release-manifest SHA-256。

终端日志可能包含本机用户名或路径。交回前可以把用户名替换为 `[REDACTED]`，但不要修改命令输出、错误信息、版本号、时间或哈希。

## 不应填写的内容

- 不需要验收者姓名、学校、邮箱、GitHub 账号或电脑主机名。
- 不需要截图替代 JSON 和终端日志。
- 不要把原始 EEG、特征、刺激或任何受限数据放进验收材料。
- 首次失败不能删除；它是判断文档和安装流程是否真实可用的重要证据。
