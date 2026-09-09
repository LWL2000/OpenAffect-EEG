# OpenAffect-EEG 独立复现与验收执行文档

## 1. 任务目标

请在一台未用于本项目开发的电脑、虚拟机或新系统账户中，仅根据本文档完成安装和验收。任务验证的是：

- 公开仓库可以从零安装；
- 发布清单中的文件完整且哈希一致；
- Python 依赖不存在已知冲突；
- 合成数据上的审计流程可以执行；
- 论文使用的两份聚合证据摘要可以从冻结来源重建。

这不是原始 EEG 模型的全量重新训练。公开仓库不包含原始 EEG、刺激材料、个体级预测、特征数组或模型权重，因此本任务不能被描述成“独立重复了论文全部实证结果”。

## 2. 验收类型

- **独立用户验收**：执行者没有参与本项目开发，只收到仓库链接和本文档，过程中没有获得作者逐步指导。
- **独立机器验收**：执行者参与过开发，或作者本人换一台电脑运行。
- **作者辅助验收**：执行过程中作者解释了具体命令、修改了环境或提供了修复方案。

三种结果都有诊断价值，但论文中必须按真实类型报告，不能混称。

## 3. 执行前要求

- Git；
- CPython 3.11 或 3.12；
- 稳定网络；
- 至少 2 GB 可用磁盘；
- 不需要 CUDA、GPU、Conda、原始 EEG 或任何受限数据。

请新建目录，不要在已有项目副本或作者提供的虚拟环境中运行。开始计时后，保留完整终端输出，包括第一次失败。

## 4. Windows PowerShell

将下面 `$RepoUrl` 的内容替换为负责人发送的 HTTPS 仓库地址，然后逐行执行：

```powershell
$RepoUrl = "把仓库的 HTTPS 地址粘贴到这里"
$RunDir = "OpenAffect-EEG-acceptance-01"

New-Item -ItemType Directory -Path $RunDir
Set-Location $RunDir
Start-Transcript -Path .\openaffect_acceptance_terminal.txt

Write-Host "START_UTC=$((Get-Date).ToUniversalTime().ToString('o'))"
git --version
py -3.12 --version
git clone --depth 1 $RepoUrl artifact
Set-Location artifact
git rev-parse HEAD
git status --short

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check ".[audit]"
.\.venv\Scripts\python.exe scripts\verify_review_artifact.py `
  --project-root . `
  --output acceptance-run

$VerifyExit = $LASTEXITCODE
Write-Host "VERIFY_EXIT_CODE=$VerifyExit"
Get-Content .\acceptance-run\independent_acceptance_report.json
git diff --exit-code
Write-Host "END_UTC=$((Get-Date).ToUniversalTime().ToString('o'))"
Stop-Transcript
```

如果没有 Python 3.12，但安装了 3.11，将所有 `py -3.12` 改为 `py -3.11`。这里直接调用虚拟环境中的 Python，不需要执行激活脚本，也不需要修改 PowerShell 执行策略。

## 5. Linux 或 macOS

将 `REPO_URL` 替换为负责人发送的 HTTPS 仓库地址。在新终端中执行：

```bash
REPO_URL="把仓库的 HTTPS 地址粘贴到这里"
RUN_DIR="OpenAffect-EEG-acceptance-01"

mkdir "$RUN_DIR"
cd "$RUN_DIR"
script -q openaffect_acceptance_terminal.txt

echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git --version
python3.12 --version
git clone --depth 1 "$REPO_URL" artifact
cd artifact
git rev-parse HEAD
git status --short

python3.12 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check ".[audit]"
.venv/bin/python scripts/verify_review_artifact.py \
  --project-root . \
  --output acceptance-run

VERIFY_EXIT=$?
echo "VERIFY_EXIT_CODE=$VERIFY_EXIT"
cat acceptance-run/independent_acceptance_report.json
git diff --exit-code
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit
```

如果只有 Python 3.11，将全部 `python3.12` 改为 `python3.11`。Ubuntu/Debian 若提示缺少 `venv`，请保留原始错误；安装相应系统包后，必须在验收备注中写明该操作。

## 6. 成功判据

终端应出现 `VERIFY_EXIT_CODE=0`。同时，`artifact/acceptance-run/independent_acceptance_report.json` 必须满足：

- 顶层 `status` 为 `pass`；
- `supported_python` 为 `pass`；
- `release_manifest` 为 `pass`；
- `dependency_consistency` 为 `pass`；
- `synthetic_toy_audit` 为 `pass`；
- `source_linked_evidence_rebuild` 为 `pass`。

任意一项失败都应按失败记录。不要手工修改仓库文件来获得 `pass`。

## 7. 必须交回的材料

请将以下材料私下发送给项目负责人，不要提交到公开 Issue：

1. `artifact/acceptance-run/independent_acceptance_report.json`；
2. `openaffect_acceptance_terminal.txt`；
3. 填写后的 `docs/independent_acceptance_form.md` 副本，或下面的等价中文记录；
4. 第一次运行失败时的报告和终端记录，即使后续已经修复成功。

中文记录至少填写：

```text
匿名验收者编号：
验收日期与时区：
是否参与过项目开发：是/否
是否获得了本文档之外的操作指导：是/否
操作系统与版本：
CPU 架构：
Python 版本：
git rev-parse HEAD 输出：
报告中的 release version：
报告中的 release-manifest SHA-256：
最终 VERIFY_EXIT_CODE：
最终 status：pass/fail
从打开文档到首次得到结果的分钟数：
不清楚的命令：
首次失败及完整错误：
为成功运行做过的修改或获得的帮助：
是否从仓库内容发现作者、单位、账号、邮箱或私有服务器信息：是/否
其他意见：
```

终端日志中的本机用户名、主机名和个人目录可以替换成 `[REDACTED]`。不要修改命令、错误文本、版本号、时间、Git 提交号、状态或哈希。

## 8. 失败处理

- `Output directory must be absent or empty`：不要删除首次记录；把输出目录改为 `acceptance-run-02` 后重试。
- `Python 3.11 or 3.12 is required`：改用支持的 CPython 版本，记录原版本和更换过程。
- `Release manifest mismatch`：停止修改文件，保存报告和日志并联系负责人。
- 下载或 `pip` 网络失败：保留错误；网络恢复后在新终端记录中重试。
- 其他失败：不要凭经验改源码。先发送完整报告、终端日志和出错命令。

首次失败本身是评估文档可用性的重要证据，不能只保留最终成功记录。截图可以补充，但不能替代 JSON、终端日志和验收表。

## 9. 结果解释边界

通过本任务可以声明：公开审计工具在该操作系统和 Python 版本上能够从零安装、执行，并重建指定聚合证据。

不能仅凭本任务声明：原始 EEG 已被第三方重新下载、所有模型已重新训练、论文全部数值获得独立复制，或论文的科学结论已被外部研究者确认。
