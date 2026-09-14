# V14 外部依赖执行清单

这三项必须由真实的人完成。脚本会核对文件、版本和哈希，但不能替代签署人、
两名人工编码者或独立复现者的身份与判断。

## 1. 获取开放 FACED 数据并交给自动管线

AMIGOS 方案已在未访问任何试次标签或 EEG 的情况下，因 EULA 和下载端点的可访问性
问题被 FACED `nm000112` v1.1.3 替代。修订记录见
`docs/confirmatory_dataset_amendment_v14.md`，不得删除原 AMIGOS 方案的审计记录。

FACED 使用 CC-BY-4.0，固定版本入口为：
`https://data.nemar.org/nm000112/v1.1.3/`。优先使用 NEMAR CLI 的按受试者、可续传下载，
避免一次下载 22.7 GB 压缩包；下载后的原始 BDF、事件表和个体标签放在仓库外。

```bash
npm install -g nemar-cli
nemar dataset download nm000112 --subjects sub-000,sub-001
```

先用少量受试者验证摄取程序，再扩展到固定版本的全部受试者。正式 QC 必须先生成并
锁定不包含评分值的结构报告，之后才允许解封 Valence/Arousal。若结构 QC 后少于 25
名参与者完整保留 28 个视频，则本确认数据集失败，不得看完结果再换另一个数据集。

以下旧 AMIGOS 步骤仅作为历史审计记录保留，不再执行。

1. 在 AMIGOS 官方页面下载 EULA：
   `https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/eula.pdf`。
2. 由符合数据方条件的申请人打印、签署并通过官方页面提交。若学生不符合申请
   人资格，应由具备永久教职或相应正式职位的负责人申请；使用单位邮箱。不要把
   密码、下载令牌或签署文件提交到 Git 仓库或聊天。
3. 获批后，将 `Data_Preprocessed_P01.mat` 至 `P40.mat` 放在仓库外的受控目录。
   不要先打开 `labels_selfassessment`、画标签分布或试跑模型。
4. 先运行仅检查结构和信号尺度的命令：

   ```bash
   python scripts/qc_amigos_structure_v14.py \
     /secure/AMIGOS/Data_Preprocessed \
     /secure/AMIGOS/amigos_structural_qc_v14.json
   ```

5. 保存 QC JSON 和 SHA-256。若状态为 `pass`，再运行解封和摄取；若不是 `pass`，
   先记录带日期的方案偏离，不得根据标签或模型结果修改规则。

   ```bash
   python scripts/ingest_amigos_v14.py \
     /secure/AMIGOS/Data_Preprocessed \
     /secure/AMIGOS/amigos_structural_qc_v14.json \
     /data/OpenAffect-EEG/derived/confirmatory_amigos_v14_inputs
   ```

随后由负责人运行 100 个拆分的确定性 Ridge、EEGNet 五种子和 LaBraM-PEFT 五种子
管线。原始 MAT、trial 级标签、EEG 数组和个体预测均不得进入公开匿名仓库。

## 2. 两名人工文献编码者

1. 先用机构图书馆或作者公开存档取得 `paper_set.csv` 中 24 篇主样本的同一版本
   全文。连续两个不同日期仍不能合法取得时，才按冻结的同年份 reserve 顺序替换。
2. 给编码者 A 和 B 相同的以下材料：
   `docs/literature_resource_coding_manual_v14.md`、24 篇全文、对应的空白 CSV，以及
   自动生成的检索证据包。明确说明证据包只是定位帮助，不能代替阅读全文。
3. 两人不得讨论单篇判断，也不得查看对方 CSV。每个非 `unclear` 判断都必须写页码、
   章节、表格、图或代码位置。
4. 两人分别完成后，先计算并保存 SHA-256，再交换文件。PowerShell 示例：

   ```powershell
   Get-FileHash .\coder_A_paper.csv -Algorithm SHA256
   Get-FileHash .\coder_A_evaluations.csv -Algorithm SHA256
   ```

5. 锁定后才计算原始一致率和 Cohen's kappa，并逐项裁决分歧。论文必须如实披露
   编码者是否为作者；自动代理不能算第二名人工编码者。

## 3. 独立 artifact 复现

1. 先冻结 V14 论文结果、匿名代码快照和最终提交提交号，再让复现者开始。用旧 V13
   或作者 GitHub 地址跑通，不能证明提交时的匿名 V14 artifact 可执行。
2. 复现者必须未参与开发，使用未用于本项目的电脑、虚拟机或新系统账户，只收到
   匿名仓库链接和 `docs/independent_acceptance_task_zh.md`，不接受逐步口头指导。
3. 复现者私下交回终端全文、`independent_acceptance_report.json` 和填好的
   `independent_acceptance_form_v14.json`；首次失败也必须保留。
4. 负责人运行 `scripts/package_independent_acceptance_v14.py`。只有验证结果同时为
   `packet_status=valid`、`acceptance_type=independent_user` 和
   `independent_user_claim_allowed=true`，论文才可写独立用户复现通过。

## 4. 下一次投稿前的匿名与时效检查

NeurIPS 2026 E&D 已于 2026-05-06 截止，当前工作只能面向下一次合适的 E&D venue。
V14 最终代码应更新到匿名托管入口并从无登录环境重新验证；论文不能链接到暴露作者
账号的 GitHub 地址。下一届规则公布后，重新核对模板、页数、匿名模式、代码字段、
数据与 Croissant 要求，不能直接假设 2026 规则原样延续。
