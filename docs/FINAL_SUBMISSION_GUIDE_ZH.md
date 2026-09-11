# OpenAffect-EEG 最终投稿指南（NeurIPS 2026 E&D）

更新日期：2026-09-11

## 1. 先核对投稿前提

NeurIPS 2026 Evaluations & Datasets Track 的全文截止时间为 2026-05-06 AoE，当前已经过期。因此：

- 若此前没有创建有效的 OpenReview 投稿，当前新投 NeurIPS 2026 的行政可行性为 **0%**；论文质量不能改变截止日期。
- 若已有有效投稿记录，应在 OpenReview 允许的修改范围内更新 PDF、补充材料和匿名 artifact URL，并遵守 AC/PC 的具体通知。
- 若没有有效投稿记录，本目录是下一届或其他适配 venue 的完整候选投稿包，不能表述为“已正式提交 NeurIPS 2026”。

官方要求与时间以 [NeurIPS 2026 E&D Call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets) 和 OpenReview 投稿页为准。

## 2. 论文的可辩护贡献

本文最稳妥的创新表述是：在固定测试支持和固定总体规模下，同时控制“同一参与者的标签校准剂量”和“来自其他参与者的重复测试刺激标签剂量”，用配对 EEG/无 EEG 预测的 CCC 差值构造 5×5 资源响应面，并提供可执行的数据、模型和评估契约。

以下组成部分本身不是新方法，不应单独声称首创：身份/试次留出、校准、bootstrap、热图、身份探针、数据卡与哈希、望远镜分解恒等式。文章的差异来自这些控制被组合成一个资源匹配的评估设计，并在两个真实 EEG 情感任务上执行。

最接近的工作包括 Pandilova 等人的个体标签倾向与 qEEG 校准研究、EEG-FM-Audit 和 Identity Trap/FMScope。当前检索没有发现与本文完整联合干预设计相同的已发表工作，但检索不能证明“文献中绝对没有”。因此正文采用 “to our knowledge” 和可逐项比较的范围陈述，不使用无条件 “first”。

## 3. 结论边界

论文支持的结论是：在本文两个任务、预先给定资源网格和所考察模型内，匹配后的 EEG 增量很小，主汇总的五身份块区间均包含零。

论文不支持以下更强结论：

- EEG 对情感预测普遍没有价值；
- 区间包含零等价于证明效应为零；
- 热图每个单元都有已校准的 95% 覆盖；
- 结果可无条件外推到临床、自然场景、所有人群或所有 foundation model；
- 固定预测上的 bootstrap 包含重新训练和超参数选择的不确定性。

主推断使用五身份块 block-t 区间。四类拓扑匹配高斯模拟中，预先声明的均匀网格均值覆盖率为 0.980–1.000；最差单元覆盖率只有 0.917，因此单元区间和热图范围只作描述。该校正是在早期诊断发现原区间覆盖不足后采用，稿件已明确披露这一点。

## 4. 提交文件

主文件：

- `paper/neurips2026/main.pdf`：匿名英文论文与附录。
- `croissant.json`：机器可读的数据集元数据。
- 匿名 artifact：`https://anonymous.4open.science/#!/r/OpenAffect-EEG-1815/`。
- `docs/REVIEWER_QUICKSTART.md`：审稿人最短执行路径。
- `OpenAffect-EEG-anonymous-submission-v13.tar.gz`：确定性匿名代码归档（最终构建后生成）。

作者自用、不应作为匿名投稿附件：

- `paper/chinese_reading/OpenAffect-EEG-zh.pdf`；
- 本指南中的行政判断和概率预测；
- 任何含作者姓名、GitHub 账号、服务器路径或非匿名提交历史的文件。

## 5. OpenReview 填写顺序

1. 确认投稿记录仍有效，并读取页面上当前允许修改的字段。
2. 上传最终 `main.pdf`，检查标题、摘要和 PDF 内文一致。
3. 在代码/数据字段填写匿名 artifact URL，不填写实名 GitHub URL。
4. 数据集论文必须提交 Croissant 元数据时，上传或链接仓库根目录的 `croissant.json`。
5. 按实际情况回答 checklist；本文不能声称完整记录了所有历史和失败训练的能耗或 wall time。
6. 下载 OpenReview 最终保存的 PDF，复核页数、字体、匿名性、链接和补充页。
7. 保存 submission ID、修改时间和 OpenReview 回执。只有出现有效 submission ID/confirmation 才能称为已经提交。

## 6. 最终核验命令

在干净的 Python 3.11 或 3.12 环境执行：

```bash
python -m pip install ".[audit]"
python scripts/verify_review_artifact.py --project-root . --output acceptance-run
python scripts/validate_croissant.py croissant.json --require-publishable-url
python -m pytest tests -q
python scripts/build_neurips_submission.py --require-final-figures
```

这些检查验证可执行契约、发布清单、统计证据哈希、Croissant 结构、测试和 PDF 构建；它们不构成真实数据的独立复现，也不替代审稿人的科学判断。

## 7. 概率判断的口径

下表是基于当前稿件、E&D 常见筛选点和本文实证边界的主观预测，不是可验证事实，也不是官方录用率。区间反映审稿人匹配、当年竞争强度、领域偏好和讨论阶段的不确定性。

| 情形 | 送外审概率 | 最终录用概率 |
|---|---:|---:|
| 现在新建 NeurIPS 2026 投稿 | 0% | 0% |
| 假设按时提交 V13 最终材料 | 93%–98% | 30%–45% |
| 已有有效投稿且允许更新为 V13 | 取决于当前阶段；通过格式/材料检查约 93%–98% | 30%–45% |

主要加分项是问题定义清楚、固定支持的配对设计、负结果的价值、两个真实任务、可执行 artifact 和对区间失败的透明修正。主要减分项是任务数量有限、没有独立新队列、统计单元仅五个身份块、全网格单元没有校准、微调覆盖有限，以及贡献更接近评估协议和实证发现而非新的学习算法。

V13 的稳健性检查降低了三个具体风险：替代身份分块的 12/12 个区间仍跨零，三组 EEGNet 初始化的 12/12 个区间仍跨零，受限 LaBraM 最后一层微调在两任务的区间也跨零。它们没有解决独立队列、完整微调面和重训练置信区间问题，因此没有把录用预测上调到 50% 以上。

投稿前最容易被忽略的变量是：审稿人是否认同“资源剂量”问题的重要性；E&D track 对 benchmark/community adoption 的期待；数据授权是否允许审稿人复现全部真实结果；匿名镜像与 PDF 是否处于同一版本；OpenReview 当前阶段是否还允许替换文件。这些因素对最终概率的影响大于继续微调措辞。
