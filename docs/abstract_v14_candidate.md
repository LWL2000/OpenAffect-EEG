# OpenAffect-EEG v14 abstract candidate

## English abstract

Apparent gains from electroencephalography (EEG) cannot be attributed to EEG
when an EEG system and its no-EEG comparator receive different non-EEG labels.
OpenAffect-EEG is an executable protocol that separately doses
target-participant calibration and repeated-stimulus labels, fixes test trials
and population training, and estimates a paired EEG increment against a
same-resource comparator. After development on EmoEEG-MC and urban-image
appraisal, we evaluated the protocol on a previously unused FACED cohort (123
participants, 3,444 trials, 28 stimuli) using a complete 5-by-5 resource grid,
five training seeds, EEGNet, and final-block-adapted LaBraM. Joint bootstrap
intervals resampled participants, stimuli, training seeds, and fold rotations.
Equivalence margins were fixed before FACED label access. LaBraM's mean matched
concordance correlation coefficient increment was -0.0053 (95% CI, -0.0112 to
0.0009, 90% CI, -0.0103 to -0.0001), establishing equivalence at both +/-0.05
and +/-0.025. EEGNet's estimate was -0.0414 (95% CI, -0.0875 to 0.0002) and
remained inconclusive. A full-grid LaBraM label-permutation control produced no
spurious positive gain. Two locked EEGNet sinusoidal positive controls failed. A
redesign frozen before restart recovered increments of +0.6512 with EEGNet and
+0.6303 with LaBraM. These post-failure diagnostics show pipeline sensitivity
but remain exploratory. Matching non-EEG labels therefore changes the
attribution of system-level gains. Under the stated estimand, adapted LaBraM's
FACED increment was practically negligible, whereas EEGNet remained uncertain.
This model- and setting-specific result does not imply that EEG is universally
uninformative. We release executable resource contracts, paired estimands, and
evidence hashes for auditing EEG added value.

**Keywords:** paired estimands; practical equivalence; affective computing; EEG
foundation models; user calibration; reproducible evaluation

## 中文摘要

当EEG系统与无EEG基线获得不同的非EEG标签资源时，两者的性能差不能直接归因于EEG。OpenAffect-EEG将目标参与者校准标签和重复刺激标签分别设为资源轴，在固定测试试次与总体训练规模的条件下，以获得相同标签的无EEG预测器构造配对增量。我们先在两个情感预测队列中开发该协议，再在此前未用于本项目的FACED队列上检验主要结论。FACED分析包含123名参与者、3,444个试次、28个刺激、完整的5×5资源网格、5个训练种子，以及EEGNet和末端模块微调的LaBraM。联合bootstrap同时纳入参与者、刺激、训练种子和折叠轮换的不确定性，等价界限在访问FACED标签前固定。LaBraM的平均匹配CCC增量为−0.0053，95%置信区间为[−0.0112, 0.0009]，在±0.05和±0.025界限下均达到实际等价；EEGNet的估计为−0.0414，95%置信区间为[−0.0875, 0.0002]，结论仍不确定。LaBraM标签置换控制未产生虚假正增益。两个锁定的EEGNet正弦正控制均失败；随后设计并在重新运行前冻结的探索性控制，在EEGNet和LaBraM上分别恢复+0.6512和+0.6303的增量，表明管线能够检测明确的可学习信号。结果说明，标签资源匹配会改变系统增益的归因；在既定统计量下，FACED中适配LaBraM的EEG增量实际可忽略，但该结论不能推广为EEG普遍无信息。

**关键词：** 配对统计量；实际等价；情感计算；脑电基础模型；用户校准；可复现评估
