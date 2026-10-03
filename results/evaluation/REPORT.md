# TangGPT 补充评估

日期：2026-10-03。对象：实验 C 的验证集最佳 checkpoint（step 3,250，29.37M 参数）。本次没有重新训练或根据测试集调整模型。

## 测试集

| 项目 | 结果 |
|---|---:|
| 测试诗歌数 | 515 |
| 有效预测 token 数 | 34,590 |
| 被截断诗歌数 | 0 |
| Token 加权交叉熵 | 3.7461 |
| 困惑度（exp(loss)） | 42.36 |

CUDA FP32 推理，PyTorch 2.14.0+cu130。损失包含正文、题目、作者与控制标记，是完整序列 next-token prediction 指标，不能单独解释为正文质量。历史验证 loss 是 batch 均值，本次按有效 token 数加权，不能把两者差值当作精确泛化差距。困惑度不适合直接跨 tokenizer 比较。

## 固定条件生成

标题依次为秋夜、春江、山行、雨后、送别、月下、归乡、江上、冬雪、闲居。每种体裁各生成 10 首，作者条件为“佚名”；temperature=0.8，top-k=40，最多生成 192 token，起始种子 20261003，每个样例递增 1。

| 体裁 | 样本数 | 句数/字数合规 | EOS 正常结束 | 出现重复整句 |
|---|---:|---:|---:|---:|
| 五绝 | 10 | 10/10 | 10/10 | 0/10 |
| 七绝 | 10 | 10/10 | 10/10 | 0/10 |
| 五律 | 10 | 10/10 | 10/10 | 0/10 |
| 七律 | 10 | 10/10 | 10/10 | 0/10 |

这是固定提示下的样本观察，不代表总体成功率，更不等于完整格律合格率。全部样例见 [samples.md](samples.md)，结构化样例见 [samples.jsonl](samples.jsonl)，配置与指标见 [metrics.json](metrics.json)。

## 人工观察与失败样例

“秋夜”样例：

> 白露墜深樹，秋風吹更香。
> 天清今夜月，一半是愁腸。

该样例满足五绝句数和字数，有秋夜与愁绪意象，但未验证押韵和平仄。

“月下”首句“明月照明月”出现词组重复；“冬雪”含“花迎臘臘梅”“不因逢雪雪”，字数正确但表达生硬。“归乡”含“縱有心相心，何其入口人”，语义连贯性较差。整句重复率不能捕捉这些问题。

“春江”样例主要描述舟行、旅宿，标题条件遵循程度尚未系统评分。题目作为输入条件，不保证每首诗都紧扣题意。

## 数据审计与记忆检查

训练/验证/测试集分别有 47,677 / 522 / 515 条记录。移除标点和控制标记、只保留 CJK 字符后，三个集合两两之间的正文完全重复数均为 0；训练集内部有 30 条规范化后重复记录。审计不转换简繁体，也没有排查全部异文和近似重复，不能宣称完全排除了数据泄漏。

40 首生成诗歌与训练集去标点正文完全匹配数为 0。使用二元字符组集合 Jaccard 检索，每首保存三个最相似训练候选，最高分为 0.0952；“北鴈又南飛”是生成内容与训练文本共享的句子。整首完全匹配检查不能排除句子复用或近似复述，Jaccard 也不是语义指标。候选与数据 SHA-256 见 [data_audit.json](data_audit.json)。

## 复现

在 TangGPT 根目录安装项目，准备原始数据并运行 `scripts/prepare_data.py`。使用 checkpoint 对应的 tokenizer，核对 metrics.json 中哈希。

```bash
python scripts/evaluate_checkpoint.py --checkpoint runs/experiment_c_large/best.pt --tokenizer artifacts/tokenizer.json --output-dir results/evaluation --samples-per-form 10
python scripts/audit_data.py --samples results/evaluation/samples.jsonl --output results/evaluation/data_audit.json
```

本次 checkpoint 从已有的本机模型包读取，路径记录在 metrics.json，权重未上传仓库。不同 PyTorch/CUDA 版本可能导致采样差异。

## 可用于简历的表述

“完成 12.19M 与 29.37M 参数 Decoder-only 唐诗语言模型的三组训练实验，分析 Dropout、模型规模与过拟合；对验证集最优的 29.37M 模型进行独立测试集评估，token 加权交叉熵为 3.7461，并整理固定条件生成、格式检查和失败案例。”

个人贡献应按实际编写、修改、调试与分析的内容调整。未评价诗歌艺术质量、未验证多随机种子，也未补测 A/B 的测试集指标。
