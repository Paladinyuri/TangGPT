# TangGPT

一个从零搭建的 Decoder-only Transformer，用于中文古诗生成。

当前版本包含模型、全唐诗数据处理、从零实现的 byte-level BPE tokenizer、
单卡训练、checkpoint、断点续训和文本生成。

## 当前结构

- Token embedding
- RMSNorm
- Rotary Position Embedding（RoPE）
- Multi-head causal self-attention
- SwiGLU feed-forward network
- Pre-Norm residual decoder blocks
- Tied language-model head
- Byte-level BPE tokenizer（训练、编码、解码、保存、加载）
- 全唐诗清洗、去重、体裁识别和稳定切分
- PyTorch Dataset、动态 padding 与 next-token labels
- AdamW、warmup + cosine decay、梯度累积和梯度裁剪
- CUDA 混合精度、验证集评估、checkpoint 与断点续训
- temperature + top-k 自回归生成

## 阅读顺序

建议按下面的顺序阅读 `src/tanggpt/model.py`：

1. `TangGPTConfig`
2. `RMSNorm`
3. `RotaryEmbedding`
4. `CausalSelfAttention`
5. `SwiGLU`
6. `DecoderBlock`
7. `TangGPT`

每读完一个模块，先在纸上写出输入和输出张量形状，再运行测试验证。

## 运行

在项目根目录执行：

```powershell
$env:PYTHONPATH = "src"
python scripts/inspect_model.py
python -m unittest discover -s tests -v
```

## 数据与 tokenizer

数据来自 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，
采用 MIT 许可证。原始语料默认位于 `data/raw/chinese-poetry/全唐诗`。

新服务器先下载数据：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/chinese-poetry/chinese-poetry.git \
  data/raw/chinese-poetry
git -C data/raw/chinese-poetry sparse-checkout set "全唐诗"
```

然后准备语料。仓库已包含正式的 `artifacts/tokenizer.json`，通常不需要在
服务器重新训练 tokenizer：

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_data.py
```

该命令执行清洗、正文去重和 98%/1%/1% 切分。若希望从头复现 BPE 训练，再运行：

```bash
python scripts/train_tokenizer.py --vocab-size 4096
```

`data/` 和训练 checkpoint 不会提交 Git，来源记录在 `data/SOURCE.md`。

## 本地端到端检查

正式租服务器前，先确认完整流程能够运行：

```powershell
$env:PYTHONPATH = "src"
python scripts/train.py `
  --config configs/smoke.json `
  --run-dir runs/smoke `
  --train-limit 32 `
  --valid-limit 16
```

## 单张 RTX 4090 训练

Linux 服务器中执行：

```bash
export PYTHONPATH=src
python scripts/train.py --config configs/server_4090.json
```

默认配置是 8,000 个 optimizer steps、batch size 64、梯度累积2次，等效
batch size 为128，约遍历训练集21次。训练中会生成：

```text
runs/server_4090/
├── config.json
├── tokenizer.json
├── train.jsonl
├── samples.txt
├── best.pt
├── last.pt
└── checkpoints/
```

从最近 checkpoint 接着训练：

```bash
python scripts/train.py \
  --config configs/server_4090.json \
  --resume runs/server_4090/last.pt
```

`last.pt` 用于恢复训练；`best.pt` 是验证 loss 最低的模型，用于最终生成。

本机 RTX 5060 Laptop 8GB 使用专用配置：

```powershell
$env:PYTHONPATH = "src"
python scripts/train.py --config configs/local_5060_8gb.json
```

该配置使用 BF16、batch size 16 和梯度累积8次，等效 batch size 仍为128。

如果发生 CUDA out of memory，先把 `configs/server_4090.json` 中的
`batch_size` 从64改为32，并把 `gradient_accumulation_steps` 从2改为4，
等效 batch size 仍为128。

## 生成唐诗

```bash
python scripts/generate.py \
  --checkpoint runs/server_4090/best.pt \
  --form 5jue \
  --title 秋夜 \
  --author 佚名 \
  --temperature 0.8 \
  --top-k 40 \
  --num-samples 5
```

结果会显示在终端，并写入 `outputs/generated_poems.txt`。

默认配置约为 12.1M 参数模型；测试使用的是非常小的配置，因此可以在 CPU 上
快速完成。

## 下一阶段

1. 整理和清洗唐诗数据
2. 实现字符级 tokenizer
3. 构造 next-token prediction 样本
4. 先让极小模型过拟合少量诗歌
5. 再编写完整训练循环和验证流程
