"""TangGPT 单文件教学版。

推荐阅读顺序：
    1. SimpleBPETokenizer：文字如何变成整数
    2. PoetryDataset / collate_batch：整数如何组成训练 batch
    3. RMSNorm / RoPE / CausalSelfAttention / SwiGLU：一个 Block 的内部
    4. TransformerBlock / TangGPT：完整模型如何拼起来
    5. train：模型如何通过 loss 和反向传播学习
    6. generate：模型如何一个 token 一个 token 地写诗

这个文件仍然以“看清完整数据流”为第一目标，但为了能在一台 8GB 笔记本
GPU 上稳定训练约 20,000 步，保留了两项必要工程能力：

    - BF16/FP16 混合精度：降低显存占用，并提高 GPU 训练速度；
    - checkpoint/断点续训：每隔若干步保存，程序中断后不用从零开始。

相比正式版，它仍然省略了验证集循环、JSONL 日志、梯度累积、多组优化器参数、
定期样例生成等功能，因此训练主线更容易阅读。

训练：
    python learning/tanggpt_minimal.py train --steps 20000

中断后继续：
    python learning/tanggpt_minimal.py train --steps 20000 --resume runs/minimal.pt

生成：
    python learning/tanggpt_minimal.py generate --title 秋夜
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# learning/ 位于项目根目录下，因此 parents[1] 就是 tanggpt/。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# 1. Byte-level BPE tokenizer
# =============================================================================


def merge_pair(
    token_ids: tuple[int, ...], pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    """把序列中的指定相邻 token 对合并成 new_id。

    例如：
        token_ids = (1, 2, 1, 2, 3)
        pair = (1, 2)
        new_id = 100
        结果为 (100, 100, 3)
    """

    # result 用于保存合并后的新序列；index 指向当前正在检查的位置。
    result: list[int] = []
    index = 0
    while index < len(token_ids):
        if index + 1 < len(token_ids) and token_ids[index:index + 2] == pair:
            # 找到目标 pair 后，用一个新 id 代替原来的两个 id。
            result.append(new_id)
            index += 2
        else:
            # 当前两个 token 不是目标 pair，只复制当前 token。
            result.append(token_ids[index])
            index += 1
    return tuple(result)


class SimpleBPETokenizer:
    """读取项目已经训练好的 BPE，并提供 encode/decode。

    正式版 tokenizer.py 还包含训练 BPE 的优化实现；教学版只保留推理阶段，
    因为语言模型训练时最需要理解的是文字与 token id 之间的转换。
    """

    def __init__(self, tokenizer_path: Path) -> None:
        # tokenizer.json 是 BPE 训练产生的结果。语言模型训练期间只需要加载，
        # 不会再次改变词表，否则相同 id 的含义会在训练过程中发生变化。
        payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        self.special_tokens: list[str] = payload["special_tokens"]
        self.merges = [tuple(pair) for pair in payload["merges"]]

        # 最初的 0..255 分别表示一种原始 byte。
        self.id_to_bytes: dict[int, bytes] = {
            byte_value: bytes([byte_value]) for byte_value in range(256)
        }

        # 特殊 token 使用 256 之后的 id，并且永远不参与 BPE 拆分。
        self.special_to_id = {
            token: 256 + index for index, token in enumerate(self.special_tokens)
        }
        self.id_to_special = {
            token_id: token for token, token_id in self.special_to_id.items()
        }

        # 每个 merge 都产生一个新 token。新 token 的 byte 内容等于左右两项拼接。
        first_merge_id = 256 + len(self.special_tokens)
        self.merge_to_id: dict[tuple[int, int], int] = {}
        self.merge_rank: dict[tuple[int, int], int] = {}
        for rank, pair in enumerate(self.merges):
            new_id = first_merge_id + rank
            self.merge_to_id[pair] = new_id
            self.merge_rank[pair] = rank
            self.id_to_bytes[new_id] = (
                self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
            )

        # 总词表 = 256 个原始 byte + 特殊 token + BPE merge token。
        self.vocab_size = first_merge_id + len(self.merges)
        special_pattern = "(" + "|".join(
            re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
        ) + ")"
        self.special_pattern = re.compile(special_pattern)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def encode_bytes(self, raw_bytes: bytes) -> list[int]:
        """先把文本视为 byte id，再按训练好的优先级逐步合并。"""

        # Python 遍历 bytes 时，每个元素正好是 0..255 的整数，因此可以直接
        # 作为最初的 token id。例如一个汉字的 UTF-8 通常由三个 byte 组成。
        token_ids = tuple(raw_bytes)
        while len(token_ids) >= 2:
            # zip(sequence, sequence[1:]) 枚举全部相邻 pair。
            present_pairs = set(zip(token_ids, token_ids[1:]))
            known_pairs = [pair for pair in present_pairs if pair in self.merge_rank]
            if not known_pairs:
                break
            # rank 越小，代表这个 pair 在 BPE 训练中越早被学习，优先合并。
            best_pair = min(known_pairs, key=self.merge_rank.__getitem__)
            token_ids = merge_pair(token_ids, best_pair, self.merge_to_id[best_pair])
        return list(token_ids)

    def encode(self, text: str) -> list[int]:
        """将包含控制 token 的 Unicode 字符串转换为 token ids。"""

        result: list[int] = []
        # 正则把 <|bos|> 等特殊 token 与普通文字分开。特殊 token 必须整体
        # 映射到一个 id，不能被当作普通字符串拆成许多 byte。
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.special_to_id:
                result.append(self.special_to_id[part])
            else:
                result.extend(self.encode_bytes(part.encode("utf-8")))
        return result

    def decode(self, token_ids: list[int]) -> str:
        """将 token ids 还原成文字。"""

        parts: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if token_id in self.id_to_special:
                flush_bytes()
                parts.append(self.id_to_special[token_id])
            else:
                byte_buffer.extend(self.id_to_bytes[token_id])
        flush_bytes()
        return "".join(parts)


# =============================================================================
# 2. Dataset 与 batch
# =============================================================================


class PoetryDataset(Dataset[torch.Tensor]):
    """读取 train.txt，每次返回一首诗的完整 token 序列。"""

    def __init__(
        self,
        text_path: Path,
        tokenizer: SimpleBPETokenizer,
        max_seq_len: int,
        limit: int | None = None,
    ) -> None:
        self.samples: list[torch.Tensor] = []
        # 处理后的数据保证“一行一首诗”，因此这里直接按行读取。
        lines = text_path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[:limit]

        for line in lines:
            # 这里执行的是：字符串 -> BPE token id 列表。
            token_ids = tokenizer.encode(line)
            # 后面要错开一位，因此完整样本最多需要 max_seq_len + 1 个 token。
            if len(token_ids) > max_seq_len + 1:
                token_ids = token_ids[: max_seq_len + 1]
                # 截断后强制以 EOS 结尾，让模型仍能学到“应该停止”。
                token_ids[-1] = tokenizer.eos_id
            if len(token_ids) >= 2:
                self.samples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


def collate_batch(samples: list[torch.Tensor], pad_id: int) -> dict[str, torch.Tensor]:
    """把长短不一的诗歌组合成一个矩形 batch。

    假设完整 token 序列为：
        [BOS, 春, 江, EOS]

    那么：
        input_ids = [BOS, 春, 江]
        labels    = [春,  江, EOS]

    labels 中 padding 的值使用 -100，因为 cross_entropy 默认忽略 -100。
    """

    # 动态 padding：只补到当前 batch 的最长样本，而不是一律补到256，
    # 可以显著减少显存和无效计算。
    max_length = max(sample.numel() - 1 for sample in samples)
    input_ids = torch.full((len(samples), max_length), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), max_length), -100, dtype=torch.long)

    for row, sample in enumerate(samples):
        length = sample.numel() - 1
        # 同一完整序列左右错开一位，就是 next-token prediction 的监督信号。
        input_ids[row, :length] = sample[:-1]
        labels[row, :length] = sample[1:]
    return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# 3. Decoder-only Transformer
# =============================================================================


@dataclass
class ModelConfig:
    """模型结构超参数。

    d_model 是每个 token 的表示维度；n_heads 是注意力头数；每个头的维度为
    d_model / n_heads。n_layers 决定重复堆叠多少个 Transformer Block。
    """

    vocab_size: int = 4096
    max_seq_len: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1024
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        # 当前配置为 384 / 6 = 64，每个注意力头处理64维向量。
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """按最后一维的均方根缩放输入。输入输出形状完全相同。"""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.float()
        inverse_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inverse_rms).to(original_dtype) * self.weight


class RoPE(nn.Module):
    """通过旋转 Q/K 中的成对维度注入 token 位置信息。"""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        # RoPE 把相邻两维看作二维平面上的坐标，所以每次跨两维取一个频率。
        dimension_ids = torch.arange(0, head_dim, 2, dtype=torch.float32)
        # 不同维度使用不同旋转频率：低维变化快，高维变化慢。
        inverse_frequency = 1.0 / (base ** (dimension_ids / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # outer 得到 [max_seq_len, head_dim/2]，即每个位置、每对维度的角度。
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, sequence, head_dim]，简写为 [B,H,T,D]。
        sequence_length = x.size(-2)
        cos = self.cos[:sequence_length].to(x.dtype)[None, None, :, :]
        sin = self.sin[:sequence_length].to(x.dtype)[None, None, :, :]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：每个位置只能看见自己和之前的位置。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        # 一次矩阵乘法同时生成 Q、K、V，再沿最后一维切成三份。
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RoPE(config.head_dim, config.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape

        # [B,T,C] -> 三个 [B,T,C]
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        # [B,T,C] -> [B,H,T,D]，其中 C = H * D。
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query = self.rope(split_heads(query))
        key = self.rope(split_heads(key))
        value = split_heads(value)

        # 数学上等价于 softmax(QK^T / sqrt(D))V。
        # is_causal=True 使用下三角 mask，位置 t 无法看到 t+1 及更后的答案。
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B,H,T,D] -> [B,T,C]
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )
        return self.output(attention)


class SwiGLU(nn.Module):
    """Transformer Block 中的逐位置前馈网络。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.value = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两个上投影产生 [B,T,d_ff]；gate 经 SiLU 后逐元素控制 value，
        # down 再把维度从 d_ff 压回 d_model，使残差相加时形状一致。
        return self.down(F.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Decoder Block：Attention 和 FFN 外各有一条残差连接。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm：先归一化再进入子层；子层输出与原 x 做残差相加。
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TangGPT(nn.Module):
    """Embedding + 多个 Decoder Block + 输出词表概率。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self.initialize_weights)
        # 输入 embedding 和输出分类器共享同一份参数。
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # input_ids: [B,T]；hidden: [B,T,C]
        # Embedding 本质是查表：把每个整数 id 替换成一个 d_model 维向量。
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)

        # logits[b,t,v] 表示第 b 个样本、第 t 个位置预测 token v 的分数。
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            # Cross entropy 会把每个位置4096个词表分数与正确的 label 比较。
            # reshape 把 [B,T,V] 展平成 [B*T,V]，labels 展平成 [B*T]。
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


# =============================================================================
# 4. 最简训练循环
# =============================================================================


def cosine_learning_rate(
    step: int,
    total_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    # progress 从0逐渐增加到1；cos(pi*progress) 从1平滑下降到-1。
    progress = step / max(total_steps - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    path: Path,
    model: TangGPT,
    optimizer: torch.optim.Optimizer,
    scaler,
    config: ModelConfig,
    step: int,
) -> None:
    """保存模型参数、优化器状态和进度。

    仅保存 model 只能用于生成；同时保存 optimizer 和 step，才能在中断后
    以相同的 AdamW 动量状态继续训练。
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "step": step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def choose_mixed_precision(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """选择训练精度，并返回 (autocast dtype, 是否需要 GradScaler)。

    BF16 与 FP16 都用16 bit 保存激活，显存占用比 FP32 更低。BF16 的指数范围
    与 FP32 接近，通常不需要缩放 loss；较旧 GPU 只能用 FP16，为避免小梯度
    下溢，需要 GradScaler。
    """

    if device.type != "cuda":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def train(args: argparse.Namespace) -> None:
    """教学训练循环：支持混合精度、定期保存和断点续训。"""

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleBPETokenizer(PROJECT_ROOT / "artifacts" / "tokenizer.json")
    config = ModelConfig(vocab_size=tokenizer.vocab_size)

    dataset = PoetryDataset(
        PROJECT_ROOT / "data" / "processed" / "train.txt",
        tokenizer,
        config.max_seq_len,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda samples: collate_batch(samples, tokenizer.pad_id),
        # pin_memory 允许 CPU batch 更快地异步复制到 CUDA 显存。
        pin_memory=device.type == "cuda",
    )

    model = TangGPT(config).to(device)
    # AdamW 为每个参数维护一阶动量和二阶动量，所以 checkpoint 也要保存它。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )

    # 5060 Laptop 支持 BF16，因此本机通常会选择 BF16；如果没有 CUDA，
    # autocast_dtype=None，代码自动回到 CPU FP32。
    autocast_dtype, use_scaler = choose_mixed_precision(device)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))
        print(f"从 {args.resume} 恢复，将从 step {start_step + 1} 继续")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device}, precision={autocast_dtype or torch.float32}, "
        f"samples={len(dataset):,}, parameters={parameter_count:,}"
    )
    print(
        f"steps={args.steps:,}, batch_size={args.batch_size}, "
        f"每 {args.save_every:,} 步保存到 {args.output}"
    )

    data_iterator = iter(loader)
    model.train()
    # 用于计算一段时间内的平均 loss、吞吐和剩余时间估计。
    interval_started = time.perf_counter()
    interval_loss = 0.0
    interval_tokens = 0
    interval_steps = 0

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        # non_blocking=True 与 pin_memory 配合，让 CPU->GPU 复制更高效。
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        interval_tokens += int((labels != -100).sum().item())

        learning_rate = cosine_learning_rate(
            step - 1, args.steps, args.learning_rate, args.learning_rate / 10
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        # ------------------------------------------------------------------
        # 每一步训练最核心的过程。
        # ------------------------------------------------------------------
        # set_to_none=True 不写一块全零梯度张量，通常更省显存、速度也略快。
        optimizer.zero_grad(set_to_none=True)

        # autocast 只把适合低精度的矩阵运算转成 BF16/FP16；某些数值敏感
        # 运算仍由 PyTorch 保持在 FP32。模型参数本身仍由优化器可靠地更新。
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            _, loss = model(input_ids, labels)
        assert loss is not None

        # FP16 时 scaler 会先放大 loss 再反向传播；BF16/FP32 时它相当于直通。
        scaler.scale(loss).backward()

        # clip 之前必须把被 scaler 放大的梯度还原，否则裁剪阈值没有意义。
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer.step 真正修改参数；scaler.update 根据是否溢出调整缩放倍数。
        scaler.step(optimizer)
        scaler.update()
        interval_loss += loss.item()
        interval_steps += 1

        if step == 1 or step % args.log_interval == 0:
            elapsed = time.perf_counter() - interval_started
            average_loss = interval_loss / interval_steps
            tokens_per_second = interval_tokens / max(elapsed, 1e-9)
            seconds_per_step = elapsed / interval_steps
            remaining_seconds = max(args.steps - step, 0) * seconds_per_step
            print(
                f"step {step:>6}/{args.steps} | loss {average_loss:.4f} | "
                f"lr {learning_rate:.2e} | grad {float(gradient_norm):.3f} | "
                f"tok/s {tokens_per_second:.0f} | ETA {remaining_seconds / 3600:.2f}h"
            )
            interval_started = time.perf_counter()
            interval_loss = 0.0
            interval_tokens = 0
            interval_steps = 0

        # 对2万步训练，定期覆盖 minimal.pt。即使电脑关机，最多损失最近
        # save_every 步，而不是丢掉整个训练过程。
        if step % args.save_every == 0:
            save_checkpoint(args.output, model, optimizer, scaler, config, step)
            print(f"checkpoint 已保存：{args.output}（step {step}）")

    save_checkpoint(args.output, model, optimizer, scaler, config, args.steps)
    print(f"模型已保存到: {args.output}")


# =============================================================================
# 5. 最简自回归生成
# =============================================================================


@torch.inference_mode()
def sample_tokens(
    model: TangGPT,
    prompt_ids: list[int],
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    """每次取最后一个位置的 logits，采样一个新 token，再接回输入。"""

    device = next(model.parameters()).device
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()

    for _ in range(max_new_tokens):
        if token_ids.size(1) >= model.config.max_seq_len:
            break
        logits, _ = model(token_ids)
        next_logits = logits[:, -1, :] / temperature

        # 只保留分数最高的 top_k 个候选，其余设为负无穷。
        k = min(top_k, next_logits.size(-1))
        threshold = torch.topk(next_logits, k).values[:, -1, None]
        next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_token), dim=1)
        if next_token.item() == eos_id:
            break
    return token_ids[0].tolist()


def generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleBPETokenizer(PROJECT_ROOT / "artifacts" / "tokenizer.json")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TangGPT(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])

    prompt = (
        f"<|bos|><|{args.form}|><|title|>{args.title}"
        f"<|author|>{args.author}<|body|>"
    )
    generated_ids = sample_tokens(
        model,
        tokenizer.encode(prompt),
        tokenizer.eos_id,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    text = tokenizer.decode(generated_ids)
    body = text.split("<|body|>", 1)[-1].split("<|eos|>", 1)[0]
    print(body.replace("<|line|>", "\n"))


# =============================================================================
# 6. 命令行入口
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TangGPT 单文件教学版")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练模型")
    # 8GB 显存的 RTX 5060 Laptop 默认使用 batch 16 + BF16；若出现 OOM，
    # 命令行改成 --batch-size 8 即可，不需要修改源代码。
    train_parser.add_argument("--steps", type=int, default=20000)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--limit", type=int)
    train_parser.add_argument("--save-every", type=int, default=1000)
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="从先前保存的 minimal.pt 继续训练",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "minimal.pt",
    )
    train_parser.set_defaults(function=train)

    generate_parser = subparsers.add_parser("generate", help="生成唐诗")
    generate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "minimal.pt",
    )
    generate_parser.add_argument("--form", choices=["5jue", "7jue", "5lv", "7lv"], default="5jue")
    generate_parser.add_argument("--title", default="秋夜")
    generate_parser.add_argument("--author", default="佚名")
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=40)
    generate_parser.set_defaults(function=generate)
    return parser


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    arguments = build_parser().parse_args()
    arguments.function(arguments)
