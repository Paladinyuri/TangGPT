"""TangGPT 的 Decoder-only Transformer 模型。

这个文件暂时只负责“模型本身”，不包含 tokenizer、数据加载和训练循环。
核心结构为：

    token ids -> token embedding -> N 个 DecoderBlock -> RMSNorm -> LM Head

每个 DecoderBlock 使用 Pre-Norm 结构：

    x = x + CausalSelfAttention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TangGPTConfig:
    """集中保存模型超参数。

    第一版默认值约为小型实验模型。真正训练前，我们会根据数据量和显卡
    显存重新选择参数；写成配置类是为了避免在模型各处散落“魔法数字”。
    """

    # 与 artifacts/tokenizer.json 的正式 BPE 词表大小保持一致。
    vocab_size: int = 4_096
    # 清洗后验证集最长样本约 203 个 BPE token，256 可完整容纳绝大多数诗歌。
    max_seq_len: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1_024
    dropout: float = 0.0
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("每个注意力头的维度必须是偶数，RoPE 才能成对旋转")
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size 和 max_seq_len 必须为正数")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    对最后一维做均方根归一化，但不减去均值。weight 是可学习的逐维缩放。
    输入和输出形状相同，通常都是 [batch, sequence, d_model]。
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 float32 计算平方均值，降低半精度训练时的数值误差；
        # 最后再转回原始 dtype，使混合精度训练仍然有效。
        x_float = x.float()
        rms_inverse = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_float * rms_inverse).to(dtype=x.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """为 query/key 预计算 RoPE 的 cos 和 sin。

    RoPE 不向 token embedding 直接添加位置向量，而是对每个注意力头中
    相邻的两维做二维旋转。旋转角度随 token 位置变化。
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        # inv_freq 的长度为 head_dim / 2，每个值对应一对旋转维度。
        dimension_ids = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (dimension_ids / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim / 2]

        # buffer 会随模型一起移动到 GPU，但不会被优化器更新。
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将旋转位置编码作用到 query 和 key。

        query/key: [batch, n_heads, sequence, head_dim]
        """

        seq_len = query.size(-2)
        cos = self.cos[:seq_len].to(dtype=query.dtype)[None, None, :, :]
        sin = self.sin[:seq_len].to(dtype=query.dtype)[None, None, :, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            # 偶数维和奇数维两两组成一个二维向量：
            # [x_even, x_odd] -> [x_even*cos - x_odd*sin,
            #                     x_even*sin + x_odd*cos]
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            rotated_even = x_even * cos - x_odd * sin
            rotated_odd = x_even * sin + x_odd * cos
            return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

        return rotate(query), rotate(key)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。

    “因果”表示第 t 个 token 只能看到 0..t，不能偷看未来 token。这是
    自回归语言模型能够进行 next-token prediction 的关键约束。
    """

    def __init__(self, config: TangGPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        # 一次线性投影同时得到 Q、K、V，计算上比三个独立 Linear 更紧凑。
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_seq_len, config.rope_base
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # qkv: [B, T, 3*C] -> 三个 [B, T, C]
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)

        # 把隐藏维拆成多个注意力头：
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        query, key = self.rope(query, key)

        # PyTorch 会选择可用的高效 attention kernel。is_causal=True 自动应用
        # 下三角 mask，保证每个位置只能访问自己及之前的位置。
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        return self.out_proj(attention_output)


class SwiGLU(nn.Module):
    """带门控的前馈网络。

    公式为 down(silu(gate(x)) * value(x))。与普通的两层 MLP 相比，
    gate 分支可以学习哪些特征应该被传递。
    """

    def __init__(self, config: TangGPTConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.value_proj(x)
        return self.down_proj(hidden)


class DecoderBlock(nn.Module):
    """一个 Pre-Norm Transformer Decoder block。"""

    def __init__(self, config: TangGPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接使信息和梯度可以绕过子层直接传播。
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        x = x + self.residual_dropout(self.feed_forward(self.ffn_norm(x)))
        return x


class TangGPT(nn.Module):
    """用于 next-token prediction 的完整 Decoder-only 语言模型。"""

    def __init__(self, config: TangGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # 输入 embedding 与输出分类矩阵共享权重，减少参数并让“读入 token”
        # 和“预测 token”使用同一个语义空间。
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """计算每个位置的下一个 token logits，并可选地计算训练 loss。

        Args:
            input_ids: [batch, sequence]，元素为 tokenizer 产生的整数 id。
            labels: [batch, sequence]。训练时通常已经相对原文右移一位；
                值为 -100 的位置会被交叉熵忽略。

        Returns:
            logits: [batch, sequence, vocab_size]
            loss: 如果提供 labels，则为标量；否则为 None。
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids 的形状必须是 [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"序列长度 {input_ids.size(1)} 超过 max_seq_len="
                f"{self.config.max_seq_len}"
            )
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels 必须与 input_ids 形状相同")

        hidden = self.token_embedding(input_ids)  # [B, T] -> [B, T, C]
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = True) -> int:
        """返回参数量；共享的 embedding/lm_head 权重只会计算一次。"""

        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
