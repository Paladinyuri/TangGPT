"""TangGPT 的自回归采样工具。"""

from __future__ import annotations

import torch

from .model import TangGPT


@torch.inference_mode()
def generate_tokens(
    model: TangGPT,
    prompt_ids: list[int],
    eos_id: int,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int | None = None,
) -> list[int]:
    """逐 token 生成，返回“提示词 + 新生成内容”的完整 token 序列。"""

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k 必须是正整数或 None")
    if len(prompt_ids) >= model.config.max_seq_len:
        raise ValueError("提示词已经达到模型最大上下文长度")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    output = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            if output.size(1) >= model.config.max_seq_len:
                break
            logits, _ = model(output)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, k).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            output = torch.cat((output, next_token), dim=1)
            if next_token.item() == eos_id:
                break
    finally:
        model.train(was_training)
    return output[0].tolist()

