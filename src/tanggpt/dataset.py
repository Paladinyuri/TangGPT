"""将清洗后的诗歌文本转换为 PyTorch 可以训练的张量。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteBPETokenizer


class PoetryDataset(Dataset[torch.Tensor]):
    """内存中的 tokenized 唐诗数据集。

    每个元素保留完整的 BOS...EOS token 序列。input/label 的错位在 collator
    中统一完成，这样 Dataset 仍然容易检查和复用。
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteBPETokenizer,
        max_seq_len: int,
        limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_tokens = max_seq_len + 1  # input 和 label 错位后各长 max_seq_len。
        self.samples: list[torch.Tensor] = []
        self.truncated_count = 0

        with self.path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file):
                if limit is not None and line_index >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                token_ids = tokenizer.encode(text)
                if len(token_ids) < 2:
                    continue
                if len(token_ids) > self.max_tokens:
                    # 保留开头的条件信息，并保证截断后的序列仍以 EOS 结束。
                    token_ids = token_ids[: self.max_tokens]
                    token_ids[-1] = tokenizer.eos_id
                    self.truncated_count += 1
                self.samples.append(torch.tensor(token_ids, dtype=torch.long))

        if not self.samples:
            raise ValueError(f"数据集为空: {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


class CausalLMCollator:
    """右侧 padding，并构造错开一位的 input_ids 和 labels。"""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        max_length = max(sample.numel() - 1 for sample in samples)
        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, max_length), self.pad_id, dtype=torch.long
        )
        # CrossEntropyLoss 的 ignore_index=-100 会跳过这些 padding 位置。
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row, sample in enumerate(samples):
            length = sample.numel() - 1
            input_ids[row, :length] = sample[:-1]
            labels[row, :length] = sample[1:]

        return {"input_ids": input_ids, "labels": labels}

