"""一个从零实现的 byte-level BPE tokenizer。

BPE 训练不断寻找语料中最常见的相邻 token 对，再把这一对合并为新 token。
我们从 256 种原始 byte 开始，因此任何 UTF-8 文本都能编码，不需要 UNK。
"""

from __future__ import annotations

import base64
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


DEFAULT_SPECIAL_TOKENS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|title|>", "<|author|>",
    "<|body|>", "<|line|>", "<|5jue|>", "<|7jue|>", "<|5lv|>",
    "<|7lv|>", "<|other|>",
)


def _merge_pair(sequence: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """把序列中所有不重叠的指定相邻对合并。"""
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index:index + 2] == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return tuple(merged)


class ByteBPETokenizer:
    """支持训练、编码、解码和序列化的 byte-level BPE。"""

    def __init__(self, special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS) -> None:
        self.special_tokens = tuple(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens 不能重复")

        # id 0..255 永远对应单个原始 byte。
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[int, int]] = []
        self.special_to_id = {token: 256 + i for i, token in enumerate(self.special_tokens)}
        self.id_to_special = {token_id: token for token, token_id in self.special_to_id.items()}
        # merge token 放在特殊 token 之后，使特殊 token id 在重新训练时保持不变。
        self._next_token_id = 256 + len(self.special_tokens)
        self._refresh_helpers()

    @property
    def vocab_size(self) -> int:
        return self._next_token_id

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def _refresh_helpers(self) -> None:
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.merge_to_id = {
            pair: 256 + len(self.special_tokens) + rank
            for rank, pair in enumerate(self.merges)
        }
        alternatives = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in alternatives) + ")"
        )

    def _pretokenize(self, text: str) -> list[bytes]:
        """切分训练片段，并排除不能被拆开的特殊 token。"""
        pieces: list[bytes] = []
        for part in self._special_pattern.split(text):
            if not part or part in self.special_to_id:
                continue
            # 汉字连续段保留在一起；标点和空白成为边界，避免跨诗行 merge。
            units = re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9]+|\s+|[^\w\s]", part)
            pieces.extend(unit.encode("utf-8") for unit in units if unit)
        return pieces

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_pair_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """在文本流上训练 BPE；应只传入训练集，防止数据泄漏。"""
        minimum_size = 256 + len(self.special_tokens)
        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size 至少应为 {minimum_size}")
        if self.merges:
            raise RuntimeError("当前 tokenizer 已训练，请创建新实例重新训练")

        # 相同片段只保存一次，frequency 记录出现次数，减少重复计算。
        frequencies: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for piece in self._pretokenize(text):
                frequencies[tuple(piece)] += 1

        # 倒排索引记录每个 pair 出现在哪些片段。合并后只重新统计受到影响的
        # 片段，不需要在每轮 merge 中扫描整个语料。
        sequences = list(frequencies)
        sequence_frequencies = [frequencies[sequence] for sequence in sequences]
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_sequences: dict[tuple[int, int], set[int]] = defaultdict(set)
        for sequence_index, (sequence, frequency) in enumerate(
            zip(sequences, sequence_frequencies)
        ):
            local_counts = Counter(zip(sequence, sequence[1:]))
            for pair, occurrences in local_counts.items():
                pair_counts[pair] += occurrences * frequency
                pair_to_sequences[pair].add(sequence_index)

        # Python 的 heap 是最小堆：负频次让最高频 pair 位于堆顶，pair 本身
        # 提供确定性的同频 tie-break。旧堆项通过与 pair_counts 对照惰性删除。
        heap = [(-frequency, pair) for pair, frequency in pair_counts.items()]
        heapq.heapify(heap)

        target_merges = vocab_size - minimum_size
        for merge_index in range(target_merges):
            while heap:
                negative_frequency, candidate = heapq.heappop(heap)
                if -negative_frequency == pair_counts.get(candidate, 0):
                    break
            else:
                break

            best_pair = candidate
            best_frequency = -negative_frequency
            if best_frequency < min_pair_frequency:
                break

            new_id = self._next_token_id
            self.merges.append(best_pair)
            self.id_to_bytes[new_id] = self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            self._next_token_id += 1

            affected_indices = list(pair_to_sequences.get(best_pair, set()))
            changed_pairs: set[tuple[int, int]] = set()
            for sequence_index in affected_indices:
                old_sequence = sequences[sequence_index]
                frequency = sequence_frequencies[sequence_index]
                old_counts = Counter(zip(old_sequence, old_sequence[1:]))

                # 先删除旧序列对全局 pair 统计和倒排索引的贡献。
                for pair, occurrences in old_counts.items():
                    pair_counts[pair] -= occurrences * frequency
                    pair_to_sequences[pair].discard(sequence_index)
                    changed_pairs.add(pair)

                new_sequence = _merge_pair(old_sequence, best_pair, new_id)
                sequences[sequence_index] = new_sequence
                new_counts = Counter(zip(new_sequence, new_sequence[1:]))

                # 再加入合并后序列的贡献。
                for pair, occurrences in new_counts.items():
                    pair_counts[pair] += occurrences * frequency
                    pair_to_sequences[pair].add(sequence_index)
                    changed_pairs.add(pair)

            for pair in changed_pairs:
                current_frequency = pair_counts.get(pair, 0)
                if current_frequency > 0:
                    heapq.heappush(heap, (-current_frequency, pair))

            if verbose and (merge_index + 1) % 100 == 0:
                print(f"已学习 {merge_index + 1}/{target_merges} 个 merge，pair 频次={best_frequency}")

        self._refresh_helpers()

    def _encode_bytes(self, raw: bytes) -> list[int]:
        sequence = tuple(raw)
        while len(sequence) >= 2:
            candidates = [
                pair for pair in set(zip(sequence, sequence[1:]))
                if pair in self.merge_ranks
            ]
            if not candidates:
                break
            best_pair = min(candidates, key=self.merge_ranks.__getitem__)
            sequence = _merge_pair(sequence, best_pair, self.merge_to_id[best_pair])
        return list(sequence)

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """将字符串编码为 token id；用户原始输入可设置 allowed_special=False。"""
        if not allowed_special:
            return self._encode_bytes(text.encode("utf-8"))
        token_ids: list[int] = []
        for part in self._special_pattern.split(text):
            if not part:
                continue
            if part in self.special_to_id:
                token_ids.append(self.special_to_id[part])
            else:
                token_ids.extend(self._encode_bytes(part.encode("utf-8")))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """把 token id 无损还原为字符串。"""
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
            elif token_id in self.id_to_bytes:
                byte_buffer.extend(self.id_to_bytes[token_id])
            else:
                raise ValueError(f"未知 token id: {token_id}")
        flush_bytes()
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "type": "byte_bpe",
            "special_tokens": list(self.special_tokens),
            "merges": [list(pair) for pair in self.merges],
            "merge_token_bytes_base64": [
                base64.b64encode(self.id_to_bytes[self.merge_to_id[pair]]).decode("ascii")
                for pair in self.merges
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or payload.get("type") != "byte_bpe":
            raise ValueError("不支持的 tokenizer 文件格式")
        tokenizer = cls(payload["special_tokens"])
        for raw_pair in payload["merges"]:
            pair = (int(raw_pair[0]), int(raw_pair[1]))
            if pair[0] not in tokenizer.id_to_bytes or pair[1] not in tokenizer.id_to_bytes:
                raise ValueError(f"merge 引用了尚未定义的 token: {pair}")
            new_id = tokenizer._next_token_id
            tokenizer.merges.append(pair)
            tokenizer.id_to_bytes[new_id] = tokenizer.id_to_bytes[pair[0]] + tokenizer.id_to_bytes[pair[1]]
            tokenizer._next_token_id += 1
        tokenizer._refresh_helpers()
        return tokenizer
