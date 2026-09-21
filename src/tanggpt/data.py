"""全唐诗数据的读取、清洗、体裁识别和稳定切分。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


SENTENCE_SPLIT = re.compile(r"[，。！？；?!;]+")
WHITESPACE = re.compile(r"\s+")
EDITORIAL_MARKERS = re.compile(r"[（）()\[\]【】{}〈〉《》]")


@dataclass(frozen=True)
class Poem:
    poem_id: str
    title: str
    author: str
    paragraphs: tuple[str, ...]
    form: str

    @property
    def body(self) -> str:
        return "".join(self.paragraphs)

    def serialize(self) -> str:
        body = "<|line|>".join(self.paragraphs)
        return (
            f"<|bos|><|{self.form}|><|title|>{self.title}"
            f"<|author|>{self.author}<|body|>{body}<|eos|>"
        )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return WHITESPACE.sub("", text).strip()


def split_clauses(paragraphs: tuple[str, ...]) -> list[str]:
    clauses: list[str] = []
    for paragraph in paragraphs:
        clauses.extend(part for part in SENTENCE_SPLIT.split(paragraph) if part)
    return clauses


def classify_form(paragraphs: tuple[str, ...]) -> str:
    """按句数和每句字数识别常见近体诗形式。"""
    clauses = split_clauses(paragraphs)
    lengths = [len(clause) for clause in clauses]
    if len(clauses) == 4 and lengths == [5] * 4:
        return "5jue"
    if len(clauses) == 4 and lengths == [7] * 4:
        return "7jue"
    if len(clauses) == 8 and lengths == [5] * 8:
        return "5lv"
    if len(clauses) == 8 and lengths == [7] * 8:
        return "7lv"
    return "other"


def load_poems(raw_dir: str | Path, max_body_chars: int = 160) -> Iterator[Poem]:
    """遍历原始 JSON，并跳过明显异常或过长记录。"""
    for json_path in sorted(Path(raw_dir).glob("poet.tang.*.json")):
        records = json.loads(json_path.read_text(encoding="utf-8"))
        for record in records:
            title = normalize_text(str(record.get("title", "")))
            author = normalize_text(str(record.get("author", "")))
            raw_paragraphs = record.get("paragraphs", [])
            if not title or not author or not isinstance(raw_paragraphs, list):
                continue
            paragraphs = tuple(
                paragraph for value in raw_paragraphs
                if (paragraph := normalize_text(str(value)))
            )
            if len(paragraphs) < 2:
                continue
            body = "".join(paragraphs)
            if len(body) > max_body_chars or EDITORIAL_MARKERS.search(body):
                continue
            yield Poem(
                poem_id=str(record.get("id", "")),
                title=title,
                author=author,
                paragraphs=paragraphs,
                form=classify_form(paragraphs),
            )


def stable_split(body: str) -> str:
    """按正文哈希稳定进行 98%/1%/1% 切分。"""
    bucket = int.from_bytes(hashlib.sha256(body.encode("utf-8")).digest()[:4], "big") % 10_000
    if bucket < 9_800:
        return "train"
    if bucket < 9_900:
        return "valid"
    return "test"


def prepare_dataset(raw_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    """清洗、按正文去重、切分，并写出每行一首诗的文本文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "valid", "test")
    }
    statistics: Counter[str] = Counter()
    seen_bodies: set[str] = set()
    try:
        for poem in load_poems(raw_dir):
            if poem.body in seen_bodies:
                statistics["duplicates"] += 1
                continue
            seen_bodies.add(poem.body)
            split = stable_split(poem.body)
            handles[split].write(poem.serialize() + "\n")
            statistics[split] += 1
            statistics[f"form_{poem.form}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (output_dir / "stats.json").write_text(
        json.dumps(dict(sorted(statistics.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dict(statistics)

