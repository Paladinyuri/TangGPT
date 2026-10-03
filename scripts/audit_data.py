"""检查正文跨集合重复，并检索生成诗歌的训练集近似匹配候选。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def extract_body(text: str) -> str:
    return text.split("<|body|>", 1)[-1].split("<|eos|>", 1)[0].replace("<|line|>", "\n").strip()


def normalize_body(text: str) -> str:
    # 保留原始简繁体，不把字符相似性解释为语义相似性。
    return re.sub(r"[^\u3400-\u9fff]", "", text)


def bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {}
    texts = {}
    report = {"normalization": "只保留 CJK 字符，移除标点和控制标记；不转换简繁体。", "splits": {}}
    for name in ("train", "valid", "test"):
        path = args.data_dir / f"{name}.txt"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts[name] = [extract_body(line) for line in lines]
        bodies = [normalize_body(text) for text in texts[name]]
        splits[name] = set(bodies)
        report["splits"][name] = {"poems": len(lines), "duplicate_normalized_bodies": len(bodies) - len(set(bodies)),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report["cross_split_body_overlap"] = {
        "train_valid": len(splits["train"] & splits["valid"]),
        "train_test": len(splits["train"] & splits["test"]),
        "valid_test": len(splits["valid"] & splits["test"]),
    }
    training = [bigrams(normalize_body(text)) for text in texts["train"]]
    index = defaultdict(list)
    for i, grams in enumerate(training):
        for gram in grams:
            index[gram].append(i)
    matches = []
    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, sample in enumerate(samples, 1):
        grams = bigrams(normalize_body(sample["body"]))
        shared = Counter(i for gram in grams for i in index.get(gram, []))
        # 检索所有至少共享一个二元字符组的正文，计算确切的集合 Jaccard。
        ranked = sorted(((count / (len(grams) + len(training[i]) - count), i)
                         for i, count in shared.items()), reverse=True)[:3]
        matches.append({"sample": number, "title": sample["title"], "form": sample["form"],
                        "body": sample["body"], "nearest_training_candidates": [
                            {"bigram_jaccard": score, "body": texts["train"][i]} for score, i in ranked]})
    report["matches"] = matches
    report["note"] = "二元字符组 Jaccard 仅用于定位人工检查候选。低分不能证明没有记忆，高分也不能直接认定抄袭。"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "matches"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
