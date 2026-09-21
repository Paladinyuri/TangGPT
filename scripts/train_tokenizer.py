"""仅使用训练集训练 byte-level BPE tokenizer。"""

import argparse
from pathlib import Path
from tanggpt.tokenizer import ByteBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if stripped := line.strip():
                yield stripped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "tokenizer.json")
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer()
    tokenizer.train(
        iter_lines(PROJECT_ROOT / "data" / "processed" / "train.txt"),
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        verbose=True,
    )
    tokenizer.save(args.output)

    example = "<|bos|><|5jue|><|title|>靜夜思<|author|>李白<|body|>牀前明月光。<|eos|>"
    encoded = tokenizer.encode(example)
    if tokenizer.decode(encoded) != example:
        raise RuntimeError("tokenizer 编码解码往返测试失败")
    print(f"词表大小: {tokenizer.vocab_size:,}")
    print(f"示例 UTF-8 byte 数: {len(example.encode('utf-8')):,}")
    print(f"示例 token 数: {len(encoded):,}")
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()

