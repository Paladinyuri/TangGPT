"""从训练好的 checkpoint 生成唐诗。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tanggpt.generation import generate_tokens
from tanggpt.model import TangGPT, TangGPTConfig
from tanggpt.tokenizer import ByteBPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORM_TOKENS = {"5jue", "7jue", "5lv", "7lv", "other"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 TangGPT 生成唐诗")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--form", choices=sorted(FORM_TOKENS), default="5jue")
    parser.add_argument("--title", default="秋夜")
    parser.add_argument("--author", default="佚名")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "generated_poems.txt")
    return parser.parse_args()


def extract_body(text: str) -> str:
    """从带控制 token 的完整结果中提取便于阅读的正文。"""
    if "<|body|>" in text:
        text = text.split("<|body|>", 1)[1]
    text = text.split("<|eos|>", 1)[0]
    return text.replace("<|line|>", "\n").strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = TangGPTConfig(**checkpoint["model_config"])
    model = TangGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer_path = args.tokenizer or args.checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = PROJECT_ROOT / "artifacts" / "tokenizer.json"
    tokenizer = ByteBPETokenizer.load(tokenizer_path)

    prompt = (
        f"<|bos|><|{args.form}|><|title|>{args.title}"
        f"<|author|>{args.author}<|body|>"
    )
    prompt_ids = tokenizer.encode(prompt)
    outputs: list[str] = []
    for sample_index in range(args.num_samples):
        generated_ids = generate_tokens(
            model,
            prompt_ids,
            tokenizer.eos_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + sample_index,
        )
        body = extract_body(tokenizer.decode(generated_ids))
        rendered = (
            f"题目：{args.title}\n体裁：{args.form}\n作者条件：{args.author}\n\n{body}"
        )
        outputs.append(rendered)
        print(f"\n===== 样本 {sample_index + 1} =====\n{rendered}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(outputs) + "\n", encoding="utf-8")
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
