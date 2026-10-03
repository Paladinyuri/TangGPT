"""固定生成条件，并按有效 token 数汇总独立数据集损失。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from torch.utils.data import DataLoader
from tanggpt.dataset import PoetryDataset, CausalLMCollator
from tanggpt.generation import generate_tokens
from tanggpt.model import TangGPT, TangGPTConfig
from tanggpt.tokenizer import ByteBPETokenizer
from tanggpt.data import classify_form, split_clauses


def body(text):
    return text.split("<|body|>", 1)[-1].split("<|eos|>", 1)[0].replace("<|line|>", "\n").strip()


def canonical(text):
    return re.sub(r"[^\u3400-\u9fff]", "", text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizer.json")
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/test.txt")
    p.add_argument("--train-data", type=Path, default=ROOT / "data/processed/train.txt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-per-form", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261003)
    args = p.parse_args()
    if args.samples_per_form < 1 or args.batch_size < 1:
        p.error("sample count and batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TangGPT(TangGPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": ckpt.get("step"),
              "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": str(device), "seed": args.seed,
              "temperature": 0.8, "top_k": 40, "max_new_tokens": 192,
              "note": "格式指标仅检查句数和字数，不判断平仄、押韵或文学质量。历史验证 loss 为 batch 均值，不能与本脚本 token 加权 loss 完全等同。"}
    if args.data.exists():
        dataset = PoetryDataset(args.data, tokenizer, model.config.max_seq_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=CausalLMCollator(tokenizer.pad_id))
        total_loss, total_tokens = 0., 0
        with torch.inference_mode():
            for batch in loader:
                labels = batch["labels"].to(device)
                _, loss = model(batch["input_ids"].to(device), labels)
                n = int((labels != -100).sum())
                total_loss += float(loss) * n
                total_tokens += n
        mean = total_loss / total_tokens
        report["evaluation"] = {"data": str(args.data), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                                "poems": len(dataset), "truncated_poems": dataset.truncated_count,
                                "tokens": total_tokens, "token_weighted_loss": mean,
                                "perplexity": math.exp(mean)}
        print(json.dumps(report["evaluation"], ensure_ascii=False), flush=True)
    else:
        report["evaluation"] = {"status": "missing_data"}
    train_bodies = set()
    if args.train_data.exists():
        train_bodies = {canonical(body(x)) for x in args.train_data.read_text(encoding="utf-8").splitlines() if x.strip()}
    rows = []
    titles = ["秋夜", "春江", "山行", "雨后", "送别", "月下", "归乡", "江上", "冬雪", "闲居"]
    for form in ("5jue", "7jue", "5lv", "7lv"):
        for i in range(args.samples_per_form):
            title = titles[i % len(titles)]
            prompt = f"<|bos|><|{form}|><|title|>{title}<|author|>佚名<|body|>"
            prompt_ids = tokenizer.encode(prompt)
            seed = args.seed + len(rows)
            ids = generate_tokens(model, prompt_ids, tokenizer.eos_id, max_new_tokens=192,
                                  temperature=0.8, top_k=40, seed=seed)
            text = body(tokenizer.decode(ids))
            clauses = split_clauses((text.replace("\n", ""),))
            row = {"form": form, "title": title, "seed": seed, "body": text,
                   "format_ok": classify_form((text.replace("\n", ""),)) == form,
                   "ended_with_eos": ids[-1] == tokenizer.eos_id,
                   "repeated_clause": len(set(clauses)) < len(clauses),
                   "exact_train_body": canonical(text) in train_bodies if train_bodies else None}
            rows.append(row)
            print(f"{form} {i+1}/{args.samples_per_form}: format={row['format_ok']}", flush=True)
    summary = {}
    for form in ("5jue", "7jue", "5lv", "7lv"):
        group = [r for r in rows if r["form"] == form]
        summary[form] = {"samples": len(group), **{k: sum(r[k] for r in group)/len(group)
                          for k in ("format_ok", "ended_with_eos", "repeated_clause")}}
    report["generation"] = summary
    report["exact_train_match_count"] = sum(r["exact_train_body"] is True for r in rows) if train_bodies else None
    report["overlap_note"] = "仅检查去标点正文完全匹配，不能排除近似复述。"
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    rendered = "\n\n".join(f"## {i+1}. {r['form']} / {r['title']}\n\n{r['body']}\n\n格式合规：{r['format_ok']}；EOS：{r['ended_with_eos']}；重复句：{r['repeated_clause']}" for i,r in enumerate(rows))
    (args.output_dir / "samples.md").write_text("# 固定条件下的全部生成样例\n\n" + rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
