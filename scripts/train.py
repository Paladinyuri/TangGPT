"""TangGPT 单卡训练入口。

示例：
    python scripts/train.py --config configs/server_4090.json
    python scripts/train.py --config configs/server_4090.json --resume runs/server_4090/last.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tanggpt.dataset import CausalLMCollator, PoetryDataset
from tanggpt.generation import generate_tokens
from tanggpt.model import TangGPT, TangGPTConfig
from tanggpt.tokenizer import ByteBPETokenizer
from tanggpt.training import (
    TrainConfig,
    append_jsonl,
    atomic_torch_save,
    build_optimizer,
    create_grad_scaler,
    evaluate,
    learning_rate_at_step,
    make_checkpoint,
    next_batch,
    resolve_precision,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TangGPT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--train-limit", type=int, help="只读取前 N 条训练样本，用于 smoke test")
    parser.add_argument("--valid-limit", type=int, help="只读取前 N 条验证样本")
    return parser.parse_args()


def save_sample(
    model: TangGPT,
    tokenizer: ByteBPETokenizer,
    run_dir: Path,
    step: int,
) -> None:
    prompt = "<|bos|><|5jue|><|title|>秋夜<|author|>佚名<|body|>"
    prompt_ids = tokenizer.encode(prompt)
    generated = generate_tokens(
        model,
        prompt_ids,
        eos_id=tokenizer.eos_id,
        max_new_tokens=80,
        temperature=0.8,
        top_k=40,
        seed=step,
    )
    text = tokenizer.decode(generated)
    with (run_dir / "samples.txt").open("a", encoding="utf-8") as file:
        file.write(f"\n===== step {step} =====\n{text}\n")


def main() -> None:
    # Windows PowerShell 可能默认使用 GBK；训练日志统一为 UTF-8。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = TangGPTConfig(**config_payload["model"])
    train_config = TrainConfig(**config_payload["training"])
    set_seed(train_config.seed)

    run_name = args.config.stem
    run_dir = args.run_dir or PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    tokenizer_path = PROJECT_ROOT / "artifacts" / "tokenizer.json"
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab={tokenizer.vocab_size}，但模型 vocab={model_config.vocab_size}"
        )

    shutil.copy2(args.config, run_dir / "config.json")
    shutil.copy2(tokenizer_path, run_dir / "tokenizer.json")

    train_dataset = PoetryDataset(
        PROJECT_ROOT / "data" / "processed" / "train.txt",
        tokenizer,
        model_config.max_seq_len,
        limit=args.train_limit,
    )
    valid_dataset = PoetryDataset(
        PROJECT_ROOT / "data" / "processed" / "valid.txt",
        tokenizer,
        model_config.max_seq_len,
        limit=args.valid_limit,
    )
    collator = CausalLMCollator(tokenizer.pad_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": train_config.batch_size,
        "num_workers": train_config.num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "persistent_workers": train_config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = TangGPT(model_config).to(device)
    optimizer = build_optimizer(model, train_config)
    autocast_dtype, use_scaler = resolve_precision(train_config.precision, device)
    scaler = create_grad_scaler(use_scaler)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"已从 {args.resume} 恢复，继续 step {start_step + 1}")

    effective_batch = train_config.batch_size * train_config.gradient_accumulation_steps
    print(f"device={device}, precision={autocast_dtype or torch.float32}")
    print(f"parameters={model.num_parameters():,}, effective_batch={effective_batch}")
    print(
        f"train={len(train_dataset):,}（截断 {train_dataset.truncated_count}），"
        f"valid={len(valid_dataset):,}（截断 {valid_dataset.truncated_count}）"
    )

    train_iterator = iter(train_loader)
    model.train()
    interval_started = time.perf_counter()
    interval_tokens = 0
    interval_loss = 0.0

    for zero_based_step in range(start_step, train_config.max_steps):
        completed_step = zero_based_step + 1
        learning_rate = learning_rate_at_step(zero_based_step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(train_config.gradient_accumulation_steps):
            batch, train_iterator = next_batch(train_iterator, train_loader)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            interval_tokens += int((labels != -100).sum().item())

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                _, loss = model(input_ids, labels)
                assert loss is not None
                scaled_loss = loss / train_config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.item() / train_config.gradient_accumulation_steps

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        interval_loss += accumulated_loss

        if completed_step % train_config.log_interval == 0:
            elapsed = time.perf_counter() - interval_started
            record = {
                "step": completed_step,
                "train_loss": interval_loss / train_config.log_interval,
                "learning_rate": learning_rate,
                "grad_norm": float(gradient_norm),
                "tokens_per_second": interval_tokens / elapsed,
            }
            append_jsonl(run_dir / "train.jsonl", record)
            print(
                f"step {completed_step:>6}/{train_config.max_steps} | "
                f"loss {record['train_loss']:.4f} | lr {learning_rate:.2e} | "
                f"grad {record['grad_norm']:.3f} | tok/s {record['tokens_per_second']:.0f}"
            )
            interval_started = time.perf_counter()
            interval_tokens = 0
            interval_loss = 0.0

        should_evaluate = (
            completed_step % train_config.eval_interval == 0
            or completed_step == train_config.max_steps
        )
        if should_evaluate:
            val_loss = evaluate(
                model,
                valid_loader,
                device,
                autocast_dtype,
                train_config.eval_batches,
            )
            append_jsonl(
                run_dir / "train.jsonl",
                {"step": completed_step, "val_loss": val_loss},
            )
            print(f"validation step {completed_step} | loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                payload = make_checkpoint(
                    model, optimizer, scaler, model_config, train_config,
                    completed_step, best_val_loss,
                )
                atomic_torch_save(payload, run_dir / "best.pt")
                print(f"已更新 best.pt（val_loss={best_val_loss:.4f}）")

        if completed_step % train_config.sample_interval == 0:
            save_sample(model, tokenizer, run_dir, completed_step)

        should_save = (
            completed_step % train_config.save_interval == 0
            or completed_step == train_config.max_steps
        )
        if should_save:
            payload = make_checkpoint(
                model, optimizer, scaler, model_config, train_config,
                completed_step, best_val_loss,
            )
            atomic_torch_save(payload, run_dir / "last.pt")
            atomic_torch_save(
                payload, run_dir / "checkpoints" / f"step_{completed_step:06d}.pt"
            )

    print(f"训练完成。最佳验证 loss: {best_val_loss:.4f}")
    print(f"输出目录: {run_dir}")


if __name__ == "__main__":
    main()
