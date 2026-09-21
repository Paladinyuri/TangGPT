"""训练、验证、学习率调度和 checkpoint 工具。"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .model import TangGPT, TangGPTConfig


@dataclass
class TrainConfig:
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    max_steps: int = 20_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"  # auto / bf16 / fp16 / fp32
    num_workers: int = 4
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 50
    save_interval: int = 1_000
    sample_interval: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.batch_size <= 0:
            raise ValueError("max_steps 和 batch_size 必须为正数")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正数")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps 必须小于 max_steps")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision 必须是 auto/bf16/fp16/fp32")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at_step(step: int, config: TrainConfig) -> float:
    """线性 warmup 后使用 cosine decay。"""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def build_optimizer(model: TangGPT, config: TrainConfig) -> torch.optim.AdamW:
    """矩阵参数使用 weight decay，bias/Norm 等一维参数不衰减。"""
    decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
    }
    # fused AdamW 在 CUDA 上通常更快；旧版 PyTorch 不支持时自动回退。
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def resolve_precision(requested: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """返回 autocast dtype，以及是否需要 GradScaler。"""
    if device.type != "cuda" or requested == "fp32":
        return None, False
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 bf16，请改用 fp16")
        return torch.bfloat16, False
    return torch.float16, True


def create_grad_scaler(enabled: bool):
    """兼容不同 PyTorch 2.x 版本的 GradScaler 构造接口。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(
    model: TangGPT,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            _, loss = model(input_ids, labels)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """先写临时文件再替换，降低中途断电损坏 checkpoint 的风险。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_checkpoint(
    model: TangGPT,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    model_config: TangGPTConfig,
    train_config: TrainConfig,
    step: int,
    best_val_loss: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "step": step,
        "best_val_loss": best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def next_batch(iterator: Iterable, loader: DataLoader):
    """取下一个 batch；遍历完一个 epoch 后自动创建新 iterator。"""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator
