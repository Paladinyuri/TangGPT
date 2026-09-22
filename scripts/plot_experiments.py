"""Plot comparable TangGPT runs from their train.jsonl logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制多组 TangGPT 训练曲线")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("NAME", "RUN_DIR"),
        required=True,
        help="实验名称和包含 train.jsonl 的运行目录，可重复传入",
    )
    parser.add_argument("--output", type=Path, default=Path("results/experiment_overview.png"))
    return parser.parse_args()


def load_log(run_dir: str) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "train.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    train = [record for record in records if "train_loss" in record]
    valid = [record for record in records if "val_loss" in record]
    return {
        "train_step": np.asarray([record["step"] for record in train]),
        "train_loss": np.asarray([record["train_loss"] for record in train]),
        "valid_step": np.asarray([record["step"] for record in valid]),
        "valid_loss": np.asarray([record["val_loss"] for record in valid]),
    }


def smooth(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < window:
        return values
    middle = np.convolve(values, np.ones(window) / window, mode="valid")
    left = np.full(window // 2, middle[0])
    right = np.full(len(values) - len(middle) - len(left), middle[-1])
    return np.concatenate([left, middle, right])


def main() -> None:
    args = parse_args()
    runs = {name: load_log(run_dir) for name, run_dir in args.run}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, run in runs.items():
        axes[0].plot(run["train_step"], smooth(run["train_loss"]), label=name)
        axes[1].plot(run["valid_step"], run["valid_loss"], label=name)
        aligned_train = np.interp(run["valid_step"], run["train_step"], run["train_loss"])
        axes[2].plot(run["valid_step"], run["valid_loss"] - aligned_train, label=name)

        best_index = int(np.argmin(run["valid_loss"]))
        best_step = int(run["valid_step"][best_index])
        best_loss = float(run["valid_loss"][best_index])
        axes[1].scatter(best_step, best_loss, s=65, zorder=5)
        print(f"{name}: best step={best_step}, val_loss={best_loss:.4f}")

    titles = ("Smoothed training loss", "Validation loss", "Generalization gap")
    ylabels = ("Loss", "Loss", "Validation loss - training loss")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
