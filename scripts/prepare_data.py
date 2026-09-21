"""将下载的全唐诗 JSON 清洗并切分为模型语料。"""

from pathlib import Path
from tanggpt.data import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw" / "chinese-poetry" / "全唐诗"
    statistics = prepare_dataset(raw_dir, PROJECT_ROOT / "data" / "processed")
    print("数据准备完成：")
    for name, value in sorted(statistics.items()):
        print(f"  {name}: {value:,}")


if __name__ == "__main__":
    main()

