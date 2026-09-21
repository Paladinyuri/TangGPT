"""创建一个默认 TangGPT，并打印结构、参数量和一次前向传播结果。"""

import torch

from tanggpt import TangGPT, TangGPTConfig


def main() -> None:
    config = TangGPTConfig()
    model = TangGPT(config)
    sample_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, _ = model(sample_ids)

    print(model)
    print(f"\n参数量: {model.num_parameters():,}")
    print(f"输入形状: {tuple(sample_ids.shape)}")
    print(f"输出形状: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

