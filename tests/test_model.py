import unittest
from pathlib import Path

import torch

from tanggpt import TangGPT, TangGPTConfig
from tanggpt.dataset import CausalLMCollator
from tanggpt.tokenizer import ByteBPETokenizer
from tanggpt.training import TrainConfig, learning_rate_at_step


def tiny_config() -> TangGPTConfig:
    """测试使用极小模型，让 CPU 上也能快速运行。"""

    return TangGPTConfig(
        vocab_size=100,
        max_seq_len=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )


class TangGPTTest(unittest.TestCase):
    def test_output_shape_loss_and_backward(self) -> None:
        model = TangGPT(tiny_config())
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 8))

        logits, loss = model(input_ids, labels)

        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertIsNotNone(loss)
        assert loss is not None  # 帮助静态类型检查器理解 loss 不再是 None。
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

        # 一次完整反向传播，确认计算图从 loss 连通到模型参数。
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_attention_is_causal(self) -> None:
        """改变未来 token，不应影响之前位置的输出。"""

        torch.manual_seed(7)
        model = TangGPT(tiny_config()).eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6]])
        second = torch.tensor([[1, 2, 3, 4, 91, 92]])

        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)

        # 前四个输入完全相同；因果 mask 下，它们不能感知第五、六个 token。
        torch.testing.assert_close(first_logits[:, :4], second_logits[:, :4])

    def test_weight_tying(self) -> None:
        model = TangGPT(tiny_config())
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_rejects_too_long_sequence(self) -> None:
        model = TangGPT(tiny_config())
        input_ids = torch.zeros((1, 17), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            model(input_ids)


class ByteBPETokenizerTest(unittest.TestCase):
    def test_round_trip_before_and_after_training(self) -> None:
        tokenizer = ByteBPETokenizer()
        text = "<|bos|>床前明月光，疑是地上霜。<|eos|>"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

        tokenizer.train([text, text, "海上生明月，天涯共此時。"], vocab_size=300)
        encoded = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertLess(len(encoded), len(text.encode("utf-8")))

    def test_save_and_load(self) -> None:
        import tempfile

        tokenizer = ByteBPETokenizer()
        tokenizer.train(["明月明月明月", "春風春風"], vocab_size=290)
        text = "<|bos|>明月照春風<|eos|>"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPETokenizer.load(path)

        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)


class TrainingUtilitiesTest(unittest.TestCase):
    def test_collator_shifts_and_ignores_padding(self) -> None:
        collator = CausalLMCollator(pad_id=0)
        batch = collator([
            torch.tensor([10, 11, 12, 13]),
            torch.tensor([20, 21, 22]),
        ])
        torch.testing.assert_close(
            batch["input_ids"], torch.tensor([[10, 11, 12], [20, 21, 0]])
        )
        torch.testing.assert_close(
            batch["labels"], torch.tensor([[11, 12, 13], [21, 22, -100]])
        )

    def test_warmup_and_cosine_learning_rate(self) -> None:
        config = TrainConfig(
            max_steps=100,
            warmup_steps=10,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
        )
        self.assertAlmostEqual(learning_rate_at_step(9, config), 1e-3)
        self.assertLess(learning_rate_at_step(99, config), 2e-4)


if __name__ == "__main__":
    unittest.main()
