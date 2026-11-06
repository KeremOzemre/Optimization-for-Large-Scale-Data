"""Small correctness checks that run without downloading TinyStories."""

import unittest

import numpy as np
import torch

from model import GPT, GPTConfig
from train import batch_at


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small_model(self):
        torch.manual_seed(42)
        return GPT(GPTConfig(block_size=8, vocab_size=32, n_layer=1,
                             n_head=4, n_embd=32, dropout=0.0, bias=False))

    def test_targets_are_shifted_one_token(self):
        tokens = np.arange(20, dtype=np.uint16)
        x, y = batch_at(tokens, [2], 4, torch.device("cpu"))
        self.assertEqual(x.tolist(), [[2, 3, 4, 5]])
        self.assertEqual(y.tolist(), [[3, 4, 5, 6]])

    def test_future_tokens_cannot_change_earlier_logits(self):
        model = self.small_model().eval()
        a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        b = torch.tensor([[1, 2, 3, 4, 20, 21, 22, 23]])
        with torch.inference_mode():
            a_logits, _ = model(a, a)
            b_logits, _ = model(b, b)
        torch.testing.assert_close(a_logits[:, :4], b_logits[:, :4], atol=1e-6, rtol=0)

    def test_input_and_output_embeddings_share_weights(self):
        model = self.small_model()
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_can_overfit_one_repeated_batch(self):
        model = self.small_model().train()
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).repeat(4, 1)
        y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]]).repeat(4, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        with torch.no_grad():
            _, first_loss = model(x, y)
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            _, final_loss = model(x, y)
        self.assertLess(final_loss.item(), first_loss.item() * 0.2)


if __name__ == "__main__":
    unittest.main()
