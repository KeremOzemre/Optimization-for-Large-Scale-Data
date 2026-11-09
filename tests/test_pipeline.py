"""Small correctness checks that run without downloading TinyStories."""

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from model import GPT, GPTConfig
from train import batch_at, fixed_batches, learning_rate, load_checkpoint_model, train_model
from prepare import preparation_settings
from run import load_config
from score_checkpoints import evaluation_windows, paired_bootstrap, score_stories


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

    @patch("tiktoken.get_encoding", return_value=SimpleNamespace(name="gpt2", n_vocab=50257, eot_token=50256))
    def test_training_seed_does_not_change_prepared_data_or_validation(self, _encoding):
        a = load_config("configs/followups/xl_full_1h_seed42.yaml")
        b = load_config("configs/followups/xxl_full_1h_seed43.yaml")
        self.assertEqual(preparation_settings(a), preparation_settings(b))
        tokens = np.arange(1000, dtype=np.uint16)
        starts = []
        for config in (a, b):
            e = config["evaluation"]
            starts.append(np.concatenate(fixed_batches(tokens, e["microbatch_size"], 256,
                                                       e["batches"], e["seed"])))
        np.testing.assert_array_equal(*starts)
        self.assertEqual(len(starts[0]) * 256, 65536)

    def test_all_yaml_configs_validate(self):
        for path in Path("configs").rglob("*.yaml"):
            if path.name != "prompts.yaml":
                with self.subTest(path=path):
                    load_config(path)

    def test_full_scoring_includes_tail_and_weights_by_tokens(self):
        model = self.small_model().eval()
        tokens = np.array([1, 2, 31, 3, 4, 5, 6, 31, 7, 8, 31], dtype=np.uint16)
        windows = list(evaluation_windows(len(tokens), 4, 2, 0, 43))
        nll, counts = score_stories(model, tokens, windows, torch.device("cpu"), "float32", eot_token=31)
        self.assertEqual(counts.tolist(), [2, 5, 3])
        expected = 0.0
        with torch.no_grad():
            for starts, length in windows:
                x, y = batch_at(tokens, starts, length, torch.device("cpu"))
                _, loss = model(x, y)
                expected += loss.item() * y.numel()
        self.assertAlmostEqual(nll.sum() / counts.sum(), expected / 10, places=6)
        delta = paired_bootstrap(nll, nll + counts * 0.2, counts, repeats=20)
        self.assertAlmostEqual(delta["delta_b_minus_a"], 0.2)
        self.assertAlmostEqual(delta["ci95_low"], 0.2)

    def test_generate_stops_when_eot_is_sampled(self):
        model = self.small_model().eval()
        with patch("torch.multinomial", return_value=torch.tensor([[31]])):
            ids = model.generate(torch.tensor([[1, 2]]), 10, eos_token_id=31)
        self.assertEqual(ids.tolist(), [[1, 2, 31]])

    @patch("tiktoken.get_encoding", return_value=SimpleNamespace(name="gpt2", n_vocab=50257, eot_token=50256))
    def test_step_budget_end_to_end_and_time_safety_cap(self, _encoding):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config("configs/smoke.yaml")
            config["data"]["cache_dir"] = str(root / "data")
            config["model"].update(layers=1, width=16, heads=2, context_length=8)
            config["training"].update(device="cpu", precision="float32", max_steps=3,
                                      time_limit_seconds=30, microbatch_size=2,
                                      gradient_accumulation_steps=2, log_interval_steps=1)
            config["schedule"]["progress"] = "steps"
            config["evaluation"].update(microbatch_size=2, batches=2, interval_seconds=0.0001)
            data_dir = Path(config["data"]["cache_dir"])
            data_dir.mkdir()
            tokens = np.tile(np.array([1, 2, 3, 4, 50256], dtype=np.uint16), 20)
            for split in ("train", "val", "test"):
                tokens.tofile(data_dir / f"{split}.bin")
            metadata = {**preparation_settings(config), "eot_token": 50256,
                        "token_counts": {split: len(tokens) for split in ("train", "val", "test")}}
            (data_dir / "meta.json").write_text(json.dumps(metadata))
            config["experiment"]["output_dir"] = str(root / "steps")
            train_model(config)
            out = Path(config["experiment"]["output_dir"])
            summary = json.loads((out / "training_summary.json").read_text())
            self.assertEqual(summary["stop_reason"], "max_steps")
            self.assertEqual(summary["steps"], 3)
            self.assertEqual(summary["tokens_processed"], 96)
            model, checkpoint = load_checkpoint_model(out / "checkpoint.pt", torch.device("cpu"))
            self.assertEqual(checkpoint["tokens_processed"], 96)
            self.assertEqual(model.generate(torch.tensor([[1, 2]]), 2).shape, (1, 4))
            with (out / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["phase"] == "validation" for row in rows))
            self.assertTrue(np.isfinite(float(rows[-1]["val_loss"])))
            self.assertTrue(np.isfinite(float(rows[-1]["train_loss"])))
            config["experiment"]["output_dir"] = str(root / "timecap")
            config["training"].update(time_limit_seconds=0.000001)
            train_model(config)
            summary = json.loads((root / "timecap/training_summary.json").read_text())
            self.assertEqual(summary["stop_reason"], "time_limit")
            self.assertLess(summary["steps"], 3)

    def test_step_schedule_progress(self):
        args = (3e-4, 0.02, 0.1)
        self.assertAlmostEqual(learning_rate(0, 100, *args), 3e-5)
        self.assertAlmostEqual(learning_rate(2, 100, *args), 3e-4)
        self.assertAlmostEqual(learning_rate(100, 100, *args), 3e-5)

    def test_accumulation_grouping_preserves_sampled_window_sequence(self):
        sequences = []
        for microbatch, accumulation in ((8, 2), (4, 4)):
            rng = np.random.default_rng(42)
            sequences.append(np.concatenate([
                rng.integers(0, 473992236 - 256, size=microbatch)
                for _ in range(3 * accumulation)]))
        np.testing.assert_array_equal(*sequences)


if __name__ == "__main__":
    unittest.main()
