"""Check the revised experiment budget without training or network access."""

import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run import load_config
from summarize_followups import EQUAL_TOKEN_TARGETS, main


class FollowupConfigTests(unittest.TestCase):
    def test_equal_token_pair_shares_budget_and_controls(self):
        configs = [load_config(f"configs/followups/{family}_full_equal_tokens_seed42.yaml")
                   for family in ("xl", "xxl")]
        for config in configs:
            with self.subTest(model=config["experiment"]["name"]):
                training = config["training"]
                per_step = (training["microbatch_size"] * training["gradient_accumulation_steps"]
                            * config["model"]["context_length"])
                self.assertEqual(per_step, 4096)
                self.assertEqual(training["max_steps"], 35000)
                self.assertEqual(training["max_steps"] * per_step, EQUAL_TOKEN_TARGETS)
                self.assertEqual(training["time_limit_seconds"], 7200)
                self.assertEqual(config["schedule"]["progress"], "steps")
                self.assertEqual(config["experiment"]["seed"], 42)
                self.assertEqual(config["data"]["preparation_seed"], 42)
                self.assertEqual(config["evaluation"]["seed"], 43)
                self.assertEqual(config["evaluation"]["microbatch_size"]
                                 * config["evaluation"]["batches"]
                                 * config["model"]["context_length"], 65536)
                self.assertEqual(Path(config["experiment"]["output_dir"]).parent,
                                 Path("runs/followups_equal_tokens_143m"))
        self.assertEqual(configs[0]["data"], configs[1]["data"])
        self.assertEqual(configs[0]["evaluation"], configs[1]["evaluation"])
        self.assertEqual(configs[0]["optimizer"], configs[1]["optimizer"])
        self.assertEqual(configs[0]["schedule"], configs[1]["schedule"])

    def summarize_pair(self, xxl_targets):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores.csv"
            fields = ["run", "training_seed", "loss", "training_hardware",
                      "training_data_sha256", "tokens_processed"]
            with scores.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for family, targets, loss in (("xl", EQUAL_TOKEN_TARGETS, 1.5),
                                              ("xxl", xxl_targets, 1.51)):
                    writer.writerow({"run": f"{family}_full_equal_tokens_seed42",
                                     "training_seed": 42, "loss": loss,
                                     "training_hardware": "test-device",
                                     "training_data_sha256": "test-cache-hash",
                                     "tokens_processed": targets})
            output = root / "summary.md"
            with patch("sys.argv", ["summarize_followups.py", "--scores", str(scores),
                                    "--output", str(output)]), redirect_stdout(io.StringIO()):
                main()
            return output.read_text()

    def test_summary_accepts_completed_revised_budget(self):
        result = self.summarize_pair(EQUAL_TOKEN_TARGETS)
        self.assertNotIn("Comparability warnings", result)
        self.assertIn("Mean difference: +0.010000", result)

    def test_summary_warns_when_time_cap_prevents_completion(self):
        result = self.summarize_pair(EQUAL_TOKEN_TARGETS - 4096)
        self.assertIn("Comparability warnings", result)
        self.assertIn("requires 143,360,000", result)

    def test_summary_does_not_accept_old_full_cache_budget(self):
        result = self.summarize_pair(473993216)
        self.assertIn("processed 473,993,216 targets", result)
        self.assertIn("requires 143,360,000", result)


if __name__ == "__main__":
    unittest.main()
