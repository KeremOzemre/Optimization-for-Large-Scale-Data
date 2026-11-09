"""Exploratory paired color-tracking probes; these are not a reasoning benchmark."""

import argparse
import csv
import json
from pathlib import Path

import tiktoken
import torch

from train import file_sha256, load_checkpoint_model, select_device


def probes():
    pairs = [("Lily", "boat", "ball"), ("Tom", "hat", "coat"), ("Mia", "box", "bag"),
             ("Ben", "car", "bike"), ("Anna", "cup", "plate"), ("Sam", "kite", "toy"),
             ("Lucy", "chair", "table"), ("Leo", "door", "gate"),
             ("Emma", "book", "pen"), ("Jack", "sock", "shoe")]
    for i, (name, target, other) in enumerate(pairs):
        for color, distractor in (("red", "blue"), ("blue", "red")):
            # Both versions have the same words; the entity-color assignments swap.
            prompt = (f"{name} had a {color} {target} and a {distractor} {other}. "
                      f"{name} played outside with a friend. Later, the friend asked about the {target}. "
                      f"{name} said that the color of the {target} was")
            yield {"pair": i, "correct": color, "prompt": prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a new output directory")
    out.mkdir(parents=True)
    device = select_device(args.device)
    enc = tiktoken.get_encoding("gpt2")
    colors = ["red", "blue"]
    options = [enc.encode_ordinary(" " + color) for color in colors]
    if any(len(ids) != 1 for ids in options):
        raise ValueError("Each answer must be exactly one GPT-2 token")
    rows, summaries = [], []
    for path in args.checkpoints:
        model, checkpoint = load_checkpoint_model(path, device)
        del checkpoint
        model_rows = []
        with torch.inference_mode():
            for probe in probes():
                ids = enc.encode_ordinary(probe["prompt"])
                if len(ids) > model.config.block_size:
                    raise ValueError("Probe exceeds the checkpoint context; do not compare truncated prompts")
                logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
                log_probs = logits[0, -1].float().log_softmax(-1)
                scores = [float(log_probs[ids[0]].item()) for ids in options]
                predicted = colors[max(range(2), key=lambda i: scores[i])]
                model_rows.append({"checkpoint": str(path), **probe, "prediction": predicted,
                                   "correct_prediction": int(predicted == probe["correct"]),
                                   "logp_red": scores[0], "logp_blue": scores[1]})
        pair_success = [all(row["correct_prediction"] for row in model_rows if row["pair"] == pair)
                        for pair in range(10)]
        summaries.append({"checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                          "accuracy": sum(row["correct_prediction"] for row in model_rows) / 20,
                          "both_versions_correct": sum(pair_success), "pairs": 10})
        rows.extend(model_rows)
        del model
    with (out / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "warning": "20 hand-written prompts, paired color swaps; exploratory, correlated, narrow task. A model can succeed on one version using a color preference. Both-version accuracy is the stronger check. No claim of general reasoning follows.",
        "probe_code_sha256": file_sha256(__file__), "torch": str(torch.__version__),
        "device": str(device), "results": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
