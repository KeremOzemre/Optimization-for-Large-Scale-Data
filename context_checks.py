"""Balanced exploratory color-tracking probes; these are not a reasoning benchmark."""

import argparse
import csv
import json
from pathlib import Path

import tiktoken
import torch

from train import file_sha256, load_checkpoint_model, select_device


PROTOCOL_VERSION = "color_tracking_v2_balanced"


def probes():
    pairs = [("Lily", "boat", "ball"), ("Tom", "hat", "coat"), ("Mia", "box", "bag"),
             ("Ben", "car", "bike"), ("Anna", "cup", "plate"), ("Sam", "kite", "toy"),
             ("Lucy", "chair", "table"), ("Leo", "door", "gate"),
             ("Emma", "book", "pen"), ("Jack", "sock", "shoe")]
    for scenario, (name, object_a, object_b) in enumerate(pairs):
        for color_a, color_b in (("red", "blue"), ("blue", "red")):
            objects = {"a": object_a, "b": object_b}
            colors = {"a": color_a, "b": color_b}
            for first, second in (("a", "b"), ("b", "a")):
                for queried_entity in ("a", "b"):
                    # Balance all three factors: color assignment, mention order,
                    # and queried object. Copying either color position scores 50%.
                    queried_object = objects[queried_entity]
                    prompt = (f"{name} had a {colors[first]} {objects[first]} and a "
                              f"{colors[second]} {objects[second]}. "
                              f"{name} played outside with a friend. Later, the friend asked "
                              f"about the {queried_object}. {name} said that the color "
                              f"of the {queried_object} was")
                    yield {
                        "scenario": scenario, "object_a": object_a, "object_b": object_b,
                        "color_a": color_a, "color_b": color_b,
                        "color_assignment": f"a_{color_a}_b_{color_b}",
                        "mention_order": f"{first}_{second}",
                        "queried_entity": queried_entity, "queried_object": queried_object,
                        "query_position": 1 if queried_entity == first else 2,
                        "correct": colors[queried_entity], "prompt": prompt,
                    }


def prediction_summary(rows):
    """Require every balanced variant, rather than one favorable prompt per scenario."""
    expected = {}
    for probe in probes():
        expected.setdefault(probe["scenario"], set()).add(
            (probe["color_assignment"], probe["mention_order"], probe["queried_entity"]))
    successful = 0
    for scenario, variants in expected.items():
        selected = [row for row in rows if row["scenario"] == scenario]
        observed = {(row["color_assignment"], row["mention_order"], row["queried_entity"])
                    for row in selected}
        successful += (observed == variants and len(selected) == len(variants)
                       and all(row["correct_prediction"] for row in selected))
    return {
        "accuracy": sum(row["correct_prediction"] for row in rows) / len(rows),
        "prompts": len(rows), "scenarios": len(expected),
        "scenarios_all_variants_correct": successful,
    }


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
        summaries.append({"checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                          **prediction_summary(model_rows)})
        rows.extend(model_rows)
        del model
    with (out / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "protocol": PROTOCOL_VERSION,
        "warning": "80 hand-written prompts across 10 scenarios, with all 8 combinations of queried entity, mention order, and color assignment. Constant-color and first/last-color rules score 50%; this does not exclude every possible shortcut. Scenarios are correlated and narrow. The strict score requires all 8 variants of a scenario to be correct. No claim of general reasoning follows. Results are not comparable to the flawed v1 first-object-only probes.",
        "probe_code_sha256": file_sha256(__file__), "torch": str(torch.__version__),
        "device": str(device), "results": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
