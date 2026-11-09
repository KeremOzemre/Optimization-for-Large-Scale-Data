"""Summarize common-validation results without selecting a best random seed."""

import argparse
import csv
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="One scores.csv from score_checkpoints.py")
    parser.add_argument("--output", required=True, help="New Markdown file")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output file")
    with Path(args.scores).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pattern = re.compile(r"(xl|xxl)_full_(1h|4h|equal_tokens)_seed(\d+)")
    grouped = {}
    for row in rows:
        match = pattern.fullmatch(row["run"])
        if match:
            family, budget, seed = match.groups()
            seed = int(seed)
            if seed != int(row["training_seed"]):
                raise ValueError(f"Checkpoint seed contradicts run name: {row['run']}")
            group = grouped.setdefault((budget, family), {})
            if seed in group:
                raise ValueError("Duplicate seed in a model/budget group")
            group[seed] = float(row["loss"])
    if not grouped:
        parser.error("No follow-up run names found in the scoring table")
    hardware = {row["training_hardware"] for row in rows}
    hashes = {row["training_data_sha256"] for row in rows}
    warnings = []
    if len(hardware) != 1 or not all(hardware):
        warnings.append("Training hardware is missing or differs; do not interpret this as a controlled time-budget comparison.")
    if len(hashes) != 1 or not all(hashes):
        warnings.append("Training-cache hashes are missing or differ; reconcile data provenance before interpreting differences.")
    for row in rows:
        if "equal_tokens" in row["run"] and int(row["tokens_processed"]) != 473993216:
            warnings.append(f"{row['run']} did not reach the equal-token target.")
    lines = ["# Follow-up validation results", "", f"Source: `{args.scores}`.", "",
             "All scores use the common protocol saved beside the source CSV. Lower CE is better. "
             "Standard deviations describe training-seed variation, while paired story intervals in "
             "`paired_comparisons.json` describe evaluation-sample variation conditional on each checkpoint.", ""]
    if warnings:
        lines += ["**Comparability warnings:**", "", *[f"- {warning}" for warning in warnings], ""]
    lines += ["| Budget | Model | Seeds | Mean CE | Sample SD | Min CE | Max CE |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for (budget, family), values in sorted(grouped.items()):
        scores = list(values.values())
        sd = f"{statistics.stdev(scores):.6f}" if len(scores) > 1 else "unavailable (one seed)"
        lines.append(f"| {budget} | {family} | {', '.join(map(str, sorted(values)))} | "
                     f"{statistics.mean(scores):.6f} | {sd} | {min(scores):.6f} | {max(scores):.6f} |")
    lines += ["", "Paired training-seed differences (XXL CE minus XL CE; positive favours XL):", ""]
    for budget in sorted({budget for budget, _ in grouped}):
        xl, xxl = grouped.get((budget, "xl"), {}), grouped.get((budget, "xxl"), {})
        common = sorted(set(xl) & set(xxl))
        if common:
            delta = [xxl[seed] - xl[seed] for seed in common]
            lines.append(f"- {budget}: " + "; ".join(f"seed {seed}: {xxl[seed] - xl[seed]:+.6f}" for seed in common)
                         + f". Mean difference: {statistics.mean(delta):+.6f}.")
    lines += ["", "Three seeds are an initial variability check. Do not equate a small mean gap with a universal ranking. "
              "Longer-budget runs use budget-specific schedules; equal-token runs still sample with replacement and do not guarantee an epoch.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
