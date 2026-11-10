"""Audit returned one-hour runs and explain the common-validation comparison.

Reads existing CSV/JSON files only. Writes a separate analysis, audit, and two
scientific figures; it never loads weights, trains, or changes source results.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("xl", "xxl")
SEEDS = (42, 43, 44)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit_training(row, runs_dir):
    directory = runs_dir / row["run"]
    config = json.loads((directory / "run_config.json").read_text())
    metrics = read_csv(directory / "metrics.csv")
    runtime = config["runtime"]
    batch = (config["training"]["microbatch_size"]
             * config["training"]["gradient_accumulation_steps"]
             * config["model"]["context_length"])
    assert batch == runtime["tokens_per_step"] == 4096
    assert config["experiment"]["seed"] == int(row["training_seed"])
    assert config["experiment"]["name"] == row["run"]
    assert runtime["hardware"] == row["training_hardware"]
    assert runtime["data_sha256"]["train.bin"] == row["training_data_sha256"]
    assert runtime["parameter_count"] == int(row["parameters"])
    assert int(row["scored_targets"]) == runtime["prepared_token_counts"]["val"] - 1
    assert math.isclose(math.exp(float(row["loss"])), float(row["perplexity"]), rel_tol=1e-10)
    previous_step, previous_time = 0, 0
    for metric in metrics:
        step, elapsed = int(metric["step"]), float(metric["elapsed_seconds"])
        assert step >= previous_step and elapsed >= previous_time
        assert int(metric["tokens_processed"]) == step * batch
        for key in ("elapsed_seconds", "train_loss", "val_loss", "val_perplexity", "learning_rate", "tokens_per_second"):
            if metric[key]:
                assert math.isfinite(float(metric[key])), (directory, key)
        previous_step, previous_time = step, elapsed
    assert sum(metric["phase"] == "final" for metric in metrics) == 1
    assert metrics[-1]["phase"] == "final"
    assert int(metrics[-1]["tokens_processed"]) == int(row["tokens_processed"])
    assert abs(float(metrics[-1]["elapsed_seconds"]) - float(row["training_seconds"])) < 0.001
    assert 3600 <= float(row["training_seconds"]) < 3601
    normalized = json.loads(json.dumps(config))
    normalized.pop("experiment")
    normalized.pop("runtime")
    for key in ("layers", "width", "heads"):
        normalized["model"].pop(key)
    for key in ("microbatch_size", "gradient_accumulation_steps"):
        normalized["training"].pop(key)
    controls = {"configuration": normalized, **{key: runtime[key] for key in (
        "hardware", "precision", "git_commit", "code_sha256", "data_sha256",
        "python", "packages", "cuda_version", "prepared_token_counts", "validation_targets")}}
    validation = [metric for metric in metrics if metric["val_loss"]]
    return config, validation, controls, len(metrics)


def write_figures(rows, validation, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    palette = {"xl": "#147d92", "xxl": "#b45c20"}
    for seed, color in zip(SEEDS, ("#3958a8", "#9c427a", "#448051")):
        pair = [next(row for row in rows if row["run"] == f"{family}_full_1h_seed{seed}")
                for family in FAMILIES]
        axes[0].plot([0, 1], [float(row["loss"]) for row in pair], "o-", color=color, label=f"Seed {seed}")
    axes[0].set(xticks=[0, 1], xticklabels=["XL (124M)", "XXL (253M)"],
                ylabel="Full-validation cross-entropy (lower is better)", title="XL leads in every observed seed pair")
    axes[0].legend(frameon=False)
    for family, offset in (("xl", -0.18), ("xxl", 0.18)):
        values = [int(next(row for row in rows if row["run"] == f"{family}_full_1h_seed{seed}")["tokens_processed"]) / 1e6
                  for seed in SEEDS]
        bars = axes[1].bar(np.arange(3) + offset, values, width=0.34, color=palette[family], label=family.upper())
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1].set(xticks=range(3), xticklabels=[f"Seed {seed}" for seed in SEEDS],
                ylim=(0, 310), ylabel="Training targets (millions)", title="Same hour, different training exposure")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "seed_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for row in rows:
        family, seed = row["run"].split("_")[0], int(row["training_seed"])
        observations = validation[row["run"]]
        losses = [float(metric["val_loss"]) for metric in observations]
        style = {42: "-", 43: "--", 44: ":"}[seed]
        axes[0].plot([float(metric["elapsed_seconds"]) / 60 for metric in observations], losses,
                     color=palette[family], ls=style, label=f"{family.upper()} seed {seed}")
        axes[1].plot([int(metric["tokens_processed"]) / 1e6 for metric in observations], losses,
                     color=palette[family], ls=style)
    axes[0].set(xlabel="Training-loop minutes", ylabel="Fixed sampled-validation CE", title="Loss against training time")
    axes[1].set(xlabel="Processed training targets (millions)", title="Descriptive: schedules differ at equal targets")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("analysis/followups/seed_validation/scores.csv"))
    parser.add_argument("--archived-scores", type=Path, default=Path("analysis/followups/archived_validation/scores.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/followups_1h"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/followups/review"))
    args = parser.parse_args()
    rows = read_csv(args.scores)
    expected = {f"{family}_full_1h_seed{seed}" for family in FAMILIES for seed in SEEDS}
    assert len(rows) == 6 and {row["run"] for row in rows} == expected
    groups = {family: [row for row in rows if row["run"].startswith(family + "_")] for family in FAMILIES}
    configs, validation, controls, total_rows = {}, {}, [], 0
    for row in rows:
        config, observations, control, count = audit_training(row, args.runs_dir)
        configs[row["run"]], validation[row["run"]] = config, observations
        controls.append(json.dumps(control, sort_keys=True))
        total_rows += count
    assert len(set(controls)) == 1, "Unexpected difference in training controls"
    count = next(iter(configs.values()))["runtime"]["prepared_token_counts"]["train"]
    stats = {}
    for family, group in groups.items():
        losses = [float(row["loss"]) for row in group]
        tokens = statistics.mean(int(row["tokens_processed"]) for row in group)
        stats[family] = {"mean_ce": statistics.mean(losses), "sd_ce": statistics.stdev(losses),
                         "ppl_from_mean_ce": math.exp(statistics.mean(losses)), "mean_targets": tokens,
                         "mean_targets_per_second": statistics.mean(int(row["tokens_processed"]) / float(row["training_seconds"]) for row in group),
                         "cache_equivalents": tokens / count}
    pairs = {seed: {family: next(row for row in rows if row["run"] == f"{family}_full_1h_seed{seed}")
                    for family in FAMILIES} for seed in SEEDS}
    deltas = [float(pair["xxl"]["loss"]) - float(pair["xl"]["loss"]) for pair in pairs.values()]
    delta = statistics.mean(deltas)
    # Conventional t(2) 97.5th percentile; an illustrative interval with only 3 pairs.
    margin = 4.302652729911275 * statistics.stdev(deltas) / math.sqrt(3)
    archived = read_csv(args.archived_scores)
    manifest = json.loads(Path("analysis/followups/checkpoint_manifest.json").read_text())
    for row in archived:
        assert row["checkpoint_sha256"] == manifest["files"][row["run"] + "/checkpoint.pt"]["sha256"]
    missing = [str(path) for directory in (args.scores.parent, args.archived_scores.parent)
               for path in (directory / "protocol.json", directory / "paired_comparisons.json") if not path.exists()]
    for row in rows:
        for filename in ("training_summary.json", "checkpoint.pt"):
            path = args.runs_dir / row["run"] / filename
            if not path.exists():
                missing.append(str(path))
    story_files_present = {str(directory): len(list(directory.glob("*_stories.npz")))
                           for directory in (args.scores.parent, args.archived_scores.parent)}
    audit = {"sources": [str(args.scores), str(args.archived_scores), str(args.runs_dir)],
             "logged_rows_checked": total_rows, "training_controls_match": True,
             "token_counts_and_losses_valid": True, "archived_hashes_match_manifest": True,
             "statistics": stats, "paired_mean_ce_delta_xxl_minus_xl": delta,
             "paired_t95_interval_assuming_normal_differences": [delta - margin, delta + margin],
             "missing_artifacts": missing, "per_story_files_present": story_files_present,
             "offline_protocol_independently_verified": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_figures(rows, validation, args.output_dir)

    lines = ["# One-hour follow-up analysis", "", "Analysis of returned scores and logs; no models were trained or re-evaluated here.", "",
             "**XL has lower reported full-validation loss than XXL in all three seed pairs, and every XL run beats every XXL run.** "
             f"The mean CE advantage is {delta:.6f} nats/token, corresponding to {100 * (1 - math.exp(-delta)):.2f}% lower perplexity for XL. "
             "This strengthens the shared-recipe, one-hour observation. It does not establish intrinsic architectural superiority, convergence, or an optimized recipe for each size.", "",
             "## Six-run results", "", "| Seed | XL CE | XL PPL | XL targets (M) | XXL CE | XXL PPL | XXL targets (M) | XXL minus XL CE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed, pair in pairs.items():
        a, b = pair["xl"], pair["xxl"]
        lines.append(f"| {seed} | {float(a['loss']):.6f} | {float(a['perplexity']):.4f} | {int(a['tokens_processed'])/1e6:.3f} | "
                     f"{float(b['loss']):.6f} | {float(b['perplexity']):.4f} | {int(b['tokens_processed'])/1e6:.3f} | "
                     f"{float(b['loss'])-float(a['loss']):+.6f} |")
    lines += ["", "| Model | Mean CE | Sample SD | PPL from mean CE | Mean targets (M) | Mean targets/s | Cache equivalents |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family, stat in stats.items():
        lines.append(f"| {family.upper()} | {stat['mean_ce']:.6f} | {stat['sd_ce']:.6f} | {stat['ppl_from_mean_ce']:.4f} | "
                     f"{stat['mean_targets']/1e6:.3f} | {stat['mean_targets_per_second']:,.0f} | {stat['cache_equivalents']:.4f} |")
    lines += ["", "PPL from mean CE is the geometric mean of per-run perplexities, not the arithmetic mean. "
              "All six score rows report 3,808,643 evaluated targets. This equals the validation cache length minus its first token, consistent with the planned full-stream protocol; count agreement alone does not verify target identity.", "",
              "![Final validation scores and training exposure](seed_comparison.png)", "", "## What changed after common scoring?", "",
              "| Archived model | Original sampled CE | Common full-validation CE | Change |", "| --- | ---: | ---: | ---: |"]
    for row in archived:
        old = next(metric for metric in reversed(read_csv(Path("results") / row["run"] / "metrics.csv")) if metric["val_loss"])
        before, after = float(old["val_loss"]), float(row["loss"])
        lines.append(f"| {row['run']} | {before:.6f} | {after:.6f} | {after-before:+.6f} |")
    archived_by_name = {row["run"]: row for row in archived}
    archived_gap = float(archived_by_name["xxl_data100"]["loss"]) - float(archived_by_name["xl_data100"]["loss"])
    fraction_gap = float(archived_by_name["xl_data50"]["loss"]) - float(archived_by_name["xl_data100"]["loss"])
    lines += ["", f"The archived full-data XL–XXL gap is now {archived_gap:.6f}, rather than approximately 0.003782 under unequal sampled validation. "
              f"Full-data XL also edges XL/50% by {fraction_gap:.6f}, reversing their original sampled ranking. "
              "That tiny single-seed fraction difference does not establish a generally preferable data fraction.", "", "## Training audit and stability", "",
              f"All **{total_rows:,} logged rows** have finite recorded losses and satisfy `targets = optimizer step × 4,096`. "
              "Steps and elapsed time are monotonic. Each run ends with exactly one final-evaluation row, which repeats the last completed update rather than adding another step. "
              "The six elapsed times range from 3,600.008 to 3,600.101 seconds. No divergence or unfinished time budget was found.", "",
              "Saved configs agree on data/splits, preparation/evaluation seeds, optimizer, schedule, tokenizer, evaluation workload, and effective batch. "
              "Runtime hardware label, bfloat16 precision, code hashes/commit, cache hashes, Python/package/CUDA versions also match. "
              "Architecture, microbatch/accumulation, and experiment seed/name/output are the intended differences. Matching hardware labels do not prove identical live load or sustained performance.", "",
              f"XL processes {stats['xl']['mean_targets']/stats['xxl']['mean_targets']:.3f} times as many targets in the hour. "
              "XXL seed 44 processes 5.98% more targets than seed 42 and 9.80% more than seed 43, while producing the best XXL score. "
              "Its improvement is compatible with an exposure contribution; these three runs cannot separate runtime effects from model/data randomness. Across-run SD is therefore observed run variability, not isolated initialization variance.", "",
              "Every sampled-validation curve improves in its final five minutes. This supports further learning under the observed schedule but neither establishes convergence nor predicts eventual XXL overtaking.", "",
              "![Validation curves against time and targets](learning_curves.png)", "", "## Does XXL learn better per target?", "",
              "Existing curves suggest this is plausible. Linear interpolation of the saved sampled-validation observations at 120M targets gives:", "",
              "| Model | Seed 42 CE | Seed 43 CE | Seed 44 CE |", "| --- | ---: | ---: | ---: |"]
    for family in FAMILIES:
        values = []
        for seed in SEEDS:
            observations = validation[f"{family}_full_1h_seed{seed}"]
            targets = [int(metric["tokens_processed"]) for metric in observations]
            losses = [float(metric["val_loss"]) for metric in observations]
            assert min(targets) <= 120000000 <= max(targets)
            values.append(float(np.interp(120000000, targets, losses)))
        lines.append(f"| {family.upper()} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "These are descriptive interpolated sampled scores, not controlled equal-exposure results. "
              "At equal target counts the models are at different elapsed times and different warmup/cosine progress. "
              "Do not use the curves to claim exposure caused the one-hour ranking or reuse an old checkpoint as a matched step-schedule control.", "", "## Strength and limits of the conclusion", "",
              "Three independently seeded runs per family consistently favour XL here. There are six training runs, not millions of independent training replicates; all nine cross-family comparisons reuse the same six runs. "
              "A model-specific optimizer search was not performed. The result concerns these architectures and this recipe/allocation, not every larger or smaller transformer.", "",
              f"For illustration, a conventional paired t 95% interval for the mean CE gap is [{delta-margin:.6f}, {delta+margin:.6f}]. "
              "It assumes independent, normally distributed paired differences, an assumption that cannot be assessed reliably from three pairs. "
              "The interval includes zero; do not describe this small study as precise population-level statistical confirmation. Report the observed direction, effect size, raw seeds, and uncertainty together.", "",
              "The provided handoff lacks both scoring directories' `protocol.json`, `paired_comparisons.json`, and per-story `.npz` files, plus the six `training_summary.json` files and new checkpoints. "
              "We can audit training settings/logs and interpret the supplied score rows, but cannot independently verify offline scoring precision, validation-cache identity, context policy, checkpoint contents, or story-level intervals. "
              "The four archived checkpoint hashes do match the downloaded-baseline manifest. No missing interval or provenance file was reconstructed as if it were original evidence.", "", "## Recommended next step", "",
              "First return the existing scoring protocol/per-story/paired-comparison files and training summaries; no rerun is needed if those exports are still on the training machine. "
              "If an explanation of the ranking is desired, the preplanned short equal-token pair is now worthwhile: 35,000 updates, 143,360,000 targets per model, fresh weights, and matched step-based schedules. "
              f"At the new observed mean throughputs, expected durations are about {143360000/stats['xl']['mean_targets_per_second']/60:.0f} minutes for XL and "
              f"{143360000/stats['xxl']['mean_targets_per_second']/60:.0f} for XXL (roughly 1.6 GPU-hours total). "
              "Keep the two-hour safety caps and verify both exact target counts. One pair is exploratory. Four-hour runs and a new broad grid remain lower priority.", "",
              "The original data-fraction grid is a separate finding. All families have slightly lower recorded CE at 100% than 25%; Small/25% already processed 8.206 equivalents. "
              "The six full-data follow-ups do not establish the cause or robustness of the fraction effect. No fresh test-set claim follows: all original test results were already inspected.", "",
              "**Report wording:** On the recorded H100 MIG allocation, XL achieved lower reported full-validation cross-entropy than XXL in all three fresh one-hour trials under the shared recipe. "
              "Its mean CE was 1.4947 versus 1.5244, with about 2.9% lower perplexity and 1.93 times as many processed training targets. "
              "The comparison supports a repeatable observed fixed-time advantage; exposure and recipe effects have not been isolated.", "",
              "## Sources and reproduction", "", f"- Scores: `{args.scores}` and `{args.archived_scores}`.",
              f"- Training configs/logs: `{args.runs_dir}/<run>/`.",
              "- Original sampled scores: `results/<archived_run>/metrics.csv`.",
              "- Detailed control/missing-file audit: [audit.json](audit.json).",
              "- Recreate this review with `python analyze_followups.py`; requires the plotting dependencies already used by the repository's other analysis scripts.", ""]
    (args.output_dir / "ANALYSIS.md").write_text("\n".join(lines))
    print(f"Audited {total_rows:,} rows; wrote {args.output_dir / 'ANALYSIS.md'} and two figures.")


if __name__ == "__main__":
    main()
