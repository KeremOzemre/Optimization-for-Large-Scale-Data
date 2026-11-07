"""Aggregate and plot results across experiment runs.

Reads every run under --runs-dir that contains both ``run_config.json`` and
``metrics.csv`` (the files written by ``train.py``), builds a summary table,
and saves comparison figures. It never trains anything, so it can be run on a
laptop once teammates copy back the run folders produced on the cluster.

Usage:
    python analyze.py                       # reads results/, writes analysis/
    python analyze.py --runs-dir results --out-dir analysis
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: save files, never open a window.
import matplotlib.pyplot as plt
import pandas as pd

# Columns written by train.py (train.CSV_FIELDS). The final row has phase "final".
PHASE_FINAL = "final"


def _get(config, *keys, default=None):
    """Safely walk nested dict keys, returning default if any is missing."""
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def parse_data_fraction(name, train_stories):
    """Data fraction as a percent, from the run name (e.g. large_data75) or null=full."""
    match = re.search(r"data(\d+)", str(name))
    if match:
        return int(match.group(1))
    if train_stories is None:
        return 100  # train_stories null means the full split.
    return None


def parse_size_label(name, param_count):
    """A short model-size label from the run name, else the parameter count in millions."""
    # Longest labels first so "xxl" is not matched as "xl".
    for label in ("xxl", "xl", "small", "medium", "large"):
        if re.match(rf"{label}_", str(name)):
            return label
    if param_count:
        return f"{param_count / 1e6:.0f}M"
    return str(name)


def load_run(run_dir):
    """Return (summary_dict, curve_dataframe) for one run, or None if incomplete."""
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.csv"
    if not config_path.exists() or not metrics_path.exists():
        return None

    config = json.loads(config_path.read_text())
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None

    name = _get(config, "experiment", "name", default=run_dir.name)
    param_count = _get(config, "runtime", "parameter_count")
    train_stories = _get(config, "data", "train_stories")

    # The run's chosen metric: the final-row validation loss, with best-ever as backup.
    final_rows = metrics[metrics["phase"] == PHASE_FINAL]
    final = final_rows.iloc[-1] if not final_rows.empty else metrics.iloc[-1]
    val_series = metrics["val_loss"].dropna()
    best_val_loss = val_series.min() if not val_series.empty else float("nan")

    # Optional test numbers, if evaluate results were saved alongside the run.
    test_path = run_dir / "test_results.json"
    test = json.loads(test_path.read_text()) if test_path.exists() else {}

    summary = {
        "run": name,
        "size": parse_size_label(name, param_count),
        "data_pct": parse_data_fraction(name, train_stories),
        "layers": _get(config, "model", "layers"),
        "width": _get(config, "model", "width"),
        "heads": _get(config, "model", "heads"),
        "context_length": _get(config, "model", "context_length"),
        "parameters": param_count,
        "params_millions": round(param_count / 1e6, 2) if param_count else None,
        "device": _get(config, "runtime", "device"),
        "precision": _get(config, "runtime", "precision"),
        "train_stories": train_stories,
        "final_val_loss": round(float(final.get("val_loss")), 4)
        if pd.notna(final.get("val_loss")) else None,
        "final_val_perplexity": round(float(final.get("val_perplexity")), 3)
        if pd.notna(final.get("val_perplexity")) else None,
        "best_val_loss": round(float(best_val_loss), 4) if pd.notna(best_val_loss) else None,
        "tokens_processed": int(final.get("tokens_processed", 0)),
        "tokens_per_second": round(float(final.get("tokens_per_second", 0)), 1),
        "steps": int(final.get("step", 0)),
        "test_loss": test.get("test_loss"),
        "test_perplexity": test.get("test_perplexity"),
        "test_bits_per_byte": test.get("test_bits_per_byte"),
    }

    curve = metrics.assign(run=name)
    return summary, curve


def collect(runs_dir):
    """Load every run under runs_dir. Returns (summary_df, long_curve_df)."""
    summaries, curves = [], []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        loaded = load_run(run_dir)
        if loaded is None:
            print(f"  skipping {run_dir.name}: no run_config.json + metrics.csv")
            continue
        summary, curve = loaded
        summaries.append(summary)
        curves.append(curve)
        print(f"  loaded {summary['run']}: "
              f"{summary['params_millions']}M params, "
              f"final val loss {summary['final_val_loss']}")
    if not summaries:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True)


def plot_loss_curves(curves, out_dir):
    """Validation loss over wall-clock time, one line per run."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, group in curves.groupby("run"):
        valid = group.dropna(subset=["val_loss"])
        if valid.empty:
            continue
        ax.plot(valid["elapsed_seconds"], valid["val_loss"], marker="o", label=run)
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Validation loss (cross-entropy)")
    ax.set_title("Validation loss vs. training time")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss_curves.png", dpi=150)
    plt.close(fig)


def _scatter_grouped(summary, x, y, group, title, xlabel, ylabel, filename, out_dir,
                     logx=False):
    """Scatter y vs x, one connected line per group value (e.g. per data fraction)."""
    plot_df = summary.dropna(subset=[x, y, group])
    if plot_df.empty:
        print(f"  skip {filename}: no rows with {x}, {y}, {group}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    order = {"small": 0, "medium": 1, "large": 2, "xl": 3, "xxl": 4}
    groups = sorted(plot_df.groupby(group), key=lambda kv: order.get(kv[0], kv[0]) if group == "size" else kv[0])
    for key, g in groups:
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", label=f"{group}={key}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def write_report_table(summary, out_dir):
    """Markdown table ready to paste into the report, sorted by final val loss."""
    columns = ["run", "size", "data_pct", "layers", "width", "heads",
               "params_millions", "tokens_processed", "steps",
               "tokens_per_second", "final_val_loss", "final_val_perplexity",
               "test_loss", "test_perplexity", "test_bits_per_byte"]
    present = [c for c in columns if c in summary.columns]
    ordered = summary.sort_values("final_val_loss", na_position="last")[present]
    (out_dir / "summary_table.md").write_text(ordered.to_markdown(index=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="results", type=Path,
                        help="Directory holding one subfolder per run (default: results)")
    parser.add_argument("--out-dir", default="analysis", type=Path,
                        help="Where to write tables and figures (default: analysis)")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        raise SystemExit(f"No runs directory at {args.runs_dir}. "
                         "Point --runs-dir at the folder with the run outputs.")

    print(f"Collecting runs from {args.runs_dir}/ ...")
    summary, curves = collect(args.runs_dir)
    if summary.empty:
        raise SystemExit("No complete runs found (need run_config.json + metrics.csv).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values("final_val_loss", na_position="last").to_csv(
        args.out_dir / "summary.csv", index=False)
    write_report_table(summary, args.out_dir)

    plot_loss_curves(curves, args.out_dir)
    _scatter_grouped(summary, "parameters", "final_val_loss", "data_pct",
                     "Final validation loss vs. model size",
                     "Parameters", "Final validation loss",
                     "val_loss_vs_params.png", args.out_dir, logx=True)
    _scatter_grouped(summary, "data_pct", "final_val_loss", "size",
                     "Final validation loss vs. training-data fraction",
                     "Training data (% of split)", "Final validation loss",
                     "val_loss_vs_data.png", args.out_dir)
    _scatter_grouped(summary, "tokens_processed", "final_val_loss", "size",
                     "Final validation loss vs. tokens processed in the budget",
                     "Tokens processed", "Final validation loss",
                     "val_loss_vs_tokens.png", args.out_dir, logx=True)
    _scatter_grouped(summary, "parameters", "tokens_per_second", "data_pct",
                     "Throughput vs. model size",
                     "Parameters", "Tokens per second",
                     "throughput_vs_params.png", args.out_dir, logx=True)

    print(f"\nWrote {args.out_dir}/summary.csv, summary_table.md, and figures.")
    print(f"Best run by final validation loss: "
          f"{summary.sort_values('final_val_loss').iloc[0]['run']}")


if __name__ == "__main__":
    main()
