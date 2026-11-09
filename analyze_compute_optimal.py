"""Compute-optimal analysis (no retraining).

Reads analysis/summary.csv (produced by analyze.py) and, using only the already
trained runs, (1) estimates the training compute of each run with the standard
C ~= 6 N D approximation (N = parameters, D = training tokens processed), (2) plots
the loss--compute frontier, and (3) fits a parabola to best-per-size validation
loss in log-parameters as a descriptive visualization. This is not an estimate
of a statistically established optimum or measured GPU utilization.

Usage:
    python analyze_compute_optimal.py            # reads analysis/summary.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZE_ORDER = ["small", "medium", "large", "xl", "xxl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="analysis/summary.csv", type=Path)
    parser.add_argument("--out-dir", default="analysis", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    # Training compute (FLOPs), standard approximation C ~= 6 * params * tokens.
    df["flops"] = 6.0 * df["parameters"] * df["tokens_processed"]

    # --- (1) Loss vs compute, coloured by size ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for size in SIZE_ORDER:
        g = df[df["size"] == size]
        if g.empty:
            continue
        ax.scatter(g["flops"], g["final_val_loss"], s=60, label=size)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated training compute  C ≈ 6 N D  (FLOPs)")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Recorded loss vs. approximate 6ND proxy (one hour)")
    ax.legend(title="model size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "loss_vs_compute.png", dpi=150)
    plt.close(fig)

    # Use the same fraction at every size; do not select a different subset per point.
    best = df[df["data_pct"] == 100].copy()
    best["size"] = pd.Categorical(best["size"], SIZE_ORDER, ordered=True)
    best = best.sort_values("size")

    x = np.log10(best["parameters"].to_numpy())   # log10 params
    y = best["final_val_loss"].to_numpy()
    a, b, c = np.polyfit(x, y, 2)                  # quadratic fit
    x_star = -b / (2 * a)                          # vertex (log10 of optimal N)
    n_star = 10 ** x_star

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(best["parameters"], y, s=70, color="C3", zorder=3,
               label="100% data, one seed per size")
    xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    ax.plot(10 ** xs, a * xs**2 + b * xs + c, color="C0",
            label="quadratic fit (log-params)")
    ax.axvline(n_star, color="gray", ls="--",
               label=f"descriptive fit vertex ≈ {n_star/1e6:.0f}M")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Recorded final validation loss")
    ax.set_title("Five-point descriptive fit; validation samples differ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "compute_optimal_fit.png", dpi=150)
    plt.close(fig)

    print("Full-data run per size (one seed; original validation samples differ):")
    for _, r in best.iterrows():
        print(f"  {r['size']:<6} {r['parameters']/1e6:6.1f}M  "
              f"val {r['final_val_loss']:.4f}  C~={r['flops']:.3e} FLOPs")
    print(f"\nQuadratic fit in log10(params): loss = {a:.3f} x^2 + {b:.3f} x + {c:.3f}")
    print(f"Descriptive quadratic vertex, NOT an established optimum: "
          f"N* ≈ {n_star/1e6:.0f}M parameters (10^{x_star:.3f})")
    print(f"Empirical winner: {best.iloc[y.argmin()]['size']} "
          f"({best['parameters'].iloc[y.argmin()]/1e6:.0f}M)")
    print(f"\nWrote {args.out_dir}/loss_vs_compute.png and compute_optimal_fit.png")


if __name__ == "__main__":
    main()
