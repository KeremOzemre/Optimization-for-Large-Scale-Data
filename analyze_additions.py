"""Figures for the two extra experiments: budget sweep and context length.

Budget sweep: full-data validation loss per size at the one-hour budget (from
analysis/summary.csv) versus the 30-minute budget (runs/{large,xl,xxl}_t1800),
as an exploratory comparison. Hardware labels and validation samples differ,
so the plot cannot isolate a causal budget or context effect.

Usage:
    python analyze_additions.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("analysis")
PARAMS = {"small": 16.08, "medium": 30.02, "large": 51.04, "xl": 123.75, "xxl": 253.09}


def final_val(run):
    m = pd.read_csv(f"runs/{run}/metrics.csv")
    rows = m[m["phase"] == "final"] if "phase" in m else m
    row = rows.iloc[-1] if not rows.empty else m.iloc[-1]
    if pd.notna(row.get("val_loss")):
        return float(row["val_loss"])
    return float(m["val_loss"].dropna().iloc[-1])


# Compare the same full-data fraction at both budgets.
summary = pd.read_csv(OUT / "summary.csv")
hour = summary[summary["data_pct"] == 100].set_index("size")["final_val_loss"]
hour = {s: hour[s] for s in PARAMS if s in hour.index}

# 30-minute budget: the three sizes we reran.
half = {s: final_val(f"{s}_t1800") for s in ("large", "xl", "xxl")}

fig, ax = plt.subplots(figsize=(8, 5))
for label, data, color in [("1 h budget", hour, "C0"), ("30 min budget", half, "C3")]:
    xs = [PARAMS[s] for s in data]
    ys = [data[s] for s in data]
    pairs = sorted(zip(xs, ys))
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    ax.plot(xs, ys, marker="o", color=color, label=label)
    best_s = min(data, key=data.get)
    ax.scatter([PARAMS[best_s]], [data[best_s]], s=180, facecolors="none",
               edgecolors=color, linewidths=2, zorder=4)
    ax.annotate(f"best: {best_s}", (PARAMS[best_s], data[best_s]),
                textcoords="offset points", xytext=(6, 10), color=color)
ax.set_xscale("log")
ax.set_xlabel("Parameters (millions)")
ax.set_ylabel("Recorded final validation loss")
ax.set_title("Exploratory budgets: hardware labels / validation samples differ")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "budget_sweep.png", dpi=150)
plt.close(fig)

print("1 h budget (100% data):", {k: round(v, 4) for k, v in hour.items()})
print("30 min budget (val):          ", {k: round(v, 4) for k, v in half.items()})
print(f"  best @1h  = {min(hour, key=hour.get)}")
print(f"  best @30m = {min(half, key=half.get)}")
print("\nContext length (xl, 1 h): ctx256 val", round(hour['xl'], 4),
      "-> ctx512 val", round(final_val('xl_ctx512'), 4))
print(f"\nWrote {OUT}/budget_sweep.png")
