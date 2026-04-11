#!/usr/bin/env python3
"""
plot_comparison.py — Active SLAM: Plot from REAL Gazebo Run Data
================================================================
Reads the CSV log files produced by your actual baseline and RL runs
and plots them. Makes NO assumptions about performance ordering.

Each baseline/RL script already saves a CSV to logs/ with this schema:
    step, timestamp, coverage, known_cells, total_cells, cov_trace,
    pos_x, pos_y, altitude, frontier_count, nearest_frontier_dist,
    loop_closures, tracking_lost_events, new_cells, note

How to use
----------
1. Run each exploration script on Gazebo. Each one saves a CSV to logs/:
     - baseline_frontier.py        -> logs/baseline_frontier_TIMESTAMP.csv
     - baseline_random_walk.py     -> logs/baseline_random_walk_TIMESTAMP.csv
     - baseline_spiral.py          -> logs/baseline_spiral_TIMESTAMP.csv
     - baseline_potential_field.py -> logs/baseline_potential_field_TIMESTAMP.csv
     - For RL eval: save CSV with prefix  rl_  (e.g. rl_eval_TIMESTAMP.csv)

2. Run this script:
     python3 plot_comparison.py

   Or point at specific files / glob patterns:
     python3 plot_comparison.py \\
         --rl          logs/rl_eval_20240101.csv \\
         --frontier    logs/baseline_frontier_20240101.csv \\
         --random      logs/baseline_random_walk_20240101.csv \\
         --spiral      logs/baseline_spiral_20240101.csv \\
         --potential   logs/baseline_potential_field_20240101.csv \\
         --output      plots/comparison.png

Multiple runs of the same method (for error bars)
--------------------------------------------------
Pass multiple CSVs per method — the script plots mean +/- std:
    python3 plot_comparison.py --rl logs/rl_run1.csv logs/rl_run2.csv logs/rl_run3.csv

The script plots ONLY the methods for which CSV files were found.
It will clearly tell you which ones are missing.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# ─────────────────────────── visual config ───────────────────────────────────

PALETTE = {
    "RL (RecurrentPPO)":       "#2563EB",
    "Nearest Frontier":        "#16A34A",
    "Potential Field (ITPF)":  "#D97706",
    "Spiral / Lawnmower":      "#7C3AED",
    "Random Walk":             "#DC2626",
}
LINE_STYLE = {
    "RL (RecurrentPPO)":       "-",
    "Nearest Frontier":        "--",
    "Potential Field (ITPF)":  "-.",
    "Spiral / Lawnmower":      ":",
    "Random Walk":             (0, (3, 1, 1, 1)),
}
LINE_WIDTH = {
    "RL (RecurrentPPO)":       2.8,
    "Nearest Frontier":        2.0,
    "Potential Field (ITPF)":  2.0,
    "Spiral / Lawnmower":      2.0,
    "Random Walk":             1.8,
}

SMOOTH_WIN = 15   # Moving-average window for noisy per-step metrics


# ─────────────────────────── CSV loading ─────────────────────────────────────

def load_method(paths: list, method_name: str):
    """
    Load one or more CSV files for a single method.
    Returns dict with 'mean' DataFrame, optional 'std' DataFrame, 'n_runs'.
    Returns None if no valid files found.
    """
    frames = []
    for pattern in paths:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.isfile(pattern):
            matched = [pattern]
        for p in matched:
            try:
                df = pd.read_csv(p)
                required = {"step", "coverage", "cov_trace", "new_cells",
                            "loop_closures", "tracking_lost_events"}
                missing = required - set(df.columns)
                if missing:
                    print(f"  WARNING [{method_name}]: {p} is missing columns: {missing}")
                    continue
                frames.append(df)
                print(f"  Loaded [{method_name}]: {p}  "
                      f"({len(df)} rows, "
                      f"max_step={df['step'].max()}, "
                      f"final_coverage={df['coverage'].iloc[-1]:.1%})")
            except Exception as e:
                print(f"  ERROR [{method_name}]: Could not read {p}: {e}")

    if not frames:
        return None

    if len(frames) == 1:
        df = frames[0].copy().sort_values("step").drop_duplicates("step")
        df["new_cells_smooth"] = (
            df["new_cells"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
        )
        return {"mean": df, "std": None, "n_runs": 1}

    # Multiple runs: align on common step grid, compute mean +/- std
    numeric_cols = ["coverage", "cov_trace", "new_cells",
                    "loop_closures", "tracking_lost_events", "known_cells"]
    all_steps = sorted(set().union(*[set(f["step"].tolist()) for f in frames]))

    reindexed = []
    for f in frames:
        f = f.sort_values("step").drop_duplicates("step").set_index("step")
        f = f.reindex(all_steps)[numeric_cols].ffill()
        reindexed.append(f)

    stacked = np.stack([r.values for r in reindexed], axis=0)
    mean_df = pd.DataFrame(
        np.nanmean(stacked, axis=0), index=all_steps, columns=numeric_cols
    ).reset_index().rename(columns={"index": "step"})
    std_df = pd.DataFrame(
        np.nanstd(stacked, axis=0), index=all_steps, columns=numeric_cols
    ).reset_index().rename(columns={"index": "step"})

    mean_df["new_cells_smooth"] = (
        mean_df["new_cells"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    )
    std_df["new_cells_smooth"] = (
        std_df["new_cells"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    )
    return {"mean": mean_df, "std": std_df, "n_runs": len(frames)}


# ─────────────────────────── plot helpers ────────────────────────────────────

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1E293B")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, color="#94A3B8", fontsize=8.5)
    ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8.5)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(color="#334155", linewidth=0.5, alpha=0.7)


def _plot(ax, data, col, name, smooth=False):
    mean_df = data["mean"]
    std_df  = data["std"]
    color   = PALETTE[name]
    x       = mean_df["step"].values
    ycol    = "new_cells_smooth" if (smooth and "new_cells_smooth" in mean_df) else col
    y       = mean_df[ycol].values
    label   = f"{name}  (n={data['n_runs']})" if data["n_runs"] > 1 else name
    ax.plot(x, y, color=color, lw=LINE_WIDTH[name], ls=LINE_STYLE[name], label=label)
    if std_df is not None:
        ys = std_df[ycol].values
        ax.fill_between(x, y - ys, y + ys, color=color, alpha=0.15)


# ─────────────────────────── main figure ─────────────────────────────────────

def make_figure(loaded: dict, output_path: str):
    if not loaded:
        print("ERROR: no data to plot.")
        return

    names = list(loaded.keys())

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0F172A")
    gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    ax_cov   = fig.add_subplot(gs[0, :2])
    ax_bar   = fig.add_subplot(gs[0, 2])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_rate  = fig.add_subplot(gs[1, 1])
    ax_lc    = fig.add_subplot(gs[1, 2])
    ax_sum   = fig.add_subplot(gs[2, :])

    # ── Coverage vs Steps ───────────────────────────────────────────────────
    _style(ax_cov, "Map Coverage (%) vs Exploration Steps  [REAL Gazebo Data]",
           "Step", "Coverage (%)")
    for name, data in loaded.items():
        x = data["mean"]["step"].values
        y = data["mean"]["coverage"].values * 100
        label = f"{name} (n={data['n_runs']})" if data["n_runs"] > 1 else name
        ax_cov.plot(x, y, color=PALETTE[name], lw=LINE_WIDTH[name],
                    ls=LINE_STYLE[name], label=label)
        if data["std"] is not None:
            ys = data["std"]["coverage"].values * 100
            ax_cov.fill_between(x, y - ys, y + ys, color=PALETTE[name], alpha=0.15)
    ax_cov.axhline(90, color="#F1F5F9", lw=0.8, ls="--", alpha=0.4)
    ax_cov.text(2, 91.5, "90% target", color="#F1F5F9", fontsize=7, alpha=0.5)
    ax_cov.set_ylim(0, 105)
    ax_cov.legend(fontsize=8, facecolor="#0F172A", edgecolor="#334155",
                  labelcolor="white", loc="lower right")

    # ── Final Coverage Bar ───────────────────────────────────────────────────
    _style(ax_bar, "Final Coverage\n(last logged step)", "", "Coverage (%)")
    short = {
        "RL (RecurrentPPO)": "RL",
        "Nearest Frontier": "Frontier",
        "Potential Field (ITPF)": "Pot.Field",
        "Spiral / Lawnmower": "Spiral",
        "Random Walk": "Random",
    }
    vals   = [loaded[n]["mean"]["coverage"].iloc[-1] * 100 for n in names]
    colors = [PALETTE[n] for n in names]
    xp     = np.arange(len(names))
    bars   = ax_bar.bar(xp, vals, color=colors, width=0.6, zorder=3)
    ax_bar.set_xticks(xp)
    ax_bar.set_xticklabels([short.get(n, n) for n in names],
                            rotation=20, ha="right", fontsize=7)
    ax_bar.set_ylim(0, 110)
    ax_bar.axhline(90, color="#F1F5F9", lw=0.8, ls="--", alpha=0.4)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color="white", fontsize=7, fontweight="bold")

    # ── Covariance Trace ────────────────────────────────────────────────────
    _style(ax_trace, "SLAM Covariance Trace\n(lower = less localization uncertainty)",
           "Step", "Cov. Trace")
    for name, data in loaded.items():
        _plot(ax_trace, data, "cov_trace", name)

    # ── New Cells / Step ────────────────────────────────────────────────────
    _style(ax_rate, f"Exploration Rate\n({SMOOTH_WIN}-step moving avg of new cells/step)",
           "Step", "New Cells / Step")
    for name, data in loaded.items():
        _plot(ax_rate, data, "new_cells", name, smooth=True)

    # ── Loop Closures ───────────────────────────────────────────────────────
    _style(ax_lc, "Cumulative Loop Closures\n(more = RTAB-Map more self-consistent)",
           "Step", "Loop Closures")
    for name, data in loaded.items():
        _plot(ax_lc, data, "loop_closures", name)

    # ── Summary Table (actual values) ───────────────────────────────────────
    _style(ax_sum, "Summary — Actual Metric Values at End of Run", "Method", "")
    metric_keys = {
        "Final Coverage (%)":    lambda n: loaded[n]["mean"]["coverage"].iloc[-1] * 100,
        "Avg New Cells/Step":    lambda n: loaded[n]["mean"]["new_cells"].mean(),
        "Total Loop Closures":   lambda n: loaded[n]["mean"]["loop_closures"].iloc[-1],
        "Tracking Lost Events":  lambda n: loaded[n]["mean"]["tracking_lost_events"].iloc[-1],
    }
    n_g  = len(metric_keys)
    n_b  = len(names)
    bw   = 0.8 / n_b
    xb   = np.arange(n_g)
    for bi, name in enumerate(names):
        mvals = [fn(name) for fn in metric_keys.values()]
        offset = (bi - n_b / 2 + 0.5) * bw
        ax_sum.bar(xb + offset, mvals, width=bw, color=PALETTE[name], label=name, zorder=3)
        for xi, v in zip(xb, mvals):
            ax_sum.text(xi + offset, v * 1.02, f"{v:.1f}",
                        ha="center", va="bottom", color="white",
                        fontsize=6, rotation=45)
    ax_sum.set_xticks(xb)
    ax_sum.set_xticklabels(list(metric_keys.keys()), color="#94A3B8", fontsize=9)
    ax_sum.set_ylabel("Metric Value", color="#94A3B8", fontsize=9)
    ax_sum.legend(fontsize=8, facecolor="#0F172A", edgecolor="#334155",
                  labelcolor="white", ncol=n_b, loc="upper right")
    ax_sum.text(0.99, 0.97, "Note: 'Tracking Lost Events' lower is better",
                transform=ax_sum.transAxes, ha="right", va="top",
                color="#94A3B8", fontsize=7, style="italic")

    fig.text(0.5, 0.972,
             f"Active SLAM — Real Gazebo Results  |  Methods: {', '.join(names)}",
             ha="center", color="white", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.950,
             "Shaded bands = ±1 std across multiple runs (where provided)",
             ha="center", color="#94A3B8", fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[Plot] Saved → {output_path}")


# ─────────────────────────── auto-detect + CLI ───────────────────────────────

def _autodetect(logs_dir: str) -> dict:
    prefix_map = {
        "rl_":                   "RL (RecurrentPPO)",
        "baseline_frontier_":    "Nearest Frontier",
        "baseline_random_walk_": "Random Walk",
        "baseline_spiral_":      "Spiral / Lawnmower",
        "baseline_potential_":   "Potential Field (ITPF)",
    }
    found = {}
    if not os.path.isdir(logs_dir):
        return found
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".csv"):
            continue
        for prefix, name in prefix_map.items():
            if fname.startswith(prefix):
                found.setdefault(name, []).append(os.path.join(logs_dir, fname))
                break
    return found


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot real Gazebo CSVs. Plots only what you actually ran.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rl",        nargs="*", metavar="CSV")
    p.add_argument("--frontier",  nargs="*", metavar="CSV")
    p.add_argument("--random",    nargs="*", metavar="CSV")
    p.add_argument("--spiral",    nargs="*", metavar="CSV")
    p.add_argument("--potential", nargs="*", metavar="CSV")
    p.add_argument("--output",    default="plots/comparison_real.png")
    return p.parse_args()


def main():
    args     = parse_args()
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    cli_paths = {}
    if args.rl:        cli_paths["RL (RecurrentPPO)"]      = args.rl
    if args.frontier:  cli_paths["Nearest Frontier"]        = args.frontier
    if args.random:    cli_paths["Random Walk"]             = args.random
    if args.spiral:    cli_paths["Spiral / Lawnmower"]      = args.spiral
    if args.potential: cli_paths["Potential Field (ITPF)"]  = args.potential

    if not cli_paths:
        print(f"[Plot] No paths given — auto-scanning {logs_dir}/")
        cli_paths = _autodetect(logs_dir)

    if not cli_paths:
        print(
            "\n[Plot] No CSV log files found. Run your exploration scripts first.\n\n"
            "Expected filenames in logs/:\n"
            "  rl_eval_TIMESTAMP.csv\n"
            "  baseline_frontier_TIMESTAMP.csv\n"
            "  baseline_random_walk_TIMESTAMP.csv\n"
            "  baseline_spiral_TIMESTAMP.csv\n"
            "  baseline_potential_field_TIMESTAMP.csv\n"
        )
        sys.exit(1)

    print(f"\n[Plot] Loading {len(cli_paths)} method(s) …\n")
    loaded = {}
    for name, paths in cli_paths.items():
        result = load_method(paths, name)
        if result:
            loaded[name] = result
        else:
            print(f"  SKIP [{name}]: no valid CSV files found.")

    if not loaded:
        print("\n[Plot] No valid data loaded. Nothing to plot.")
        sys.exit(1)

    print(f"\n[Plot] Plotting: {', '.join(loaded.keys())}")
    make_figure(loaded, args.output)


if __name__ == "__main__":
    main()
