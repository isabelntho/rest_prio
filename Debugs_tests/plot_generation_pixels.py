# =============================================================================
# plot_generation_pixels.py - small-multiple grid of one individual's selected
# restoration pixels across generations, from a save_snapshots=True run.
#
# Requires an X_history_*.npz (outputs/intermediate_results/), which only
# exists when the run was launched with save_snapshots=True. It has key "X",
# shape (n_gens, pop_size, n_var), where columns [:n_restoration_pixels] line
# up 1:1 (same order) with eligible_pixels.csv in the run's r_inputs export.
#
# Two X_history shapes are handled:
#   - sparse:  n_gens == len(GENERATIONS) -> the run used
#              snapshot_generations={1, 11, 26, 51, 101, 151} (optimization_engine.py),
#              so each row already IS one of GENERATIONS, in order.
#   - dense:   n_gens > len(GENERATIONS)  -> every generation was snapshotted,
#              so row index == generation number (0-indexed).
#
# USAGE (from project root):
#   pixi run python Debugs_tests/plot_generation_pixels.py <run_dir> <x_history_npz>
#
# Example:
#   pixi run python Debugs_tests/plot_generation_pixels.py \
#       outputs/r_inputs/<run_label> \
#       outputs/intermediate_results/X_history_<timestamp>.npz
# =============================================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GENERATIONS = [1, 11, 26, 51]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="r_inputs export dir for the run (has eligible_pixels.csv)")
    ap.add_argument("x_history", help="X_history_*.npz path")
    ap.add_argument("--individual", type=int, default=0, help="population index to track (default 0)")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    coords = pd.read_csv(run_dir / "eligible_pixels.csv")
    x, y = coords["x"].to_numpy(), coords["y"].to_numpy()
    n_restore = len(coords)

    X = np.load(args.x_history)["X"]  # (n_gens, pop_size, n_var)
    n_gens = X.shape[0]

    # Pairing the wrong npz with a run folder is silent otherwise: the slice below
    # just takes the first n_restore columns and maps them onto foreign coordinates.
    if X.shape[2] < n_restore:
        raise SystemExit(
            f"{args.x_history} has {X.shape[2]} decision variables but {run_dir} lists "
            f"{n_restore} eligible pixels - they are not from the same run.")

    if n_gens == len(GENERATIONS):
        rows, labels = list(range(n_gens)), GENERATIONS
    else:
        rows, labels = [], []
        for g in GENERATIONS:
            r = min(g, n_gens - 1)
            if r not in rows:
                rows.append(r)
                labels.append(g if r == g else f"{g} (clamped to {r})")

    fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 3.5),
                              sharex=True, sharey=True)
    if len(rows) == 1:
        axes = [axes]

    for ax, row, label in zip(axes, rows, labels):
        selected = X[row, args.individual, :n_restore] == 1
        ax.scatter(x[selected], y[selected], s=1, c="#E05C2A", marker="s", linewidths=0)
        ax.set_title(f"Gen {label} (n={selected.sum():,})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Selected restoration pixels - individual {args.individual} - {run_dir.name}")
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        Path("outputs/figs/quick_look") / f"{run_dir.name}_gen_grid_ind{args.individual}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
