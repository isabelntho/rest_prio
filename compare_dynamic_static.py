"""
Compare dynamic (timing-based) vs static restoration optimisation results.

Usage
-----
    pixi run python compare_dynamic_static.py

The script auto-discovers the most recent static and dynamic result pickles in
``results_files/``.  Edit the CONFIGURATION section below to point at specific
files or to change figure output paths.

Plots produced
--------------
1. Parallel coordinates — non-dominated front for each dynamic scenario
2. Cost vs RP scatter — static Pareto vs each dynamic scenario (normalised)
3. Timing distribution — how patches are spread across time steps (dynamic only)
4. Summary table printed to console
"""

from __future__ import annotations

import os
import pickle
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# CONFIGURATION — edit paths here, or leave as None for auto-discovery
# ---------------------------------------------------------------------------

RESULTS_DIR = "results_files"

# Static pkl — None = use most recent non-dynamic file
STATIC_PKL: str | None = None

# Dynamic pkls — None = use most recent file per recovery scenario
DYNAMIC_PKLS: dict[str, str | None] = {
    "fast":    None,
    "gradual": None,
    "delayed": None,
    "partial": None,
}

# Output directory for figures (None = show interactively only)
FIG_DIR: str | None = "figs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _latest_file(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def _normalise_cols(F: np.ndarray) -> np.ndarray:
    """Min-max normalise each column of F to [0, 1]."""
    F = F.astype(float)
    out = np.zeros_like(F)
    for j in range(F.shape[1]):
        lo, hi = F[:, j].min(), F[:, j].max()
        out[:, j] = (F[:, j] - lo) / (hi - lo) if hi > lo else 0.5
    return out


def _savefig(fig: plt.Figure, name: str) -> None:
    if FIG_DIR:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Auto-discover files
# ---------------------------------------------------------------------------

def discover_files() -> tuple[str, dict[str, str]]:
    # Static
    static_path = STATIC_PKL or _latest_file(
        os.path.join(RESULTS_DIR, "res_*.pkl")
    )
    if static_path is None:
        raise FileNotFoundError(f"No static result pkl found in {RESULTS_DIR}/")

    # Dynamic — one per scenario
    dyn_paths: dict[str, str] = {}
    for scenario, override in DYNAMIC_PKLS.items():
        if override:
            dyn_paths[scenario] = override
        else:
            found = _latest_file(
                os.path.join(RESULTS_DIR, f"dynamic_dynamic_{scenario}_*.pkl")
            )
            if found:
                dyn_paths[scenario] = found

    return static_path, dyn_paths


# ---------------------------------------------------------------------------
# Load and normalise results
# ---------------------------------------------------------------------------

def load_static(path: str) -> dict:
    """Load static pkl, return dict with standardised keys."""
    r = _load_pkl(path)
    # Static pkls are single-scenario — no 'scenarios' wrapper
    obj_names = r.get("objective_names", [])
    F = np.asarray(r["objectives"], dtype=float)
    nd = np.asarray(r.get("is_nondominated", np.ones(len(F), dtype=bool)))
    return {
        "label": "static",
        "objective_names": obj_names,
        "F": F,
        "F_nd": F[nd],
        "is_nondominated": nd,
        "n_solutions": len(F),
        "n_nd": int(nd.sum()),
        "path": path,
    }


def load_dynamic(path: str, scenario_name: str) -> dict:
    """Load dynamic pkl, return dict with standardised keys."""
    r = _load_pkl(path)
    obj_names = r.get("objective_names", [])
    F = np.asarray(r["objectives"], dtype=float)
    nd = np.asarray(r.get("is_nondominated", np.ones(len(F), dtype=bool)))
    return {
        "label": scenario_name,
        "objective_names": obj_names,
        "F": F,
        "F_nd": F[nd],
        "is_nondominated": nd,
        "n_solutions": len(F),
        "n_nd": int(nd.sum()),
        "time_steps": r.get("time_steps"),
        "decisions": np.asarray(r.get("decisions", [])),
        "per_step_max_pixels": r.get("per_step_max_pixels"),
        "path": path,
    }


# ---------------------------------------------------------------------------
# Plot 1 — Parallel coordinates for all dynamic scenarios
# ---------------------------------------------------------------------------

_SCENARIO_COLORS = {
    "fast":    "#e41a1c",
    "gradual": "#377eb8",
    "delayed": "#4daf4a",
    "partial": "#984ea3",
}


def plot_parallel_coords_dynamic(dyn_results: dict[str, dict]) -> plt.Figure:
    """
    One subplot per recovery scenario; non-dominated solutions only.
    Objectives: final_landscape_rp | cumulative_landscape_rp | total_cost
    """
    n = len(dyn_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    raw_labels = ["final_landscape_rp", "cumulative_landscape_rp", "final_landscape_context", "total_cost"]
    display_labels = ["Final\nLandscape RP\n(↓ more gain)", "Cumulative\nLandscape RP\n(↓ more)", "Final\nLandscape Ctx\n(↓ more)", "Total\nCost (↓ less)"]
    n_obj = len(raw_labels)
    x = np.arange(n_obj)

    def _orient(F: np.ndarray) -> np.ndarray:
        """All objectives stored as minimise-is-better; no flip needed."""
        return F.copy()

    # Check there is something to plot
    if not any(len(v["F_nd"]) for v in dyn_results.values()):
        plt.close(fig)
        return None

    for ax, (scenario, res) in zip(axes, dyn_results.items()):
        F_nd = _orient(res["F_nd"])
        if len(F_nd) == 0:
            ax.set_title(f"{scenario}\n(no ND solutions)")
            continue

        color = _SCENARIO_COLORS.get(scenario, "steelblue")

        # Per-scenario normalisation: reveals within-scenario trade-offs.
        # (Global normalisation squashes scenarios with different cost scales.)
        F_all_oriented = _orient(res["F"])
        s_min = F_all_oriented.min(axis=0)
        s_max = F_all_oriented.max(axis=0)

        def _norm(M, lo=s_min, hi=s_max):
            out = np.zeros_like(M, dtype=float)
            for j in range(n_obj):
                out[:, j] = (M[:, j] - lo[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else 0.5
            return out

        F_norm = _norm(F_nd)
        F_all_norm = _norm(F_all_oriented)

        nd_mask = res["is_nondominated"]
        for i in range(len(res["F"])):
            if not nd_mask[i]:
                ax.plot(x, F_all_norm[i], color="lightgrey", lw=0.6, zorder=1)

        # Non-dominated — highlighted
        for i in range(len(F_norm)):
            ax.plot(x, F_norm[i], color=color, alpha=0.7, lw=1.2, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Normalised value (per-scenario)" if ax == axes[0] else "")
        ax.set_title(
            f"{scenario}\n{res['n_nd']} ND / {res['n_solutions']} total",
            fontsize=10, color=color,
        )
        ax.grid(True, axis="y", alpha=0.3, ls="--")

    fig.suptitle(
        "Dynamic optimisation — parallel coordinates by recovery scenario\n"
        "(per-scenario normalisation; lower = better on all axes; ctx improvement negated)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Cost vs RP: static Pareto vs dynamic scenarios
# ---------------------------------------------------------------------------

def plot_cost_vs_rp(static: dict, dyn_results: dict[str, dict]) -> plt.Figure:
    """
    Scatter of normalised cost (x) vs normalised RP proxy (y) for all runs.

    Static proxies used:
        RP proxy  = -(abiotic_anomaly + biotic_anomaly) / 2  (flip sign → higher = better)
        Cost      =  implementation_cost

    Dynamic equivalents:
        RP proxy  = -final_landscape_rp   (flip sign; stored negative)
        Cost      =  total_cost
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Static ---
    F_s = static["F"]
    names_s = static["objective_names"]

    cost_col_s = next(
        (i for i, n in enumerate(names_s) if "cost" in n.lower()), None
    )
    abiotic_col = next(
        (i for i, n in enumerate(names_s) if "abiotic" in n.lower()), None
    )
    biotic_col = next(
        (i for i, n in enumerate(names_s) if "biotic" in n.lower()), None
    )

    if cost_col_s is not None and abiotic_col is not None and biotic_col is not None:
        cost_s = F_s[:, cost_col_s]
        rp_s = -(F_s[:, abiotic_col] + F_s[:, biotic_col]) / 2.0  # flip: higher = more degraded = more gain
        nd_s = static["is_nondominated"]

        cost_s_norm = (cost_s - cost_s.min()) / (cost_s.max() - cost_s.min() + 1e-12)
        rp_s_norm = (rp_s - rp_s.min()) / (rp_s.max() - rp_s.min() + 1e-12)

        ax.scatter(
            cost_s_norm[~nd_s], rp_s_norm[~nd_s],
            c="lightgrey", s=25, alpha=0.5, label="Static (dominated)", zorder=1,
        )
        ax.scatter(
            cost_s_norm[nd_s], rp_s_norm[nd_s],
            c="black", s=50, marker="D", alpha=0.8,
            label=f"Static ND ({nd_s.sum()})", zorder=3,
        )

    # --- Dynamic scenarios ---
    for scenario, res in dyn_results.items():
        F_d = res["F"]
        names_d = res["objective_names"]
        nd_d = res["is_nondominated"]

        cost_col_d = next(
            (i for i, n in enumerate(names_d) if "cost" in n.lower()), None
        )
        rp_col_d = next(
            (i for i, n in enumerate(names_d) if n == "final_landscape_rp"), None
        )
        if cost_col_d is None or rp_col_d is None:
            continue

        cost_d = F_d[:, cost_col_d]
        rp_d = -F_d[:, rp_col_d]  # flip: more negative = more potential gain

        cost_d_norm = (cost_d - cost_d.min()) / (cost_d.max() - cost_d.min() + 1e-12)
        rp_d_norm = (rp_d - rp_d.min()) / (rp_d.max() - rp_d.min() + 1e-12)

        color = _SCENARIO_COLORS.get(scenario, "steelblue")
        ax.scatter(
            cost_d_norm[~nd_d], rp_d_norm[~nd_d],
            c=color, s=20, alpha=0.3, zorder=2,
        )
        ax.scatter(
            cost_d_norm[nd_d], rp_d_norm[nd_d],
            c=color, s=60, edgecolors="white", linewidths=0.5,
            label=f"Dynamic {scenario} ND ({nd_d.sum()})", zorder=4,
        )

    ax.set_xlabel("Normalised implementation cost (per-run range)", fontsize=11)
    ax.set_ylabel("Normalised RP gain proxy (per-run range)", fontsize=11)
    ax.set_title(
        "Cost vs RP gain: static (diamond) vs dynamic scenarios\n"
        "(each run normalised independently — scales are NOT directly comparable)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Timing distribution across time steps (dynamic only)
# ---------------------------------------------------------------------------

def plot_timing_distribution(dyn_results: dict[str, dict]) -> plt.Figure:
    """
    For each scenario, show the distribution of patches per time step across
    all non-dominated solutions.
    """
    n = len(dyn_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (scenario, res) in zip(axes, dyn_results.items()):
        decisions = res.get("decisions")  # (n_solutions, n_patches) int 0..n_steps
        nd = res["is_nondominated"]
        time_steps = res.get("time_steps", [])

        if decisions is None or len(decisions) == 0 or not any(nd):
            ax.set_title(f"{scenario}\n(no data)")
            continue

        D_nd = decisions[nd].astype(int)
        n_steps = int(D_nd.max()) if D_nd.max() > 0 else len(time_steps)
        step_labels = [str(t) for t in time_steps[:n_steps]] if time_steps else [f"Step {k}" for k in range(1, n_steps + 1)]

        # Fraction of patches assigned to each time step (0 = never)
        counts = np.array([
            [(row == k).sum() for k in range(n_steps + 1)]
            for row in D_nd
        ], dtype=float)  # (n_nd, n_steps+1)
        n_patches = D_nd.shape[1]
        counts /= n_patches  # fraction

        # Plot mean ± std across solutions
        never_frac = counts[:, 0]
        step_fracs = counts[:, 1:]

        color = _SCENARIO_COLORS.get(scenario, "steelblue")
        positions = np.arange(n_steps)

        for k in range(n_steps):
            vals = step_fracs[:, k]
            ax.bar(
                k, vals.mean(),
                yerr=vals.std(), color=color, alpha=0.7,
                capsize=4, width=0.6,
            )
            ax.text(k, vals.mean() + vals.std() + 0.002, f"{vals.mean():.1%}",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(positions)
        ax.set_xticklabels(step_labels, fontsize=9)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Fraction of patches" if ax == axes[0] else "")
        never_mean = never_frac.mean()
        ax.set_title(
            f"{scenario}\n{never_mean:.1%} patches never restored",
            fontsize=9, color=color,
        )
        ax.set_ylim(0, min(1.0, step_fracs.mean(axis=0).max() * 2.5 + 0.1))
        ax.grid(True, axis="y", alpha=0.3, ls="--")

    fig.suptitle(
        "Dynamic — patch timing distribution (mean ± SD across ND solutions)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(static: dict, dyn_results: dict[str, dict]) -> None:
    print("\n" + "=" * 68)
    print(f"{'Run':<25} {'ND':>5} {'Total':>7}  Objectives")
    print("-" * 68)

    # Static
    names_s = " | ".join(static["objective_names"])
    F_nd_s = static["F_nd"]
    cost_s = F_nd_s[:, -1] if len(F_nd_s) else np.array([np.nan])
    print(
        f"{'static':<25} {static['n_nd']:>5} {static['n_solutions']:>7}  "
        f"cost [{cost_s.min():.2e}–{cost_s.max():.2e}]"
    )

    # Dynamic
    for scenario, res in dyn_results.items():
        F_nd_d = res["F_nd"]
        names_d = res["objective_names"]
        cost_col = next(
            (i for i, n in enumerate(names_d) if "cost" in n.lower()), -1
        )
        cost_d = F_nd_d[:, cost_col] if len(F_nd_d) and cost_col >= 0 else np.array([np.nan])
        ts = res.get("time_steps")
        ts_str = f"  steps={ts}" if ts else ""
        print(
            f"{'dynamic_' + scenario:<25} {res['n_nd']:>5} {res['n_solutions']:>7}  "
            f"cost [{cost_d.min():.2e}–{cost_d.max():.2e}]{ts_str}"
        )

    print("=" * 68)
    print(f"Static objectives : {static['objective_names']}")
    if dyn_results:
        first = next(iter(dyn_results.values()))
        print(f"Dynamic objectives: {first['objective_names']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    static_path, dyn_paths = discover_files()

    print(f"Static  : {static_path}")
    for s, p in dyn_paths.items():
        print(f"Dynamic [{s}]: {p}")

    static = load_static(static_path)
    dyn_results = {
        scenario: load_dynamic(path, scenario)
        for scenario, path in dyn_paths.items()
    }

    print_summary(static, dyn_results)

    if not dyn_results:
        print("No dynamic results found. Exiting.")
        return

    print("Plotting…")

    fig1 = plot_parallel_coords_dynamic(dyn_results)
    if fig1:
        _savefig(fig1, "dynamic_parallel_coords.png")

    fig2 = plot_cost_vs_rp(static, dyn_results)
    _savefig(fig2, "cost_vs_rp_comparison.png")

    fig3 = plot_timing_distribution(dyn_results)
    _savefig(fig3, "dynamic_timing_distribution.png")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
