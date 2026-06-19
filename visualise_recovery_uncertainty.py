"""
Recovery Uncertainty Visualisation
====================================
Produces 4 panels illustrating how recovery-function uncertainty propagates
from ecological assumption → trade-off surface → spatial decision → timing:

  Panel 1  Recovery curves         — what the 4 uncertainty scenarios assume
  Panel 2  Pareto scatter          — how RP-gain vs cost trade-offs shift
  Panel 3  Spatial agreement map   — which patches are robust vs sensitive
  Panel 4  Timing distribution     — when optimal timing shifts

Usage
-----
    pixi run python visualise_recovery_uncertainty.py

Output saved to figs/ (overview + 4 individual panels).
"""

from __future__ import annotations

import os
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration  — mirrors run_dynamic.py
# ---------------------------------------------------------------------------
RESULTS_DIR    = "results_files"
FIG_DIR        = "figs"

# Initial-conditions loading (must match the dynamic run settings)
ECOSYSTEM      = "combined"       # "combined" → ecosystem_for_loader = "all"
REGION         = "Bern"
CONDITION_SCE  = "global_all"
AGGREGATION    = 2
PATCH_SIZE     = 2
OBJECTIVES     = ["abiotic", "biotic", "cost"]

SCENARIO_COLORS = {
    "fast":    "#e41a1c",
    "gradual": "#377eb8",
    "delayed": "#4daf4a",
    "partial": "#984ea3",
}
SCENARIO_LABELS = {
    "fast":    "Fast",
    "gradual": "Gradual",
    "delayed": "Delayed",
    "partial": "Partial",
}

FORCE_DIRS = None
# Set FORCE_DIRS to a list of pkl paths to use specific files, e.g.:
#FORCE_DIRS = [
#     "results_files\\dynamic_dynamic_gradual_seed104_20260529_211734.pkl",
#     "results_files\\dynamic_dynamic_fast_seed104_20260529_195605.pkl",
#     "results_files\\dynamic_dynamic_partial_seed104_20260530_000033.pkl",
#     "results_files\\dynamic_dynamic_delayed_seed104_20260529_223904.pkl",
#]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_file(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def _load(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _savefig(fig, name: str, individual: bool = False) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if individual:
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dynamic_results() -> tuple[dict[str, dict], dict]:
    """Load the most recent pkl per recovery scenario.

    Returns (results, run_meta) where run_meta contains aggregation_factor
    and patch_size read from the stored run_config so spatial visualisation
    uses the exact same settings as the optimisation run.
    """
    results = {}
    run_meta = {"aggregation_factor": AGGREGATION, "patch_size": PATCH_SIZE}  # fallback defaults

    # Build a scenario→path map from FORCE_DIRS if provided
    forced_map: dict[str, str] = {}
    if FORCE_DIRS is not None:
        for p in FORCE_DIRS:
            basename = os.path.basename(p)
            for scen in SCENARIO_COLORS:
                if f"_{scen}_" in basename or f"_{scen}." in basename:
                    forced_map[scen] = p
                    break

    for scenario in SCENARIO_COLORS:
        if FORCE_DIRS is not None:
            path = forced_map.get(scenario)
            if path is None:
                print(f"  WARNING: no FORCE_DIRS entry matched scenario '{scenario}'")
                continue
        else:
            path = _latest_file(
                os.path.join(RESULTS_DIR, f"dynamic_dynamic_{scenario}_*.pkl")
            )
        if path is None:
            print(f"  WARNING: no pkl found for scenario '{scenario}'")
            continue
        r = _load(path)
        F = np.asarray(r["objectives"], dtype=float)
        nd = np.asarray(r.get("is_nondominated", np.ones(len(F), dtype=bool)))
        decisions = np.asarray(r.get("decisions", []))
        results[scenario] = {
            "F": F,
            "F_nd": F[nd],
            "nd": nd,
            "decisions": decisions,       # (n_solutions, n_patches) integer 0..n_steps
            "time_steps": r.get("time_steps", [2025, 2030, 2040, 2050]),
            "n_nd": int(nd.sum()),
            "path": path,
        }
        # Extract run settings from the first pkl we can read
        if "run_config" in r and "_set" not in run_meta:
            cfg = r["run_config"]
            run_meta["aggregation_factor"] = cfg.get("aggregation_factor", AGGREGATION)
            run_meta["patch_size"] = cfg.get("patch_size", PATCH_SIZE)
            run_meta["_set"] = True
            cfg = r["run_config"]
            run_meta["aggregation_factor"] = cfg.get("aggregation_factor", AGGREGATION)
            run_meta["patch_size"] = cfg.get("patch_size", PATCH_SIZE)
            run_meta["_set"] = True
        prob = r.get("problem_info", {})
        print(
            f"  [{scenario:8s}] {int(nd.sum()):3d} ND / {len(F)} total — "
            f"{os.path.basename(path)} "
            f"(agg={run_meta['aggregation_factor']}, "
            f"{prob.get('n_restoration_patches', decisions.shape[1] if decisions.ndim==2 else '?')} patches)"
        )
    return results, run_meta


def load_static_result() -> dict | None:
    path = _latest_file(os.path.join(RESULTS_DIR, "res_*.pkl"))
    if path is None:
        print("  WARNING: no static pkl found")
        return None
    r = _load(path)
    F = np.asarray(r["objectives"], dtype=float)
    nd = np.asarray(r.get("is_nondominated", np.ones(len(F), dtype=bool)))
    # Static objectives: [abiotic_anomaly, biotic_anomaly, implementation_cost]
    # Use col 2 (cost) and derive RP gain proxy as -(abiotic+biotic)/2
    return {"F": F, "F_nd": F[nd], "nd": nd, "path": path}


def load_initial_conditions_for_spatial(aggregation_factor: int = AGGREGATION,
                                         patch_size: int = PATCH_SIZE):
    """Load initial conditions matching the aggregation and patch settings of the pkl files."""
    if sys.path and sys.path[0] != ".":
        sys.path.insert(0, ".")
    from Core_optimisation.data_loader import load_initial_conditions
    from Core_optimisation.patch_approach import create_patch_mappings

    ecosystem_for_loader = "all" if ECOSYSTEM == "combined" else ECOSYSTEM
    print(f"  Loading initial conditions (agg={aggregation_factor}, patch_size={patch_size}) …")
    ic = load_initial_conditions(
        workspace_dir=".",
        objectives=OBJECTIVES,
        region=REGION,
        ecosystem=ecosystem_for_loader,
        aggregation_factor=aggregation_factor,
        condition_scenario=CONDITION_SCE,
    )
    pm = create_patch_mappings(ic, patch_size=patch_size)
    ic["patch_mappings"] = pm
    print(f"  {ic['n_restoration_pixels']:,} restoration pixels, "
          f"{pm['restoration_patches']['n_patches']:,} patches")
    return ic


# ---------------------------------------------------------------------------
# Panel 1 — Recovery curves
# ---------------------------------------------------------------------------

def _fast_recovery(t):      return 1.0 - np.exp(-t / 5.0)
def _gradual_recovery(t):   return np.clip(t / 35.0, 0.0, 1.0)
def _delayed_recovery(t):   return 1.0 / (1.0 + np.exp(-0.5 * (t - 20.0)))
def _partial_recovery(t):   return 0.6 * (1.0 - np.exp(-t / 12.0))

RECOVERY_FNS = {
    "fast":    _fast_recovery,
    "gradual": _gradual_recovery,
    "delayed": _delayed_recovery,
    "partial": _partial_recovery,
}

TIME_AXIS_LABELS = {2025: "2025", 2030: "2030", 2040: "2040", 2050: "2050"}
BASE_YEAR = 2025


def plot_recovery_curves(ax=None) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.get_figure()

    t = np.linspace(0, 35, 350)
    # Year x-axis ticks correspond to years since 2025 restore
    step_years = [2025, 2030, 2040, 2050]

    for scenario, fn in RECOVERY_FNS.items():
        y = fn(t)
        ax.plot(t, y, color=SCENARIO_COLORS[scenario], lw=2.2,
                label=SCENARIO_LABELS[scenario])

    # Vertical dotted lines at planning years
    for yr in step_years:
        elapsed = yr - BASE_YEAR
        ax.axvline(elapsed, color="grey", lw=0.7, ls=":", alpha=0.8)

    # Partial cap annotation
    #ax.axhline(0.6, color=SCENARIO_COLORS["partial"], lw=0.8, ls="--", alpha=0.5)
    #ax.text(30.5, 0.6, "60% cap", fontsize=7, color=SCENARIO_COLORS["partial"], va="center")

    ax.set_xlim(0, 35)
    ax.set_ylim(0, 1.08)
    #remove axis ticks
    ax.set_yticks([])
    ax.set_xlabel("Time (years)", fontsize=9)
    ax.set_ylabel("Ecosystem condition recovered", fontsize=9)
    #ax.set_title("(a) Recovery assumptions", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    #ax.grid(True, alpha=0.25, ls="--")

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 2 — Pareto scatter: RP gain vs total cost
# ---------------------------------------------------------------------------

def plot_pareto_scatter(dyn_results: dict, static_result: dict | None, ax=None) -> plt.Figure:
    """RP gain vs cost, dynamic scenarios only (static uses different objective units)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.get_figure()

    # Dynamic: F cols = [final_landscape_rp, cumulative_landscape_rp, final_landscape_context, total_cost]
    # Cumulative RP gain = -cumulative_landscape_rp (stored negative; more negative = more gain)

    # Global min/max across all scenarios for normalisation
    all_rp   = np.concatenate([-res["F"][:, 1] for res in dyn_results.values()])
    all_cost = np.concatenate([ res["F"][:, 3] for res in dyn_results.values()])
    rp_min, rp_max     = all_rp.min(),   all_rp.max()
    cost_min, cost_max = all_cost.min(), all_cost.max()

    def _norm_rp(v):   return (v - rp_min)   / (rp_max   - rp_min)   if rp_max   > rp_min   else v * 0
    def _norm_cost(v): return (v - cost_min) / (cost_max - cost_min) if cost_max > cost_min else v * 0

    for scenario, res in dyn_results.items():
        color = SCENARIO_COLORS[scenario]
        F = res["F"]
        nd = res["nd"]
        rp_gain = _norm_rp(-F[:, 1])
        cost    = _norm_cost(F[:, 3])
        # Dominated — light background dots
        if (~nd).any():
            ax.scatter(cost[~nd], rp_gain[~nd], c=color, s=6, alpha=0.18, zorder=1)
        # ND — solid markers
        ax.scatter(cost[nd], rp_gain[nd], c=color, s=6, alpha=0.18,
                   zorder=3,
                   label=SCENARIO_LABELS[scenario])
        # Connect ND front (sorted by cost)
        #idx = np.argsort(cost[nd])
        #ax.plot(cost[nd][idx], rp_gain[nd][idx], color=color, lw=1.0,
        #        alpha=0.55, zorder=2)

    ax.set_xlabel("Implementation cost", fontsize=9)
    ax.set_ylabel("Cumulative restoration potential", fontsize=9)
    #ax.set_title("(b) Trade-off surface by scenario", fontsize=10, fontweight="bold")
    #ax.legend(fontsize=8, framealpha=0.8, ncol=1)
    ax.grid(True, alpha=0.25, ls="--")
    # Annotate: partial scenario achieves lower absolute gain due to 60% cap
    #ax.annotate("Partial scenario\ncapped at 60% recovery",
    #            xy=(0.02, 0.18), xycoords="axes fraction",
    #            fontsize=7, color=SCENARIO_COLORS["partial"],
    #            ha="left", va="bottom")

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 3 — Spatial agreement map
# ---------------------------------------------------------------------------

def _patch_freq_to_grid(patch_freq: np.ndarray, patch_to_pixels: dict,
                        n_patches: int, rest_eligible_indices: np.ndarray,
                        shape: tuple) -> np.ndarray:
    """
    Map per-patch scalar values to a 2-D global pixel grid.

    patch_to_pixels values are eligible-space indices (0..n_rest_pixels-1).
    rest_eligible_indices converts those to global flat indices.
    """
    n_pixels_flat = shape[0] * shape[1]
    pixel_freq = np.zeros(n_pixels_flat, dtype=float)
    for pid in range(min(n_patches, len(patch_freq))):
        elig_idx = patch_to_pixels.get(pid)
        if elig_idx is not None and len(elig_idx):
            global_idx = rest_eligible_indices[np.asarray(elig_idx, dtype=int)]
            pixel_freq[global_idx] = patch_freq[pid]
    return pixel_freq.reshape(shape)


def compute_spatial_agreement(dyn_results: dict, ic: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two 2-D arrays of shape ic['shape']:
      mean_freq   — mean across scenarios of fraction-of-ND-solutions that select each patch
      std_freq    — std across those scenario frequencies (scenario sensitivity)
    """
    shape = ic["shape"]
    rest_patches = ic["patch_mappings"]["restoration_patches"]
    patch_to_pixels = rest_patches["patch_to_pixels"]
    n_patches = rest_patches["n_patches"]
    rest_eligible_indices = ic["restoration_eligible_indices"]  # global flat indices

    per_scenario_pixel_freq = []

    for scenario, res in dyn_results.items():
        decisions = res["decisions"]  # (n_solutions, n_patches) integer 0..n_steps
        nd_mask = res["nd"]
        dec_nd = decisions[nd_mask]   # ND solutions only

        if len(dec_nd) == 0:
            continue

        # Fraction of ND solutions selecting each patch (>0 = any timing step)
        patch_freq = (dec_nd > 0).mean(axis=0)   # (n_patches,)

        per_scenario_pixel_freq.append(
            _patch_freq_to_grid(patch_freq, patch_to_pixels, n_patches,
                                rest_eligible_indices, shape)
        )

    if not per_scenario_pixel_freq:
        return np.zeros(shape), np.zeros(shape)

    stack = np.stack(per_scenario_pixel_freq, axis=0)  # (n_scenarios, H, W)
    mean_freq = stack.mean(axis=0)
    std_freq  = stack.std(axis=0)
    return mean_freq, std_freq


def compute_spatial_divergence_by_step(dyn_results: dict, ic: dict) -> dict:
    """
    For each scenario, compute per-patch mean timing step (weighted across ND solutions).
    Returns structured data for plot_spatial_divergence_by_step().

    Returns dict:
      {
        "scenarios": { scenario: {"mean_timing": (H,W), "selection_freq": (H,W)} },
        "n_scenarios_selecting": (H,W)   # 0-4, how many scenarios select each patch
        "timing_std": (H,W)              # std of mean_timing across scenarios (for selected patches)
        "shape": (H, W),
        "rest_mask": (H, W) bool,
      }
    """
    shape = ic["shape"]
    rest_patches = ic["patch_mappings"]["restoration_patches"]
    patch_to_pixels = rest_patches["patch_to_pixels"]
    n_patches = rest_patches["n_patches"]
    rest_eligible_indices = ic["restoration_eligible_indices"]  # global flat indices
    n_pixels_flat = shape[0] * shape[1]
    rest_mask = ic["restoration_eligible_mask"]

    scenario_timing  = {}   # scenario → (H,W) float, mean timing step (NaN if unselected)
    scenario_selfreq = {}   # scenario → (H,W) float, fraction of ND solutions selecting patch

    for scenario, res in dyn_results.items():
        dec_nd = res["decisions"][res["nd"]]   # (n_nd, n_patches)
        if len(dec_nd) == 0:
            continue

        # Selection frequency (any step)
        sel_freq = (dec_nd > 0).mean(axis=0)      # (n_patches,)
        # Mean timing step for SELECTED assignments (ignore 0=never)
        timing_sum   = (dec_nd * (dec_nd > 0)).sum(axis=0).astype(float)
        timing_count = (dec_nd > 0).sum(axis=0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_timing = np.where(timing_count > 0, timing_sum / timing_count, np.nan)

        pix_selfreq = np.full(n_pixels_flat, np.nan)
        pix_timing  = np.full(n_pixels_flat, np.nan)

        for pid in range(min(n_patches, len(sel_freq))):
            elig_idx = patch_to_pixels.get(pid)
            if elig_idx is not None and len(elig_idx):
                global_idx = rest_eligible_indices[np.asarray(elig_idx, dtype=int)]
                pix_selfreq[global_idx] = sel_freq[pid]
                pix_timing[global_idx]  = mean_timing[pid]

        scenario_selfreq[scenario] = pix_selfreq.reshape(shape)
        scenario_timing[scenario]  = pix_timing.reshape(shape)

    # Agreement: how many scenarios select each patch (freq > 0.1 threshold)
    THRESH = 0.10
    n_scen_sel = np.zeros(shape, dtype=float)
    timing_stack = []
    for scenario in scenario_selfreq:
        sel = scenario_selfreq[scenario]
        n_scen_sel += np.where(~np.isnan(sel) & (sel > THRESH), 1.0, 0.0)
        t = scenario_timing[scenario].copy()
        t[np.isnan(sel) | (sel <= THRESH)] = np.nan
        timing_stack.append(t)

    # Std of mean timing across scenarios (only where ≥2 scenarios agree)
    timing_arr = np.stack(timing_stack, axis=0)  # (n_scen, H, W)
    with np.errstate(invalid="ignore"):
        timing_std = np.nanstd(timing_arr, axis=0)
    timing_std[n_scen_sel < 2] = np.nan   # suppress where <2 scenarios overlap

    return {
        "scenarios":            scenario_timing,
        "selection_freq":       scenario_selfreq,
        "n_scenarios_selecting": n_scen_sel,
        "timing_std":           timing_std,
        "shape":                shape,
        "rest_mask":            rest_mask,
    }


def plot_spatial_divergence_by_step(
    step_data: dict, rest_mask: np.ndarray
) -> plt.Figure:
    """
    2-row figure.
    Top row (one panel per scenario): colour = mean timing step assigned to each patch
      (1=2025, 2=2030, 3=2040, 4=2050; white = never selected)
    Bottom row: left = agreement count (0–4 scenarios agree), right = timing std map.

    This answers: do scenarios select the SAME patches and differ only in timing,
    or do they select genuinely DIFFERENT spatial areas?
    """
    scenarios = list(step_data["scenarios"].keys())
    n_scen = len(scenarios)
    rest_mask = step_data["rest_mask"]
    shape = step_data["shape"]

    # ---- Figure layout: top row = n_scen maps; bottom = 2 maps + 1 spacer ----
    fig = plt.figure(figsize=(4.2 * n_scen, 9))
    gs_top = gridspec.GridSpec(1, n_scen, top=0.93, bottom=0.52,
                               left=0.03, right=0.92, wspace=0.06)
    gs_bot = gridspec.GridSpec(1, 3, top=0.46, bottom=0.05,
                               left=0.03, right=0.92, wspace=0.12)

    # Timing colormap: 4 discrete levels for steps 1–4
    from matplotlib.colors import BoundaryNorm, ListedColormap
    step_cmap = ListedColormap(["#fee08b", "#fc8d59", "#d73027", "#4d0026"])
    step_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], step_cmap.N)
    step_labels = {1: "2025", 2: "2030", 3: "2040", 4: "2050"}

    # Pre-compute crop box from rest_mask so all panels share the same spatial extent
    rows, cols = np.where(rest_mask)
    pad = 5
    r0 = max(rows.min() - pad, 0);  r1 = min(rows.max() + pad + 1, shape[0])
    c0 = max(cols.min() - pad, 0);  c1 = min(cols.max() + pad + 1, shape[1])
    rest_crop = rest_mask[r0:r1, c0:c1].astype(float)   # 0/1 full restoration area

    def _crop(arr):
        return arr[r0:r1, c0:c1]

    # ---- Top row: per-scenario timing maps ----
    im_timing = None
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs_top[0, col])
        timing = step_data["scenarios"][scenario]
        timing_plot = np.where(rest_mask, timing, np.nan)
        timing_crop = _crop(timing_plot)

        # Grey background = full restoration area
        ax.imshow(rest_crop, cmap="Greys_r", vmin=0, vmax=4,
                  interpolation="nearest", alpha=0.12)

        im_timing = ax.imshow(timing_crop, cmap=step_cmap, norm=step_norm,
                              interpolation="nearest")
        ax.set_title(f"{SCENARIO_LABELS.get(scenario, scenario)}", fontsize=10,
                     fontweight="bold", color=SCENARIO_COLORS.get(scenario, "k"))
        ax.axis("off")

    # Colorbar for timing
    cax_t = fig.add_axes([0.93, 0.52, 0.015, 0.41])
    cb_t = fig.colorbar(im_timing, cax=cax_t, ticks=[1, 2, 3, 4])
    cb_t.ax.set_yticklabels(["2025", "2030", "2040", "2050"], fontsize=8)
    cb_t.set_label("Typical restoration year", fontsize=8)

    # Row label
    fig.text(0.01, 0.72, "Most common\ntiming step\nper scenario",
             ha="left", va="center", fontsize=8, rotation=90, color="dimgrey")

    # ---- Bottom left: agreement count (discrete: grey / blue / green) ----
    ax_agree = fig.add_subplot(gs_bot[0, 0])
    n_sel = step_data["n_scenarios_selecting"]
    n_scen_total = len(scenarios)

    agree_coded = np.full(shape, np.nan)
    agree_coded[rest_mask & (n_sel == 0)] = 0
    agree_coded[rest_mask & (n_sel > 0) & (n_sel < 4)] = 1
    agree_coded[rest_mask & (n_sel >= 4)] = 2

    agree_disc_cmap = ListedColormap(["#aaaaaa", "#4575b4", "#1a9641"])
    agree_disc_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], agree_disc_cmap.N)

    ax_agree.set_facecolor("white")
    ax_agree.imshow(_crop(agree_coded), cmap=agree_disc_cmap, norm=agree_disc_norm,
                    interpolation="nearest")
    ax_agree.set_title("Scenario agreement", fontsize=9, fontweight="bold")
    ax_agree.axis("off")

    from matplotlib.patches import Patch as _Patch
    agree_legend = [
        _Patch(facecolor="#aaaaaa", label="Eligible, not selected"),
        _Patch(facecolor="#4575b4", label=f"Sensitive"),
        _Patch(facecolor="#1a9641", label=f"Robust"),
    ]
    ax_agree.legend(handles=agree_legend, loc="lower right", fontsize=7,
                    framealpha=0.9, edgecolor="grey")

    # ---- Bottom middle: timing std ----
    ax_tstd = fig.add_subplot(gs_bot[0, 1])
    tstd_crop = _crop(np.where(rest_mask, step_data["timing_std"], np.nan))
    ax_tstd.imshow(rest_crop, cmap="Greys_r", vmin=0, vmax=4,
                   interpolation="nearest", alpha=0.12)
    vmax_tstd = 1.5
    im_tstd = ax_tstd.imshow(tstd_crop, cmap="PuBu", vmin=0, vmax=vmax_tstd,
                              interpolation="nearest")
    ax_tstd.set_title("Timing uncertainty:\nstd of timing step across scenarios\n(where ≥2 agree on selecting)",
                      fontsize=9, fontweight="bold")
    ax_tstd.axis("off")
    cb_ts = fig.colorbar(im_tstd, ax=ax_tstd, fraction=0.046, pad=0.04)
    cb_ts.set_label("Std of timing step (step units)", fontsize=8)

    # ---- Bottom right: interpretive legend ----
    ax_leg = fig.add_subplot(gs_bot[0, 2])
    ax_leg.axis("off")
    txt = (
        "Reading the maps\n\n"
        "Top row: Each panel shows where\n"
        "a scenario concentrates restoration\n"
        "and at which planning step.\n\n"
        "Agreement (bottom left):\n"
        " 4 scenarios agree  →  robust priority\n"
        " 1–2 scenarios       →  scenario-sensitive\n\n"
        "Timing uncertainty (bottom middle):\n"
        " Low std  →  scenarios agree on WHEN\n"
        " High std →  scenarios differ on timing\n"
        "  even for patches they all select"
    )
    ax_leg.text(0.05, 0.95, txt, transform=ax_leg.transAxes,
                fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="grey", lw=0.8))

    fig.suptitle(
        "Spatial divergence of restoration decisions across recovery scenarios\n"
        "Top: per-scenario timing assignment   |   Bottom: cross-scenario agreement and timing uncertainty",
        fontsize=11, fontweight="bold", y=0.98,
    )
    return fig


def export_spatial_divergence_tifs(step_data: dict, ic: dict,
                                    out_dir: str = "r_inputs/spatial_divergence") -> None:
    """
    Write n_scenarios_selecting and timing_std as georeferenced GeoTIFFs so the
    two-panel figure can be reproduced in R (see Core_optimisation/plot_spatial_divergence_bottom_panels.R).
    CRS and affine transform are taken from ic, which must contain 'transform' and 'crs'.
    Non-eligible pixels are written as NaN.
    """
    try:
        import rasterio as rio
        from affine import Affine
    except ImportError:
        print("  WARNING: rasterio/affine not available — skipping GeoTIFF export.")
        return

    os.makedirs(out_dir, exist_ok=True)
    rest_mask = step_data["rest_mask"]
    crs       = str(ic.get("crs", "EPSG:2056"))
    transform_raw = ic.get("transform")
    if transform_raw is None:
        print("  WARNING: ic has no 'transform' — skipping GeoTIFF export.")
        return
    affine_t = transform_raw if isinstance(transform_raw, Affine) else Affine(*list(transform_raw)[:6])

    def _write(arr: np.ndarray, fname: str) -> None:
        masked = np.where(rest_mask, arr, np.nan).astype("float32")
        path = os.path.join(out_dir, fname)
        with rio.open(path, "w", driver="GTiff",
                      height=masked.shape[0], width=masked.shape[1],
                      count=1, dtype="float32", crs=crs, transform=affine_t,
                      nodata=float("nan")) as dst:
            dst.write(masked, 1)
        print(f"  Saved: {path}")

    _write(step_data["n_scenarios_selecting"].astype(float), "n_scenarios_selecting.tif")
    _write(step_data["timing_std"],                          "timing_std.tif")


def plot_spatial_divergence_bottom_only(step_data: dict) -> plt.Figure:
    """
    Two-panel figure: scenario robustness (left) and timing SD (right).
    No title. Improvements over the full figure:
      - Robustness: 5 discrete levels (0–4 scenarios) so per-cell agreement
        gradients are visible rather than collapsed into 3 categories.
      - SD: YlOrRd colormap (yellow = agree, red = disagree) with tick labels
        that express values in plain-language planning-step equivalents.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    rest_mask = step_data["rest_mask"]
    shape = step_data["shape"]
    n_sel = step_data["n_scenarios_selecting"]

    rows, cols = np.where(rest_mask)
    pad = 5
    r0 = max(rows.min() - pad, 0);  r1 = min(rows.max() + pad + 1, shape[0])
    c0 = max(cols.min() - pad, 0);  c1 = min(cols.max() + pad + 1, shape[1])

    def _crop(arr):
        return arr[r0:r1, c0:c1]

    rest_crop = rest_mask[r0:r1, c0:c1].astype(float)

    fig, (ax_agree, ax_tstd) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ---- Left: robustness — 5 discrete levels (0-4 scenarios selecting) ----
    agree_coded = np.full(shape, np.nan)
    for v in range(5):
        agree_coded[rest_mask & (n_sel == v)] = v

    agree_cmap = ListedColormap([
        "#d9d9d9",  # 0 — eligible, not selected
        "#c6dbef",  # 1 scenario
        "#6baed6",  # 2 scenarios
        "#2171b5",  # 3 scenarios
        "#1a9641",  # 4 scenarios — robust
    ])
    agree_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], agree_cmap.N)

    ax_agree.set_facecolor("white")
    ax_agree.imshow(_crop(agree_coded), cmap=agree_cmap, norm=agree_norm,
                    interpolation="nearest")
    ax_agree.axis("off")

    legend_elements = [
        Patch(facecolor="#d9d9d9", label="Eligible, not selected"),
        Patch(facecolor="#c6dbef", label="1 scenario"),
        Patch(facecolor="#6baed6", label="2 scenarios"),
        Patch(facecolor="#2171b5", label="3 scenarios"),
        Patch(facecolor="#1a9641", label="4 scenarios (robust)"),
    ]
    ax_agree.legend(handles=legend_elements, loc="lower right", fontsize=8,
                    framealpha=0.95, edgecolor="grey",
                    title="Scenarios selecting patch", title_fontsize=8)
    _add_switzerland_inset(ax_agree)

    # ---- Right: timing SD — YlOrRd so "hot" = high disagreement ----
    tstd_crop = _crop(np.where(rest_mask, step_data["timing_std"], np.nan))
    ax_tstd.set_facecolor("white")
    ax_tstd.imshow(rest_crop, cmap="Greys_r", vmin=0, vmax=4,
                   interpolation="nearest", alpha=0.15)
    im_tstd = ax_tstd.imshow(tstd_crop, cmap="YlOrRd", vmin=0, vmax=1.5,
                              interpolation="nearest")
    ax_tstd.axis("off")

    cb = fig.colorbar(im_tstd, ax=ax_tstd, fraction=0.046, pad=0.04,
                      ticks=[0, 0.5, 1.0, 1.5])
    cb.ax.set_yticklabels(
        ["0\n(full agreement)", "0.5\n(~1 step apart)", "1.0\n(~2 steps apart)", "≥1.5\n(max disagreement)"],
        fontsize=7,
    )
    cb.set_label("Timing disagreement across scenarios", fontsize=8)

    fig.tight_layout()
    return fig


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 5):
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, arr.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def plot_spatial_agreement(n_scenarios_selecting: np.ndarray, rest_mask: np.ndarray,
                           n_total_scenarios: int = 4, ax=None) -> plt.Figure:
    """Discrete 3-category agreement map: grey=eligible+unselected, blue=sensitive, green=robust."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # Recode into 3 categories; NaN = non-eligible → renders as white background
    coded = np.full(n_scenarios_selecting.shape, np.nan)
    coded[rest_mask & (n_scenarios_selecting == 0)] = 0                                                # grey
    coded[rest_mask & (n_scenarios_selecting > 0) & (n_scenarios_selecting < 4)] = 1  # blue
    coded[rest_mask & (n_scenarios_selecting >= 4)] = 2                               # green

    coded_crop = _crop_to_mask(coded, rest_mask)

    cmap = ListedColormap(["#aaaaaa", "#4575b4", "#1a9641"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ax.set_facecolor("white")
    ax.imshow(coded_crop, cmap=cmap, norm=norm, interpolation="nearest")
    #ax.set_title("(c) Scenario agreement", fontsize=10, fontweight="bold")
    ax.axis("off")

    legend_elements = [
        Patch(facecolor="#aaaaaa", label="Not selected"),
        Patch(facecolor="#4575b4", label=f"Sensitive"),
        Patch(facecolor="#1a9641", label=f"Robust"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9,
              edgecolor="grey")

    # --- Inset: Switzerland with Bern highlighted ---
    _add_switzerland_inset(ax)

    if standalone:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
    return fig


def _add_switzerland_inset(ax):
    """Add a small locator inset (top-right) showing Switzerland with Bern highlighted."""
    try:
        import geopandas as gpd
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ch_path   = r"Y:\EU_BioES_SELINA\WP3\4. Spatially_Explicit_EC\Data\CH_shps\swissBOUNDARIES3D_1_4_TLM_LANDESGEBIET.shp"
        kant_path = r"Y:\EU_BioES_SELINA\WP3\4. Spatially_Explicit_EC\Data\CH_shps\swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"

        ch   = gpd.read_file(ch_path)
        kant = gpd.read_file(kant_path)
        bern = kant[kant["NAME"] == "Bern"]

        if bern.crs != ch.crs:
            bern = bern.to_crs(ch.crs)

        axins = inset_axes(ax, width="30%", height="25%", loc="upper right",
                           borderpad=0.5)
        axins.set_aspect("equal")

        ch.plot(ax=axins, color="lightgrey", edgecolor="white", linewidth=0.3)
        bern.plot(ax=axins, color="#d62728", edgecolor="white", linewidth=0.3)

        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("grey")
    except Exception as e:
        import warnings
        warnings.warn(f"Switzerland inset skipped: {e}")


# ---------------------------------------------------------------------------
# Panel 4 — Timing distribution
# ---------------------------------------------------------------------------

def plot_timing_distribution(dyn_results: dict, ax=None) -> plt.Figure:
    """Stacked bar: fraction of ND-selected patches assigned to each time step."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    scenarios  = list(dyn_results.keys())
    time_steps = dyn_results[scenarios[0]]["time_steps"]
    n_steps    = len(time_steps)
    n_scen     = len(scenarios)

    step_colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_steps))
    bar_w = 0.65

    bottoms = np.zeros(n_scen)
    patches_legend = []

    for k, yr in enumerate(time_steps):
        fracs = []
        for scenario in scenarios:
            res = dyn_results[scenario]
            dec = res["decisions"][res["nd"]]   # (n_nd, n_patches)
            # Fraction of all patches (across all ND solutions) assigned to step k+1
            total_assignments = dec.size
            at_step = (dec == (k + 1)).sum()
            fracs.append(at_step / total_assignments if total_assignments > 0 else 0.0)
        bars = ax.bar(range(n_scen), fracs, bar_w, bottom=bottoms,
                      color=step_colors[k], label=str(yr))
        patches_legend.append(bars[0])
        bottoms += np.array(fracs)

    # "Never" remainder
    fracs_never = 1.0 - bottoms
    ax.bar(range(n_scen), fracs_never, bar_w, bottom=bottoms,
           color="lightgrey", label="Never", alpha=0.8)

    ax.set_xticks(range(n_scen))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8,
                       rotation=25, ha="right")
    ax.set_ylabel("Fraction of patch-decisions", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("(d) Timing distribution across scenarios", fontsize=10, fontweight="bold")
    ax.legend(title="Restoration year", fontsize=8, title_fontsize=8,
              loc="upper right", framealpha=0.8)
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined figure assembly
# ---------------------------------------------------------------------------

def make_overview_figure(dyn_results: dict, static_result: dict | None,
                         mean_freq: np.ndarray, std_freq: np.ndarray,
                         rest_mask: np.ndarray,
                         n_scenarios_selecting: np.ndarray | None = None) -> plt.Figure:
    """1x3 layout: one row = curves + pareto + spatial map"""
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    ax_curves  = fig.add_subplot(gs[0, 0])
    ax_pareto  = fig.add_subplot(gs[0, 1])
    ax_sens    = fig.add_subplot(gs[0, 2])

    plot_recovery_curves(ax=ax_curves)
    plot_pareto_scatter(dyn_results, static_result, ax=ax_pareto)
    plot_spatial_agreement(n_scenarios_selecting, rest_mask, ax=ax_sens)

    ax_curves.set_box_aspect(1)
    ax_pareto.set_box_aspect(1)
    ax_sens.set_box_aspect(1)
    #plot_timing_distribution(dyn_results, ax=ax_timing)

    # Spatial panels — crop and render inside the provided axes
    rest_mask_crop = _crop_to_mask(rest_mask, rest_mask)
    mean_crop = _crop_to_mask(np.where(rest_mask, mean_freq, np.nan), rest_mask)
    std_crop  = _crop_to_mask(np.where(rest_mask, std_freq,  np.nan), rest_mask)

    # Compute vmax over actually-selected patches only
    def _vmax(arr, pct=95, floor=0.05):
        v = arr[~np.isnan(arr)]
        v = v[v > 0]
        return max(float(np.percentile(v, pct)) if v.size > 10 else floor, floor)

    vm_mean = _vmax(mean_crop)
    vm_std  = _vmax(std_crop)

    #im1 = ax_mean.imshow(mean_crop, cmap="YlOrRd", vmin=0, vmax=vm_mean,
    #                     interpolation="nearest")
    #ax_mean.set_title("(c) Mean selection frequency across recovery scenarios",
    #                  fontsize=10, fontweight="bold")
    #ax_mean.axis("off")
    #fig.colorbar(im1, ax=ax_mean, fraction=0.025, pad=0.02,
    #             label="Fraction of ND solutions selecting patch")

    #im2 = ax_sens.imshow(std_crop, cmap="PuBu", vmin=0, vmax=vm_std,
    #                     interpolation="nearest")
    #ax_sens.set_title("(e) Scenario sensitivity\n(std across scenarios)",
    #                  fontsize=10, fontweight="bold")
    #ax_sens.axis("off")
    #fig.colorbar(im2, ax=ax_sens, fraction=0.04, pad=0.02,
    #             label="Std of selection frequency")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Loading dynamic results …")
    dyn_results, run_meta = load_dynamic_results()
    if not dyn_results:
        print("ERROR: no dynamic results found. Exiting.")
        return
    print(f"  Run settings from pkl: aggregation_factor={run_meta['aggregation_factor']}, "
          f"patch_size={run_meta['patch_size']}")

    print("Loading static result …")
    static_result = load_static_result()

    print("Loading initial conditions for spatial map …")
    ic = load_initial_conditions_for_spatial(
        aggregation_factor=run_meta["aggregation_factor"],
        patch_size=run_meta["patch_size"],
    )
    rest_mask = ic["restoration_eligible_mask"]

    print("Computing spatial agreement …")
    mean_freq, std_freq = compute_spatial_agreement(dyn_results, ic)
    print(f"  Max mean freq: {mean_freq[rest_mask].max():.3f}, "
          f"max std: {std_freq[rest_mask].max():.3f}")

    print("Computing per-step spatial divergence …")
    step_data = compute_spatial_divergence_by_step(dyn_results, ic)

    # ---- Individual panels ------------------------------------------------
    print("Saving individual panels …")

    fig1 = plot_recovery_curves()
    _savefig(fig1, "recovery_curves.png", individual=True)
    plt.close(fig1)

    fig2 = plot_pareto_scatter(dyn_results, static_result)
    _savefig(fig2, "pareto_scatter_scenarios.png", individual=True)
    plt.close(fig2)

    fig3 = plot_spatial_agreement(step_data["n_scenarios_selecting"], rest_mask)
    _savefig(fig3, "spatial_agreement.png", individual=True)
    plt.close(fig3)

    fig4 = plot_timing_distribution(dyn_results)
    _savefig(fig4, "timing_distribution_scenarios.png", individual=True)
    plt.close(fig4)

    fig5 = plot_spatial_divergence_by_step(step_data, rest_mask)
    _savefig(fig5, "spatial_divergence_by_step.png", individual=True)
    plt.close(fig5)

    fig5b = plot_spatial_divergence_bottom_only(step_data)
    _savefig(fig5b, "spatial_divergence_bottom_panels.png", individual=True)
    plt.close(fig5b)

    print("Exporting spatial divergence rasters for R …")
    export_spatial_divergence_tifs(step_data, ic)

    # ---- Combined overview ------------------------------------------------
    print("Saving overview figure …")
    fig_ov = make_overview_figure(dyn_results, static_result, mean_freq, std_freq, rest_mask,
                                  n_scenarios_selecting=step_data["n_scenarios_selecting"])
    _savefig(fig_ov, "recovery_uncertainty_overview.png", individual=True)
    plt.close(fig_ov)

    print("Done.")


if __name__ == "__main__":
    main()
