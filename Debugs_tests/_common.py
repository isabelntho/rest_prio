"""Shared helpers for the Debugs_tests diagnostic modules.

Importing this module (a) puts the repo root on sys.path so `Core_optimisation`,
`visualisations` and `utils` import cleanly, and (b) makes stdout tolerant of the
Unicode check marks the shared visualisations code prints on a cp1252 console.

It also centralises the boilerplate that used to be copy-pasted across ~25 scripts:
the standard Bern / global_all initial-conditions load, the direct-evaluation BASE
params, the run constants (POP_SIZE / N_GENERATIONS / RANDOM_SEEDS), the union-find
adjacency / n_components metric, and a "newest pkl per seed" selector.

Because each diagnostic is launched as `pixi run python Debugs_tests/<mod>.py`, the
script's own directory is sys.path[0], so `from _common import ...` resolves without
any packaging.
"""
import os
import sys
import glob

REPO_ROOT = r"c:\Users\inicholson\Documents\rest_prio"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The visualisations helpers print a Unicode check mark on save; the user's console is
# cp1252, so make stdout tolerant rather than editing the shared module.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from Core_optimisation.data_loader import load_initial_conditions
from Core_optimisation.resto_anom import RestorationProblem

# --- standard locations -----------------------------------------------------
RESULTS_DIR = os.path.join(REPO_ROOT, "outputs", "results_files")
INTER_DIR = os.path.join(REPO_ROOT, "outputs", "intermediate_results")
DIAG_DIR = os.path.join(REPO_ROOT, "Debugs_tests", "diagnostics")

# --- standard config --------------------------------------------------------
# Direct-evaluation base (raw objectives) used by the spillover/benefit checks.
BASE = {
    "max_restoration_fraction": 0.05, "abiotic_effect": 0.01, "biotic_effect": 0.01,
    "normalize_objectives": False,
}
POP_SIZE = 50
N_GENERATIONS = 100
RANDOM_SEEDS = [606, 707, 808]


def use_agg():
    """Select the headless matplotlib backend (call before importing pyplot)."""
    import matplotlib
    matplotlib.use("Agg")


def load_ic(objectives, **overrides):
    """Load the standard Bern / ecosystem=all / global_all initial conditions.

    `overrides` can replace any load_initial_conditions kwarg (e.g. aggregation_factor).
    """
    kwargs = dict(region="Bern", ecosystem="all", sample_fraction=None,
                  sample_seed=42, aggregation_factor=None,
                  condition_scenario="global_all")
    kwargs.update(overrides)
    return load_initial_conditions(REPO_ROOT, objectives=list(objectives), **kwargs)


def make_problem(neighbor_radius, ic=None,
                 objectives=("restoration_benefit", "cost"), **params):
    """RestorationProblem with the BASE params + a neighbor_radius (spillover checks).

    Returns (problem, ic). Loads the standard IC if `ic` is not supplied.
    """
    if ic is None:
        ic = load_ic(objectives)
    return RestorationProblem(ic, {**BASE, "neighbor_radius": neighbor_radius, **params}), ic


def cluster_metrics(sel, nbr):
    """adjacency (shared edges among selected) and n_components via the neighbour table.

    Union-find over the 4-neighbour table; `sel` is a boolean vector over restoration
    pixels and `nbr` the (n_rest, 4) neighbour-index table (-1 = no neighbour).
    """
    valid = nbr >= 0
    sel_nb = np.where(valid, sel[np.clip(nbr, 0, None)], False) & valid
    adjacency = int(sel_nb[sel].sum() // 2)
    idx = np.flatnonzero(sel)
    pos = {p: i for i, p in enumerate(idx)}
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in idx:
        for q in nbr[p]:
            if q >= 0 and sel[q] and q in pos:
                ra, rb = find(pos[p]), find(pos[q])
                if ra != rb:
                    parent[ra] = rb
    n_comp = len({find(i) for i in range(len(idx))})
    return adjacency, n_comp


def newest_per_seed(pattern_fmt, seeds):
    """{seed: newest matching pkl}. `pattern_fmt` is a glob with a `{seed}` placeholder,
    resolved against RESULTS_DIR; newest = latest file mtime."""
    out = {}
    for seed in seeds:
        hits = glob.glob(os.path.join(RESULTS_DIR, pattern_fmt.format(seed=seed)))
        if hits:
            out[seed] = max(hits, key=os.path.getmtime)
    return out


def newest_xhist():
    """Newest outputs/intermediate_results/X_history_*.npz (raises if none exist)."""
    hits = glob.glob(os.path.join(INTER_DIR, "X_history_*.npz"))
    if not hits:
        raise FileNotFoundError(
            f"No X_history_*.npz in {INTER_DIR}. Run a snapshot (save_snapshots=True) first.")
    return max(hits, key=os.path.getmtime)


def render_coverage_gif(xhist_path, ic, out_gif, title_prefix, step=1, fps=8, thresh=0.5):
    """Render a per-generation coverage GIF from an X_history npz.

    Each frame = the whole population's restoration selection frequency at that generation
    (fraction of individuals selecting each pixel), grey = eligible-but-unselected. The
    title reports cumulative coverage: the % of eligible pixels touched by ANY individual
    in ANY generation up to that frame (the explored footprint, incl. early discards).

    `title_prefix` is prepended to the per-frame title (e.g. "S=2 scored"). The caller is
    responsible for selecting a headless backend first (use_agg()).

    Returns (final_cov_pct, n_never_touched, n_rest).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    X = np.load(xhist_path)["X"]                # (n_gens, pop, n_var) int8
    n_gens, pop, n_var = X.shape
    shape = ic["shape"]
    rest_idx = np.asarray(ic["restoration_eligible_indices"])
    n_rest = len(rest_idx)

    # crop to the restoration bounding box so the map is legible
    rows, cols = np.unravel_index(rest_idx, shape)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

    sel = X[:, :, :n_rest] > thresh             # (n_gens, pop, n_rest) bool
    ever = np.zeros(n_rest, dtype=bool)         # cumulative touched pixels
    gens = list(range(0, n_gens, step))

    # frozen grey background of all eligible pixels
    base = np.full(shape, np.nan, dtype=float)
    base.flat[rest_idx] = 0.0
    grey = np.where(~np.isnan(base[r0:r1, c0:c1]), 0.0, np.nan)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(grey, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    freq_map = np.full(shape, np.nan, dtype=float)
    im = ax.imshow(freq_map[r0:r1, c0:c1], cmap="YlOrRd", vmin=0, vmax=100,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("population selection frequency (%)")
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("")

    def draw(g):
        gen_sel = sel[g]                        # (pop, n_rest)
        freq = gen_sel.mean(axis=0) * 100.0     # % of pop selecting each pixel
        ever[:] |= gen_sel.any(axis=0)
        fm = np.full(shape, np.nan, dtype=float)
        fm.flat[rest_idx] = freq
        im.set_data(fm[r0:r1, c0:c1])
        cum = 100.0 * ever.sum() / n_rest
        title.set_text(f"{title_prefix}, gen {g+1}/{n_gens}  |  "
                       f"cumulative coverage {cum:.1f}% of eligible")
        return im, title

    anim = FuncAnimation(fig, draw, frames=gens, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    final_cov = 100.0 * ever.sum() / n_rest
    return final_cov, int((~ever).sum()), n_rest
