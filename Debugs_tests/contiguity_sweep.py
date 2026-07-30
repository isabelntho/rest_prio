"""Price-of-contiguity: 2-objective potential-vs-cost NSGA-II at several minimum-patch
levels S, plus the figures/tables read back from the saved pkls.

Subcommands (pixi run python Debugs_tests/contiguity_sweep.py <cmd>):

  run        RUN ONLY. Runs the potential-vs-cost problem at several min-patch levels S
             (+ optional scattered baseline) across several seeds, and SAVES one pkl per
             (level, seed) as res_*_contig_<level>_seed_<seed>.pkl. Process-level
             parallelism only (each run forced n_jobs=1; GRID_WORKERS runs concurrent as
             separate processes). Edit SMOKE below for a fast wiring check. Does NOT plot.

  fronts     Overlay the pooled potential-vs-cost Pareto fronts by contiguity level and
             print the summary table (hv / patch sizes / n_components / adjacency). Reads
             the saved pkls; re-runnable to tweak the figure.

  spatial    For each level, an example-solution map + per-pixel RFOP, from the same pkls.

  jaccard    Jaccard (spatial) vs objective similarity of Pareto solutions, faceted by S.

  snapshot [S]
             Run ONE random_region level (default S=5) with save_snapshots=True, then
             render a per-generation coverage GIF (exploration footprint growing gen by
             gen). Same config as a sweep `run` for that level, so it reflects the real
             run. This is a full optimisation - it re-runs (the sweep pkls have no
             per-generation snapshots to animate).

  animate [S]
             Re-render the coverage GIF from the newest X_history_*.npz without re-running
             (pass the S you snapshotted, for the title/filename only).

The plot subcommands share one pkl gatherer / newness rule so they always agree on which
run represents each level.
"""
# --- cap nested numpy/BLAS threading BEFORE importing numpy-heavy modules ----
# Spawned worker processes re-import this module top-to-bottom and inherit this env, so
# the ProcessPool x per-run eval threads do not oversubscribe on top of threaded BLAS.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import sys
import glob
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from _common import (
    REPO_ROOT, RESULTS_DIR, DIAG_DIR, load_ic, use_agg, newest_xhist, render_coverage_gif,
)
import numpy as np
from Core_optimisation.resto_anom import run_optimization_instance
from Core_optimisation.spatial_operations import _label_components, build_restoration_neighbor_table


# ===========================================================================
# run  (was contiguity_price_sweep.py)
# ===========================================================================
SMOKE = False                     # True = fast wiring check; False = real sweep
# Process-level parallelism ONLY (see run_custom_parallel.py / grid_parallel.py):
# each run is forced to n_jobs=1, and GRID_WORKERS runs execute concurrently as
# separate processes. GRID_WORKERS <= 1 = sequential in-process fallback.
# Bound GRID_WORKERS by RAM, not cores: each worker holds its own ~1.3M-pixel
# raster copy (the IC is pickled once into every worker via the pool initializer).
GRID_WORKERS = 6                  # number of runs executing concurrently (processes)
RUN_POP_SIZE = 50
RUN_N_GENERATIONS = 100
RUN_RANDOM_SEEDS = [606, 707, 808, 909, 102]
MIN_PATCH_LEVELS = [1, 2, 5, 10, 25]  # min clump size in pixels = hectares; 1 = no floor
INCLUDE_SCATTERED = False              # add the unconstrained bitflip baseline

if SMOKE:
    RUN_POP_SIZE, RUN_N_GENERATIONS = 12, 2
    MIN_PATCH_LEVELS = [1, 100]
    RUN_RANDOM_SEEDS = [101]
    INCLUDE_SCATTERED = True
    GRID_WORKERS = 2

RUN_BASE_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01, "normalize_objectives": True,
    "burden_sharing": "no", "rp_formulation": "sum", "rp_threshold": 0.0,
    "region_seeds": 25, "region_seeds_min": 5, "region_growth_bias": "scored",
    "region_mutation_edits": 100, "region_random_share": 1,
}

LOG_DIR = os.path.join(REPO_ROOT, "outputs", "logs")

# --- worker plumbing (module-level so it pickles under Windows 'spawn') ------
# initial_conditions is shipped to each worker ONCE via the pool initializer
# (initargs), not re-pickled per task, and stashed here.
_IC = None


def _init_worker(ic):
    """Pool initializer: cache the shared, read-only initial conditions."""
    global _IC
    _IC = ic


def _run_one(task):
    """Run a single (level, seed) optimisation. Returns a small picklable tuple.

    Process-level parallelism IS the parallelism here, so the inner pymoo eval thread
    pool is forced serial (n_jobs=1). Redirects this run's verbose stdout/stderr to a
    per-run log file so concurrent runs do not interleave, and restores the streams
    afterwards (a worker process is reused across several tasks).
    """
    run_label, _label, params, seed = task
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{run_label}.log")
    t0 = time.perf_counter()
    _old_out, _old_err = sys.stdout, sys.stderr
    ok, err = False, None
    try:
        with open(log_path, "w", encoding="ascii", errors="replace") as logf:
            sys.stdout = sys.stderr = logf
            try:
                run_optimization_instance(
                    initial_conditions=_IC, scenario_params=params,
                    pop_size=RUN_POP_SIZE, n_generations=RUN_N_GENERATIONS,
                    save_results=(not SMOKE), verbose=True, n_jobs=1,
                    random_seed=seed, use_repair=True, use_patch_approach=False,
                    pixel_tolerance=0.05, algorithm_type="nsga2", run_label=run_label,
                )
                ok = True
            except Exception as e:  # noqa: BLE001 - report, don't crash the pool
                err = repr(e)
    finally:
        sys.stdout, sys.stderr = _old_out, _old_err
    elapsed = time.perf_counter() - t0
    return run_label, ok, elapsed, err


def _build_tasks():
    """Build the flat list of (run_label, label, params, seed) runs."""
    levels = []
    if INCLUDE_SCATTERED:
        levels.append(("scattered", {**RUN_BASE_PARAMS, "sampling_strategy": "scattered"}))
    for lvl_s in MIN_PATCH_LEVELS:
        levels.append((f"region S={lvl_s}",
                       {**RUN_BASE_PARAMS, "sampling_strategy": "region_grow", "min_patch_size": lvl_s}))

    tasks = []
    for seed in RUN_RANDOM_SEEDS:
        for label, params in levels:
            safe_label = label.replace(" ", "_").replace("=", "")
            run_label = f"contig_random_{safe_label}_seed_{seed}"
            tasks.append((run_label, label, params, seed))
    return tasks


def cmd_run():
    ic = load_ic(["restoration_benefit", "cost"])
    _init_worker(ic)  # stash for the serial fallback path

    tasks = _build_tasks()
    n_runs = len(tasks)
    ok = 0
    t0 = time.perf_counter()

    mode = f"PARALLEL ({GRID_WORKERS} workers x n_jobs=1)" if GRID_WORKERS > 1 else "SERIAL"
    print(f"\n===== Contiguity sweep: {n_runs} runs, {mode}, "
          f"pop={RUN_POP_SIZE}, gens={RUN_N_GENERATIONS} =====")
    print(f"Per-run logs -> {LOG_DIR}\\<run_label>.log")

    if GRID_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=GRID_WORKERS,
                                 initializer=_init_worker, initargs=(ic,)) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks}
            for done, fut in enumerate(as_completed(futs), 1):
                run_label, r_ok, elapsed, err = fut.result()
                if r_ok:
                    ok += 1
                    print(f"[{done}/{n_runs}] OK   {run_label} ({elapsed/60:.1f} min)")
                else:
                    print(f"[{done}/{n_runs}] FAIL {run_label} ({elapsed/60:.1f} min): {err}")
    else:
        # Serial fallback (GRID_WORKERS <= 1): same runs, one at a time, still routed
        # through _run_one so each run keeps writing its own per-run log file.
        for done, task in enumerate(tasks, 1):
            run_label = task[0]
            print(f"\n===== [{done}/{n_runs}] {run_label} =====")
            _, r_ok, elapsed, err = _run_one(task)
            if r_ok:
                ok += 1
                print(f"  [ok] {run_label} ({elapsed/60:.1f} min)")
            else:
                print(f"  [x] {run_label} failed: {err}")

    elapsed = time.perf_counter() - t0
    print(f"\n===== SWEEP COMPLETE: {ok}/{n_runs} runs saved in {elapsed/60:.1f} min =====")
    if SMOKE:
        print("SMOKE mode: save_results was off, so no pkls were written.")
    else:
        print("Pkls: outputs/results_files/res_<timestamp>_contig_<level>_seed_<seed>.pkl")
        print("Make the figure + table:  pixi run python Debugs_tests/contiguity_sweep.py fronts")


# ===========================================================================
# shared plot gathering  (was in plot_contiguity_fronts.py)
# ===========================================================================
OUT_DIR = DIAG_DIR
EXCLUDE_FROM_PLOT = {"scattered"}   # off the figure (out of scale); still in the table
PKL_GLOB = "*_contig_*.pkl"         # which runs to gather
# PLOT_SEEDS: the main knob for choosing WHICH sweep to plot. The results folder can hold
# runs from earlier sweeps (different seeds and/or a different objective). Set to the seed
# list of the sweep you want; None = every seed found (fine for a single sweep on disk).
PLOT_SEEDS = {606, 707, 808, 909, 102}
# Safety backstop: never pool runs from a different objective formulation (older
# 'restoration_potential' vs newer 'restoration_benefit').
POTENTIAL_OBJ = "restoration_benefit"

_TS_RE = re.compile(r"res_(\d{8}_\d{4})_contig_")
_INHERIT = object()   # sentinel: gather_pkls(seeds=...) defaults to PLOT_SEEDS


def label_sort_key(level):
    """scattered first, then region_S<k> by ascending k."""
    if "scattered" in level:
        return (-1,)
    m = re.search(r"S(\d+)", level)
    return (int(m.group(1)) if m else 10**9,)


def _newness_key(path):
    """Sort key for 'newest': embedded run timestamp (res_YYYYMMDD_HHMM_...) first, file
    mtime as a tiebreak. The timestamp string sorts chronologically as text."""
    m = _TS_RE.search(os.path.basename(path))
    return (m.group(1) if m else "", os.path.getmtime(path))


def gather_pkls(seeds=_INHERIT):
    """level -> list of the newest pkl per seed.

    Groups every matching file by (level, seed), then keeps the newest per combo (embedded
    run timestamp, file mtime tiebreak), so re-running some seeds/levels supersedes older
    pkls. Only multi-seed files (res_*_contig_<level>_seed_<seed>.pkl) are pooled.
    """
    from collections import defaultdict
    sel = PLOT_SEEDS if seeds is _INHERIT else seeds
    allow = {str(s) for s in sel} if sel is not None else None
    groups = defaultdict(list)   # (level, seed) -> [paths]
    skipped = 0
    for p in glob.glob(os.path.join(RESULTS_DIR, PKL_GLOB)):
        base = os.path.basename(p)
        if "_contig_" not in base:
            continue
        tail = base.split("_contig_", 1)[1].rsplit(".pkl", 1)[0]
        if "_seed_" not in tail:
            skipped += 1
            continue
        level, seed = tail.rsplit("_seed_", 1)
        if allow is not None and seed not in allow:
            continue
        groups[(level, seed)].append(p)
    if skipped:
        print(f"(skipped {skipped} old-format contig pkl(s) without a _seed_ tag)")

    by = {}   # level -> [newest path per seed]
    for (level, seed), paths in sorted(groups.items()):
        newest = sorted(paths, key=_newness_key)[-1]
        by.setdefault(level, []).append(newest)
    return by


def _obj_indices(names):
    pot = names.index(POTENTIAL_OBJ)
    cost = names.index("implementation_cost") if "implementation_cost" in names else names.index("cost")
    return pot, cost


def _adjacency(sel, nbr):
    valid = nbr >= 0
    sel_nb = np.where(valid, sel[np.clip(nbr, 0, None)], False) & valid
    return int(sel_nb[sel].sum() // 2)


# ===========================================================================
# fronts  (was plot_contiguity_fronts.py)
# ===========================================================================
def _stats_from_level(paths, nbr_bundle):
    """Pool non-dominated fronts + descriptive contiguity across a level's seed runs."""
    nbr, rows, cols, shape = nbr_bundle
    n_rest = nbr.shape[0]
    F_list, Fall_list, hvs = [], [], []
    nc, ms, mn, ad = [], [], [], []
    n_skipped = 0
    for path in paths:
        with open(path, "rb") as f:
            res = pickle.load(f)
        names = list(res["objective_names"])
        if POTENTIAL_OBJ not in names:
            n_skipped += 1
            continue
        pot, cost = _obj_indices(names)
        raw = np.asarray(res["objectives_raw"], float)
        nd = np.asarray(res["is_nondominated"], bool)
        F_list.append(raw[nd][:, [pot, cost]])
        Fall_list.append(raw[:, [pot, cost]])
        hv = res.get("algorithm_info", {}).get("hypervolume_history", [np.nan])
        hvs.append(float(hv[-1]) if len(hv) else float("nan"))
        dec = np.asarray(res["decisions"])
        for i in np.where(nd)[0]:
            sel = dec[i, :n_rest].astype(bool)
            sizes = [c.size for c in _label_components(sel, shape, rows, cols)]
            nc.append(len(sizes)); ms.append(np.mean(sizes) if sizes else 0)
            mn.append(min(sizes) if sizes else 0); ad.append(_adjacency(sel, nbr))
    F = np.vstack(F_list) if F_list else np.empty((0, 2))
    F_all = np.vstack(Fall_list) if Fall_list else np.empty((0, 2))
    return {
        "F": F, "F_all": F_all, "n": F.shape[0],
        "n_seeds": len(paths) - n_skipped, "n_skipped": n_skipped,
        "hv": float(np.nanmean(hvs)) if hvs else float("nan"),
        "n_components": float(np.mean(nc)) if nc else 0.0,
        "mean_patch": float(np.mean(ms)) if ms else 0.0,
        "min_patch": float(np.min(mn)) if mn else 0.0,
        "adjacency": float(np.mean(ad)) if ad else 0.0,
    }


def _get_nbr_bundle(pkl_paths):
    """Build the neighbour table once from a pkl's initial_conditions (reload if absent)."""
    for p in pkl_paths:
        with open(p, "rb") as f:
            ic = pickle.load(f).get("initial_conditions", {})
        if ic.get("shape") is not None and ic.get("restoration_eligible_indices") is not None:
            nbr, rows, cols = build_restoration_neighbor_table(ic)
            return nbr, rows, cols, tuple(ic["shape"])
    ic = load_ic(["restoration_benefit", "cost"])
    nbr, rows, cols = build_restoration_neighbor_table(ic)
    return nbr, rows, cols, tuple(ic["shape"])


def cmd_fronts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "contiguity_fronts/contiguity_price.png")
    by_level = gather_pkls()
    if not by_level:
        print(f"No pkls matching {PKL_GLOB} in {RESULTS_DIR}. Run the sweep first "
              f"(contiguity_sweep.py run with SMOKE=False).")
        return
    levels = sorted(by_level, key=label_sort_key)
    all_paths = [p for lvl in levels for p in by_level[lvl]]
    nbr_bundle = _get_nbr_bundle(all_paths)

    stats = {lvl: _stats_from_level(by_level[lvl], nbr_bundle) for lvl in levels}

    skipped_total = sum(stats[lvl]["n_skipped"] for lvl in levels)
    empty = [lvl for lvl in levels if stats[lvl]["n_seeds"] == 0]
    levels = [lvl for lvl in levels if stats[lvl]["n_seeds"] > 0]
    if skipped_total:
        print(f"(skipped {skipped_total} run(s) whose objective is not '{POTENTIAL_OBJ}' "
              f"- e.g. older restoration_potential runs)")
    if empty:
        print(f"(no '{POTENTIAL_OBJ}' runs for: {', '.join(empty)} - dropped)")
    if not levels:
        print(f"No '{POTENTIAL_OBJ}' runs found. Run the sweep first "
              f"(contiguity_sweep.py run with SMOKE=False).")
        return

    print("Using pkls (level -> #seeds):")
    for lvl in levels:
        print(f"  {lvl:<14} {stats[lvl]['n_seeds']} seed(s)")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_levels = [l for l in levels if l not in EXCLUDE_FROM_PLOT]
    pop = [stats[l]["F_all"] for l in plot_levels if stats[l]["F_all"].size]
    if pop:
        P = np.vstack(pop)
        ax.scatter(-P[:, 0], P[:, 1], s=9, color="lightgrey", alpha=0.6,
                   edgecolors="none", zorder=1, label="_population")
    plotted = 0
    for lvl in plot_levels:
        F = stats[lvl]["F"]
        if F.size:
            ax.scatter(-F[:, 0], F[:, 1], s=18, alpha=0.85, zorder=2,
                       label=f"{lvl} (n={stats[lvl]['n']}, {stats[lvl]['n_seeds']} seeds)")
            plotted += 1
    ax.set_xlabel("restoration benefit (-potential sum)")
    ax.set_ylabel("implementation cost")
    ax.set_title("Price of contiguity: potential-vs-cost front by minimum patch size")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    excluded = ", ".join(sorted(EXCLUDE_FROM_PLOT)) or "none"
    print(f"\nfront figure -> {out_path}   (excluded from plot: {excluded})")

    print("\n=== price-of-contiguity summary (pooled across seeds) ===")
    print(f"{'level':<14}{'seeds':>6}{'n':>5}{'hv':>11}{'mean_patch':>12}"
          f"{'min_patch':>11}{'n_comp':>9}{'adjacency':>11}")
    for lvl in levels:
        st = stats[lvl]
        print(f"{lvl:<14}{st['n_seeds']:>6}{st['n']:>5}{st['hv']:>11.4g}"
              f"{st['mean_patch']:>12.0f}{st['min_patch']:>11.0f}"
              f"{st['n_components']:>9.1f}{st['adjacency']:>11.0f}")


# ===========================================================================
# spatial  (was plot_contiguity_spatial.py)
# ===========================================================================
SPATIAL_OUT_DIR = os.path.join(DIAG_DIR, "contiguity_spatial")
CHOOSE = "best_sum"        # example-solution pick: best_sum|best_first|random|index
SOLUTION_ID = None        # only used when CHOOSE == "index"
SHOW_ELIGIBLE = True
ACTION_TYPE = "combined"
FREQ_CMAP = "YlOrRd"
INCLUDE_EXCLUDED = False   # also map levels EXCLUDE_FROM_PLOT (e.g. scattered)


def cmd_spatial():
    import matplotlib
    matplotlib.use("Agg")   # headless: the visualisations functions call plt.show()
    from visualisations import plot_example_solution, create_selection_frequency_map

    os.makedirs(SPATIAL_OUT_DIR, exist_ok=True)
    by_level = gather_pkls(seeds=PLOT_SEEDS)
    if not by_level:
        print("No contig pkls found. Run contiguity_sweep.py run (SMOKE=False) first.")
        return

    levels = sorted(by_level, key=label_sort_key)
    if not INCLUDE_EXCLUDED:
        levels = [l for l in levels if l not in EXCLUDE_FROM_PLOT]

    print(f"Levels to map: {levels}")
    for lvl in levels:
        pkl = sorted(by_level[lvl], key=_newness_key)[-1]
        safe = lvl.replace(" ", "_").replace(">=", "ge").replace("=", "")
        sol_png = os.path.join(SPATIAL_OUT_DIR, f"solution_{safe}.png")
        rfop_png = os.path.join(SPATIAL_OUT_DIR, f"rfop_{safe}.png")

        print(f"\n=== {lvl} ===")
        print(f"  pkl: {os.path.basename(pkl)}")

        plot_example_solution(
            pkl, choose=CHOOSE, solution_id=SOLUTION_ID,
            show_eligible=SHOW_ELIGIBLE, action_type=ACTION_TYPE,
            title=f"{lvl}: example solution ({CHOOSE})", save_path=sol_png,
        )
        print(f"  example solution -> {sol_png}")

        create_selection_frequency_map(
            pkl, save_path=rfop_png, cmap=FREQ_CMAP, show_eligible=SHOW_ELIGIBLE,
            action_type=ACTION_TYPE, title=f"{lvl}: RFOP",
        )
        print(f"  RFOP map         -> {rfop_png}")

    print(f"\nAll maps written under {SPATIAL_OUT_DIR}")


# ===========================================================================
# jaccard  (was plot_jaccard_vs_objective.py)
# ===========================================================================
JACCARD_OUT_PATH = os.path.join(DIAG_DIR, "contiguity_spatial", "jaccard_vs_objective.png")
NCOLS = 3
POINT_SIZE = 10
POINT_ALPHA = 0.45


def _restoration_selection(res):
    ic = res["initial_conditions"]
    n_rest = len(np.asarray(ic["restoration_eligible_indices"]))
    dec = np.asarray(res["decisions"])
    nd = np.asarray(res["is_nondominated"], bool)
    return dec[nd][:, :n_rest] > 0.5


def _objective_vectors(res):
    names = list(res["objective_names"])
    pot, cost = _obj_indices(names)
    raw = np.asarray(res["objectives_raw"], float)
    nd = np.asarray(res["is_nondominated"], bool)
    return raw[nd][:, [pot, cost]]


def _jaccard(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else np.nan


def _collect_level(paths):
    per_seed = []
    for path in paths:
        with open(path, "rb") as f:
            res = pickle.load(f)
        if POTENTIAL_OBJ not in list(res["objective_names"]):
            continue
        seed = os.path.basename(path).rsplit("_seed_", 1)[1].rsplit(".pkl", 1)[0]
        sel = _restoration_selection(res)
        F = _objective_vectors(res)
        if sel.shape[0] >= 2:
            per_seed.append((seed, sel, F))
    return per_seed


def _level_pairs(per_seed):
    import itertools
    if not per_seed:
        return np.empty(0), np.empty(0), []
    allF = np.vstack([F for _, _, F in per_seed])
    lo = allF.min(axis=0)
    rng = allF.max(axis=0) - lo
    rng[rng == 0] = 1.0
    diag = np.sqrt(allF.shape[1])

    obj_sim, jac, seeds = [], [], []
    for seed, sel, F in per_seed:
        Fn = (F - lo) / rng
        for i, j in itertools.combinations(range(sel.shape[0]), 2):
            d = np.linalg.norm(Fn[i] - Fn[j]) / diag
            obj_sim.append(1.0 - d)
            jac.append(_jaccard(sel[i], sel[j]))
            seeds.append(seed)
    return np.asarray(obj_sim), np.asarray(jac), seeds


def cmd_jaccard():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(JACCARD_OUT_PATH), exist_ok=True)
    by_level = gather_pkls(seeds=PLOT_SEEDS)
    if not by_level:
        print("No contig pkls found. Run contiguity_sweep.py run first.")
        return
    levels = [l for l in sorted(by_level, key=label_sort_key) if l not in EXCLUDE_FROM_PLOT]

    data = {}
    for lvl in levels:
        data[lvl] = _level_pairs(_collect_level(by_level[lvl]))
    levels = [l for l in levels if data[l][0].size]
    if not levels:
        print(f"No '{POTENTIAL_OBJ}' runs with >=2 non-dominated solutions per seed.")
        return

    seed_list = sorted({s for lvl in levels for s in data[lvl][2]})
    cmap = plt.get_cmap("tab10")
    seed_color = {s: cmap(i % 10) for i, s in enumerate(seed_list)}

    ncols = min(NCOLS, len(levels))
    nrows = int(np.ceil(len(levels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for k, lvl in enumerate(levels):
        ax = axes[k // ncols][k % ncols]
        xs, ys, seeds = data[lvl]
        for s in seed_list:
            m = np.array([sd == s for sd in seeds])
            if m.any():
                ax.scatter(xs[m], ys[m], s=POINT_SIZE, alpha=POINT_ALPHA,
                           color=seed_color[s], edgecolors="none", label=f"seed {s}")
        ok = np.isfinite(xs) & np.isfinite(ys)
        r = (np.corrcoef(xs[ok], ys[ok])[0, 1]
             if ok.sum() >= 2 and np.std(xs[ok]) > 0 and np.std(ys[ok]) > 0 else np.nan)
        ax.set_title(f"{lvl}  (pairs={ok.sum()}, r={r:.2f})", fontsize=10)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    for k in range(len(levels), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    for r_i in range(nrows):
        axes[r_i][0].set_ylabel("Jaccard similarity (spatial)")
    for c_i in range(ncols):
        axes[nrows - 1][c_i].set_xlabel("objective similarity")

    handles = [plt.Line2D([0], [0], marker="o", ls="", color=seed_color[s], label=f"seed {s}")
               for s in seed_list]
    fig.legend(handles=handles, loc="lower center", ncol=len(seed_list),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Spatial (Jaccard) vs objective similarity of Pareto solutions, by min patch S",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(JACCARD_OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"jaccard-vs-objective figure -> {JACCARD_OUT_PATH}")

    print(f"\n{'level':<14}{'pairs':>7}{'mean_jac':>10}{'mean_objsim':>13}{'r':>7}")
    for lvl in levels:
        xs, ys, _ = data[lvl]
        ok = np.isfinite(xs) & np.isfinite(ys)
        r = (np.corrcoef(xs[ok], ys[ok])[0, 1]
             if ok.sum() >= 2 and np.std(xs[ok]) > 0 and np.std(ys[ok]) > 0 else np.nan)
        print(f"{lvl:<14}{ok.sum():>7}{np.nanmean(ys):>10.3f}{np.nanmean(xs):>13.3f}{r:>7.2f}")


# ===========================================================================
# snapshot / animate  (per-generation coverage GIF for a random_region config)
# ===========================================================================
SNAP_LEVEL = 5            # default min-patch level to snapshot (override: snapshot <S>)
SNAP_SEED = 606
SNAP_N_JOBS = 12
SNAP_STEP = 1             # animate every STEP-th generation
SNAP_FPS = 8
SNAP_THRESH = 0.5


def _snap_gif_path(S):
    return os.path.join(SPATIAL_OUT_DIR, f"coverage_per_gen_contig_S{S}.gif")


def cmd_snapshot(arg):
    """Run ONE random_region level (default S=SNAP_LEVEL) with save_snapshots=True, then
    render the per-generation coverage GIF. Same config as a sweep `run` for that level
    (region_grow, region_random_share=1, scored), so the GIF reflects the real run."""
    S = int(arg) if arg else SNAP_LEVEL
    os.makedirs(SPATIAL_OUT_DIR, exist_ok=True)
    run_label = f"contig_random_snap_S{S}_seed_{SNAP_SEED}"
    params = {**RUN_BASE_PARAMS, "sampling_strategy": "region_grow", "min_patch_size": S}

    ic = load_ic(["restoration_benefit", "cost"])
    print(f"===== random_region snapshot: S={S}, seed={SNAP_SEED}, "
          f"pop={RUN_POP_SIZE}, gens={RUN_N_GENERATIONS} -> {run_label} =====")
    res = run_optimization_instance(
        initial_conditions=ic, scenario_params=params,
        pop_size=RUN_POP_SIZE, n_generations=RUN_N_GENERATIONS, save_results=True,
        verbose=True, n_jobs=SNAP_N_JOBS, random_seed=SNAP_SEED, use_repair=True,
        use_patch_approach=False, pixel_tolerance=0.05, algorithm_type="nsga2",
        run_label=run_label, save_snapshots=True,
    )
    xh = res.get("X_history_path")
    print(f"\nX_history_path = {xh}")
    if not (xh and os.path.exists(xh)):
        print("No X_history written - cannot build the GIF."); return
    _render_snap_gif(xh, ic, S)


def _render_snap_gif(xhist_path, ic, S):
    use_agg()
    gif = _snap_gif_path(S)
    final_cov, n_never, n_rest = render_coverage_gif(
        xhist_path, ic, gif, f"random_region S={S}, scored",
        step=SNAP_STEP, fps=SNAP_FPS, thresh=SNAP_THRESH)
    print(f"raster {ic['shape']}, restoration-eligible pixels {n_rest}")
    print(f"Final cumulative coverage (all gens, all individuals): {final_cov:.1f}%")
    print(f"Pixels never touched: {n_never} ({100 - final_cov:.1f}%)")
    print(f"animation -> {gif}")


def cmd_animate(arg):
    """Re-render the coverage GIF from the newest X_history_*.npz (no rerun). Pass the S
    you snapshotted for the title/filename (default SNAP_LEVEL); it only labels the GIF -
    the newest X_history on disk is used regardless."""
    S = int(arg) if arg else SNAP_LEVEL
    os.makedirs(SPATIAL_OUT_DIR, exist_ok=True)
    path = newest_xhist()
    print(f"Loading {os.path.basename(path)} ...")
    ic = load_ic(["restoration_benefit", "cost"])
    _render_snap_gif(path, ic, S)


# ===========================================================================
COMMANDS = {
    "run": lambda a: cmd_run(),
    "fronts": lambda a: cmd_fronts(),
    "spatial": lambda a: cmd_spatial(),
    "jaccard": lambda a: cmd_jaccard(),
    "snapshot": cmd_snapshot,
    "animate": cmd_animate,
}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    arg = argv[2] if len(argv) > 2 else None
    if cmd not in COMMANDS:
        print(__doc__)
        print(f"Subcommands: {', '.join(COMMANDS)}")
        return
    COMMANDS[cmd](arg)


if __name__ == "__main__":
    main(sys.argv)
