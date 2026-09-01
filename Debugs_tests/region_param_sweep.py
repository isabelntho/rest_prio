"""Region-operator parameter sweep: which region_* settings actually help the search?

The five region knobs (region_seeds, region_seeds_min, region_growth_bias,
region_mutation_edits, region_random_share) were set by hand and never validated. They are
all consumed in one place, _build_operators (Core_optimisation/resto_anom.py:1719-1783),
and feed RegionGrowingSampling / RegionGrowingMutation. Three of them (seeds, seeds_min,
random_share) affect ONLY the initial population; growth_bias affects sampling, mutation
and the repair; mutation_edits affects only mutation.

This is a ONE-AT-A-TIME sweep: a baseline arm holding the production values from
run_custom_nsga2.py, plus one arm per alternative level of one knob.

There is also a `regionswap` control arm. region_grow leaves crossover=None, so
_build_algorithm falls back to pixel-level HUX (resto_anom.py:1865-1866), which shatters
the contiguous regions the sampler builds: plans seeded with 25 vs 500 regions both
collapse to ~2900 components within a couple of generations. That would make the three
initial-population-only knobs look inert for a reason unrelated to those knobs. The arm
sets region_crossover="region_swap" to use the contiguity-preserving RegionSwapCrossover
that region_evolve already uses, separating "knob does nothing" from "crossover erased it".

It reports four metric families, because the question is not only "which setting optimises
best" but "which setting stops the population collapsing onto one basin":

  search      hv_first/hv_last/hv_change_pct, gens_to_90pct, wall time
  front       n_nd, n_unique_F, per-objective min/max/range/cv, benefit-cost correlation
  structure   n_components, mean_patch, min_patch, adjacency (over the non-dominated plans)
  diversity   centroid_spread, cells_touched, top_cell_share, mean pairwise Jaccard

Subcommands (pixi run python Debugs_tests/region_param_sweep.py <cmd>):

  run [arms] RUN ONLY. Runs every (arm, seed) and SAVES one pkl per run as
             res_*_regionop_<arm>_seed_<seed>.pkl. Process-level parallelism only
             (evaluation is sequential within a run; GRID_WORKERS runs concurrent as
             separate processes).
             Prints a pre-flight arm table and checks the region_seeds cap before
             submitting anything. Edit SMOKE below for a fast wiring check. Does NOT plot.

             `arms` optionally restricts the run to a comma-separated list of arm labels,
             or to one group name from KNOBS, so a follow-up block can be added without
             re-running what is already on disk. `report` pools whatever it finds, so
             partial runs accumulate:
               run region_swap_combo      # just the 3 new region_swap combination arms
               run min1,neutral           # two specific arms

  report     Read the saved pkls back, write the long CSV (one row per arm x seed), print
             the per-knob comparison table, and save the small-multiples figure.
             Re-runnable to tweak the output.

  maps [arms]
             WHERE each arm puts restoration. One pooled selection-frequency (RFOP) map per
             arm - every non-dominated plan from every seed - as a small-multiples grid on
             a SHARED colour scale, plus a second grid of the same maps as a difference
             from `base` (symmetric about zero: red = selected more than base, blue =
             less). One map per arm rather than one per run, since pooling across seeds is
             what separates a real spatial shift from seed noise.

Fixed across all arms so the knobs are isolated:
  - sampling_strategy="region_grow" (under region_evolve, resto_anom.py:1753-1757 hardcodes
    growth_bias='neutral' on the sampler and never passes random_share, so two of the five
    knobs would be inert).
  - warm_seeding=False: resto_anom.py:2235-2246 wraps the sampler in WarmStartSampling,
    which would overwrite part of the initial population and dilute exactly the three
    sampling-only knobs. run_custom_nsga2.py already runs with WARM_SEEDING=False.
  - hv_patience = n_generations + 1, i.e. HV early stopping OFF, so every arm gets an equal
    budget. Convergence speed is recovered as gens_to_90pct instead.
  - min_patch_size=2 (production value). See the cap check in _preflight().
"""
# --- cap nested numpy/BLAS threading BEFORE importing numpy-heavy modules ----
# Spawned worker processes re-import this module top-to-bottom and inherit this env, so
# the ProcessPool x per-run eval threads do not oversubscribe on top of threaded BLAS.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import csv
import sys
import glob
import time
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from _common import (
    REPO_ROOT, RESULTS_DIR, DIAG_DIR, load_ic, use_agg, spread_metrics,
)
import numpy as np
from Core_optimisation.resto_anom import run_optimization_instance
from Core_optimisation.spatial_operations import _label_components, build_restoration_neighbor_table


# ===========================================================================
# run
# ===========================================================================
SMOKE = False                     # True = fast wiring check; False = real sweep
# Process-level parallelism ONLY: each run evaluates sequentially, and GRID_WORKERS runs
# execute concurrently as separate processes. GRID_WORKERS <= 1 = sequential fallback.
# Bound GRID_WORKERS by RAM, not cores: each worker holds its own ~1.3M-pixel raster copy.
GRID_WORKERS = 6
RUN_POP_SIZE = 50
RUN_N_GENERATIONS = 60
RUN_RANDOM_SEEDS = [606, 707, 808]
OBJECTIVES = ["restoration_benefit", "cost"]

# Baseline = the production region config (run_custom_nsga2.py:126-170). Every arm below
# changes exactly ONE of the five region_* keys.
BASE_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01, "normalize_objectives": True,
    "burden_sharing": "no", "rp_formulation": "sum", "rp_threshold": 0.0,
    "sampling_strategy": "region_grow", "repair_scored": False, "min_patch_size": 2,
    "region_seeds": 25, "region_seeds_min": 5, "region_growth_bias": "scored",
    "region_mutation_edits": 100, "region_random_share": 0.5,
    "region_crossover": "hux",   # "hux" (pixel-level, default) | "region_swap"
}

# (arm label, group, level label, override). `group` decides which report section the arm
# appears in; `level` is its x-axis label within that section. "base" carries no override
# and is the reference row repeated at the top of every group.
ARMS = [
    ("base",     None,                    "-",           {}),
    ("seeds5",   "region_seeds",          "5",           {"region_seeds": 5}),
    ("seeds50",  "region_seeds",          "50",          {"region_seeds": 50}),
    ("seeds100", "region_seeds",          "100",         {"region_seeds": 100}),
    ("seeds500", "region_seeds",          "500",         {"region_seeds": 500}),
    ("min1",     "region_seeds_min",      "1",           {"region_seeds_min": 1}),
    ("min25",    "region_seeds_min",      "25",          {"region_seeds_min": 25}),  # == seeds
    ("neutral",  "region_growth_bias",    "neutral",     {"region_growth_bias": "neutral"}),
    ("edits25",  "region_mutation_edits", "25",          {"region_mutation_edits": 25}),
    ("edits400", "region_mutation_edits", "400",         {"region_mutation_edits": 400}),
    ("share0",   "region_random_share",   "0.0",         {"region_random_share": 0.0}),
    ("share1",   "region_random_share",   "1.0",         {"region_random_share": 1.0}),
    # Not one of the five knobs, but a control on all of them: with the default HUX
    # crossover the sampler's regions are shredded within a couple of generations, so the
    # three initial-population-only knobs (seeds / seeds_min / random_share) may show no
    # effect for a reason that has nothing to do with those knobs. This arm swaps in the
    # contiguity-preserving RegionSwapCrossover that region_evolve already uses, so the
    # sweep can tell "knob does nothing" apart from "crossover erased the knob".
    ("regionswap", "region_crossover",    "swap",        {"region_crossover": "region_swap"}),

    # --- follow-up: the two diversity winners, combined -----------------------
    # Round 1 found min1 and neutral are the only arms that deliver diversity, and they win
    # on DIFFERENT axes: neutral on spatial distinctness (jaccard 0.27 vs base 0.75, 117
    # cells, spread 20), min1 on objective range (benefit range 12.3 vs 0.55). Neither
    # dominates the other, so this tests whether the two compose. Default HUX crossover -
    # the region_swap versions below are a separate question.
    ("min1_neutral", "hux_combo", "min1+neutral", {"region_seeds_min": 1,
                                                   "region_growth_bias": "neutral"}),

    # --- follow-up block: rescuing region_swap --------------------------------
    # Round 1 (60 gens, 3 seeds) found region_swap is the ONLY setting that preserves the
    # seeded structure (25 patches at ~1746 px, vs ~1690 fragments of ~13 px under HUX) -
    # but the population collapses to a single plan by generation 1 (n_nd=1, zero objective
    # range, HV falling 10.8%). Its problem is diversity, not structure. min1 and neutral
    # were the two arms that DID produce diversity under HUX (n_nd 35 and 39 vs base 5), so
    # this is a 2x2 of seeds_min x growth_bias under region_swap. `regionswap` above is the
    # (min=5, scored) corner, so only the other three corners are new.
    ("swap_min1",         "region_swap_combo", "min1",         {"region_crossover": "region_swap",
                                                                "region_seeds_min": 1}),
    ("swap_neutral",      "region_swap_combo", "neutral",      {"region_crossover": "region_swap",
                                                                "region_growth_bias": "neutral"}),
    ("swap_min1_neutral", "region_swap_combo", "min1+neutral", {"region_crossover": "region_swap",
                                                                "region_seeds_min": 1,
                                                                "region_growth_bias": "neutral"}),
]

# Order the knob groups appear in the report table / figure columns.
KNOBS = ["region_seeds", "region_seeds_min", "region_growth_bias",
         "region_mutation_edits", "region_random_share", "hux_combo",
         "region_crossover", "region_swap_combo"]

# Groups that are combinations rather than a single BASE_PARAMS key. Their report section
# is anchored on `regionswap` (the min=5/scored corner) instead of `base`, since the
# question is what the combinations add ON TOP of plain region_swap.
COMBO_GROUPS = {"region_swap_combo": "regionswap"}

if SMOKE:
    RUN_POP_SIZE, RUN_N_GENERATIONS = 12, 2
    RUN_RANDOM_SEEDS = [101]
    GRID_WORKERS = 2
    ARMS = [a for a in ARMS if a[0] in ("base", "regionswap", "swap_min1_neutral")]

LOG_DIR = os.path.join(REPO_ROOT, "outputs", "logs")

# --- worker plumbing (module-level so it pickles under Windows 'spawn') ------
# initial_conditions is shipped to each worker ONCE via the pool initializer (initargs),
# not re-pickled per task, and stashed here.
_IC = None


def _init_worker(ic):
    """Pool initializer: cache the shared, read-only initial conditions."""
    global _IC
    _IC = ic


def _run_one(task):
    """Run a single (arm, seed) optimisation. Returns a small picklable tuple.

    Process-level parallelism IS the parallelism here; evaluation within a run is
    sequential. Redirects this run's verbose stdout/stderr to a per-run
    log file so concurrent runs do not interleave, and restores the streams afterwards (a
    worker process is reused across several tasks).
    """
    run_label, _arm, params, seed = task
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{run_label}.log")
    t0 = time.perf_counter()
    _old_out, _old_err = sys.stdout, sys.stderr
    ok, err = False, None
    try:
        with open(log_path, "w", encoding="ascii", errors="replace") as logf:
            sys.stdout = sys.stderr = logf
            try:
                res = run_optimization_instance(
                    initial_conditions=_IC, scenario_params=params,
                    pop_size=RUN_POP_SIZE, n_generations=RUN_N_GENERATIONS,
                    # SMOKE runs save too, so `report` is testable end to end. They use
                    # seed 101, which REPORT_SEEDS filters out of the real sweep's report.
                    save_results=True, verbose=True,
                    random_seed=seed, use_repair=True, use_patch_approach=False,
                    pixel_tolerance=0.05, algorithm_type="nsga2", run_label=run_label,
                    # equal budget for every arm: no HV early stop, and no warm-start
                    # seeds overwriting what the region sampler produced.
                    hv_patience=RUN_N_GENERATIONS + 1, warm_seeding=False,
                    skip_diagnostics=True,
                )
                # run_optimization_instance returns None on failure rather than raising.
                ok = res is not None
                if not ok:
                    err = "run_optimization_instance returned None"
            except Exception as e:  # noqa: BLE001 - report, don't crash the pool
                err = repr(e)
    finally:
        sys.stdout, sys.stderr = _old_out, _old_err
    elapsed = time.perf_counter() - t0
    return run_label, ok, elapsed, err


def _build_tasks(only=None):
    """Build the flat list of (run_label, arm, params, seed) runs.

    `only` is an optional comma-separated arm filter, so a follow-up block can be run
    without re-running the arms already on disk.
    """
    wanted = {a.strip() for a in only.split(",")} if only else None
    tasks = []
    for seed in RUN_RANDOM_SEEDS:
        for arm, _grp, _lvl, override in ARMS:
            if wanted is not None and arm not in wanted:
                continue
            params = {**BASE_PARAMS, **override}
            tasks.append((f"regionop_{arm}_seed_{seed}", arm, params, seed))
    return tasks


def _preflight(ic, arms_to_run=None):
    """Print the arm table and verify nothing silently rewrites an arm's parameters.

    Two ways an arm can end up running something other than its label says:
      1. resto_anom.py:1729-1734 caps region_seeds at max_action_pixels // min_patch_size
         when min_patch_size > 1, and clamps region_seeds_min to match.
      2. spatial_operations.py:990 clamps the per-individual seed count to
         s_lo = max(1, min(region_seeds_min, region_seeds)), so region_seeds_min > seeds
         silently collapses to a fixed seed count.
    Both are cheap to check here and expensive to discover after the full sweep.
    """
    n_rest = int(ic["n_restoration_pixels"])
    budget = int(BASE_PARAMS["max_restoration_fraction"] * n_rest)   # resto_anom.py:626
    S = int(BASE_PARAMS["min_patch_size"])
    cap = max(1, budget // S) if S > 1 else None

    print(f"\nEligible restoration pixels: {n_rest:,}   budget: {budget:,} px "
          f"({BASE_PARAMS['max_restoration_fraction']:.0%})   min_patch_size: {S}")
    if cap is not None:
        print(f"region_seeds cap (budget // min_patch_size): {cap:,}")

    print(f"\n{'arm':>18} {'group':>18} {'seeds':>7} {'min':>5} {'bias':>8} "
          f"{'edits':>6} {'share':>6} {'crossover':>11} {'avg patch':>10}")
    problems = []
    for arm, group, _lvl, override in ARMS:
        if arms_to_run is not None and arm not in arms_to_run:
            continue
        p = {**BASE_PARAMS, **override}
        seeds, smin = int(p["region_seeds"]), int(p["region_seeds_min"])
        eff_lo = max(1, min(smin, seeds))
        print(f"{arm:>18} {str(group or '-'):>18} {seeds:>7} {smin:>5} "
              f"{p['region_growth_bias']:>8} {p['region_mutation_edits']:>6} "
              f"{p['region_random_share']:>6} {p['region_crossover']:>11} "
              f"{budget / max(seeds, 1):>10.0f}")
        if cap is not None and seeds > cap:
            problems.append(f"{arm}: region_seeds {seeds} > cap {cap} -> silently capped")
        if smin > seeds:
            problems.append(f"{arm}: region_seeds_min {smin} > region_seeds {seeds} "
                            f"-> seed count clamped to {eff_lo}")

    if problems:
        print("\n" + "!" * 76)
        print("PRE-FLIGHT: these arms will NOT run the parameters their label claims:")
        for msg in problems:
            print(f"  - {msg}")
        print("!" * 76)
    else:
        print("\nPre-flight OK: every arm runs the parameters its label claims.")
    return not problems


def cmd_run(only=None):
    """Run the sweep. `only` = comma-separated arm labels, or a group name from KNOBS,
    to run a subset (e.g. a follow-up block) without re-running what is already on disk."""
    known = {a for a, _g, _l, _o in ARMS}
    if only:
        # allow a group name as shorthand for all its arms
        expanded = set()
        for tok in (t.strip() for t in only.split(",")):
            grp = [a for a, g, _l, _o in ARMS if g == tok]
            expanded.update(grp if grp else {tok})
        unknown = expanded - known
        if unknown:
            print(f"Unknown arm(s): {', '.join(sorted(unknown))}")
            print(f"Known arms: {', '.join(sorted(known))}")
            print(f"Known groups: {', '.join(KNOBS)}")
            return
        only = ",".join(sorted(expanded))

    ic = load_ic(OBJECTIVES)
    _init_worker(ic)  # stash for the serial fallback path

    arms_to_run = set(only.split(",")) if only else None
    if not _preflight(ic, arms_to_run):
        print("\nAborting: fix the arm definitions above, or adjust min_patch_size.")
        return

    tasks = _build_tasks(only)
    if not tasks:
        print("No arms selected.")
        return
    n_runs = len(tasks)
    ok = 0
    t0 = time.perf_counter()

    mode = f"PARALLEL ({GRID_WORKERS} workers)" if GRID_WORKERS > 1 else "SERIAL"
    print(f"\n===== Region-param sweep: {len(ARMS)} arms x {len(RUN_RANDOM_SEEDS)} seeds "
          f"= {n_runs} runs, {mode}, pop={RUN_POP_SIZE}, gens={RUN_N_GENERATIONS} =====")
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
        # Serial fallback: same runs, one at a time, still routed through _run_one so each
        # run keeps writing its own per-run log file.
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
    print(f"\n===== SWEEP COMPLETE: {ok}/{n_runs} runs in {elapsed/60:.1f} min =====")
    print("Pkls: outputs/results_files/res_<timestamp>_regionop_<arm>_seed_<seed>.pkl")
    if SMOKE:
        print("SMOKE mode: tiny pkls at seed 101; REPORT_SEEDS keeps them out of the "
              "real sweep's report.")
    print("Next: pixi run python Debugs_tests/region_param_sweep.py report")
    print("Check an arm actually got its parameters:")
    print(f"  grep 'Using region-growing operators' {LOG_DIR}\\regionop_*.log")


# ===========================================================================
# report
# ===========================================================================
OUT_DIR = DIAG_DIR
PKL_GLOB = "*_regionop_*.pkl"
# REPORT_SEEDS: which sweep to analyse. outputs/results_files holds runs from many earlier
# sweeps; set to the seed list of the sweep you want. None = every seed found.
REPORT_SEEDS = set(RUN_RANDOM_SEEDS)
# Safety backstop: never pool runs from a different objective formulation (older
# 'restoration_potential' vs newer 'restoration_benefit').
BENEFIT_OBJ = "restoration_benefit"

_TS_RE = re.compile(r"res_(\d{8}_\d{4})_regionop_")
_NAME_RE = re.compile(r"_regionop_(.+)_seed_(\d+)\.pkl$")

# Headline metrics for the figure: (column key, axis label, whether higher is better)
FIG_METRICS = [
    ("hv_last", "final HV", True),
    ("n_nd", "non-dominated count", True),
    ("centroid_spread", "centroid spread (px)", True),
    ("n_components", "patches per plan", None),
]


def _newness_key(path):
    """Sort key for 'newest': embedded run timestamp (res_YYYYMMDD_HHMM_...) first, file
    mtime as a tiebreak. The timestamp string sorts chronologically as text."""
    m = _TS_RE.search(os.path.basename(path))
    return (m.group(1) if m else "", os.path.getmtime(path))


def gather_pkls(seeds=None):
    """(arm, seed) -> newest matching pkl path.

    Groups every matching file by (arm, seed) and keeps the newest per combo, so re-running
    some arms supersedes older pkls without having to delete anything.
    """
    seeds = REPORT_SEEDS if seeds is None else seeds
    by = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, PKL_GLOB)):
        m = _NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        if seeds is not None and seed not in seeds:
            continue
        key = (arm, seed)
        if key not in by or _newness_key(path) > _newness_key(by[key]):
            by[key] = path
    return by


def _get_nbr_bundle(pkl_paths):
    """Build the neighbour table once from a pkl's initial_conditions (reload if absent)."""
    for p in pkl_paths:
        with open(p, "rb") as f:
            ic = pickle.load(f).get("initial_conditions", {})
        if ic.get("shape") is not None and ic.get("restoration_eligible_indices") is not None:
            nbr, rows, cols = build_restoration_neighbor_table(ic)
            return nbr, rows, cols, tuple(ic["shape"]), ic
    ic = load_ic(OBJECTIVES)
    nbr, rows, cols = build_restoration_neighbor_table(ic)
    return nbr, rows, cols, tuple(ic["shape"]), ic


def _adjacency(sel, nbr):
    valid = nbr >= 0
    sel_nb = np.where(valid, sel[np.clip(nbr, 0, None)], False) & valid
    return int(sel_nb[sel].sum() // 2)


def _mean_pairwise_jaccard(D, max_pairs=300, rng=None):
    """Mean Jaccard overlap between non-dominated plans (1 = all plans identical).

    Subsampled to max_pairs because the front can hold ~pop_size plans of ~1M bits each.
    """
    n = len(D)
    if n < 2:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    vals = []
    for a, b in pairs:
        inter = np.count_nonzero(D[a] & D[b])
        union = np.count_nonzero(D[a] | D[b])
        if union:
            vals.append(inter / union)
    return float(np.mean(vals)) if vals else float("nan")


def _metrics_from_run(res, nbr_bundle):
    """One flat row of metrics for a single run, or None if the run is not comparable."""
    nbr, rows, cols, shape, ic = nbr_bundle
    names = list(res["objective_names"])
    if BENEFIT_OBJ not in names:
        return None
    n_rest = nbr.shape[0]

    row = {}

    # --- search ---------------------------------------------------------
    ai = res.get("algorithm_info", {})
    hv = np.asarray(ai.get("hypervolume_history", []), float)
    if hv.size:
        row["hv_first"] = float(hv[0])
        row["hv_last"] = float(hv[-1])
        row["hv_change_pct"] = (float(hv[-1] - hv[0]) / abs(hv[0]) * 100.0
                                if hv[0] else float("nan"))
        # convergence speed: first generation within 10% of the final HV. With early
        # stopping disabled this is the only signal of "how fast", since every arm runs
        # the full generation budget.
        reached = np.where(hv >= 0.9 * hv[-1])[0]
        row["gens_to_90pct"] = int(reached[0]) + 1 if reached.size else len(hv)
    else:
        row.update(hv_first=float("nan"), hv_last=float("nan"),
                   hv_change_pct=float("nan"), gens_to_90pct=-1)
    row["gens_run"] = int(ai.get("actual_generations", len(hv)))
    row["termination_reason"] = str(ai.get("termination_reason", ""))

    # --- front ----------------------------------------------------------
    F = np.asarray(res["objectives_raw"], float)
    nd = np.asarray(res["is_nondominated"], bool)
    Fn = F[nd]
    row["n_nd"] = int(Fn.shape[0])
    row["n_unique_F"] = int(np.unique(np.round(Fn, 6), axis=0).shape[0]) if Fn.size else 0
    for j, nm in enumerate(names):
        c = Fn[:, j] if Fn.size else np.array([np.nan])
        short = "benefit" if nm == BENEFIT_OBJ else ("cost" if "cost" in nm else nm[:10])
        row[f"{short}_min"] = float(np.min(c))
        row[f"{short}_max"] = float(np.max(c))
        row[f"{short}_range"] = float(np.max(c) - np.min(c))
        row[f"{short}_cv"] = (float(np.std(c) / abs(np.mean(c)))
                              if np.mean(c) else float("nan"))
    if Fn.shape[0] > 1 and len(names) >= 2:
        a, b = Fn[:, 0], Fn[:, 1]
        row["obj_corr"] = (float(np.corrcoef(a, b)[0, 1])
                           if a.std() > 0 and b.std() > 0 else float("nan"))
    else:
        row["obj_corr"] = float("nan")

    # --- structure + diversity (over the non-dominated plans) -----------
    dec = np.asarray(res["decisions"])
    D = dec[nd][:, :n_rest].astype(bool)
    nc, ms, mn, ad = [], [], [], []
    for sel in D:
        sizes = [c.size for c in _label_components(sel, shape, rows, cols)]
        nc.append(len(sizes))
        ms.append(np.mean(sizes) if sizes else 0)
        mn.append(min(sizes) if sizes else 0)
        ad.append(_adjacency(sel, nbr))
    row["n_components"] = float(np.mean(nc)) if nc else 0.0
    row["mean_patch"] = float(np.mean(ms)) if ms else 0.0
    row["min_patch"] = float(np.min(mn)) if mn else 0.0
    row["adjacency"] = float(np.mean(ad)) if ad else 0.0

    sm = spread_metrics(dec[nd], ic)
    row["centroid_spread"] = sm["centroid_spread"]
    row["cells_touched"] = sm["cells_touched"]
    row["cells_total"] = sm["cells_total"]
    row["top_cell_share"] = sm["top_cell_share"]
    row["mean_jaccard"] = _mean_pairwise_jaccard(D)
    return row


ARM_KNOB = {arm: grp for arm, grp, _lvl, _ov in ARMS}
ARM_LEVEL = {arm: lvl for arm, _grp, lvl, _ov in ARMS}


def _group_ref(group):
    """Arm whose row is repeated at the top of a group's section, and its level label.

    Single-knob groups are read against `base`; combination groups against the arm named
    in COMBO_GROUPS (plain region_swap), since the question there is what the combination
    adds on top of that, not on top of the production baseline.
    """
    ref = COMBO_GROUPS.get(group, "base")
    lvl = ARM_LEVEL.get(ref, "-") if ref != "base" else str(BASE_PARAMS.get(group, "-"))
    return ref, f"{lvl}*"


# (column key, header, format spec). 'g' for quantities whose magnitude is unknown in
# advance (HV is ~1e-6 and would print as 0.0000 under a fixed-decimal spec). Width is
# derived from the header so the table never misaligns when a header is the wider of the two.
TABLE_COLS = [
    ("n_nd", "n_nd", ".0f"), ("hv_last", "hv_last", ".4g"),
    ("hv_change_pct", "hv_chg%", ".2f"), ("gens_to_90pct", "gen@90%", ".1f"),
    ("n_components", "patches", ".0f"), ("mean_patch", "mean_px", ".0f"),
    ("centroid_spread", "spread", ".0f"), ("cells_touched", "cells", ".0f"),
    ("mean_jaccard", "jaccard", ".3f"), ("benefit_range", "ben_range", ".4g"),
    ("cost_range", "cost_range", ".4g"),
]


def _print_table(by_arm):
    """Per-knob comparison table, baseline repeated at the top of each group."""
    widths = [max(len(h), 9) for _, h, _ in TABLE_COLS]
    hdr = f"{'arm':>18} {'level':>12} {'n':>3} " + " ".join(
        f"{h:>{w}}" for (_, h, _), w in zip(TABLE_COLS, widths))
    for knob in KNOBS:
        arms = [a for a, k, _l, _o in ARMS if k == knob and a in by_arm]
        if not arms:
            continue
        print(f"\n--- {knob} " + "-" * max(0, 68 - len(knob)))
        print(hdr)
        ref, ref_lvl = _group_ref(knob)
        ordered = ([ref] + arms) if ref in by_arm else arms
        for arm in ordered:
            rows = by_arm[arm]
            lvl = ref_lvl if arm == ref else ARM_LEVEL[arm]
            line = f"{arm:>18} {lvl:>12} {len(rows):>3} "
            cells = []
            for (key, _h, spec), w in zip(TABLE_COLS, widths):
                # mean_jaccard is NaN for a single-solution front, so a column can be
                # all-NaN; nanmean would warn. Filter first and print '-'.
                vals = [r[key] for r in rows if np.isfinite(r[key])]
                v = np.mean(vals) if vals else float("nan")
                cells.append(f"{v:>{w}{spec}}" if np.isfinite(v) else f"{'-':>{w}}")
            print(line + " ".join(cells))


def _plot(by_arm, out_png):
    use_agg()
    import matplotlib.pyplot as plt

    knobs = [k for k in KNOBS if any(kk == k and a in by_arm for a, kk, _l, _o in ARMS)]
    if not knobs:
        print("Nothing to plot.")
        return
    nrow, ncol = len(FIG_METRICS), len(knobs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow), squeeze=False)

    for j, knob in enumerate(knobs):
        arms = [a for a, k, _l, _o in ARMS if k == knob and a in by_arm]
        # The reference arm sits inside each group at ITS OWN level for this knob, so the
        # axis reads as a parameter axis rather than "reference plus alternatives".
        ref, ref_lvl = _group_ref(knob)
        ref_rows = by_arm.get(ref, [])
        group = ([(ref, ref_lvl.rstrip("*"))] if ref_rows else []) + \
                [(a, ARM_LEVEL[a]) for a in arms]
        # numeric knobs sort numerically; categorical ones (growth_bias, crossover,
        # combos) keep the order they are declared in ARMS
        try:
            group = sorted(group, key=lambda it: float(it[1]))
        except ValueError:
            pass
        labels = [f"{lv}*" if a == ref else lv for a, lv in group]

        for i, (metric, ylab, _higher) in enumerate(FIG_METRICS):
            ax = axes[i][j]
            means, los, his = [], [], []
            for a, _lv in group:
                vals = np.array([r[metric] for r in by_arm[a]], float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    means.append(np.nan); los.append(0.0); his.append(0.0)
                    continue
                means.append(vals.mean())
                los.append(vals.mean() - vals.min())
                his.append(vals.max() - vals.mean())
            x = np.arange(len(group))
            ax.errorbar(x, means, yerr=[los, his], fmt="o-", capsize=3, ms=5, lw=1.2)
            if ref_rows:
                bl = np.nanmean([r[metric] for r in ref_rows])
                ax.axhline(bl, color="grey", ls="--", lw=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_xlim(-0.5, len(group) - 0.5)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if i == 0:
                ax.set_title(knob.replace("region_", ""), fontsize=10)
            ax.tick_params(labelsize=8)

    fig.suptitle(f"Region-operator OAT sweep  (pop={RUN_POP_SIZE}, gens={RUN_N_GENERATIONS}, "
                 f"seeds={len(RUN_RANDOM_SEEDS)}; * = baseline, dashed = baseline mean)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nFigure -> {out_png}")


# ===========================================================================
# maps
# ===========================================================================
# Arms are compared as ONE pooled selection-frequency map each (all non-dominated plans
# from all seeds), not 3 maps per arm - the question is where an arm sends restoration,
# and pooling across seeds is what separates a real spatial shift from seed noise.
MAP_NCOL = 4
MAP_CMAP = "YlOrRd"        # sequential, single ramp light->dark (house convention)
MAP_DIFF_CMAP = "RdBu_r"   # diverging, neutral midpoint, used symmetric about 0


def _arm_frequency(paths, n_rest):
    """Pooled selection frequency over every non-dominated plan in `paths`.

    Returns (freq over restoration pixels in [0,1], n_plans). Pkls are opened one at a
    time and only the frequency vector is kept, so this stays flat in memory.
    """
    acc = np.zeros(n_rest, dtype=np.float64)
    n_plans = 0
    for path in paths:
        with open(path, "rb") as f:
            res = pickle.load(f)
        if BENEFIT_OBJ not in list(res["objective_names"]):
            continue
        nd = np.asarray(res["is_nondominated"], bool)
        D = np.asarray(res["decisions"])[nd][:, :n_rest].astype(bool)
        if D.size == 0:
            continue
        acc += D.sum(axis=0)
        n_plans += D.shape[0]
    return (acc / n_plans if n_plans else acc), n_plans


def _map_grid(panels, ic, out_png, title, cmap, vmin, vmax, cbar_label, ncol=None):
    """Small-multiples grid of maps on ONE shared colour scale and ONE shared colorbar.

    A per-panel scale would make the panels look similar regardless of how different they
    are; the shared scale is what makes the comparison honest.

    `ncol` overrides MAP_NCOL, so a caller with e.g. 5 panels can keep them on one row
    rather than leaving a blank cell.
    """
    use_agg()
    import matplotlib.pyplot as plt

    shape = tuple(ic["shape"])
    rest_idx = np.asarray(ic["restoration_eligible_indices"])
    rows, cols = np.unravel_index(rest_idx, shape)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

    # frozen grey backdrop of all eligible pixels, so "not selected" reads as context
    base = np.full(shape, np.nan)
    base.flat[rest_idx] = 0.0
    grey = np.where(~np.isnan(base[r0:r1, c0:c1]), 0.0, np.nan)

    n = len(panels)
    ncol = min(ncol or MAP_NCOL, n)
    nrow = int(np.ceil(n / ncol))
    # Height = panels + a fixed strip for the suptitle and the shared colorbar, so the
    # strip does not eat into the panels (and does not collapse on a one-row grid).
    # Width has a floor: a single-panel grid is only 3.1 in wide, too narrow for the
    # suptitle and the colorbar, which then run off both edges.
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(3.1 * ncol, 7.0), 3.0 * nrow + 1.2),
                             squeeze=False)
    im = None
    for k, (label, sub, vals) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        ax.imshow(grey, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        arr = np.full(shape, np.nan)
        arr.flat[rest_idx] = vals
        im = ax.imshow(arr[r0:r1, c0:c1], cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(sub, fontsize=7.5, color="0.35")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # Reserve a FIXED height in inches for the colorbar strip and the suptitle, converted
    # to a fraction. A fixed fraction collapses on a one-row grid (short figure) and the
    # colorbar lands on top of the panel labels.
    fig_h = fig.get_figheight()
    fig.suptitle(title, fontsize=11, wrap=True)
    fig.tight_layout(rect=(0, 0.95 / fig_h, 1, 1.0 - 0.45 / fig_h))
    cax = fig.add_axes([0.25, 0.50 / fig_h, 0.5, 0.14 / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  -> {out_png}")


def cmd_maps(only=None):
    """Per-arm pooled RFOP maps, plus the same maps as a difference from the reference."""
    by_path = gather_pkls()
    if not by_path:
        print(f"No pkls matching {PKL_GLOB} in {RESULTS_DIR} for seeds {REPORT_SEEDS}.")
        return
    wanted = {a.strip() for a in only.split(",")} if only else None

    paths_by_arm = {}
    for (arm, _seed), path in sorted(by_path.items()):
        if wanted is not None and arm not in wanted:
            continue
        paths_by_arm.setdefault(arm, []).append(path)
    if not paths_by_arm:
        print("No arms selected.")
        return

    _, _, _, _, ic = _get_nbr_bundle(sorted(by_path.values(), key=_newness_key, reverse=True))
    n_rest = int(np.asarray(ic["restoration_eligible_indices"]).size)

    order = [a for a, _g, _l, _o in ARMS if a in paths_by_arm]
    freqs = {}
    print(f"Pooling non-dominated plans for {len(order)} arm(s)...")
    for arm in order:
        freqs[arm], n_plans = _arm_frequency(paths_by_arm[arm], n_rest)
        touched = int((freqs[arm] > 0).sum())
        print(f"  {arm:>18}: {n_plans:>4} plans, {touched:>7,} px ever selected "
              f"({100.0 * touched / n_rest:.1f}% of eligible)")

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # --- absolute: where does each arm put restoration? ---------------------
    panels = []
    for arm in order:
        f = freqs[arm] * 100.0
        touched = 100.0 * (freqs[arm] > 0).sum() / n_rest
        panels.append((f"{arm}  ({ARM_LEVEL.get(arm, '?')})",
                       f"{touched:.1f}% of eligible touched", np.where(f > 0, f, np.nan)))
    _map_grid(panels, ic, os.path.join(OUT_DIR, f"region_param_maps_{ts}.png"),
              "Where each arm puts restoration - selection frequency across the "
              "non-dominated set, pooled over seeds",
              MAP_CMAP, 0, 100, "% of non-dominated plans selecting the pixel")

    # --- difference from the reference arm ----------------------------------
    # Symmetric limits so the neutral midpoint sits exactly at zero (no shift = no colour).
    ref = "base" if "base" in freqs else order[0]
    others = [a for a in order if a != ref]
    if others:
        diffs = {a: (freqs[a] - freqs[ref]) * 100.0 for a in others}
        # Every arm spends the same pixel budget, so sum(freq) is the mean plan size for
        # both arms. Half the summed absolute difference is therefore the amount of that
        # budget that MOVED somewhere else - 0% = identical footprint, 100% = disjoint.
        mass = float(freqs[ref].sum())
        # Colour limit from a high percentile of the non-zero differences, not the max: a
        # handful of 0->100 pixels would otherwise saturate the scale and wash every panel
        # out. Values beyond the limit clip to the end colour.
        pooled = np.concatenate([np.abs(d)[np.abs(d) > 0] for d in diffs.values()])
        lim = float(np.percentile(pooled, 99)) if pooled.size else 1.0
        lim = max(lim, 1e-6)
        panels = []
        for arm in others:
            d = diffs[arm]
            moved = float(np.abs(d).sum() / 100.0 / 2.0 / mass) if mass else float("nan")
            panels.append((f"{arm}  ({ARM_LEVEL.get(arm, '?')})",
                           f"{moved:.0%} of the budget relocated",
                           np.where(np.abs(d) > 0, d, np.nan)))
        _map_grid(panels, ic, os.path.join(OUT_DIR, f"region_param_maps_diff_{ts}.png"),
                  f"Shift relative to '{ref}' - red = arm selects it more, blue = less "
                  f"(colour scale clipped at the 99th percentile)",
                  MAP_DIFF_CMAP, -lim, lim,
                  f"difference in selection frequency vs {ref} (percentage points)")


def cmd_report():
    by_path = gather_pkls()
    if not by_path:
        print(f"No pkls matching {PKL_GLOB} in {RESULTS_DIR} for seeds {REPORT_SEEDS}.")
        print("Run the sweep first: pixi run python Debugs_tests/region_param_sweep.py run")
        return
    print(f"Found {len(by_path)} run pkl(s) across "
          f"{len({a for a, _ in by_path})} arm(s).")

    nbr_bundle = _get_nbr_bundle(sorted(by_path.values(), key=_newness_key, reverse=True))

    rows, skipped = [], 0
    for (arm, seed), path in sorted(by_path.items()):
        with open(path, "rb") as f:
            res = pickle.load(f)
        m = _metrics_from_run(res, nbr_bundle)
        if m is None:
            skipped += 1
            continue
        rows.append({"arm": arm, "knob": ARM_KNOB.get(arm, "?"),
                     "level": ARM_LEVEL.get(arm, "?"), "seed": seed, **m})
        print(f"  {arm:>10} seed {seed}: {m['n_nd']} non-dominated, "
              f"HV {m['hv_last']:.4g}")
    if skipped:
        print(f"({skipped} pkl(s) skipped: objective '{BENEFIT_OBJ}' not in the run)")
    if not rows:
        print("No comparable runs found.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(OUT_DIR, f"region_param_sweep_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="ascii", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV ({len(rows)} rows) -> {csv_path}")

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    _print_table(by_arm)
    _plot(by_arm, os.path.join(OUT_DIR, f"region_param_sweep_{ts}.png"))


# ===========================================================================
COMMANDS = {
    "run": cmd_run,          # optional arg: comma-separated arm labels, or a group name
    "report": lambda a: cmd_report(),
    "maps": cmd_maps,        # optional arg: comma-separated arm labels
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
