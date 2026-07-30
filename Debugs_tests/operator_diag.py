"""Region / operator behaviour diagnostics for the region-growing search.

Subcommands (pixi run python Debugs_tests/operator_diag.py <cmd> [arg]):

  sweep                Operator sweep: does reaching CLUSTERED plans unlock objective
                       spread + HV movement? Runs the same 3-objective scenario under
                       different operator arms (scattered / region scored / region neutral
                       / patch) and reports HV movement + cost/clustering spread per arm.
                       Writes operator_sweep_<ts>.csv/.png under diagnostics/.

  instrument           Operator bottleneck probe: mutation vs crossover vs repair. Wraps
                       HUX to observe each generation's parent/child overlap + contiguity,
                       replaying repair and mutation on the same children, to see which
                       operator is responsible for the region front barely improving.
                       Writes operator_bottleneck_probe.csv.

  front-check          Region-growing front check: is it a genuine 3-way trade-off or a
                       relocated collapse? Inspects the non-dominated set of region_fixed /
                       region_diverse / scattered for objective spread + pairwise
                       correlations.

  region-evolve [n]    region_evolve spread test: does the search explore the landscape?
                       Runs the region_evolve strategy (default 30 gens; pass n to change),
                       measures spatial spread of the non-dominated set, and regenerates the
                       RFOP map.
"""
import os
import sys
import csv
import copy
import time
from datetime import datetime
import numpy as np

from _common import REPO_ROOT, RESULTS_DIR, DIAG_DIR, load_ic, use_agg

OBJECTIVES_3 = ["restoration_potential", "spatial_clustering", "cost"]


# ===========================================================================
# sweep  (was operator_sweep.py)
# ===========================================================================
SWEEP_N_GENERATIONS = 40          # equal budget per arm; fast first-pass signal
SWEEP_N_PARTITIONS = 12
SWEEP_N_JOBS = 12
SWEEP_SEEDS = [101]               # add seeds later for stability
SWEEP_PIXEL_TOLERANCE = 0.05

SWEEP_BASE_PARAMS = {
    "max_restoration_fraction": 0.05,
    "spatial_clustering": 0,
    "biotic_effect": 0.01,
    "abiotic_effect": 0.01,
    "normalize_objectives": False,
    "patch_score_temperature": 2.0,
    "patch_repair_top_k": 100,
    "burden_sharing": "no",
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    "clustering_metric": "adjacency",   # pixel-valid; comparable across all arms
}

# (label, use_patch_approach, patch_size, param_overrides)
SWEEP_ARMS = [
    ("scattered",         False, 2,  {"sampling_strategy": "scattered"}),
    ("region_scored_S1",  False, 2,  {"sampling_strategy": "region_grow", "region_seeds": 1,   "region_growth_bias": "scored"}),
    ("region_scored_S25", False, 2,  {"sampling_strategy": "region_grow", "region_seeds": 25,  "region_growth_bias": "scored"}),
    ("region_neutral_S25", False, 2, {"sampling_strategy": "region_grow", "region_seeds": 25,  "region_growth_bias": "neutral"}),
    ("patch_T2",          True,  2,  {}),
    ("patch_T8",          True,  8,  {}),
]


def _sweep_spread_row(ps, obj_names, gen_idx):
    """Return per-objective (mean, cv, rng, best) at a generation index."""
    fmean = np.asarray(ps["f_mean_history"], float)[gen_idx]
    fstd = np.asarray(ps["f_std_history"], float)[gen_idx]
    fmin = np.asarray(ps["f_min_history"], float)[gen_idx]
    fmax = np.asarray(ps["f_max_history"], float)[gen_idx]
    out = {}
    for j, nm in enumerate(obj_names):
        mean = fmean[j]
        out[nm] = dict(mean=mean, cv=(fstd[j] / abs(mean) if mean else float("nan")),
                       rng=fmax[j] - fmin[j], best=fmin[j])
    return out


def cmd_sweep():
    use_agg()
    import matplotlib.pyplot as plt
    from Core_optimisation.resto_anom import run_optimization_instance

    os.makedirs(DIAG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print("=== OPERATOR SWEEP ===")
    print(f"Region=Bern scenario=global_all objectives={OBJECTIVES_3}")
    print(f"Budget: {SWEEP_N_GENERATIONS} gens x {SWEEP_N_PARTITIONS} partitions  seeds={SWEEP_SEEDS}\n")

    ic_base = load_ic(OBJECTIVES_3)

    rows = []
    curves = {}
    for label, use_patch, patch_size, overrides in SWEEP_ARMS:
        params = {**SWEEP_BASE_PARAMS, **overrides}
        # Patch arms need a fresh ic per size (patch init is cached on the dict).
        ic = copy.copy(ic_base) if use_patch else ic_base
        for seed in SWEEP_SEEDS:
            tag = f"{label}_seed{seed}"
            print(f"--- {tag} (patch={use_patch} size={patch_size}) ---")
            t0 = time.perf_counter()
            res = run_optimization_instance(
                initial_conditions=ic, scenario_params=params,
                pop_size=None, n_generations=SWEEP_N_GENERATIONS,
                save_results=False, verbose=False, skip_diagnostics=True,
                hv_patience=SWEEP_N_GENERATIONS + 1, n_jobs=SWEEP_N_JOBS, random_seed=seed,
                use_repair=True, use_patch_approach=use_patch, patch_size=patch_size,
                patch_constraint_type="pixel_count", pixel_tolerance=SWEEP_PIXEL_TOLERANCE,
                save_snapshots=False, n_partitions=SWEEP_N_PARTITIONS, warm_seeding=True,
                run_label=tag)
            dt = time.perf_counter() - t0
            if res is None:
                print(f"  x {tag} returned None ({dt:.0f}s)\n")
                continue
            ai = res["algorithm_info"]
            obj_names = res["objective_names"]
            hv = np.asarray(ai["hypervolume_history"], float)
            ps = ai["population_statistics"]
            first = _sweep_spread_row(ps, obj_names, 0)
            last = _sweep_spread_row(ps, obj_names, -1)
            curves.setdefault(label, []).append(hv)
            hv_chg = (hv[-1] - hv[0]) / abs(hv[0]) * 100 if hv[0] else float("nan")

            def g(d, nm, key):
                return d.get(nm, {}).get(key, float("nan"))
            cost_nm = "implementation_cost"
            clus_nm = "spatial_clustering"
            rp_nm = "restoration_potential"
            row = dict(
                arm=label, seed=seed, wall_s=round(dt, 1),
                hv_first=hv[0], hv_last=hv[-1], hv_change_pct=round(hv_chg, 3),
                cost_cv_first=g(first, cost_nm, "cv"), cost_cv_last=g(last, cost_nm, "cv"),
                cost_best_first=g(first, cost_nm, "best"), cost_best_last=g(last, cost_nm, "best"),
                clus_rng_first=g(first, clus_nm, "rng"), clus_rng_last=g(last, clus_nm, "rng"),
                clus_best_first=g(first, clus_nm, "best"), clus_best_last=g(last, clus_nm, "best"),
                rp_best_first=g(first, rp_nm, "best"), rp_best_last=g(last, rp_nm, "best"),
            )
            rows.append(row)
            print(f"  wall={dt:.0f}s  HV {hv[0]:.4g}->{hv[-1]:.4g} ({hv_chg:+.2f}%)")
            print(f"  cost CV {row['cost_cv_first']:.4f}->{row['cost_cv_last']:.4f}  "
                  f"cost best {row['cost_best_first']:.1f}->{row['cost_best_last']:.1f}")
            print(f"  clustering best(neg adj) {row['clus_best_first']:.0f}->{row['clus_best_last']:.0f}  "
                  f"rp best {row['rp_best_first']:.4g}->{row['rp_best_last']:.4g}\n")

    csv_path = os.path.join(DIAG_DIR, f"operator_sweep_{ts}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV  -> {csv_path}")

    plt.figure(figsize=(8, 5))
    for label, _u, _p, _o in SWEEP_ARMS:
        cs = curves.get(label)
        if not cs:
            continue
        ml = min(len(c) for c in cs)
        stacked = np.vstack([c[:ml] for c in cs])
        base = stacked[:, 0:1]
        norm = stacked / np.where(base == 0, 1, base)
        gens = np.arange(1, ml + 1)
        plt.plot(gens, norm.mean(axis=0), marker="", linewidth=1.6, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume / gen-1 hypervolume (within-arm)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(DIAG_DIR, f"operator_sweep_{ts}.png")
    plt.savefig(png_path, dpi=130)
    print(f"Plot -> {png_path}")


# ===========================================================================
# instrument  (was instrument_operators.py)
# ===========================================================================
INSTR_N_GENERATIONS = 18
INSTR_N_PARTITIONS = 12
INSTR_N_JOBS = 12
INSTR_SEED = 101
INSTR_N_COMP_SAMPLE = 24   # children sampled per generation for component counts

INSTR_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01,
    "normalize_objectives": True, "burden_sharing": "no",
    "rp_formulation": "sum", "rp_threshold": 0.0, "clustering_metric": "adjacency",
    "sampling_strategy": "region_grow", "region_seeds": 25, "region_seeds_min": 1,
    "region_random_share": 0.3, "region_growth_bias": "scored",
    "region_mutation_edits": 100,
}


def _instr_jaccard(A, B):
    """Row-wise Jaccard of two boolean matrices (N, n_rest)."""
    inter = np.logical_and(A, B).sum(1)
    union = np.logical_or(A, B).sum(1)
    return inter / np.maximum(union, 1)


def _make_logging_hux():
    """Build the LoggingHUX class lazily (needs pymoo + resto_anom imported)."""
    from pymoo.operators.crossover.hux import HUX
    from scipy import ndimage
    from Core_optimisation.resto_anom import _build_operators

    class LoggingHUX(HUX):
        """HUX that observes parent/child overlap + contiguity each generation."""
        LOG = []

        def __init__(self, **kw):
            super().__init__(**kw)
            self._probe_repair = None
            self._probe_mut = None
            self._gen = 0

        def _ensure_probes(self, problem):
            if self._probe_repair is not None:
                return
            _, rep, mut, _ = _build_operators(
                problem.initial_conditions, problem.scenario_params, problem,
                False, True, 'pixel_count', 0.05, False)
            self._probe_repair = rep
            self._probe_mut = mut
            ic = problem.initial_conditions
            self._shape = ic['shape']
            self._elig = np.asarray(ic['restoration_eligible_indices'])
            self._n_rest = problem.n_restoration_pixels

        def _components(self, sel_block, sample):
            """Mean connected-components over a sample of selections (boolean, N x n_rest)."""
            W = self._shape[1]
            idx = np.arange(sel_block.shape[0])
            if idx.size > sample:
                idx = np.random.default_rng(0).choice(idx, sample, replace=False)
            vals = []
            for i in idx:
                b = sel_block[i]
                m2 = np.zeros(self._shape, dtype=bool)
                rr, cc = np.divmod(self._elig[b], W)
                m2[rr, cc] = True
                _, nc = ndimage.label(m2)
                vals.append(nc)
            return float(np.mean(vals)) if vals else float('nan')

        def _do(self, problem, X, **kwargs):
            off = super()._do(problem, X, **kwargs)
            try:
                if X.ndim == 3 and off.ndim == 3 and X.shape[0] == 2:
                    self._ensure_probes(problem)
                    self._gen += 1
                    nr = self._n_rest

                    pA_full = np.vstack([X[0], X[0]])          # each child's parent A
                    pB_full = np.vstack([X[1], X[1]])
                    xo_full = np.vstack([off[0], off[1]])       # crossover-only children (full n_var)

                    # crossover + repair (no mutation) -> question (c)
                    xoR = self._probe_repair._do(problem, xo_full.copy())
                    # crossover + mutation + repair (full pipeline) -> question (a)
                    full = self._probe_repair._do(problem, self._probe_mut._do(problem, xo_full.copy()))

                    pA = (pA_full[:, :nr] == 1)
                    pB = (pB_full[:, :nr] == 1)
                    xo = (xo_full[:, :nr] == 1)
                    xoRb = (xoR[:, :nr] == 1)
                    fb = (full[:, :nr] == 1)

                    def maxJ(c):
                        return float(np.mean(np.maximum(_instr_jaccard(c, pA), _instr_jaccard(c, pB))))

                    rec = {
                        "gen": self._gen,
                        "par_par_J": float(np.mean(_instr_jaccard(pA, pB))),
                        "par_comp": self._components(pA, INSTR_N_COMP_SAMPLE),
                        "xo_maxJ": maxJ(xo),
                        "xo_comp": self._components(xo, INSTR_N_COMP_SAMPLE),
                        "xoR_maxJ": maxJ(xoRb),
                        "xoR_comp": self._components(xoRb, INSTR_N_COMP_SAMPLE),
                        "full_maxJ": maxJ(fb),
                        "full_comp": self._components(fb, INSTR_N_COMP_SAMPLE),
                    }
                    LoggingHUX.LOG.append(rec)
            except Exception as e:
                print(f"  [instrument] gen logging failed: {e}")
            return off

    return LoggingHUX


def cmd_instrument():
    import Core_optimisation.resto_anom as ra
    from Core_optimisation.resto_anom import run_optimization_instance

    LoggingHUX = _make_logging_hux()
    LoggingHUX.LOG = []
    ra.HUX = LoggingHUX   # monkeypatch: _build_algorithm constructs crossover=HUX()

    print("=== OPERATOR BOTTLENECK PROBE (region_grow diverse) ===")
    print(f"{INSTR_N_GENERATIONS} gens, seed={INSTR_SEED}\n")
    ic = load_ic(OBJECTIVES_3)

    res = run_optimization_instance(
        initial_conditions=ic, scenario_params=INSTR_PARAMS, pop_size=None,
        n_generations=INSTR_N_GENERATIONS, save_results=False, verbose=False,
        skip_diagnostics=True, hv_patience=INSTR_N_GENERATIONS + 1, n_jobs=INSTR_N_JOBS,
        random_seed=INSTR_SEED, use_repair=True, use_patch_approach=False,
        pixel_tolerance=0.05, save_snapshots=False, n_partitions=INSTR_N_PARTITIONS,
        warm_seeding=False, run_label="instrument")

    rep_log = res.get("repair_diagnostics", []) if res else []
    mut_log = res.get("mutation_diagnostics", []) if res else []

    def rep_at(g):
        for r in rep_log:
            if r.get("generation") == g:
                return r.get("fraction_repaired", float('nan')), r.get("mean_bits_changed", float('nan'))
        return float('nan'), float('nan')

    def mut_at(g):
        for m in mut_log:
            if m.get("generation") == g:
                return m.get("mean_raw_flips", float('nan'))
        return float('nan')

    print("\nPer-generation operator effects "
          "(J = Jaccard vs nearest parent; comp = connected components):")
    hdr = (f"{'gen':>3} {'parJ':>6} {'parComp':>8} | {'xo_J':>6} {'xoComp':>8} | "
           f"{'xoR_J':>6} {'xoRComp':>8} | {'full_J':>6} {'fullComp':>9} | "
           f"{'rep_fire':>8} {'rep_dpix':>8} {'mut_dpix':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for rec in LoggingHUX.LOG:
        g = rec["gen"]
        fire, dpix = rep_at(g)
        mdp = mut_at(g)
        rec2 = {**rec, "rep_fire": fire, "rep_dpix": dpix, "mut_dpix": mdp}
        rows.append(rec2)
        print(f"{g:>3} {rec['par_par_J']:>6.3f} {rec['par_comp']:>8.1f} | "
              f"{rec['xo_maxJ']:>6.3f} {rec['xo_comp']:>8.1f} | "
              f"{rec['xoR_maxJ']:>6.3f} {rec['xoR_comp']:>8.1f} | "
              f"{rec['full_maxJ']:>6.3f} {rec['full_comp']:>9.1f} | "
              f"{fire:>8.3f} {dpix:>8.1f} {mdp:>8.1f}")

    os.makedirs(DIAG_DIR, exist_ok=True)
    csv_path = os.path.join(DIAG_DIR, "operator_bottleneck_probe.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {csv_path}")


# ===========================================================================
# front-check  (was region_front_check.py)
# ===========================================================================
FRONT_N_GENERATIONS = 50
FRONT_N_PARTITIONS = 12
FRONT_N_JOBS = 12
FRONT_SEED = 101
FRONT_PIXEL_TOLERANCE = 0.05

FRONT_BASE = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01,
    "normalize_objectives": True, "burden_sharing": "no",
    "rp_formulation": "sum", "rp_threshold": 0.0,
    "clustering_metric": "adjacency",
}

FRONT_ARMS = [
    ("region_fixed",   {"sampling_strategy": "region_grow", "region_seeds": 25,
                        "region_growth_bias": "scored"}),
    ("region_diverse", {"sampling_strategy": "region_grow", "region_seeds": 25,
                        "region_seeds_min": 1, "region_random_share": 0.3,
                        "region_growth_bias": "scored"}),
    ("scattered",      {"sampling_strategy": "scattered"}),
]


def _front_analyse(res):
    names = res["objective_names"]
    F = np.asarray(res["objectives_raw"], float)          # (n_sol, n_obj)
    nd = np.asarray(res["is_nondominated"], bool)
    Fn = F[nd]
    n_uniq = np.unique(np.round(Fn, 6), axis=0).shape[0]
    print(f"    non-dominated: {Fn.shape[0]} of {F.shape[0]}  ({n_uniq} unique objective vectors)")
    print(f"    {'objective':>22} {'min':>12} {'max':>12} {'range':>12} {'CV':>8}")
    for j, nm in enumerate(names):
        c = Fn[:, j]
        cv = c.std() / abs(c.mean()) if c.mean() else float("nan")
        print(f"    {nm:>22} {c.min():>12.4g} {c.max():>12.4g} {c.max()-c.min():>12.4g} {cv:>8.3f}")
    print("    pairwise correlation across non-dominated set:")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ca, cb = Fn[:, a], Fn[:, b]
            r = float(np.corrcoef(ca, cb)[0, 1]) if ca.std() > 0 and cb.std() > 0 else float("nan")
            print(f"      {names[a][:14]:>14} vs {names[b][:14]:<14} r = {r:+.3f}")


def cmd_front_check():
    from Core_optimisation.resto_anom import run_optimization_instance
    print("=== REGION FRONT CHECK ===")
    print(f"{FRONT_N_GENERATIONS} gens x {FRONT_N_PARTITIONS} partitions  seed={FRONT_SEED}\n")
    ic_base = load_ic(OBJECTIVES_3)

    for label, overrides in FRONT_ARMS:
        params = {**FRONT_BASE, **overrides}
        print(f"--- {label} ---")
        t0 = time.perf_counter()
        res = run_optimization_instance(
            initial_conditions=copy.copy(ic_base), scenario_params=params,
            pop_size=None, n_generations=FRONT_N_GENERATIONS, save_results=False,
            verbose=False, skip_diagnostics=True, hv_patience=FRONT_N_GENERATIONS + 1,
            n_jobs=FRONT_N_JOBS, random_seed=FRONT_SEED, use_repair=True, use_patch_approach=False,
            pixel_tolerance=FRONT_PIXEL_TOLERANCE, save_snapshots=False,
            n_partitions=FRONT_N_PARTITIONS, warm_seeding=False, run_label=label)
        dt = time.perf_counter() - t0
        if res is None:
            print(f"  returned None ({dt:.0f}s)\n")
            continue
        ai = res["algorithm_info"]
        hv = np.asarray(ai["hypervolume_history"], float)
        print(f"    wall={dt:.0f}s  HV {hv[0]:.4g}->{hv[-1]:.4g} "
              f"({(hv[-1]-hv[0])/abs(hv[0])*100:+.2f}%)")
        _front_analyse(res)
        print()


# ===========================================================================
# region-evolve  (was region_evolve_spread_test.py)
# ===========================================================================
REGEV_SEED = 101
REGEV_LABEL = "region_evolve_spread"
REGEV_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01,
    "normalize_objectives": True, "burden_sharing": "no",
    "rp_formulation": "sum", "rp_threshold": 0.0, "clustering_metric": "adjacency",
    "sampling_strategy": "region_evolve", "region_seeds": 25, "region_seeds_min": 5,
    "region_seed_grid": 16, "region_growth_bias": "scored", "region_mutation_edits": 100,
}


def _regev_spread_metrics(dec_nd, ic, grid=16):
    """Spatial-spread summary of the non-dominated restoration plans."""
    shape = ic["shape"]; W = shape[1]
    elig = np.asarray(ic["restoration_eligible_indices"])
    n_rest = elig.size
    D = dec_nd[:, :n_rest].astype(bool)
    rows, cols = np.divmod(elig, W)
    cents = np.array([[rows[d].mean(), cols[d].mean()] for d in D if d.any()])
    cent_spread = float(np.hypot(cents[:, 0].std(), cents[:, 1].std())) if len(cents) else 0.0
    H = shape[0]
    cell = (rows * grid // H) * grid + (cols * grid // W)
    freq = D.mean(axis=0)
    ever = freq > 0
    cells_touched = np.unique(cell[ever]).size
    cells_total = np.unique(cell).size
    sel_per_cell = np.bincount(cell, weights=freq, minlength=cell.max() + 1)
    conc = float(sel_per_cell.max() / sel_per_cell.sum()) if sel_per_cell.sum() else 1.0
    return dict(n_nd=len(D), centroid_spread=cent_spread,
                cells_touched=cells_touched, cells_total=cells_total,
                top_cell_share=conc)


def cmd_region_evolve(arg):
    import glob
    from Core_optimisation.resto_anom import run_optimization_instance
    n_gen = int(arg) if (arg and arg != "run") else 30
    print(f"=== region_evolve spread test ({n_gen} gens, seed {REGEV_SEED}) ===")
    ic = load_ic(["restoration_potential", "spatial_clustering", "cost"])

    t0 = time.perf_counter()
    res = run_optimization_instance(
        initial_conditions=ic, scenario_params=REGEV_PARAMS, pop_size=None,
        n_generations=n_gen, save_results=True, verbose=True, skip_diagnostics=True,
        hv_patience=n_gen + 1, n_jobs=12, random_seed=REGEV_SEED, use_repair=True,
        use_patch_approach=False, pixel_tolerance=0.05, save_snapshots=False,
        n_partitions=12, warm_seeding=False, run_label=REGEV_LABEL)
    dt = time.perf_counter() - t0
    if res is None:
        print("run returned None"); return

    hv = np.asarray(res["algorithm_info"]["hypervolume_history"], float)
    dec = np.asarray(res["decisions"])
    nd = np.asarray(res["is_nondominated"], bool)
    m = _regev_spread_metrics(dec[nd], ic)
    print(f"\nwall={dt:.0f}s  HV {hv[0]:.4g}->{hv[-1]:.4g} ({(hv[-1]-hv[0])/abs(hv[0])*100:+.2f}%)")
    print(f"non-dominated plans: {m['n_nd']}")
    print(f"centroid spread (px): {m['centroid_spread']:.0f}   (higher = plans sit in different areas)")
    print(f"grid cells ever restored: {m['cells_touched']} / {m['cells_total']}")
    print(f"top-cell share of all selections: {m['top_cell_share']*100:.1f}%   (lower = less blob-concentrated)")

    try:
        from visualisations import create_selection_frequency_map
        pkls = sorted(glob.glob(os.path.join(RESULTS_DIR, f"res_*{REGEV_LABEL}*.pkl")),
                      key=os.path.getmtime)
        if pkls:
            out_png = os.path.join(DIAG_DIR, "rfop_region_evolve.png")
            os.makedirs(DIAG_DIR, exist_ok=True)
            create_selection_frequency_map(pkls[-1], save_path=out_png,
                                           title="region_evolve RFOP")
            print(f"RFOP map -> {out_png}   (pkl: {os.path.basename(pkls[-1])})")
    except Exception as e:
        print(f"(RFOP map skipped: {e})")


# ===========================================================================
COMMANDS = {
    "sweep": lambda a: cmd_sweep(),
    "instrument": lambda a: cmd_instrument(),
    "front-check": lambda a: cmd_front_check(),
    "region-evolve": cmd_region_evolve,
}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    arg = argv[2] if len(argv) > 2 else "run"
    if cmd not in COMMANDS:
        print(__doc__)
        print(f"Subcommands: {', '.join(COMMANDS)}")
        return
    COMMANDS[cmd](arg)


if __name__ == "__main__":
    main(sys.argv)
