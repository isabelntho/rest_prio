"""S=2 exploration-footprint diagnostics: how much of the map does the scored benefit
search actually touch?

Subcommands (pixi run python Debugs_tests/coverage_S2.py <cmd> [mode]):

  snapshot     Single-seed S=2 scored benefit run that SAVES per-generation snapshots
               (save_snapshots=True). Writes outputs/intermediate_results/X_history_*.npz
               of shape (n_gens, pop_size, n_var) - the full population at EVERY
               generation, including solutions discarded before the final front. Mirrors
               the scored config of spillover_benefit.py bias.

  animate      Render a per-generation coverage GIF from the newest X_history_*.npz
               (run `snapshot` first). Each frame = population selection frequency at that
               generation; title reports cumulative coverage (% eligible ever touched).

  fullpop      Coverage map from the FULL final population (incl. dominated individuals)
               of the scored biasS2 runs (res_*_biasS2_scored_seed_*.pkl written by
               spillover_benefit.py bias). Final generation only.

  sweep [run|analyse]
               Exploration-vs-benefit price of region_random_share (single seed, scored
               growth): per share, best_benefit vs coverage bought. `analyse` re-plots the
               cumulative-coverage curves from the saved csv without re-running.
"""
import os
import sys
import csv
import glob
import pickle
import numpy as np

from _common import (
    REPO_ROOT, RESULTS_DIR, INTER_DIR, POP_SIZE, N_GENERATIONS, load_ic, use_agg,
    newest_xhist, render_coverage_gif,
)

OUT_DIR = os.path.join(REPO_ROOT, "Debugs_tests", "diagnostics", "bias_S2")

# Scored S=2 config shared by snapshot + sweep (mirrors spillover_benefit.py bias, scored).
SCORED_S2_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01, "normalize_objectives": True,
    "burden_sharing": "no", "rp_formulation": "sum", "rp_threshold": 0.0,
    "region_seeds": 25, "region_seeds_min": 5, "region_mutation_edits": 100,
    "sampling_strategy": "region_grow", "min_patch_size": 2,
    "region_growth_bias": "scored",
}
THRESH = 0.5


# ===========================================================================
# snapshot  (was snapshot_run_S2.py)
# ===========================================================================
def cmd_snapshot():
    from Core_optimisation.resto_anom import run_optimization_instance
    seed = 606
    run_label = f"snap_S2_scored_seed_{seed}"
    ic = load_ic(["restoration_benefit", "cost"])
    res = run_optimization_instance(
        initial_conditions=ic, scenario_params=SCORED_S2_PARAMS,
        pop_size=POP_SIZE, n_generations=N_GENERATIONS, save_results=True,
        verbose=True, n_jobs=12, random_seed=seed, use_repair=True,
        use_patch_approach=False, pixel_tolerance=0.05, algorithm_type="nsga2",
        run_label=run_label, save_snapshots=True,
    )
    xh = res.get("X_history_path")
    print(f"\nX_history_path = {xh}")
    if xh and os.path.exists(xh):
        X = np.load(xh)["X"]
        print(f"X_history shape = {X.shape}  (n_gens, pop_size, n_var)")


# ===========================================================================
# animate  (was animate_coverage_S2.py)
# ===========================================================================
ANIM_STEP = 1                # plot every STEP-th generation (1 = all)
ANIM_FPS = 8


def cmd_animate():
    use_agg()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = newest_xhist()
    print(f"Loading {os.path.basename(path)} ...")
    ic = load_ic(["restoration_benefit", "cost"])
    gif = os.path.join(OUT_DIR, "coverage_per_gen_S2_scored.gif")
    final_cov, n_never, n_rest = render_coverage_gif(
        path, ic, gif, "S=2 scored", step=ANIM_STEP, fps=ANIM_FPS, thresh=THRESH)
    print(f"raster {ic['shape']}, restoration-eligible pixels {n_rest}")
    print(f"\nFinal cumulative coverage (all gens, all individuals): {final_cov:.1f}%")
    print(f"Pixels never touched: {n_never} ({100 - final_cov:.1f}%)")
    print(f"animation -> {gif}")


# ===========================================================================
# fullpop  (was full_pop_coverage_S2.py)
# ===========================================================================
def cmd_fullpop():
    use_agg()
    from utils import pickle_load
    from visualisations import create_selection_frequency_map, _convert_patch_decisions_to_pixel_matrix

    bias = "scored"
    os.makedirs(OUT_DIR, exist_ok=True)
    pkls = sorted(glob.glob(os.path.join(RESULTS_DIR, f"res_*_biasS2_{bias}_seed_*.pkl")))
    if not pkls:
        print(f"No biasS2_{bias} pkls found."); return

    pooled, ic, pinfo = [], None, None
    for p in pkls:
        r = pickle_load(p)                       # raw: 'decisions' = FULL final population
        dec = r["decisions"]                     # (pop_size, n_var) - all individuals
        conv = _convert_patch_decisions_to_pixel_matrix(dec, r["initial_conditions"], results=r)
        pooled.append(np.asarray(conv if conv is not None else dec))
        ic = r["initial_conditions"]; pinfo = r["problem_info"]
        nd = np.asarray(r["is_nondominated"], bool)
        print(f"{os.path.basename(p)}: full pop = {dec.shape[0]}, non-dominated = {int(nd.sum())}")

    X = np.vstack(pooled)
    tmp = os.path.join(OUT_DIR, f"_fullpop_{bias}.pkl")
    with open(tmp, "wb") as f:
        pickle.dump({"decisions": X, "initial_conditions": ic, "n_solutions": X.shape[0],
                     "objectives": np.zeros((X.shape[0], 2)), "problem_info": pinfo,
                     "objective_names": ["restoration_benefit", "implementation_cost"],
                     "scenario_params": {}}, f)   # no is_nondominated -> keep ALL individuals

    png = os.path.join(OUT_DIR, f"coverage_S2_{bias}_fullpop.png")
    create_selection_frequency_map(
        tmp, save_path=png, cmap="YlOrRd", show_eligible=True, action_type="combined",
        title=f"S=2 benefit, {bias}: FULL final population ({X.shape[0]} individuals, incl. dominated)")
    print(f"\ncoverage map -> {png}")


# ===========================================================================
# sweep  (was random_share_sweep_S2.py)
# ===========================================================================
SWEEP_SEED = 606
SWEEP_SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]
CSV_PATH = os.path.join(OUT_DIR, "random_share_sweep_S2.csv")


def _best_benefit(res):
    names = list(res["objective_names"]); bi = names.index("restoration_benefit")
    F = np.asarray(res["objectives_raw"], float)
    nd = np.asarray(res["is_nondominated"], bool)
    return float(F[nd][:, bi].min())          # most negative = most benefit


def _plot_curves(curves):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for share in sorted(curves):
        c = curves[share]
        if c is None:
            continue
        ax.plot(np.arange(1, len(c) + 1), c, label=f"share={share:.2f}")
    ax.set_xlabel("generation")
    ax.set_ylabel("cumulative coverage (% eligible ever touched)")
    ax.set_title("S=2 scored: exploration footprint vs region_random_share")
    ax.legend(title="region_random_share")
    ax.grid(True, alpha=0.3)
    png = os.path.join(OUT_DIR, "random_share_coverage_curves_S2.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"coverage curves -> {png}")


def _sweep_run():
    from Core_optimisation.resto_anom import run_optimization_instance
    os.makedirs(OUT_DIR, exist_ok=True)
    ic = load_ic(["restoration_benefit", "cost"])
    rest_idx = np.asarray(ic["restoration_eligible_indices"])
    n_rest = len(rest_idx)
    print(f"restoration-eligible pixels = {n_rest}")

    rows = []
    curves = {}   # share -> cumulative coverage % per generation
    for share in SWEEP_SHARES:
        label = f"rndshare_S2_share{int(share*100):03d}_seed_{SWEEP_SEED}"
        print(f"\n===== share={share} -> {label} =====")
        res = run_optimization_instance(
            initial_conditions=ic,
            scenario_params={**SCORED_S2_PARAMS, "region_random_share": share},
            pop_size=POP_SIZE, n_generations=N_GENERATIONS, save_results=True,
            verbose=True, n_jobs=12, random_seed=SWEEP_SEED, use_repair=True,
            use_patch_approach=False, pixel_tolerance=0.05, algorithm_type="nsga2",
            run_label=label, save_snapshots=True,
        )
        best = _best_benefit(res)
        nd = np.asarray(res["is_nondominated"], bool)
        dec = np.asarray(res["decisions"])[:, :n_rest] > THRESH
        pareto_cov = 100.0 * (dec[nd].any(axis=0)).sum() / n_rest
        fullpop_cov = 100.0 * (dec.any(axis=0)).sum() / n_rest

        xh = res.get("X_history_path")
        cum_curve = None
        cumulative_cov = float("nan")
        if xh and os.path.exists(xh):
            X = np.load(xh)["X"]                      # (n_gens, pop, n_var)
            sel = X[:, :, :n_rest] > THRESH
            ever = np.zeros(n_rest, dtype=bool)
            cum_curve = np.empty(sel.shape[0], dtype=float)
            for g in range(sel.shape[0]):
                ever |= sel[g].any(axis=0)
                cum_curve[g] = 100.0 * ever.sum() / n_rest
            cumulative_cov = float(cum_curve[-1])
            del X, sel
            try:
                os.remove(xh)                          # reclaim ~2.5 GB
            except OSError:
                pass
        curves[share] = cum_curve
        rows.append((share, best, pareto_cov, fullpop_cov, cumulative_cov))
        print(f"  best_benefit={best:.2f}  pareto_cov={pareto_cov:.1f}%  "
              f"fullpop_cov={fullpop_cov:.1f}%  cumulative_cov={cumulative_cov:.1f}%")

    # write summary csv (with the per-gen curves appended as extra columns)
    max_g = max((len(c) for c in curves.values() if c is not None), default=0)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["share", "best_benefit", "pareto_cov", "fullpop_cov", "cumulative_cov"]
                   + [f"cum_gen_{i+1}" for i in range(max_g)])
        for share, best, pc, fc, cc in rows:
            c = curves.get(share)
            clist = list(c) if c is not None else []
            w.writerow([share, best, pc, fc, cc] + clist)
    print(f"\nsummary csv -> {CSV_PATH}")

    print(f"\n{'share':>7}{'best_benefit':>14}{'pareto_cov':>12}{'fullpop_cov':>13}{'cumul_cov':>11}")
    for share, best, pc, fc, cc in rows:
        print(f"{share:>7.2f}{best:>14.2f}{pc:>11.1f}%{fc:>12.1f}%{cc:>10.1f}%")

    _plot_curves(curves)


def _sweep_analyse():
    if not os.path.exists(CSV_PATH):
        print(f"No csv at {CSV_PATH}; run the sweep first."); return
    curves = {}
    with open(CSV_PATH) as f:
        r = csv.reader(f); next(r)
        n_meta = 5
        for row in r:
            share = float(row[0])
            curve = [float(x) for x in row[n_meta:] if x != ""]
            curves[share] = np.asarray(curve) if curve else None
    _plot_curves(curves)


def cmd_sweep(mode):
    use_agg()
    if mode == "analyse":
        _sweep_analyse()
    else:
        _sweep_run()


# ===========================================================================
COMMANDS = {
    "snapshot": lambda m: cmd_snapshot(),
    "animate": lambda m: cmd_animate(),
    "fullpop": lambda m: cmd_fullpop(),
    "sweep": cmd_sweep,
}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    mode = argv[2] if len(argv) > 2 else "run"
    if cmd not in COMMANDS:
        print(__doc__)
        print(f"Subcommands: {', '.join(COMMANDS)}")
        return
    COMMANDS[cmd](mode)


if __name__ == "__main__":
    main(sys.argv)
