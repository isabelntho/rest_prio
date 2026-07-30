"""Spillover / restoration_benefit direction suite: does the spatially-explicit
restoration_benefit objective reward clustering or dispersal?

Subcommands (pixi run python Debugs_tests/spillover_benefit.py <cmd> [mode]):

  check                    Unit check of the restoration_benefit objective:
                           empty->0, non-empty<0, spillover real (r3 more negative
                           than r0), and benefit is a single (non-split) objective.

  direction                Test 1 (direct, no GA): sweep a COMPACTNESS gradient at
                           fixed budget K (grow_region_plan with 1..K seeds) and read
                           benefit at radius 3 vs 0. More negative as plans fragment
                           -> spillover REWARDS DISPERSAL.

  sensitivity              Test 1b: does the dispersal direction hold across
                           neighbor_radius x neighbor_effect_decay? gap =
                           mean(scattered) - mean(compact); <0 = dispersal.

  confirm [smoke|run|analyse]
                           Test 2: free, spatially-unbiased optimiser (scattered/bitflip,
                           NO contiguity constraint) on (benefit, cost). High-benefit
                           non-dominated solutions should come out scattered.

  bias [smoke|run|analyse]
                           Test 3: at S=2, do NEUTRAL region operators disperse (and
                           score MORE benefit) than SCORED ones? Same seeds/pop/gens,
                           min_patch=2 fixed, only region_growth_bias changes; writes the
                           res_*_biasS2_* pkls that coverage_S2.py reads.

benefit is NEGATED throughout: MORE NEGATIVE = more improvement = better.
"""
import os
import sys
import glob
import time
import pickle
import numpy as np

from _common import (
    REPO_ROOT, RESULTS_DIR, BASE, POP_SIZE, N_GENERATIONS, RANDOM_SEEDS,
    load_ic, make_problem, cluster_metrics, use_agg,
)
from Core_optimisation.resto_anom import RestorationProblem, run_optimization_instance
from Core_optimisation.spatial_operations import build_restoration_neighbor_table, grow_region_plan


# ===========================================================================
# check  (was check_restoration_benefit.py)
# ===========================================================================
def _benefit_raw(problem, x):
    raw = problem.evaluate_raw_objectives(x)
    i = problem.objective_names.index("restoration_benefit")
    return raw[i]


def cmd_check():
    prob3, ic = make_problem(neighbor_radius=3)
    print(f"objective_names = {prob3.objective_names}")
    assert "restoration_benefit" in prob3.objective_names, "benefit not registered"
    assert "abiotic_anomaly" not in prob3.objective_names, "abiotic leaked as objective"
    assert "biotic_anomaly" not in prob3.objective_names, "biotic leaked as objective"

    n_rest = int(ic["n_restoration_pixels"])
    n_var = prob3.n_var
    K = prob3.max_action_pixels
    nbr, rows, cols = build_restoration_neighbor_table(ic)
    rng = np.random.default_rng(0)

    # (1) empty plan
    x0 = np.zeros(n_var, dtype=int)
    b_empty = _benefit_raw(prob3, x0)
    print(f"\n[1] empty-plan benefit          = {b_empty:.6f}  (expect 0)")
    assert abs(b_empty) < 1e-9

    # a compact plan and a scattered plan, matched budget K
    compact = grow_region_plan(nbr, n_rest, K, n_seeds=1,
                               scores=np.zeros(n_rest), mode="neutral", rng=rng)
    x_comp = np.zeros(n_var, dtype=int); x_comp[:n_rest] = compact.astype(int)
    scat_idx = rng.choice(n_rest, size=K, replace=False)
    x_scat = np.zeros(n_var, dtype=int); x_scat[scat_idx] = 1

    # (2) non-empty -> negative (improvement accrued)
    b_comp3 = _benefit_raw(prob3, x_comp)
    b_scat3 = _benefit_raw(prob3, x_scat)
    print(f"[2] compact benefit (r=3)       = {b_comp3:.4f}   scattered = {b_scat3:.4f}  (expect <0)")
    assert b_comp3 < 0 and b_scat3 < 0

    # (3) spillover real: r=3 MORE NEGATIVE than r=0 for same plan
    prob0, _ = make_problem(neighbor_radius=0)
    b_comp0 = _benefit_raw(prob0, x_comp)
    b_scat0 = _benefit_raw(prob0, x_scat)
    print(f"[3] compact benefit (r=0)       = {b_comp0:.4f}   scattered = {b_scat0:.4f}")
    print(f"    spillover gain (compact)    = {b_comp0 - b_comp3:.4f}  (improvement added by spillover; expect >0)")
    print(f"    spillover gain (scattered)  = {b_scat0 - b_scat3:.4f}  (expect >0)")
    assert b_comp3 < b_comp0 and b_scat3 < b_scat0

    # (4) arrangement signal (more negative = more improvement)
    print(f"\n[4] scattered - compact @ r=3   = {b_scat3 - b_comp3:.4f}  (negative -> scattered improves MORE)")
    print(f"    scattered - compact @ r=0   = {b_scat0 - b_comp0:.4f}  (direct-only, no spillover)")
    print("\nAll assertions passed.")


# ===========================================================================
# direction  (was spillover_direction.py)
# ===========================================================================
DIR_SEEDS_GRID = [1, 2, 5, 25, 100, 500, 2000]   # region seeds: low = compact, high = fragmented
DIR_N_REPS = 6                                    # replicate plans per level (different rng)


def cmd_direction():
    ic = load_ic(["restoration_benefit", "cost"])
    n_rest = int(ic["n_restoration_pixels"])
    prob3, _ = make_problem(3, ic)
    prob0, _ = make_problem(0, ic)
    n_var = prob3.n_var
    K = prob3.max_action_pixels
    bi = prob3.objective_names.index("restoration_benefit")
    nbr, rows, cols = build_restoration_neighbor_table(ic)
    scores0 = np.zeros(n_rest)

    def benefit(prob, sel):
        x = np.zeros(n_var, dtype=int); x[:n_rest] = sel.astype(int)
        return prob.evaluate_raw_objectives(x)[bi]

    print(f"budget K = {K} pixels; benefit is NEGATED (more negative = more improvement)\n")
    print(f"{'plan':>14}{'benefit_r3':>12}{'benefit_r0':>12}{'spillover':>11}"
          f"{'adjacency':>11}{'n_comp':>9}")

    def row(label, plans):
        b3 = np.mean([benefit(prob3, s) for s in plans])
        b0 = np.mean([benefit(prob0, s) for s in plans])
        adj, nc = np.mean([cluster_metrics(s, nbr) for s in plans], axis=0)
        print(f"{label:>14}{b3:>12.2f}{b0:>12.2f}{b0-b3:>11.2f}{adj:>11.0f}{nc:>9.0f}")

    for s in DIR_SEEDS_GRID:
        plans = [grow_region_plan(nbr, n_rest, K, n_seeds=s, scores=scores0,
                                  mode="neutral", rng=np.random.default_rng(1000 + r))
                 for r in range(DIR_N_REPS)]
        row(f"seeds={s}", plans)

    # pure random selection = fully scattered reference
    rand_plans = []
    for r in range(DIR_N_REPS):
        rng = np.random.default_rng(9000 + r)
        sel = np.zeros(n_rest, dtype=bool); sel[rng.choice(n_rest, size=K, replace=False)] = True
        rand_plans.append(sel)
    row("random", rand_plans)

    print("\nReading: scan benefit_r3 down the rows (compact -> fragmented).")
    print("  more negative as it fragments  -> spillover REWARDS DISPERSAL")
    print("  less negative as it fragments  -> spillover REWARDS CLUSTERING")
    print("  'spillover' col = improvement contributed by neighbour effect (r3 vs r0).")


# ===========================================================================
# sensitivity  (was spillover_sensitivity.py)
# ===========================================================================
SENS_RADII = [0, 1, 2, 3, 5, 8]
SENS_DECAYS = [0.05, 0.2, 0.5, 1.0]
SENS_N_REPS = 4


def cmd_sensitivity():
    ic = load_ic(["restoration_benefit", "cost"])
    n_rest = int(ic["n_restoration_pixels"])
    nbr, _, _ = build_restoration_neighbor_table(ic)
    K = int(BASE["max_restoration_fraction"] * n_rest)
    scores0 = np.zeros(n_rest)

    # fixed plans, generated once (independent of the effect params)
    compact = [grow_region_plan(nbr, n_rest, K, n_seeds=1, scores=scores0, mode="neutral",
                                rng=np.random.default_rng(100 + r)) for r in range(SENS_N_REPS)]
    scatter = []
    for r in range(SENS_N_REPS):
        rng = np.random.default_rng(900 + r)
        s = np.zeros(n_rest, dtype=bool); s[rng.choice(n_rest, size=K, replace=False)] = True
        scatter.append(s)

    def benefit(prob, sel):
        x = np.zeros(prob.n_var, dtype=int); x[:n_rest] = sel.astype(int)
        return prob.evaluate_raw_objectives(x)[prob.objective_names.index("restoration_benefit")]

    print(f"budget K = {K}; cell = mean(scattered) - mean(compact) benefit "
          f"(<0 = DISPERSAL, >0 = CLUSTERING)\n")
    print(f"{'radius':>7}" + "".join(f"{'decay='+str(d):>13}" for d in SENS_DECAYS))
    for rad in SENS_RADII:
        cells = []
        for dec in SENS_DECAYS:
            prob = RestorationProblem(ic, {**BASE, "neighbor_radius": rad, "neighbor_effect_decay": dec})
            bc = np.mean([benefit(prob, s) for s in compact])
            bs = np.mean([benefit(prob, s) for s in scatter])
            cells.append(bs - bc)
        print(f"{rad:>7}" + "".join(f"{g:>13.2f}" for g in cells))

    print("\nReading: sign is the direction. If every non-zero-radius cell is < 0, the")
    print("dispersal result is robust to the spillover radius/decay. radius=0 ~ 0 (control).")


# ===========================================================================
# confirm  (was spillover_confirm_run.py)
# ===========================================================================
CONFIRM_LABEL_PREFIX = "spillfree"
CONFIRM_PARAMS = {
    "max_restoration_fraction": 0.05, "abiotic_effect": 0.01, "biotic_effect": 0.01,
    "normalize_objectives": True, "sampling_strategy": "scattered",
    # no min_patch_size -> no contiguity constraint; scattered operators -> no spatial prior
}


def _confirm_run_sweep(ic, pop_size, n_generations, seeds, smoke):
    t0 = time.perf_counter()
    ok = 0
    for i, seed in enumerate(seeds, 1):
        run_label = f"{CONFIRM_LABEL_PREFIX}_seed_{seed}"
        print(f"\n===== [{i}/{len(seeds)}] scattered free run, seed={seed} "
              f"(pop={pop_size}, gens={n_generations}) -> {run_label} =====")
        try:
            run_optimization_instance(
                initial_conditions=ic, scenario_params=CONFIRM_PARAMS,
                pop_size=pop_size, n_generations=n_generations,
                save_results=(not smoke), verbose=True, n_jobs=12,
                random_seed=seed, use_repair=True, use_patch_approach=False,
                pixel_tolerance=0.05, algorithm_type="nsga2", run_label=run_label,
            )
            ok += 1
        except Exception as e:
            print(f"  [x] {run_label} failed: {e}")
    print(f"\n===== RUN COMPLETE: {ok}/{len(seeds)} in {(time.perf_counter()-t0)/60:.1f} min =====")


def _confirm_analyse():
    from visualisations import load_results  # noqa: F401  (kept for parity; not used directly)
    ic = load_ic(["restoration_benefit", "cost"])
    n_rest = int(ic["n_restoration_pixels"])
    nbr, _, _ = build_restoration_neighbor_table(ic)
    K = int(CONFIRM_PARAMS["max_restoration_fraction"] * n_rest)

    # random-selection null for adjacency/n_components at budget K
    rng = np.random.default_rng(0)
    adj_null, nc_null = [], []
    for _ in range(20):
        sel = np.zeros(n_rest, dtype=bool); sel[rng.choice(n_rest, size=K, replace=False)] = True
        a, c = cluster_metrics(sel, nbr); adj_null.append(a); nc_null.append(c)
    print(f"random null (K={K}): adjacency ~ {np.mean(adj_null):.0f}, n_components ~ {np.mean(nc_null):.0f}")

    pkls = sorted(glob.glob(os.path.join(RESULTS_DIR, f"res_*_{CONFIRM_LABEL_PREFIX}_seed_*.pkl")))
    if not pkls:
        print("No spillfree pkls yet. Run without 'analyse' first.")
        return
    print(f"\n{'seed':>6}{'nsol':>6}{'corr(benefit,adj)':>18}{'corr(benefit,ncomp)':>20}"
          f"{'best-benefit adj':>18}{'best n_comp':>12}")
    for p in pkls:
        with open(p, "rb") as f:                 # raw pkl: all arrays are full population
            res = pickle.load(f)
        names = list(res["objective_names"])
        bi = names.index("restoration_benefit")
        F = np.asarray(res["objectives_raw"], float)
        nd = np.asarray(res["is_nondominated"], bool)
        dec = np.asarray(res["decisions"])[nd][:, :n_rest] > 0.5
        ben = F[nd][:, bi]                       # negated: more negative = more benefit
        adj = np.array([cluster_metrics(d, nbr)[0] for d in dec])
        nco = np.array([cluster_metrics(d, nbr)[1] for d in dec])
        # benefit magnitude = -ben (higher = better). Correlate with compactness.
        mag = -ben
        ca = np.corrcoef(mag, adj)[0, 1] if len(mag) > 1 else np.nan
        cc = np.corrcoef(mag, nco)[0, 1] if len(mag) > 1 else np.nan
        best = int(np.argmax(mag))
        seed = os.path.basename(p).split("_seed_")[1].split(".")[0]
        print(f"{seed:>6}{len(mag):>6}{ca:>18.2f}{cc:>20.2f}{adj[best]:>18.0f}{nco[best]:>12.0f}")
    print("\nReading: negative corr(benefit,adjacency) and positive corr(benefit,n_comp)")
    print("-> higher-benefit solutions are MORE scattered = confirms spillover rewards")
    print("dispersal. Best-benefit adjacency near the random null = fully scattered.")


def cmd_confirm(mode):
    if mode == "analyse":
        _confirm_analyse(); return
    smoke = (mode == "smoke")
    pop_size, n_generations, seeds = POP_SIZE, N_GENERATIONS, list(RANDOM_SEEDS)
    if smoke:
        pop_size, n_generations, seeds = 12, 3, [606]
    ic = load_ic(["restoration_benefit", "cost"])
    _confirm_run_sweep(ic, pop_size, n_generations, seeds, smoke)
    if not smoke:
        _confirm_analyse()


# ===========================================================================
# bias  (was spillover_bias_S2.py)
# ===========================================================================
BIAS_S = 2
BIAS_BIASES = ["scored", "neutral"]
BIAS_OUT_DIR = os.path.join(REPO_ROOT, "Debugs_tests", "diagnostics", "bias_S2")
BIAS_LABEL_PREFIX = "biasS2"
BIAS_BASE_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01, "normalize_objectives": True,
    "burden_sharing": "no", "rp_formulation": "sum", "rp_threshold": 0.0,
    "region_seeds": 25, "region_seeds_min": 5, "region_mutation_edits": 100,
    "sampling_strategy": "region_grow", "min_patch_size": BIAS_S,
}


def _bias_run_sweep(ic, pop_size, n_generations, seeds, smoke):
    n = len(BIAS_BIASES) * len(seeds); done = ok = 0; t0 = time.perf_counter()
    for bias in BIAS_BIASES:
        for seed in seeds:
            done += 1
            run_label = f"{BIAS_LABEL_PREFIX}_{bias}_seed_{seed}"
            print(f"\n===== [{done}/{n}] S={BIAS_S} benefit growth_bias={bias} seed={seed} -> {run_label} =====")
            try:
                run_optimization_instance(
                    initial_conditions=ic, scenario_params={**BIAS_BASE_PARAMS, "region_growth_bias": bias},
                    pop_size=pop_size, n_generations=n_generations, save_results=(not smoke),
                    verbose=True, n_jobs=12, random_seed=seed, use_repair=True,
                    use_patch_approach=False, pixel_tolerance=0.05, algorithm_type="nsga2",
                    run_label=run_label,
                )
                ok += 1
            except Exception as e:
                print(f"  [x] {run_label} failed: {e}")
    print(f"\n===== RUN COMPLETE: {ok}/{n} in {(time.perf_counter()-t0)/60:.1f} min =====")


def _bias_pkls(bias, seeds):
    out = {}
    for seed in seeds:
        hits = glob.glob(os.path.join(RESULTS_DIR, f"res_*_{BIAS_LABEL_PREFIX}_{bias}_seed_{seed}.pkl"))
        if hits:
            out[seed] = max(hits, key=os.path.getmtime)
    return out


def _bias_sel(res, convert):
    dec = convert(res["decisions"], res["initial_conditions"], results=res)
    if dec is None:
        dec = res["decisions"]
    nr = len(np.asarray(res["initial_conditions"]["restoration_eligible_indices"]))
    return (np.asarray(dec)[:, :nr] > 0.5)


def _bias_analyse(seeds):
    from visualisations import (
        load_results, _convert_patch_decisions_to_pixel_matrix, create_selection_frequency_map,
    )
    os.makedirs(BIAS_OUT_DIR, exist_ok=True)
    ic = load_ic(["restoration_benefit", "cost"])
    nbr, _, _ = build_restoration_neighbor_table(ic)
    print(f"\n=== S={BIAS_S} benefit: scored vs neutral region operators ===")
    print(f"{'bias':>8}{'seeds':>6}{'nsol':>6}{'best_benefit':>14}{'ever':>8}{'core_frac':>10}{'adjacency':>11}{'n_comp':>9}")
    for bias in BIAS_BIASES:
        pkls = _bias_pkls(bias, seeds)
        if not pkls:
            print(f"{bias:>8}   (no pkls yet)"); continue
        nsols, bests, evers, cores, adjs, ncs = [], [], [], [], [], []
        pooled = []
        for p in pkls.values():
            res = load_results(p)                       # nondominated-filtered
            with open(p, "rb") as f:
                raw = pickle.load(f)
            names = list(raw["objective_names"]); bi = names.index("restoration_benefit")
            F = np.asarray(raw["objectives_raw"], float); nd = np.asarray(raw["is_nondominated"], bool)
            bests.append(float(F[nd][:, bi].min()))     # most negative = most benefit
            Sm = _bias_sel(res, _convert_patch_decisions_to_pixel_matrix); nsol = Sm.shape[0]; nsols.append(nsol)
            freq = Sm.sum(0) / nsol
            evers.append(int((freq > 0).sum()))
            cores.append(int((freq >= 0.999).sum()) / int(Sm.sum(1)[0]))
            a, c = np.mean([cluster_metrics(Sm[i], nbr) for i in range(nsol)], axis=0)
            adjs.append(a); ncs.append(c)
            conv = _convert_patch_decisions_to_pixel_matrix(res["decisions"], res["initial_conditions"], results=res)
            pooled.append(np.asarray(conv if conv is not None else res["decisions"]))
        print(f"{bias:>8}{len(pkls):>6}{np.mean(nsols):>6.0f}{np.mean(bests):>14.2f}"
              f"{np.mean(evers):>8.0f}{np.mean(cores):>10.2f}{np.mean(adjs):>11.0f}{np.mean(ncs):>9.0f}")
        # pooled RFOP map
        X = np.vstack(pooled)
        src = load_results(list(pkls.values())[0])
        tmp = os.path.join(BIAS_OUT_DIR, f"_pool_{bias}.pkl")
        with open(tmp, "wb") as f:
            pickle.dump({"decisions": X, "initial_conditions": src["initial_conditions"],
                         "n_solutions": X.shape[0], "objectives": np.zeros((X.shape[0], 2)),
                         "problem_info": src["problem_info"],
                         "objective_names": ["restoration_benefit", "implementation_cost"],
                         "scenario_params": {}}, f)
        png = os.path.join(BIAS_OUT_DIR, f"rfop_S2_{bias}.png")
        create_selection_frequency_map(tmp, save_path=png, cmap="YlOrRd", show_eligible=True,
                                       action_type="combined", title=f"S=2 benefit, {bias} operators: RFOP")
        print(f"  RFOP map -> {png}")
    print("\nReading: benefit is negated (more negative = more benefit). If NEUTRAL gets")
    print("more-negative best_benefit AND lower core_frac / higher n_comp, the scored")
    print("operators were mis-steering (sacrificing benefit by concentrating).")


def cmd_bias(mode):
    use_agg()
    seeds = list(RANDOM_SEEDS)
    if mode == "analyse":
        _bias_analyse(seeds); return
    smoke = (mode == "smoke")
    pop_size, n_generations = POP_SIZE, N_GENERATIONS
    if smoke:
        pop_size, n_generations, seeds = 12, 3, [606]
    ic = load_ic(["restoration_benefit", "cost"])
    _bias_run_sweep(ic, pop_size, n_generations, seeds, smoke)
    if not smoke:
        _bias_analyse(seeds)


# ===========================================================================
COMMANDS = {
    "check": lambda m: cmd_check(),
    "direction": lambda m: cmd_direction(),
    "sensitivity": lambda m: cmd_sensitivity(),
    "confirm": cmd_confirm,
    "bias": cmd_bias,
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
