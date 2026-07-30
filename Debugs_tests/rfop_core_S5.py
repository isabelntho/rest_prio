"""S=5 RFOP-core robustness: is the always-selected restoration core real, or an
artifact of the seeding / scored operators?

Subcommands (pixi run python Debugs_tests/rfop_core_S5.py <cmd> [mode]):

  initpop      Per-pixel frequency-of-occurrence maps of the INITIAL population (50
               individuals) at S=5, one per region_random_share (0.0 / 0.5 / 1.0). The
               initial population is regenerated from RegionGrowingSampling with the exact
               contiguity-sweep scores/params (same np seed for all three shares), so the
               only difference is the share.

  init-sensitivity [smoke|run|analyse]
               Does the RFOP skew survive a diversified initial population? Varies ONLY
               region_random_share (mutation + repair stay scored), a few seeds each, and
               reports the RFOP-skew metrics per share. Core persists at share=1.0 ->
               operator/objective-driven; core dissolves -> seeding artifact.

  neutral [smoke|run|analyse]
               Does the RFOP core survive when the scored bias is removed from ALL region
               operators (region_growth_bias='neutral')? Compares scored vs neutral, and
               the cross-bias core overlap. Core persists under neutral (+ high Jaccard)
               -> genuine structure; dissolves -> the scored operators drove it.

  plots        Neutral-operator visuals: pooled RFOP map (all seeds) + potential-vs-cost
               Pareto front with every seed drawn (scored faint for context). Reads the
               `neutral` run pkls.
"""
import os
import sys
import glob
import time
import pickle
import numpy as np

from _common import REPO_ROOT, RESULTS_DIR, POP_SIZE, N_GENERATIONS, RANDOM_SEEDS, load_ic, use_agg, newest_per_seed

S = 5
# Region config shared across the S=5 subcommands (matches the contiguity sweep).
BASE_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01, "normalize_objectives": True,
    "burden_sharing": "no", "rp_formulation": "sum", "rp_threshold": 0.0,
    "region_seeds": 25, "region_seeds_min": 5, "region_mutation_edits": 100,
    "sampling_strategy": "region_grow", "min_patch_size": S,
}


def _sel_matrix(res, convert):
    dec = convert(res["decisions"], res["initial_conditions"], results=res)
    if dec is None:
        dec = res["decisions"]
    nr = len(np.asarray(res["initial_conditions"]["restoration_eligible_indices"]))
    return (np.asarray(dec)[:, :nr] > 0.5)


# ===========================================================================
# initpop  (was init_pop_maps_S5.py)
# ===========================================================================
INITPOP_SHARES = [0.0, 0.5, 1.0]
INITPOP_N_POP = 50
INITPOP_SEED = 606
INITPOP_OUT_DIR = os.path.join(REPO_ROOT, "Debugs_tests", "diagnostics", "init_pop_S5")
INITPOP_PARAMS = {
    "max_restoration_fraction": 0.05, "spatial_clustering": 0,
    "biotic_effect": 0.01, "abiotic_effect": 0.01, "normalize_objectives": True,
    "burden_sharing": "no", "rp_formulation": "sum", "rp_threshold": 0.0,
    "region_seeds": 25, "region_seeds_min": 5, "region_growth_bias": "scored",
    "region_mutation_edits": 100, "sampling_strategy": "region_grow", "min_patch_size": S,
}


class _Stub:
    """Minimal stand-in for the pymoo problem: only fields RegionGrowingSampling._do reads."""
    def __init__(self, n_var, n_rest):
        self.n_var = n_var
        self.n_restoration_pixels = n_rest


def _build_region_scores(ic):
    from Core_optimisation.resto_anom import build_repair_scores
    scores = np.asarray(build_repair_scores(ic, INITPOP_PARAMS), dtype=np.float64)
    cost2d = ic.get("implementation_cost")
    rmask = ic.get("restoration_eligible_mask")
    if cost2d is not None and rmask is not None:
        cpix = np.asarray(cost2d)[rmask].astype(np.float64)
        if cpix.size == scores.size:
            def _z(a):
                sd = a.std()
                return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a)
            scores = _z(scores) - _z(cpix)   # same blend resto_anom uses for 'scored'
    return scores


def cmd_initpop():
    use_agg()
    from Core_optimisation.spatial_operations import RegionGrowingSampling
    from visualisations import create_selection_frequency_map

    os.makedirs(INITPOP_OUT_DIR, exist_ok=True)
    ic = load_ic(["restoration_potential", "cost"])
    n_rest = int(ic["n_restoration_pixels"])
    n_conv = len(ic.get("conversion_eligible_indices", []))
    n_var = n_rest + n_conv
    max_action = int(INITPOP_PARAMS["max_restoration_fraction"] * n_rest)

    # match resto_anom's min-patch seed cap so the initial regions are >= S on average
    region_seeds = min(int(INITPOP_PARAMS["region_seeds"]), max(1, max_action // S))
    region_seeds_min = min(int(INITPOP_PARAMS["region_seeds_min"]), region_seeds)

    region_scores = _build_region_scores(ic)
    problem = _Stub(n_var, n_rest)
    print(f"n_rest={n_rest} n_conv={n_conv} budget={max_action} "
          f"seeds<={region_seeds} (min {region_seeds_min})")

    for share in INITPOP_SHARES:
        np.random.seed(INITPOP_SEED)   # same global seed -> only `share` differs across maps
        sampler = RegionGrowingSampling(
            ic, max_action, region_scores, region_seeds=region_seeds,
            growth_bias="scored", region_seeds_min=region_seeds_min, random_share=share,
        )
        X = sampler._do(problem, INITPOP_N_POP)  # (N_POP, n_var), restoration block filled
        per_ind = X[:, :n_rest].sum(1)
        print(f"\nshare={share}: pixels/individual min/med/max = "
              f"{per_ind.min()}/{int(np.median(per_ind))}/{per_ind.max()}")

        # wrap as a minimal results dict so we can reuse the RFOP plotter unchanged
        res = {
            "decisions": X.astype(int),
            "initial_conditions": ic,
            "n_solutions": INITPOP_N_POP,
            "objectives": np.zeros((INITPOP_N_POP, 2), dtype=float),
            "objective_names": ["restoration_potential", "implementation_cost"],
            "problem_info": {"n_pixels": n_rest},
            "scenario_params": INITPOP_PARAMS,
        }   # no 'is_nondominated' -> load_results keeps all 50
        tmp = os.path.join(INITPOP_OUT_DIR, f"_initpop_share{str(share).replace('.', 'p')}.pkl")
        with open(tmp, "wb") as f:
            pickle.dump(res, f)

        png = os.path.join(INITPOP_OUT_DIR, f"initpop_freq_share{str(share).replace('.', 'p')}.png")
        create_selection_frequency_map(
            tmp, save_path=png, cmap="YlOrRd", show_eligible=True,
            action_type="combined",
            title=f"Initial population (50 indiv), S={S}, region_random_share={share}",
        )
        print(f"  map -> {png}")

    print(f"\nAll initial-population maps under {INITPOP_OUT_DIR}")


# ===========================================================================
# init-sensitivity  (was init_sensitivity_S5.py)
# ===========================================================================
INITSENS_SHARES = [0.0, 0.5, 1.0]
INITSENS_LABEL_PREFIX = "initS5"


def _share_tag(share):
    return f"share{str(share).replace('.', 'p')}"


def _initsens_run_sweep(ic, pop_size, n_generations, shares, seeds, smoke):
    from Core_optimisation.resto_anom import run_optimization_instance
    n_runs = len(shares) * len(seeds)
    done = ok = 0
    t0 = time.perf_counter()
    for share in shares:
        for seed in seeds:
            done += 1
            run_label = f"{INITSENS_LABEL_PREFIX}_{_share_tag(share)}_seed_{seed}"
            print(f"\n===== [{done}/{n_runs}] S={S} region_random_share={share} "
                  f"seed={seed} (pop={pop_size}, gens={n_generations}) -> {run_label} =====")
            params = {**BASE_PARAMS, "region_growth_bias": "scored", "region_random_share": share}
            try:
                run_optimization_instance(
                    initial_conditions=ic, scenario_params=params,
                    pop_size=pop_size, n_generations=n_generations,
                    save_results=(not smoke), verbose=True, n_jobs=12,
                    random_seed=seed, use_repair=True, use_patch_approach=False,
                    pixel_tolerance=0.05, algorithm_type="nsga2", run_label=run_label,
                )
                ok += 1
            except Exception as e:
                print(f"  [x] {run_label} failed: {e}")
    print(f"\n===== RUN COMPLETE: {ok}/{n_runs} in {(time.perf_counter()-t0)/60:.1f} min =====")
    if smoke:
        print("SMOKE: save_results off, no pkls written. Wiring OK if this printed.")


def _initsens_run_metrics(pkl):
    from visualisations import load_results, _convert_patch_decisions_to_pixel_matrix
    res = load_results(pkl)                 # nondominated filtered
    S_ = _sel_matrix(res, _convert_patch_decisions_to_pixel_matrix)
    nsol = S_.shape[0]
    freq = S_.sum(0) / nsol
    ever = int((freq > 0).sum())
    area = int(S_.sum(1)[0])
    core = int((freq >= 0.999).sum())
    among = freq[freq > 0]
    rfop90 = float((among >= 0.90).mean()) if ever else float("nan")
    sets = [set(np.flatnonzero(S_[i]).tolist()) for i in range(nsol)]
    js = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
          for i in range(nsol) for j in range(i + 1, nsol) if (sets[i] | sets[j])]
    return dict(nsol=nsol, area=area, ever=ever, core=core,
                core_frac=core / area if area else np.nan,
                rfop90=rfop90, meanjac=float(np.mean(js)) if js else np.nan)


def _initsens_analyse(shares, seeds):
    print(f"\n=== S={S} init-sensitivity: RFOP skew vs region_random_share ===")
    print("(per-run metrics averaged over seeds; core_frac = %budget always-selected)")
    print(f"{'share':>7}{'seeds':>6}{'nsol':>6}{'ever':>8}{'core_frac':>10}"
          f"{'RFOP>=90%':>10}{'meanJac':>9}")
    rows = []
    for share in shares:
        pkls = newest_per_seed(f"res_*_{INITSENS_LABEL_PREFIX}_{_share_tag(share)}_seed_{{seed}}.pkl", seeds)
        if not pkls:
            print(f"{share:>7}   (no pkls yet)")
            continue
        ms = [_initsens_run_metrics(p) for p in pkls.values()]
        agg = {k: float(np.mean([m[k] for m in ms])) for k in
               ("nsol", "ever", "core_frac", "rfop90", "meanjac")}
        print(f"{share:>7}{len(pkls):>6}{agg['nsol']:>6.0f}{agg['ever']:>8.0f}"
              f"{agg['core_frac']:>10.2f}{agg['rfop90']:>10.2f}{agg['meanjac']:>9.2f}")
        rows.append((share, agg))
    if len(rows) >= 2:
        print("\nRead: if core_frac / RFOP>=90% / meanJac stay high at share=1.0, the")
        print("always-selected core PERSISTS under a diversified init -> not a seeding")
        print("artifact. If they drop sharply, the skew was largely from the seeding.")
    return rows


def cmd_init_sensitivity(mode):
    if mode == "analyse":
        _initsens_analyse(INITSENS_SHARES, list(RANDOM_SEEDS)); return
    smoke = (mode == "smoke")
    pop_size, n_generations, shares, seeds = POP_SIZE, N_GENERATIONS, INITSENS_SHARES, list(RANDOM_SEEDS)
    if smoke:
        pop_size, n_generations, shares, seeds = 12, 3, [0.0, 1.0], [606]
    ic = load_ic(["restoration_potential", "cost"])
    _initsens_run_sweep(ic, pop_size, n_generations, shares, seeds, smoke)
    if not smoke:
        _initsens_analyse(shares, seeds)


# ===========================================================================
# neutral  (was neutral_operator_S5.py)
# ===========================================================================
NEUT_BIASES = ["scored", "neutral"]
NEUT_LABEL_PREFIX = "neutopS5"


def _neut_newest_per_seed(bias, seeds):
    return newest_per_seed(f"res_*_{NEUT_LABEL_PREFIX}_{bias}_seed_{{seed}}.pkl", seeds)


def _neut_run_sweep(ic, pop_size, n_generations, seeds, smoke):
    from Core_optimisation.resto_anom import run_optimization_instance
    n_runs = len(NEUT_BIASES) * len(seeds)
    done = ok = 0
    t0 = time.perf_counter()
    for bias in NEUT_BIASES:
        for seed in seeds:
            done += 1
            run_label = f"{NEUT_LABEL_PREFIX}_{bias}_seed_{seed}"
            print(f"\n===== [{done}/{n_runs}] S={S} growth_bias={bias} seed={seed} "
                  f"(pop={pop_size}, gens={n_generations}) -> {run_label} =====")
            params = {**BASE_PARAMS, "region_growth_bias": bias}
            try:
                run_optimization_instance(
                    initial_conditions=ic, scenario_params=params,
                    pop_size=pop_size, n_generations=n_generations,
                    save_results=(not smoke), verbose=True, n_jobs=12,
                    random_seed=seed, use_repair=True, use_patch_approach=False,
                    pixel_tolerance=0.05, algorithm_type="nsga2", run_label=run_label,
                )
                ok += 1
            except Exception as e:
                print(f"  [x] {run_label} failed: {e}")
    print(f"\n===== RUN COMPLETE: {ok}/{n_runs} in {(time.perf_counter()-t0)/60:.1f} min =====")


def _neut_run_metrics(pkl):
    from visualisations import load_results, _convert_patch_decisions_to_pixel_matrix
    res = load_results(pkl)
    Sm = _sel_matrix(res, _convert_patch_decisions_to_pixel_matrix)
    nsol = Sm.shape[0]
    freq = Sm.sum(0) / nsol
    ever = int((freq > 0).sum())
    area = int(Sm.sum(1)[0])
    core = int((freq >= 0.999).sum())
    among = freq[freq > 0]
    rfop90 = float((among >= 0.90).mean()) if ever else float("nan")
    sets = [set(np.flatnonzero(Sm[i]).tolist()) for i in range(nsol)]
    js = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
          for i in range(nsol) for j in range(i + 1, nsol) if (sets[i] | sets[j])]
    return dict(nsol=nsol, ever=ever, core_frac=core / area if area else np.nan,
                rfop90=rfop90, meanjac=float(np.mean(js)) if js else np.nan)


def _neut_cross_bias_core(pkls_by_bias):
    """Overlap of the always-selected core between the scored and neutral runs
    (do the two biases land on the SAME pixels?). Pooled per bias, Jaccard of cores."""
    from visualisations import load_results, _convert_patch_decisions_to_pixel_matrix
    cores = {}
    for bias, pkls in pkls_by_bias.items():
        acc = None
        for p in pkls.values():
            res = load_results(p)
            Sm = _sel_matrix(res, _convert_patch_decisions_to_pixel_matrix)
            c = set(np.flatnonzero((Sm.sum(0) / Sm.shape[0]) >= 0.999).tolist())
            acc = c if acc is None else (acc & c)   # core common to all this bias's seeds
        cores[bias] = acc or set()
    if len(cores) == 2:
        a, b = cores["scored"], cores["neutral"]
        u = len(a | b)
        j = len(a & b) / u if u else float("nan")
        print(f"\nscored-core={len(a)}  neutral-core={len(b)}  "
              f"shared={len(a & b)}  Jaccard(scored,neutral)={j:.2f}")


def _neut_analyse(seeds):
    print(f"\n=== S={S} neutral-operator test: RFOP skew vs growth_bias ===")
    print(f"{'bias':>8}{'seeds':>6}{'nsol':>6}{'ever':>8}{'core_frac':>10}{'RFOP>=90%':>10}{'meanJac':>9}")
    pkls_by_bias = {}
    for bias in NEUT_BIASES:
        pkls = _neut_newest_per_seed(bias, seeds)
        pkls_by_bias[bias] = pkls
        if not pkls:
            print(f"{bias:>8}   (no pkls yet)")
            continue
        ms = [_neut_run_metrics(p) for p in pkls.values()]
        agg = {k: float(np.mean([m[k] for m in ms])) for k in
               ("nsol", "ever", "core_frac", "rfop90", "meanjac")}
        print(f"{bias:>8}{len(pkls):>6}{agg['nsol']:>6.0f}{agg['ever']:>8.0f}"
              f"{agg['core_frac']:>10.2f}{agg['rfop90']:>10.2f}{agg['meanjac']:>9.2f}")
    if all(pkls_by_bias.get(b) for b in NEUT_BIASES):
        _neut_cross_bias_core(pkls_by_bias)
        print("\nRead: high core_frac under 'neutral' AND high Jaccard(scored,neutral)")
        print("-> both un-biased and biased search converge on the SAME core = genuine")
        print("structure. Low neutral core_frac / low Jaccard -> scored operators drove it.")


def cmd_neutral(mode):
    if mode == "analyse":
        _neut_analyse(list(RANDOM_SEEDS)); return
    smoke = (mode == "smoke")
    pop_size, n_generations, seeds = POP_SIZE, N_GENERATIONS, list(RANDOM_SEEDS)
    if smoke:
        pop_size, n_generations, seeds = 12, 3, [606]
    ic = load_ic(["restoration_potential", "cost"])
    _neut_run_sweep(ic, pop_size, n_generations, seeds, smoke)
    if not smoke:
        _neut_analyse(seeds)


# ===========================================================================
# plots  (was neutral_plots_S5.py)
# ===========================================================================
NEUT_OUT_DIR = os.path.join(REPO_ROOT, "Debugs_tests", "diagnostics", "neutral_S5")
BIAS_COLORS = {"neutral": "#2A7BE0", "scored": "#E05C2A"}


def _obj_idx(names):
    pot = names.index("restoration_potential")
    cost = names.index("implementation_cost") if "implementation_cost" in names else names.index("cost")
    return pot, cost


def _plots_pooled_rfop(seeds, bias="neutral"):
    from visualisations import create_selection_frequency_map, _convert_patch_decisions_to_pixel_matrix
    from visualisations import load_results
    pkls = _neut_newest_per_seed(bias, seeds)
    decs, ic, pinfo = [], None, None
    for p in pkls.values():
        res = load_results(p)                       # nondominated-filtered
        d = _convert_patch_decisions_to_pixel_matrix(res["decisions"], res["initial_conditions"], results=res)
        if d is None:
            d = res["decisions"]
        decs.append(np.asarray(d))
        ic = res["initial_conditions"]
        pinfo = res["problem_info"]
    X = np.vstack(decs)
    pooled = {
        "decisions": X, "initial_conditions": ic, "n_solutions": X.shape[0],
        "objectives": np.zeros((X.shape[0], 2)), "problem_info": pinfo,
        "objective_names": ["restoration_potential", "implementation_cost"],
        "scenario_params": {},
    }   # no is_nondominated -> keep all pooled solutions
    tmp = os.path.join(NEUT_OUT_DIR, f"_pooled_{bias}.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(pooled, f)
    png = os.path.join(NEUT_OUT_DIR, f"rfop_{bias}_allseeds.png")
    create_selection_frequency_map(
        tmp, save_path=png, cmap="YlOrRd", show_eligible=True, action_type="combined",
        title=f"Neutral operators, S=5: pooled RFOP ({len(pkls)} seeds)")
    print(f"pooled RFOP ({bias}, {X.shape[0]} sols) -> {png}")


def _plots_front_all_seeds(seeds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for bias in ("scored", "neutral"):
        first = True
        for seed, p in sorted(_neut_newest_per_seed(bias, seeds).items()):
            with open(p, "rb") as f:
                raw = pickle.load(f)
            names = list(raw["objective_names"])
            pot, cost = _obj_idx(names)
            F = np.asarray(raw["objectives_raw"], float)
            nd = np.asarray(raw["is_nondominated"], bool)
            Fn = F[nd]
            alpha = 0.9 if bias == "neutral" else 0.35
            size = 22 if bias == "neutral" else 14
            ax.scatter(-Fn[:, pot], Fn[:, cost], s=size, alpha=alpha,
                       color=BIAS_COLORS[bias], edgecolors="none",
                       label=(bias if first else None), zorder=3 if bias == "neutral" else 2)
            first = False
    ax.set_xlabel("restoration benefit (-potential sum)")
    ax.set_ylabel("implementation cost")
    ax.set_title("S=5 Pareto fronts, all seeds: neutral vs scored operators")
    ax.legend(title="growth_bias")
    fig.tight_layout()
    png = os.path.join(NEUT_OUT_DIR, "front_S5_neutral_vs_scored_allseeds.png")
    fig.savefig(png, dpi=140)
    print(f"front (all seeds) -> {png}")


def cmd_plots():
    use_agg()
    os.makedirs(NEUT_OUT_DIR, exist_ok=True)
    seeds = list(RANDOM_SEEDS)
    _plots_pooled_rfop(seeds, "neutral")
    _plots_front_all_seeds(seeds)
    print(f"\nOutputs under {NEUT_OUT_DIR}")


# ===========================================================================
COMMANDS = {
    "initpop": lambda m: cmd_initpop(),
    "init-sensitivity": cmd_init_sensitivity,
    "neutral": cmd_neutral,
    "plots": lambda m: cmd_plots(),
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
