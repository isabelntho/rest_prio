"""
Two-objective NSGA-II run script.

Companion to run_custom.py. Identical structure and scenario modes, but runs
a TWO-objective optimisation (restoration_potential + implementation_cost)
using NSGA-II instead of the 3-objective NSGA-III used by run_custom.py.

Rationale: restoration_potential vs implementation_cost is the real headline
trade-off once spatial_clustering is dropped (clustering shadows cost). With
only two objectives the Pareto front is a curve, so NSGA-II's crowding-distance
diversity works well; NSGA-III is unnecessary here.

Edit the configuration variables below and run directly:
    python -m Core_optimisation.run_custom_nsga2
"""
import cProfile
import pstats
import io
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from .resto_anom import run_optimization_instance, main
from .data_loader import load_initial_conditions
from .grid_parallel import run_tag
from .logger_setup import setup_logger
from .paths import LOGS_DIR, R_INPUTS_DIR

# Algorithm selector passed to run_optimization_instance. "nsga2" uses NSGA-II
# with an explicit POP_SIZE (below).
ALGORITHM = "nsga2"

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
# Used in output filenames and the run_registry.jsonl log.
# Examples: "baseline", "patch_size2_highbudget", "testing_new_repair"
RUN_LABEL = "fact_2obj"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "all"            runs scenario="all" using the scenario sampling logic inside main()
#   "condition_grid" sweeps all 13 condition scenarios x SEEDS
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
#   "factorial"      fully-crossed design over FACTORIAL_FORMS x FACTORIAL_SCALINGS x
#                    FACTORIAL_CONSTRUCTIONS x FACTORIAL_POLICIES x SEEDS (iEMSs Block 4)
SCENARIO_MODE = "factorial"

# Condition scenario tag - selects pre-computed anomaly rasters from data/anomaly_scenarios/
# Available tags:
#   global_all | global_drop_smd | global_drop_sbd | global_drop_soc |
#   global_drop_uzl | global_drop_tsd | global_drop_can | global_drop_cdi |
#   global_drop_swf_h | global_drop_swf_t | global_drop_lai | global_drop_ndvi |
#   upper_q75_all
CONDITION_SCENARIO = "global_all"

# Seeds used by both condition_grid and policy_grid modes.
#   List[int] - runs each scenario/variant once per seed; labels: <name>_seed<n>
#   None       - runs each scenario/variant once using RANDOM_SEED; labels: <name>
#SEEDS = [101, 102, 103, 104, 105] # 5 seed replicates (iEMSs run matrix, Block 0)
SEEDS = [101, 102, 103]  # single-seed "quick" grid (list so condition_grid can iterate it)
print(f"\n=== RESTORATION OPTIMIZATION (NSGA-II, 2-OBJ) FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
print(f"Scenario mode: {SCENARIO_MODE}")

log_path = setup_logger(log_dir=str(LOGS_DIR), run_label=f"{ECOSYSTEM_TO_RUN}_{REGION.lower()}_nsga2")
if log_path:
    print(f"Verbose output -> {log_path}")

# TWO-objective formulation: restoration_benefit (spatially-explicit ecological
# benefit) vs cost. "cost" maps to implementation_cost; "restoration_benefit" is the
# total abiotic+biotic anomaly improvement INCLUDING neighbour spillover, so it depends
# on WHICH pixels AND their arrangement (unlike the static restoration_potential).
# NSGA-II handles the 2-objective front.
OBJECTIVES = ["restoration_benefit", "cost"]
# Available objective names:
#   "abiotic"               - minimise abiotic condition anomaly (restoration pixels)
#   "biotic"                - minimise biotic condition anomaly (restoration pixels)
#   "cost"                  - minimise implementation cost
#   "connectivity"          - maximise connectivity gain (precomputed per pixel, conversion pixels)
#   "landscape"             - legacy landscape anomaly via SN-density recalculation (conversion pixels)
#   "landscape_context"     - minimise mean abiotic anomaly of eligible neighbours within 500 m
#   "restoration_potential" - minimise mean of per-pixel abiotic + biotic baseline anomaly;
#                             single combined ecological condition score (lower = more degraded)
#   "spatial_clustering"    - maximise spatial compactness of the selected pixels
#   "es_future_val"         - maximise total ES performance of selected pixels under future scenarios
#   "es_future_robustness"  - minimise total ES instability of selected pixels under future scenarios
SAMPLE_FRACTION = None
SAMPLE_SEED = 42
# NSGA-II uses POP_SIZE directly (unlike NSGA-III, which sizes the population
# from N_PARTITIONS). N_PARTITIONS is left defined for API compatibility but is
# unused when ALGORITHM == "nsga2".
POP_SIZE = 100
N_GENERATIONS = 100
N_JOBS = 12
# Worker PROCESSES for condition_grid (parallelises whole optimisation instances,
# one task per condition tag).
GRID_WORKERS = 4
RANDOM_SEED = 101
# In "custom" mode, a non-None SEEDS list replicates the single run once per seed (labels
# <RUN_LABEL>_seed<n>); SEEDS=None runs once at RANDOM_SEED. The grid modes always iterate
# SEEDS themselves - this flag only wires seed replication into the single-run branch.
_MULTISEED = (SCENARIO_MODE == "custom") and (SEEDS is not None)
N_SAMPLES_PER_PARAM = 3
N_PARTITIONS = 12  # unused for NSGA-II; kept for run_optimization_instance API
WARM_SEEDING = False
# Expected number of bitflips per individual per generation (k in prob_var = k / n_var).
# None keeps the historical default of 200.
MUTATION_FLIP_COUNT = None  # e.g. 100, 200, 400

# Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
custom_scenario_params = {
    "max_restoration_fraction": 0.05,
    "spatial_clustering": 0,
    "biotic_effect": 0.01,
    "abiotic_effect": 0.01,
    "normalize_objectives": True,
    "burden_sharing": "no",
    # Axis 2 - restoration_potential objective formulation:
    #   "sum"       = total improvement (default)
    #   "threshold" = area of restored pixels reaching 'good' condition (> rp_threshold)
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    # Pixel-mode sampling strategy for the 2-objective potential-vs-cost run:
    #   "scattered"   = AdaptiveSampling + bitflip: the true UNCONSTRAINED front
    #                   (cherry-pick anywhere) - the baseline the contiguity sweep
    #                   is compared against. Produces very scattered plans.
    #   "region_grow" = contiguous regions; pair with min_patch_size to impose the
    #                   minimum-patch-size constraint (the "price of contiguity" sweep).
    "sampling_strategy": "region_grow",
    # Neutral repair: with "scattered", sampling (AdaptiveSampling) and mutation
    # (bitflip) are already unbiased; repair_scored=False passes scores=None to
    # AdaptiveRepair so it only enforces the budget (constraints-only, no score bias).
    # Gives an operator-unbiased front on restoration_benefit. Set True (or drop) to
    # restore score-based repair.
    "repair_scored": False,
    # Minimum-patch-size constraint (price-of-contiguity sweep axis). Every
    # 4-connected component of selected pixels must be >= min_patch_size pixels,
    # enforced by MinPatchSizeRepair. 1 = OFF (region_grow with no size floor).
    # Ignored on the "scattered" path (no contiguity constraint). The sweep is driven
    # by Debugs_tests/contiguity_price_sweep.py, which overrides this across levels.
    "min_patch_size": 2,
    # Region operator knobs (used by region_grow / the min-patch-size repair).
    "region_seeds": 25,             # max seed regions per individual (avg region ~ budget/seeds)
    "region_seeds_min": 5,          # min seed regions per individual
    "region_growth_bias": "scored", # "scored" (high value / low cost) or "neutral" (random)
    "region_random_share": 0.5,     # fraction of region seeds placed at random (vs scored)
    "region_mutation_edits": 100,   # grow/shrink edit size per mutated individual
    # Warm-start pre-optimisation budget (only used when WARM_SEEDING is True and
    # use_patch_approach is False). Objectives without a per-pixel score
    # (restoration_benefit, spatial_clustering) get a short single-objective GA
    # seed; these set that GA's population and generations. Keep small - one run
    # per such objective happens before every optimisation run.
    "warm_seed_preopt_pop": 40,
    "warm_seed_preopt_gens": 30,
}

# Named policy scenarios for SCENARIO_MODE == "policy_grid".
POLICY_VARIANTS = {
    "policy_baseline":         {},                                                  # mirrors condition baseline
    "policy_ambitious":        {"max_restoration_fraction": 0.10},                 # double the budget
    "policy_very_ambitious":   {"max_restoration_fraction": 0.20},                 # 4x the budget
    "policy_burden_shared":    {"burden_sharing": "yes"},                          # equal burden across regions
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10,
                                "burden_sharing": "yes"},                          # ambitious + burden shared
}

# Benchmark condition scenarios run in policy_grid mode with baseline params x SEEDS.
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# -- Factorial design (SCENARIO_MODE == "factorial") - iEMSs Block 4 --
FACTORIAL_FORMS = ["sum", "threshold"]
FACTORIAL_SCALINGS = ["global", "upper_q75"]
# Construction and policy axes collapsed to a single level each so this factorial
# varies ONLY forms x scalings x seeds. Re-add entries to cross them back in.
FACTORIAL_CONSTRUCTIONS = [
    "all",
]
FACTORIAL_POLICIES = {
    "status_quo":    {},                                  # baseline budget, no burden sharing
}

# Patch approach settings. The contiguity work uses the PIXEL representation
# (region operators + min-patch-size constraint), so the patch approach is off;
# set True only to run the fixed-2x2-grain patch variant instead.
USE_PATCH_APPROACH = False
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.05

# Spatial aggregation: block-coarsen all input rasters by this integer factor before optimisation.
AGGREGATION_FACTOR = None

# Set to True to save per-generation population snapshots for animation
SAVE_SNAPSHOTS = False

# Set to True to capture paired pre-repair vs post-repair genotype/phenotype diversity.
CAPTURE_REPAIR_DIAG = True
N_CAPTURE_GENS = 5

# Set to True to profile the run with cProfile and print the top 30 hotspots afterwards.
PROFILE = False
PROFILE_TOP_N = 30

# Snapshot of the full configuration block - written to run_registry.jsonl alongside results.
run_config = {
    "ecosystem": ECOSYSTEM_TO_RUN,
    "region": REGION,
    "scenario_mode": SCENARIO_MODE,
    "algorithm_type": ALGORITHM,
    "objectives": OBJECTIVES,
    "sample_fraction": SAMPLE_FRACTION,
    "sample_seed": SAMPLE_SEED,
    "pop_size": POP_SIZE,
    "n_generations": N_GENERATIONS,
    "n_jobs": N_JOBS,
    "random_seed": RANDOM_SEED,
    "n_samples_per_param": N_SAMPLES_PER_PARAM,
    "use_patch_approach": USE_PATCH_APPROACH,
    "patch_size": PATCH_SIZE,
    "patch_constraint_type": PATCH_CONSTRAINT_TYPE,
    "pixel_tolerance": PIXEL_TOLERANCE,
    "save_snapshots": SAVE_SNAPSHOTS,
    "aggregation_factor": AGGREGATION_FACTOR,
    "custom_scenario_params": custom_scenario_params,
    "seeds": SEEDS,
    "benchmark_scenarios": BENCHMARK_SCENARIOS,
    "factorial_forms": FACTORIAL_FORMS,
    "factorial_scalings": FACTORIAL_SCALINGS,
    "factorial_constructions": FACTORIAL_CONSTRUCTIONS,
    "factorial_policies": list(FACTORIAL_POLICIES.keys()),
}

if ECOSYSTEM_TO_RUN == "all":
    runs = [
        ("forest", "forest"),
        ("agricultural", "agricultural"),
        ("grassland", "grassland"),
    ]
elif ECOSYSTEM_TO_RUN == "combined":
    runs = [("combined", "all")]
else:
    runs = [(ECOSYSTEM_TO_RUN, ECOSYSTEM_TO_RUN)]


def _run_condition_grid(ecosystem_for_loader, all_results, run_times, run_label):
    """Run the condition grid across a process pool (one task per tag).

    Independent (tag x seed) runs are grouped by tag so each worker loads that
    tag's rasters once and reuses them across seeds (see grid_parallel.run_tag).
    GRID_WORKERS controls how many tags optimise concurrently; GRID_WORKERS <= 1
    falls back to an in-process sequential loop (still with per-tag caching).

    The heavy initial_conditions are loaded INSIDE each worker from the picklable
    tag string, so they never cross a process boundary. cfg carries only plain
    dicts/lists/scalars.
    """
    _condition_tags = [
        "global_all",
        "global_drop_smd", "global_drop_sbd", "global_drop_soc",
        "global_drop_uzl", "global_drop_cdi", "global_drop_swf_h",
        "global_drop_swf_t", "global_drop_ndvi",
        "upper_q75_all",
    ]
    _use_seeds = SEEDS is not None
    _seeds     = SEEDS if _use_seeds else [RANDOM_SEED]
    _grid_total = len(_condition_tags) * len(_seeds)
    _grid_times = {}

    # One parent dir for the whole grid: r_inputs/{timestamp}_{RUN_LABEL}/
    _grid_ts = datetime.now().strftime('%Y%m%d_%H%M')
    _grid_r_parent = os.path.join(str(R_INPUTS_DIR), f"{_grid_ts}_{RUN_LABEL}")
    os.makedirs(_grid_r_parent, exist_ok=True)
    print(f"  Grid R export parent: {_grid_r_parent}/")
    print(f"  Condition grid: {_grid_total} runs ({len(_condition_tags)} tags x "
          f"{len(_seeds)} seeds), GRID_WORKERS={GRID_WORKERS}")

    # Run settings shared by every task. Must be picklable (plain dicts/lists/scalars).
    # The four extra keys (vs run_custom_parallel's cfg) are read by grid_parallel._invoke
    # so this grid runs NSGA-II with the nsga2 script's mutation/diag settings.
    cfg = {
        "objectives": OBJECTIVES,
        "region": REGION,
        "ecosystem": ecosystem_for_loader,
        "sample_fraction": SAMPLE_FRACTION,
        "sample_seed": SAMPLE_SEED,
        "aggregation_factor": AGGREGATION_FACTOR,
        "scenario_params": custom_scenario_params,
        "pop_size": POP_SIZE,
        "n_generations": N_GENERATIONS,
        "use_patch_approach": USE_PATCH_APPROACH,
        "patch_size": PATCH_SIZE,
        "patch_constraint_type": PATCH_CONSTRAINT_TYPE,
        "pixel_tolerance": PIXEL_TOLERANCE,
        "save_snapshots": SAVE_SNAPSHOTS,
        "n_partitions": N_PARTITIONS,
        "warm_seeding": WARM_SEEDING,
        "run_config": run_config,
        "r_parent": _grid_r_parent,
        # NSGA-II-specific knobs forwarded by grid_parallel._invoke:
        "algorithm_type": ALGORITHM,
        "mutation_flip_count": MUTATION_FLIP_COUNT,
        "capture_repair_diag": CAPTURE_REPAIR_DIAG,
        "n_capture_gens": N_CAPTURE_GENS,
        # Only echo per-run verbose output for a single worker, else parallel
        # stdout interleaves into noise.
        "verbose": GRID_WORKERS == 1,
    }
    tasks = [
        {"tag": _tag, "seeds": _seeds, "use_seeds": _use_seeds, "cfg": cfg}
        for _tag in _condition_tags
    ]

    _grid_start = time.perf_counter()

    def _record(summaries):
        for run_lbl, ok, elapsed, err in summaries:
            _grid_times[run_lbl] = elapsed
            if ok:
                # Full results are on disk; store a marker so all_results counts succeed.
                all_results[run_lbl] = True
                print(f"  [ok] {run_lbl} completed ({elapsed/60:.1f} min)")
            else:
                msg = f": {err}" if err else ""
                print(f"  [x] {run_lbl} failed{msg} ({elapsed/60:.1f} min)")

    if GRID_WORKERS <= 1:
        # Sequential, in-process fallback (still loads each tag's rasters once).
        for task in tasks:
            print(f"  [tag] {task['tag']} (sequential)")
            _record(run_tag(task))
    else:
        with ProcessPoolExecutor(max_workers=GRID_WORKERS) as ex:
            futures = {ex.submit(run_tag, task): task["tag"] for task in tasks}
            for fut in as_completed(futures):
                _tag = futures[fut]
                try:
                    _record(fut.result())
                except Exception as _e:  # a worker crashed entirely
                    print(f"  [x] tag '{_tag}' worker crashed: {_e}")

    _grid_elapsed = time.perf_counter() - _grid_start
    _grid_succeeded = len(all_results)
    print(f"\n=== CONDITION GRID COMPLETE ===")
    print(f"  {_grid_succeeded}/{_grid_total} runs succeeded")
    print(f"  Total grid time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
    if _grid_times:
        _avg = sum(_grid_times.values()) / len(_grid_times)
        print(f"  Average per run: {_avg/60:.1f} min ({_avg:.0f} s)")
    print(f"  Results PKLs: results_files/res_<timestamp>_<run_label>.pkl (one per grid element)")

    run_times[run_label] = _grid_elapsed


def _run():
    """Main execution body - separated so cProfile can wrap it cleanly."""
    global all_results, run_times
    all_results = {}
    run_times = {}
    _script_start = time.perf_counter()

    for run_label, ecosystem_for_loader in runs:
        print(f"=== Starting optimisation for {run_label.upper()} ecosystem ===")
        _run_start = time.perf_counter()

        try:
            if SCENARIO_MODE == "all":
                results = main(
                    workspace_dir=".",
                    scenario="all",
                    objectives=OBJECTIVES,
                    n_samples_per_param=N_SAMPLES_PER_PARAM,
                    pop_size=POP_SIZE,
                    n_generations=N_GENERATIONS,
                    save_results=True,
                    verbose=True,
                    random_seed=RANDOM_SEED,
                    sample_fraction=SAMPLE_FRACTION,
                    sample_seed=SAMPLE_SEED,
                    ecosystem=ecosystem_for_loader,
                    lulc_path=None,
                )
            elif SCENARIO_MODE == "condition_grid":
                # Parallelised across a process pool (GRID_WORKERS), one task per
                # condition tag; see _run_condition_grid.
                _run_condition_grid(ecosystem_for_loader, all_results, run_times, run_label)
            elif SCENARIO_MODE == "policy_grid":
                # Run each named policy variant x SEEDS against CONDITION_SCENARIO.
                _pg_use_seeds = SEEDS is not None
                _pg_seeds     = SEEDS if _pg_use_seeds else [RANDOM_SEED]
                _pg_total     = len(POLICY_VARIANTS) * len(_pg_seeds) + len(BENCHMARK_SCENARIOS) * len(_pg_seeds)

                print(f"\n  Policy variants to run ({len(POLICY_VARIANTS)} x {len(_pg_seeds)} seeds):")
                for _vname, _vparams in POLICY_VARIANTS.items():
                    _diff = {k: v for k, v in _vparams.items()} or {"(baseline - no overrides)": ""}
                    print(f"    {_vname}: {_diff}")
                if BENCHMARK_SCENARIOS:
                    print(f"  Benchmark scenarios ({len(BENCHMARK_SCENARIOS)} x {len(_pg_seeds)} seeds):")
                    for _btag in BENCHMARK_SCENARIOS:
                        print(f"    {_btag}")

                results    = None
                _grid_start = time.perf_counter()
                _grid_done  = 0
                _grid_times = {}
                import os as _os
                from datetime import datetime as _dt
                _grid_ts = _dt.now().strftime('%Y%m%d_%H%M')
                _pg_r_parent = _os.path.join(str(R_INPUTS_DIR), f"{_grid_ts}_{RUN_LABEL}")
                _os.makedirs(_pg_r_parent, exist_ok=True)
                print(f"  Grid R export parent: {_pg_r_parent}/")
                # -- policy variants --
                _ic_policy = load_initial_conditions(
                    ".",
                    objectives=OBJECTIVES,
                    region=REGION,
                    ecosystem=ecosystem_for_loader,
                    sample_fraction=SAMPLE_FRACTION,
                    sample_seed=SAMPLE_SEED,
                    aggregation_factor=AGGREGATION_FACTOR,
                    condition_scenario=CONDITION_SCENARIO,
                )

                for _vname, _vparams in POLICY_VARIANTS.items():
                    for _seed in _pg_seeds:
                        _run_lbl = f"{_vname}_seed{_seed}" if _pg_use_seeds else _vname
                        _item_start    = time.perf_counter()
                        _merged_params = {**custom_scenario_params, **_vparams}
                        _grid_config   = {**run_config, "policy_variant": _vname,
                                          "condition_scenario": CONDITION_SCENARIO,
                                          "random_seed": _seed}
                        _grid_done += 1
                        print(f"  policy_grid [{_grid_done}/{_pg_total}]: {_run_lbl}")
                        try:
                            _presults = run_optimization_instance(
                                initial_conditions=_ic_policy,
                                scenario_params=_merged_params,
                                pop_size=POP_SIZE,
                                n_generations=N_GENERATIONS,
                                save_results=True,
                                verbose=True,
                                n_jobs=N_JOBS,
                                random_seed=_seed,
                                use_repair=True,
                                use_patch_approach=USE_PATCH_APPROACH,
                                patch_size=PATCH_SIZE,
                                patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                                pixel_tolerance=PIXEL_TOLERANCE,
                                save_snapshots=SAVE_SNAPSHOTS,
                                capture_repair_diag=CAPTURE_REPAIR_DIAG,
                                n_capture_gens=N_CAPTURE_GENS,
                                n_partitions=N_PARTITIONS,
                                warm_seeding=WARM_SEEDING,
                                run_label=_run_lbl,
                                run_config=_grid_config,
                                r_export_parent=_pg_r_parent,
                                mutation_flip_count=MUTATION_FLIP_COUNT,
                                algorithm_type=ALGORITHM,
                            )
                            _item_elapsed = time.perf_counter() - _item_start
                            _grid_times[_run_lbl] = _item_elapsed
                            if _presults is not None:
                                all_results[_run_lbl] = _presults
                                print(f"  [ok] {_run_lbl} completed ({_item_elapsed/60:.1f} min)")
                            else:
                                print(f"  [x] {_run_lbl} returned no results ({_item_elapsed/60:.1f} min)")
                        except Exception as _e:
                            _grid_times[_run_lbl] = time.perf_counter() - _item_start
                            print(f"  [x] {_run_lbl} failed: {_e}")

                # -- benchmark scenarios --
                for _btag in BENCHMARK_SCENARIOS:
                    _ic_bench = load_initial_conditions(
                        ".",
                        objectives=OBJECTIVES,
                        region=REGION,
                        ecosystem=ecosystem_for_loader,
                        sample_fraction=SAMPLE_FRACTION,
                        sample_seed=SAMPLE_SEED,
                        aggregation_factor=AGGREGATION_FACTOR,
                        condition_scenario=_btag,
                    )
                    for _seed in _pg_seeds:
                        _run_lbl   = f"{_btag}_seed{_seed}" if _pg_use_seeds else _btag
                        _item_start = time.perf_counter()
                        _grid_config = {**run_config, "benchmark_scenario": _btag,
                                        "condition_scenario": _btag,
                                        "random_seed": _seed}
                        _grid_done += 1
                        print(f"  policy_grid [{_grid_done}/{_pg_total}]: {_run_lbl} (benchmark)")
                        try:
                            _bresults = run_optimization_instance(
                                initial_conditions=_ic_bench,
                                scenario_params=custom_scenario_params,
                                pop_size=POP_SIZE,
                                n_generations=N_GENERATIONS,
                                save_results=True,
                                verbose=True,
                                n_jobs=N_JOBS,
                                random_seed=_seed,
                                use_repair=True,
                                use_patch_approach=USE_PATCH_APPROACH,
                                patch_size=PATCH_SIZE,
                                patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                                pixel_tolerance=PIXEL_TOLERANCE,
                                save_snapshots=SAVE_SNAPSHOTS,
                                capture_repair_diag=CAPTURE_REPAIR_DIAG,
                                n_capture_gens=N_CAPTURE_GENS,
                                n_partitions=N_PARTITIONS,
                                warm_seeding=WARM_SEEDING,
                                run_label=_run_lbl,
                                run_config=_grid_config,
                                r_export_parent=_pg_r_parent,
                                mutation_flip_count=MUTATION_FLIP_COUNT,
                                algorithm_type=ALGORITHM,
                            )
                            _item_elapsed = time.perf_counter() - _item_start
                            _grid_times[_run_lbl] = _item_elapsed
                            if _bresults is not None:
                                all_results[_run_lbl] = _bresults
                                print(f"  [ok] {_run_lbl} completed ({_item_elapsed/60:.1f} min)")
                            else:
                                print(f"  [x] {_run_lbl} returned no results ({_item_elapsed/60:.1f} min)")
                        except Exception as _e:
                            _grid_times[_run_lbl] = time.perf_counter() - _item_start
                            print(f"  [x] {_run_lbl} failed: {_e}")

                _grid_elapsed   = time.perf_counter() - _grid_start
                _grid_succeeded = len(all_results)
                print(f"\n=== POLICY GRID COMPLETE ===")
                print(f"  {_grid_succeeded}/{_pg_total} runs succeeded")
                print(f"  Total time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
                if _grid_times:
                    _avg = sum(_grid_times.values()) / len(_grid_times)
                    print(f"  Average per run: {_avg/60:.1f} min ({_avg:.0f} s)")
                print(f"  Results PKLs: results_files/res_<timestamp>_<label>.pkl")
            elif SCENARIO_MODE == "factorial":
                # Fully-crossed design: form x scaling x construction x policy x SEEDS.
                results = None
                _use_seeds = SEEDS is not None
                _seeds     = SEEDS if _use_seeds else [RANDOM_SEED]
                _grid_start = time.perf_counter()
                _grid_total = (len(FACTORIAL_FORMS) * len(FACTORIAL_SCALINGS)
                               * len(FACTORIAL_CONSTRUCTIONS) * len(FACTORIAL_POLICIES)
                               * len(_seeds))
                _grid_done  = 0
                _grid_times = {}
                import os as _os
                from datetime import datetime as _dt
                _grid_ts = _dt.now().strftime('%Y%m%d_%H%M')
                _grid_r_parent = _os.path.join(str(R_INPUTS_DIR), f"{_grid_ts}_{RUN_LABEL}")
                _os.makedirs(_grid_r_parent, exist_ok=True)
                print(f"  Factorial design: {_grid_total} runs "
                      f"({len(FACTORIAL_FORMS)} form x {len(FACTORIAL_SCALINGS)} scaling x "
                      f"{len(FACTORIAL_CONSTRUCTIONS)} construction x "
                      f"{len(FACTORIAL_POLICIES)} policy x {len(_seeds)} seed)")
                print(f"  Grid R export parent: {_grid_r_parent}/")

                _ic_cache = {}  # condition_scenario tag -> initial_conditions (or None if missing)
                for _seed in _seeds:
                    for _scaling in FACTORIAL_SCALINGS:
                        for _construction in FACTORIAL_CONSTRUCTIONS:
                            _tag = f"{_scaling}_{_construction}"
                            if _tag not in _ic_cache:
                                try:
                                    _ic_cache[_tag] = load_initial_conditions(
                                        ".",
                                        objectives=OBJECTIVES,
                                        region=REGION,
                                        ecosystem=ecosystem_for_loader,
                                        sample_fraction=SAMPLE_FRACTION,
                                        sample_seed=SAMPLE_SEED,
                                        aggregation_factor=AGGREGATION_FACTOR,
                                        condition_scenario=_tag,
                                    )
                                except Exception as _e:
                                    print(f"  [x] could not load condition rasters for tag '{_tag}': {_e}")
                                    _ic_cache[_tag] = None
                            _ic = _ic_cache[_tag]
                            if _ic is None:
                                _skipped = len(FACTORIAL_FORMS) * len(FACTORIAL_POLICIES)
                                _grid_done += _skipped
                                print(f"  [x] skipping {_skipped} cells needing missing tag '{_tag}'")
                                continue
                            for _form in FACTORIAL_FORMS:
                                for _pol_name, _pol_overrides in FACTORIAL_POLICIES.items():
                                    _run_lbl = (f"form-{_form}__scal-{_scaling}__"
                                                f"con-{_construction}__pol-{_pol_name}__seed{_seed}")
                                    _item_start = time.perf_counter()
                                    _grid_done += 1
                                    print(f"  factorial [{_grid_done}/{_grid_total}]: {_run_lbl}")
                                    _merged_params = {
                                        **custom_scenario_params,
                                        "rp_formulation": _form,
                                        **_pol_overrides,
                                    }
                                    _grid_config = {
                                        **run_config,
                                        "condition_scenario": _tag,
                                        "random_seed": _seed,
                                        "factor_form": _form,
                                        "factor_scaling": _scaling,
                                        "factor_construction": _construction,
                                        "factor_policy": _pol_name,
                                    }
                                    try:
                                        _fresults = run_optimization_instance(
                                            initial_conditions=_ic,
                                            scenario_params=_merged_params,
                                            pop_size=POP_SIZE,
                                            n_generations=N_GENERATIONS,
                                            save_results=True,
                                            verbose=True,
                                            n_jobs=N_JOBS,
                                            random_seed=_seed,
                                            use_repair=True,
                                            use_patch_approach=USE_PATCH_APPROACH,
                                            patch_size=PATCH_SIZE,
                                            patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                                            pixel_tolerance=PIXEL_TOLERANCE,
                                            save_snapshots=SAVE_SNAPSHOTS,
                                            capture_repair_diag=CAPTURE_REPAIR_DIAG,
                                            n_capture_gens=N_CAPTURE_GENS,
                                            n_partitions=N_PARTITIONS,
                                            warm_seeding=WARM_SEEDING,
                                            run_label=_run_lbl,
                                            run_config=_grid_config,
                                            r_export_parent=_grid_r_parent,
                                            mutation_flip_count=MUTATION_FLIP_COUNT,
                                            algorithm_type=ALGORITHM,
                                        )
                                        _item_elapsed = time.perf_counter() - _item_start
                                        _grid_times[_run_lbl] = _item_elapsed
                                        if _fresults is not None:
                                            all_results[_run_lbl] = _fresults
                                            print(f"  [ok] {_run_lbl} completed ({_item_elapsed/60:.1f} min)")
                                        else:
                                            print(f"  [x] {_run_lbl} returned no results ({_item_elapsed/60:.1f} min)")
                                    except Exception as _e:
                                        _grid_times[_run_lbl] = time.perf_counter() - _item_start
                                        print(f"  [x] {_run_lbl} failed: {_e}")

                _grid_elapsed   = time.perf_counter() - _grid_start
                _grid_succeeded = len(all_results)
                print(f"\n=== FACTORIAL GRID COMPLETE ===")
                print(f"  {_grid_succeeded}/{_grid_total} runs succeeded")
                print(f"  Total grid time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
                if _grid_times:
                    _avg = sum(_grid_times.values()) / len(_grid_times)
                    print(f"  Average per run: {_avg/60:.1f} min ({_avg:.0f} s)")
                print(f"  Results PKLs: results_files/res_<timestamp>_<run_label>.pkl")
            else:
                # Data load is seed-independent - load once, reuse across seeds.
                initial_conditions = load_initial_conditions(
                    ".",
                    objectives=OBJECTIVES,
                    region=REGION,
                    ecosystem=ecosystem_for_loader,
                    sample_fraction=SAMPLE_FRACTION,
                    sample_seed=SAMPLE_SEED,
                    aggregation_factor=AGGREGATION_FACTOR,
                    condition_scenario=CONDITION_SCENARIO,
                )
                print(f"[ok] Data loaded for {run_label}")

                # SEEDS (when set) replicates the run once per seed; SEEDS=None -> one run
                # at RANDOM_SEED (original behaviour). Each seed gets its own random_seed,
                # run_label and run_config so the pkls and registry record the right seed.
                _seeds = SEEDS if _MULTISEED else [RANDOM_SEED]
                if _MULTISEED:
                    print(f"  Custom run replicated across {len(_seeds)} seeds: {_seeds}")

                results = None
                for _seed in _seeds:
                    _run_lbl = f"{RUN_LABEL}_seed{_seed}" if _MULTISEED else RUN_LABEL
                    _seed_start = time.perf_counter()
                    if _MULTISEED:
                        print(f"\n--- seed {_seed} -> {_run_lbl} ---")
                    results = run_optimization_instance(
                        initial_conditions=initial_conditions,
                        scenario_params=custom_scenario_params,
                        pop_size=POP_SIZE,
                        n_generations=N_GENERATIONS,
                        save_results=True,
                        verbose=True,
                        n_jobs=N_JOBS,
                        random_seed=_seed,
                        use_repair=True,
                        use_patch_approach=USE_PATCH_APPROACH,
                        patch_size=PATCH_SIZE,
                        patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                        pixel_tolerance=PIXEL_TOLERANCE,
                        save_snapshots=SAVE_SNAPSHOTS,
                        capture_repair_diag=CAPTURE_REPAIR_DIAG,
                        n_capture_gens=N_CAPTURE_GENS,
                        n_partitions=N_PARTITIONS,
                        warm_seeding=WARM_SEEDING,
                        run_label=_run_lbl,
                        run_config={**run_config, "random_seed": _seed} if _MULTISEED else run_config,
                        mutation_flip_count=MUTATION_FLIP_COUNT,
                        algorithm_type=ALGORITHM,
                    )
                    if _MULTISEED:
                        # store per-seed (the single-run post-block below is skipped when
                        # _MULTISEED, so it does not double-store the last seed's result)
                        _res_key = f"{run_label}_seed{_seed}"
                        if results is not None:
                            all_results[_res_key] = results
                        run_times[_res_key] = time.perf_counter() - _seed_start
                        print(f"  [{'ok' if results is not None else 'x'}] {_run_lbl} "
                              f"({run_times[_res_key]/60:.1f} min)")

            if not _MULTISEED and SCENARIO_MODE not in ("condition_grid", "policy_grid", "factorial"):
                if results is not None:
                    all_results[run_label] = results
                    run_times[run_label] = time.perf_counter() - _run_start
                    print(f"\n[ok] {run_label.title()} optimisation completed successfully!")
                else:
                    run_times[run_label] = time.perf_counter() - _run_start
                    print(f"\n[x] {run_label.title()} optimisation failed.")
            else:
                run_times[run_label] = time.perf_counter() - _run_start

        except Exception as e:
            run_times[run_label] = time.perf_counter() - _run_start
            print(f"\n[x] Error optimising {run_label} ecosystem: {e}")
            continue

    print(f"\n{'='*80}")
    print("=== OPTIMISATION SUMMARY ===")
    print(f"{'='*80}")
    _total_elapsed = time.perf_counter() - _script_start

    if _MULTISEED or SCENARIO_MODE in ("condition_grid", "policy_grid", "factorial"):
        grid_succeeded = len(all_results)
        _label = f"custom x {len(SEEDS)} seeds" if _MULTISEED else SCENARIO_MODE
        print(f"{_label}: {grid_succeeded} runs stored in all_results")
        print(f"\nTotal wall-clock time: {_total_elapsed/60:.1f} min ({_total_elapsed:.0f} s)")
    else:
        successful_runs = len(all_results)
        total_runs = len(runs)
        print(f"Successfully completed {successful_runs}/{total_runs} ecosystem optimisations:")
        for run_label, _ in runs:
            status = "[ok] SUCCESS" if run_label in all_results else "[x] FAILED"
            elapsed = run_times.get(run_label)
            time_str = f"  ({elapsed/60:.1f} min)" if elapsed is not None else ""
            print(f"  {run_label.title():<12}: {status}{time_str}")

        print(f"\nTotal wall-clock time: {_total_elapsed/60:.1f} min ({_total_elapsed:.0f} s)")

        if successful_runs > 0:
            print(f"[ok] Completed with outputs for {successful_runs} ecosystems.")
        else:
            print("[x] No optimisations completed successfully.")


# The __main__ guard is REQUIRED: condition_grid uses a ProcessPoolExecutor, and on
# the 'spawn' start method (Windows default) each worker re-imports this module. Without
# the guard, every worker would re-launch the whole grid. Workers import this module as
# '__mp_main__', so this block is skipped in them.
if __name__ == "__main__":
    if PROFILE:
        _profiler = cProfile.Profile()
        _profiler.enable()
        _run()
        _profiler.disable()
        _buf = io.StringIO()
        pstats.Stats(_profiler, stream=_buf).sort_stats("cumulative").print_stats(PROFILE_TOP_N)
        _profile_text = _buf.getvalue()
        print("\n" + _profile_text)
        _profile_path = f"logs/profile_{ECOSYSTEM_TO_RUN}_{RUN_LABEL}.txt"
        with open(_profile_path, "w") as _f:
            _f.write(_profile_text)
        print(f"Profile saved -> {_profile_path}")
    else:
        _run()
