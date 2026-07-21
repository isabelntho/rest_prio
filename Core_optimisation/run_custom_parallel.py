"""
Parallel variant of run_custom.py.
==================================

Identical configuration and behaviour to run_custom.py, except the
``condition_grid`` scenario mode runs its independent (tag × seed) optimisations
across a process pool instead of a sequential for-loop. See grid_parallel.py for
the worker and the rationale (GIL-bound within-run eval → parallelise at the grid
level with processes).

Run from the project root as a module so the package-relative imports and the
``spawn`` worker re-import resolve:

    python -m Core_optimisation.run_custom_parallel

Set GRID_WORKERS below: 1 = sequential, in-process fallback (still loads each
tag's rasters once, unlike the original); N = up to N tags optimised at once.
Bound N by RAM, not cores — each concurrent run holds its own ~1.3M-pixel
rasters.

NOTE: the module-level execution is guarded by ``if __name__ == "__main__"``.
This is mandatory on Windows: ``spawn`` re-imports this module in every worker,
and without the guard that would re-trigger the grid recursively.
"""
import cProfile
import pstats
import io
import time
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from .resto_anom import run_optimization_instance, main
from .data_loader import load_initial_conditions
from .grid_parallel import run_tag, run_factorial_cell
from .logger_setup import setup_logger

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
RUN_LABEL = "iEMSs_fullfact_clustobj"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "all"            runs scenario="all" using the scenario sampling logic inside main()
#   "condition_grid" sweeps all 13 condition scenarios x SEEDS  (PARALLELISED here)
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
#   "factorial"      fully-crossed design (iEMSs Block 4)
SCENARIO_MODE = "factorial"

# Number of worker PROCESSES for condition_grid. 1 = sequential in-process
# fallback. Bound by available RAM (each concurrent run holds its own rasters).
GRID_WORKERS = 4

# Condition scenario tag — selects pre-computed anomaly rasters from inputs/anomaly_scenarios/
CONDITION_SCENARIO = "global_all"

# Seeds used by both condition_grid and policy_grid modes.
SEEDS = [101, 102, 103, 104, 105]  # 5 seed replicates (iEMSs run matrix, Block 0)

print(f"\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
print(f"Scenario mode: {SCENARIO_MODE}")

log_path = setup_logger(log_dir="logs", run_label=f"{ECOSYSTEM_TO_RUN}_{REGION.lower()}")
if log_path:
    print(f"Verbose output → {log_path}")

OBJECTIVES = ["restoration_potential", "spatial_clustering", "cost"]  # iEMSs headline objectives
# Available objective names:
#   "abiotic"               – minimise abiotic condition anomaly (restoration pixels)
#   "biotic"                – minimise biotic condition anomaly (restoration pixels)
#   "cost"                  – minimise implementation cost
#   "connectivity"          – maximise connectivity gain (precomputed per pixel)
#   "landscape_context"     – minimise mean abiotic anomaly of eligible neighbours within 500 m
#   "restoration_potential" – minimise mean per-pixel abiotic + biotic baseline anomaly
#   "spatial_clustering"    – maximise spatial compactness of the selected pixels
#                             (configuration-dependent; distinct from the spatial_clustering
#                             sampling-bias knob in custom_scenario_params)
#   "es_future_val"         – maximise total ES performance under future scenarios
#   "es_future_robustness"  – minimise total ES instability under future scenarios

SAMPLE_FRACTION = None
SAMPLE_SEED = 42
POP_SIZE = 92
N_GENERATIONS = 100
N_JOBS = 12  # within-run thread pool (GIL-bound; ignored by the parallel grid path)
RANDOM_SEED = 100
N_SAMPLES_PER_PARAM = 3
N_PARTITIONS = 12
WARM_SEEDING = True

# Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
custom_scenario_params = {
    "max_restoration_fraction": 0.05,
    "spatial_clustering": 0,
    "biotic_effect": 0.01,
    "abiotic_effect": 0.01,
    "normalize_objectives": True,
    "patch_score_temperature": 2.0,
    "patch_repair_top_k": 100,
    "burden_sharing": "no",
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    # Metric for the "spatial_clustering" objective (only used when it is in OBJECTIVES):
    #   "adjacency"  = shared-edge count (compactness)
    #   "components" = number of disconnected clusters (fragmentation)
    "clustering_metric": "adjacency",
}

# Named policy scenarios for SCENARIO_MODE == "policy_grid".
POLICY_VARIANTS = {
    "policy_baseline":         {},
    "policy_ambitious":        {"max_restoration_fraction": 0.10},
    "policy_very_ambitious":   {"max_restoration_fraction": 0.20},
    "policy_burden_shared":    {"burden_sharing": "yes"},
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10,
                                "burden_sharing": "yes"},
}

# Benchmark condition scenarios run in policy_grid mode with baseline params × SEEDS.
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# ── Factorial design (SCENARIO_MODE == "factorial") — iEMSs Block 4 ──────────
FACTORIAL_FORMS = ["sum", "threshold"]
FACTORIAL_SCALINGS = ["global", "upper_q75"]
FACTORIAL_CONSTRUCTIONS = [
    "all",
    "drop_sbd",
    "drop_soc",
    "drop_ndvi",
    # "drop_<highest_leverage_1>",   # ← fill from Block 2 R3c Jaccard
    # "drop_<highest_leverage_2>",   # ← fill from Block 2 R3c Jaccard
]
FACTORIAL_POLICIES = {
    "status_quo":    {},
    "ambitious":     {"max_restoration_fraction": 0.10},
    "burden_shared": {"burden_sharing": "yes"},
}

# Patch approach settings
USE_PATCH_APPROACH = True
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.05 #previously 0.15

# Spatial aggregation factor (None/1 disables).
AGGREGATION_FACTOR = None

# Save per-generation population snapshots for animation.
SAVE_SNAPSHOTS = False

# Profile the run with cProfile.
PROFILE = False
PROFILE_TOP_N = 30

# Snapshot of the full configuration block — written to run_registry.jsonl alongside results.
run_config = {
    "ecosystem": ECOSYSTEM_TO_RUN,
    "region": REGION,
    "scenario_mode": SCENARIO_MODE,
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
    "grid_workers": GRID_WORKERS,
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

    Independent (tag × seed) runs are grouped by tag so that each worker loads
    that tag's rasters once and reuses them across seeds. GRID_WORKERS controls
    how many tags are optimised concurrently; GRID_WORKERS == 1 falls back to an
    in-process sequential loop (still with per-tag caching).
    """
    # Agricultural-focus LOO set (iEMSs Blocks 1+2) + the q75 benchmark for R3.
    _condition_tags = [
        "global_all",
        "global_drop_smd", "global_drop_sbd", "global_drop_soc",
        "global_drop_uzl", "global_drop_cdi", "global_drop_swf_h",
        "global_drop_swf_t", "global_drop_ndvi",
        "upper_q75_all",
    ]
    _use_seeds = SEEDS is not None
    _seeds = SEEDS if _use_seeds else [RANDOM_SEED]
    _grid_total = len(_condition_tags) * len(_seeds)
    _grid_times = {}

    # One parent dir for the whole grid: r_inputs/{timestamp}_{RUN_LABEL}/
    _grid_ts = datetime.now().strftime('%Y%m%d_%H%M')
    _grid_r_parent = os.path.join("r_inputs", f"{_grid_ts}_{RUN_LABEL}")
    os.makedirs(_grid_r_parent, exist_ok=True)
    print(f"  Grid R export parent: {_grid_r_parent}/")
    print(f"  Condition grid: {_grid_total} runs ({len(_condition_tags)} tags × {len(_seeds)} seeds), "
          f"GRID_WORKERS={GRID_WORKERS}")

    # Run settings shared by every task. Must be picklable (plain dicts/lists/scalars).
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
        # Only echo per-run verbose output when there is a single worker, else
        # parallel stdout interleaves into noise.
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
                print(f"  ✓ {run_lbl} completed ({elapsed/60:.1f} min)")
            else:
                msg = f": {err}" if err else ""
                print(f"  ✗ {run_lbl} failed{msg} ({elapsed/60:.1f} min)")

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
                except Exception as _e:  # noqa: BLE001 — a worker crashed entirely
                    print(f"  ✗ tag '{_tag}' worker crashed: {_e}")

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


def _run_factorial(ecosystem_for_loader, all_results, run_times, run_label):
    """Run the factorial design across a process pool (one task per (tag, seed) cell).

    The fully-crossed design is form × scaling × construction × policy × seed.
    scaling × construction selects the condition raster tag; form + policy are
    scenario_params overrides applied per cell. Tasks are grouped by (tag, seed)
    so each worker loads that tag's rasters once and reuses them across the
    form × policy combinations at that seed. GRID_WORKERS controls how many cells
    run concurrently; GRID_WORKERS == 1 falls back to an in-process loop.
    """
    _use_seeds = SEEDS is not None
    _seeds = SEEDS if _use_seeds else [RANDOM_SEED]
    _grid_total = (len(FACTORIAL_FORMS) * len(FACTORIAL_SCALINGS)
                   * len(FACTORIAL_CONSTRUCTIONS) * len(FACTORIAL_POLICIES)
                   * len(_seeds))
    _grid_times = {}

    _grid_ts = datetime.now().strftime('%Y%m%d_%H%M')
    _grid_r_parent = os.path.join("r_inputs", f"{_grid_ts}_{RUN_LABEL}")
    os.makedirs(_grid_r_parent, exist_ok=True)
    print(f"  Factorial design: {_grid_total} runs "
          f"({len(FACTORIAL_FORMS)} form × {len(FACTORIAL_SCALINGS)} scaling × "
          f"{len(FACTORIAL_CONSTRUCTIONS)} construction × "
          f"{len(FACTORIAL_POLICIES)} policy × {len(_seeds)} seed)")
    print(f"  Grid R export parent: {_grid_r_parent}/")
    print(f"  GRID_WORKERS={GRID_WORKERS}, "
          f"{len(FACTORIAL_SCALINGS) * len(FACTORIAL_CONSTRUCTIONS) * len(_seeds)} (tag, seed) cells")

    cfg = {
        "objectives": OBJECTIVES,
        "region": REGION,
        "ecosystem": ecosystem_for_loader,
        "sample_fraction": SAMPLE_FRACTION,
        "sample_seed": SAMPLE_SEED,
        "aggregation_factor": AGGREGATION_FACTOR,
        # Base params; form (rp_formulation) and policy overrides are merged per cell.
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
        "verbose": GRID_WORKERS == 1,
    }
    tasks = [
        {"scaling": _scaling, "construction": _construction, "seed": _seed,
         "forms": FACTORIAL_FORMS, "policies": FACTORIAL_POLICIES, "cfg": cfg}
        for _seed in _seeds
        for _scaling in FACTORIAL_SCALINGS
        for _construction in FACTORIAL_CONSTRUCTIONS
    ]

    _grid_start = time.perf_counter()

    def _record(summaries):
        for run_lbl, ok, elapsed, err in summaries:
            _grid_times[run_lbl] = elapsed
            if ok:
                all_results[run_lbl] = True
                print(f"  ✓ {run_lbl} completed ({elapsed/60:.1f} min)")
            else:
                msg = f": {err}" if err else ""
                print(f"  ✗ {run_lbl} failed{msg} ({elapsed/60:.1f} min)")

    if GRID_WORKERS <= 1:
        for task in tasks:
            print(f"  [cell] {task['scaling']}_{task['construction']} seed{task['seed']} (sequential)")
            _record(run_factorial_cell(task))
    else:
        with ProcessPoolExecutor(max_workers=GRID_WORKERS) as ex:
            futures = {
                ex.submit(run_factorial_cell, task):
                    (task["scaling"], task["construction"], task["seed"])
                for task in tasks
            }
            for fut in as_completed(futures):
                _key = futures[fut]
                try:
                    _record(fut.result())
                except Exception as _e:  # noqa: BLE001 — a worker crashed entirely
                    print(f"  ✗ cell {_key} worker crashed: {_e}")

    _grid_elapsed = time.perf_counter() - _grid_start
    _grid_succeeded = len(all_results)
    print(f"\n=== FACTORIAL GRID COMPLETE ===")
    print(f"  {_grid_succeeded}/{_grid_total} runs succeeded")
    print(f"  Total grid time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
    if _grid_times:
        _avg = sum(_grid_times.values()) / len(_grid_times)
        print(f"  Average per run: {_avg/60:.1f} min ({_avg:.0f} s)")
    print(f"  Results PKLs: results_files/res_<timestamp>_<run_label>.pkl")

    run_times[run_label] = _grid_elapsed


def _run():
    """Main execution body — separated so cProfile can wrap it cleanly."""
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
                _run_condition_grid(ecosystem_for_loader, all_results, run_times, run_label)
                continue  # per-run summary handled inside; run_times already set
            elif SCENARIO_MODE == "policy_grid":
                # Run each named policy variant × SEEDS against CONDITION_SCENARIO.
                _pg_use_seeds = SEEDS is not None
                _pg_seeds     = SEEDS if _pg_use_seeds else [RANDOM_SEED]
                _pg_total     = len(POLICY_VARIANTS) * len(_pg_seeds) + len(BENCHMARK_SCENARIOS) * len(_pg_seeds)

                print(f"\n  Policy variants to run ({len(POLICY_VARIANTS)} × {len(_pg_seeds)} seeds):")
                for _vname, _vparams in POLICY_VARIANTS.items():
                    _diff = {k: v for k, v in _vparams.items()} or {"(baseline — no overrides)": ""}
                    print(f"    {_vname}: {_diff}")
                if BENCHMARK_SCENARIOS:
                    print(f"  Benchmark scenarios ({len(BENCHMARK_SCENARIOS)} × {len(_pg_seeds)} seeds):")
                    for _btag in BENCHMARK_SCENARIOS:
                        print(f"    {_btag}")

                results    = None
                _grid_start = time.perf_counter()
                _grid_done  = 0
                _grid_times = {}
                _grid_ts = datetime.now().strftime('%Y%m%d_%H%M')
                _pg_r_parent = os.path.join("r_inputs", f"{_grid_ts}_{RUN_LABEL}")
                os.makedirs(_pg_r_parent, exist_ok=True)
                print(f"  Grid R export parent: {_pg_r_parent}/")
                # ── policy variants ──────────────────────────────────────────────
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
                                n_partitions=N_PARTITIONS,
                                warm_seeding=WARM_SEEDING,
                                run_label=_run_lbl,
                                run_config=_grid_config,
                                r_export_parent=_pg_r_parent,
                            )
                            _item_elapsed = time.perf_counter() - _item_start
                            _grid_times[_run_lbl] = _item_elapsed
                            if _presults is not None:
                                all_results[_run_lbl] = _presults
                                print(f"  ✓ {_run_lbl} completed ({_item_elapsed/60:.1f} min)")
                            else:
                                print(f"  ✗ {_run_lbl} returned no results ({_item_elapsed/60:.1f} min)")
                        except Exception as _e:
                            _grid_times[_run_lbl] = time.perf_counter() - _item_start
                            print(f"  ✗ {_run_lbl} failed: {_e}")

                # ── benchmark scenarios ──────────────────────────────────────────
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
                                n_partitions=N_PARTITIONS,
                                warm_seeding=WARM_SEEDING,
                                run_label=_run_lbl,
                                run_config=_grid_config,
                                r_export_parent=_pg_r_parent,
                            )
                            _item_elapsed = time.perf_counter() - _item_start
                            _grid_times[_run_lbl] = _item_elapsed
                            if _bresults is not None:
                                all_results[_run_lbl] = _bresults
                                print(f"  ✓ {_run_lbl} completed ({_item_elapsed/60:.1f} min)")
                            else:
                                print(f"  ✗ {_run_lbl} returned no results ({_item_elapsed/60:.1f} min)")
                        except Exception as _e:
                            _grid_times[_run_lbl] = time.perf_counter() - _item_start
                            print(f"  ✗ {_run_lbl} failed: {_e}")

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
                _run_factorial(ecosystem_for_loader, all_results, run_times, run_label)
                continue  # per-run summary handled inside; run_times already set
            else:
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
                print(f"✓ Data loaded for {run_label}")

                results = run_optimization_instance(
                    initial_conditions=initial_conditions,
                    scenario_params=custom_scenario_params,
                    pop_size=POP_SIZE,
                    n_generations=N_GENERATIONS,
                    save_results=True,
                    verbose=True,
                    n_jobs=N_JOBS,
                    random_seed=RANDOM_SEED,
                    use_repair=True,
                    use_patch_approach=USE_PATCH_APPROACH,
                    patch_size=PATCH_SIZE,
                    patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                    pixel_tolerance=PIXEL_TOLERANCE,
                    save_snapshots=SAVE_SNAPSHOTS,
                    n_partitions=N_PARTITIONS,
                    warm_seeding=WARM_SEEDING,
                    run_label=RUN_LABEL,
                    run_config=run_config,
                )

            if SCENARIO_MODE not in ("condition_grid", "policy_grid", "factorial"):
                if results is not None:
                    all_results[run_label] = results
                    run_times[run_label] = time.perf_counter() - _run_start
                    print(f"\n✓ {run_label.title()} optimisation completed successfully!")
                else:
                    run_times[run_label] = time.perf_counter() - _run_start
                    print(f"\n✗ {run_label.title()} optimisation failed.")
            else:
                run_times[run_label] = time.perf_counter() - _run_start

        except Exception as e:
            run_times[run_label] = time.perf_counter() - _run_start
            print(f"\n✗ Error optimising {run_label} ecosystem: {e}")
            continue

    print(f"\n{'='*80}")
    print("=== OPTIMISATION SUMMARY ===")
    print(f"{'='*80}")
    _total_elapsed = time.perf_counter() - _script_start

    if SCENARIO_MODE in ("condition_grid", "policy_grid", "factorial"):
        grid_succeeded = len(all_results)
        print(f"{SCENARIO_MODE}: {grid_succeeded} runs stored in all_results")
        print(f"\nTotal wall-clock time: {_total_elapsed/60:.1f} min ({_total_elapsed:.0f} s)")
    else:
        successful_runs = len(all_results)
        total_runs = len(runs)
        print(f"Successfully completed {successful_runs}/{total_runs} ecosystem optimisations:")
        for run_label, _ in runs:
            status = "✓ SUCCESS" if run_label in all_results else "✗ FAILED"
            elapsed = run_times.get(run_label)
            time_str = f"  ({elapsed/60:.1f} min)" if elapsed is not None else ""
            print(f"  {run_label.title():<12}: {status}{time_str}")

        print(f"\nTotal wall-clock time: {_total_elapsed/60:.1f} min ({_total_elapsed:.0f} s)")

        if successful_runs > 0:
            print(f"✓ Completed with outputs for {successful_runs} ecosystems.")
        else:
            print("✗ No optimisations completed successfully.")


# Mandatory on Windows: spawn re-imports this module in every worker process.
# Without this guard, that re-import would re-trigger the grid recursively.
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
        print(f"Profile saved → {_profile_path}")
    else:
        _run()
