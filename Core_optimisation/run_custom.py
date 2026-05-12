"""
Custom single-scenario and multi-ecosystem run script.

Edit the configuration variables below and run directly:
    python run_custom.py
"""
import cProfile
import pstats
import io
import time
from .resto_anom import run_optimization_instance, main
from .data_loader import load_initial_conditions
from .logger_setup import setup_logger

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
# Used in output filenames and the run_registry.jsonl log.
# Examples: "baseline", "patch_size2_highbudget", "testing_new_repair"
RUN_LABEL = "policy_seed_expansion"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "all"            runs scenario="all" using the scenario sampling logic inside main()
#   "condition_grid" sweeps all 13 condition scenarios x SEEDS
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
SCENARIO_MODE = "policy_grid"

# Condition scenario tag — selects pre-computed anomaly rasters from inputs/anomaly_scenarios/
# Available tags:
#   global_all | global_drop_smd | global_drop_sbd | global_drop_soc |
#   global_drop_uzl | global_drop_tsd | global_drop_can | global_drop_cdi |
#   global_drop_swf_h | global_drop_swf_t | global_drop_lai | global_drop_ndvi |
#   upper_q75_all
CONDITION_SCENARIO = "global_all"

# Seeds used by both condition_grid and policy_grid modes.
#   List[int] — runs each scenario/variant once per seed; labels: <name>_seed<n>
#   None       — runs each scenario/variant once using RANDOM_SEED; labels: <name>
SEEDS = [100, 101, 102, 103, 104] # None

print(f"\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
print(f"Scenario mode: {SCENARIO_MODE}")

log_path = setup_logger(log_dir="logs", run_label=f"{ECOSYSTEM_TO_RUN}_{REGION.lower()}")
if log_path:
    print(f"Verbose output → {log_path}")

OBJECTIVES = ["abiotic", "biotic", "cost"]
# Available objective names:
#   "abiotic"      – minimise abiotic condition anomaly (restoration pixels)
#   "biotic"       – minimise biotic condition anomaly (restoration pixels)
#   "cost"         – minimise implementation cost
#   "connectivity" – maximise connectivity gain (precomputed per pixel, conversion pixels)
#                    Replaces the older "landscape" objective as the default landscape metric.
#   "landscape"    – legacy landscape anomaly via SN-density recalculation (conversion pixels)
SAMPLE_FRACTION = None
SAMPLE_SEED = 42
POP_SIZE = 50
N_GENERATIONS = 100
N_JOBS = 12
RANDOM_SEED = 100 #42
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
}

# Named policy scenarios for SCENARIO_MODE == "policy_grid".
# Each entry is a dict of parameter overrides applied on top of custom_scenario_params.
# An empty dict {} means "no overrides" — mirrors the condition baseline run.
# Run labels will be the key name (e.g. "policy_ambitious"), with no seed suffix,
# using CONDITION_SCENARIO and RANDOM_SEED defined above.
POLICY_VARIANTS = {
    "policy_baseline":         {},                                                  # mirrors condition baseline
    "policy_ambitious":        {"max_restoration_fraction": 0.10},                 # double the budget
    "policy_very_ambitious":   {"max_restoration_fraction": 0.20},                 # 4x the budget
    "policy_burden_shared":    {"burden_sharing": "yes"},                          # equal burden across regions
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10,
                                "burden_sharing": "yes"},                          # ambitious + burden shared
}

# Benchmark condition scenarios run in policy_grid mode with baseline params × SEEDS.
# Labels: <tag>_seed<n>  (or <tag> when SEEDS is None).
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# Patch approach settings
USE_PATCH_APPROACH = True
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.15

# Spatial aggregation: block-coarsen all input rasters by this integer factor before optimisation.
# 2 = halve resolution in each dimension (~4x fewer pixels). Set to None or 1 to disable.
AGGREGATION_FACTOR = None

# Set to True to save per-generation population snapshots for animation
# Output: intermediate_results/X_history_{timestamp}.npz  shape=(n_gens, pop_size, n_var) int8
SAVE_SNAPSHOTS = False

# Set to True to profile the run with cProfile and print the top 30 hotspots afterwards.
# Results are also written to logs/profile_<run_label>.txt
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
                # Sweep all 13 condition scenario tags.
                # If SEEDS is a list, run each tag × seed.
                # If SEEDS is None, run each tag once with RANDOM_SEED.
                _condition_tags = [
                    "global_all",
                    "global_drop_smd", "global_drop_sbd", "global_drop_soc",
                    "global_drop_uzl", "global_drop_tsd", "global_drop_can",
                    "global_drop_cdi", "global_drop_swf_h", "global_drop_swf_t",
                    "global_drop_lai", "global_drop_ndvi",
                    "upper_q75_all",
                ]
                _use_seeds = SEEDS is not None
                _seeds     = SEEDS if _use_seeds else [RANDOM_SEED]
                results = None
                _grid_start = time.perf_counter()
                _grid_total = len(_condition_tags) * len(_seeds)
                _grid_done = 0
                _grid_times = {}
                for _tag in _condition_tags:
                    for _seed in _seeds:
                        _run_label = f"{_tag}_seed{_seed}" if _use_seeds else _tag
                        _item_start = time.perf_counter()
                        print(f"  condition_grid [{_grid_done + 1}/{_grid_total}]: {_run_label}")
                        try:
                            _ic = load_initial_conditions(
                                ".",
                                objectives=OBJECTIVES,
                                region=REGION,
                                ecosystem=ecosystem_for_loader,
                                sample_fraction=SAMPLE_FRACTION,
                                sample_seed=SAMPLE_SEED,
                                aggregation_factor=AGGREGATION_FACTOR,
                                condition_scenario=_tag,
                            )
                            _grid_config = {**run_config, "condition_scenario": _tag, "random_seed": _seed}
                            _grid_results = run_optimization_instance(
                                initial_conditions=_ic,
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
                                run_label=_run_label,
                                run_config=_grid_config,
                            )
                            _item_elapsed = time.perf_counter() - _item_start
                            _grid_times[_run_label] = _item_elapsed
                            if _grid_results is not None:
                                all_results[_run_label] = _grid_results
                                print(f"  ✓ {_run_label} completed ({_item_elapsed/60:.1f} min)")
                            else:
                                print(f"  ✗ {_run_label} returned no results ({_item_elapsed/60:.1f} min)")
                        except Exception as _e:
                            _grid_times[_run_label] = time.perf_counter() - _item_start
                            print(f"  ✗ {_run_label} failed: {_e}")
                        _grid_done += 1

                _grid_elapsed = time.perf_counter() - _grid_start
                _grid_succeeded = sum(1 for lbl in all_results)
                print(f"\n=== CONDITION GRID COMPLETE ===")
                print(f"  {_grid_succeeded}/{_grid_total} runs succeeded")
                print(f"  Total grid time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
                if _grid_times:
                    _avg = sum(_grid_times.values()) / len(_grid_times)
                    print(f"  Average per run: {_avg/60:.1f} min ({_avg:.0f} s)")
                print(f"  Results PKLs: results_files/res_<timestamp>_<run_label>.pkl (one per grid element)")
            elif SCENARIO_MODE == "policy_grid":
                # Run each named policy variant × SEEDS against CONDITION_SCENARIO.
                # Also run BENCHMARK_SCENARIOS × SEEDS with baseline params.
                # Labels: <variant>_seed<n>  /  <benchmark_tag>_seed<n>
                # (no seed suffix when SEEDS is None — uses RANDOM_SEED)
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

            if SCENARIO_MODE != "condition_grid":
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

    if SCENARIO_MODE in ("condition_grid", "policy_grid"):
        # Grid modes have their own per-run summary printed inline; just show wall time.
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
