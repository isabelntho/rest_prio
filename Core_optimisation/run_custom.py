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
from .paths import LOGS_DIR, R_INPUTS_DIR

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
# Used in output filenames and the run_registry.jsonl log.
# Examples: "baseline", "patch_size2_highbudget", "testing_new_repair"
RUN_LABEL = "refactor_test"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "all"            runs scenario="all" using the scenario sampling logic inside main()
#   "condition_grid" sweeps all 13 condition scenarios x SEEDS
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
#   "factorial"      fully-crossed design over FACTORIAL_FORMS x FACTORIAL_SCALINGS x
#                    FACTORIAL_CONSTRUCTIONS x FACTORIAL_POLICIES x SEEDS (iEMSs Block 4)
SCENARIO_MODE = "custom"

# Condition scenario tag — selects pre-computed anomaly rasters from data/anomaly_scenarios/
# Available tags:
#   global_all | global_drop_smd | global_drop_sbd | global_drop_soc |
#   global_drop_uzl | global_drop_tsd | global_drop_can | global_drop_cdi |
#   global_drop_swf_h | global_drop_swf_t | global_drop_lai | global_drop_ndvi |
#   upper_q75_all
CONDITION_SCENARIO = "global_all"

# Seeds used by both condition_grid and policy_grid modes.
#   List[int] — runs each scenario/variant once per seed; labels: <name>_seed<n>
#   None       — runs each scenario/variant once using RANDOM_SEED; labels: <name>
#SEEDS = [101, 102, 103, 104, 105] # 5 seed replicates (iEMSs run matrix, Block 0)
SEEDS = 101
print(f"\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
print(f"Scenario mode: {SCENARIO_MODE}")

log_path = setup_logger(log_dir=str(LOGS_DIR), run_label=f"{ECOSYSTEM_TO_RUN}_{REGION.lower()}")
if log_path:
    print(f"Verbose output → {log_path}")

OBJECTIVES = ["restoration_potential", "spatial_clustering", "cost"]  # iEMSs headline objectives
#OBJECTIVES = ["abiotic", "biotic", "cost"]
#["restoration_potential", "cost", "es_future_val", "es_future_robustness"] # test ES future value as an objective
# Available objective names:
#   "abiotic"               – minimise abiotic condition anomaly (restoration pixels)
#   "biotic"                – minimise biotic condition anomaly (restoration pixels)
#   "cost"                  – minimise implementation cost
#   "connectivity"          – maximise connectivity gain (precomputed per pixel, conversion pixels)
#                             Replaces the older "landscape" objective as the default landscape metric.
#   "landscape"             – legacy landscape anomaly via SN-density recalculation (conversion pixels)
#   "landscape_context"     – minimise mean abiotic anomaly of eligible neighbours within 500 m;
#                             rewards selecting pixels whose surroundings are already in good condition
#   "restoration_potential" – minimise mean of per-pixel abiotic + biotic baseline anomaly;
#                             single combined ecological condition score (lower = more degraded = higher potential)
#   "spatial_clustering"    – maximise spatial compactness of the selected pixels
#                             (number of orthogonal shared edges between selected pixels;
#                             configuration-dependent, computed each evaluation). Rewards
#                             clumped solutions. NB: distinct from the custom_scenario_params
#                             'spatial_clustering' knob, which only soft-biases sampling.
#   "es_future_val"         – maximise total ES performance of selected pixels under future scenarios
#                             (source: Mean_sum_of_change_ES.tif; higher sum = greater future ES gain)
#   "es_future_robustness"  – minimise total ES instability of selected pixels under future scenarios
#                             (source: Undesirable_deviation_sum_of_change_ES.tif; lower sum = more robust)
SAMPLE_FRACTION = None
SAMPLE_SEED = 42
POP_SIZE = 92 #previously 50?
N_GENERATIONS = 100
N_JOBS = 12
RANDOM_SEED = 101 #42
N_SAMPLES_PER_PARAM = 3
N_PARTITIONS = 12 #6 # for 4 objectives 6
WARM_SEEDING = True
#previously n partitions 12, pop size 50

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
    # Axis 2 — restoration_potential objective formulation:
    #   "sum"       = total improvement (default)
    #   "threshold" = area of restored pixels reaching 'good' condition (> rp_threshold)
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    # Metric for the "spatial_clustering" objective (only used when it is in OBJECTIVES):
    #   "adjacency"  = shared-edge count (compactness); strongly correlated with cost
    #   "components" = number of disconnected clusters (fragmentation); test for decoupling
    "clustering_metric": "adjacency",
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

# ── Factorial design (SCENARIO_MODE == "factorial") — iEMSs Block 4 ──────────
# Fully-crossed design over the decision-making + model formulation factors.
# Every cell is the product (form × scaling × construction × policy) × SEEDS.
#
#   FACTORIAL_FORMS         objective target form → scenario_params['rp_formulation']
#                           {"sum", "threshold"}  (Axis 2: what counts as success)
#   FACTORIAL_SCALINGS      condition reference benchmark → raster tag PREFIX.
#                           "global" = anomaly, "upper_q75" = q75 (Axis: scaling).
#   FACTORIAL_CONSTRUCTIONS condition-indicator construction → raster tag SUFFIX.
#                           "all" = full indicator set; "drop_<ind>" / reduced builds.
#                           PLACEHOLDER — fill once the 2–3 construction levels are
#                           defined (and the matching crossed rasters exist in
#                           data/anomaly_scenarios/, produced by ec_anomalies.r).
#   FACTORIAL_POLICIES      policy/governance lever → overrides on custom_scenario_params.
#
# scaling × construction together select the condition_scenario raster tag,
# built as f"{scaling}_{construction}" (e.g. "global_all", "upper_q75_drop_smd"),
# which must match the .tif files written by ec_anomalies.r and resolved in
# data_loader.load_initial_conditions.
#
# Each cell's factor levels are written to its run_config as flat factor_* keys
# (factor_form / factor_scaling / factor_construction / factor_policy) so they
# survive export_to_r → metadata.json and are recoverable for variance
# partitioning in R (load_run_factorial / compute_rfop_variance_partition).
FACTORIAL_FORMS = ["sum", "threshold"]
FACTORIAL_SCALINGS = ["global", "upper_q75"]
# Construction axis = 3 levels: "all" + the 2 HIGHEST-LEVERAGE agricultural drops.
# Which 2 comes from Block 2's R3c Jaccard, so RUN BLOCKS 1+2 FIRST, then fill the
# two drop_<ind> slots below. Design: 2 form × 2 scaling × 3 policy × 3 construction
# × 5 seeds = 180 runs. (Left at just "all" until filled, so a premature factorial
# run stays small rather than expanding to the full 9-level LOO.)
# Each tag needs rasters at BOTH scalings ("global_<c>" and "upper_q75_<c>"); all
# agricultural q75 drops are generated by ec_anomalies.r, so any 2 picks are ready.
#   Available drops: smd sbd soc uzl cdi swf_h swf_t ndvi
FACTORIAL_CONSTRUCTIONS = [
    "all",
    "drop_smd",   # ← fill from Block 2 R3c Jaccard
    #"drop_<highest_leverage_2>",   # ← fill from Block 2 R3c Jaccard
]
FACTORIAL_POLICIES = {
    "status_quo":    {},                                  # baseline budget, no burden sharing
    "ambitious":     {"max_restoration_fraction": 0.10},  # larger target area
    "burden_shared": {"burden_sharing": "yes"},           # equal burden across regions
}

# Patch approach settings
USE_PATCH_APPROACH = True
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.05#.15

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
                # Agricultural-focus LOO set (iEMSs Blocks 1+2). LOO is restricted to
                # indicators present in the agricultural EC set (setup.r): smd/sbd/soc
                # (abiotic) + uzl/cdi/swf_h/swf_t/ndvi (biotic). tsd/can/lai are
                # forest-only — dropping them is a no-op over agricultural pixels — so
                # they are excluded. upper_q75_all is the q75 benchmark for R3.
                _condition_tags = [
                    "global_all",
                    "global_drop_smd", "global_drop_sbd", "global_drop_soc",
                    "global_drop_uzl", "global_drop_cdi", "global_drop_swf_h",
                    "global_drop_swf_t", "global_drop_ndvi",
                    "upper_q75_all",
                ]
                _use_seeds = SEEDS is not None
                _seeds     = SEEDS if _use_seeds else [RANDOM_SEED]
                results = None
                _grid_start = time.perf_counter()
                _grid_total = len(_condition_tags) * len(_seeds)
                _grid_done = 0
                _grid_times = {}
                # One parent dir for the whole grid: r_inputs/{timestamp}_{RUN_LABEL}/
                import os as _os
                from datetime import datetime as _dt
                _grid_ts = _dt.now().strftime('%Y%m%d_%H%M')
                _grid_r_parent = _os.path.join(str(R_INPUTS_DIR), f"{_grid_ts}_{RUN_LABEL}")
                _os.makedirs(_grid_r_parent, exist_ok=True)
                print(f"  Grid R export parent: {_grid_r_parent}/")
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
                                r_export_parent=_grid_r_parent,
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
                _grid_times = {}                # One parent dir for the whole grid: r_inputs/{timestamp}_{RUN_LABEL}/
                import os as _os
                from datetime import datetime as _dt
                _grid_ts = _dt.now().strftime('%Y%m%d_%H%M')
                _pg_r_parent = _os.path.join(str(R_INPUTS_DIR), f"{_grid_ts}_{RUN_LABEL}")
                _os.makedirs(_pg_r_parent, exist_ok=True)
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
                # Fully-crossed design (iEMSs Block 4):
                #   form × scaling × construction × policy × SEEDS
                # scaling × construction select the condition raster tag
                # ("{scaling}_{construction}"); form + policy are scenario_params
                # overrides. initial_conditions are cached per unique tag because
                # loading rasters dominates runtime and patch geometry is tag-only.
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
                      f"({len(FACTORIAL_FORMS)} form × {len(FACTORIAL_SCALINGS)} scaling × "
                      f"{len(FACTORIAL_CONSTRUCTIONS)} construction × "
                      f"{len(FACTORIAL_POLICIES)} policy × {len(_seeds)} seed)")
                print(f"  Grid R export parent: {_grid_r_parent}/")

                _ic_cache = {}  # condition_scenario tag → initial_conditions (or None if missing)
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
                                    print(f"  ✗ could not load condition rasters for tag '{_tag}': {_e}")
                                    _ic_cache[_tag] = None
                            _ic = _ic_cache[_tag]
                            if _ic is None:
                                _skipped = len(FACTORIAL_FORMS) * len(FACTORIAL_POLICIES)
                                _grid_done += _skipped
                                print(f"  ✗ skipping {_skipped} cells needing missing tag '{_tag}'")
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
                                            n_partitions=N_PARTITIONS,
                                            warm_seeding=WARM_SEEDING,
                                            run_label=_run_lbl,
                                            run_config=_grid_config,
                                            r_export_parent=_grid_r_parent,
                                        )
                                        _item_elapsed = time.perf_counter() - _item_start
                                        _grid_times[_run_lbl] = _item_elapsed
                                        if _fresults is not None:
                                            all_results[_run_lbl] = _fresults
                                            print(f"  ✓ {_run_lbl} completed ({_item_elapsed/60:.1f} min)")
                                        else:
                                            print(f"  ✗ {_run_lbl} returned no results ({_item_elapsed/60:.1f} min)")
                                    except Exception as _e:
                                        _grid_times[_run_lbl] = time.perf_counter() - _item_start
                                        print(f"  ✗ {_run_lbl} failed: {_e}")

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
