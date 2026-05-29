"""
Planning Unit Optimization — Standalone Run Script
==================================================

Test the planning unit approach where the decision variable is a single binary
per coarse spatial unit (a regular grid cell or an administrative boundary)
rather than per pixel or per patch.

UNIT TYPES
----------
  'grid'  : Regular N×N pixel grid cells.
             UNIT_SIZE_PX = 20  →  20×20 px = 2 km at 100 m resolution
             UNIT_SIZE_PX = 50  →  50×50 px = 5 km at 100 m resolution
  'admin' : Administrative boundaries (municipalities / districts / cantons).
            Requires access to the Swiss boundaries shapefiles on the Y: drive.

HOW TO RUN
----------
    cd c:/Users/inicholson/Documents/rest_prio
    pixi run python -m Core_optimisation.run_planning_unit_optimization
or with pixi environment active:
    python -m Core_optimisation.run_planning_unit_optimization

RESULTS
-------
Results are saved to results_files/ in the same PKL format as the patch approach
and are directly compatible with run_vis.py and the existing analysis notebooks.

Each run also records a row in run_registry.jsonl.

Created: May 2026
"""

import time as _time
import sys
import os

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# --- Unit type ---
# 'grid'  : regular pixel grid cells
# 'admin' : administrative boundaries
UNIT_TYPE = "grid"

# Grid mode: cell side length in pixels.
# At 100 m resolution: 20 px = 2 km,  50 px = 5 km,  100 px = 10 km
# Ignored when UNIT_TYPE == 'admin'.
UNIT_SIZE_PX = 20

# Admin mode: administrative level.
# 'Bern' → municipalities within Bern canton
# 'CH'   → Swiss cantons
# Ignored when UNIT_TYPE == 'grid'.
ADMIN_REGION = "Bern"

# Admin mode: cap on eligible pixels per unit.
# Units with more eligible pixels are split into two halves.
# None = no cap.
MAX_UNIT_PIXELS = None

# --- Spatial extent ---
REGION    = "Bern"
ECOSYSTEM = "combined"   # 'forest','agricultural','grassland','fg','all','combined'

# --- Objectives ---
OBJECTIVES = ["abiotic", "biotic", "cost"]
#OBJECTIVES = ["restoration_potential", "cost", "landscape_context"] # "landscape" is legacy, replaced by "connectivity"

# --- Scenario parameters (same format as run_custom.py) ---
SCENARIO_PARAMS = {
    "max_restoration_fraction" : 0.05,
    "spatial_clustering"       : 0,
    "biotic_effect"            : 0.01,
    "abiotic_effect"           : 0.01,
    "normalize_objectives"     : True,
    "patch_score_temperature"  : 2,
    "patch_repair_top_k"       : 100,
    "burden_sharing"           : "no",
}

# --- Budget tolerance (±fraction of target pixels) ---
# Wider than patches (±10–15 %) to accommodate unit-size variability
PIXEL_TOLERANCE = 0.20

# --- Algorithm settings ---
N_GENERATIONS = 100
N_PARTITIONS  = 12      # NSGA-III reference directions; 4 obj + 6 partitions → 84 dirs
POP_SIZE      = 92      # population size (must be ≥ n_partitions × 3 for NSGA-III)
HV_PATIENCE   = 15     # early stopping patience
HV_MIN_IMPROVEMENT = 1e-6
N_JOBS        = 8      # parallel objective evaluations (None = all cores)
WARM_SEEDING  = True

# --- Seeds ---
# List → one run per seed.  Integer → single run.  None → single run, random seed.
SEEDS = [100, 101, 102, 103, 104, 105, 106, 108, 109]

# --- Optional spatial aggregation (speeds up for large extents) ---
# 2 = halve resolution in each dimension (~4× fewer pixels), None = disabled
AGGREGATION_FACTOR = None
SAMPLE_FRACTION    = None
SAMPLE_SEED        = 42

# --- Condition scenario ---
CONDITION_SCENARIO = "global_all"

# --- Scenario mode ---
# 'custom'         : run once (or once per seed) using SCENARIO_PARAMS and CONDITION_SCENARIO
# 'condition_grid' : sweep all 13 condition scenario tags × SEEDS
# 'policy_grid'    : sweep each POLICY_VARIANTS entry × SEEDS, plus BENCHMARK_SCENARIOS × SEEDS
SCENARIO_MODE = "policy_grid"

# Named policy variants for SCENARIO_MODE == 'policy_grid'.
# Each entry is a dict of parameter overrides applied on top of SCENARIO_PARAMS.
# An empty dict {} means "no overrides" — mirrors the condition baseline.
POLICY_VARIANTS = {
    "policy_baseline"        : {},
    "policy_ambitious"       : {"max_restoration_fraction": 0.10},
    "policy_very_ambitious"  : {"max_restoration_fraction": 0.20},
    "policy_burden_shared"   : {"burden_sharing": "yes"},
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10, "burden_sharing": "yes"},
}

# Benchmark condition scenarios run in policy_grid mode with baseline params × SEEDS.
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# --- Output ---
SAVE_RESULTS = True
OUTPUT_DIR   = "."      # PKL files go to results_files/ inside this directory
VERBOSE      = True

# ---------------------------------------------------------------------------
# RUN LABEL
# ---------------------------------------------------------------------------
_unit_tag = (
    f"grid{UNIT_SIZE_PX}px"
    if UNIT_TYPE == "grid"
    else f"admin_{ADMIN_REGION.lower()}"
)
BASE_LABEL = f"pu_{_unit_tag}"

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _run():
    from Core_optimisation.data_loader import load_initial_conditions
    from Core_optimisation.planning_unit_approach import run_planning_unit_instance
    from Core_optimisation.logger_setup import setup_logger
    import traceback

    setup_logger(log_dir="logs", run_label=f"{ECOSYSTEM}_{REGION.lower()}_pu")

    workspace_dir = "."
    ecosystem_for_loader = ECOSYSTEM if ECOSYSTEM != "combined" else "all"
    seeds = SEEDS if isinstance(SEEDS, list) else ([SEEDS] if SEEDS is not None else [None])
    all_results = {}
    _script_start = _time.perf_counter()

    # ------------------------------------------------------------------
    # Helper: run one planning-unit instance and record outcome
    # ------------------------------------------------------------------
    def _run_one(label, ic, scenario_params, seed, extra_cfg=None, r_export_parent=None):
        run_cfg = {
            "unit_type"          : UNIT_TYPE,
            "unit_size_px"       : UNIT_SIZE_PX if UNIT_TYPE == "grid" else None,
            "admin_region"       : ADMIN_REGION if UNIT_TYPE == "admin" else None,
            "max_unit_pixels"    : MAX_UNIT_PIXELS,
            "region"             : REGION,
            "ecosystem"          : ECOSYSTEM,
            "objectives"         : OBJECTIVES,
            "scenario_params"    : scenario_params,
            "pixel_tolerance"    : PIXEL_TOLERANCE,
            "n_generations"      : N_GENERATIONS,
            "n_partitions"       : N_PARTITIONS,
            "pop_size"           : POP_SIZE,
            "hv_patience"        : HV_PATIENCE,
            "n_jobs"             : N_JOBS,
            "warm_seeding"       : WARM_SEEDING,
            "random_seed"        : seed,
            "aggregation_factor" : AGGREGATION_FACTOR,
            "condition_scenario" : CONDITION_SCENARIO,
        }
        if extra_cfg:
            run_cfg.update(extra_cfg)
        _t0 = _time.perf_counter()
        try:
            results = run_planning_unit_instance(
                initial_conditions=ic,
                scenario_params=scenario_params,
                unit_type=UNIT_TYPE,
                unit_size_px=UNIT_SIZE_PX,
                workspace_dir=workspace_dir if UNIT_TYPE == "admin" else None,
                region=ADMIN_REGION,
                max_unit_pixels=MAX_UNIT_PIXELS,
                pixel_tolerance=PIXEL_TOLERANCE,
                n_generations=N_GENERATIONS,
                n_partitions=N_PARTITIONS,
                pop_size=POP_SIZE,
                hv_patience=HV_PATIENCE,
                hv_min_improvement=HV_MIN_IMPROVEMENT,
                n_jobs=N_JOBS,
                random_seed=seed,
                warm_seeding=WARM_SEEDING,
                save_results=SAVE_RESULTS,
                output_dir=OUTPUT_DIR,
                run_label=label,
                run_config=run_cfg,
                verbose=VERBOSE,
                r_export_parent=r_export_parent,
            )
            elapsed = _time.perf_counter() - _t0
            if results is not None:
                all_results[label] = results
                nd = results['n_nondominated_solutions']
                hv = results['algorithm_info'].get('final_hypervolume') or 0
                print(f"  ✓ {label}: {nd} Pareto solutions, HV={hv:.5f}, {elapsed/60:.1f} min")
            else:
                print(f"  ✗ {label}: no results returned ({elapsed/60:.1f} min)")
        except Exception as exc:
            elapsed = _time.perf_counter() - _t0
            print(f"  ✗ {label}: FAILED after {elapsed/60:.1f} min")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # SCENARIO_MODE: custom
    # ------------------------------------------------------------------
    if SCENARIO_MODE == "custom":
        for seed in seeds:
            label = f"{BASE_LABEL}_seed{seed}" if seed is not None else BASE_LABEL
            print(f"\n{'='*60}")
            print(f"  Run: {label}")
            print(f"{'='*60}")
            ic = load_initial_conditions(
                workspace_dir,
                objectives=OBJECTIVES,
                region=REGION,
                ecosystem=ecosystem_for_loader,
                sample_fraction=SAMPLE_FRACTION,
                sample_seed=SAMPLE_SEED,
                aggregation_factor=AGGREGATION_FACTOR,
                condition_scenario=CONDITION_SCENARIO,
            )
            _run_one(label, ic, SCENARIO_PARAMS, seed)

        total = _time.perf_counter() - _script_start
        print(f"\n{'='*60}")
        print(f"All runs complete: {len(all_results)}/{len(seeds)} succeeded")
        print(f"Results saved to: results_files/")
        print(f"Total time: {total/60:.1f} min")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # SCENARIO_MODE: condition_grid
    # ------------------------------------------------------------------
    elif SCENARIO_MODE == "condition_grid":
        _condition_tags = [
            "global_all",
            "global_drop_smd", "global_drop_sbd", "global_drop_soc",
            "global_drop_uzl", "global_drop_tsd", "global_drop_can",
            "global_drop_cdi", "global_drop_swf_h", "global_drop_swf_t",
            "global_drop_lai", "global_drop_ndvi",
            "upper_q75_all",
        ]
        _use_seeds = seeds != [None]
        _grid_total = len(_condition_tags) * len(seeds)
        _grid_done = 0
        print(f"\n  condition_grid: {len(_condition_tags)} tags × {len(seeds)} seeds = {_grid_total} runs")
        _grid_start = _time.perf_counter()

        from datetime import datetime as _dt
        _grid_ts = _dt.now().strftime('%Y%m%d_%H%M')
        _grid_r_parent = os.path.join(OUTPUT_DIR, "r_inputs", f"{_grid_ts}_{BASE_LABEL}")
        os.makedirs(_grid_r_parent, exist_ok=True)
        print(f"  Grid R export parent: {_grid_r_parent}/")

        for _tag in _condition_tags:
            _ic = load_initial_conditions(
                workspace_dir,
                objectives=OBJECTIVES,
                region=REGION,
                ecosystem=ecosystem_for_loader,
                sample_fraction=SAMPLE_FRACTION,
                sample_seed=SAMPLE_SEED,
                aggregation_factor=AGGREGATION_FACTOR,
                condition_scenario=_tag,
            )
            for _seed in seeds:
                _run_lbl = f"{_tag}_seed{_seed}" if _use_seeds else _tag
                _grid_done += 1
                print(f"  condition_grid [{_grid_done}/{_grid_total}]: {_run_lbl}")
                _run_one(_run_lbl, _ic, SCENARIO_PARAMS, _seed,
                         extra_cfg={"condition_scenario": _tag, "random_seed": _seed},
                         r_export_parent=_grid_r_parent)

        _grid_elapsed = _time.perf_counter() - _grid_start
        print(f"\n=== CONDITION GRID COMPLETE ===")
        print(f"  {len(all_results)}/{_grid_total} runs succeeded")
        print(f"  Total time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
        if all_results:
            print(f"  Average per run: {_grid_elapsed/len(all_results)/60:.1f} min")
        print(f"  Results PKLs: results_files/res_<timestamp>_<run_label>.pkl")

    # ------------------------------------------------------------------
    # SCENARIO_MODE: policy_grid
    # ------------------------------------------------------------------
    elif SCENARIO_MODE == "policy_grid":
        _use_seeds = seeds != [None]
        _pg_total = len(POLICY_VARIANTS) * len(seeds) + len(BENCHMARK_SCENARIOS) * len(seeds)
        _grid_done = 0
        _grid_start = _time.perf_counter()

        print(f"\n  Policy variants ({len(POLICY_VARIANTS)} × {len(seeds)} seeds):")
        for _vname, _vparams in POLICY_VARIANTS.items():
            _diff = {k: v for k, v in _vparams.items()} or {"(baseline — no overrides)": ""}
            print(f"    {_vname}: {_diff}")
        if BENCHMARK_SCENARIOS:
            print(f"  Benchmark scenarios ({len(BENCHMARK_SCENARIOS)} × {len(seeds)} seeds):")
            for _btag in BENCHMARK_SCENARIOS:
                print(f"    {_btag}")

        from datetime import datetime as _dt
        _grid_ts = _dt.now().strftime('%Y%m%d_%H%M')
        _pg_r_parent = os.path.join(OUTPUT_DIR, "r_inputs", f"{_grid_ts}_{BASE_LABEL}")
        os.makedirs(_pg_r_parent, exist_ok=True)
        print(f"  Grid R export parent: {_pg_r_parent}/")

        # Load IC for policy variants (shared condition scenario)
        _ic_policy = load_initial_conditions(
            workspace_dir,
            objectives=OBJECTIVES,
            region=REGION,
            ecosystem=ecosystem_for_loader,
            sample_fraction=SAMPLE_FRACTION,
            sample_seed=SAMPLE_SEED,
            aggregation_factor=AGGREGATION_FACTOR,
            condition_scenario=CONDITION_SCENARIO,
        )

        for _vname, _vparams in POLICY_VARIANTS.items():
            _merged = {**SCENARIO_PARAMS, **_vparams}
            for _seed in seeds:
                _run_lbl = f"{_vname}_seed{_seed}" if _use_seeds else _vname
                _grid_done += 1
                print(f"  policy_grid [{_grid_done}/{_pg_total}]: {_run_lbl}")
                _run_one(_run_lbl, _ic_policy, _merged, _seed,
                         extra_cfg={"policy_variant": _vname,
                                    "condition_scenario": CONDITION_SCENARIO,
                                    "random_seed": _seed},
                         r_export_parent=_pg_r_parent)

        # Benchmark scenarios (baseline params, varied condition)
        for _btag in BENCHMARK_SCENARIOS:
            _ic_bench = load_initial_conditions(
                workspace_dir,
                objectives=OBJECTIVES,
                region=REGION,
                ecosystem=ecosystem_for_loader,
                sample_fraction=SAMPLE_FRACTION,
                sample_seed=SAMPLE_SEED,
                aggregation_factor=AGGREGATION_FACTOR,
                condition_scenario=_btag,
            )
            for _seed in seeds:
                _run_lbl = f"{_btag}_seed{_seed}" if _use_seeds else _btag
                _grid_done += 1
                print(f"  policy_grid [{_grid_done}/{_pg_total}]: {_run_lbl} (benchmark)")
                _run_one(_run_lbl, _ic_bench, SCENARIO_PARAMS, _seed,
                         extra_cfg={"benchmark_scenario": _btag,
                                    "condition_scenario": _btag,
                                    "random_seed": _seed},
                         r_export_parent=_pg_r_parent)

        _grid_elapsed = _time.perf_counter() - _grid_start
        print(f"\n=== POLICY GRID COMPLETE ===")
        print(f"  {len(all_results)}/{_pg_total} runs succeeded")
        print(f"  Total time: {_grid_elapsed/60:.1f} min ({_grid_elapsed:.0f} s)")
        if all_results:
            print(f"  Average per run: {_grid_elapsed/len(all_results)/60:.1f} min")
        print(f"  Results PKLs: results_files/res_<timestamp>_<label>.pkl")

    else:
        raise ValueError(f"Unknown SCENARIO_MODE: {SCENARIO_MODE!r}. "
                         "Choose 'custom', 'condition_grid', or 'policy_grid'.")

    return all_results


if __name__ == "__main__":
    _run()
