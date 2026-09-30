"""Single optimisation driver, configured by a preset from Core_optimisation.configs.

Replaces run_custom.py / run_custom_parallel.py / run_custom_nsga2.py, which were
near-clones of one another: the same _run() skeleton, the same SCENARIO_MODE
ladder, the same load_initial_conditions and run_optimization_instance call
sites, differing only in a block of module-level constants. That block now lives
in configs/<preset>.py and the shared machinery lives here.

Run from the project root as a module so the package-relative imports and the
'spawn' worker re-import resolve:

    python -m Core_optimisation.run nsga2_2obj
    python -m Core_optimisation.run --list

Scenario modes: custom | condition_grid | policy_grid | factorial. The legacy
"all" mode (optimization_engine.main(scenario="all")) is deliberately NOT supported here:
it suppresses per-run saving and never passes r_export_parent, so it writes
nothing to outputs/r_inputs/ - the only tree produce_figures.R and the paper2
qmds read - and its combined-pickle format has readers only in
Documentation/archive/. Use policy_grid to sweep scenario parameters instead.

NOTE: the module-level execution is guarded by ``if __name__ == "__main__"``.
This is mandatory on Windows: ``spawn`` re-imports this module in every worker,
and without the guard that would re-trigger the grid recursively.
"""
import argparse
import cProfile
import importlib
import io
import os
import pstats
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from .configs._defaults import DEFAULTS
from .data_loader import load_initial_conditions
from .grid_parallel import run_custom_seed, run_factorial_cell, run_tag
from .logger_setup import setup_logger
from .paths import DATA_DIR, LOGS_DIR, R_INPUTS_DIR
from .optimization_engine import run_optimization_instance

GRID_MODES = ("condition_grid", "policy_grid", "factorial")


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

def _load_preset(name):
    """Overlay a preset's module-level constants onto DEFAULTS.

    Raises on any public name the contract does not define, so a typo'd constant
    fails loudly instead of silently doing nothing (the old scripts' behaviour).
    """
    try:
        mod = importlib.import_module(f".configs.{name}", package=__package__)
    except ImportError as e:
        raise SystemExit(f"Unknown config preset {name!r}: {e}\n"
                         f"Available: {', '.join(_available_presets())}")

    cfg = dict(DEFAULTS)
    unknown = []
    for key, value in vars(mod).items():
        if key.startswith("__") or key == "_condition_tags" and value is None:
            continue
        if key in DEFAULTS:
            cfg[key] = value
        elif not key.startswith("_") or key == "_condition_tags":
            unknown.append(key)

    if unknown:
        raise SystemExit(
            f"Config {name!r} sets name(s) not in the contract: {', '.join(sorted(unknown))}\n"
            f"Add them to configs/_defaults.py, or fix the spelling.")

    cfg["_preset"] = name
    return cfg


def _available_presets():
    here = os.path.join(os.path.dirname(__file__), "configs")
    return sorted(f[:-3] for f in os.listdir(here)
                  if f.endswith(".py") and not f.startswith("_"))


def _seed_list(cfg):
    """SEEDS as a list. Accepts a bare int (run_custom.py wrote SEEDS = 101)."""
    seeds = cfg["SEEDS"]
    if seeds is None:
        return None
    return [seeds] if isinstance(seeds, int) else list(seeds)


def _multiseed(cfg):
    return cfg["SCENARIO_MODE"] == "custom" and cfg["CUSTOM_MULTISEED"] \
        and _seed_list(cfg) is not None


def _build_run_config(cfg):
    """Provenance snapshot written to run_registry.jsonl alongside results."""
    return {
        "ecosystem": cfg["ECOSYSTEM_TO_RUN"],
        "region": cfg["REGION"],
        "scenario_mode": cfg["SCENARIO_MODE"],
        "algorithm_type": cfg["ALGORITHM"],
        "objectives": cfg["OBJECTIVES"],
        "sample_fraction": cfg["SAMPLE_FRACTION"],
        "sample_seed": cfg["SAMPLE_SEED"],
        "pop_size": cfg["POP_SIZE"],
        "n_generations": cfg["N_GENERATIONS"],
        "random_seed": cfg["RANDOM_SEED"],
        "n_samples_per_param": cfg["N_SAMPLES_PER_PARAM"],
        "use_patch_approach": cfg["USE_PATCH_APPROACH"],
        "patch_size": cfg["PATCH_SIZE"],
        "patch_constraint_type": cfg["PATCH_CONSTRAINT_TYPE"],
        "pixel_tolerance": cfg["PIXEL_TOLERANCE"],
        "save_snapshots": cfg["SAVE_SNAPSHOTS"],
        "snapshot_generations": cfg["SNAPSHOT_GENERATIONS"],
        "aggregation_factor": cfg["AGGREGATION_FACTOR"],
        "custom_scenario_params": cfg["custom_scenario_params"],
        "seeds": cfg["SEEDS"],
        "benchmark_scenarios": cfg["BENCHMARK_SCENARIOS"],
        "factorial_forms": cfg["FACTORIAL_FORMS"],
        "factorial_scalings": cfg["FACTORIAL_SCALINGS"],
        "factorial_constructions": cfg["FACTORIAL_CONSTRUCTIONS"],
        "factorial_policies": list(cfg["FACTORIAL_POLICIES"].keys()),
        "grid_workers": cfg["GRID_WORKERS"],
    }


def _grid_cfg(cfg, ecosystem_for_loader, r_parent):
    """Run settings shared by every grid task (condition_grid and factorial).

    Must be picklable (plain dicts/lists/scalars) - it crosses the process
    boundary to the workers. The heavy initial_conditions are loaded INSIDE each
    worker from the picklable condition tag, so they never cross that boundary.
    """
    out = {
        "objectives": cfg["OBJECTIVES"],
        "region": cfg["REGION"],
        "ecosystem": ecosystem_for_loader,
        "sample_fraction": cfg["SAMPLE_FRACTION"],
        "sample_seed": cfg["SAMPLE_SEED"],
        "aggregation_factor": cfg["AGGREGATION_FACTOR"],
        "scenario_params": cfg["custom_scenario_params"],
        "pop_size": cfg["POP_SIZE"],
        "n_generations": cfg["N_GENERATIONS"],
        "use_patch_approach": cfg["USE_PATCH_APPROACH"],
        "patch_size": cfg["PATCH_SIZE"],
        "patch_constraint_type": cfg["PATCH_CONSTRAINT_TYPE"],
        "pixel_tolerance": cfg["PIXEL_TOLERANCE"],
        "save_snapshots": cfg["SAVE_SNAPSHOTS"],
        "snapshot_generations": cfg["SNAPSHOT_GENERATIONS"],
        "n_partitions": cfg["N_PARTITIONS"],
        "warm_seeding": cfg["WARM_SEEDING"],
        "run_config": cfg["_run_config"],
        "r_parent": r_parent,
        "algorithm_type": cfg["ALGORITHM"],
        "mutation_flip_count": cfg["MUTATION_FLIP_COUNT"],
        "capture_repair_diag": cfg["CAPTURE_REPAIR_DIAG"],
        "n_capture_gens": cfg["N_CAPTURE_GENS"],
        # Only echo per-run verbose output for a single worker, else parallel
        # stdout interleaves into noise.
        "verbose": cfg["GRID_WORKERS"] == 1,
    }
    # Omitted when None so run_optimization_instance's own default (15, HV early
    # stopping ON) applies - the distinction the nsga3 presets rely on.
    if cfg["HV_PATIENCE"] is not None:
        out["hv_patience"] = cfg["HV_PATIENCE"]
    return out


def _hv_kw(cfg):
    """Splatted into direct run_optimization_instance calls; empty = engine default."""
    return {} if cfg["HV_PATIENCE"] is None else {"hv_patience": cfg["HV_PATIENCE"]}


def _resolve_runs(cfg):
    """(run_label, ecosystem_for_loader) pairs implied by ECOSYSTEM_TO_RUN."""
    eco = cfg["ECOSYSTEM_TO_RUN"]
    if eco == "all":
        return [("forest", "forest"), ("agricultural", "agricultural"),
                ("grassland", "grassland")]
    if eco == "combined":
        return [("combined", "all")]
    return [(eco, eco)]


def _resolve_condition_tags(cfg):
    """Condition tags for condition_grid mode.

    An explicit _condition_tags list in the preset wins. Otherwise the tags are
    derived from CONDITION_GRID_FAMILY.
    """
    if cfg["_condition_tags"] is not None:
        tags = list(cfg["_condition_tags"])
    else:
        # -- Indicator-weighting vertex set (see data/ec_anomalies.r) --
        # These tags vary HOW MUCH each indicator counts, holding the indicator
        # set fixed - the complement of the drop_* (leave-one-out) tags, which
        # vary WHICH indicators enter. Each weighting tag's abiotic_ and biotic_
        # rasters are the SAME all-indicator weighted composite, so the
        # objective's 1:1 abiotic+biotic sum expresses the weight vector
        # directly (2 * effect * w(C)) with no change to the objective code.
        #   global_all    - untouched status-quo anchor (equal-weight category means)
        #   global_w_flat - every indicator 1/n
        #   global_w_cat  - abiotic block 0.5, biotic block 0.5
        #   global_w_<k>  - k up-weighted to 2/n, the rest shrunk proportionally
        weight_focal = [
            "smd", "sbd", "soc",                                          # abiotic
            "uzl", "tsd", "can", "cdi", "swf_h", "swf_t", "lai", "ndvi",  # biotic
        ]
        weighting_tags = ["global_w_flat", "global_w_cat"] + \
            [f"global_w_{v}" for v in weight_focal]

        # Extended weighting campaign (the simplex sample): explicit Dirichlet
        # weight vectors at fixed L1 distances from flat, because the vertices
        # above only reach 0.150 and the screen's dose-response keeps declining
        # well past that. w_flat is re-run here rather than borrowed from the
        # vertex grid: uncertainty_analysis's `noise` stage derives each
        # campaign's seed floor from that campaign's OWN reference cell, so it
        # has to be present in this grid's r_inputs folder.
        simplex_tags = ["global_w_flat"] + [
            f"global_w_d{b}{d}"
            for b in cfg["SIMPLEX_BANDS"] for d in cfg["SIMPLEX_DRAWS"]
        ]

        family = cfg["CONDITION_GRID_FAMILY"]
        if family == "weighting_simplex":
            tags = simplex_tags
        elif family == "weighting_vertex":
            tags = ["global_all"] + weighting_tags
        else:
            raise ValueError(
                f"CONDITION_GRID_FAMILY={family!r} unknown - use "
                "'weighting_vertex' or 'weighting_simplex', or set an explicit "
                "_condition_tags list in the preset.")

    # Fail now, not an hour in: a tag whose rasters were never generated would
    # otherwise surface as a per-worker load error partway through the grid.
    anom_dir = DATA_DIR / ("CH_wide" if cfg["REGION"] == "CH" else "anomaly_scenarios")
    missing = [t for t in tags
               if not all((anom_dir / f"{c}_{t}.tif").exists()
                          for c in ("abiotic", "biotic"))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} condition tag(s) have no rasters in {anom_dir}: "
            f"{', '.join(missing)}\n"
            "For the simplex family, generate them by setting "
            "GENERATE_SIMPLEX_WEIGHT_SCENARIOS <- TRUE in data/ec_anomalies.r "
            "(and GENERATE_WEIGHT_SCENARIOS <- FALSE) and running it once.")
    return tags


def _grid_parent(cfg):
    """One parent dir for a whole grid: r_inputs/{timestamp}_{RUN_LABEL}/."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parent = os.path.join(str(R_INPUTS_DIR), f"{stamp}_{cfg['RUN_LABEL']}")
    os.makedirs(parent, exist_ok=True)
    print(f"  Grid R export parent: {parent}/")
    return parent


def _dispatch(cfg, worker, tasks, key_of, all_results, times):
    """Submit tasks to a process pool, or run them in-process when GRID_WORKERS <= 1.

    Every worker returns ``(run_label, ok, elapsed, error)`` tuples - run_tag and
    run_factorial_cell a list of them, run_custom_seed a single one.
    """
    def record(summaries):
        if summaries and not isinstance(summaries, list):
            summaries = [summaries]
        for run_lbl, ok, elapsed, err in summaries:
            times[run_lbl] = elapsed
            if ok:
                # Full results are on disk; store a marker so all_results counts succeed.
                all_results[run_lbl] = True
                print(f"  [ok] {run_lbl} completed ({elapsed/60:.1f} min)")
            else:
                msg = f": {err}" if err else ""
                print(f"  [x] {run_lbl} failed{msg} ({elapsed/60:.1f} min)")

    if cfg["GRID_WORKERS"] <= 1:
        for task in tasks:
            print(f"  [task] {key_of(task)} (sequential)")
            record(worker(task))
        return

    with ProcessPoolExecutor(max_workers=cfg["GRID_WORKERS"]) as ex:
        futures = {ex.submit(worker, task): key_of(task) for task in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                record(fut.result())
            except Exception as e:  # a worker crashed entirely
                print(f"  [x] task {key} worker crashed: {e}")


def _grid_summary(title, succeeded, total, elapsed, times):
    print(f"\n=== {title} COMPLETE ===")
    print(f"  {succeeded}/{total} runs succeeded")
    print(f"  Total grid time: {elapsed/60:.1f} min ({elapsed:.0f} s)")
    if times:
        avg = sum(times.values()) / len(times)
        print(f"  Average per run: {avg/60:.1f} min ({avg:.0f} s)")
    print("  Results PKLs: results_files/res_<timestamp>_<run_label>.pkl")


# ---------------------------------------------------------------------------
# Scenario modes
# ---------------------------------------------------------------------------

def _run_condition_grid(cfg, ecosystem_for_loader, all_results, run_times, run_label):
    """Condition grid: one task per tag, every seed of that tag in one worker.

    Grouping by tag means each worker loads that tag's rasters once and reuses
    them across seeds (see grid_parallel.run_tag).
    """
    tags = _resolve_condition_tags(cfg)
    if cfg["_condition_tags"] is None:
        print(f"  Condition grid family: {cfg['CONDITION_GRID_FAMILY']} ({len(tags)} tags)")

    seeds = _seed_list(cfg)
    use_seeds = seeds is not None
    seeds = seeds if use_seeds else [cfg["RANDOM_SEED"]]
    total = len(tags) * len(seeds)
    times = {}

    parent = _grid_parent(cfg)
    print(f"  Condition grid: {total} runs ({len(tags)} tags x {len(seeds)} seeds), "
          f"GRID_WORKERS={cfg['GRID_WORKERS']}")

    grid_cfg = _grid_cfg(cfg, ecosystem_for_loader, parent)
    tasks = [{"tag": t, "seeds": seeds, "use_seeds": use_seeds, "cfg": grid_cfg}
             for t in tags]

    start = time.perf_counter()
    _dispatch(cfg, run_tag, tasks, lambda t: t["tag"], all_results, times)
    elapsed = time.perf_counter() - start

    _grid_summary("CONDITION GRID", len(all_results), total, elapsed, times)
    run_times[run_label] = elapsed


def _run_factorial(cfg, ecosystem_for_loader, all_results, run_times, run_label):
    """Factorial design: one task per (tag, seed) cell.

    The fully-crossed design is form x scaling x construction x policy x seed;
    scaling x construction selects the condition raster tag, and form + policy
    are scenario_params overrides applied per cell. Grouping by (tag, seed) means
    each worker loads that tag's rasters once and reuses them across the
    form x policy combinations at that seed.
    """
    seeds = _seed_list(cfg)
    seeds = seeds if seeds is not None else [cfg["RANDOM_SEED"]]
    forms, scalings = cfg["FACTORIAL_FORMS"], cfg["FACTORIAL_SCALINGS"]
    constructions, policies = cfg["FACTORIAL_CONSTRUCTIONS"], cfg["FACTORIAL_POLICIES"]
    total = len(forms) * len(scalings) * len(constructions) * len(policies) * len(seeds)
    times = {}

    print(f"  Factorial design: {total} runs "
          f"({len(forms)} form x {len(scalings)} scaling x "
          f"{len(constructions)} construction x {len(policies)} policy x "
          f"{len(seeds)} seed)")
    parent = _grid_parent(cfg)
    print(f"  GRID_WORKERS={cfg['GRID_WORKERS']}, "
          f"{len(scalings) * len(constructions) * len(seeds)} (tag, seed) cells")

    grid_cfg = _grid_cfg(cfg, ecosystem_for_loader, parent)
    tasks = [{"scaling": sc, "construction": co, "seed": sd,
              "forms": forms, "policies": policies, "cfg": grid_cfg}
             for sd in seeds for sc in scalings for co in constructions]

    start = time.perf_counter()
    _dispatch(cfg, run_factorial_cell, tasks,
              lambda t: f"{t['scaling']}_{t['construction']} seed{t['seed']}",
              all_results, times)
    elapsed = time.perf_counter() - start

    _grid_summary("FACTORIAL GRID", len(all_results), total, elapsed, times)
    run_times[run_label] = elapsed


def _run_custom_parallel(cfg, ecosystem_for_loader, all_results, run_times, run_label, seeds):
    """Custom mode across a process pool - one task per seed.

    Seed replicates of a single scenario are fully independent, exactly like the
    grid cells, so they parallelise the same way. Each worker loads
    CONDITION_SCENARIO's rasters itself and writes its own line-buffered log, so
    verbose output stays readable AND live while several runs execute at once.
    """
    grid_cfg = {
        **_grid_cfg(cfg, ecosystem_for_loader, None),  # r_parent None = per-run R export
        # Safe to force on: each worker's stdout goes to its own file, so parallel
        # runs cannot interleave. Without this a parallel run is silent for hours.
        "verbose": True,
        "log_dir": str(LOGS_DIR),
    }
    tasks = [{"tag": cfg["CONDITION_SCENARIO"], "seed": s,
              "run_label": f"{cfg['RUN_LABEL']}_seed{s}", "cfg": grid_cfg}
             for s in seeds]

    print(f"  Custom scenario x {len(seeds)} seeds: {seeds}, "
          f"GRID_WORKERS={cfg['GRID_WORKERS']}")
    print(f"  Per-run logs -> {LOGS_DIR}/<run_label>.log")

    times = {}
    start = time.perf_counter()
    _dispatch(cfg, run_custom_seed, tasks, lambda t: t["run_label"], all_results, times)
    elapsed = time.perf_counter() - start

    print("\n=== CUSTOM RUN COMPLETE ===")
    print(f"  {len(all_results)}/{len(tasks)} runs succeeded")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed:.0f} s)")
    print(f"  Results PKLs: results_files/res_<timestamp>_{cfg['RUN_LABEL']}_seed<n>.pkl")
    run_times.update(times)
    run_times[run_label] = elapsed


def _run_policy_grid(cfg, ecosystem_for_loader, all_results, run_times, run_label):
    """Run each POLICY_VARIANTS entry, then each BENCHMARK_SCENARIOS tag, x SEEDS.

    Sequential: the policy variants share one initial_conditions load (they only
    differ in scenario_params), so there is no per-task raster cost to amortise
    across processes the way the grids do.
    """
    seeds = _seed_list(cfg)
    use_seeds = seeds is not None
    seeds = seeds if use_seeds else [cfg["RANDOM_SEED"]]
    variants, benchmarks = cfg["POLICY_VARIANTS"], cfg["BENCHMARK_SCENARIOS"]
    total = (len(variants) + len(benchmarks)) * len(seeds)

    print(f"\n  Policy variants to run ({len(variants)} x {len(seeds)} seeds):")
    for name, params in variants.items():
        print(f"    {name}: {params or {'(baseline - no overrides)': ''}}")
    if benchmarks:
        print(f"  Benchmark scenarios ({len(benchmarks)} x {len(seeds)} seeds):")
        for tag in benchmarks:
            print(f"    {tag}")

    times = {}
    done = 0
    start = time.perf_counter()
    parent = _grid_parent(cfg)

    def one(ic, params, lbl, seed, run_cfg):
        item_start = time.perf_counter()
        try:
            res = run_optimization_instance(
                initial_conditions=ic,
                scenario_params=params,
                pop_size=cfg["POP_SIZE"],
                n_generations=cfg["N_GENERATIONS"],
                save_results=True,
                verbose=True,
                random_seed=seed,
                use_repair=True,
                use_patch_approach=cfg["USE_PATCH_APPROACH"],
                patch_size=cfg["PATCH_SIZE"],
                patch_constraint_type=cfg["PATCH_CONSTRAINT_TYPE"],
                pixel_tolerance=cfg["PIXEL_TOLERANCE"],
                save_snapshots=cfg["SAVE_SNAPSHOTS"],
                snapshot_generations=cfg["SNAPSHOT_GENERATIONS"],
                capture_repair_diag=cfg["CAPTURE_REPAIR_DIAG"],
                n_capture_gens=cfg["N_CAPTURE_GENS"],
                n_partitions=cfg["N_PARTITIONS"],
                warm_seeding=cfg["WARM_SEEDING"],
                run_label=lbl,
                run_config=run_cfg,
                r_export_parent=parent,
                mutation_flip_count=cfg["MUTATION_FLIP_COUNT"],
                algorithm_type=cfg["ALGORITHM"],
                **_hv_kw(cfg),
            )
            times[lbl] = time.perf_counter() - item_start
            if res is not None:
                all_results[lbl] = res
                print(f"  [ok] {lbl} completed ({times[lbl]/60:.1f} min)")
            else:
                print(f"  [x] {lbl} returned no results ({times[lbl]/60:.1f} min)")
        except Exception as e:
            times[lbl] = time.perf_counter() - item_start
            print(f"  [x] {lbl} failed: {e}")

    # -- policy variants (one shared IC: they differ only in scenario_params) --
    ic_policy = load_initial_conditions(
        ".",
        objectives=cfg["OBJECTIVES"],
        region=cfg["REGION"],
        ecosystem=ecosystem_for_loader,
        sample_fraction=cfg["SAMPLE_FRACTION"],
        sample_seed=cfg["SAMPLE_SEED"],
        aggregation_factor=cfg["AGGREGATION_FACTOR"],
        condition_scenario=cfg["CONDITION_SCENARIO"],
    )
    for name, overrides in variants.items():
        for seed in seeds:
            lbl = f"{name}_seed{seed}" if use_seeds else name
            done += 1
            print(f"  policy_grid [{done}/{total}]: {lbl}")
            one(ic_policy,
                {**cfg["custom_scenario_params"], **overrides},
                lbl, seed,
                {**cfg["_run_config"], "policy_variant": name,
                 "condition_scenario": cfg["CONDITION_SCENARIO"], "random_seed": seed})

    # -- benchmark scenarios (each is a different condition tag) --
    for tag in benchmarks:
        ic_bench = load_initial_conditions(
            ".",
            objectives=cfg["OBJECTIVES"],
            region=cfg["REGION"],
            ecosystem=ecosystem_for_loader,
            sample_fraction=cfg["SAMPLE_FRACTION"],
            sample_seed=cfg["SAMPLE_SEED"],
            aggregation_factor=cfg["AGGREGATION_FACTOR"],
            condition_scenario=tag,
        )
        for seed in seeds:
            lbl = f"{tag}_seed{seed}" if use_seeds else tag
            done += 1
            print(f"  policy_grid [{done}/{total}]: {lbl} (benchmark)")
            one(ic_bench, cfg["custom_scenario_params"], lbl, seed,
                {**cfg["_run_config"], "benchmark_scenario": tag,
                 "condition_scenario": tag, "random_seed": seed})

    elapsed = time.perf_counter() - start
    print("\n=== POLICY GRID COMPLETE ===")
    print(f"  {len(all_results)}/{total} runs succeeded")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed:.0f} s)")
    if times:
        avg = sum(times.values()) / len(times)
        print(f"  Average per run: {avg/60:.1f} min ({avg:.0f} s)")
    print("  Results PKLs: results_files/res_<timestamp>_<label>.pkl")
    run_times[run_label] = elapsed


def _run_custom(cfg, ecosystem_for_loader, all_results, run_times, run_label):
    """Single custom scenario, optionally replicated across SEEDS in-process.

    Data load is seed-independent, so it happens once and is reused across seeds.
    """
    initial_conditions = load_initial_conditions(
        ".",
        objectives=cfg["OBJECTIVES"],
        region=cfg["REGION"],
        ecosystem=ecosystem_for_loader,
        sample_fraction=cfg["SAMPLE_FRACTION"],
        sample_seed=cfg["SAMPLE_SEED"],
        aggregation_factor=cfg["AGGREGATION_FACTOR"],
        condition_scenario=cfg["CONDITION_SCENARIO"],
    )
    print(f"[ok] Data loaded for {run_label}")

    multiseed = _multiseed(cfg)
    seeds = _seed_list(cfg) if multiseed else [cfg["RANDOM_SEED"]]
    if multiseed:
        print(f"  Custom run replicated across {len(seeds)} seeds: {seeds} "
              f"(sequential: GRID_WORKERS={cfg['GRID_WORKERS']})")

    results = None
    for seed in seeds:
        lbl = f"{cfg['RUN_LABEL']}_seed{seed}" if multiseed else cfg["RUN_LABEL"]
        seed_start = time.perf_counter()
        if multiseed:
            print(f"\n--- seed {seed} -> {lbl} ---")
        results = run_optimization_instance(
            initial_conditions=initial_conditions,
            scenario_params=cfg["custom_scenario_params"],
            pop_size=cfg["POP_SIZE"],
            n_generations=cfg["N_GENERATIONS"],
            save_results=True,
            verbose=True,
            random_seed=seed,
            use_repair=True,
            use_patch_approach=cfg["USE_PATCH_APPROACH"],
            patch_size=cfg["PATCH_SIZE"],
            patch_constraint_type=cfg["PATCH_CONSTRAINT_TYPE"],
            pixel_tolerance=cfg["PIXEL_TOLERANCE"],
            save_snapshots=cfg["SAVE_SNAPSHOTS"],
            snapshot_generations=cfg["SNAPSHOT_GENERATIONS"],
            capture_repair_diag=cfg["CAPTURE_REPAIR_DIAG"],
            n_capture_gens=cfg["N_CAPTURE_GENS"],
            n_partitions=cfg["N_PARTITIONS"],
            warm_seeding=cfg["WARM_SEEDING"],
            run_label=lbl,
            run_config=({**cfg["_run_config"], "random_seed": seed}
                        if multiseed else cfg["_run_config"]),
            mutation_flip_count=cfg["MUTATION_FLIP_COUNT"],
            algorithm_type=cfg["ALGORITHM"],
            **_hv_kw(cfg),
        )
        if multiseed:
            key = f"{run_label}_seed{seed}"
            if results is not None:
                all_results[key] = results
            run_times[key] = time.perf_counter() - seed_start
            print(f"  [{'ok' if results is not None else 'x'}] {lbl} "
                  f"({run_times[key]/60:.1f} min)")

    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _run(cfg):
    """Main execution body - separated so cProfile can wrap it cleanly."""
    cfg["_run_config"] = _build_run_config(cfg)
    all_results, run_times = {}, {}
    script_start = time.perf_counter()
    mode = cfg["SCENARIO_MODE"]
    multiseed = _multiseed(cfg)
    runs = _resolve_runs(cfg)

    for run_label, ecosystem_for_loader in runs:
        print(f"=== Starting optimisation for {run_label.upper()} ecosystem ===")
        run_start = time.perf_counter()
        results = None

        try:
            if mode == "condition_grid":
                _run_condition_grid(cfg, ecosystem_for_loader, all_results, run_times, run_label)
            elif mode == "policy_grid":
                _run_policy_grid(cfg, ecosystem_for_loader, all_results, run_times, run_label)
            elif mode == "factorial":
                _run_factorial(cfg, ecosystem_for_loader, all_results, run_times, run_label)
            elif multiseed and cfg["GRID_WORKERS"] > 1:
                # Seed replicates are independent runs, so they parallelise
                # exactly like the grid modes.
                _run_custom_parallel(cfg, ecosystem_for_loader, all_results, run_times,
                                     run_label, _seed_list(cfg))
            elif mode == "custom":
                results = _run_custom(cfg, ecosystem_for_loader, all_results,
                                      run_times, run_label)
            else:
                raise ValueError(
                    f"SCENARIO_MODE={mode!r} unknown - use one of "
                    f"custom, {', '.join(GRID_MODES)}.")

            if not multiseed and mode not in GRID_MODES:
                run_times[run_label] = time.perf_counter() - run_start
                if results is not None:
                    all_results[run_label] = results
                    print(f"\n[ok] {run_label.title()} optimisation completed successfully!")
                else:
                    print(f"\n[x] {run_label.title()} optimisation failed.")
            elif run_label not in run_times:
                run_times[run_label] = time.perf_counter() - run_start

        except Exception as e:
            run_times[run_label] = time.perf_counter() - run_start
            print(f"\n[x] Error optimising {run_label} ecosystem: {e}")
            continue

    _final_summary(cfg, runs, all_results, run_times,
                   time.perf_counter() - script_start, multiseed)
    return all_results


def _final_summary(cfg, runs, all_results, run_times, total_elapsed, multiseed):
    print(f"\n{'='*80}")
    print("=== OPTIMISATION SUMMARY ===")
    print(f"{'='*80}")

    if multiseed or cfg["SCENARIO_MODE"] in GRID_MODES:
        label = (f"custom x {len(_seed_list(cfg))} seeds" if multiseed
                 else cfg["SCENARIO_MODE"])
        print(f"{label}: {len(all_results)} runs stored in all_results")
        print(f"\nTotal wall-clock time: {total_elapsed/60:.1f} min ({total_elapsed:.0f} s)")
        return

    succeeded = len(all_results)
    print(f"Successfully completed {succeeded}/{len(runs)} ecosystem optimisations:")
    for run_label, _ in runs:
        status = "[ok] SUCCESS" if run_label in all_results else "[x] FAILED"
        elapsed = run_times.get(run_label)
        time_str = f"  ({elapsed/60:.1f} min)" if elapsed is not None else ""
        print(f"  {run_label.title():<12}: {status}{time_str}")

    print(f"\nTotal wall-clock time: {total_elapsed/60:.1f} min ({total_elapsed:.0f} s)")
    if succeeded > 0:
        print(f"[ok] Completed with outputs for {succeeded} ecosystems.")
    else:
        print("[x] No optimisations completed successfully.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m Core_optimisation.run",
        description="Run an optimisation configured by a preset in Core_optimisation/configs/.")
    parser.add_argument("preset", nargs="?",
                        help="config preset name, e.g. nsga2_2obj")
    parser.add_argument("--list", action="store_true",
                        help="list available presets and exit")
    args = parser.parse_args(argv)

    if args.list or not args.preset:
        print("Available config presets:")
        for name in _available_presets():
            print(f"  {name}")
        return 0

    cfg = _load_preset(args.preset)

    print(f"\n=== RESTORATION OPTIMIZATION [{args.preset}] FOR "
          f"{cfg['ECOSYSTEM_TO_RUN'].upper()} ECOSYSTEM, REGION {cfg['REGION']} ===")
    print(f"Scenario mode: {cfg['SCENARIO_MODE']}")

    log_path = setup_logger(
        log_dir=str(LOGS_DIR),
        run_label=f"{cfg['ECOSYSTEM_TO_RUN']}_{cfg['REGION'].lower()}{cfg['LOG_SUFFIX']}")
    if log_path:
        print(f"Verbose output -> {log_path}")

    if not cfg["PROFILE"]:
        _run(cfg)
        return 0

    profiler = cProfile.Profile()
    profiler.enable()
    _run(cfg)
    profiler.disable()
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(cfg["PROFILE_TOP_N"])
    text = buf.getvalue()
    print("\n" + text)
    path = os.path.join(str(LOGS_DIR), f"profile_{cfg['ECOSYSTEM_TO_RUN']}_{cfg['RUN_LABEL']}.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Profile saved -> {path}")
    return 0


# The __main__ guard is REQUIRED: the grid modes use a ProcessPoolExecutor, and on
# the 'spawn' start method (Windows default) each worker re-imports this module.
# Without the guard, every worker would re-launch the whole grid. Workers import
# this module as '__mp_main__', so this block is skipped in them.
if __name__ == "__main__":
    sys.exit(main())
