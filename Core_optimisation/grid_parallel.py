"""
Process-parallel drivers for the independent-run grids.
========================================================

The condition grid and the factorial design are both sets of fully independent
optimisation runs. The pymoo solver is sequential across generations, and the
within-generation ``n_jobs`` thread pool is GIL-bound (per-individual
``_evaluate`` is mostly Python-level work), so a single run leaves most cores
idle. The runs themselves share nothing - so the real parallelism lives at the
grid level and is best exploited with *processes*, which sidestep the GIL.

Each worker function below is the unit of work submitted to a process pool. It
loads ``initial_conditions`` once for its condition-scenario tag and reuses it
across every run it owns, so the heavy raster I/O + connectivity computation +
admin-shapefile clip happens once per worker rather than once per run. The large
arrays never cross a process boundary, and only a lightweight summary is
returned - full results are persisted to disk by
``run_optimization_instance(save_results=True)``.

Workers:
  - ``run_tag``            : condition grid - one tag, all of its seeds.
  - ``run_factorial_cell`` : factorial design - one (tag, seed) cell, all of its
                             form x policy combinations.

This module deliberately has NO import-time side effects so it can be safely
re-imported by ``spawn`` worker processes on Windows.
"""

import time

from .data_loader import load_initial_conditions
from .resto_anom import run_optimization_instance


def _invoke(ic, scenario_params, run_label, run_config, seed, cfg):
    """Run one optimisation instance with the grid's shared settings.

    Process-level parallelism IS the parallelism here, so the inner thread pool
    is forced serial (``n_jobs=1``) - it is GIL-bound and ineffective, and using
    it would only oversubscribe cores against the other workers.
    """
    return run_optimization_instance(
        initial_conditions=ic,
        scenario_params=scenario_params,
        pop_size=cfg["pop_size"],
        n_generations=cfg["n_generations"],
        save_results=True,
        # Quiet by default: parallel workers interleave stdout. Echoed only when
        # the caller runs a single worker (GRID_WORKERS == 1) for debugging.
        verbose=cfg.get("verbose", False),
        n_jobs=1,
        random_seed=seed,
        use_repair=True,
        use_patch_approach=cfg["use_patch_approach"],
        patch_size=cfg["patch_size"],
        patch_constraint_type=cfg["patch_constraint_type"],
        pixel_tolerance=cfg["pixel_tolerance"],
        save_snapshots=cfg["save_snapshots"],
        n_partitions=cfg["n_partitions"],
        warm_seeding=cfg["warm_seeding"],
        run_label=run_label,
        run_config=run_config,
        r_export_parent=cfg["r_parent"],
        # Defaults reproduce the historical behaviour for callers (e.g.
        # run_custom_parallel.py) whose cfg omits these keys: nsga3, no explicit
        # mutation flip count, no repair-diagnostic capture. run_custom_nsga2.py
        # sets them so its grid runs NSGA-II with its diag settings.
        algorithm_type=cfg.get("algorithm_type", "nsga3"),
        mutation_flip_count=cfg.get("mutation_flip_count", None),
        capture_repair_diag=cfg.get("capture_repair_diag", False),
        n_capture_gens=cfg.get("n_capture_gens", 5),
    )


def run_tag(task):
    """Condition grid: run every seed for a single condition tag in one process.

    Parameters
    ----------
    task : dict
        Picklable work item with keys:
          - 'tag'       : str   condition-scenario raster tag (e.g. 'global_all')
          - 'seeds'     : list[int]
          - 'use_seeds' : bool  whether to suffix run labels with the seed
          - 'cfg'       : dict  run settings shared across all tasks

    Returns
    -------
    list[tuple]
        One ``(run_label, succeeded, elapsed_seconds, error_or_None)`` per seed.
    """
    tag = task["tag"]
    seeds = task["seeds"]
    use_seeds = task["use_seeds"]
    cfg = task["cfg"]

    # Load rasters/derived layers ONCE for this tag, reuse across seeds.
    ic = load_initial_conditions(
        ".",
        objectives=cfg["objectives"],
        region=cfg["region"],
        ecosystem=cfg["ecosystem"],
        sample_fraction=cfg["sample_fraction"],
        sample_seed=cfg["sample_seed"],
        aggregation_factor=cfg["aggregation_factor"],
        condition_scenario=tag,
    )

    summaries = []
    for seed in seeds:
        run_label = f"{tag}_seed{seed}" if use_seeds else tag
        t0 = time.perf_counter()
        try:
            res = _invoke(
                ic, cfg["scenario_params"], run_label,
                {**cfg["run_config"], "condition_scenario": tag, "random_seed": seed},
                seed, cfg,
            )
            summaries.append((run_label, res is not None, time.perf_counter() - t0, None))
        except Exception as e:  # noqa: BLE001 - report, never crash the whole pool
            summaries.append((run_label, False, time.perf_counter() - t0, str(e)))

    return summaries


def run_factorial_cell(task):
    """Factorial design: run all form x policy combinations for one (tag, seed).

    The condition raster tag is ``f"{scaling}_{construction}"``. A cell loads its
    IC once and runs every (form, policy) combination at the given seed, so the
    raster load is amortised over ``len(forms) * len(policies)`` runs. Choosing
    the cell - rather than the whole tag - as the task keeps workers busy even
    when only a single construction level is active.

    Parameters
    ----------
    task : dict
        Picklable work item with keys:
          - 'scaling'      : str
          - 'construction' : str
          - 'seed'         : int
          - 'forms'        : list[str]            FACTORIAL_FORMS
          - 'policies'     : dict[str, dict]      name -> scenario_params overrides
          - 'cfg'          : dict                 run settings (cfg['scenario_params']
                                                  is the base custom_scenario_params)

    Returns
    -------
    list[tuple]
        One ``(run_label, succeeded, elapsed_seconds, error_or_None)`` per
        (form, policy) combination. If the tag's rasters cannot be loaded, every
        combination in the cell is reported as failed (mirrors the sequential
        "skip cells needing missing tag" behaviour).
    """
    scaling = task["scaling"]
    construction = task["construction"]
    seed = task["seed"]
    forms = task["forms"]
    policies = task["policies"]
    cfg = task["cfg"]
    tag = f"{scaling}_{construction}"

    def _label(form, pol_name):
        return (f"form-{form}__scal-{scaling}__"
                f"con-{construction}__pol-{pol_name}__seed{seed}")

    # Load IC once for this tag; on failure, report all combinations as skipped.
    try:
        ic = load_initial_conditions(
            ".",
            objectives=cfg["objectives"],
            region=cfg["region"],
            ecosystem=cfg["ecosystem"],
            sample_fraction=cfg["sample_fraction"],
            sample_seed=cfg["sample_seed"],
            aggregation_factor=cfg["aggregation_factor"],
            condition_scenario=tag,
        )
    except Exception as e:  # noqa: BLE001
        msg = f"load failed for tag '{tag}': {e}"
        return [(_label(form, pol), False, 0.0, msg)
                for form in forms for pol in policies]

    summaries = []
    for form in forms:
        for pol_name, pol_overrides in policies.items():
            run_label = _label(form, pol_name)
            merged_params = {**cfg["scenario_params"], "rp_formulation": form, **pol_overrides}
            run_config = {
                **cfg["run_config"],
                "condition_scenario": tag,
                "random_seed": seed,
                "factor_form": form,
                "factor_scaling": scaling,
                "factor_construction": construction,
                "factor_policy": pol_name,
            }
            t0 = time.perf_counter()
            try:
                res = _invoke(ic, merged_params, run_label, run_config, seed, cfg)
                summaries.append((run_label, res is not None, time.perf_counter() - t0, None))
            except Exception as e:  # noqa: BLE001
                summaries.append((run_label, False, time.perf_counter() - t0, str(e)))

    return summaries
