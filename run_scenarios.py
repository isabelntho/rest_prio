# =============================================================================
# SCENARIO ORCHESTRATION
# =============================================================================
# This module handles everything that operates *across* scenarios:
#   - running a single scenario via the run_one() wrapper
#   - looping over a batch of scenarios
#   - expanding the full scenario space and running all combinations
#   - assembling and saving combined multi-scenario results
#
# The per-scenario optimization engine lives in resto_anom.py.
# This module imports from resto_anom; resto_anom does NOT import from here.
# =============================================================================

from datetime import datetime

from scenarios import expand_scenarios, define_scenario_parameters
from results_saving import save_combined_results, save_parameter_summary
from resto_anom import run_optimization_instance, _filter_initial_conditions_for_return


def run_one(initial_conditions, scenario_params, run_settings):
    """
    Run a single optimization instance from a run_settings dict.

    Args:
        initial_conditions: Initial objective conditions dict.
        scenario_params: Single scenario parameter dict.
        run_settings: Dict of algorithm/run configuration keys:
            pop_size, n_generations, save_results, verbose,
            skip_diagnostics, hv_patience, hv_min_improvement,
            n_jobs, use_repair, use_patch_approach, patch_size,
            patch_constraint_type, pixel_tolerance, output_dir.

    Returns:
        dict or None: Optimization results.
    """
    return run_optimization_instance(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params,
        pop_size=run_settings["pop_size"],
        n_generations=run_settings["n_generations"],
        save_results=run_settings["save_results"],
        verbose=run_settings["verbose"],
        skip_diagnostics=run_settings.get("skip_diagnostics", False),
        hv_patience=run_settings.get("hv_patience", 15),
        hv_min_improvement=run_settings.get("hv_min_improvement", 1e-6),
        n_jobs=run_settings.get("n_jobs", None),
        use_repair=run_settings.get("use_repair", True),
        use_patch_approach=run_settings.get("use_patch_approach", False),
        patch_size=run_settings.get("patch_size", 100),
        patch_constraint_type=run_settings.get("patch_constraint_type", "pixel_count"),
        pixel_tolerance=run_settings.get("pixel_tolerance", 0.05),
        output_dir=run_settings.get("output_dir", "."),
    )


def run_scenario_batch(initial_conditions, scenario_combinations, run_settings):
    """
    Run optimization for each scenario in scenario_combinations.

    Args:
        initial_conditions: Initial objective conditions dict.
        scenario_combinations: List of scenario parameter dicts.
        run_settings: Dict of algorithm/run configuration (see run_one).

    Returns:
        dict: Results keyed by scenario index (failed scenarios are omitted).
    """
    all_results = {}
    verbose = run_settings.get("verbose", True)

    if verbose:
        print("\n=== MULTI SCENARIO OPTIMIZATION ===")
        print(f"Total scenarios to run: {len(scenario_combinations)}")

    for i, scenario_params in enumerate(scenario_combinations):
        if verbose:
            print(f"\n--- RUNNING SCENARIO {i + 1}/{len(scenario_combinations)} ---")

        result = run_one(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            run_settings=run_settings,
        )

        if result is not None:
            all_results[i] = result
            if verbose:
                print(f"✓ Scenario {i} completed: {result['n_solutions']} solutions found")
        else:
            if verbose:
                print(f"✗ Scenario {i} failed")

    return all_results


def build_combined_results(
    initial_conditions,
    all_results,
    n_samples_per_param=3,
    random_seed=42,
    pop_size=50,
    n_generations=100,
    scenario_combinations=None,  # accepted for forward-compat; not used internally
):
    """
    Assemble a combined results dict from a completed batch run.

    Args:
        initial_conditions: Initial conditions dict (heavy gdf will be stripped).
        all_results: Dict of per-scenario result dicts keyed by scenario index.
        n_samples_per_param: Used to reconstruct total scenario count.
        random_seed: Used to reconstruct total scenario count.
        pop_size: Algorithm setting recorded for reference.
        n_generations: Algorithm setting recorded for reference.
        scenario_combinations: Ignored — kept for call-site compatibility.

    Returns:
        dict: combined_results ready for saving or downstream use.
    """
    initial_conditions_filtered = _filter_initial_conditions_for_return(initial_conditions)

    return {
        "scenarios": all_results,
        "n_scenarios_run": len(all_results),
        "n_scenarios_total": len(expand_scenarios(n_samples_per_param, random_seed)),
        "scenario_parameters": define_scenario_parameters(),
        "n_samples_per_param": n_samples_per_param,
        "random_seed": random_seed,
        "algorithm_info": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "timestamp": datetime.now().isoformat(),
        },
        "initial_conditions": initial_conditions_filtered,
    }


def finalise_combined_results(combined_results, save_results=True, verbose=True):
    """
    Save combined results and parameter summary to disk.

    Args:
        combined_results: Dict returned by build_combined_results.
        save_results: Skip all I/O when False.
        verbose: Print progress messages.
    """
    if not save_results:
        return

    all_results = combined_results.get("scenarios", {})
    if not all_results:
        return

    save_combined_results(combined_results, verbose=verbose)

    save_parameter_summary(
        output_dir=".",
        n_samples_per_param=combined_results.get("n_samples_per_param", 3),
        random_seed=combined_results.get("random_seed", 42),
        verbose=verbose,
    )


def run_all_scenarios_optimization(
    initial_conditions,
    n_samples_per_param=3,
    pop_size=50,
    n_generations=100,
    save_results=True,
    verbose=True,
    random_seed=42,
    hv_patience=15,
    hv_min_improvement=1e-6,
    n_jobs=None,
):
    """
    Expand the full scenario space and run optimization for every combination.

    Args:
        initial_conditions: Initial objective conditions dict.
        n_samples_per_param: Number of samples per continuous parameter.
        pop_size: NSGA-II population size.
        n_generations: Generation budget.
        save_results: Whether to persist combined results to disk.
        verbose: Print progress messages.
        random_seed: Seed for scenario expansion.
        hv_patience: HV convergence patience forwarded to each run.
        hv_min_improvement: HV improvement threshold forwarded to each run.
        n_jobs: Parallelism setting forwarded to each run.

    Returns:
        dict: combined_results from build_combined_results.
    """
    scenario_combinations = expand_scenarios(
        n_samples_per_param=n_samples_per_param, random_seed=random_seed
    )

    run_settings = {
        "pop_size": pop_size,
        "n_generations": n_generations,
        "save_results": False,   # individual saves suppressed; combined save at end
        "verbose": verbose,
        "skip_diagnostics": True,
        "hv_patience": hv_patience,
        "hv_min_improvement": hv_min_improvement,
        "n_jobs": n_jobs,
        "use_repair": True,
    }

    all_results = run_scenario_batch(
        initial_conditions=initial_conditions,
        scenario_combinations=scenario_combinations,
        run_settings=run_settings,
    )

    combined_results = build_combined_results(
        initial_conditions=initial_conditions,
        all_results=all_results,
        n_samples_per_param=n_samples_per_param,
        random_seed=random_seed,
        pop_size=pop_size,
        n_generations=n_generations,
    )

    if verbose:
        print("\n=== ALL SCENARIOS COMPLETE ===")
        print(f"Successfully completed: {len(all_results)}/{len(scenario_combinations)} scenarios")

    finalise_combined_results(combined_results, save_results=save_results, verbose=verbose)

    return combined_results
