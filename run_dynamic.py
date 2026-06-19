"""
Dynamic Restoration Optimisation — Entry Point
===============================================
Run the timing-based dynamic optimisation across multiple recovery scenarios
and random seeds.  Edit the configuration variables below and run with:

    python -m Core_optimisation.run_dynamic

or from the workspace root:

    python run_dynamic.py

For a quick smoke-test, set AGGREGATION_FACTOR=4 and N_GENERATIONS=20.
"""

import os
import sys
import time

# Add workspace root to path when run as a top-level script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from Core_optimisation.data_loader import load_initial_conditions
from Core_optimisation.logger_setup import setup_logger
from Core_optimisation.patch_approach import create_patch_mappings
from Core_optimisation.dynamic_optimisation import (
    RECOVERY_FUNCTIONS,
    run_dynamic_optimization_instance,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ECOSYSTEM_TO_RUN = "combined"   # "combined" | "forest" | "agricultural" | "grassland"
REGION = "Bern"
CONDITION_SCENARIO = "global_all"

# Planning years — each patch is assigned one of these years (or never=0)
TIME_STEPS = [2025, 2030, 2040, 2050]

# Maximum fraction of restoration pixels that may be restored across the FULL planning
# horizon.  Divided equally across time steps internally (per_step = fraction / n_steps).
PER_STEP_BUDGET_FRACTION = 0.10  # 10% total

# NSGA-III settings (4 objectives, n_partitions=6 → 84 reference directions)
N_PARTITIONS = 6
N_GENERATIONS = 100
POP_SIZE = None      # None = auto (number of reference directions = 84 with defaults)
N_JOBS = 1           # Parallel evaluation threads (1 = serial for safety)
DISCOUNT_RATE = 0.03 # Annual discount rate for present-value cumulative RP objective

# Random seeds — one run per (recovery_fn × seed) combination
SEEDS = [103]

# Patch size (pixels) used to define restoration patches
PATCH_SIZE = 2

# Spatial aggregation: block-coarsen inputs by this factor before optimisation.
# Use 4 for fast testing (~16× fewer pixels).  None or 1 disables aggregation.
AGGREGATION_FACTOR = 2

# Recovery functions to evaluate (subset of RECOVERY_FUNCTIONS keys)
RECOVERY_SCENARIOS = ["fast", "gradual", "delayed", "partial"]

# Output directory for pickled results
OUTPUT_DIR = "results_files"

# Raw rasters to load.  dynamic_optimisation.py derives the 3 optimisation
# objectives from these internally:
#
#   abiotic + biotic  →  F[0]=final_landscape_rp  and  F[1]=cumulative_landscape_rp
#                        Computed via restoration_effect() (direct + neighbour
#                        spillover, recovery-weighted) — same spatial methodology
#                        as the static optimisation.
#
#                        F[2]=final_landscape_context  (proportion of good-condition
#                        neighbours at t_end; binary threshold abiotic>0 AND biotic>0,
#                        500 m physical radius, computed from updated raster state)
#
#   cost              →  F[3]=total_cost  (used directly)
OBJECTIVES = ["abiotic", "biotic", "cost"]

# ---------------------------------------------------------------------------
# Run configuration snapshot (written to results for reproducibility)
# ---------------------------------------------------------------------------
run_config_base = {
    "ecosystem": ECOSYSTEM_TO_RUN,
    "region": REGION,
    "condition_scenario": CONDITION_SCENARIO,
    "time_steps": TIME_STEPS,
    "per_step_budget_fraction": PER_STEP_BUDGET_FRACTION,
    "n_partitions": N_PARTITIONS,
    "n_generations": N_GENERATIONS,
    "pop_size": POP_SIZE,
    "patch_size": PATCH_SIZE,
    "aggregation_factor": AGGREGATION_FACTOR,
    "recovery_scenarios": RECOVERY_SCENARIOS,
    "seeds": SEEDS,
    "objectives": OBJECTIVES,
    "discount_rate": DISCOUNT_RATE,
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ecosystem_for_loader = "all" if ECOSYSTEM_TO_RUN == "combined" else ECOSYSTEM_TO_RUN

log_path = setup_logger(log_dir="logs", run_label=f"dynamic_{ECOSYSTEM_TO_RUN}")
if log_path:
    print(f"Verbose output → {log_path}")

print(f"\n=== DYNAMIC RESTORATION OPTIMISATION ===")
print(f"  Ecosystem : {ECOSYSTEM_TO_RUN}")
print(f"  Region    : {REGION}")
print(f"  Time steps: {TIME_STEPS}")
print(f"  Scenarios : {RECOVERY_SCENARIOS}")
print(f"  Seeds     : {SEEDS}")
print(f"  Aggregation factor: {AGGREGATION_FACTOR}")


# ---------------------------------------------------------------------------
# Load initial conditions (once, shared across all runs)
# ---------------------------------------------------------------------------

print("\nLoading initial conditions …")
t0 = time.perf_counter()

initial_conditions = load_initial_conditions(
    workspace_dir=".",
    objectives=OBJECTIVES,
    region=REGION,
    ecosystem=ecosystem_for_loader,
    aggregation_factor=AGGREGATION_FACTOR,
    condition_scenario=CONDITION_SCENARIO,
)

# Inject patch mappings (required by DynamicRestorationProblem)
patch_mappings = create_patch_mappings(initial_conditions, patch_size=PATCH_SIZE)
initial_conditions["patch_mappings"] = patch_mappings
initial_conditions["n_restoration_patches"] = patch_mappings["restoration_patches"]["n_patches"]
initial_conditions["n_conversion_patches"] = patch_mappings["conversion_patches"]["n_patches"]

print(
    f"  Loaded in {time.perf_counter()-t0:.1f}s — "
    f"{initial_conditions['n_restoration_pixels']:,} restoration pixels, "
    f"{initial_conditions['n_restoration_patches']:,} patches"
)

# ---------------------------------------------------------------------------
# Main loop: recovery scenario × seed
# ---------------------------------------------------------------------------

all_results = {}
total_runs = len(RECOVERY_SCENARIOS) * len(SEEDS)
run_idx = 0

for scenario_name in RECOVERY_SCENARIOS:
    if scenario_name not in RECOVERY_FUNCTIONS:
        print(f"  WARNING: unknown recovery scenario '{scenario_name}', skipping")
        continue
    recovery_fn = RECOVERY_FUNCTIONS[scenario_name]

    for seed in SEEDS:
        run_idx += 1
        run_label = f"dynamic_{scenario_name}_seed{seed}"
        run_config = {**run_config_base, "recovery_fn": scenario_name, "random_seed": seed}

        print(f"\n[{run_idx}/{total_runs}] {run_label}")
        t_run = time.perf_counter()

        try:
            result = run_dynamic_optimization_instance(
                initial_conditions=initial_conditions,
                time_steps=TIME_STEPS,
                recovery_fn=recovery_fn,
                recovery_fn_name=scenario_name,
                per_step_budget_fraction=PER_STEP_BUDGET_FRACTION,
                pop_size=POP_SIZE,
                n_generations=N_GENERATIONS,
                n_partitions=N_PARTITIONS,
                n_jobs=N_JOBS,
                save_results=True,
                output_dir=OUTPUT_DIR,
                verbose=True,
                random_seed=seed,
                run_label=run_label,
                run_config=run_config,
                discount_rate=DISCOUNT_RATE,
            )

            elapsed = time.perf_counter() - t_run
            if result is not None:
                all_results[run_label] = result
                print(
                    f"  ✓ {run_label} completed in {elapsed/60:.1f} min "
                    f"({result['n_nondominated_solutions']} non-dominated solutions)"
                )
            else:
                print(f"  ✗ {run_label} returned no results ({elapsed/60:.1f} min)")

        except Exception as exc:
            elapsed = time.perf_counter() - t_run
            print(f"  ✗ {run_label} FAILED after {elapsed/60:.1f} min: {exc}")
            import traceback
            traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n=== Dynamic optimisation complete: {len(all_results)}/{total_runs} runs succeeded ===")
for label, res in all_results.items():
    nd = res.get("n_nondominated_solutions", "?")
    path = res.get("output_path", "(not saved)")
    print(f"  {label}: {nd} non-dominated solutions → {path}")
