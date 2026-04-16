"""
Custom single-scenario and multi-ecosystem run script.

Edit the configuration variables below and run directly:
    python run_custom.py
"""
from resto_anom import run_optimization_instance, main
from data_loader import load_initial_conditions
from logger_setup import setup_logger

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "fg"

# Short human-readable label describing what this run is testing.
# Used in output filenames and the run_registry.jsonl log.
# Examples: "baseline", "patch_size2_highbudget", "testing_new_repair"
RUN_LABEL = "cost_corrected_noWS"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom" runs exactly one scenario using custom_scenario_params
#   "all" runs scenario="all" using the scenario sampling logic inside main()
SCENARIO_MODE = "custom"

print(f"\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
print(f"Scenario mode: {SCENARIO_MODE}")

log_path = setup_logger(log_dir="logs", run_label=f"{ECOSYSTEM_TO_RUN}_{REGION.lower()}")
if log_path:
    print(f"Verbose output → {log_path}")

OBJECTIVES = ["abiotic", "biotic", "cost"]
SAMPLE_FRACTION = None
SAMPLE_SEED = 42
POP_SIZE = 50
N_GENERATIONS = 100
N_JOBS = 12
RANDOM_SEED = 42
N_SAMPLES_PER_PARAM = 3
WARM_SEEDING = False

# Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
custom_scenario_params = {
    "max_restoration_fraction": 0.05,
    "spatial_clustering": 0,
    "biotic_effect": 0.01,
    "abiotic_effect": 0.01,
    "normalize_objectives": True,
    "patch_score_temperature": 2.0,
    "patch_repair_top_k": 100,
}

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
SAVE_SNAPSHOTS = True

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

all_results = {}

for run_label, ecosystem_for_loader in runs:
    print(f"=== Starting optimisation for {run_label.upper()} ecosystem ===")

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
        else:
            initial_conditions = load_initial_conditions(
                ".",
                objectives=OBJECTIVES,
                region=REGION,
                ecosystem=ecosystem_for_loader,
                sample_fraction=SAMPLE_FRACTION,
                sample_seed=SAMPLE_SEED,
                aggregation_factor=AGGREGATION_FACTOR,
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
                warm_seeding=WARM_SEEDING,
                run_label=RUN_LABEL,
                run_config=run_config,
            )

        if results is not None:
            all_results[run_label] = results
            print(f"\n✓ {run_label.title()} optimisation completed successfully!")
        else:
            print(f"\n✗ {run_label.title()} optimisation failed.")

    except Exception as e:
        print(f"\n✗ Error optimising {run_label} ecosystem: {e}")
        continue

print(f"\n{'='*80}")
print("=== OPTIMISATION SUMMARY ===")
print(f"{'='*80}")
successful_runs = len(all_results)
total_runs = len(runs)
print(f"Successfully completed {successful_runs}/{total_runs} ecosystem optimisations:")
for run_label, _ in runs:
    status = "✓ SUCCESS" if run_label in all_results else "✗ FAILED"
    print(f"  {run_label.title():<12}: {status}")

if successful_runs > 0:
    print(f"\n✓ Completed with outputs for {successful_runs} ecosystems.")
else:
    print("\n✗ No optimisations completed successfully.")
