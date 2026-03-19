"""
Patch Size Experiment Runner
=============================
Runs restoration optimization across multiple patch sizes and random seeds.

Created: March 2026
"""

import os
from datetime import datetime
from resto_anom import run_single_scenario_optimization
from data_loader import load_initial_conditions

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ecosystems to optimize
ECOSYSTEMS = ["forest"]#, "agricultural", "grassland"]  # All three ecosystems
REGION = "Bern"

# Seeds to test
SEEDS = [100, 101, 102, 103, 104, 105, 106]  # 5 different seeds

# Patch sizes to test
PATCH_SIZES = [2]  # 3 different patch sizes

# Optimization parameters
OBJECTIVES = ["abiotic", "biotic", "cost"]
SAMPLE_FRACTION = None
SAMPLE_SEED = 42  # Fixed for consistent spatial sampling
POP_SIZE = 20
N_GENERATIONS = 30
N_JOBS = 12

# Patch configuration
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.05

# Scenario parameters
scenario_params = {
    "max_restoration_fraction": 0.1,
    "spatial_clustering": 0,
    "burden_sharing": "no",
    "abiotic_effect": 0.01,
    "biotic_effect": 0.01,
}

# Output directory
OUTPUT_DIR = "patch_tests"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run optimization for all ecosystems, patch sizes and seeds."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Track results
    run_number = 0
    total_runs = len(ECOSYSTEMS) * len(PATCH_SIZES) * len(SEEDS)
    results_summary = []
    
    # Loop through ecosystems, patch sizes and seeds
    for ecosystem in ECOSYSTEMS:
        print(f"\n{'#'*80}")
        print(f"# ECOSYSTEM: {ecosystem.upper()}")
        print(f"{'#'*80}\n")
        
        for patch_size in PATCH_SIZES:
            # Load initial conditions for each patch size to avoid reusing patch mappings
            print(f"\nLoading initial conditions for {ecosystem} with patch size {patch_size}...")
            initial_conditions = load_initial_conditions(
                ".",
                objectives=OBJECTIVES,
                region=REGION,
                ecosystem=ecosystem,
                sample_fraction=SAMPLE_FRACTION,
                sample_seed=SAMPLE_SEED
            )
            print(f"✓ Initial conditions loaded\n")
            
            for seed in SEEDS:
                run_number += 1
                
                print(f"\n{'='*80}")
                print(f"RUN {run_number}/{total_runs}")
                print(f"Ecosystem: {ecosystem} | Patch size: {patch_size}×{patch_size} | Seed: {seed}")
                print(f"{'='*80}")
                
                # Create subdirectory for this run
                run_output_dir = os.path.join(OUTPUT_DIR, f"{ecosystem}_{patch_size}_{seed}")
                os.makedirs(run_output_dir, exist_ok=True)
                
                try:
                    results = run_single_scenario_optimization(
                        initial_conditions=initial_conditions,
                        scenario_params=scenario_params,
                        pop_size=POP_SIZE,
                        n_generations=N_GENERATIONS,
                        save_results=True,
                        verbose=True,
                        n_jobs=N_JOBS,
                        use_repair=True,
                        random_seed=seed,
                        use_patch_approach=True,
                        patch_size=patch_size,
                        patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                        pixel_tolerance=PIXEL_TOLERANCE,
                        output_dir=run_output_dir
                    )
                    
                    if results is not None:
                        status = "✓ SUCCESS"
                        hv = results.get('hypervolume', None)
                        n_solutions = len(results['results_df']) if 'results_df' in results else 0
                    else:
                        status = "✗ FAILED"
                        hv = None
                        n_solutions = 0
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
                    status = "✗ ERROR"
                    hv = None
                    n_solutions = 0
                
                # Record result
                results_summary.append({
                    'ecosystem': ecosystem,
                    'patch_size': patch_size,
                    'seed': seed,
                    'status': status,
                    'hypervolume': hv,
                    'n_solutions': n_solutions
                })
                
                print(f"{status}")
    
    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Ecosystem':<15} {'Patch Size':<12} {'Seed':<8} {'Status':<12} {'HV':<12} {'Solutions':<12}")
    print(f"{'-'*80}")
    
    for result in results_summary:
        hv_str = f"{result['hypervolume']:.4f}" if result['hypervolume'] is not None else "N/A"
        print(f"{result['ecosystem']:<15} {result['patch_size']:<12} {result['seed']:<8} {result['status']:<12} {hv_str:<12} {result['n_solutions']:<12}")
    
    print(f"{'='*80}\n")
    print(f"✓ Experiment complete! Results saved to: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
