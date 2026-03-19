"""
Multi-Seed Restoration Optimization
====================================
Runs the restoration optimization across multiple random seeds for a single ecosystem type.
This helps assess the robustness and variability of optimization outcomes.

Created: February 2026
"""

import os
import json
import numpy as np
from datetime import datetime
from resto_anom import run_single_scenario_optimization
from data_loader import load_initial_conditions

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ecosystem to optimize
ECOSYSTEM = "forest"  # Options: 'forest', 'agricultural', 'grassland'

# Region for optimization
REGION = "Bern"  # Options: 'Bern', 'CH' for Switzerland-wide

# Number of random seeds to run
N_SEEDS = 10

# Seed configuration
BASE_SEED = 100  # Starting seed value
SEEDS = [BASE_SEED + i for i in range(N_SEEDS)]  # Seeds: 100, 101, 102, ..., 109

# Optimization parameters
OBJECTIVES = ["abiotic", "biotic", "cost"]
SAMPLE_FRACTION = 0.3  # Use 20% of eligible pixels
SAMPLE_SEED = 42  # Fixed seed for spatial sampling (to keep the spatial sample consistent)
POP_SIZE = 20
N_GENERATIONS = 30
N_JOBS = 12

# Patch approach configuration
USE_PATCH_APPROACH = True  # Set to True to use patch-based optimization
PATCH_SIZE = 10  # Size of patches (pixels) - only used if USE_PATCH_APPROACH=True
PATCH_CONSTRAINT_TYPE = 'pixel_count'  # 'pixel_count' (recommended) or 'patch_count'
PIXEL_TOLERANCE = 0.05  # Tolerance for pixel_count constraint (±5%)

# Scenario parameters (single scenario to test across seeds)
scenario_params = {
    "max_restoration_fraction": 0.1,
    "spatial_clustering": 0,
    "burden_sharing": "no",
    "abiotic_effect": 0.01,
    "biotic_effect": 0.01,
}

# Output directory
OUTPUT_DIR = "multi_seed_results"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_optimization_with_seed(seed, initial_conditions, run_index, total_runs):
    """
    Run optimization with a specific random seed.
    
    Args:
        seed: Random seed to use
        initial_conditions: Initial conditions dictionary
        run_index: Current run number (0-indexed)
        total_runs: Total number of runs
        
    Returns:
        dict: Results dictionary with seed information
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_index + 1}/{total_runs}: SEED = {seed}")
    print(f"{'='*80}")
    
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
            random_seed=seed,  # Pass seed to optimization
            use_patch_approach=USE_PATCH_APPROACH,
            patch_size=PATCH_SIZE,
            patch_constraint_type=PATCH_CONSTRAINT_TYPE,
            pixel_tolerance=PIXEL_TOLERANCE
        )
        
        if results is not None:
            # Add seed information to results
            results['random_seed'] = seed
            results['run_index'] = run_index
            print(f"\n✓ Seed {seed} completed successfully!")
            return results
        else:
            print(f"\n✗ Seed {seed} failed - no results returned.")
            return None
            
    except Exception as e:
        print(f"\n✗ Seed {seed} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_summary_results(all_results, output_dir):
    """
    Save summary statistics across all seed runs.
    
    Args:
        all_results: Dictionary of results keyed by seed
        output_dir: Directory to save summary
    """
    summary = {
        'ecosystem': ECOSYSTEM,
        'region': REGION,
        'n_seeds': len(all_results),
        'successful_seeds': list(all_results.keys()),
        'scenario_params': scenario_params,
        'optimization_params': {
            'pop_size': POP_SIZE,
            'n_generations': N_GENERATIONS,
            'sample_fraction': SAMPLE_FRACTION,
            'objectives': OBJECTIVES,
            'use_patch_approach': USE_PATCH_APPROACH,
            'patch_size': PATCH_SIZE if USE_PATCH_APPROACH else None,
            'patch_constraint_type': PATCH_CONSTRAINT_TYPE if USE_PATCH_APPROACH else None,
            'pixel_tolerance': PIXEL_TOLERANCE if USE_PATCH_APPROACH else None
        },
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Extract key metrics from each run
    seed_metrics = []
    for seed, results in all_results.items():
        if 'results_df' in results and results['results_df'] is not None:
            df = results['results_df']
            metrics = {
                'seed': seed,
                'n_solutions': len(df),
                'hypervolume': results.get('hypervolume', None),
                'mean_abiotic_improvement': df['obj_1'].mean() if 'obj_1' in df else None,
                'mean_biotic_improvement': df['obj_2'].mean() if 'obj_2' in df else None,
                'mean_cost': df['obj_3'].mean() if 'obj_3' in df else None,
                'max_abiotic_improvement': df['obj_1'].max() if 'obj_1' in df else None,
                'max_biotic_improvement': df['obj_2'].max() if 'obj_2' in df else None,
                'min_cost': df['obj_3'].min() if 'obj_3' in df else None,
            }
            seed_metrics.append(metrics)
    
    summary['seed_metrics'] = seed_metrics
    
    # Save summary
    summary_file = os.path.join(output_dir, f'multi_seed_summary_{ECOSYSTEM}_{summary["timestamp"]}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    return summary


def main():
    """
    Main execution function for multi-seed optimization.
    """
    print(f"\n{'='*80}")
    print(f"MULTI-SEED RESTORATION OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Ecosystem:        {ECOSYSTEM}")
    print(f"Region:           {REGION}")
    print(f"Number of seeds:  {N_SEEDS}")
    print(f"Seeds:            {SEEDS}")
    print(f"Population size:  {POP_SIZE}")
    print(f"Generations:      {N_GENERATIONS}")
    print(f"Sample fraction:  {SAMPLE_FRACTION}")
    if USE_PATCH_APPROACH:
        print(f"Patch approach:   YES ({PATCH_SIZE}×{PATCH_SIZE} patches)")
        print(f"  Constraint:     {PATCH_CONSTRAINT_TYPE}")
        if PATCH_CONSTRAINT_TYPE == 'pixel_count':
            print(f"  Tolerance:      ±{PIXEL_TOLERANCE*100:.1f}%")
    else:
        print(f"Patch approach:   NO (pixel-based)")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load initial conditions once (using fixed sample_seed for consistency)
    print("Loading initial conditions...")
    initial_conditions = load_initial_conditions(
        ".",
        objectives=OBJECTIVES,
        region=REGION,
        ecosystem=ECOSYSTEM,
        sample_fraction=SAMPLE_FRACTION,
        sample_seed=SAMPLE_SEED
    )
    print("✓ Initial conditions loaded\n")
    
    # Run optimization for each seed
    all_results = {}
    successful_seeds = []
    failed_seeds = []
    
    for idx, seed in enumerate(SEEDS):
        results = run_optimization_with_seed(seed, initial_conditions, idx, N_SEEDS)
        
        if results is not None:
            all_results[seed] = results
            successful_seeds.append(seed)
        else:
            failed_seeds.append(seed)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"MULTI-SEED OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Ecosystem:           {ECOSYSTEM}")
    print(f"Total runs:          {N_SEEDS}")
    print(f"Successful runs:     {len(successful_seeds)}")
    print(f"Failed runs:         {len(failed_seeds)}")
    
    if successful_seeds:
        print(f"\nSuccessful seeds:    {successful_seeds}")
    if failed_seeds:
        print(f"Failed seeds:        {failed_seeds}")
    
    # Save summary results
    if all_results:
        summary = save_summary_results(all_results, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print(f"RESULTS OVERVIEW")
        print(f"{'='*80}")
        
        if 'seed_metrics' in summary and summary['seed_metrics']:
            print(f"\n{'Seed':<8} {'N Solutions':<15} {'Hypervolume':<15}")
            print(f"{'-'*40}")
            for metrics in summary['seed_metrics']:
                seed = metrics['seed']
                n_sols = metrics['n_solutions']
                hv = metrics['hypervolume']
                hv_str = f"{hv:.4f}" if hv is not None else "N/A"
                print(f"{seed:<8} {n_sols:<15} {hv_str:<15}")
    
    print(f"\n{'='*80}")
    print(f"✓ Multi-seed optimization complete!")
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    results = main()
