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
from resto_anom import run_optimization_instance
from data_loader import load_initial_conditions

# =============================================================================
# CONFIGURATION
# =============================================================================
import itertools

# Ecosystem to optimize
ECOSYSTEM = "fg"  # Options: 'forest', 'agricultural', 'grassland'

# Region for optimization
REGION = "Bern"  # Options: 'Bern', 'CH' for Switzerland-wide

# --- EXPERIMENT GRID ---
EXPERIMENT_GRID = {
    # 'input_data': [
    #     {'code': 'S', 'path': 'path/to/segmented_lulc.tif'},
    #     {'code': 'N', 'path': None}
    # ],
    'max_restoration_fraction': [0.1, 0.3],
    'patch_size': [2, 3, 5],
    'seed': [100, 101, 102],
    #'approach': ['patch', 'pixel']
}

# Optimization parameters
OBJECTIVES = ["abiotic", "biotic", "cost"]
SAMPLE_FRACTION = 0.3  # Use 20% of eligible pixels
SAMPLE_SEED = 42  # Fixed seed for spatial sampling (to keep the spatial sample consistent)
POP_SIZE = 20
N_GENERATIONS = 30
N_JOBS = 12

# Patch approach configuration
# These will be overridden by the experiment grid
USE_PATCH_APPROACH = True
PATCH_SIZE = 10
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
BASE_OUTPUT_DIR = "multi_seed_results"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_optimization_with_seed(seed, initial_conditions, run_index, total_runs, use_patch_approach, patch_size, experiment_id, output_dir):
    """
    Run optimization with a specific random seed.
    
    Args:
        seed: Random seed to use
        initial_conditions: Initial conditions dictionary
        run_index: Current run number (0-indexed)
        total_runs: Total number of runs
        use_patch_approach: Boolean, whether to use patch-based optimization
        patch_size: Size of patches
        experiment_id: Unique ID for the experiment run
        output_dir: Directory to save optimization outputs
        
    Returns:
        dict: Results dictionary with seed information
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_index + 1}/{total_runs}: SEED = {seed}, EXPERIMENT = {experiment_id}")
    print(f"{'='*80}")
    
    try:
        results = run_optimization_instance(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            pop_size=POP_SIZE,
            n_generations=N_GENERATIONS,
            save_results=True,
            verbose=True,
            n_jobs=N_JOBS,
            use_repair=True,
            random_seed=seed,  # Pass seed to optimization
            use_patch_approach=use_patch_approach,
            patch_size=patch_size,
            patch_constraint_type=PATCH_CONSTRAINT_TYPE,
            pixel_tolerance=PIXEL_TOLERANCE,
            output_dir=output_dir
        )
        
        if results is not None:
            # Add seed information to results
            results['random_seed'] = seed
            results['run_index'] = run_index
            results['experiment_id'] = experiment_id
            print(f"\n✓ Seed {seed} for experiment {experiment_id} completed successfully!")
            return results
        else:
            print(f"\n✗ Seed {seed} for experiment {experiment_id} failed - no results returned.")
            return None
            
    except Exception as e:
        print(f"\n✗ Seed {seed} for experiment {experiment_id} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_summary_results(all_results, output_dir, experiment_id, experiment_params):
    """
    Save summary statistics across all seed runs for a given experiment.
    
    Args:
        all_results: Dictionary of results keyed by seed
        output_dir: Directory to save summary
        experiment_id: Unique ID for the experiment
        experiment_params: Dictionary of parameters for the experiment
    """
    summary = {
        'experiment_id': experiment_id,
        'experiment_params': experiment_params,
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
            'patch_size': experiment_params['patch_size'] if USE_PATCH_APPROACH else None,
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
    summary_file = os.path.join(output_dir, f'summary_{experiment_id}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Experiment summary saved to: {summary_file}")
    
    return summary


def main():
    """
    Main execution function for grid search restoration optimization.
    """
    print(f"{'='*80}")

    # Generate all combinations of experiments
    keys, values = zip(*EXPERIMENT_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total experiments to run: {len(experiments)}")
    
    all_experiment_results = {}

    for exp_idx, params in enumerate(experiments):
        max_restoration_fraction = params['max_restoration_fraction']
        patch_size = params['patch_size']
        seed = params['seed']

        use_patch_approach = USE_PATCH_APPROACH
        scenario_params['max_restoration_fraction'] = max_restoration_fraction
        scenario_params['ecosystem_label'] = ECOSYSTEM

        # Generate experiment ID
        id_code = f"MRF{int(max_restoration_fraction * 100):02d}"
        ps_code = patch_size
        se_code = seed
        ap_code = 'PA' if use_patch_approach else 'PI'
        
        experiment_id = f"ID{id_code}_PS{ps_code}_SE{se_code}_{ap_code}"

        print(f"\n--- Starting Experiment {exp_idx + 1}/{len(experiments)}: {experiment_id} ---")

        # Create output directory for this experiment
        output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_id)
        os.makedirs(output_dir, exist_ok=True)

        # Load initial conditions
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

        # Run optimization
        results = run_optimization_with_seed(
            seed, 
            initial_conditions, 
            exp_idx, 
            len(experiments), 
            use_patch_approach, 
            patch_size,
            experiment_id,
            output_dir
        )
        
        if results:
            all_experiment_results[experiment_id] = results
            # Save summary for this single run
            save_summary_results({seed: results}, output_dir, experiment_id, params)

    print(f"\n{'='*80}")
    print(f"✓ Grid search complete!")
    print(f"{'='*80}\n")
    
    return all_experiment_results


if __name__ == "__main__":
    results = main()
