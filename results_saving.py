"""
Results Saving Utilities
========================
Functions for saving optimization results, parameter summaries, and combined scenario results.

Created: January 2026
"""

import os
import json
import pickle
from datetime import datetime

def save_parameter_summary(output_dir=".", n_samples_per_param=3, random_seed=42, verbose=True):
    """
    Save a summary of parameter ranges and sampled values.
    
    Args:
        output_dir: Directory to save files
        n_samples_per_param: Number of samples per continuous parameter
        random_seed: Random seed used for sampling
        verbose: Print save status
    """
    from resto_anom import define_scenario_parameters, sample_scenario_parameters
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get parameter ranges and sampled values
    param_ranges = define_scenario_parameters()
    sampled_combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
    
    # Create parameter summary
    param_summary = {
        'sampling_info': {
            'n_samples_per_param': n_samples_per_param,
            'random_seed': random_seed,
            'total_scenarios': len(sampled_combinations),
            'timestamp': datetime.now().isoformat()
        },
        'parameter_ranges': {},
        'sampled_values': {},
        'scenarios': []
    }
    
    # Process parameter ranges
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            param_summary['parameter_ranges'][param_name] = {
                'type': 'continuous',
                'min': param_range[0],
                'max': param_range[1]
            }
        else:
            param_summary['parameter_ranges'][param_name] = {
                'type': 'categorical',
                'values': param_range
            }
    
    # Extract unique sampled values for each parameter
    for param_name in param_ranges.keys():
        values = [combo[param_name] for combo in sampled_combinations]
        unique_values = sorted(list(set(values)))
        param_summary['sampled_values'][param_name] = unique_values
    
    # Add all scenario combinations
    for i, combo in enumerate(sampled_combinations):
        param_summary['scenarios'].append({
            'scenario_id': i,
            'parameters': combo
        })
    
    # Save to JSON
    summary_filename = os.path.join(output_dir, f"parameter_summary_{timestamp}.json")
    with open(summary_filename, 'w') as f:
        json.dump(param_summary, f, indent=2)
    
    if verbose:
        print(f"✓ Parameter summary saved to: {summary_filename}")
        print(f"   Total scenarios: {len(sampled_combinations)}")
        print(f"   Continuous parameters: {sum(1 for r in param_ranges.values() if isinstance(r, tuple))}")
        print(f"   Categorical parameters: {sum(1 for r in param_ranges.values() if not isinstance(r, tuple))}")
    
    return summary_filename

def save_scenario_results(results, output_dir=".", verbose=True):
    """
    Save single scenario optimization results to files.
    
    Args:
        results: Results dictionary from single scenario optimization
        output_dir: Directory to save files
        verbose: Print save status
        
    Returns:
        tuple: (pickle_filename, summary_filename)
    """
    if verbose:
        print("\nSaving scenario results...")
    
    timestamp = datetime.now().strftime('%d%m')
    
    # Get ecosystem information for filename
    ecosystem = results.get('initial_conditions', {}).get('ecosystem', 'all')
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    # Save complete results (pickle format)
    x = 1
    while True:
        results_filename = os.path.join(
            output_dir,
            f"results{ecosystem_suffix}_{timestamp}_{x}.pkl"
        )
        if not os.path.exists(results_filename):
            break
        x += 1
    print(f"  Saving to: {results_filename}")
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary (JSON format)
    summary_filename = os.path.join(output_dir, f"summary{ecosystem_suffix}_{timestamp}.json")
    
    objectives = results['objectives']
    
    # Get objective names from the problem (avoid circular import)
    # We can get them from the results dict directly
    objective_names = results['objective_names']
    
    summary = {
        'ecosystem': ecosystem,
        'scenario_params': results['scenario_params'],
        'optimization_info': {
            'n_solutions': results['n_solutions'],
            'n_pixels': results['problem_info']['n_pixels'],
            'timestamp': results['algorithm_info']['timestamp'],
            'objectives_used': objective_names
        },
        'objective_statistics': {}
    }
    
    # Add statistics for each objective that was used
    for i, obj_name in enumerate(objective_names):
        summary['objective_statistics'][obj_name] = {
            'min': float(objectives[:, i].min()),
            'max': float(objectives[:, i].max()),
            'mean': float(objectives[:, i].mean())
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"✓ Complete results saved to: {results_filename}")
        print(f"✓ Summary saved to: {summary_filename}")
    
    return results_filename, summary_filename

def save_combined_results(combined_results, output_dir=".", verbose=True):
    """
    Save combined multi-scenario results.
    
    Args:
        combined_results: Combined results from all scenarios
        output_dir: Directory to save files
        verbose: Print save status
    """
    if verbose:
        print(f"\nSaving combined scenario results...")
    
    timestamp = datetime.now().strftime('%d%m')
    
    # Get ecosystem information from first scenario (should be same for all)
    first_scenario_results = next(iter(combined_results['scenarios'].values()))
    ecosystem = first_scenario_results.get('initial_conditions', {}).get('ecosystem', 'all')
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    x = 1
    while True:
        results_filename = os.path.join(output_dir, f"results_{timestamp}_{x}.pkl")
        summary_filename = os.path.join(output_dir, f"results_sc_{timestamp}_{x}.json")
        if not os.path.exists(results_filename):
            break
        x += 1
    print(f"  Saving combined results to: {results_filename}")
    with open(results_filename, 'wb') as f:
        pickle.dump(combined_results, f)
    
    # Convert parameter ranges to JSON-serializable format
    param_ranges = combined_results.get('scenario_parameters', {})
    json_param_ranges = {}
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            json_param_ranges[param_name] = {'type': 'continuous', 'min': param_range[0], 'max': param_range[1]}
        else:
            json_param_ranges[param_name] = {'type': 'categorical', 'values': param_range}
    
    summary = {
        'ecosystem': ecosystem,
        'combined_info': {
            'n_scenarios_run': combined_results['n_scenarios_run'],
            'n_scenarios_total': combined_results['n_scenarios_total'],
            'parameter_ranges': json_param_ranges,
            'sampling_info': {
                'n_samples_per_param': combined_results.get('n_samples_per_param', 3),
                'random_seed': combined_results.get('random_seed', 42)
            },
            'timestamp': combined_results['algorithm_info']['timestamp']
        },
        'scenarios': {}
    }
    
    # Add summary for each scenario
    for scenario_id, scenario_results in combined_results['scenarios'].items():
        objectives = scenario_results['objectives']
        
        # Get objective names from this scenario (avoid circular import)
        objective_names = scenario_results['objective_names']
        
        objective_ranges = {}
        for i, obj_name in enumerate(objective_names):
            objective_ranges[obj_name] = [float(objectives[:, i].min()), float(objectives[:, i].max())]
        
        summary['scenarios'][f'scenario_{scenario_id}'] = {
            'params': scenario_results['scenario_params'],
            'n_solutions': scenario_results['n_solutions'],
            'objectives_used': objective_names,
            'objective_ranges': objective_ranges
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"✓ Combined results saved to: {results_filename}")
        print(f"✓ Combined summary saved to: {summary_filename}")