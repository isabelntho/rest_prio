#!/usr/bin/env python
"""
Multi-Seed Optimization Analysis
==================================
Analyzes optimization results across multiple random seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import os
import json
import glob
import pickle
from scipy.spatial.distance import cdist
from sklearn.metrics import jaccard_score

# Custom unpickler to handle missing classes
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If the class is not found, return a dummy class
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            # Return a dummy class for missing classes
            return type(name, (), {})

def safe_pickle_load(file_path):
    """Safely load a pickle file, handling missing classes."""
    try:
        with open(file_path, 'rb') as f:
            return CustomUnpickler(f).load()
    except Exception as e:
        raise e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date code for the multi-seed run files to analyze (MMDD format)
# Files should be named: optimization_report_{ecosystem}_{FILE_DATE}_{seed}.json
#                   and: results_{ecosystem}_{FILE_DATE}_{seed}.pkl
FILE_DATE = "2602"  # Change this to analyze a different run

# File numbering range (e.g., 1-10 or 11-20)
FILE_START = 11  # Starting file number
FILE_END = 20    # Ending file number (inclusive)

# =============================================================================
# Section 1: Load Multi-Seed Summary and Display Metrics
# =============================================================================

print("="*80)
print("MULTI-SEED OPTIMIZATION ANALYSIS")
print("="*80)

# Load multi-seed summary file
summary_files = glob.glob("multi_seed_results/multi_seed_summary_*.json")

if not summary_files:
    print("✗ No multi-seed summary files found in multi_seed_results/")
    exit(1)

# Get the most recent summary file
latest_summary_file = max(summary_files, key=os.path.getmtime)

# Read and parse the summary
with open(latest_summary_file, 'r') as f:
    json_text = f.read().replace('NaN', 'null')
    multiseed_summary = json.loads(json_text)

print(f"✓ Loaded multi-seed summary for ecosystem: {multiseed_summary['ecosystem']}")
print(f"  Number of seeds: {multiseed_summary['n_seeds']}")
print(f"  Seeds: {', '.join(map(str, multiseed_summary['successful_seeds']))}")

# =============================================================================
# Section 1: Calculate Seed Metrics from Individual Results Files
# =============================================================================

print("\n" + "="*80)
print("SEED-LEVEL PERFORMANCE METRICS")
print("="*80)

ecosystem = multiseed_summary['ecosystem']
seeds = multiseed_summary['successful_seeds']

print("\nCalculating metrics from individual results files...")

# Load individual pickle files and calculate metrics
seed_metrics = []

for i in range(FILE_START, FILE_END + 1):
    pkl_file = f"results_{ecosystem}_{FILE_DATE}_{i}.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"  Warning: {pkl_file} not found")
        continue
    
    try:
        results = safe_pickle_load(pkl_file)
        
        # Extract objectives
        F = results.get('objectives') if isinstance(results, dict) else getattr(results, 'F', None)
        
        if F is not None and len(F) > 0:
            # Extract seed from results file if available, otherwise calculate from file number
            seed = results.get('random_seed', None) if isinstance(results, dict) else None
            if seed is None:
                # If not stored in results, calculate: assuming BASE_SEED=100 for files 1-10, 110 for files 11-20, etc.
                seed = 100 + (i - 1)  # File 1→100, File 11→110, File 20→119
            
            # Calculate metrics
            n_solutions = len(F)
            
            # Objectives are typically: [abiotic, biotic, cost]
            # Abiotic and biotic are negative counts (minimize negative = maximize positive)
            abiotic_values = -F[:, 0]  # Negate to get improvement
            biotic_values = -F[:, 1]   # Negate to get improvement
            cost_values = F[:, 2]       # Cost is to minimize
            
            # Calculate hypervolume
            ref_point = np.max(F, axis=0) * 1.1
            hv_indicator = HV(ref_point=ref_point)
            hypervolume = hv_indicator(F)
            
            seed_metrics.append({
                'seed': seed,
                'n_solutions': n_solutions,
                'hypervolume': hypervolume,
                'mean_abiotic_improvement': float(np.mean(abiotic_values)),
                'max_abiotic_improvement': float(np.max(abiotic_values)),
                'min_abiotic_improvement': float(np.min(abiotic_values)),
                'mean_biotic_improvement': float(np.mean(biotic_values)),
                'max_biotic_improvement': float(np.max(biotic_values)),
                'min_biotic_improvement': float(np.min(biotic_values)),
                'mean_cost': float(np.mean(cost_values)),
                'min_cost': float(np.min(cost_values)),
                'max_cost': float(np.max(cost_values))
            })
            
            print(f"  Processed {pkl_file} for seed {seed}: {n_solutions} solutions, HV={hypervolume:.6f}")
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")

if len(seed_metrics) > 0:
    seed_metrics_df = pd.DataFrame(seed_metrics)
    
    # Display summary table
    print("\nSummary Statistics Across Seeds:")
    summary_cols = ['seed', 'n_solutions', 'hypervolume', 
                   'mean_abiotic_improvement', 'mean_biotic_improvement', 'mean_cost',
                   'max_abiotic_improvement', 'max_biotic_improvement', 'min_cost']
    print(seed_metrics_df[summary_cols].round(4).to_string(index=False))
    
    # Plot hypervolume by seed
    print("\nGenerating seed metrics plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Hypervolume plot
    mean_hv = seed_metrics_df['hypervolume'].mean()
    std_hv = seed_metrics_df['hypervolume'].std()
    
    axes[0].bar(seed_metrics_df['seed'].astype(str), seed_metrics_df['hypervolume'], 
                color='steelblue', alpha=0.8)
    axes[0].axhline(mean_hv, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_hv:.4f}')
    axes[0].set_xlabel('Random Seed')
    axes[0].set_ylabel('Hypervolume')
    axes[0].set_title(f'Hypervolume by Seed (Mean: {mean_hv:.4f} | SD: {std_hv:.4f})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Objective ranges plot
    objective_ranges = pd.DataFrame({
        'seed': seed_metrics_df['seed'],
        'abiotic': seed_metrics_df['max_abiotic_improvement'] - seed_metrics_df['mean_abiotic_improvement'],
        'biotic': seed_metrics_df['max_biotic_improvement'] - seed_metrics_df['mean_biotic_improvement'],
        'cost': seed_metrics_df['mean_cost'] - seed_metrics_df['min_cost']
    })
    
    x = np.arange(len(objective_ranges))
    width = 0.25
    
    axes[1].bar(x - width, objective_ranges['abiotic'], width, label='Abiotic', alpha=0.8)
    axes[1].bar(x, objective_ranges['biotic'], width, label='Biotic', alpha=0.8)
    axes[1].bar(x + width, objective_ranges['cost'], width, label='Cost', alpha=0.8)
    axes[1].set_xlabel('Random Seed')
    axes[1].set_ylabel('Range')
    axes[1].set_title('Objective Range by Seed (Max - Mean)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(objective_ranges['seed'].astype(str))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Min/Max values plot
    for obj_type, obj_name in [('abiotic_improvement', 'Abiotic'), 
                                ('biotic_improvement', 'Biotic'), 
                                ('cost', 'Cost')]:
        max_col = f'max_{obj_type}'
        min_col = f'min_{obj_type}'
        if max_col in seed_metrics_df.columns and min_col in seed_metrics_df.columns:
            axes[2].plot(seed_metrics_df['seed'], seed_metrics_df[max_col], 
                        'o-', label=f'{obj_name} Max', linewidth=2)
            axes[2].plot(seed_metrics_df['seed'], seed_metrics_df[min_col], 
                        's--', label=f'{obj_name} Min', linewidth=2)
    
    axes[2].set_xlabel('Random Seed')
    axes[2].set_ylabel('Objective Value')
    axes[2].set_title('Minimum and Maximum Objective Values by Seed')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiseed_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved metrics plot to multiseed_metrics.png")
    plt.close(fig)
else:
    print("\n⚠ No seed metrics found in summary - skipping metrics plot")
    plt.show()

# =============================================================================
# Section 1.5: Check for Duplicates Within Each Seed
# =============================================================================

print("\n" + "="*80)
print("DUPLICATE DETECTION WITHIN SEEDS")
print("="*80)

print("\nChecking for duplicate solutions within each seed...")

duplicate_stats = []

# Load pickle files for each seed
for i in range(FILE_START, FILE_END + 1):
    pkl_file = f"results_{ecosystem}_{FILE_DATE}_{i}.pkl"
    
    if not os.path.exists(pkl_file):
        continue
    
    try:
        results = safe_pickle_load(pkl_file)
        
        # Extract X and F
        X = results.get('decisions') if isinstance(results, dict) else getattr(results, 'X', None)
        F = results.get('objectives') if isinstance(results, dict) else getattr(results, 'F', None)
        
        if X is not None and F is not None:
            # Extract seed from results file
            seed = results.get('random_seed', None) if isinstance(results, dict) else None
            if seed is None:
                seed = 100 + (i - 1)  # Fallback calculation
            
            # Count total solutions
            n_total = len(X)
            
            # Count unique decision vectors (exact comparison for binary)
            unique_X = np.unique(X, axis=0)
            n_unique_decisions = len(unique_X)
            
            # Count unique objective vectors (with tolerance for floats)
            # Round to 6 decimal places for comparison
            F_rounded = np.round(F, decimals=6)
            unique_F = np.unique(F_rounded, axis=0)
            n_unique_objectives = len(unique_F)
            
            # Calculate duplicate percentages
            duplicate_decisions_pct = (1 - n_unique_decisions / n_total) * 100
            duplicate_objectives_pct = (1 - n_unique_objectives / n_total) * 100
            
            duplicate_stats.append({
                'seed': seed,
                'n_total': n_total,
                'n_unique_decisions': n_unique_decisions,
                'n_duplicate_decisions': n_total - n_unique_decisions,
                'duplicate_decisions_pct': duplicate_decisions_pct,
                'n_unique_objectives': n_unique_objectives,
                'n_duplicate_objectives': n_total - n_unique_objectives,
                'duplicate_objectives_pct': duplicate_objectives_pct
            })
            
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")

if duplicate_stats:
    duplicate_df = pd.DataFrame(duplicate_stats)
    
    # Check if any duplicates exist
    total_decision_duplicates = duplicate_df['n_duplicate_decisions'].sum()
    total_objective_duplicates = duplicate_df['n_duplicate_objectives'].sum()
    
    if total_decision_duplicates > 0 or total_objective_duplicates > 0:
        print("\n⚠ DUPLICATES DETECTED:")
        print(duplicate_df[['seed', 'n_total', 'n_unique_decisions', 'n_duplicate_decisions', 
                            'duplicate_decisions_pct', 'n_unique_objectives', 'n_duplicate_objectives',
                            'duplicate_objectives_pct']].to_string(index=False))
        
        # Summary statistics
        print(f"\nSummary Across All Seeds:")
        print(f"  Mean duplicate decisions: {duplicate_df['duplicate_decisions_pct'].mean():.2f}%")
        print(f"  Mean duplicate objectives: {duplicate_df['duplicate_objectives_pct'].mean():.2f}%")
        print(f"  Max duplicate decisions: {duplicate_df['duplicate_decisions_pct'].max():.2f}% (seed {duplicate_df.loc[duplicate_df['duplicate_decisions_pct'].idxmax(), 'seed']})")
        print(f"  Max duplicate objectives: {duplicate_df['duplicate_objectives_pct'].max():.2f}% (seed {duplicate_df.loc[duplicate_df['duplicate_objectives_pct'].idxmax(), 'seed']})")
    else:
        print("\n✓ No duplicates detected in any seed")
else:
    print("No data available for duplicate analysis")

# =============================================================================
# Section 2: Combined Reference Set Analysis
# =============================================================================

print("\n" + "="*80)
print("COMBINED REFERENCE SET ANALYSIS")
print("="*80)

print(f"\nAnalyzing combined reference set for {ecosystem} ecosystem")
print(f"Seeds: {seeds}")

# Load all solutions from pickle files
all_solutions = []
seed_labels = []

# Use the configured file date pattern
pkl_files = []
for i in range(FILE_START, FILE_END + 1):
    pkl_file = f"results_{ecosystem}_{FILE_DATE}_{i}.pkl"
    if os.path.exists(pkl_file):
        pkl_files.append(pkl_file)
    else:
        print(f"  Warning: {pkl_file} not found")

print(f"  Found {len(pkl_files)} pickle files from {FILE_DATE} run")

for idx, pkl_file in enumerate(pkl_files):
    try:
        results = safe_pickle_load(pkl_file)
        
        # Debug: print structure for first file
        if idx == 0:
            print(f"\n  Results is a {type(results).__name__}")
            if isinstance(results, dict):
                print(f"  Dictionary keys: {list(results.keys())}")
        
        # Extract objectives from dictionary
        objectives = None
        if isinstance(results, dict):
            # Results is a dictionary - use key access
            objectives = results.get('objectives')
        elif hasattr(results, 'F'):
            # Results is an object - use attribute access
            objectives = results.F
        
        if objectives is not None:
            # Extract seed from results file
            seed = results.get('random_seed', None) if isinstance(results, dict) else None
            if seed is None:
                # Calculate from file position: FILE_START + idx gives current file number
                file_num = FILE_START + idx
                seed = 100 + (file_num - 1)
            
            all_solutions.append(objectives)
            seed_labels.extend([seed] * len(objectives))
            print(f"  Loaded {pkl_file} for seed {seed} ({len(objectives)} solutions)")
        else:
            print(f"  Warning: Could not find objective data in {pkl_file}")
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")

if len(all_solutions) > 0:
    # Combine all solutions
    combined_F = np.vstack(all_solutions)
    seed_labels = np.array(seed_labels)
    
    print(f"\nTotal solutions across all seeds: {len(combined_F)}")
    
    # Find non-dominated solutions in the combined set
    nds = NonDominatedSorting()
    non_dominated_indices = nds.do(combined_F, only_non_dominated_front=True)
    
    combined_pareto = combined_F[non_dominated_indices]
    pareto_seed_labels = seed_labels[non_dominated_indices]
    
    print(f"Non-dominated solutions in combined set: {len(combined_pareto)}")
    
    # Calculate contribution of each seed to the combined Pareto front
    seed_contributions = pd.Series(pareto_seed_labels).value_counts().sort_index()
    contribution_pct = (seed_contributions / len(combined_pareto) * 100).round(2)
    
    print("\nSeed contributions to combined Pareto front:")
    for seed, count in seed_contributions.items():
        pct = contribution_pct[seed]
        print(f"  Seed {seed}: {count} solutions ({pct}%)")
    
    # Create contribution dataframe
    contribution_df = pd.DataFrame({
        'seed': seed_contributions.index,
        'n_solutions': seed_contributions.values,
        'percentage': contribution_pct.values
    })
    
    # Plot seed contributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of contributions
    axes[0].bar(contribution_df['seed'].astype(str), contribution_df['n_solutions'], 
                color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Random Seed')
    axes[0].set_ylabel('Number of Solutions')
    axes[0].set_title('Contribution to Combined Pareto Front')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(contribution_df['n_solutions'], labels=contribution_df['seed'].astype(str),
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Percentage Contribution by Seed')
    
    plt.tight_layout()
    plt.savefig('seed_contributions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved contribution plot to seed_contributions.png")
    plt.close(fig)
    
    # Calculate hypervolume of combined Pareto front
    ref_point = np.max(combined_F, axis=0) * 1.1  # 10% beyond worst point
    hv_indicator = HV(ref_point=ref_point)
    combined_hv = hv_indicator(combined_pareto)
    
    print(f"\nCombined Pareto front hypervolume: {combined_hv:.6f}")
    
    # Compare to individual seed hypervolumes
    if 'seed_metrics' in multiseed_summary:
        individual_hvs = [m['hypervolume'] for m in multiseed_summary['seed_metrics'] 
                         if m.get('hypervolume') is not None]
        if individual_hvs:
            mean_individual_hv = np.mean(individual_hvs)
            improvement = ((combined_hv - mean_individual_hv) / mean_individual_hv * 100)
            print(f"Mean individual seed HV: {mean_individual_hv:.6f}")
            print(f"Improvement from combining: {improvement:.2f}%")
    
    # =============================================================================
    # Section 2.5: Objective Space Consistency Across Seeds
    # =============================================================================
    
    print("\n" + "-"*80)
    print("OBJECTIVE SPACE CONSISTENCY ANALYSIS")
    print("-"*80)
    
    print("\nComputing objective space quality metrics for each seed...")
    
    # Calculate ideal and nadir points for normalization
    ideal_point = np.min(combined_pareto, axis=0)
    nadir_point = np.max(combined_pareto, axis=0)
    
    print(f"Ideal point (min): {ideal_point}")
    print(f"Nadir point (max): {nadir_point}")
    print(f"Objective ranges: {nadir_point - ideal_point}")
    
    # Normalize combined Pareto front
    range_point = nadir_point - ideal_point
    # Avoid division by zero
    range_point = np.where(range_point == 0, 1.0, range_point)
    combined_pareto_norm = (combined_pareto - ideal_point) / range_point
    
    # For each seed, compute IGD to the combined (pooled) Pareto front
    from scipy.spatial.distance import cdist as sp_cdist
    
    consistency_metrics = []
    
    for seed in np.unique(seed_labels):
        # Get this seed's solutions
        seed_mask = seed_labels == seed
        seed_F = combined_F[seed_mask]
        
        # Normalize this seed's solutions using same ideal/nadir
        seed_F_norm = (seed_F - ideal_point) / range_point
        
        # Compute IGD: average distance from combined Pareto to this seed's solutions
        # Using normalized objectives to ensure all objectives contribute equally
        distances = sp_cdist(combined_pareto_norm, seed_F_norm, metric='euclidean')
        min_distances = np.min(distances, axis=1)  # For each point in combined, dist to nearest in seed
        igd = np.mean(min_distances)
        
        # Per-objective ranges
        obj_metrics = {
            'seed': seed,
            'n_solutions': len(seed_F),
            'igd_to_combined': igd
        }
        
        for obj_idx, obj_name in enumerate(['abiotic', 'biotic', 'cost']):
            obj_metrics[f'{obj_name}_min'] = float(np.min(seed_F[:, obj_idx]))
            obj_metrics[f'{obj_name}_max'] = float(np.max(seed_F[:, obj_idx]))
            obj_metrics[f'{obj_name}_range'] = float(np.max(seed_F[:, obj_idx]) - np.min(seed_F[:, obj_idx]))
        
        consistency_metrics.append(obj_metrics)
    
    consistency_df = pd.DataFrame(consistency_metrics)
    
    print("\nObjective Space Consistency by Seed:")
    print(consistency_df[['seed', 'n_solutions', 'igd_to_combined', 
                          'abiotic_min', 'abiotic_max', 'biotic_min', 'biotic_max',
                          'cost_min', 'cost_max']].to_string(index=False))
    
    # Summary statistics
    print(f"\nIGD Statistics:")
    print(f"  Mean IGD to combined front: {consistency_df['igd_to_combined'].mean():.6f}")
    print(f"  Std IGD: {consistency_df['igd_to_combined'].std():.6f}")
    print(f"  Min IGD: {consistency_df['igd_to_combined'].min():.6f} (seed {consistency_df.loc[consistency_df['igd_to_combined'].idxmin(), 'seed']})")
    print(f"  Max IGD: {consistency_df['igd_to_combined'].max():.6f} (seed {consistency_df.loc[consistency_df['igd_to_combined'].idxmax(), 'seed']})")
    
    # Check range consistency across seeds
    print(f"\nObjective Range Variability Across Seeds:")
    for obj_name in ['abiotic', 'biotic', 'cost']:
        range_col = f'{obj_name}_range'
        mean_range = consistency_df[range_col].mean()
        std_range = consistency_df[range_col].std()
        cv = (std_range / mean_range * 100) if mean_range > 0 else 0
        print(f"  {obj_name.capitalize()}: mean={mean_range:.4f}, std={std_range:.4f}, CV={cv:.2f}%")
    
    # Plot IGD by seed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(consistency_df['seed'].astype(str), consistency_df['igd_to_combined'], 
                color='coral', alpha=0.8)
    axes[0].axhline(consistency_df['igd_to_combined'].mean(), color='red', linestyle='--', 
                    linewidth=2, label='Mean IGD')
    axes[0].set_xlabel('Seed')
    axes[0].set_ylabel('IGD to Combined Pareto Front')
    axes[0].set_title('Objective Space Quality by Seed')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot objective ranges
    x = np.arange(len(consistency_df))
    width = 0.25
    
    axes[1].bar(x - width, consistency_df['abiotic_range'], width, label='Abiotic', alpha=0.8)
    axes[1].bar(x, consistency_df['biotic_range'], width, label='Biotic', alpha=0.8)
    axes[1].bar(x + width, consistency_df['cost_range'], width, label='Cost', alpha=0.8)
    axes[1].set_xlabel('Seed')
    axes[1].set_ylabel('Objective Range')
    axes[1].set_title('Objective Range Coverage by Seed')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(consistency_df['seed'].astype(str))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('objective_consistency.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved objective consistency plot to objective_consistency.png")
    plt.close(fig)
    
else:
    print("No solutions found to analyze")

# =============================================================================
# Section 3: Spatial Agreement Analysis
# =============================================================================

print("\n" + "="*80)
print("SPATIAL AGREEMENT ANALYSIS")
print("="*80)

print("\nAnalyzing spatial agreement of solutions...")

# Load pickle files to get decision variable data
pkl_files = {}

# Use the configured file date pattern
matching_files = []
for i in range(FILE_START, FILE_END + 1):
    pkl_file = f"results_{ecosystem}_{FILE_DATE}_{i}.pkl"
    if os.path.exists(pkl_file):
        matching_files.append(pkl_file)
    else:
        print(f"  Warning: {pkl_file} not found")

print(f"  Found {len(matching_files)} pickle files from {FILE_DATE} run")

# Load the files
for idx, pkl_file in enumerate(matching_files):
    # Calculate seed from file number
    file_num = FILE_START + idx
    seed = 100 + (file_num - 1)  # Default calculation
    
    try:
        results = safe_pickle_load(pkl_file)
        
        # Extract X and F from dictionary or object
        X = results.get('decisions') if isinstance(results, dict) else getattr(results, 'X', None)
        F = results.get('objectives') if isinstance(results, dict) else getattr(results, 'F', None)
        
        if X is not None and F is not None:
            pkl_files[seed] = {
                'X': X,  # Decision variables
                'F': F,  # Objective values
                'file': pkl_file
            }
            print(f"  Loaded {pkl_file} for seed {seed} ({len(F)} solutions)")
        else:
            print(f"  Warning: Could not extract X or F from {pkl_file}")
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")

if len(pkl_files) > 0:
    # Combine all solutions from all seeds
    all_X = []
    all_F = []
    all_seed_labels = []
    
    for seed, data in pkl_files.items():
        all_X.append(data['X'])
        all_F.append(data['F'])
        all_seed_labels.extend([seed] * len(data['X']))
    
    combined_X = np.vstack(all_X)
    combined_F = np.vstack(all_F)
    all_seed_labels = np.array(all_seed_labels)
    
    # Find non-dominated solutions
    nds = NonDominatedSorting()
    pareto_indices = nds.do(combined_F, only_non_dominated_front=True)
    
    pareto_X = combined_X[pareto_indices]
    pareto_F = combined_F[pareto_indices]
    pareto_seeds = all_seed_labels[pareto_indices]
    
    print(f"\nAnalyzing spatial agreement for {len(pareto_X)} Pareto solutions...")
    
    # Calculate pairwise distances in objective space
    obj_distances = cdist(pareto_F, pareto_F, metric='euclidean')
    
    # Normalize distances to [0, 1]
    obj_distances_norm = obj_distances / np.max(obj_distances)
    
    # For each solution, find K nearest neighbors in objective space from DIFFERENT seeds
    K = 5
    spatial_agreements = []
    
    for i in range(len(pareto_X)):
        # Get solutions from different seeds only
        different_seed_mask = pareto_seeds != pareto_seeds[i]
        different_seed_indices = np.where(different_seed_mask)[0]
        
        if len(different_seed_indices) == 0:
            # Skip if no solutions from other seeds (shouldn't happen with multiple seeds)
            continue
        
        # Get distances only to solutions from different seeds
        distances_to_different = obj_distances[i][different_seed_indices]
        
        # Find K nearest neighbors from different seeds
        k_actual = min(K, len(different_seed_indices))
        nearest_in_different = np.argsort(distances_to_different)[:k_actual]
        nearest_indices = different_seed_indices[nearest_in_different]
        
        # Calculate Jaccard similarity with each neighbor
        jaccard_scores = []
        for j in nearest_indices:
            # Jaccard similarity: intersection / union for binary vectors
            intersection = np.sum(pareto_X[i] & pareto_X[j])
            union = np.sum(pareto_X[i] | pareto_X[j])
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
        
        mean_jaccard = np.mean(jaccard_scores)
        spatial_agreements.append({
            'solution_id': i,
            'seed': pareto_seeds[i],
            'mean_jaccard': mean_jaccard,
            'min_jaccard': np.min(jaccard_scores),
            'max_jaccard': np.max(jaccard_scores),
            'mean_obj_distance': np.mean(obj_distances_norm[i][nearest_indices]),
            'n_neighbors': len(jaccard_scores)
        })
    
    agreement_df = pd.DataFrame(spatial_agreements)
    
    print(f"\nSpatial Agreement Statistics (Across Seeds Only):")
    print(f"  Mean Jaccard similarity: {agreement_df['mean_jaccard'].mean():.4f}")
    print(f"  Std Jaccard similarity: {agreement_df['mean_jaccard'].std():.4f}")
    print(f"  Min Jaccard similarity: {agreement_df['mean_jaccard'].min():.4f}")
    print(f"  Max Jaccard similarity: {agreement_df['mean_jaccard'].max():.4f}")
    print(f"  Mean neighbors per solution: {agreement_df['n_neighbors'].mean():.1f}")
    
    # Plot spatial agreement vs objective distance
    fig, axes = plt.subplots(1, 1, figsize=(14, 5))
    
    # Scatter: Jaccard vs objective distance
    scatter = axes.scatter(agreement_df['mean_obj_distance'], 
                              agreement_df['mean_jaccard'],
                              c=agreement_df['seed'], 
                              cmap='viridis', 
                              alpha=0.6, s=50)
    axes.set_xlabel('Mean Objective Distance to Neighbors')
    axes.set_ylabel('Mean Jaccard Similarity')
    axes.set_title('Spatial Agreement vs Objective Space Proximity')
    axes.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes, label='Seed')

    
    plt.tight_layout()
    plt.savefig('spatial_agreement_forest.png', dpi=150, bbox_inches='tight')
    print("✓ Saved spatial agreement plot to spatial_agreement_forest.png")
    plt.close(fig)
    
    # Analyze agreement by seed
    seed_agreement = agreement_df.groupby('seed')['mean_jaccard'].agg(['mean', 'std', 'count'])
    print("\nSpatial Agreement by Seed:")
    print(seed_agreement)
    
    # =============================================================================
    # Section 3.5: Within-Seed vs Across-Seed Spatial Similarity
    # =============================================================================
    
    print("\n" + "-"*80)
    print("WITHIN-SEED VS ACROSS-SEED SPATIAL SIMILARITY")
    print("-"*80)
    
    print("\nComputing Jaccard similarity distributions...")
    
    # Within-seed similarities: between pairs of solutions from the same seed
    within_seed_jaccard = []
    
    for seed in np.unique(pareto_seeds):
        seed_mask = pareto_seeds == seed
        seed_solutions = pareto_X[seed_mask]
        
        if len(seed_solutions) < 2:
            continue
        
        # Sample pairs within this seed (limit to avoid combinatorial explosion)
        n_pairs = min(100, len(seed_solutions) * (len(seed_solutions) - 1) // 2)
        
        for _ in range(n_pairs):
            i, j = np.random.choice(len(seed_solutions), size=2, replace=False)
            intersection = np.sum(seed_solutions[i] & seed_solutions[j])
            union = np.sum(seed_solutions[i] | seed_solutions[j])
            jaccard = intersection / union if union > 0 else 0.0
            within_seed_jaccard.append(jaccard)
    
    # Across-seed similarities: between objective-matched neighbors from different seeds
    # Already computed in spatial_agreements list above
    across_seed_jaccard = agreement_df['mean_jaccard'].values
    
    print(f"\nWithin-Seed Similarity (same seed, random pairs):")
    print(f"  N pairs: {len(within_seed_jaccard)}")
    print(f"  Mean Jaccard: {np.mean(within_seed_jaccard):.4f}")
    print(f"  Std Jaccard: {np.std(within_seed_jaccard):.4f}")
    print(f"  Median Jaccard: {np.median(within_seed_jaccard):.4f}")
    print(f"  Min Jaccard: {np.min(within_seed_jaccard):.4f}")
    print(f"  Max Jaccard: {np.max(within_seed_jaccard):.4f}")
    
    print(f"\nAcross-Seed Similarity (objective-matched from different seeds):")
    print(f"  N pairs: {len(across_seed_jaccard)}")
    print(f"  Mean Jaccard: {np.mean(across_seed_jaccard):.4f}")
    print(f"  Std Jaccard: {np.std(across_seed_jaccard):.4f}")
    print(f"  Median Jaccard: {np.median(across_seed_jaccard):.4f}")
    print(f"  Min Jaccard: {np.min(across_seed_jaccard):.4f}")
    print(f"  Max Jaccard: {np.max(across_seed_jaccard):.4f}")
    
    # Statistical comparison
    from scipy import stats
    if len(within_seed_jaccard) > 0 and len(across_seed_jaccard) > 0:
        # Mann-Whitney U test (non-parametric)
        statistic, pvalue = stats.mannwhitneyu(within_seed_jaccard, across_seed_jaccard, alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {statistic:.2f}")
        print(f"  P-value: {pvalue:.6f}")
        if pvalue < 0.05:
            print(f"  → Distributions are significantly different (p < 0.05)")
        else:
            print(f"  → Distributions are not significantly different (p >= 0.05)")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograms
    axes[0].hist(within_seed_jaccard, bins=30, alpha=0.6, label='Within-Seed', 
                 color='steelblue', edgecolor='black', density=True)
    axes[0].hist(across_seed_jaccard, bins=30, alpha=0.6, label='Across-Seed', 
                 color='coral', edgecolor='black', density=True)
    axes[0].axvline(np.mean(within_seed_jaccard), color='steelblue', linestyle='--', 
                    linewidth=2, label=f'Within Mean: {np.mean(within_seed_jaccard):.3f}')
    axes[0].axvline(np.mean(across_seed_jaccard), color='coral', linestyle='--', 
                    linewidth=2, label=f'Across Mean: {np.mean(across_seed_jaccard):.3f}')
    axes[0].set_xlabel('Jaccard Similarity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Within-Seed vs Across-Seed Spatial Similarity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plots
    data_to_plot = [within_seed_jaccard, across_seed_jaccard]
    bp = axes[1].boxplot(data_to_plot, labels=['Within-Seed', 'Across-Seed'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    axes[1].set_ylabel('Jaccard Similarity')
    axes[1].set_title('Spatial Similarity Distribution Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('within_vs_across_seed_similarity.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved similarity comparison plot to within_vs_across_seed_similarity.png")
    plt.close(fig)
    
    # Calculate pairwise overlap for a few example solution pairs (cross-seed only)
    print("\n### Example Cross-Seed Pairwise Comparisons:")
    # Find 3 pairs of solutions from different seeds that are close in objective space
    close_pairs = []
    for i in range(min(20, len(pareto_X))):
        # Get solutions from different seeds
        different_seed_mask = pareto_seeds != pareto_seeds[i]
        different_seed_indices = np.where(different_seed_mask)[0]
        
        if len(different_seed_indices) > 0:
            # Find nearest neighbor from different seed
            distances_to_different = obj_distances_norm[i][different_seed_indices]
            nearest_idx_in_different = np.argmin(distances_to_different)
            nearest_idx = different_seed_indices[nearest_idx_in_different]
            
            if distances_to_different[nearest_idx_in_different] < 0.1:  # Only if very close
                close_pairs.append((i, nearest_idx))
                if len(close_pairs) >= 3:
                    break
    
    for idx, (i, j) in enumerate(close_pairs):
        overlap = np.sum(pareto_X[i] & pareto_X[j])
        union = np.sum(pareto_X[i] | pareto_X[j])
        jaccard = overlap / union if union > 0 else 0.0
        obj_dist = obj_distances_norm[i][j]
        
        print(f"\nPair {idx+1}: Solution {i} (seed {pareto_seeds[i]}) vs Solution {j} (seed {pareto_seeds[j]})")
        print(f"  Objective distance: {obj_dist:.4f}")
        print(f"  Jaccard similarity: {jaccard:.4f}")
        print(f"  Overlapping pixels: {overlap}")
        print(f"  Union pixels: {union}")
        print(f"  Pixels in sol {i}: {np.sum(pareto_X[i])}")
        print(f"  Pixels in sol {j}: {np.sum(pareto_X[j])}")
else:
    print("No pickle files could be loaded for spatial analysis")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
