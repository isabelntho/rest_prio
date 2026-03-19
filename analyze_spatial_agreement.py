"""
Spatial Agreement Analysis
===========================
Analyzes spatial agreement (Jaccard similarity) vs objective space proximity
for multi-seed optimization results.

This script:
1. Loads optimization results from multiple seeds
2. Identifies Pareto-optimal solutions across all seeds
3. For each solution, finds K nearest neighbors in objective space (from different seeds)
4. Calculates Jaccard similarity (spatial agreement) between solutions
5. Saves results for plotting

Created: March 2026
"""

import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from patch_approach import convert_patch_results_to_pixel_decisions


# Custom unpickler to handle missing classes
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return type(name, (), {})


def safe_pickle_load(file_path):
    """Safely load a pickle file, handling missing classes."""
    with open(file_path, 'rb') as f:
        return CustomUnpickler(f).load()


def analyze_spatial_agreement(ecosystem, file_date, file_start, file_end, 
                              k_neighbors=5, output_file=None, use_patch_approach=False):
    """
    Analyze spatial agreement across multi-seed optimization results.
    
    Parameters
    ----------
    ecosystem : str
        Ecosystem type ('forest', 'agricultural', 'grassland')
    file_date : str
        Date code for files (MMDD format)
    file_start : int
        Starting file number
    file_end : int
        Ending file number (inclusive)
    k_neighbors : int
        Number of nearest neighbors to consider
    output_file : str, optional
        Path to save results CSV. If None, uses default naming
    use_patch_approach : bool
        Whether results are patch-based (requires conversion)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with spatial agreement metrics
    """
    
    # Load individual pickle files
    pkl_files = {}
    matching_files = []
    
    for i in range(file_start, file_end + 1):
        pkl_file = f"results_{ecosystem}_{file_date}_{i}.pkl"
        if os.path.exists(pkl_file):
            matching_files.append(pkl_file)
        else:
            print(f"  Warning: {pkl_file} not found")
    
    if not matching_files:
        raise FileNotFoundError(f"No matching files found for pattern: results_{ecosystem}_{file_date}_*.pkl")
    
    print(f"Found {len(matching_files)} result files\n")
    
    # Load results and extract decisions/objectives
    for idx, pkl_file in enumerate(matching_files):
        file_num = file_start + idx
        seed = 100 + (file_num - 1)
        
        try:
            results = safe_pickle_load(pkl_file)
            
            # Check if results are patch-based
            has_patch_mappings = 'patch_mappings' in results if isinstance(results, dict) else False
            
            # Handle patch-based results
            if use_patch_approach:
                if has_patch_mappings:
                    print(f"  Converting patch-based results for seed {seed}...")
                    results = convert_patch_results_to_pixel_decisions(
                        results, 
                        results['patch_mappings']
                    )
                else:
                    print(f"  ✗ Warning: use_patch_approach=True but {pkl_file} has no patch_mappings!")
                    print(f"     Skipping this file. Check if it was generated with USE_PATCH_APPROACH=True")
                    continue
            else:
                # If analyzing pixel-based but file contains patches
                if has_patch_mappings:
                    print(f"  ✗ Warning: use_patch_approach=False but {pkl_file} contains patch_mappings!")
                    print(f"     This appears to be a patch-based result. Set use_patch_approach=True or use different files.")
                    continue
            
            # Extract X and F
            X = results.get('decisions') if isinstance(results, dict) else getattr(results, 'X', None)
            F = results.get('objectives') if isinstance(results, dict) else getattr(results, 'F', None)
            
            if X is not None and F is not None:
                pkl_files[seed] = {
                    'X': X,
                    'F': F,
                    'file': pkl_file
                }
                is_patch = 'patch_mappings' in results if isinstance(results, dict) else False
                approach_str = " (patch-based)" if is_patch else " (pixel-based)"
                print(f"  ✓ Loaded seed {seed}: {len(F)} solutions, decision shape: {X.shape}{approach_str}")
            else:
                print(f"  ✗ Warning: Could not extract X or F from {pkl_file}")
        except Exception as e:
            print(f"  ✗ Error loading {pkl_file}: {e}")
    
    if not pkl_files:
        raise ValueError("No valid result files could be loaded")
    
    # Validate that all decision arrays have the same shape
    print(f"\nValidating decision array dimensions...")
    decision_shapes = {seed: data['X'].shape for seed, data in pkl_files.items()}
    unique_shapes = set(decision_shapes.values())
    
    if len(unique_shapes) > 1:
        print(f"\n✗ ERROR: Mismatched decision array dimensions detected!")
        print(f"\nShape summary:")
        for shape in unique_shapes:
            matching_seeds = [seed for seed, s in decision_shapes.items() if s == shape]
            print(f"  Shape {shape}: seeds {matching_seeds}")
        
        raise ValueError(
            f"Cannot combine results with different decision vector sizes. "
            f"Found {len(unique_shapes)} different shapes: {unique_shapes}. "
            f"This usually means you're mixing pixel-based and patch-based results, "
            f"or results with different sampling fractions. "
            f"Ensure all files were generated with the same configuration (same USE_PATCH_APPROACH and SAMPLE_FRACTION)."
        )
    
    print(f"✓ All decision arrays have consistent shape: {list(unique_shapes)[0]}")
    
    #print(f"\n{'='*80}")
    #print(f"CALCULATING SPATIAL AGREEMENT")
    #print(f"{'='*80}\n")
    
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
    
    #print(f"Total solutions: {len(combined_X)}")
    
    # Find non-dominated solutions across all seeds
    nds = NonDominatedSorting()
    pareto_indices = nds.do(combined_F, only_non_dominated_front=True)
    
    pareto_X = combined_X[pareto_indices]
    pareto_F = combined_F[pareto_indices]
    pareto_seeds = all_seed_labels[pareto_indices]
    
    #print(f"Pareto-optimal solutions: {len(pareto_X)}")
    
    # Calculate pairwise distances in objective space
    obj_distances = cdist(pareto_F, pareto_F, metric='euclidean')
    
    # Normalize distances to [0, 1]
    obj_distances_norm = obj_distances / np.max(obj_distances)
    
    # For each solution, find K nearest neighbors from different seeds
    #print(f"\nCalculating Jaccard similarities...")
    spatial_agreements = []
    
    for i in range(len(pareto_X)):
        # Get solutions from different seeds only
        different_seed_mask = pareto_seeds != pareto_seeds[i]
        different_seed_indices = np.where(different_seed_mask)[0]
        
        if len(different_seed_indices) == 0:
            continue
        
        # Get distances only to solutions from different seeds
        distances_to_different = obj_distances[i][different_seed_indices]
        
        # Find K nearest neighbors from different seeds
        k_actual = min(k_neighbors, len(different_seed_indices))
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
            'min_obj_distance': np.min(obj_distances_norm[i][nearest_indices]),
            'max_obj_distance': np.max(obj_distances_norm[i][nearest_indices]),
            'n_neighbors': len(jaccard_scores)
        })
    
    agreement_df = pd.DataFrame(spatial_agreements)
    
    # Generate output filename if not provided
    if output_file is None:
        suffix = "_patch" if use_patch_approach else "_pixel"
        output_file = f"spatial_agreement_{ecosystem}_{file_date}{suffix}.csv"
    
    # Save results
    agreement_df.to_csv(output_file, index=False)
    
    #print(f"\n{'='*80}")
    #print(f"RESULTS SUMMARY")
    #print(f"{'='*80}")
    #print(f"Solutions analyzed:     {len(agreement_df)}")
    #print(f"Mean Jaccard:           {agreement_df['mean_jaccard'].mean():.4f} ± {agreement_df['mean_jaccard'].std():.4f}")
    #print(f"Jaccard range:          [{agreement_df['mean_jaccard'].min():.4f}, {agreement_df['mean_jaccard'].max():.4f}]")
    #print(f"Mean obj distance:      {agreement_df['mean_obj_distance'].mean():.4f} ± {agreement_df['mean_obj_distance'].std():.4f}")
    #print(f"\n✓ Results saved to: {output_file}")
    #print(f"{'='*80}\n")
    
    return agreement_df


def main():
    """
    Main execution function.
    Can be run standalone or imported.
    """
    # Configuration
    ECOSYSTEM = "forest"
    FILE_DATE = "2602"
    FILE_START = 11
    FILE_END = 20
    K_NEIGHBORS = 5
    USE_PATCH_APPROACH = False  # Set to True for patch-based results
    
    # Run analysis
    agreement_df = analyze_spatial_agreement(
        ecosystem=ECOSYSTEM,
        file_date=FILE_DATE,
        file_start=FILE_START,
        file_end=FILE_END,
        k_neighbors=K_NEIGHBORS,
        use_patch_approach=USE_PATCH_APPROACH
    )
    
    return agreement_df


if __name__ == "__main__":
    results = main()
