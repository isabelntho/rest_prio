"""
Restoration Optimization initial code
======================================
Objectives:
1. Abiotic condition anomaly (minimize)
2. Biotic condition anomaly (minimize)  
3. Landscape condition anomaly (minimize)
4. Implementation cost (minimize)

Decision variable: Binary (0/1) for not restore/restore
Restoration effect: Triggers action in cell or neighboring cells

Created: December 2025
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import os
import numpy as np
import pandas as pd
import rasterio as rio
from scipy import ndimage
from datetime import datetime
from multiprocessing import Pool

from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Import spatial operations from separate module
from spatial_operations import apply_burden_sharing, apply_spatial_clustering, _enforce_exact_pixel_count
from spatial_operations import AdaptiveSampling, AdaptiveRepair, compute_sn_dens

# Import data loading functions from separate module
from data_loader import load_initial_conditions, ECOSYSTEM_TYPES, FOCAL_CLASSES, load_admin_regions, load_lulc_raster,create_ecosystem_mask, get_region_reference

# Import results saving functions from separate module
from results_saving import (
    save_parameter_summary, save_scenario_results, save_combined_results,
    save_results_with_reports
)

# =============================================================================
# WEIGHTING FUNCTION
# =============================================================================

def anomaly_improvement_weight(anomaly_values, shape='exponential', scale=1.0):
    """
    Compute improvement weights based on baseline anomaly values.
    
    Weight function w(a₀) is monotonic, peaks at anomaly=0, and decreases as |anomaly| increases.
    
    Args:
        anomaly_values: Array of baseline anomaly values
        shape: Shape of the weighting function ('exponential' or 'gaussian')
        scale: Scale parameter controlling the decay rate (higher = slower decay)
        
    Returns:
        Array of weights in [0,1] with same shape as anomaly_values
    """
    # Ensure we work with absolute anomaly values
    abs_anomaly = np.abs(anomaly_values)
    
    if shape == 'exponential':
        # Exponential decay: w(a) = exp(-|a|/scale)
        weights = np.exp(-abs_anomaly / scale)
    elif shape == 'gaussian':
        # Gaussian decay: w(a) = exp(-|a|²/(2*scale²))
        weights = np.exp(-(abs_anomaly**2) / (2 * scale**2))
    else:
        raise ValueError(f"Unknown weight shape: {shape}. Use 'exponential' or 'gaussian'")
    
    # Ensure weights are in [0,1] and handle any numerical issues
    weights = np.clip(weights, 0.0, 1.0)
    
    return weights

def build_repair_scores(initial_conditions, scenario_params):
    """
    Build a per eligible pixel score used only for deterministic exact count repair.
    Higher score means keep or add this pixel when enforcing the count constraint.
    """
    elig = initial_conditions["eligible_mask"]

    a0 = initial_conditions["abiotic_anomaly"][elig]
    b0 = initial_conditions["biotic_anomaly"][elig]

    wshape = scenario_params.get("anomaly_weight_shape", "exponential")
    wscale = scenario_params.get("anomaly_weight_scale", 1.0)

    wa = anomaly_improvement_weight(a0, shape=wshape, scale=wscale)
    wb = anomaly_improvement_weight(b0, shape=wshape, scale=wscale)

    # keep consistent with your restoration parameterisation
    sa = float(scenario_params.get("abiotic_effect", 0.0)) * wa
    sb = float(scenario_params.get("biotic_effect", 0.0)) * wb

    scores = sa + sb

    # Optional: small cost penalty if cost exists in initial_conditions and scenario uses it
    if "implementation_cost" in initial_conditions:
        c = initial_conditions["implementation_cost"][elig].astype(np.float64)
        c = c / (np.nanmean(c) + 1e-12)
        scores = scores - 1e-6 * c

    return np.asarray(scores, dtype=np.float64)

# =============================================================================
# SCENARIO PARAMETERS
# =============================================================================

def define_scenario_parameters():
    """
    Define continuous ranges for scenario parameters
    Returns:
        dict: Dictionary with parameter names as keys and (min, max) tuples for continuous ranges
              or lists for categorical parameters
    """
    scenario_parameters = {
        'max_restoration_fraction': [0.1, 0.3],#(0.10, 0.50), # Proportion of study area to restore (min, max)
        'spatial_clustering': [0, 0.3],#, 1.0), # Degree of spatial clustering (0=random, 1=highly clustered)
        'burden_sharing': ['no'], # Equal sharing across admin regions (categorical)
        # Separate improvement parameters for each anomaly type
        'abiotic_effect': (0.005, 0.02), # Reduction in abiotic anomaly (min, max)
        'biotic_effect': (0.005, 0.02), # Reduction in biotic anomaly (min, max)
        #'landscape_effect': [0.01],#(0.005, 0.02), # Reduction in landscape anomaly (min, max)
        # Anomaly-dependent improvement weighting
        'anomaly_weight_shape': ['exponential'], # Shape of weighting function
        'anomaly_weight_scale': [1.0], # Scale parameter for weight decay
    }
    
    return scenario_parameters

def sample_scenario_parameters(n_samples_per_param=3, random_seed=42):
    """
    Sample values from continuous parameter ranges and combine with categorical parameters.
    
    Args:
        n_samples_per_param: Number of samples to draw from each continuous parameter
        random_seed: Random seed for reproducible sampling
        
    Returns:
        list: List of parameter combinations (dicts)
    """
    from itertools import product
    import numpy as np
    
    # Set random seed for reproducible sampling
    np.random.seed(random_seed)
    
    params = define_scenario_parameters()
    sampled_values = {}
    
    # Sample from continuous ranges or use categorical values
    for param_name, param_range in params.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Continuous parameter - sample uniformly between min and max
            min_val, max_val = param_range
            samples = np.linspace(min_val, max_val, n_samples_per_param)
            sampled_values[param_name] = samples.tolist()
        else:
            # Categorical parameter - use all values
            sampled_values[param_name] = param_range
    
    # Generate all combinations of sampled values
    param_names = list(sampled_values.keys())
    param_value_lists = list(sampled_values.values())
    
    combinations = []
    for combo in product(*param_value_lists):
        combo_dict = dict(zip(param_names, combo))
        combinations.append(combo_dict)
    
    return combinations

def get_scenario_combinations():
    """
    Get all possible combinations of scenario parameters (legacy function).
    Now uses sampling approach with 3 values per continuous parameter.
    
    Returns:
        list: List of parameter combinations (dicts)
    """
    return sample_scenario_parameters(n_samples_per_param=3)

def get_scenario_by_index(scenario_index, n_samples_per_param=3, random_seed=42):
    """
    Get a specific scenario by index using sampled parameters.
    
    Args:
        scenario_index: Index of the scenario (0-based)
        n_samples_per_param: Number of samples per continuous parameter
        random_seed: Random seed for reproducible sampling
        
    Returns:
        dict: Parameter combination for the scenario
    """
    combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
    
    if 0 <= scenario_index < len(combinations):
        return combinations[scenario_index]
    else:
        raise ValueError(f"Scenario index {scenario_index} out of range. Available: 0-{len(combinations)-1}")

# =============================================================================
# RESTORATION EFFECT FUNCTION
# =============================================================================

def fast_selective_landscape_approximation(convert_vars, initial_conditions):
    """
    Fast landscape calculation that combines both approximation and selective processing.
    
    This approach:
    a) Only processes neighborhoods around converted pixels (selective)
    b) Uses mathematical approximation instead of full compute_sn_dens (approximate)
    
    Expected speedup: 50-200x faster than full recalculation
    """
    if not np.any(convert_vars):
        return initial_conditions['landscape_anomaly']
    
    # Get conversion locations from conversion eligible indices
    conversion_eligible_indices = initial_conditions['conversion_eligible_indices']
    converted_indices = conversion_eligible_indices[convert_vars == 1]
    shape = initial_conditions['shape']
    
    if len(converted_indices) == 0:
        return initial_conditions['landscape_anomaly']
    
    # Start with baseline landscape (no copying yet for efficiency)
    baseline_landscape = initial_conditions['landscape_anomaly']
    
    # Parameters
    pixel_size = 100  # meters per pixel - adjust to match your data
    radius_m = 300    # influence radius in meters
    radius_px = int(radius_m / pixel_size)
    
    # Create conversion mask for affected areas only
    affected_regions = set()
    conversion_locations = []
    
    # Identify all affected neighborhoods
    for converted_idx in converted_indices:
        row, col = divmod(converted_idx, shape[1])
        conversion_locations.append((row, col))
        
        # Mark neighborhood for processing
        r_min = max(0, row - radius_px)
        r_max = min(shape[0], row + radius_px + 1)
        c_min = max(0, col - radius_px)
        c_max = min(shape[1], col + radius_px + 1)
        
        affected_regions.add((r_min, r_max, c_min, c_max))
    
    # Merge overlapping regions for efficiency
    merged_regions = merge_overlapping_regions(affected_regions)
    
    # Only copy baseline if we actually need to modify it
    updated_landscape = baseline_landscape.copy()
    
    # Optional: Print debug info occasionally (every 100th call)
    # print(f"  Fast landscape: processing {len(merged_regions)} regions around {len(converted_indices)} conversions")
    
    # Process each affected region
    for region_bounds in merged_regions:
        r_min, r_max, c_min, c_max = region_bounds
        
        # Extract region
        region_shape = (r_max - r_min, c_max - c_min)
        
        # Find conversions within this region
        region_conversions = []
        for conv_row, conv_col in conversion_locations:
            if r_min <= conv_row < r_max and c_min <= conv_col < c_max:
                # Convert to region-local coordinates
                local_row = conv_row - r_min
                local_col = conv_col - c_min
                region_conversions.append((local_row, local_col))
        
        if not region_conversions:
            continue
        
        # Approximate landscape improvement for this region
        region_landscape_change = approximate_region_landscape_change(
            region_conversions, region_shape, radius_px
        )
        
        # Apply changes to the corresponding area in the full landscape
        updated_landscape[r_min:r_max, c_min:c_max] += region_landscape_change
    
    return updated_landscape


def merge_overlapping_regions(regions):
    """
    Merge overlapping rectangular regions to minimize redundant processing.
    """
    if not regions:
        return []
    
    # Convert set to list and sort
    region_list = list(regions)
    region_list.sort()
    
    merged = []
    current = region_list[0]
    
    for next_region in region_list[1:]:
        # Check if regions overlap
        r_min1, r_max1, c_min1, c_max1 = current
        r_min2, r_max2, c_min2, c_max2 = next_region
        
        # Check for overlap
        if (r_max1 >= r_min2 and r_min1 <= r_max2 and 
            c_max1 >= c_min2 and c_min1 <= c_max2):
            # Merge regions
            current = (
                min(r_min1, r_min2),
                max(r_max1, r_max2),
                min(c_min1, c_min2),
                max(c_max1, c_max2)
            )
        else:
            # No overlap, add current to merged list
            merged.append(current)
            current = next_region
    
    merged.append(current)
    return merged


def approximate_region_landscape_change(conversions, region_shape, radius_px):
    """
    Approximate landscape change within a region using mathematical models.
    No compute_sn_dens needed - pure mathematical approximation.
    """
    # Create conversion mask for this region
    conversion_mask = np.zeros(region_shape, dtype=np.float32)
    for row, col in conversions:
        if 0 <= row < region_shape[0] and 0 <= col < region_shape[1]:
            conversion_mask[row, col] = 1.0
    
    # Create circular influence kernel
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    kernel = ((x**2 + y**2) <= radius_px**2).astype(np.float32)
    
    # Distance-weighted kernel (closer pixels have more influence)
    distances = np.sqrt(x**2 + y**2)
    distances[distances == 0] = 1  # Avoid division by zero
    kernel = kernel / (1 + distances * 0.1)  # Gradual distance decay
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Apply convolution to estimate landscape density improvement
    from scipy.ndimage import convolve
    density_improvement = convolve(conversion_mask, kernel, mode='constant', cval=0.0)
    
    # Convert density improvement to anomaly change
    # Density increase = anomaly decrease (improvement)
    anomaly_improvement = -density_improvement
    
    # Scale the effect based on conversion effectiveness
    # This parameter can be tuned based on validation against full calculations
    effect_strength = 0.15  # Adjust based on your data
    
    return effect_strength * anomaly_improvement


def recalculate_landscape_anomaly_with_conversions(initial_conditions, conversion_mask_2d):
    """
    Efficiently recalculate landscape anomaly after applying conversion actions.
    Converts pixels to focal classes and recalculates landscape density.
    
    Args:
        initial_conditions: Dict with LULC data and focal classes
        conversion_mask_2d: 2D boolean mask of pixels to convert
        
    Returns:
        numpy.ndarray: Updated landscape anomaly values
    """
    import tempfile
    import os as temp_os
    
    # Get original landscape LULC data and conversion info
    lulc_data = initial_conditions['landscape_lulc_data'].copy()
    focal_classes = initial_conditions['landscape_focal_classes']
    lulc_meta = initial_conditions['landscape_lulc_meta'].copy()
    
    # Apply conversions: convert pixels to a focal class (use first focal class)
    # In future, could use different conversion rules or random selection from focal classes
    target_lulc_value = focal_classes[0]  # Convert to first focal class (e.g., forest)
    lulc_data[conversion_mask_2d] = target_lulc_value
    
    # Fix metadata to ensure proper rasterio compatibility
    lulc_meta.update({
        'width': int(lulc_data.shape[1]),
        'height': int(lulc_data.shape[0]),
        'count': 1,
        'dtype': lulc_data.dtype
    })
    
    # Create temporary file with modified LULC
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
        temp_lulc_path = tmp_file.name
    
    try:
        # Write modified LULC to temporary file
        with rio.open(temp_lulc_path, 'w', **lulc_meta) as tmp_dst:
            tmp_dst.write(lulc_data, 1)
        
        # Recalculate landscape density with modified LULC
        landscape_density, _ = compute_sn_dens(temp_lulc_path, focal_classes, radius_m=300)
        
        # Convert density to anomaly (higher density = lower anomaly)
        landscape_anomaly = 1.0 - landscape_density
        
        # Handle NaN values
        landscape_anomaly = np.nan_to_num(landscape_anomaly, nan=0.0)
        
        return landscape_anomaly
        
    finally:
        # Clean up temporary file
        if temp_os.path.exists(temp_lulc_path):
            temp_os.unlink(temp_lulc_path)

def restoration_effect(restore_vars, convert_vars, initial_conditions, effect_params=None):
    """
    Define what happens when restoration or conversion is selected.
    This function calculates the effect of different actions on neighboring cells.
    
    Args:
        restore_vars: Binary array (0/1) for restoration decisions
        convert_vars: Binary array (0/1) for conversion decisions  
        initial_conditions: Dict with initial objective values
        effect_params: Parameters controlling restoration effects (dict)
        
    Returns:
        dict: Updated objective values after restoration effects
    """
    if effect_params is None:
        # Default values if no effect_params provided
        effect_params = {
            'abiotic_effect': 0.01,       # Improvement for abiotic anomaly
            'biotic_effect': 0.01,        # Improvement for biotic anomaly
            #'landscape_effect': 0.01,     # Improvement for landscape anomaly
            'neighbor_radius': 1,            # Effect radius in cells (fixed)
            'neighbor_effect_decay': 0.0,     # Effect strength for neighbors (fixed)
            'anomaly_weight_shape': 'exponential', # Shape of anomaly-dependent weighting
            'anomaly_weight_scale': 1.0     # Scale parameter for weight decay
        }
    
    # Set fixed values for spatial parameters (not scenario parameters)
    effect_params = effect_params.copy()
    effect_params['neighbor_radius'] = 1  # Fixed value
    effect_params['neighbor_effect_decay'] = 0.0  # Fixed value
    
    # Get separate improvement effects for each anomaly type
    abiotic_effect = effect_params.get('abiotic_effect', 0.01)
    biotic_effect = effect_params.get('biotic_effect', 0.01)
    #landscape_effect = effect_params.get('landscape_effect', 0.01)
    effect_params['abiotic_improvement'] = abiotic_effect
    effect_params['biotic_improvement'] = biotic_effect
    #effect_params['landscape_improvement'] = landscape_effect
    
    # Ensure neighbor_radius is integer for array indexing
    effect_params['neighbor_radius'] = int(round(effect_params['neighbor_radius']))
    
    shape = initial_conditions['shape']
    # Get separate eligible masks and indices for restoration and conversion
    restoration_eligible_mask = initial_conditions['restoration_eligible_mask']
    conversion_eligible_mask = initial_conditions['conversion_eligible_mask']
    restoration_eligible_indices = initial_conditions['restoration_eligible_indices']
    conversion_eligible_indices = initial_conditions['conversion_eligible_indices']
    
    # For backward compatibility
    eligible_mask = initial_conditions['eligible_mask']
    eligible_indices = initial_conditions['eligible_indices']
    
    # Create 2D masks from 1D decision variables
    restoration_mask_2d = np.zeros(shape, dtype=bool)
    conversion_mask_2d = np.zeros(shape, dtype=bool)
    
    if np.any(restore_vars):
        # Convert restoration eligible indices with restoration decisions back to 2D coordinates
        restored_indices = restoration_eligible_indices[restore_vars == 1]
        rows, cols = np.divmod(restored_indices, shape[1])
        restoration_mask_2d[rows, cols] = True
    
    if np.any(convert_vars):
        # Convert conversion eligible indices with conversion decisions back to 2D coordinates
        converted_indices = conversion_eligible_indices[convert_vars == 1]
        rows, cols = np.divmod(converted_indices, shape[1])
        conversion_mask_2d[rows, cols] = True
    
    # Initialize updated conditions
    updated_conditions = {}
    
    # Process only objectives that are available in initial_conditions
    available_anomaly_objectives = [obj for obj in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly'] 
                                   if obj in initial_conditions]
    
    for objective in available_anomaly_objectives:
        original_values = initial_conditions[objective].copy()
        updated_values = original_values.copy()
        
        # Determine which action type affects this objective
        if objective in ['abiotic_anomaly', 'biotic_anomaly']:
            # Restoration affects abiotic and biotic anomalies
            action_mask = restoration_mask_2d
            improvement_key = f'{objective.split("_")[0]}_improvement'
        elif objective == 'landscape_anomaly':
            # Conversion affects landscape anomaly
            if np.any(conversion_mask_2d):
                # Use fast selective approximation (combines approximation + selective processing)
                updated_landscape = fast_selective_landscape_approximation(
                    convert_vars, initial_conditions
                )
                updated_conditions[objective] = updated_landscape
                continue
            else:
                # No conversions - use original landscape anomaly
                updated_conditions[objective] = original_values
                continue
        else:
            # Skip unknown objectives
            updated_conditions[objective] = updated_values
            continue
        
        if np.any(action_mask):
            # Direct effect on action cells
            improvement = effect_params[improvement_key]
            
            # Calculate anomaly-dependent improvement weights
            weight_shape = effect_params.get('anomaly_weight_shape', 'exponential')
            weight_scale = effect_params.get('anomaly_weight_scale', 1.0)
            
            # Get weights for action pixels based on their baseline anomaly
            baseline_anomalies = original_values[action_mask]
            improvement_weights = anomaly_improvement_weight(
                baseline_anomalies, shape=weight_shape, scale=weight_scale
            )
            
            # Apply weighted improvement: improvementᵢ = improvement × w(a₀ᵢ)
            weighted_improvements = improvement * improvement_weights
            
            # Apply improvement:
            updated_values[action_mask] = (
                original_values[action_mask] + weighted_improvements
            )

            # Neighbor effects
            if effect_params['neighbor_radius'] > 0:
                # Create kernel for neighbor effects
                radius = effect_params['neighbor_radius']
                y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                kernel = (x*x + y*y) <= radius*radius
                
                # Apply dilation to find neighbor cells
                neighbor_mask = ndimage.binary_dilation(action_mask, structure=kernel)
                neighbor_mask = neighbor_mask & ~action_mask  # Exclude direct action cells
                
                # Apply reduced improvement to neighbors with anomaly weighting
                neighbor_improvement = improvement * effect_params['neighbor_effect_decay']
                
                # Calculate weights for neighbor pixels based on their baseline anomaly
                neighbor_baseline_anomalies = original_values[neighbor_mask]
                neighbor_weights = anomaly_improvement_weight(
                    neighbor_baseline_anomalies, shape=weight_shape, scale=weight_scale
                )
                
                # Apply weighted neighbor improvement
                weighted_neighbor_improvements = neighbor_improvement * neighbor_weights
                
                updated_values[neighbor_mask] = (
                    original_values[neighbor_mask] + weighted_neighbor_improvements
                )
        
        # Only apply changes to appropriate eligible pixels based on objective type
        # This prevents affecting NaN→0 pixels outside the study area
        if objective in ['abiotic_anomaly', 'biotic_anomaly']:
            # Use restoration eligible mask for restoration-affected objectives
            updated_values = np.where(restoration_eligible_mask, updated_values, original_values)
        elif objective == 'landscape_anomaly':
            # Use conversion eligible mask for conversion-affected objectives
            updated_values = np.where(conversion_eligible_mask, updated_values, original_values)
        else:
            # Use general eligible mask for other objectives
            updated_values = np.where(eligible_mask, updated_values, original_values)
        
        # Safety check: ensure no NaN or inf values
        if np.any(np.isnan(updated_values)) or np.any(np.isinf(updated_values)):
            print(f"WARNING: {objective} contains NaN or inf values after restoration effect!")
            updated_values = np.nan_to_num(updated_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        updated_conditions[objective] = updated_values
    
    # Implementation cost (only if cost objective is being used)
    if 'implementation_cost' in initial_conditions:
        base_cost = initial_conditions['implementation_cost'].copy()
        total_cost = 0
        if np.any(restoration_mask_2d):
            total_cost += np.sum(base_cost[restoration_mask_2d])
        if np.any(conversion_mask_2d):
            # Could have different cost structure for conversion in future
            total_cost += np.sum(base_cost[conversion_mask_2d])
        updated_conditions['implementation_cost'] = total_cost
    
    # Population proximity (only if proximity objective is being used)
    # Proximity is a fixed spatial property - no restoration improvement effect
    if 'population_proximity' in initial_conditions:
        proximity_map = initial_conditions['population_proximity'].copy()
        total_distance = 0.0
        total_pixels = 0
        if np.any(restoration_mask_2d):
            total_distance += np.sum(proximity_map[restoration_mask_2d])
            total_pixels += np.sum(restoration_mask_2d)
        if np.any(conversion_mask_2d):
            total_distance += np.sum(proximity_map[conversion_mask_2d])
            total_pixels += np.sum(conversion_mask_2d)
        avg_distance = total_distance / total_pixels if total_pixels > 0 else 0.0
        updated_conditions['population_proximity'] = avg_distance
    
    return updated_conditions

# =============================================================================
# CUSTOM SAMPLING OPERATORS
# =============================================================================









# =============================================================================
# OPTIMIZATION PROBLEM DEFINITION
# =============================================================================


def diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10):
    """
    Diagnose why optimization might not find solutions.
    
    Args:
        initial_conditions: Dict with initial conditions
        scenario_params: Dict with scenario parameters
        n_samples: Number of sample solutions to test
    """
    print("\n" + "="*70)
    print("OPTIMIZATION DIAGNOSTICS")
    print("="*70)
    
    # Create problem instance
    problem = RestorationProblem(initial_conditions, scenario_params, n_jobs=1)  # Use single process for diagnostics
    
    #print(f"\nPROBLEM SETUP:")
    #print(f"   - Decision variables: {problem.n_var} total ({problem.n_pixels} restore + {problem.n_pixels} convert)")
    #print(f"   - Objectives: {problem.n_obj} ({problem.objective_names})")
    #print(f"   - Constraints: {problem.n_constr}")
    #print(f"   - Max restored pixels: {problem.max_restored_pixels}")
    #print(f"   - Restoration fraction: {scenario_params['max_restoration_fraction']*100:.1f}%")
    #print(f"   - Effect parameters:")
    #if 'improvement_effect' in scenario_params:
    #    print(f"     improvement_effect: {scenario_params['improvement_effect']:.4f} (applied to all anomaly types)")
    #print(f"     neighbor_radius: 1 (fixed)")
    #print(f"     neighbor_effect_decay: 0.5 (fixed)")
    
    # Check baseline objectives
    #print(f"\nBASELINE OBJECTIVES (no restoration):")
    x_none = np.zeros(problem.n_var, dtype=int)
    out_none = {}
    problem._evaluate(x_none, out_none)
    
    # for i, obj_name in enumerate(problem.objective_names):
    #     if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
    #         # Display actual anomaly sum (reverse the negative)
    #         actual_sum = -out_none['F'][i]
    #         print(f"   - {obj_name}: sum={actual_sum:.4e} (objective={out_none['F'][i]:.4e})")
    #     else:
    #         print(f"   - {obj_name}: {out_none['F'][i]:.4e}")
    #print(f"   - Constraint violation: {out_none['G'][0]:.4e} (should be ≤ 0)")
    
    # Test sample solutions
    #print(f"\n3. SAMPLE SOLUTIONS (random restoration patterns):")
    
    feasible_count = 0
    infeasible_count = 0
    
    for i in range(n_samples):
        # Generate random solution at max allowed restoration
        x = np.zeros(problem.n_var, dtype=int)
        n_restore = problem.max_restored_pixels
        if n_restore > 0 and n_restore <= problem.n_pixels:
            # Only select from restoration decisions (first half of decision vector)
            restore_indices = np.random.permutation(problem.n_pixels)[:n_restore]
            x[restore_indices] = 1
        
        # Evaluate
        out = {}
        problem._evaluate(x, out)
        
        constraint_violation = out['G'][0]
        is_feasible = constraint_violation <= 0
        
        if is_feasible:
            feasible_count += 1
        else:
            infeasible_count += 1
        
        if i < 3:  # Print first 3 samples in detail
            for j, obj_name in enumerate(problem.objective_names):
                # For anomaly objectives, we minimize negative sum, so lower objective = higher anomaly sum (better)
                if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
                    # Reverse sign for display: show actual anomaly sums
                    baseline_sum = -out_none['F'][j]
                    restored_sum = -out['F'][j]
                    improvement = ((restored_sum - baseline_sum) / abs(baseline_sum) * 100) if baseline_sum != 0 else 0
                    #print(f"       {obj_name}: sum={restored_sum:.4e} (baseline={baseline_sum:.4e}, Δ {improvement:+.2f}%)")
                else:
                    improvement = ((out_none['F'][j] - out['F'][j]) / out_none['F'][j] * 100) if out_none['F'][j] != 0 else 0
                    #print(f"       {obj_name}: {out['F'][j]:.4e} (Δ {improvement:+.2f}%)")
    
    print(f"\n   Summary: {feasible_count}/{n_samples} feasible, {infeasible_count}/{n_samples} infeasible")
    
    if feasible_count > 0:
        print(f"   ✓ Feasible solutions exist, Objectives show improvement with restoration")
    else:
        print(f"   ✗ WARNING: No feasible solutions found in {n_samples} samples!")
        print(f"   → Check if max_restored_pixels constraint is too restrictive")
    
    # Check for common issues
    issues = []
    
    if problem.max_restored_pixels == 0:
        issues.append("   ✗ Max restored pixels is 0 - no restoration possible!")
    
    if problem.max_restored_pixels > problem.n_pixels:
        issues.append(f"   ✗ Max restored pixels ({problem.max_restored_pixels}) > available pixels ({problem.n_pixels})")
    
    # Check if objectives are all zero or constant
    if out_none['F'][0] == 0:
        issues.append("   ✗ Baseline objective is zero - may indicate data loading issue")
    
    if np.any(np.isnan(out_none['F'])):
        issues.append("   ✗ NaN detected in objectives - data contains unmasked NaN values")
    
    if not issues:
        print("   ✓ No obvious setup issues detected")
    else:
        for issue in issues:
            print(issue)
    
    # Test with different restoration amounts
    print(f"\n OBJECTIVE SENSITIVITY TO RESTORATION AMOUNT:")
    test_fractions = [0.05, 0.10, 0.15, 0.20]
    
    for frac in test_fractions:
        n_restore = int(frac * problem.n_pixels)  # Use n_pixels, not n_var
        if n_restore > 0 and n_restore <= problem.n_pixels:
            x = np.zeros(problem.n_var, dtype=int)
            # Only select from restoration decisions (first half)
            restore_indices = np.random.permutation(problem.n_pixels)[:n_restore]
            x[restore_indices] = 1
            
            out = {}
            problem._evaluate(x, out)
            
            # Calculate total improvement across all objectives
            total_improvement = 0
            for j in range(len(out['F'])):
                if out_none['F'][j] != 0:
                    improvement = (out_none['F'][j] - out['F'][j]) / out_none['F'][j]
                    total_improvement += improvement
            
            print(f"   {frac*100:>5.1f}% restored ({n_restore:>6} pixels): Avg improvement = {total_improvement/len(out['F'])*100:>6.2f}%")
    
    print(f"\nINITIAL DATA SPATIAL CORRELATIONS:")
    anomaly_objs = [obj for obj in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly'] 
                if obj in initial_conditions]
    for i in range(len(anomaly_objs)):
        for j in range(i+1, len(anomaly_objs)):
        # Correlation of pixel values (only eligible pixels)
            eligible_mask = initial_conditions['eligible_mask']
            
            # Ensure both data and mask are 2D and have the same shape
            data1_full = initial_conditions[anomaly_objs[i]]
            data2_full = initial_conditions[anomaly_objs[j]]
            
            # Apply mask to extract eligible pixels only
            data1 = data1_full[eligible_mask]
            data2 = data2_full[eligible_mask]
            
            if len(data1) > 0 and len(data2) > 0:
                corr = np.corrcoef(data1, data2)[0, 1]
                print(f"   {anomaly_objs[i]} vs {anomaly_objs[j]}: {corr:.4f}")
            else:
                print(f"   {anomaly_objs[i]} vs {anomaly_objs[j]}: No eligible pixels")

    # Check if improvement pushes values above threshold
    print(f"\nANOMALY VALUE DISTRIBUTIONS (threshold diagnostic):")
    print(f"   Testing with {problem.max_restored_pixels} restored pixels...")
    
    # Test with max restoration
    x_test = np.zeros(problem.n_var, dtype=int)
    if problem.max_restored_pixels > 0:
        # Only select from restoration decisions (first half of decision vector)
        restore_indices = np.random.permutation(problem.n_pixels)[:problem.max_restored_pixels]
        x_test[restore_indices] = 1
    
    # Get updated conditions after restoration (pass both restoration and conversion decisions)
    x_test_restore = x_test[:problem.n_pixels]
    x_test_convert = x_test[problem.n_pixels:]
    updated = restoration_effect(x_test_restore, x_test_convert, initial_conditions)
    
    eligible_mask = initial_conditions['eligible_mask']
    
    for obj_name in anomaly_objs:
        before = initial_conditions[obj_name][eligible_mask]
        after = updated[obj_name][eligible_mask]
        
        # For restored pixels only
        restored_2d = np.zeros(initial_conditions['shape'], dtype=bool)
        # Use only restoration decisions from the test vector
        rows, cols = np.divmod(initial_conditions['eligible_indices'][x_test_restore == 1], initial_conditions['shape'][1])
        restored_2d[rows, cols] = True
        restored_mask = restored_2d[eligible_mask]
        
        before_restored = before[restored_mask]
        after_restored = after[restored_mask]
        
        threshold = 0.0
        
        print(f"\n   {obj_name}:")
        print(f"      ALL eligible pixels:")
        print(f"         Before: min={np.min(before):.3f}, max={np.max(before):.3f}, mean={np.mean(before):.3f}")
        print(f"         % above threshold ({threshold}): {100*np.sum(before > threshold)/len(before):.1f}%")
        
        print(f"      RESTORED pixels only (n={len(before_restored)}):")
        print(f"         Before: min={np.min(before_restored):.3f}, max={np.max(before_restored):.3f}, mean={np.mean(before_restored):.3f}")
        print(f"         After:  min={np.min(after_restored):.3f}, max={np.max(after_restored):.3f}, mean={np.mean(after_restored):.3f}")
        print(f"         % above threshold BEFORE: {100*np.sum(before_restored > threshold)/len(before_restored):.1f}%")
        print(f"         % above threshold AFTER:  {100*np.sum(after_restored > threshold)/len(after_restored):.1f}%")

    print("\n" + "="*70)
    print("END DIAGNOSTICS")
    print("="*70 + "\n")
    
    return problem

# =============================================================================
# OPTIMIZATION PROBLEM DEFINITION
# =============================================================================

class RestorationProblem(ElementwiseProblem):
    """
    Multi-objective restoration optimization problem.
    
    Objectives (all to minimize, depending on what's loaded):
    - Abiotic condition anomaly
    - Biotic condition anomaly
    - Landscape condition anomaly  
    - Implementation cost
    - Population proximity (average distance to population centers)
    """
    
    def __init__(self, initial_conditions, scenario_params, n_jobs=None):
        """
        Initialize the optimization problem.
        
        Args:
            initial_conditions: Dict with initial objective states
            scenario_params: Dict with scenario parameters (e.g., max_restoration_fraction, effect_params)
            n_jobs: Number of parallel jobs to use (None for automatic, 1 for serial, -1 for all cores)
        """
        self.initial_conditions = initial_conditions
        self.scenario_params = scenario_params
        
        # Extract effect parameters from scenario_params (excluding fixed spatial parameters)
        abiotic_effect = scenario_params.get('abiotic_effect', 0.01)
        biotic_effect = scenario_params.get('biotic_effect', 0.01)
        #landscape_effect = scenario_params.get('landscape_effect', 0.01)
        self.effect_params = {
            'abiotic_effect': abiotic_effect,
            'biotic_effect': biotic_effect,
            #'landscape_effect': landscape_effect,
            'abiotic_improvement': abiotic_effect,
            'biotic_improvement': biotic_effect,
            #'landscape_improvement': landscape_effect,
            'neighbor_radius': 1,  # Fixed value
            'neighbor_effect_decay': 0.5  # Fixed value
        }
        
        # Determine which objectives are available
        self.objective_names = []
        if 'abiotic_anomaly' in initial_conditions:
            self.objective_names.append('abiotic_anomaly')
        if 'biotic_anomaly' in initial_conditions:
            self.objective_names.append('biotic_anomaly')
        if 'landscape_anomaly' in initial_conditions:
            self.objective_names.append('landscape_anomaly')
        if 'implementation_cost' in initial_conditions:
            self.objective_names.append('implementation_cost')
        if 'population_proximity' in initial_conditions:
            self.objective_names.append('population_proximity')
        
        n_objectives = len(self.objective_names)
        if n_objectives == 0:
            raise ValueError("No objectives found in initial_conditions")
        
        n_restoration_pixels = initial_conditions['n_restoration_pixels']
        n_conversion_pixels = initial_conditions['n_conversion_pixels']
        max_restoration_fraction = scenario_params['max_restoration_fraction']
        
        # Calculate max restored pixels based on restoration eligible pixels
        self.max_restored_pixels = int(max_restoration_fraction * n_restoration_pixels)
        
        # Store pixel counts for splitting decision vector
        self.n_restoration_pixels = n_restoration_pixels
        self.n_conversion_pixels = n_conversion_pixels
        self.n_pixels = n_restoration_pixels  # For backward compatibility
        
        # Binary decision variables: 0 = no action, 1 = action
        # First n_restoration_pixels elements = restoration decisions
        # Next n_conversion_pixels elements = conversion decisions
        total_decision_vars = n_restoration_pixels + n_conversion_pixels
        
        # Setup parallelization
        elementwise_runner = None
        if n_jobs is None or n_jobs != 1:
            try:
                # Create process pool for parallel evaluation
                if n_jobs == -1:
                    # Use all available cores
                    pool = Pool()
                elif n_jobs is None:
                    # Use automatic number of cores (typically CPU count)
                    pool = Pool()
                else:
                    # Use specified number of cores
                    pool = Pool(processes=n_jobs)
                
                elementwise_runner = StarmapParallelization(pool.starmap)
                print(f"Parallelization enabled with {pool._processes if hasattr(pool, '_processes') else 'auto'} processes")
            except Exception as e:
                print(f"Warning: Could not setup parallelization: {e}")
                print("Falling back to sequential evaluation")
                elementwise_runner = None
        
        # Initialize the problem
        if elementwise_runner is not None:
            super().__init__(
                n_var=total_decision_vars,  # Restoration decisions + conversion decisions
                n_obj=n_objectives,         # Variable number of objectives
                n_constr=1,                 # Constraint on total restoration area
                xl=0,                       # Lower bound: no action
                xu=1,                       # Upper bound: action
                type_var=int,                # Integer (binary) variables
                elementwise=True,
                elementwise_runner=elementwise_runner
            )
        else:
            super().__init__(
                n_var=total_decision_vars,  # Restoration decisions + conversion decisions
                n_obj=n_objectives,         # Variable number of objectives
                n_constr=1,                 # Constraint on total restoration area
                xl=0,                       # Lower bound: no action
                xu=1,                       # Upper bound: action
                type_var=int,                # Integer (binary) variables
                elementwise=True
            )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution (restoration plan).
        
        Args:
            x: Decision variables (binary array) - already clustered/burden-shared by sampling/repair
                First n_restoration_pixels elements: restoration decisions for restoration-eligible pixels
                Next n_conversion_pixels elements: conversion decisions for conversion-eligible pixels
            out: Output dictionary for objectives and constraints
        """
        # Initialize debug counter
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        
        # Split decision vector into restore and convert actions
        x_restore = x[:self.n_restoration_pixels]
        x_convert = x[self.n_restoration_pixels:self.n_restoration_pixels + self.n_conversion_pixels] 
        
        # Enable conversion actions now that we have fast landscape calculation
        if 'landscape_anomaly' not in self.initial_conditions:
            # Force convert actions to zero if landscape objective not available
            x_convert[:] = 0
        
        # Use both restoration and conversion decisions
        n_restored = np.sum(x_restore)
        n_converted = np.sum(x_convert)
        n_total_actions = n_restored + n_converted
        
        # Debug: Track action patterns for first few evaluations
        #if self._debug_count <= 10:
            #print(f"DEBUG Eval {self._debug_count}: {n_restored} restore + {n_converted} convert = {n_total_actions} total")
        
        # Apply restoration effects to both restoration and conversion decisions
        updated_conditions = restoration_effect(x_restore, x_convert, self.initial_conditions, self.effect_params)
        
        # Calculate objective values (all to minimize) - only for available objectives
        objectives = []
        
        for obj_name in self.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                # Anomaly objectives: MINIMIZE negative sum (equivalent to MAXIMIZE sum)
                # Higher anomalies are better, so we want to maximize the sum
                # NSGA-II minimizes, so we minimize the negative
                threshold = 0.0  # or different per objective
                pixels_above = np.sum(updated_conditions[obj_name] > threshold)
                obj_value = -pixels_above  # Maximize count above threshold
                
                # Debug abiotic/biotic anomaly if needed
                #if self._debug_count <= 3 and obj_name == 'abiotic_anomaly':
                #    print(f"  {obj_name}: pixels_above={pixels_above}, obj_value={obj_value}")
                
            elif obj_name == 'landscape_anomaly':
                # Landscape objective: MINIMIZE the actual sum of landscape anomalies
                # Use the actual values, not just count above threshold
                landscape_vals = updated_conditions[obj_name]
                obj_value = np.sum(landscape_vals)  # Minimize total landscape impact
                
                # Debug landscape anomaly specifically
                #if self._debug_count <= 10:
                #    unique_vals = np.unique(landscape_vals)
                #    print(f"  Landscape anomaly: sum={obj_value:.6f}, obj_value={obj_value:.6f}")
                #    print(f"  Landscape range: min={landscape_vals.min():.6f}, max={landscape_vals.max():.6f}")
                #    print(f"  Landscape unique values: {len(unique_vals)} unique (showing first 5: {unique_vals[:5]})")

            elif obj_name == 'implementation_cost':
                # Cost objective: minimize total cost (lower is better)
                obj_value = updated_conditions[obj_name]
            elif obj_name == 'population_proximity':
                # Proximity objective: minimize average distance to population centers
                obj_value = updated_conditions[obj_name]
            else:
                raise ValueError(f"Unknown objective: {obj_name}")
            
            objectives.append(obj_value)
        
        out["F"] = objectives
        
        # Constraint: total number of pixels with actions (restore + convert)
        out["G"] = [abs(n_total_actions - self.max_restored_pixels)]  # Should be 0 due to exact count enforcement

# =============================================================================
# OPTIMIZATION EXECUTION
# =============================================================================

class HVCallback:
    """
    Hypervolume-based early stopping callback.
    Monitors hypervolume improvement and stops optimization if no significant improvement 
    is observed for a specified number of generations.
    """
    
    def __init__(self, patience=15, min_improvement=1e-6, verbose=True):
        """
        Initialize hypervolume callback.
        
        Args:
            patience: Number of generations to wait for improvement before stopping
            min_improvement: Minimum relative hypervolume improvement threshold
            verbose: Print convergence information
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.verbose = verbose
        self.hv_history = []
        self.best_hv = 0.0
        self.no_improvement_count = 0
        self.converged = False
    
    def __call__(self, algorithm):
        """
        Called at each generation to check for convergence.
        
        Args:
            algorithm: The optimization algorithm object
        """
        # Get current population objectives
        if hasattr(algorithm, 'pop') and algorithm.pop is not None:
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                # Calculate hypervolume
                try:
                    # Create reference point (worst case for each objective)
                    ref_point = np.max(F, axis=0) + 1.0
                    
                    # Calculate hypervolume using pymoo's HV indicator
                    hv_indicator = HV(ref_point=ref_point)
                    current_hv = hv_indicator(F)
                    
                    self.hv_history.append(current_hv)
                    
                    # Check for improvement
                    if current_hv > self.best_hv:
                        relative_improvement = (current_hv - self.best_hv) / (self.best_hv + 1e-10)
                        if relative_improvement >= self.min_improvement:
                            self.best_hv = current_hv
                            self.no_improvement_count = 0
                            if self.verbose and algorithm.n_gen % 10 == 0:
                                print(f"   HV improved: {current_hv:.6f} (+{relative_improvement*100:.4f}%)")
                        else:
                            self.no_improvement_count += 1
                    else:
                        self.no_improvement_count += 1
                    
                    # Check for convergence
                    if self.no_improvement_count >= self.patience:
                        self.converged = True
                        if self.verbose:
                            print(f"   Early stopping: No HV improvement for {self.patience} generations")
                    
                except Exception as e:
                    # If hypervolume calculation fails, just continue
                    if self.verbose and algorithm.n_gen == 1:
                        print(f"   Warning: Hypervolume calculation failed: {e}")
                    self.hv_history.append(0.0)

def run_single_scenario_optimization(initial_conditions, scenario_params, pop_size=50, 
                                   n_generations=100, save_results=True, verbose=True, skip_diagnostics=False,
                                   hv_patience=15, hv_min_improvement=1e-6, n_jobs=None):
    """
    Run the multi-objective restoration optimization for a single scenario.
    
    Args:
        initial_conditions: Initial objective conditions
        scenario_params: Dict with scenario parameters
        pop_size: Population size for NSGA-II
        n_generations: Number of optimization generations
        save_results: Whether to save results to files
        verbose: Print progress information
        skip_diagnostics: Skip diagnostic output (useful for multi-scenario runs)
        hv_patience: Generations to wait for hypervolume improvement before stopping
        hv_min_improvement: Minimum relative hypervolume improvement threshold
        n_jobs: Number of parallel jobs to use (None for automatic, 1 for serial, -1 for all cores)
        
    Returns:
        dict: Optimization results
    """
    if verbose:
        print(f"\n=== SINGLE SCENARIO OPTIMIZATION ===") 
        print(f"Scenario parameters: {scenario_params}")
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_generations} (with HV early stopping: patience={hv_patience})")
        #print(f"Max restoration: {scenario_params['max_restoration_fraction']*100:.1f}% of eligible area")
        print(f"Eligible pixels: {initial_conditions['n_pixels']}")
        
        # Show which objectives are being used
        problem_temp = RestorationProblem(initial_conditions, scenario_params, n_jobs=1)  # Use single process for setup
        print(f"Objectives: {problem_temp.objective_names} ({len(problem_temp.objective_names)} total)")
    
    # Create optimization problem
    problem = RestorationProblem(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params,
        n_jobs=n_jobs
    )
    
    # Print additional diagnostic information
    if verbose:
        print(f"\nOptimization setup details:")
        print(f"  Max restored pixels allowed: {problem.max_restored_pixels}")
        print(f"  Number of objectives: {len(problem.objective_names)}")
        
        # Sample objective values without restoration
        sample_obj_str = "  Baseline objectives (no restoration): "
        for obj_name in problem.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
                val = np.sum(initial_conditions[obj_name])
                sample_obj_str += f"{obj_name}={val:.2e}, "
        print(sample_obj_str.rstrip(", "))
        
        if problem.max_restored_pixels == 0:
            print("  WARNING: max_restored_pixels is 0! No restoration possible.")
        
        # Only run diagnostics if not skipped
        if not skip_diagnostics:
            #print("\nRunning diagnostics to check optimization setup...")
            diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10)
    
    # Create custom sampling and repair operators based on scenario parameters
    burden_sharing = scenario_params.get('burden_sharing', 'no')
    clustering_strength = scenario_params.get('spatial_clustering', 0.0)
    
    # Build scores for repair operator
    scores = build_repair_scores(initial_conditions, scenario_params)
    
    # Use consolidated operators for all scenarios
    sampling = AdaptiveSampling(
        initial_conditions, 
        problem.max_restored_pixels, 
        scenario_params
    )
    repair = AdaptiveRepair(
        initial_conditions, 
        problem.max_restored_pixels, 
        scenario_params, 
        scores
    )
    
    if verbose:
        strategy_desc = []
        if burden_sharing == 'yes':
            strategy_desc.append("burden-sharing")
        if clustering_strength > 0.0:
            strategy_desc.append(f"clustering({clustering_strength})")
        if not strategy_desc:
            strategy_desc.append("random with score-based repair")
        print(f"  Using adaptive operators: {', '.join(strategy_desc)}")
    
    # Create optimization algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,                # Custom or standard sampling
        crossover=HUX(),                  # Half-uniform crossover
        mutation=BitflipMutation(prob=0.05),       # Bit-flip mutation
        repair=repair                     # Custom repair to maintain properties
    )
    
    # Set up termination (use standard generation-based termination)
    termination = get_termination("n_gen", n_generations)
    
    # Progress callback
    class ProgressCallback:
        def __init__(self, verbose=True):
            self.verbose = verbose
            self.start_time = None
            self.hv_callback = HVCallback(patience=hv_patience, min_improvement=hv_min_improvement, verbose=verbose)
            
        def __call__(self, algorithm):
            if self.start_time is None:
                self.start_time = datetime.now()
            
            # Call hypervolume callback for early stopping check
            self.hv_callback(algorithm)
            
            gen = algorithm.n_gen
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.verbose and gen % 10 == 0:
                progress = (gen / n_generations) * 100
                eta = (elapsed / gen) * (n_generations - gen) if gen > 0 else 0
                print(f"   Generation {gen}/{n_generations} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed/60:.1f}min - ETA: {eta/60:.1f}min")
            
            # Stop optimization if hypervolume has converged
            if self.hv_callback.converged:
                algorithm.termination.force_termination = True
    
    # Run optimization
    if verbose:
        print("Starting optimization...")
    
    try:
        callback = ProgressCallback(verbose=verbose)
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=42, 
            verbose=False,
            callback=callback
        )
        
        # Debug: Check what's in the result
        if verbose:
            print(f"\nDEBUG - Result object:")
            print(f"  result is None: {result is None}")
            if result is not None:
                # Check final population
                if hasattr(result, 'pop') and result.pop is not None:
                    print(f"\n  Final population size: {len(result.pop)}")
                    if len(result.pop) > 0:
                        pop_F = result.pop.get("F")
                        if pop_F is not None:
                            print(f"  Population objectives shape: {pop_F.shape}")
                            print(f"  Sample objectives (first 3 solutions):")
                            for i in range(min(3, len(pop_F))):
                                print(f"    Sol {i}: {pop_F[i]}")
                            
                            # Check for diversity
                            print(f"\n  Objective statistics across population:")
                            for j, obj_name in enumerate(problem.objective_names):
                                obj_vals = pop_F[:, j]
                                print(f"    {obj_name}: min={np.min(obj_vals):.4e}, max={np.max(obj_vals):.4e}, std={np.std(obj_vals):.4e}")
                            
                            # Check correlation between objectives
                            if pop_F.shape[1] > 1:
                                print(f"\n  Objective correlations (high correlation = no trade-offs):")
                                for i in range(pop_F.shape[1]):
                                    for j in range(i+1, pop_F.shape[1]):
                                        corr = np.corrcoef(pop_F[:, i], pop_F[:, j])[0, 1]
                                        print(f"    {problem.objective_names[i]} vs {problem.objective_names[j]}: {corr:.4f}")
        
        if result is not None and hasattr(result, 'F') and result.F is not None and len(result.F) > 0:
            if verbose:
                convergence_reason = "hypervolume plateau" if callback.hv_callback.converged else "generation limit"
                final_gen = len(callback.hv_callback.hv_history)
                print(f"✓ Optimization completed after {final_gen} generations ({convergence_reason})")
                if callback.hv_callback.hv_history:
                    print(f"   Final hypervolume: {callback.hv_callback.hv_history[-1]:.6f}")
                print(f"   Found {len(result.F)} Pareto-optimal solutions")
            
            # Prepare results
            # Create a copy of initial_conditions without the geodataframe to avoid large pickle files
            initial_conditions_filtered = initial_conditions.copy()
            if 'admin_data' in initial_conditions_filtered and initial_conditions_filtered['admin_data'] is not None:
                admin_data_filtered = initial_conditions_filtered['admin_data'].copy()
                # Remove the geodataframe but keep other admin data
                if 'gdf' in admin_data_filtered:
                    del admin_data_filtered['gdf']
                initial_conditions_filtered['admin_data'] = admin_data_filtered

            # CAPTURE FULL POPULATION DATA FOR DIAGNOSTICS
            full_population_data = None
            if hasattr(result, 'pop') and result.pop is not None:
                pop_F = result.pop.get("F")
                pop_X = result.pop.get("X")
                if pop_F is not None and pop_X is not None:
                    full_population_data = {
                        'objectives': pop_F,  # Full population objectives
                        'decisions': pop_X,   # Full population decisions
                        'population_size': len(pop_F)
                    }  
            
            optimization_results = {
                'scenario_params': scenario_params,
                'objective_names' : problem.objective_names,
                'objectives': result.F,           # Objective values
                'decisions': result.X,            # Decision variables (restoration plans)
                'n_solutions': len(result.F),
                'problem_info': {
                    'n_pixels': initial_conditions['n_pixels']
                },
                'algorithm_info': {
                    'pop_size': pop_size,
                    'n_generations': n_generations,
                    'actual_generations': len(callback.hv_callback.hv_history),
                    'converged_early': callback.hv_callback.converged,
                    'termination_reason': 'hypervolume_convergence' if callback.hv_callback.converged else 'generation_limit',
                    'convergence_reason': 'hypervolume_plateau' if callback.hv_callback.converged else 'generation_limit',
                    'hypervolume_history': callback.hv_callback.hv_history,
                    'final_hypervolume': callback.hv_callback.hv_history[-1] if callback.hv_callback.hv_history else None,
                    'hv_patience': hv_patience,
                    'hv_min_improvement': hv_min_improvement,
                    'sampling_method': 'adaptive',
                    'repair_operators': ['score_based_repair', 'random_repair'],
                    'timestamp': datetime.now().isoformat()
                },
                'initial_conditions': initial_conditions_filtered,
                'hv_callback': callback.hv_callback,# Store for evolution tracking
                'full_population': full_population_data  
            
        }
            
            # Save results with comprehensive reporting if requested
            if save_results:
                # Use simplified reporting - generates all essential reports
                save_results_with_reports(optimization_results, verbose=verbose)
            
            return optimization_results
            
        else:
            if verbose:
                print("✗ Optimization failed - no solutions found")
            return None
            
    except Exception as e:
        if verbose:
            print(f"✗ Error during optimization: {e}")
        return None

def run_all_scenarios_optimization(initial_conditions, n_samples_per_param=3, pop_size=50, 
                                 n_generations=100, save_results=True, verbose=True, random_seed=42):
    """
    Run optimization for all scenario combinations using sampled parameter values.
    
    Args:
        initial_conditions: Initial objective conditions
        n_samples_per_param: Number of samples to draw from each continuous parameter
        pop_size: Population size for NSGA-II
        n_generations: Number of optimization generations
        save_results: Whether to save results to files
        verbose: Print progress information
        random_seed: Random seed for reproducible parameter sampling
        
    Returns:
        dict: Combined results from all scenarios
    """
    scenario_combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
    
    if verbose:
        print(f"\n=== MULTI-SCENARIO OPTIMIZATION ===")
        print(f"Total scenarios to run: {len(scenario_combinations)}")
        for i, params in enumerate(scenario_combinations):
            print(f"  Scenario {i}: {params}")
        print()
    
    all_results = {}
    
    # Run optimization for each scenario
    for i, scenario_params in enumerate(scenario_combinations):
        if verbose:
            print(f"\n--- RUNNING SCENARIO {i+1}/{len(scenario_combinations)} ---")
            print(f"Parameters for this scenario:")
            for param_name, param_value in scenario_params.items():
                if isinstance(param_value, float):
                    print(f"  {param_name}: {param_value:.4f}")
                else:
                    print(f"  {param_name}: {param_value}")
        
        scenario_results = run_single_scenario_optimization(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            pop_size=pop_size,
            n_generations=n_generations,
            save_results=False,  # Don't save individual files when running all scenarios
            verbose=verbose,
            skip_diagnostics=True  # Skip diagnostics for multi-scenario runs
        )
        
        if scenario_results is not None:
            all_results[i] = scenario_results
            if verbose:
                n_solutions = scenario_results['n_solutions']
                print(f"✓ Scenario {i} completed: {n_solutions} solutions found")
        else:
            if verbose:
                print(f"✗ Scenario {i} failed")
    
    # Prepare combined results
    # Create a copy of initial_conditions without the geodataframe to avoid large pickle files
    initial_conditions_filtered = initial_conditions.copy()
    if 'admin_data' in initial_conditions_filtered and initial_conditions_filtered['admin_data'] is not None:
        admin_data_filtered = initial_conditions_filtered['admin_data'].copy()
        # Remove the geodataframe but keep other admin data
        if 'gdf' in admin_data_filtered:
            del admin_data_filtered['gdf']
        initial_conditions_filtered['admin_data'] = admin_data_filtered
    
    combined_results = {
        'scenarios': all_results,
        'n_scenarios_run': len(all_results),
        'n_scenarios_total': len(scenario_combinations),
        'scenario_parameters': define_scenario_parameters(),
        'n_samples_per_param': n_samples_per_param,
        'random_seed': random_seed,
        'algorithm_info': {
            'pop_size': pop_size,
            'n_generations': n_generations,
            'timestamp': datetime.now().isoformat()
        },
        'initial_conditions': initial_conditions_filtered
    }
    
    # Save combined results
    if save_results and all_results:
        save_combined_results(combined_results, verbose=verbose)
        # Also save parameter summary for reference
        save_parameter_summary(output_dir=".", n_samples_per_param=n_samples_per_param, 
                             random_seed=random_seed, verbose=verbose)
    
    if verbose:
        print(f"\n=== ALL SCENARIOS COMPLETE ===")
        print(f"Successfully completed: {len(all_results)}/{len(scenario_combinations)} scenarios")
    
    return combined_results

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main(workspace_dir=".", scenario='all', objectives=None, n_samples_per_param=3, 
         pop_size=50, n_generations=100, save_results=True, verbose=True, random_seed=42,
         sample_fraction=None, sample_seed=42, ecosystem='all', lulc_path=None):
    """
    Main execution function for restoration optimization.
    
    Args:
        workspace_dir: Directory containing input data
        scenario: Scenario to run ('all' for all scenarios, or integer index for specific scenario)
        objectives: List of objectives to use (e.g., ['abiotic', 'biotic', 'landscape'])
                   Available: 'abiotic', 'biotic', 'landscape', 'cost'
                   If None, uses all available objectives
        n_samples_per_param: Number of samples to draw from each continuous parameter
        pop_size: Population size for optimization
        n_generations: Number of optimization generations
        save_results: Whether to save results
        verbose: Print progress information
        random_seed: Random seed for reproducible parameter sampling
        sample_fraction: Fraction of eligible pixels to use for optimization (e.g., 0.25 for 25%)
                        If None, uses all eligible pixels
        sample_seed: Random seed for spatial sampling (default: 42)
        
    Returns:
        dict: Optimization results
    """
    
    if scenario == 'all':
        # Run all scenarios
        return run_all_scenarios_optimization(
            initial_conditions=load_initial_conditions(workspace_dir, objectives=objectives, 
                                                     region='Bern', ecosystem=ecosystem, sample_fraction=sample_fraction, 
                                                     sample_seed=sample_seed, lulc_path=lulc_path),
            n_samples_per_param=n_samples_per_param,
            pop_size=pop_size,
            n_generations=n_generations,
            save_results=save_results,
            verbose=verbose,
            random_seed=random_seed
        )
    else:
        # Run single scenario by index
        if verbose:
            print("=== RESTORATION OPTIMIZATION WORKFLOW ===")
            if objectives is not None:
                print(f"Using objectives: {objectives}")
            print(f"Running scenario {scenario}")
        
        initial_conditions = load_initial_conditions(workspace_dir, objectives=objectives,
                                                   region='Bern', ecosystem=ecosystem, sample_fraction=sample_fraction, 
                                                   sample_seed=sample_seed, lulc_path=lulc_path)
        
        try:
            scenario_combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
            if 0 <= scenario < len(scenario_combinations):
                scenario_params = scenario_combinations[scenario]
            else:
                raise ValueError(f"Scenario index {scenario} out of range. Available: 0-{len(scenario_combinations)-1}")
            
            return run_single_scenario_optimization(
                initial_conditions=initial_conditions,
                scenario_params=scenario_params,
                pop_size=pop_size,
                n_generations=n_generations,
                save_results=save_results,
                verbose=verbose
            )
            
        except ValueError as e:
            if verbose:
                print(f"\n✗ Error: {e}")
            return None

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Available ecosystem types: 'all', 'forest', 'agricultural', 'grassland'
    # Choose which ecosystem to optimize for:
    ECOSYSTEM_TO_RUN = "grassland" #Change this to 'forest', 'agricultural', 'grassland', or 'all'
    
    # Available regions: 'Bern', 'CH'
    # This affects the reference raster used for data validation
    REGION = 'Bern'  # Change to 'CH' for Switzerland-wide optimization
    
    print(f"\\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
    # =========================================================================
    # IMPLEMENTATION 1: CUSTOM PARAMETERS
    # =========================================================================
    # Define your own parameter values and run optimization
    
    custom_scenario_params = {
         'max_restoration_fraction': 0.05,  # Restore 30% of eligible area
         'spatial_clustering': 0,         # High spatial clustering
         'burden_sharing': 'no',           # Equal sharing across regions
         'abiotic_effect': 0.01,      # Effect on abiotic anomaly
         'biotic_effect': 0.01#,       # Effect on biotic anomaly
         #'landscape_effect': 0.01,    # Effect on landscape anomaly
    }
    
    # Load data for selected ecosystem
    initial_conditions = load_initial_conditions(".", objectives=['abiotic', 'biotic', 'landscape', 'cost'], 
                                                 region=REGION,  # Defines validation reference
                                                 ecosystem=ECOSYSTEM_TO_RUN,  # This applies LULC masking
                                                 sample_fraction=None, sample_seed=42)
    
    # Run optimization
    results = run_single_scenario_optimization(
        initial_conditions=initial_conditions,
        scenario_params=custom_scenario_params,
        pop_size=50,
        n_generations=100,
        save_results=True,
        verbose=True,
        n_jobs=4
    )
    
    # =========================================================================
    # IMPLEMENTATION 2: ALL SCENARIOS OPTIMIZATION
    # =========================================================================
    # Run optimization for all parameter combinations (sampled from ranges)
    
    # results = run_all_scenarios_optimization(
    #    initial_conditions=load_initial_conditions(".", objectives=['abiotic', 'biotic', 'landscape', 'cost'], 
    #                                              sample_fraction=0.10, sample_seed=42),  # Use 25% sample
    #    n_samples_per_param=2,  # 3 samples per continuous parameter
    #    pop_size=20,
    #    n_generations=50,
    #    save_results=True,
    #    verbose=True,
    #    random_seed=42
    # )
    
    if results is not None:
        print("\n✓ Optimization completed successfully!")
    else:
        print("\n✗ Optimization failed.")