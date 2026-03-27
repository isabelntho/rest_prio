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
from email.mime import base
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
from spatial_operations import AdaptiveSampling, AdaptiveRepair, compute_sn_dens, compute_sn_dens_array

# Import data loading functions from separate module
from data_loader import load_initial_conditions, ECOSYSTEM_TYPES, FOCAL_CLASSES, load_admin_regions, load_lulc_raster,create_ecosystem_mask, get_region_reference
from time import perf_counter
# Import results saving functions from separate module
from results_saving import (
    save_parameter_summary, save_scenario_results, save_combined_results,
    save_results_with_reports
)
# Import patch-based optimization approach
from patch_approach import (
    create_patch_mappings,
    PatchRepair,
    PatchAwareSampling,
    aggregate_patch_scores_from_pixel_scores,
)

from scenarios import (
    define_scenario_parameters,
    sample_scenario_parameters,
    expand_scenarios,
)

import psutil
from time import time

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
    neg_mask = anomaly_values < 0
    # Ensure we work with absolute anomaly values
    abs_anomaly = np.abs(anomaly_values)
    
    if shape == 'exponential':
        # Exponential decay: w(a) = exp(-|a|/scale)
        weights = 1-(np.exp(-abs_anomaly / scale))
    elif shape == 'gaussian':
        # Gaussian decay: w(a) = exp(-|a|²/(2*scale²))
        weights = np.exp(-(abs_anomaly**2) / (2 * scale**2))
    else:
        raise ValueError(f"Unknown weight shape: {shape}. Use 'exponential' or 'gaussian'")
    gamma = 3.0
    weights = weights ** gamma
    weights = np.where(neg_mask, weights, 0.0)
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
# PATCH-BASED APPROACH INITIALIZATION
# =============================================================================

def initialize_patch_approach(initial_conditions, patch_size=100):
    """
    Add patch mappings to initial_conditions for patch-based optimization.
    
    This function creates patch definitions for both restoration and conversion
    eligible areas, enabling patch-level decision making while maintaining
    pixel-level objective calculations.
    
    Args:
        initial_conditions: Dict with shape, masks, and indices
        patch_size: Size of each patch in pixels (default: 100x100)
    
    Returns:
        Updated initial_conditions dict with patch mappings added
    """
    print(f"\nInitializing patch-based approach (patch size: {patch_size}x{patch_size} pixels)...")
    
    # Create patch mappings for restoration and conversion areas
    patch_mappings = create_patch_mappings(
        initial_conditions, 
        patch_size=patch_size
    )
    
    # Add patch mappings to initial_conditions
    initial_conditions['patch_mappings'] = patch_mappings
    initial_conditions['patch_approach_enabled'] = True
    initial_conditions['patch_size'] = patch_size
    
    # Store patch counts for easy access
    initial_conditions['n_restoration_patches'] = patch_mappings['restoration_patches']['n_patches']
    initial_conditions['n_conversion_patches'] = patch_mappings['conversion_patches']['n_patches']
    
    print(f"  Restoration patches: {initial_conditions['n_restoration_patches']}")
    print(f"  Conversion patches: {initial_conditions['n_conversion_patches']}")
    
    # Calculate average pixels per patch for information
    if initial_conditions['n_restoration_patches'] > 0:
        avg_pixels_per_restoration_patch = (
            initial_conditions['n_restoration_pixels'] / 
            initial_conditions['n_restoration_patches']
        )
        print(f"  Avg pixels per restoration patch: {avg_pixels_per_restoration_patch:.1f}")
    
    if initial_conditions['n_conversion_patches'] > 0:
        avg_pixels_per_conversion_patch = (
            initial_conditions['n_conversion_pixels'] / 
            initial_conditions['n_conversion_patches']
        )
        print(f"  Avg pixels per conversion patch: {avg_pixels_per_conversion_patch:.1f}")
    
    return initial_conditions

# =============================================================================
# RESTORATION EFFECT FUNCTIONS
# =============================================================================

def conversion_mask(convert_vars, initial_conditions):
    """
    Landscape calculation using full recalculation with compute_sn_dens.
    
    Previously used mathematical approximation for speed, but now uses full 
    recalculation for accuracy. The approximation functions are kept for 
    potential future use but are disabled.
    
    Note: No longer "fast" or "selective" - performs full landscape recalculation.
    """
    if not np.any(convert_vars):
        return initial_conditions['landscape_anomaly']
    
    # Get conversion locations from conversion eligible indices
    conversion_eligible_indices = initial_conditions['conversion_eligible_indices']
    converted_indices = conversion_eligible_indices[convert_vars == 1]
    shape = initial_conditions['shape']
    
    if len(converted_indices) == 0:
        return initial_conditions['landscape_anomaly']
    
    # Create 2D conversion mask from 1D conversion decisions
    conversion_mask_2d = np.zeros(shape, dtype=bool)
    rows, cols = np.divmod(converted_indices, shape[1])
    conversion_mask_2d[rows, cols] = True
    
    # Use full recalculation with compute_sn_dens for accuracy
    updated_landscape = recalculate_landscape_anomaly_with_conversions(
        initial_conditions, conversion_mask_2d
    )
    
    return updated_landscape

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
    import numpy as np

    # Get original landscape LULC data and conversion info
    lulc_data = initial_conditions["landscape_lulc_data"].copy()
    focal_classes = initial_conditions["landscape_focal_classes"]
    lulc_meta = initial_conditions["landscape_lulc_meta"]

    # Apply conversions: convert pixels to a focal class (use first focal class)
    target_lulc_value = focal_classes[0]
    lulc_data[conversion_mask_2d] = target_lulc_value

    # Recalculate landscape density using the in memory array
    landscape_density = compute_sn_dens_array(
        lulc_data,
        nodata=lulc_meta.get("nodata"),
        res=lulc_meta["transform"][0],
        focal_classes=focal_classes,
        radius_m=300
    )

    # Convert density to anomaly (higher density = lower anomaly)
    landscape_anomaly = 1.0 - landscape_density

    # Handle NaN values
    landscape_anomaly = np.nan_to_num(landscape_anomaly, nan=0.0)

    return landscape_anomaly

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
    assert effect_params is not None
    
    # Get separate improvement effects for each anomaly type
    abiotic_effect = effect_params['abiotic_effect']
    biotic_effect = effect_params['biotic_effect']
    
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
            improvement_key = f'{objective.split("_")[0]}_effect'
        elif objective == 'landscape_anomaly':
            # Conversion affects landscape anomaly
            if np.any(conversion_mask_2d):
                # Use fast selective approximation (combines approximation + selective processing)
                updated_landscape = conversion_mask(
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
            #weighted_improvements = improvement * improvement_weights
            
            # Apply improvement:
            #updated_values[action_mask] = (
            #    original_values[action_mask] + weighted_improvements
            #)
            updated_values[action_mask] = baseline_anomalies + (effect_params[improvement_key] * improvement_weights)

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
    
    return updated_conditions

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
    print("\n=== Quick problem diagnostics ===")
    
    # Create problem instance
    problem = RestorationProblem(initial_conditions, scenario_params, n_jobs=1)  # Use single process for diagnostics
    
    # Check baseline objectives
    #print(f"\nBASELINE OBJECTIVES (no restoration):")
    x_none = np.zeros(problem.n_var, dtype=int)
    out_none = {}
    problem._evaluate(x_none, out_none)
    
    feasible_count = 0
    infeasible_count = 0
    
    for i in range(n_samples):
        # Generate random solution at max allowed restoration
        x = np.zeros(problem.n_var, dtype=int)
        n_restore = problem.max_action_pixels
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
                else:
                    improvement = ((out_none['F'][j] - out['F'][j]) / out_none['F'][j] * 100) if out_none['F'][j] != 0 else 0
    
    print(f"\n   Summary: {feasible_count}/{n_samples} feasible, {infeasible_count}/{n_samples} infeasible")
    
    if feasible_count > 0:
        print(f"   ✓ Feasible solutions exist, Objectives show improvement with restoration")
    else:
        print(f"   ✗ WARNING: No feasible solutions found in {n_samples} samples!")
        print(f"   → Check if max_action_pixels constraint is too restrictive")
    
    # Check for common issues
    issues = []
    
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
    print(f"   Baseline objectives (no restoration):")
    for j, obj_name in enumerate(problem.objective_names):
        print(f"      {obj_name}: {out_none['F'][j]:.6f}")
    
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
            n_improved_objectives = 0
            for j in range(len(out['F'])):
                # For difference-based objectives where baseline is 0, check absolute improvement
                if out_none['F'][j] == 0:
                    # Objective is 0 at baseline - check if restoration makes it more negative (better)
                    if out['F'][j] < 0:
                        improvement = abs(out['F'][j])  # Absolute improvement from 0
                        total_improvement += improvement
                        n_improved_objectives += 1
                elif out_none['F'][j] != 0:
                    # Standard relative improvement calculation
                    improvement = (out_none['F'][j] - out['F'][j]) / abs(out_none['F'][j])
                    total_improvement += improvement
                    n_improved_objectives += 1
            
            avg_improvement = (total_improvement / n_improved_objectives) if n_improved_objectives > 0 else 0.0
            print(f"   {frac*100:>5.1f}% restored ({n_restore:>6} pixels): Avg improvement = {avg_improvement*100:>6.2f}%")
            
            # Show per-objective details for first test fraction
            if frac == test_fractions[0]:
                print(f"      Per-objective details:")
                for j, obj_name in enumerate(problem.objective_names):
                    baseline_val = out_none['F'][j]
                    restored_val = out['F'][j]
                    change = restored_val - baseline_val
                    print(f"         {obj_name}: {baseline_val:.6f} → {restored_val:.6f} (Δ={change:.6f})")

# =============================================================================

class RestorationProblem(ElementwiseProblem):
    """
    Multi-objective restoration optimization problem.
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
        #self._eval_t0 = perf_counter()
        self._t0_wall = None
        self._eval_n = 0
        #self._eval_print_every = 10   # tune, 1, 5, 10, 25
        self._slow_eval_seconds = 10.0   # print only if an eval exceeds this
        self._proc = None
        
        # Extract effect parameters from scenario_params (excluding fixed spatial parameters)
        abiotic_effect = scenario_params.get('abiotic_effect', 0.01)
        biotic_effect = scenario_params.get('biotic_effect', 0.01)
        #landscape_effect = scenario_params.get('landscape_effect', 0.01)
        self.effect_params = {
            'abiotic_effect': abiotic_effect,
            'biotic_effect': biotic_effect,
            'neighbor_radius': 3,  # Fixed value
            'neighbor_effect_decay': 0.2  # Fixed value
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
        
        n_objectives = len(self.objective_names)
        if n_objectives == 0:
            raise ValueError("No objectives found in initial_conditions")

        # Objective normalization configuration.
        # Optimization uses normalized objectives in _evaluate; raw objectives are
        # still computed and can be exported for reporting/interpretation.
        self.normalize_objectives = bool(scenario_params.get('normalize_objectives', True))
        self.objective_scales = self._build_objective_scales()
        
        n_restoration_pixels = initial_conditions['n_restoration_pixels']
        n_conversion_pixels = initial_conditions['n_conversion_pixels']
        max_restoration_fraction = scenario_params['max_restoration_fraction']
        
        # Calculate max action pixels based on restoration eligible pixels
        self.max_action_pixels = int(max_restoration_fraction * n_restoration_pixels)
        
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
    def _mem_gb(self):
        if self._proc is None:
            self._proc = psutil.Process(os.getpid())
        rss = self._proc.memory_info().rss
        return rss / (1024**3)

    def _build_objective_scales(self):
        """Build per-objective positive scales for objective normalization."""
        eps = 1e-12
        scales = {}

        rest_mask = self.initial_conditions.get("restoration_eligible_mask")
        conv_mask = self.initial_conditions.get("conversion_eligible_mask")

        for obj_name in self.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                base = self.initial_conditions[obj_name]
                if rest_mask is not None:
                    scale = float(np.nansum(np.abs(base[rest_mask])))
                else:
                    scale = float(np.nansum(np.abs(base)))
            elif obj_name == 'landscape_anomaly':
                l0 = self.initial_conditions['landscape_anomaly']
                scale = float(np.nansum(np.abs(l0)))
            elif obj_name == 'implementation_cost':
                c = self.initial_conditions['implementation_cost']
                if rest_mask is not None and conv_mask is not None:
                    scale = float(np.nansum(np.abs(c[rest_mask])) + np.nansum(np.abs(c[conv_mask])))
                else:
                    scale = float(np.nansum(np.abs(c)))
            else:
                scale = 1.0

            if (not np.isfinite(scale)) or scale <= eps:
                scale = 1.0
            scales[obj_name] = scale

        return scales

    def _normalize_objective_vector(self, raw_objectives):
        """Normalize raw objective vector using precomputed scales."""
        if not self.normalize_objectives:
            return raw_objectives

        normalized = []
        for i, obj_name in enumerate(self.objective_names):
            scale = self.objective_scales.get(obj_name, 1.0)
            normalized.append(float(raw_objectives[i]) / float(scale))
        return normalized

    def evaluate_raw_objectives(self, x):
        """Return raw (non-normalized) objective values for a decision vector."""
        x_restore = x[:self.n_restoration_pixels]
        x_convert = x[self.n_restoration_pixels:self.n_restoration_pixels + self.n_conversion_pixels]

        # Conversion objectives only valid when landscape objective is present.
        if 'landscape_anomaly' not in self.initial_conditions:
            x_convert = x_convert.copy()
            x_convert[:] = 0

        updated_conditions = restoration_effect(x_restore, x_convert, self.initial_conditions, self.effect_params)

        raw_objectives = []
        for obj_name in self.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                base = self.initial_conditions[obj_name]
                mask = self.initial_conditions["restoration_eligible_mask"]
                obj_value = -np.sum((updated_conditions[obj_name] - base)[mask])
            elif obj_name == 'landscape_anomaly':
                l0 = self.initial_conditions["landscape_anomaly"]
                l1 = updated_conditions["landscape_anomaly"]
                eps = 1e-12
                obj_value = np.sum(l1 - l0) / (np.sum(l0) + eps)
            elif obj_name == 'implementation_cost':
                obj_value = updated_conditions[obj_name]
            else:
                raise ValueError(f"Unknown objective: {obj_name}")

            raw_objectives.append(float(obj_value))

        return raw_objectives

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
        self._eval_n += 1

        if self._t0_wall is None:
            self._t0_wall = time()

        t_eval0 = time()
        mem0 = self._mem_gb()

        # Split decision vector into restore and convert actions
        t0 = time()
        x_restore = x[:self.n_restoration_pixels]
        x_convert = x[self.n_restoration_pixels:self.n_restoration_pixels + self.n_conversion_pixels]
        t_split = time() - t0
        
        # Enable conversion actions now that we have fast landscape calculation
        if 'landscape_anomaly' not in self.initial_conditions:
            # Force convert actions to zero if landscape objective not available
            x_convert[:] = 0
        
        # Use both restoration and conversion decisions
        n_restored = np.sum(x_restore)
        n_converted = np.sum(x_convert)
        n_total_actions = n_restored + n_converted
        
        # Compute raw objectives then normalize for optimization.
        t0 = time()
        raw_objectives = self.evaluate_raw_objectives(x)
        objectives = self._normalize_objective_vector(raw_objectives)
        out["F"] = objectives
        out["F_raw"] = raw_objectives
        
        # Constraint: total number of pixels with actions (restore + convert)
        out["G"] = [abs(n_total_actions - self.max_action_pixels)]  # Should be 0 due to exact count enforcement
        
        # Log constraint violations
        if not hasattr(self, 'constraint_log'):
            self.constraint_log = []
        
        # Log constraint violations (G[0] > 0 means violation)
        for i, g_val in enumerate(out["G"]):
            if g_val > 0:
                self.constraint_log.append({
                    'generation': getattr(self, 'current_gen', 0), 
                    'violation_value': g_val,
                    'constraint_type': 'budget'
                })


# =============================================================================
# PATCH-BASED RESTORATION PROBLEM
# =============================================================================

class PatchRestorationProblem(RestorationProblem):
    """
    Patch-based restoration optimization problem.
    
    Decision variables are at the patch level (groups of pixels), but objectives
    are calculated at the pixel level using the existing restoration_effect() logic.
    
    This provides a coarser decision space while maintaining pixel-level accuracy
    for objective calculations.
    
    Constraint types:
    - 'patch_count': Fixed number of patches (variable pixel count)
    - 'pixel_count': Fixed number of pixels (variable patch count) - RECOMMENDED
    """
    
    def __init__(self, initial_conditions, scenario_params, n_jobs=None,
                 patch_constraint_type='pixel_count', pixel_tolerance=0.05):
        """
        Initialize the patch-based optimization problem.
        
        Args:
            initial_conditions: Dict with initial conditions including patch_mappings
            scenario_params: Dict with scenario parameters
            n_jobs: Number of parallel jobs to use
            patch_constraint_type: 'patch_count' or 'pixel_count'
            pixel_tolerance: Tolerance for pixel_count constraint (default 0.05 = ±5%)
        """
        # Check if patch approach is initialized
        if not initial_conditions.get('patch_approach_enabled', False):
            raise ValueError(
                "Patch approach not initialized. Call initialize_patch_approach() first."
            )
        
        # Store patch mapping information
        self.patch_mappings = initial_conditions['patch_mappings']
        self.restoration_patches = self.patch_mappings['restoration_patches']
        self.conversion_patches = self.patch_mappings['conversion_patches']
        
        self.n_restoration_patches = initial_conditions['n_restoration_patches']
        self.n_conversion_patches = initial_conditions['n_conversion_patches']
        
        # Initialize parent class with pixel-level information
        # This sets up all the objective functions and constraints
        # We'll override n_var after parent initialization
        super().__init__(initial_conditions, scenario_params, n_jobs=n_jobs)
        
        # Override decision variable dimensions to use patches instead of pixels
        # Decision vector structure: [restoration_patches, conversion_patches]
        self.n_var = self.n_restoration_patches + self.n_conversion_patches
        
        # Store constraint configuration
        self.patch_constraint_type = patch_constraint_type
        self.pixel_tolerance = pixel_tolerance
        
        # Calculate target values for different constraint types
        if patch_constraint_type == 'pixel_count':
            self.target_constraint_value = self.max_action_pixels
        elif patch_constraint_type == 'patch_count':
            # Estimate max patches from max pixels
            if self.n_restoration_patches > 0:
                avg_pixels_per_patch = (
                    initial_conditions['n_restoration_pixels'] / 
                    self.n_restoration_patches
                )
                self.target_constraint_value = max(1, int(
                    self.max_action_pixels / avg_pixels_per_patch
                ))
            else:
                self.target_constraint_value = 0
        
        print(f"Patch-based problem initialized:")
        print(f"  Decision variables: {self.n_var} patches "
              f"({self.n_restoration_patches} restoration + {self.n_conversion_patches} conversion)")
        print(f"  Constraint type: {patch_constraint_type}")
        print(f"  Target value: {self.target_constraint_value}")
        if patch_constraint_type == 'pixel_count':
            print(f"  Tolerance: ±{pixel_tolerance*100:.1f}%")
        print(f"  (pixel-based equivalent: {self.max_action_pixels} pixels)")
    
    def _evaluate(self, x_patches, out, *args, **kwargs):
        """
        Evaluate a patch-based solution.
        
        Args:
            x_patches: Patch-level decision variables (binary array)
                      First n_restoration_patches: restoration patch decisions
                      Next n_conversion_patches: conversion patch decisions
            out: Output dictionary for objectives and constraints
        """
        from patch_approach import convert_patch_decisions_to_pixels
        
        # Split patch decisions into restoration and conversion
        x_restore_patches = x_patches[:self.n_restoration_patches]
        x_convert_patches = x_patches[self.n_restoration_patches:]
        
        # Convert patch-level decisions to pixel-level decisions
        x_restore_pixels = convert_patch_decisions_to_pixels(
            x_restore_patches,
            self.restoration_patches,
            self.n_restoration_pixels
        )
        
        x_convert_pixels = convert_patch_decisions_to_pixels(
            x_convert_patches,
            self.conversion_patches,
            self.n_conversion_pixels
        )
        
        # Create combined pixel-level decision vector for parent class evaluation
        x_pixels = np.concatenate([x_restore_pixels, x_convert_pixels])
        
        # Use parent class evaluation with pixel-level decisions
        super()._evaluate(x_pixels, out, *args, **kwargs)
        
        # Override constraint based on constraint type
        if self.patch_constraint_type == 'patch_count':
            n_patches_used = np.sum(x_restore_patches) + np.sum(x_convert_patches)
            constraint_value = abs(n_patches_used - self.target_constraint_value)
        
        elif self.patch_constraint_type == 'pixel_count':
            # Count actual restoration + conversion pixels separately
            n_restore_pixels = np.sum(x_restore_pixels)
            n_convert_pixels = np.sum(x_convert_pixels)
            n_pixels_used = n_restore_pixels + n_convert_pixels
            
            # Use LARGER tolerance for constraint evaluation than repair uses
            # This accounts for discretization errors from whole-patch constraints
            # that repair cannot perfectly fix
            evaluation_tolerance = self.pixel_tolerance * 1.5  # 50% larger tolerance
            
            min_pixels = int(self.target_constraint_value * (1 - evaluation_tolerance))
            max_pixels = int(self.target_constraint_value * (1 + evaluation_tolerance))
            
            # DEBUG: Print constraint check details
            if not hasattr(self, '_constraint_debug_count'):
                self._constraint_debug_count = 0
            if self._constraint_debug_count < 5:
                print(f"  DEBUG CONSTRAINT: target={self.target_constraint_value}, tolerance={evaluation_tolerance:.3f}, range=[{min_pixels}, {max_pixels}]")
                print(f"                    restore={n_restore_pixels}, convert={n_convert_pixels}, total={n_pixels_used}")
                self._constraint_debug_count += 1
            
            if min_pixels <= n_pixels_used <= max_pixels:
                constraint_value = 0  # Accept as feasible
            else:
                # Penalize violations outside the evaluation tolerance
                constraint_value = min(
                    abs(n_pixels_used - min_pixels),
                    abs(n_pixels_used - max_pixels)
                )
                if self._constraint_debug_count <= 5:
                    print(f"                    VIOLATION: G={constraint_value}")
        
        out["G"] = [constraint_value]

    def evaluate_raw_objectives(self, x_patches):
        """Return raw objectives for patch-level decisions via pixel conversion."""
        from patch_approach import convert_patch_decisions_to_pixels

        x_restore_patches = x_patches[:self.n_restoration_patches]
        x_convert_patches = x_patches[self.n_restoration_patches:]

        x_restore_pixels = convert_patch_decisions_to_pixels(
            x_restore_patches,
            self.restoration_patches,
            self.n_restoration_pixels
        )
        x_convert_pixels = convert_patch_decisions_to_pixels(
            x_convert_patches,
            self.conversion_patches,
            self.n_conversion_pixels
        )
        x_pixels = np.concatenate([x_restore_pixels, x_convert_pixels])
        return super().evaluate_raw_objectives(x_pixels)


# =============================================================================
# EXECUTION
# =============================================================================

def build_fixed_ref_point(problem, sampling, n_samples=200, margin=0.05, seed=42, verbose=False):
    """
    Build a fixed hypervolume reference point using a warm up sample of solutions.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Use the provided Sampling operator to generate candidate solutions
    # This returns a Population in pymoo, so we extract X
    pop = sampling.do(problem, n_samples)
    X = pop.get("X")

    F_list = []
    for k in range(X.shape[0]):
        out = {}
        problem._evaluate(X[k], out)
        F_list.append(out["F"])
        if verbose and (k + 1) % 25 == 0:
            print(f"  Warm-up ref point evaluation: {k + 1}/{X.shape[0]}")

    Fw = np.asarray(F_list, dtype=float)

    worst = np.max(Fw, axis=0)
    span = np.maximum(np.ptp(Fw, axis=0), 1e-12)

    ref_point = worst + margin * span
    return ref_point


class HVCallback:
    """
    Hypervolume-based early stopping callback.
    Monitors hypervolume improvement and stops optimization if no significant improvement 
    is observed for a specified number of generations.
    """
    
    def __init__(self, patience=15, min_improvement=1e-6, verbose=True, ref_point=None):
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
        self.ref_point = ref_point
        self.hv_history = []
        self.best_hv = 0.0
        self.no_improvement_count = 0
        self.converged = False
        
        # Population statistics tracking
        self.f_mean_history = []  # Mean of F per generation
        self.f_std_history = []   # Std of F per generation
        self.f_min_history = []   # Min of F per generation  
        self.f_max_history = []   # Max of F per generation
    
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
                # Calculate and store population statistics
                f_mean = np.mean(F, axis=0)  # Mean per objective
                f_std = np.std(F, axis=0)    # Std per objective
                f_min = np.min(F, axis=0)    # Min per objective
                f_max = np.max(F, axis=0)    # Max per objective
                
                self.f_mean_history.append(f_mean.tolist())
                self.f_std_history.append(f_std.tolist())
                self.f_min_history.append(f_min.tolist())
                self.f_max_history.append(f_max.tolist())
                
                # Calculate hypervolume
                try:
                    # Create reference point (worst case for each objective)
                    if self.ref_point is None:
                        raise ValueError("HVCallback requires a fixed ref_point")

                    hv_indicator = HV(ref_point=self.ref_point)
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

def run_one(initial_conditions, scenario_params, run_settings):
    """
    Thin wrapper around run_single_scenario_optimization.
    Keeps old behaviour but centralises the call site.
    """
    return run_single_scenario_optimization(
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
        patch_size=run_settings.get("patch_size", 100)
    )

def run_scenario_batch(
    initial_conditions,
    scenario_combinations,
    run_settings,
):
    """
    Runs scenarios using run_one wrapper.
    Returns all_results dict keyed by scenario index.
    """

    all_results = {}

    verbose = run_settings.get("verbose", True)

    if verbose:
        print("\n=== MULTI SCENARIO OPTIMIZATION ===")
        print(f"Total scenarios to run: {len(scenario_combinations)}")

    for i, scenario_params in enumerate(scenario_combinations):

        if verbose:
            print(f"\n--- RUNNING SCENARIO {i+1}/{len(scenario_combinations)} ---")

        result = run_one(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            run_settings=run_settings
        )

        if result is not None:
            all_results[i] = result

            if verbose:
                print(f"✓ Scenario {i} completed: {result['n_solutions']} solutions found")
        else:
            if verbose:
                print(f"✗ Scenario {i} failed")

    return all_results


def _filter_initial_conditions_for_return(initial_conditions):
    """
    Keeps current behaviour, remove heavy geodataframe from admin_data if present.
    """
    initial_conditions_filtered = initial_conditions.copy()

    if (
        "admin_data" in initial_conditions_filtered
        and initial_conditions_filtered["admin_data"] is not None
    ):
        admin_data_filtered = initial_conditions_filtered["admin_data"].copy()
        if "gdf" in admin_data_filtered:
            del admin_data_filtered["gdf"]
        initial_conditions_filtered["admin_data"] = admin_data_filtered

    return initial_conditions_filtered

def build_combined_results(
    initial_conditions,
    all_results,
    n_samples_per_param=3,
    random_seed=42,
    pop_size=50,
    n_generations=100
):
    """
    Returns the same combined_results structure you currently build.
    """
    initial_conditions_filtered = _filter_initial_conditions_for_return(initial_conditions)

    combined_results = {
        "scenarios": all_results,
        "n_scenarios_run": len(all_results),
        "n_scenarios_total": len(expand_scenarios(n_samples_per_param, random_seed)),
        "scenario_parameters": define_scenario_parameters(),
        "n_samples_per_param": n_samples_per_param,
        "random_seed": random_seed,
        "algorithm_info": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "timestamp": datetime.now().isoformat()
        },
        "initial_conditions": initial_conditions_filtered
    }

    return combined_results

def finalise_combined_results(combined_results, save_results=True, verbose=True):
    """
    Keeps current saving behaviour and conditions.
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
        verbose=verbose
    )

def run_single_scenario_optimization(initial_conditions, scenario_params, pop_size=50, 
                                   n_generations=100, save_results=True, verbose=True, skip_diagnostics=False,
                                   hv_patience=15, hv_min_improvement=1e-6, n_jobs=None, use_repair=True, 
                                   random_seed=None, use_patch_approach=False, patch_size=100,
                                   patch_constraint_type='pixel_count', pixel_tolerance=0.05, output_dir="."):
    """
    Run the multi-objective restoration optimization for a single scenario.
    
    Args:
        initial_conditions: Initial objective conditions
        scenario_params: Dict with scenario parameters
        pop_size: Population size for NSGA-II
        n_generations: Number of optimization generations
        save_results: Whether to save results to files
        verbose: Print progress information
        skip_diagnostics: Skip optimization setup diagnostics
        hv_patience: Generations to wait for hypervolume improvement before stopping
        hv_min_improvement: Minimum hypervolume improvement to reset patience counter
        n_jobs: Number of parallel jobs (None = use all cores)
        use_repair: Whether to use repair operator for constraints
        random_seed: Random seed for reproducibility
        use_patch_approach: Use patch-based optimization instead of pixel-based
        patch_size: Size of patches in pixels (for patch approach)
        patch_constraint_type: Constraint type for patch approach:
            - 'pixel_count': Constrain total number of pixels (RECOMMENDED for fair comparison)
            - 'patch_count': Constrain number of patches (original, less fair)
        pixel_tolerance: Tolerance for pixel_count constraint (default 0.05 = ±5%)
        skip_diagnostics: Skip diagnostic output (useful for multi-scenario runs)
        hv_patience: Generations to wait for hypervolume improvement before stopping
        hv_min_improvement: Minimum relative hypervolume improvement threshold
        n_jobs: Number of parallel jobs to use (None for automatic, 1 for serial, -1 for all cores)
        random_seed: Random seed for reproducibility (None uses random initialization)
        use_patch_approach: If True, use patch-based decision vector; if False, use pixel-based (default)
        patch_size: Size of patches in pixels (e.g., 100 = 100x100 patches). Only used if use_patch_approach=True
        output_dir: Directory to save results (default: current directory)
        
    Returns:
        dict: Optimization results
    """
    # Auto-initialize patch approach if requested and not already initialized
    if use_patch_approach and not initial_conditions.get('patch_approach_enabled', False):
        if verbose:
            print(f"Initializing patch approach with patch_size={patch_size}...")
        initial_conditions = initialize_patch_approach(initial_conditions, patch_size=patch_size)
    
    if verbose:
        approach_str = "PATCH-BASED" if use_patch_approach else "PIXEL-BASED"
        print(f"\n=== SINGLE SCENARIO OPTIMIZATION ({approach_str}) ===") 
        print(f"Scenario parameters: {scenario_params}")
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_generations} (with HV early stopping: patience={hv_patience})")
        #print(f"Max restoration: {scenario_params['max_restoration_fraction']*100:.1f}% of eligible area")
        
        if use_patch_approach:
            print(f"Eligible patches: {initial_conditions['n_restoration_patches']} restoration + "
                  f"{initial_conditions['n_conversion_patches']} conversion")
        else:
            print(f"Eligible pixels: {initial_conditions['n_pixels']}")
        
        # Show which objectives are being used
        #problem_temp = RestorationProblem(initial_conditions, scenario_params, n_jobs=1)  # Use single process for setup
        #print(f"Objectives: {problem_temp.objective_names} ({len(problem_temp.objective_names)} total)")
    
    # Create optimization problem - choose between pixel and patch based approach
    # Force serial execution to avoid pickle errors
    # Parallelization at problem level conflicts with pymoo's own parallelization
    if use_patch_approach:
        problem = PatchRestorationProblem(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            n_jobs=1,  # Serial execution - parallelization handled by pymoo if needed
            patch_constraint_type=patch_constraint_type,
            pixel_tolerance=pixel_tolerance
        )
    else:
        problem = RestorationProblem(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            n_jobs=1  # Serial execution - parallelization handled by pymoo if needed
        )
    import numpy as np
    # Print additional diagnostic information
    if verbose:
        print(f"\nOptimization setup details:")
        print(f"  Max action pixels allowed: {problem.max_action_pixels}")
        print(f"  Number of objectives: {len(problem.objective_names)} ({', '.join(problem.objective_names)})")
        
        # Sample objective values without restoration
        sample_obj_str = "  Baseline objectives (no restoration): "
        for obj_name in problem.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
                val = np.sum(initial_conditions[obj_name])
                sample_obj_str += f"{obj_name}={val:.2e}, "
        print(sample_obj_str.rstrip(", "))
        
        if problem.max_action_pixels == 0:
            print("  WARNING: max_action_pixels is 0! No actions possible.")
        
        # Only run diagnostics if not skipped
        if not skip_diagnostics:
            #print("\nRunning diagnostics to check optimization setup...")
            diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10)
    
    # Create custom sampling and repair operators based on scenario parameters
    burden_sharing = scenario_params.get('burden_sharing', 'no')
    clustering_strength = scenario_params.get('spatial_clustering', 0.0)
    
    # For patch approach, use patch-aware operators
    if use_patch_approach:
        # Build restoration-informed patch scores (used by both sampling and repair).
        restoration_scores = build_repair_scores(initial_conditions, scenario_params)
        patch_scores = aggregate_patch_scores_from_pixel_scores(
            patch_mappings=initial_conditions['patch_mappings'],
            restoration_pixel_scores=restoration_scores,
            conversion_pixel_scores=None,
            mode='mean'
        )

        patch_score_temperature = float(scenario_params.get('patch_score_temperature', 0.25))
        patch_random_share = float(scenario_params.get('patch_random_share', 0.15))
        patch_repair_top_k = int(scenario_params.get('patch_repair_top_k', 12))

        # Use intelligent sampling that respects target pixel count
        sampling = PatchAwareSampling(
            patch_mappings=initial_conditions['patch_mappings'],
            target_pixels=problem.target_constraint_value,
            pixel_tolerance=pixel_tolerance,
            patch_scores=patch_scores,
            score_temperature=patch_score_temperature,
            random_share=patch_random_share,
        )
        
        # Create PatchRepair with appropriate constraint type
        if use_repair:
            repair = PatchRepair(
                constraint_type=patch_constraint_type,
                target_value=problem.target_constraint_value,
                patch_mappings=initial_conditions['patch_mappings'],
                pixel_tolerance=pixel_tolerance,
                patch_scores=patch_scores,
                score_temperature=patch_score_temperature,
                top_k=patch_repair_top_k,
            )
        else:
            repair = None
        
        if verbose:
            print(
                "Using stochastic score-guided patch operators "
                f"({patch_constraint_type}, temp={patch_score_temperature}, "
                f"random_share={patch_random_share}, top_k={patch_repair_top_k})"
            )
    else:
        # Build scores for repair operator (pixel-based only)
        scores = build_repair_scores(initial_conditions, scenario_params)
        
        # Use consolidated operators for pixel-based scenarios
        sampling = AdaptiveSampling(
            initial_conditions, 
            problem.max_action_pixels, 
            scenario_params
        )
        repair = AdaptiveRepair(
            initial_conditions, 
            problem.max_action_pixels, 
            scenario_params, 
            scores
        ) if use_repair else None
        
        if verbose:
            strategy_desc = []
            if burden_sharing == 'yes':
                strategy_desc.append("burden-sharing")
            if clustering_strength > 0.0:
                strategy_desc.append(f"clustering({clustering_strength})")
            if not strategy_desc:
                strategy_desc.append("random with score-based repair")
            print(f"Using adaptive operators: {', '.join(strategy_desc)}")
    
    hv_warmup_samples = int(
        scenario_params.get(
            "hv_warmup_samples",
            40 if use_patch_approach else 200
        )
    )
    if verbose:
        print(f"Building fixed HV ref point with {hv_warmup_samples} warm-up samples...")

    fixed_ref = build_fixed_ref_point(
        problem=problem,
        sampling=sampling,
        n_samples=hv_warmup_samples,
        margin=0.05,
        seed=42
        ,verbose=verbose
    )
    
    # Create optimization algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,                # Custom or standard sampling
        crossover=HUX(),                  # Half-uniform crossover
        mutation=BitflipMutation(prob=0.1), #increased from 0.05 5.02.2026      # Bit-flip mutation
        repair=repair                     # Custom repair to maintain properties
    )
    
    # Diagnostic: Check initial population quality for patch approach
    if use_patch_approach and verbose:
        # Directly call _do to get numpy array (avoid Individual wrapping)
        test_sample = sampling._do(problem, 5)
        if hasattr(problem, 'patch_mappings'):
            patch_mappings = problem.patch_mappings
            pixel_counts = []
            for i in range(test_sample.shape[0]):
                x = test_sample[i]
                pixels = 0
                for j, selected in enumerate(x[:problem.n_restoration_patches]):
                    if selected == 1:
                        pixels += len(patch_mappings['restoration_patches']['patch_to_pixels'][j])
                for j, selected in enumerate(x[problem.n_restoration_patches:]):
                    if selected == 1:
                        pixels += len(patch_mappings['conversion_patches']['patch_to_pixels'][j])
                pixel_counts.append(pixels)
            target_min = int(problem.target_constraint_value * (1 - pixel_tolerance))
            target_max = int(problem.target_constraint_value * (1 + pixel_tolerance))
            print(f"  Initial sampling check: {len([p for p in pixel_counts if target_min <= p <= target_max])}/5 within target [{target_min}, {target_max}]")
    
    # Set up termination (use standard generation-based termination)
    termination = get_termination("n_gen", n_generations)
    
    # Progress callback
    class ProgressCallback:
        def __init__(self, verbose=True):
            self.verbose = verbose
            self.start_time = None
            self.hv_callback = HVCallback(patience=hv_patience, min_improvement=hv_min_improvement, verbose=verbose, ref_point=fixed_ref)
            
        def __call__(self, algorithm):
            if self.start_time is None:
                self.start_time = datetime.now()
            
            # Call hypervolume callback for early stopping check
            self.hv_callback(algorithm)
            
            # Update current generation in problem for constraint logging
            if hasattr(algorithm, 'problem'):
                algorithm.problem.current_gen = algorithm.n_gen
            
            gen = algorithm.n_gen
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.verbose and gen % 10 == 0:
                progress = (gen / n_generations) * 100
                eta = (elapsed / gen) * (n_generations - gen) if gen > 0 else 0
                
                # Report constraint violations
                violation_info = ""
                if hasattr(algorithm, 'problem') and hasattr(algorithm.problem, 'constraint_log'):
                    gen_violations = [c for c in algorithm.problem.constraint_log 
                                    if c.get('generation', 0) == gen]
                    if gen_violations:
                        violation_info = f" - Violations: {len(gen_violations)}"
                
                print(f"   Generation {gen}/{n_generations} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed/60:.1f}min - ETA: {eta/60:.1f}min{violation_info}")
            
            # Stop optimization if hypervolume has converged
            # TEMPORARILY DISABLED: Hypervolume-based early stopping
            # if self.hv_callback.converged:
            #     algorithm.termination.force_termination = True
    
    # Run optimization
    if verbose:
        from datetime import datetime
        print(f"Starting optimisation at {datetime.now():%H:%M}...")
    
    try:
        callback = ProgressCallback(verbose=verbose)
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=random_seed,  # Use provided seed for reproducibility
            verbose=False,
            callback=callback
        )
        
        if verbose:
            #print(f"\nDEBUG - Result object:")
            #print(f"  result is None: {result is None}")
            if result is not None:
                # Check final population
                if hasattr(result, 'pop') and result.pop is not None:
                    if len(result.pop) > 0:
                        pop_F = result.pop.get("F")
                        #if pop_F is not None:
                            #print(f"\nSample objectives (first 3 solutions):")
                            #for i in range(min(3, len(pop_F))):
                                #print(f"    Sol {i}: {pop_F[i]}")
                            
                            # Check for diversity
                            #print(f"Objective statistics across population:")
                            #for j, obj_name in enumerate(problem.objective_names):
                            #    obj_vals = pop_F[:, j]
                            #    print(f"    {obj_name}: min={np.min(obj_vals):.4e}, max={np.max(obj_vals):.4e}, std={np.std(obj_vals):.4e}")
                            
                            # Check correlation between objectives
                            #if pop_F.shape[1] > 1:
                            #    print(f"Objective correlations (high correlation = no trade-offs):")
                            #    for i in range(pop_F.shape[1]):
                            #        for j in range(i+1, pop_F.shape[1]):
                            #            corr = np.corrcoef(pop_F[:, i], pop_F[:, j])[0, 1]
                            #            print(f"    {problem.objective_names[i]} vs {problem.objective_names[j]}: {corr:.4f}")
        
        if result is not None and hasattr(result, 'F') and result.F is not None and len(result.F) > 0:
            if verbose:
                convergence_reason = "hypervolume plateau" if callback.hv_callback.converged else "generation limit"
                final_gen = len(callback.hv_callback.hv_history)
                print(f"✓ Optimization completed after {final_gen} generations ({convergence_reason})")
                if callback.hv_callback.hv_history:
                    print(f"Final hypervolume: {callback.hv_callback.hv_history[-1]:.6f}")
                print(f"Found {len(result.F)} Pareto-optimal solutions")
                import numpy as np

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
                    pop_F_raw = np.asarray([problem.evaluate_raw_objectives(xi) for xi in pop_X], dtype=float)
                    full_population_data = {
                        'objectives': pop_F_raw,  # Raw full population objectives (for reporting)
                        'objectives_normalized': pop_F,  # Normalized objectives used by optimizer
                        'decisions': pop_X,   # Full population decisions
                        'population_size': len(pop_F)
                    }  
            
            # Analyze decision patterns
            #decision_analysis = analyze_decision_patterns(result.X, initial_conditions)
            #decision_analysis['constraints']['restoration_budget_fraction'] = scenario_params.get('max_restoration_fraction', 0.0)
            
            objectives_normalized = np.asarray(result.F, dtype=float)
            objectives_raw = np.asarray([problem.evaluate_raw_objectives(xi) for xi in result.X], dtype=float)

            optimization_results = {
                'scenario_params': scenario_params,
                'objective_names' : problem.objective_names,
                'objectives': objectives_raw,      # Raw objective values (for reporting/interpretation)
                'objectives_raw': objectives_raw,
                'objectives_normalized': objectives_normalized,  # Used during optimization
                'decisions': result.X,            # Decision variables (restoration plans)
                #'decision_analysis': decision_analysis,  # Analysis of restore/convert patterns
                'n_solutions': len(result.F),
                'problem_info': {
                    'n_pixels': initial_conditions['n_pixels'],
                    'n_restoration_pixels': initial_conditions.get('n_restoration_pixels', 0),
                    'n_conversion_pixels': initial_conditions.get('n_conversion_pixels', 0),
                    'max_action_pixels': problem.max_action_pixels,
                    'is_patch_based': use_patch_approach,
                    'n_restoration_patches': initial_conditions.get('n_restoration_patches', 0) if use_patch_approach else None,
                    'n_conversion_patches': initial_conditions.get('n_conversion_patches', 0) if use_patch_approach else None,
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
                    'population_statistics': {
                        'f_mean_history': callback.hv_callback.f_mean_history,
                        'f_std_history': callback.hv_callback.f_std_history,
                        'f_min_history': callback.hv_callback.f_min_history,
                        'f_max_history': callback.hv_callback.f_max_history
                    },
                    'hv_patience': hv_patience,
                    'hv_min_improvement': hv_min_improvement,
                    'sampling_method': 'adaptive',
                    'repair_operators': ['score_based_repair', 'random_repair'],
                    'objective_normalization': {
                        'enabled': bool(problem.normalize_objectives),
                        'scales': {k: float(v) for k, v in problem.objective_scales.items()}
                    },
                    'timestamp': datetime.now().isoformat()
                },
                'initial_conditions': initial_conditions_filtered,
                'hv_callback': callback.hv_callback,# Store for evolution tracking
                'full_population': full_population_data  
            
        }
            
            # Add patch mappings if using patch approach
            if use_patch_approach and 'patch_mappings' in initial_conditions:
                optimization_results['patch_mappings'] = initial_conditions['patch_mappings']
            
            # Save results with comprehensive reporting if requested
            if save_results:
                # Use simplified reporting - generates all essential reports
                save_results_with_reports(optimization_results, output_dir=output_dir, verbose=verbose)
            
            return optimization_results
            
        else:
            if verbose:
                print("✗ Optimization failed - no solutions found")
                # Diagnostic: why did it fail?
                if result is None:
                    print("  Reason: result is None")
                elif not hasattr(result, 'F'):
                    print("  Reason: result has no F attribute")
                elif result.F is None:
                    print("  Reason: result.F is None")
                    # Check the population for feasibility
                    if hasattr(result, 'pop') and result.pop is not None and len(result.pop) > 0:
                        pop_G = result.pop.get("G")
                        pop_X = result.pop.get("X")
                        if pop_G is not None:
                            n_feasible = np.sum(np.all(pop_G <= 0, axis=1))
                            n_total = len(pop_G)
                            print(f"  Final population: {n_total} solutions, {n_feasible} feasible")
                            if n_feasible == 0 and pop_X is not None:
                                # Check actual pixel counts for first few solutions
                                print(f"  Checking actual pixel counts for first 5 solutions:")
                                for i in range(min(5, len(pop_X))):
                                    x_patches = pop_X[i]
                                    # Count pixels
                                    pixels = 0
                                    patch_mappings = initial_conditions['patch_mappings']
                                    for j in range(problem.n_restoration_patches):
                                        if x_patches[j] == 1:
                                            pixels += len(patch_mappings['restoration_patches']['patch_to_pixels'][j])
                                    for j in range(problem.n_conversion_patches):
                                        if x_patches[problem.n_restoration_patches + j] == 1:
                                            pixels += len(patch_mappings['conversion_patches']['patch_to_pixels'][j])
                                    print(f"    Sol {i}: pixels={pixels}, G={pop_G[i]}")
                elif len(result.F) == 0:
                    print("  Reason: result.F has length 0 (no feasible solutions)")
                    # Check if there's a population
                    if hasattr(result, 'pop') and result.pop is not None:
                        pop_G = result.pop.get("G")
                        if pop_G is not None:
                            n_feasible = np.sum(np.all(pop_G <= 0, axis=1))
                            print(f"  Population: {len(pop_G)} solutions, {n_feasible} feasible")
                            if n_feasible == 0:
                                print(f"  Constraint violations (first 5):")
                                for i in range(min(5, len(pop_G))):
                                    print(f"    Solution {i}: G={pop_G[i]}")
            return None
            
    except Exception as e:
        if verbose:
            print(f"✗ Error during optimization: {e}")
        return None

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
    n_jobs=None
):
    scenario_combinations = expand_scenarios(n_samples_per_param=n_samples_per_param, random_seed=random_seed)

    run_settings = {
        "pop_size": pop_size,
        "n_generations": n_generations,
        "save_results": False,          # ← keep old multi scenario behaviour
        "verbose": verbose,
        "skip_diagnostics": True,
        "hv_patience": hv_patience,
        "hv_min_improvement": hv_min_improvement,
        "n_jobs": n_jobs,
        "use_repair": True
    }

    all_results = run_scenario_batch(
        initial_conditions=initial_conditions,
        scenario_combinations=scenario_combinations,
        run_settings=run_settings,
    )

    combined_results = build_combined_results(
        initial_conditions=initial_conditions,
        all_results=all_results,
        scenario_combinations=scenario_combinations,
        n_samples_per_param=n_samples_per_param,
        random_seed=random_seed,
        pop_size=pop_size,
        n_generations=n_generations
    )

    if verbose:
        print("\n=== ALL SCENARIOS COMPLETE ===")
        print(f"Successfully completed: {len(all_results)}/{len(scenario_combinations)} scenarios")

    finalise_combined_results(combined_results, save_results=save_results, verbose=verbose)

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
        objectives: List of objectives to use (e.g., ['abiotic', 'biotic', 'landscape', 'cost])
                   If None, uses all available objectives
        n_samples_per_param: Number of samples to draw from each continuous parameter
        pop_size: Population size for optimization
        n_generations: Number of optimization generations
        save_results: Whether to save results
        verbose: Print progress information
        random_seed: Random seed for reproducible parameter sampling
        sample_fraction: Fraction of eligible pixels to use for optimization
                        If None, uses all eligible pixels
        sample_seed: Random seed for spatial sampling (default: 42)
        
    Returns:
        dict: Optimization results
    """

    if scenario == "all":
        initial_conditions = load_initial_conditions(
            workspace_dir,
            objectives=objectives,
            region="Bern",
            ecosystem=ecosystem,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            ecosystem_lulc_path=lulc_path,
            landscape_lulc_path=lulc_path,
        )

        return run_all_scenarios_optimization(
            initial_conditions=initial_conditions,
            n_samples_per_param=n_samples_per_param,
            pop_size=pop_size,
            n_generations=n_generations,
            save_results=save_results,
            verbose=verbose,
            random_seed=random_seed
        )

    if verbose:
        print("=== RESTORATION OPTIMIZATION WORKFLOW ===")
        if objectives is not None:
            print(f"Using objectives: {objectives}")
        print(f"Running scenario {scenario}")

    initial_conditions = load_initial_conditions(
        workspace_dir,
        objectives=objectives,
        region="Bern",
        ecosystem=ecosystem,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        ecosystem_lulc_path=lulc_path,
        landscape_lulc_path=lulc_path,
    )

    scenario_combinations = sample_scenario_parameters(n_samples_per_param, random_seed)

    if not isinstance(scenario, int):
        raise ValueError("scenario must be 'all' or an int index")

    if not (0 <= scenario < len(scenario_combinations)):
        raise ValueError(f"Scenario index {scenario} out of range. Available: 0-{len(scenario_combinations)-1}")

    scenario_params = scenario_combinations[scenario]

    return run_one(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params,
        pop_size=pop_size,
        n_generations=n_generations,
        save_results=save_results,
        verbose=verbose,
        skip_diagnostics=False
    )

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Available ecosystem run modes:
    # - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
    # - 'fg': run one optimisation using forest + grassland pixels
    # - 'all': run three separate optimisations (one per ecosystem)
    # - 'combined': run one optimisation without ecosystem filtering
    ECOSYSTEM_TO_RUN = "fg"  # change to a single ecosystem if needed

    # Region used for validation reference in load_initial_conditions
    REGION = "Bern"  # change to 'CH' for Switzerland-wide optimisation

    # Choose scenario mode:
    #   "custom" runs exactly one scenario using custom_scenario_params
    #   "all" runs scenario="all" using the scenario sampling logic inside main()
    SCENARIO_MODE = "custom"  # "custom" or "all"

    print(f"\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
    print(f"Scenario mode: {SCENARIO_MODE}")

    # Shared run controls
    OBJECTIVES = ["abiotic", "biotic","cost"]# "landscape", "cost"]
    SAMPLE_FRACTION = None
    SAMPLE_SEED = 42
    POP_SIZE = 50
    N_GENERATIONS = 100
    N_JOBS = 12
    RANDOM_SEED = 42
    N_SAMPLES_PER_PARAM = 3

    # Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
    custom_scenario_params = {
        "max_restoration_fraction": 0.1,
        "spatial_clustering": 0,
        "biotic_effect": 0.01,
        "abiotic_effect": 0.01,
        "normalize_objectives": True,
    }
    
    # Patch approach settings (optional - set to enable patch-based optimization)
    USE_PATCH_APPROACH = False  # Set to True to use patch-based decision vector
    PATCH_SIZE = 2  # Size of patches in pixels (e.g., 100 = 100x100 patches)
    PATCH_CONSTRAINT_TYPE = 'pixel_count'  # 'pixel_count' (recommended) or 'patch_count'
    PIXEL_TOLERANCE = 0.05  # Tolerance for pixel_count constraint (±5%)

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
                )

                results = run_single_scenario_optimization(
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
                    pixel_tolerance=PIXEL_TOLERANCE
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