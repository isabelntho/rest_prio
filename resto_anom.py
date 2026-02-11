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
    
    return problem

# =============================================================================
# PROBLEM DEFINITION
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
        
        # Apply restoration effects to both restoration and conversion decisions
        t0 = time()
        updated_conditions = restoration_effect(x_restore, x_convert, self.initial_conditions, self.effect_params)
        t_effect = time() - t0
        
        # inside RestorationProblem._evaluate, after updated_conditions is computed

        a0 = self.initial_conditions["abiotic_anomaly"]
        b0 = self.initial_conditions["biotic_anomaly"]
        l0 = self.initial_conditions["landscape_anomaly"] if "landscape_anomaly" in self.initial_conditions else None

        a1 = updated_conditions["abiotic_anomaly"]
        b1 = updated_conditions["biotic_anomaly"]
        l1 = updated_conditions["landscape_anomaly"] if "landscape_anomaly" in updated_conditions else None

        # masks for acted pixels
        shape = self.initial_conditions["shape"]
        rest_mask = np.zeros(shape, dtype=bool)
        conv_mask = np.zeros(shape, dtype=bool)

        if np.any(x_restore):
            idx = self.initial_conditions["restoration_eligible_indices"][x_restore == 1]
            rr, cc = np.divmod(idx, shape[1])
            rest_mask[rr, cc] = True

        if np.any(x_convert):
            idx = self.initial_conditions["conversion_eligible_indices"][x_convert == 1]
            rr, cc = np.divmod(idx, shape[1])
            conv_mask[rr, cc] = True

        cost = updated_conditions.get("implementation_cost", np.nan)

        n_rest = int(np.sum(x_restore))
        n_conv = int(np.sum(x_convert))

        t0 = time()
        # Calculate objective values (all to minimize) - only for available objectives
        objectives = []
        
        for obj_name in self.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                # Anomaly objectives: MINIMIZE negative sum (equivalent to MAXIMIZE sum)
                # Higher anomalies are better, so we want to maximize the sum
                # NSGA-II minimizes, so we minimize the negative
                
                #previous: threshold objective
                #threshold = 0.0  # or different per objective
                #pixels_above = np.sum(updated_conditions[obj_name] > threshold)
                #obj_value = -pixels_above  # Maximize count above threshold
                
                #now, trying a different construction which is to maximise improvement relative to the baseline
                base = self.initial_conditions[obj_name]
                mask = self.initial_conditions["restoration_eligible_mask"]
                obj_value = -np.sum((updated_conditions[obj_name] - base)[mask])
                
            elif obj_name == 'landscape_anomaly':
                # Landscape objective: MINIMIZE the actual sum of landscape anomalies
                # Use the actual values, not just count above threshold
                #landscape_vals = updated_conditions[obj_name]
                #obj_value = np.sum(landscape_vals)  # Minimize total landscape impact
                l0 = self.initial_conditions["landscape_anomaly"]
                l1 = updated_conditions["landscape_anomaly"]
                eps = 1e-12
                obj_value = np.sum(l1 - l0) / (np.sum(l0) + eps)
 
            elif obj_name == 'implementation_cost':
                # Cost objective: minimize total cost (lower is better)
                obj_value = updated_conditions[obj_name]
            else:
                raise ValueError(f"Unknown objective: {obj_name}")
            
            objectives.append(obj_value)
        
        out["F"] = objectives
        
        # Constraint: total number of pixels with actions (restore + convert)
        out["G"] = [abs(n_total_actions - self.max_action_pixels)]  # Should be 0 due to exact count enforcement


# =============================================================================
# EXECUTION
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
    import numpy as np
    # Print additional diagnostic information
    if verbose:
        print(f"\nOptimization setup details:")
        print(f"  Max action pixels allowed: {problem.max_action_pixels}")
        print(f"  Number of objectives: {len(problem.objective_names)}")
        
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
    
    # Build scores for repair operator
    scores = build_repair_scores(initial_conditions, scenario_params)
    
    # Use consolidated operators for all scenarios
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
    )
    
    if verbose:
        strategy_desc = []
        if burden_sharing == 'yes':
            strategy_desc.append("burden-sharing")
        if clustering_strength > 0.0:
            strategy_desc.append(f"clustering({clustering_strength})")
        if not strategy_desc:
            strategy_desc.append("random with score-based repair")
        print(f"Using adaptive operators: {', '.join(strategy_desc)}")
    
    # Create optimization algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,                # Custom or standard sampling
        crossover=HUX(),                  # Half-uniform crossover
        mutation=BitflipMutation(prob=0.1), #increased from 0.05 5.02.2026      # Bit-flip mutation
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
            seed=42, 
            verbose=False,
            callback=callback
        )
        
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
                    full_population_data = {
                        'objectives': pop_F,  # Full population objectives
                        'decisions': pop_X,   # Full population decisions
                        'population_size': len(pop_F)
                    }  
            
            # Analyze decision patterns
            decision_analysis = analyze_decision_patterns(result.X, initial_conditions)
            decision_analysis['constraints']['restoration_budget_fraction'] = scenario_params.get('max_restoration_fraction', 0.0)
            
            optimization_results = {
                'scenario_params': scenario_params,
                'objective_names' : problem.objective_names,
                'objectives': result.F,           # Objective values
                'decisions': result.X,            # Decision variables (restoration plans)
                'decision_analysis': decision_analysis,  # Analysis of restore/convert patterns
                'n_solutions': len(result.F),
                'problem_info': {
                    'n_pixels': initial_conditions['n_pixels'],
                    'n_restoration_pixels': initial_conditions.get('n_restoration_pixels', 0),
                    'n_conversion_pixels': initial_conditions.get('n_conversion_pixels', 0),
                    'max_action_pixels': problem.max_action_pixels
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
            landscape_lulc_path=lulc_path
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
        landscape_lulc_path=lulc_path
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

    # Available ecosystem types: 'all', 'forest', 'agricultural', 'grassland'
    ECOSYSTEM_TO_RUN = "all"  # change to a single ecosystem if needed

    # Region used for validation reference in load_initial_conditions
    REGION = "Bern"  # change to 'CH' for Switzerland-wide optimisation

    # Choose scenario mode:
    #   "custom" runs exactly one scenario using custom_scenario_params
    #   "all" runs scenario="all" using the scenario sampling logic inside main()
    SCENARIO_MODE = "custom"  # "custom" or "all"

    print(f"\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM, REGION {REGION} ===")
    print(f"Scenario mode: {SCENARIO_MODE}")

    # Shared run controls
    OBJECTIVES = ["abiotic", "biotic", "landscape", "cost"]
    SAMPLE_FRACTION = 0.2
    SAMPLE_SEED = 42
    POP_SIZE = 20
    N_GENERATIONS = 30
    N_JOBS = 12
    RANDOM_SEED = 42
    N_SAMPLES_PER_PARAM = 3

    # Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
    custom_scenario_params = {
        "max_restoration_fraction": 0.1,
        "spatial_clustering": 0,
        "burden_sharing": "no",
        "abiotic_effect": 0.01,
        "biotic_effect": 0.01,
    }

    if ECOSYSTEM_TO_RUN == "all":
        ecosystems_to_run = ["forest", "agricultural", "grassland"]
    else:
        ecosystems_to_run = [ECOSYSTEM_TO_RUN]

    all_results = {}

    for ecosystem in ecosystems_to_run:
        print(f"=== Starting optimisation for {ecosystem.upper()} ecosystem ===")

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
                    ecosystem=ecosystem,
                    lulc_path=None
                )
            else:
                initial_conditions = load_initial_conditions(
                    ".",
                    objectives=OBJECTIVES,
                    region=REGION,
                    ecosystem=ecosystem,
                    sample_fraction=SAMPLE_FRACTION,
                    sample_seed=SAMPLE_SEED
                )

                results = run_single_scenario_optimization(
                    initial_conditions=initial_conditions,
                    scenario_params=custom_scenario_params,
                    pop_size=POP_SIZE,
                    n_generations=N_GENERATIONS,
                    save_results=True,
                    verbose=True,
                    n_jobs=N_JOBS
                )

            if results is not None:
                all_results[ecosystem] = results
                print(f"\n✓ {ecosystem.title()} optimisation completed successfully!")
            else:
                print(f"\n✗ {ecosystem.title()} optimisation failed.")

        except Exception as e:
            print(f"\n✗ Error optimising {ecosystem} ecosystem: {e}")
            continue

    print(f"\n{'='*80}")
    print("=== OPTIMISATION SUMMARY ===")
    print(f"{'='*80}")
    successful_runs = len(all_results)
    total_runs = len(ecosystems_to_run)
    print(f"Successfully completed {successful_runs}/{total_runs} ecosystem optimisations:")
    for ecosystem in ecosystems_to_run:
        status = "✓ SUCCESS" if ecosystem in all_results else "✗ FAILED"
        print(f"  {ecosystem.title():<12}: {status}")

    if successful_runs > 0:
        print(f"\n✓ Completed with outputs for {successful_runs} ecosystems.")
    else:
        print("\n✗ No optimisations completed successfully.")
