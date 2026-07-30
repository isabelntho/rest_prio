"""
Restoration Optimization initial code
======================================
Objectives:
1. Abiotic condition anomaly (minimize)
2. Biotic condition anomaly (minimize)  
3. Landscape condition anomaly (minimize) (not used/implementation not finished)
4. Implementation cost (minimize)

Decision variable: Binary (0/1) for not restore/restore
Restoration effect: Triggers action in cell or neighboring cells
"""

# --- Imports ---
import os
import numpy as np
from scipy import ndimage
from datetime import datetime
import threading
from multiprocessing.pool import ThreadPool

from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.operators.crossover.hux import HUX
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .spatial_operations import AdaptiveSampling, AdaptiveRepair, compute_sn_dens_array, InstrumentedBitflipMutation, build_region_assignments_cache, RegionGrowingSampling, RegionGrowingMutation, SpatialCoverageSampling, RegionEvolveMutation, RegionSwapCrossover, MinPatchSizeRepair, WarmStartSampling, grow_region_plan, build_restoration_neighbor_table
from .data_loader import load_initial_conditions
from .results_saving import save_results_with_reports
from .paths import OUTPUT_DIR
from .patch_approach import (
    create_patch_mappings,
    PatchRepair,
    PatchAwareSampling,
    aggregate_patch_scores_from_pixel_scores,
    assign_patches_to_regions,
)
from .scenarios import sample_scenario_parameters
from time import time

# --- Weighting ---

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
    n_elig = int(elig.sum())

    wshape = scenario_params.get("anomaly_weight_shape", "exponential")
    wscale = scenario_params.get("anomaly_weight_scale", 0.5)  # 0.5 spreads score signal across more mildly-degraded pixels vs. old default of 1.0

    scores = np.zeros(n_elig, dtype=np.float64)

    if "abiotic_anomaly" in initial_conditions:
        a0 = initial_conditions["abiotic_anomaly"][elig]
        wa = anomaly_improvement_weight(a0, shape=wshape, scale=wscale)
        sa = float(scenario_params.get("abiotic_effect", 0.0)) * wa
        scores = scores + sa

    if "biotic_anomaly" in initial_conditions:
        b0 = initial_conditions["biotic_anomaly"][elig]
        wb = anomaly_improvement_weight(b0, shape=wshape, scale=wscale)
        sb = float(scenario_params.get("biotic_effect", 0.0)) * wb
        scores = scores + sb

    # Optional: small cost penalty if cost exists in initial_conditions and scenario uses it
    if "implementation_cost" in initial_conditions:
        c = initial_conditions["implementation_cost"][elig].astype(np.float64)
        c = c / (np.nanmean(c) + 1e-12)
        scores = scores - 1e-6 * c

    return np.asarray(scores, dtype=np.float64)


def build_per_objective_repair_scores(initial_conditions, scenario_params):
    """Per-objective pixel scores for direction-aware patch repair.

    Returns a dict with keys 'abiotic', 'biotic', 'cost', and optionally
    'landscape_context'.  Each value is a 1-D float64 array over eligible
    pixels, normalized to [0, 1].  Higher score always means "prefer this
    pixel for the corresponding objective":
      - abiotic:           degradation weight (same as composite abiotic term)
      - biotic:            degradation weight (same as composite biotic term)
      - cost:              inverted, normalised cost  (lower cost → higher score)
      - landscape_context: inverted context score (low neighbour anomaly → high
                           score, i.e. neighbours already in good condition)
    """
    elig = initial_conditions["eligible_mask"]
    n_elig = int(elig.sum())

    wshape = scenario_params.get("anomaly_weight_shape", "exponential")
    wscale = scenario_params.get("anomaly_weight_scale", 0.5)

    if "abiotic_anomaly" in initial_conditions:
        a0 = initial_conditions["abiotic_anomaly"][elig]
        wa = anomaly_improvement_weight(a0, shape=wshape, scale=wscale)
    else:
        wa = np.zeros(n_elig, dtype=np.float64)

    if "biotic_anomaly" in initial_conditions:
        b0 = initial_conditions["biotic_anomaly"][elig]
        wb = anomaly_improvement_weight(b0, shape=wshape, scale=wscale)
    else:
        wb = np.zeros(n_elig, dtype=np.float64)

    if "implementation_cost" in initial_conditions:
        c = initial_conditions["implementation_cost"][elig].astype(np.float64)
        c_min, c_max = np.nanmin(c), np.nanmax(c)
        span = c_max - c_min
        if span > 1e-12:
            cost_score = 1.0 - (c - c_min) / span
        else:
            cost_score = np.full(len(c), 0.5, dtype=np.float64)
    else:
        cost_score = np.full(len(wa), 0.5, dtype=np.float64)

    scores = {
        "abiotic": np.asarray(wa, dtype=np.float64),
        "biotic": np.asarray(wb, dtype=np.float64),
        "cost": cost_score,
    }

    if "landscape_context_1d" in initial_conditions:
        ctx = initial_conditions["landscape_context_1d"].astype(np.float64)
        # Anomaly convention: HIGHER context anomaly = neighbours in better condition.
        # Favour good-condition surroundings → higher ctx gets the higher repair score.
        ctx_min, ctx_max = np.nanmin(ctx), np.nanmax(ctx)
        span = ctx_max - ctx_min
        if span > 1e-12:
            scores["landscape_context"] = (ctx - ctx_min) / span
        else:
            scores["landscape_context"] = np.full(len(ctx), 0.5, dtype=np.float64)

    if "restoration_potential_1d" in initial_conditions:
        rp = initial_conditions["restoration_potential_1d"].astype(np.float64)
        # Invert: lower restoration_potential (more degraded) → higher repair score
        rp_min, rp_max = np.nanmin(rp), np.nanmax(rp)
        span = rp_max - rp_min
        if span > 1e-12:
            scores["restoration_potential"] = 1.0 - (rp - rp_min) / span
        else:
            scores["restoration_potential"] = np.full(len(rp), 0.5, dtype=np.float64)

    # NOTE: spatial_clustering deliberately has NO per-pixel score. Clustering is a
    # property of the whole selection, not of any single pixel, so there is no valid
    # per-patch preference to seed or repair toward. A uniform placeholder is worse
    # than nothing: argsort over a constant collapses to patch-index (raster scan)
    # order, which warm-seeds a solid band along the north edge of the raster. The
    # objective is left to emerge from evaluation + selection; the score machinery
    # below subsets reference-direction weights to the objectives that DO have scores.

    return scores


def _map_objectives_to_pixel_scores(problem, initial_conditions, scenario_params):
    """Map run objectives to their per-pixel score arrays.

    Returns (per_obj_pixel_scores, score_keys, score_obj_indices):
      per_obj_pixel_scores : dict from build_per_objective_repair_scores.
      score_keys           : score-dict keys for objectives that HAVE a per-pixel
                             score, row-aligned with...
      score_obj_indices    : ...the objective's column index in the n_obj weight
                             vector.
    Objectives WITHOUT a meaningful per-pixel score (spatial_clustering, which is
    configuration-level, and restoration_benefit, which is arrangement-dependent
    via spillover) are intentionally omitted here - they get no static warm-seed
    and are excluded from direction-aware repair blending. Callers that want to
    seed those objectives handle them separately (e.g. single-objective pre-opt).
    """
    per_obj_pixel_scores = build_per_objective_repair_scores(initial_conditions, scenario_params)
    obj_to_score_key = {
        'abiotic_anomaly':       'abiotic',
        'biotic_anomaly':        'biotic',
        'implementation_cost':   'cost',
        'landscape_context':     'landscape_context',
        'restoration_potential': 'restoration_potential',
    }
    # Walk objective_names once so score_keys (which scores) and score_obj_indices
    # (their column in the n_obj weight vector) stay row-aligned.
    score_keys = []
    score_obj_indices = []
    for oi, obj in enumerate(problem.objective_names):
        key = obj_to_score_key.get(obj)
        if key is not None and key in per_obj_pixel_scores:
            score_keys.append(key)
            score_obj_indices.append(oi)
    return per_obj_pixel_scores, score_keys, score_obj_indices


# --- Patch Approach Initialization ---

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
    
    #print(f"  Restoration patches: {initial_conditions['n_restoration_patches']}")
    #print(f"  Conversion patches: {initial_conditions['n_conversion_patches']}")
    
    # Calculate average pixels per patch for information
    if initial_conditions['n_restoration_patches'] > 0:
        avg_pixels_per_restoration_patch = (
            initial_conditions['n_restoration_pixels'] / 
            initial_conditions['n_restoration_patches']
        )
        #print(f"  Avg pixels per restoration patch: {avg_pixels_per_restoration_patch:.1f}")
    
    if initial_conditions['n_conversion_patches'] > 0:
        avg_pixels_per_conversion_patch = (
            initial_conditions['n_conversion_pixels'] / 
            initial_conditions['n_conversion_patches']
        )
        #print(f"  Avg pixels per conversion patch: {avg_pixels_per_conversion_patch:.1f}")
    
    return initial_conditions

# --- Restoration Effect Functions ---

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
    
    # Process only objectives that are available in initial_conditions AND are
    # actual optimisation objectives. Keys loaded purely as data dependencies for
    # computed objectives (e.g. abiotic/biotic anomaly used to build the precomputed
    # restoration_potential_1d / landscape_context_1d arrays) are listed in
    # _dependency_keys; their updated anomalies are never read by evaluate_raw_objectives,
    # so skipping them here avoids full-raster copies + dilation every evaluation.
    _dep_only = initial_conditions.get('_dependency_keys', set())
    # restoration_benefit reads the UPDATED abiotic/biotic anomalies (incl. spillover)
    # at evaluation time, so those two must be processed even when they are loaded as
    # data-only dependencies (kept out of objective_names). Without this they would be
    # skipped as _dependency_keys and the benefit branch would have no updated values.
    _benefit_deps = ({'abiotic_anomaly', 'biotic_anomaly'}
                     if initial_conditions.get('restoration_benefit_enabled', False) else set())
    available_anomaly_objectives = [obj for obj in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']
                                   if obj in initial_conditions and (obj not in _dep_only or obj in _benefit_deps)]
    
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
                
                neighbor_improvement = improvement * effect_params['neighbor_effect_decay']

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

# --- Optimization Problems ---


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

        # Extract effect parameters from scenario_params
        abiotic_effect = scenario_params.get('abiotic_effect', 0.01)
        biotic_effect = scenario_params.get('biotic_effect', 0.01)
        self.effect_params = {
            'abiotic_effect': abiotic_effect,
            'biotic_effect': biotic_effect,
            # Neighbour spillover: restoration improves un-restored eligible neighbours
            # within neighbor_radius, decayed by neighbor_effect_decay. Defaults preserve
            # the previous fixed values; overridable for sensitivity / disabling (radius 0).
            'neighbor_radius': int(scenario_params.get('neighbor_radius', 3)),
            'neighbor_effect_decay': float(scenario_params.get('neighbor_effect_decay', 0.2)),
        }

        # restoration_potential objective formulation (Axis 2 — uncertainty in how
        # the restoration target is operationalised):
        #   'sum'       – total improvement: minimise summed baseline restoration_potential
        #                 over selected pixels (selects the most degraded pixels).
        #   'threshold' – area exceeding a target: maximise the count of selected pixels
        #                 whose post-restoration condition crosses into 'good' state
        #                 (restoration_potential + improvement > rp_threshold).
        # The two formulations represent alternative restoration targets over the same
        # ecological objective; switching between them isolates the effect of objective
        # construction on spatial priorities.
        self.rp_formulation = str(scenario_params.get('rp_formulation', 'sum')).lower()
        self.rp_threshold = float(scenario_params.get('rp_threshold', 0.0))

        # spatial_clustering objective metric:
        #   'adjacency'            - count orthogonal shared edges between selected
        #                            pixels (compactness; correlates strongly with
        #                            cost in tests).
        #   'components'           - count disconnected clusters (4-connectivity);
        #                            minimised. Insensitive to cluster size/shape, so
        #                            it measures pure fragmentation and may decouple
        #                            from cost.
        #   'inter_patch_adjacency'- like 'adjacency' but counts ONLY shared edges
        #                            that cross a patch boundary (pixels in different
        #                            patches), excluding the internal edges guaranteed
        #                            inside each selected patch (4 for a 2x2 patch).
        #                            Measures inter-patch contiguity; needs the patch
        #                            approach.
        self.clustering_metric = str(scenario_params.get('clustering_metric', 'adjacency')).lower()
        # First-order per-pixel condition gain from restoration, applied to the combined
        # (abiotic + biotic) restoration_potential score used by the threshold formulation.
        self.rp_improvement = 0.5 * (abiotic_effect + biotic_effect)

        # Determine which objectives are available
        self.objective_names = []
        _dep_only = set(initial_conditions.get('_dependency_keys', set()))
        if 'abiotic_anomaly' in initial_conditions and 'abiotic_anomaly' not in _dep_only:
            self.objective_names.append('abiotic_anomaly')
        if 'biotic_anomaly' in initial_conditions and 'biotic_anomaly' not in _dep_only:
            self.objective_names.append('biotic_anomaly')
        if 'landscape_anomaly' in initial_conditions:
            self.objective_names.append('landscape_anomaly')
        if 'connectivity_gain_1d' in initial_conditions:
            self.objective_names.append('connectivity_gain')
        if 'landscape_context_1d' in initial_conditions:
            self.objective_names.append('landscape_context')
        if 'restoration_potential_1d' in initial_conditions:
            self.objective_names.append('restoration_potential')
        if initial_conditions.get('restoration_benefit_enabled', False):
            self.objective_names.append('restoration_benefit')
        if initial_conditions.get('spatial_clustering_enabled', False):
            self.objective_names.append('spatial_clustering')
        if 'implementation_cost' in initial_conditions:
            self.objective_names.append('implementation_cost')
        if 'es_future_val_1d' in initial_conditions:
            self.objective_names.append('es_future_val')
        if 'es_future_robustness_1d' in initial_conditions:
            self.objective_names.append('es_future_robustness')
        
        n_objectives = len(self.objective_names)
        if n_objectives == 0:
            raise ValueError("No objectives found in initial_conditions")

        # Objective normalization configuration.
        # Optimization uses normalized objectives in _evaluate; raw objectives are
        # still computed and can be exported for reporting/interpretation.
        self.normalize_objectives = bool(scenario_params.get('normalize_objectives', True))
        self.objective_scales = self._compute_normalization_denominators()
        
        n_restoration_pixels = initial_conditions['n_restoration_pixels']
        n_conversion_pixels = initial_conditions['n_conversion_pixels']
        max_restoration_fraction = scenario_params['max_restoration_fraction']
        
        # Calculate max action pixels based on restoration eligible pixels
        self.max_action_pixels = int(max_restoration_fraction * n_restoration_pixels)

        # Minimum patch (connected-component) size constraint for the "price of
        # contiguity" sweep. min_patch_size <= 1 disables it (feature off, and the
        # problem keeps its single budget constraint). When > 1 a second constraint
        # is added: every 4-connected component of selected restoration pixels must
        # have >= min_patch_size pixels (enforced by MinPatchSizeRepair; reported in
        # out["G"][1] as an audit). Clamped to the budget so it stays feasible.
        self.min_patch_size = int(scenario_params.get('min_patch_size', 1))
        # The min-patch-size constraint is only repaired on the region-based sampling
        # paths (MinPatchSizeRepair is wired for region_grow / region_evolve). On the
        # "scattered" path nothing repairs it, so leaving it active makes every scattered
        # solution infeasible. Auto-disable it there rather than silently return no front.
        _strategy = str(scenario_params.get('sampling_strategy', 'scattered')).lower()
        if self.min_patch_size > 1 and _strategy not in ('region_grow', 'region_evolve'):
            print(f"WARNING: min_patch_size ({self.min_patch_size}) is only enforced on the "
                  f"region_grow / region_evolve paths; sampling_strategy='{_strategy}' has no "
                  f"MinPatchSizeRepair. Disabling the constraint (min_patch_size=1) for this run.")
            self.min_patch_size = 1
        if self.min_patch_size > self.max_action_pixels:
            print(f"WARNING: min_patch_size ({self.min_patch_size}) > budget "
                  f"({self.max_action_pixels}); clamping to budget.")
            self.min_patch_size = self.max_action_pixels

        # Store pixel counts for splitting decision vector
        self.n_restoration_pixels = n_restoration_pixels
        self.n_conversion_pixels = n_conversion_pixels
        self.n_pixels = n_restoration_pixels  # For backward compatibility
        
        # Binary decision variables: 0 = no action, 1 = action
        # First n_restoration_pixels elements = restoration decisions
        # Next n_conversion_pixels elements = conversion decisions
        total_decision_vars = n_restoration_pixels + n_conversion_pixels

        # Thread safety for shared mutable state written inside _evaluate()
        self._eval_lock = threading.Lock()
        self.constraint_log = []

        # Setup parallelization
        elementwise_runner = None
        if n_jobs is None or n_jobs != 1:
            try:
                pool = ThreadPool() if n_jobs in (None, -1) else ThreadPool(processes=n_jobs)
                elementwise_runner = StarmapParallelization(pool.starmap)
                print(f"Parallelization enabled with {pool._processes if hasattr(pool, '_processes') else 'auto'} threads")
            except Exception as e:
                print(f"Warning: Could not setup parallelization: {e}")
                print("Falling back to sequential evaluation")
                elementwise_runner = None

        # Initialize the problem
        kwargs = dict(
            n_var=total_decision_vars, n_obj=n_objectives,
            n_constr=(2 if self.min_patch_size > 1 else 1),
            xl=0, xu=1, type_var=int, elementwise=True,
        )
        if elementwise_runner is not None:
            kwargs['elementwise_runner'] = elementwise_runner
        super().__init__(**kwargs)

    def _compute_normalization_denominators(self):
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
            elif obj_name == 'connectivity_gain':
                cg = self.initial_conditions['connectivity_gain_1d']
                scale = float(np.nansum(np.abs(cg)))
            elif obj_name == 'landscape_context':
                ctx = self.initial_conditions['landscape_context_1d']
                # Scale = sum of absolute values over all restoration-eligible pixels
                scale = float(np.nansum(np.abs(ctx)))
            elif obj_name == 'restoration_potential':
                rp = self.initial_conditions['restoration_potential_1d']
                if getattr(self, 'rp_formulation', 'sum') == 'threshold':
                    # Count-based objective: scale by the number of eligible pixels
                    # so the normalised value falls in [-1, 0].
                    scale = float(len(rp))
                else:
                    scale = float(np.nansum(np.abs(rp)))
            elif obj_name == 'restoration_benefit':
                # Same scale basis as abiotic+biotic anomaly objectives: total absolute
                # baseline anomaly over restoration-eligible pixels (both components).
                _ab = self.initial_conditions['abiotic_anomaly']
                _bi = self.initial_conditions['biotic_anomaly']
                if rest_mask is not None:
                    scale = float(np.nansum(np.abs(_ab[rest_mask])) + np.nansum(np.abs(_bi[rest_mask])))
                else:
                    scale = float(np.nansum(np.abs(_ab)) + np.nansum(np.abs(_bi)))
            elif obj_name == 'implementation_cost':
                c = self.initial_conditions['implementation_cost']
                if rest_mask is not None and conv_mask is not None:
                    scale = float(np.nansum(np.abs(c[rest_mask])) + np.nansum(np.abs(c[conv_mask])))
                else:
                    scale = float(np.nansum(np.abs(c)))
            elif obj_name == 'spatial_clustering':
                # max_action_pixels is not yet set when this runs, so derive it here.
                _max_frac = float(self.scenario_params['max_restoration_fraction'])
                _n_rest = int(self.initial_conditions['n_restoration_pixels'])
                _max_pix = max(int(_max_frac * _n_rest), 1)
                if getattr(self, 'clustering_metric', 'adjacency') == 'components':
                    # Worst case: every selected pixel its own component → ~max_pix
                    # clusters. Scale by the pixel budget so normalised ∈ ~[0, 1].
                    scale = float(_max_pix)
                else:
                    # 'adjacency' or 'inter_patch_adjacency': a perfectly compact
                    # block of N pixels has at most ~2N shared edges (inter-patch
                    # edges are a subset), so scale by 2 x the pixel budget ->
                    # normalised in ~[-1, 0].
                    scale = float(2 * _max_pix)
            elif obj_name == 'es_future_val':
                esv = self.initial_conditions['es_future_val_1d']
                scale = float(np.nansum(np.abs(esv)))
            elif obj_name == 'es_future_robustness':
                esr = self.initial_conditions['es_future_robustness_1d']
                scale = float(np.nansum(np.abs(esr)))
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

        # Conversion objectives only valid when a landscape/connectivity objective is present.
        has_conversion_objective = (
            'landscape_anomaly' in self.initial_conditions
            or 'connectivity_gain_1d' in self.initial_conditions
        )
        if not has_conversion_objective:
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
            elif obj_name == 'connectivity_gain':
                # Precomputed per-pixel gain; sum over converted pixels.
                # Negated: minimisation problem → maximise gain ↔ minimise negative gain.
                cg = self.initial_conditions['connectivity_gain_1d']
                obj_value = -float(np.sum(cg[x_convert == 1]))
            elif obj_name == 'landscape_context':
                # Precomputed per-pixel focal-mean neighbour anomaly; sum over restored pixels.
                # Anomaly convention (see restoration_effect / anomaly_improvement_weight):
                # HIGHER anomaly = better condition, lower/negative = more degraded.
                # Good-condition surroundings => HIGH ctx. Negated so the minimiser maximises
                # summed neighbour condition, i.e. favours restoring pixels embedded in
                # good-condition surroundings.
                ctx = self.initial_conditions['landscape_context_1d']
                obj_value = -float(np.sum(ctx[x_restore == 1]))
            elif obj_name == 'restoration_potential':
                # Precomputed per-pixel mean of abiotic and biotic baseline anomaly.
                # Lower value = pixel more degraded on both dimensions = higher potential.
                rp = self.initial_conditions['restoration_potential_1d']
                sel = (x_restore == 1)
                if self.rp_formulation == 'threshold':
                    # Area-exceeding-threshold formulation: count selected pixels whose
                    # post-restoration condition crosses into 'good' state. Negated so the
                    # minimiser maximises the restored area reaching the target.
                    post_rp = rp[sel] + self.rp_improvement
                    obj_value = -float(np.count_nonzero(post_rp > self.rp_threshold))
                else:
                    # Total-improvement formulation: minimise summed baseline potential
                    # (selecting more degraded pixels gives a more negative sum).
                    obj_value = float(np.sum(rp[sel]))
            elif obj_name == 'restoration_benefit':
                # Spatially-explicit benefit: total abiotic+biotic anomaly improvement
                # achieved by the plan, INCLUDING neighbour spillover (restoration_effect
                # improves each restored cell and its un-restored eligible neighbours).
                # Summed over restoration-eligible pixels; negated so the minimiser
                # maximises improvement. Unlike restoration_potential this depends on the
                # spatial arrangement (spillover overlap), not just which pixels.
                mask = self.initial_conditions["restoration_eligible_mask"]
                d_ab = updated_conditions['abiotic_anomaly'] - self.initial_conditions['abiotic_anomaly']
                d_bi = updated_conditions['biotic_anomaly'] - self.initial_conditions['biotic_anomaly']
                obj_value = -float(np.sum((d_ab + d_bi)[mask]))
            elif obj_name == 'spatial_clustering':
                # Spatial clustering of the selected restoration pixels. Map the
                # selected eligible pixels back to the 2D raster, then score one of
                # two metrics (self.clustering_metric):
                shape = self.initial_conditions['shape']
                sel_idx = self.initial_conditions['restoration_eligible_indices'][x_restore == 1]
                sel_mask = np.zeros(shape, dtype=bool)
                rows, cols = np.divmod(sel_idx, shape[1])
                sel_mask[rows, cols] = True
                if self.clustering_metric == 'components':
                    # Number of disconnected clusters (4-connectivity). Fewer = more
                    # clumped. Minimised directly. Insensitive to cluster size/shape
                    # -- measures pure fragmentation -- so it need not track cost the
                    # way edge count (proportional to amount of contiguous land bought) does.
                    _, n_comp = ndimage.label(sel_mask)
                    obj_value = float(n_comp)
                elif self.clustering_metric == 'inter_patch_adjacency':
                    # Inter-patch adjacency: orthogonal shared edges between selected
                    # pixels that lie in DIFFERENT patches. Excludes the internal
                    # edges guaranteed inside each fully-selected patch (4 for a 2x2
                    # patch), so it measures how much selected patches touch each
                    # other rather than trivial within-patch compactness. Negated so
                    # the minimiser maximises inter-patch contact. Requires the patch
                    # approach; without it every pixel is its own patch, so all
                    # adjacencies are inter-patch and this reduces to full adjacency.
                    pm = self.initial_conditions.get('patch_mappings')
                    patch_grid = (pm['restoration_patches']['patch_grid']
                                  if pm is not None else None)
                    if patch_grid is None:
                        horiz = np.count_nonzero(sel_mask[:, :-1] & sel_mask[:, 1:])
                        vert = np.count_nonzero(sel_mask[:-1, :] & sel_mask[1:, :])
                    else:
                        # Bitwise & binds tighter than !=, so parenthesise the
                        # patch-id comparison.
                        horiz = np.count_nonzero(
                            sel_mask[:, :-1] & sel_mask[:, 1:]
                            & (patch_grid[:, :-1] != patch_grid[:, 1:])
                        )
                        vert = np.count_nonzero(
                            sel_mask[:-1, :] & sel_mask[1:, :]
                            & (patch_grid[:-1, :] != patch_grid[1:, :])
                        )
                    obj_value = -float(horiz + vert)
                else:
                    # 'adjacency' (default): orthogonal like-adjacencies -- pairs of
                    # selected pixels sharing an edge. More shared edges = more
                    # compact. Negated so the pymoo minimiser maximises it. Cheap:
                    # two boolean shift-AND reductions, no neighbourhood convolution.
                    horiz = np.count_nonzero(sel_mask[:, :-1] & sel_mask[:, 1:])
                    vert = np.count_nonzero(sel_mask[:-1, :] & sel_mask[1:, :])
                    obj_value = -float(horiz + vert)
            elif obj_name == 'es_future_val':
                # Sum of per-pixel ES performance over selected restoration pixels.
                # Higher = greater total future ES gain → maximise (negate for pymoo minimisation).
                # Sum (not mean) is consistent with other objectives and avoids concentration artefacts.
                esv = self.initial_conditions['es_future_val_1d']
                obj_value = -float(np.sum(esv[x_restore == 1]))
            elif obj_name == 'es_future_robustness':
                # Sum of per-pixel ES instability over selected restoration pixels.
                # Lower = less total undesirable deviation = more robust → minimise directly.
                esr = self.initial_conditions['es_future_robustness_1d']
                obj_value = float(np.sum(esr[x_restore == 1]))
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
        x_restore = x[:self.n_restoration_pixels]
        x_convert = x[self.n_restoration_pixels:self.n_restoration_pixels + self.n_conversion_pixels]

        # Enable conversion actions when a landscape or connectivity objective is present
        has_conversion_objective = (
            'landscape_anomaly' in self.initial_conditions
            or 'connectivity_gain_1d' in self.initial_conditions
        )
        if not has_conversion_objective:
            x_convert[:] = 0
        
        # Use both restoration and conversion decisions
        n_restored = np.sum(x_restore)
        n_converted = np.sum(x_convert)
        n_total_actions = n_restored + n_converted
        
        raw_objectives = self.evaluate_raw_objectives(x)
        objectives = self._normalize_objective_vector(raw_objectives)
        out["F"] = objectives
        out["F_raw"] = raw_objectives
        
        # Constraint: total number of pixels with actions (restore + convert)
        out["G"] = [abs(n_total_actions - self.max_action_pixels)]  # Should be 0 due to exact count enforcement

        # Minimum-patch-size constraint (price-of-contiguity sweep). Feasible when
        # every 4-connected component of selected restoration pixels has >= S pixels;
        # violation = max(0, S - smallest_component). MinPatchSizeRepair guarantees
        # this, so out["G"][1] audits the repair (should stay 0).
        if self.min_patch_size > 1:
            out["G"].append(self._min_patch_size_violation(x_restore))

        # Log constraint violations (thread-safe: constraint_log is shared state)
        _ctypes = ['budget', 'min_patch_size']
        for i, g_val in enumerate(out["G"]):
            if g_val > 0:
                with self._eval_lock:
                    self.constraint_log.append({
                        'generation': getattr(self, 'current_gen', 0),
                        'violation_value': g_val,
                        'constraint_type': _ctypes[i] if i < len(_ctypes) else f'constraint_{i}'
                    })

    def _min_patch_size_violation(self, x_restore):
        """max(0, min_patch_size - smallest 4-connected component of the selection)."""
        sel_idx = self.initial_conditions['restoration_eligible_indices'][x_restore == 1]
        if sel_idx.size == 0:
            return float(self.min_patch_size)
        shape = self.initial_conditions['shape']
        m = np.zeros(shape, dtype=bool)
        rr, cc = np.divmod(sel_idx, shape[1])
        m[rr, cc] = True
        lab, nc = ndimage.label(m)
        if nc == 0:
            return float(self.min_patch_size)
        smallest = int(np.bincount(lab.ravel())[1:].min())
        return float(max(0, self.min_patch_size - smallest))


# --- Patch-Based Problem ---

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
        #print(f"  Constraint type: {patch_constraint_type}")
        #print(f"  Target value: {self.target_constraint_value}")
        #if patch_constraint_type == 'pixel_count':
            #print(f"  Tolerance: ±{pixel_tolerance*100:.1f}%")
        #print(f"  (pixel-based equivalent: {self.max_action_pixels} pixels)")
    
    def _evaluate(self, x_patches, out, *args, **kwargs):
        """
        Evaluate a patch-based solution.
        
        Args:
            x_patches: Patch-level decision variables (binary array)
                      First n_restoration_patches: restoration patch decisions
                      Next n_conversion_patches: conversion patch decisions
            out: Output dictionary for objectives and constraints
        """
        from .patch_approach import convert_patch_decisions_to_pixels
        
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
            
            # DEBUG: Print constraint check details (thread-safe counter)
            if not hasattr(self, '_constraint_debug_count'):
                self._constraint_debug_count = 0
            if self._constraint_debug_count < 5:
                #print(f"  DEBUG CONSTRAINT: target={self.target_constraint_value}, tolerance={evaluation_tolerance:.3f}, range=[{min_pixels}, {max_pixels}]")
                #print(f"                    restore={n_restore_pixels}, convert={n_convert_pixels}, total={n_pixels_used}")
                with self._eval_lock:
                    self._constraint_debug_count += 1
            
            if min_pixels <= n_pixels_used <= max_pixels:
                constraint_value = 0  # Accept as feasible
            else:
                # Penalize violations outside the evaluation tolerance
                constraint_value = min(
                    abs(n_pixels_used - min_pixels),
                    abs(n_pixels_used - max_pixels)
                )
                if self._constraint_debug_count < 5:
                    print(f"                    VIOLATION: G={constraint_value}")
        
        out["G"] = [constraint_value]

        # Parity with RestorationProblem: keep out["G"] length == n_constr when the
        # minimum-patch-size constraint is active (computed on the pixel selection).
        if self.min_patch_size > 1:
            out["G"].append(self._min_patch_size_violation(x_restore_pixels))

    def evaluate_raw_objectives(self, x_patches):
        """Return raw objectives for patch-level decisions via pixel conversion."""
        from .patch_approach import convert_patch_decisions_to_pixels

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


# --- Execution Utilities ---

def build_fixed_ref_point(problem, sampling, n_samples=200, margin=0.05, seed=42, verbose=False, n_jobs=1):
    """
    Build a fixed hypervolume reference point using a warm up sample of solutions.
    """
    # Use the provided Sampling operator to generate candidate solutions
    # This returns a Population in pymoo, so we extract X
    pop = sampling.do(problem, n_samples)
    X = pop.get("X")

    def _eval_one(x):
        out = {}
        problem._evaluate(x, out)
        return out["F"]

    if n_jobs != 1:
        # scipy.ndimage.convolve releases the GIL, so threads genuinely run in
        # parallel without pickling overhead. n_jobs=-1/None => use all cores.
        from concurrent.futures import ThreadPoolExecutor
        n_workers = None if n_jobs in (-1, None) else n_jobs
        if verbose:
            print(f"  Warm-up ref point: evaluating {X.shape[0]} samples with {n_workers or 'all'} threads...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            F_list = list(executor.map(_eval_one, [X[k] for k in range(X.shape[0])]))
    else:
        if verbose:
            print(f"  Warm-up ref point: evaluating {X.shape[0]} samples sequentially...")
        F_list = []
        for k in range(X.shape[0]):
            F_list.append(_eval_one(X[k]))
            if verbose and (k + 1) % 10 == 0:
                print(f"    {k + 1}/{X.shape[0]} done")

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

        # X snapshot buffering (populated only when snapshot_dir is set)
        self.snapshot_dir = None       # set by ProgressCallback when save_snapshots=True
        self.X_batch = []              # in-memory buffer of int8 arrays
        self.batch_files = []          # paths of flushed batch .npz files
        self.batch_size = 10           # flush every N generations
        self._n_generations_est = 100  # updated by ProgressCallback for memory estimate
        self._x_memory_warned = False  # print estimate only once
    
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

                # Capture full population X snapshot
                if self.snapshot_dir is not None:
                    X_pop = algorithm.pop.get("X")
                    if X_pop is not None:
                        if not self._x_memory_warned and algorithm.n_gen == 1:
                            batch_mb = self.batch_size * X_pop.shape[0] * X_pop.shape[1] / 1e6
                            total_mb = self._n_generations_est * X_pop.shape[0] * X_pop.shape[1] / 1e6
                            print(f"   X snapshots: {X_pop.shape[1]} vars × {X_pop.shape[0]} pop, "
                                  f"batch RAM ~{batch_mb:.0f} MB, est. total on disk ~{total_mb:.0f} MB")
                            if total_mb > 1000:
                                print(f"   Warning: estimated X_history size exceeds 1 GB — consider reduce n_generations")
                            self._x_memory_warned = True
                        self.X_batch.append(X_pop.astype(np.int8))
                        if len(self.X_batch) >= self.batch_size:
                            self._flush_batch()

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
                    # Always append so hv_history stays length-aligned with mutation_log
                    self.hv_history.append(float('nan'))
                    if self.verbose:
                        print(f"   Warning: HV calculation failed at gen {algorithm.n_gen}: {e}")
            else:
                # F is None or empty — keep hv_history length-aligned
                if self.verbose:
                    print(f"   Warning: HVCallback skipped at gen {algorithm.n_gen}: pop F unavailable")
                self.hv_history.append(float('nan'))
        else:
            # pop unavailable — keep hv_history length-aligned
            if self.verbose:
                print(f"   Warning: HVCallback skipped at gen {getattr(algorithm, 'n_gen', '?')}: pop unavailable")
            self.hv_history.append(float('nan'))

    def _flush_batch(self):
        """Stack the in-memory X buffer and write it to a temporary .npz file."""
        if not self.X_batch:
            return
        batch_arr = np.stack(self.X_batch, axis=0)  # (batch_size, pop_size, n_var) int8
        batch_idx = len(self.batch_files)
        batch_path = os.path.join(self.snapshot_dir, f"X_batch_{batch_idx:04d}.npz")
        np.savez(batch_path, X=batch_arr)
        self.batch_files.append(batch_path)
        self.X_batch.clear()


class ProgressCallback:
    """
    Progress reporting callback for NSGA-II optimization.

    Reports generation progress and delegates hypervolume tracking to
    HVCallback.  Constructed with explicit parameters rather than relying
    on closed-over variables so it can be instantiated cleanly outside
    the optimization runner.
    """

    def __init__(self, verbose=True, n_generations=100,
                 hv_patience=15, hv_min_improvement=1e-6, ref_point=None,
                 save_snapshots=False, snapshot_dir=None):
        """
        Args:
            verbose: Print progress every 10 generations.
            n_generations: Total generation budget (used for ETA calculation).
            hv_patience: Patience parameter forwarded to HVCallback.
            hv_min_improvement: Min improvement threshold forwarded to HVCallback.
            ref_point: Fixed hypervolume reference point forwarded to HVCallback.
            save_snapshots: If True, capture full population X at every generation.
            snapshot_dir: Directory to write temporary batch .npz files.
        """
        self.verbose = verbose
        self.n_generations = n_generations
        self.start_time = None
        self.hv_callback = HVCallback(
            patience=hv_patience,
            min_improvement=hv_min_improvement,
            verbose=verbose,
            ref_point=ref_point,
        )
        if save_snapshots and snapshot_dir is not None:
            os.makedirs(snapshot_dir, exist_ok=True)
            self.hv_callback.snapshot_dir = snapshot_dir
            self.hv_callback._n_generations_est = n_generations

    def __call__(self, algorithm):
        if self.start_time is None:
            self.start_time = datetime.now()

        # Delegate HV tracking and early-stopping check.
        self.hv_callback(algorithm)

        # Expose current generation to the problem for constraint logging.
        if hasattr(algorithm, 'problem'):
            algorithm.problem.current_gen = algorithm.n_gen

        # Sync generation counter to mutation operator for its log.
        if hasattr(algorithm, 'mating') and hasattr(algorithm.mating, 'mutation'):
            if hasattr(algorithm.mating.mutation, '_current_gen'):
                algorithm.mating.mutation._current_gen = algorithm.n_gen

        gen = algorithm.n_gen
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if self.verbose and gen % 10 == 0:
            progress = (gen / self.n_generations) * 100
            eta = (elapsed / gen) * (self.n_generations - gen) if gen > 0 else 0

            violation_info = ""
            if hasattr(algorithm, 'problem') and hasattr(algorithm.problem, 'constraint_log'):
                gen_violations = [c for c in algorithm.problem.constraint_log
                                  if c.get('generation', 0) == gen]
                if gen_violations:
                    violation_info = f" - Violations: {len(gen_violations)}"

            print(f"   Generation {gen}/{self.n_generations} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed/60:.1f}min - ETA: {eta/60:.1f}min{violation_info}")

        # Hypervolume-based early stopping: once the HVCallback has seen no
        # improvement for hv_patience generations, tell pymoo to terminate.
        if self.hv_callback.converged:
            algorithm.termination.force_termination = True


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


# --- Optimization Run Helpers ---


class SingleObjectiveView(ElementwiseProblem):
    """Expose ONE objective of a multi-objective RestorationProblem as n_obj=1.

    Reuses the base problem's full _evaluate (all objective logic plus the budget
    / min-patch constraints) and slices out a single objective column, so a
    single-objective GA can search for that objective's extreme. This is how
    warm seeding handles arrangement-dependent objectives (restoration_benefit,
    spatial_clustering) that have no static per-pixel score: the extreme is found
    by the real objective evaluation rather than a proxy.

    Operator-relevant attributes are copied from the base problem so the same
    sampling/repair/mutation/crossover operators run unchanged against this view.
    """

    def __init__(self, base, obj_index):
        self.base = base
        self.obj_index = int(obj_index)
        # Operators read these off `problem`; mirror them from the base problem.
        self.initial_conditions = base.initial_conditions
        self.n_restoration_pixels = base.n_restoration_pixels
        self.n_conversion_pixels = base.n_conversion_pixels
        self.n_pixels = getattr(base, 'n_pixels', base.n_restoration_pixels)
        self.max_action_pixels = base.max_action_pixels
        self.min_patch_size = getattr(base, 'min_patch_size', 1)
        super().__init__(
            n_var=base.n_var, n_obj=1,
            n_constr=(2 if self.min_patch_size > 1 else 1),
            xl=0, xu=1, type_var=int, elementwise=True,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        o = {}
        self.base._evaluate(x, o, *args, **kwargs)
        F = np.asarray(o["F"], dtype=float)
        out["F"] = [float(F[self.obj_index])]
        if o.get("G") is not None:
            out["G"] = o["G"]


def _single_objective_seed(problem, obj_index, base_sampling, repair, mutation,
                           crossover, pop, gens, seed, verbose=False):
    """Run a short single-objective GA on one objective; return its best genotype.

    Reuses the run's operators so the seed is feasible and (for region strategies)
    contiguous. The GA runs serially over a SingleObjectiveView. Diagnostic state
    on the repair operator is snapshotted and restored so this pre-run does not
    pollute the main run's repair logs / capture schedule.
    """
    from pymoo.algorithms.soo.nonconvex.ga import GA

    view = SingleObjectiveView(problem, obj_index)
    _mut = mutation if mutation is not None else InstrumentedBitflipMutation(
        prob=1.0, prob_var=200.0 / problem.n_var)
    _cx = crossover if crossover is not None else HUX()

    # Snapshot mutable diagnostic state on the repair so the pre-run leaves it as
    # it found it (capture disabled during the pre-run; counters/logs restored).
    _saved = None
    if repair is not None:
        _saved = (
            getattr(repair, 'capture_diag', None),
            getattr(repair, 'total_calls', None),
            list(getattr(repair, 'call_log', [])),
            list(getattr(repair, 'repair_log', [])),
        )
        if hasattr(repair, 'capture_diag'):
            repair.capture_diag = False

    try:
        ga = GA(
            pop_size=int(pop),
            sampling=base_sampling,
            crossover=_cx,
            mutation=_mut,
            repair=repair,
            eliminate_duplicates=True,
        )
        res = minimize(view, ga, get_termination("n_gen", int(gens)),
                       seed=int(seed), verbose=False)
    except Exception as e:
        if verbose:
            print(f"  Warm-start pre-opt for objective index {obj_index} failed: {e}")
        res = None
    finally:
        if _saved is not None:
            cap, tc, cl, rl = _saved
            if cap is not None:
                repair.capture_diag = cap
            if tc is not None:
                repair.total_calls = tc
            if hasattr(repair, 'call_log'):
                repair.call_log[:] = cl
            if hasattr(repair, 'repair_log'):
                repair.repair_log[:] = rl

    if res is None or getattr(res, 'X', None) is None:
        return None
    return np.asarray(res.X, dtype=int).ravel()


def build_warm_start_seeds(problem, initial_conditions, scenario_params,
                           base_sampling, repair, mutation, crossover,
                           sampling_strategy, random_seed, verbose=False):
    """Build warm-seed genotypes for the non-patch path (hybrid).

    - Objectives WITH a per-pixel score get a cheap static seed: greedy top-k by
      score for 'scattered'; a single contiguous grown region for the region
      strategies.
    - Arrangement-dependent objectives (no per-pixel score, e.g. restoration_benefit,
      spatial_clustering) get a short single-objective pre-optimisation GA.

    Returns an int array (n_seed, n_var), or None when nothing is seeded.
    """
    n_var = problem.n_var
    n_rest = problem.n_restoration_pixels
    k = min(int(problem.max_action_pixels), n_rest)
    rng = np.random.RandomState(None if random_seed is None else int(random_seed))
    strategy = str(sampling_strategy).lower()
    region_mode = strategy in ('region_grow', 'region_evolve')

    per_obj_pixel_scores, score_keys, score_obj_indices = _map_objectives_to_pixel_scores(
        problem, initial_conditions, scenario_params)

    seeds = []

    # --- Static-score seeds (objectives that have a per-pixel score) ---
    nbr = None
    if region_mode and score_keys:
        nbr, _rows, _cols = build_restoration_neighbor_table(initial_conditions)
    for key in score_keys:
        score = np.asarray(per_obj_pixel_scores[key], dtype=np.float64)
        if score.shape[0] != n_rest:
            # Score length must align with restoration-eligible pixels; skip if not.
            if verbose:
                print(f"  Warm-start: skipping static seed '{key}' "
                      f"(len {score.shape[0]} != n_rest {n_rest})")
            continue
        row = np.zeros(n_var, dtype=int)
        if region_mode:
            sel = grow_region_plan(nbr, n_rest, k, 1, score, 'scored', rng)
            row[:n_rest] = np.asarray(sel, dtype=int)
        else:
            # Greedy top-k by score; tiny tie jitter avoids raster-order bias
            # (mirrors PatchAwareSampling._build_extreme_solution).
            jitter = 1e-9 * rng.random(n_rest)
            top = np.argsort(-(score + jitter))[:k]
            row[top] = 1
        seeds.append(row)

    # --- Pre-optimisation seeds (arrangement-dependent objectives) ---
    mapped = set(score_obj_indices)
    arrangement_indices = [oi for oi in range(len(problem.objective_names))
                           if oi not in mapped]
    if arrangement_indices:
        pop = int(scenario_params.get('warm_seed_preopt_pop', 40))
        gens = int(scenario_params.get('warm_seed_preopt_gens', 30))
        base_seed = 0 if random_seed is None else int(random_seed)
        for oi in arrangement_indices:
            if verbose:
                print(f"  Warm-start: pre-optimising objective "
                      f"'{problem.objective_names[oi]}' (pop={pop}, gens={gens})...")
            seed_x = _single_objective_seed(
                problem, oi, base_sampling, repair, mutation, crossover,
                pop=pop, gens=gens, seed=base_seed + 1000 + oi, verbose=verbose)
            if seed_x is not None and seed_x.shape[0] == n_var:
                seeds.append(seed_x)

    if not seeds:
        return None
    return np.vstack(seeds).astype(int)


def _build_operators(initial_conditions, scenario_params, problem, use_patch_approach,
                     use_repair, patch_constraint_type, pixel_tolerance, verbose, warm_seeding=True,
                     capture_repair_diag=False, n_capture_gens=5, n_generations=None, diag_dir=None):
    """
    Build sampling and repair operators for the chosen optimization mode.

    Returns
    -------
    tuple
        (sampling, repair) — repair is None when use_repair is False.
    """
    mutation = None   # region pixel modes supply a custom mutation
    crossover = None  # region_evolve supplies a region-swap crossover
    if use_patch_approach:
        restoration_scores = build_repair_scores(initial_conditions, scenario_params)
        patch_scores = aggregate_patch_scores_from_pixel_scores(
            patch_mappings=initial_conditions['patch_mappings'],
            restoration_pixel_scores=restoration_scores,
            conversion_pixel_scores=None,
            mode='mean',
        )
        patch_score_temperature = float(scenario_params.get('patch_score_temperature', 0.25))
        patch_random_share = float(scenario_params.get('patch_random_share', 0.15))
        patch_repair_top_k = int(scenario_params.get('patch_repair_top_k', 12))

        # Build per-objective patch scores for direction-aware repair.
        # Each individual is assigned to an NSGA-III reference direction whose
        # weights indicate how much that direction cares about each objective.
        # PatchRepair uses these weights at repair time to blend the three score
        # vectors, so cost-axis individuals get repaired toward cheap patches,
        # abiotic-axis individuals toward high-abiotic patches, etc.
        # Map internal objective names -> per-pixel score keys (shared helper).
        # Objectives WITHOUT a meaningful per-patch score (e.g. spatial_clustering,
        # which is configuration-level) are intentionally absent. They get no
        # warm-seed and are excluded from direction-aware repair blending via
        # _score_obj_indices - never given a placeholder score, which would
        # otherwise collapse argsort-based seeding to raster scan order.
        per_obj_pixel_scores, _score_keys, _score_obj_indices = _map_objectives_to_pixel_scores(
            problem, initial_conditions, scenario_params)
        per_obj_patch_scores = np.stack([
            aggregate_patch_scores_from_pixel_scores(
                patch_mappings=initial_conditions['patch_mappings'],
                restoration_pixel_scores=per_obj_pixel_scores[key],
                conversion_pixel_scores=None,
                mode='mean',
            )
            for key in _score_keys
        ], axis=0)  # shape (n_scored_objectives, n_patches)

        repair_ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

        # Burden-sharing: build region assignments only when requested.
        # build_region_assignments_cache must be called first so the cache exists
        # when assign_patches_to_regions inspects it (the cache is normally built
        # lazily inside apply_burden_sharing, which is only used in the non-patch
        # path, so it would never be populated here without this explicit call).
        burden_sharing_enabled = scenario_params.get('burden_sharing', 'no') == 'yes'
        if burden_sharing_enabled:
            build_region_assignments_cache(initial_conditions)
        patch_region_assignments = (
            assign_patches_to_regions(
                initial_conditions['patch_mappings'], initial_conditions
            )
            if burden_sharing_enabled else None
        )

        sampling = PatchAwareSampling(
            patch_mappings=initial_conditions['patch_mappings'],
            target_pixels=problem.target_constraint_value,
            pixel_tolerance=pixel_tolerance,
            patch_scores=patch_scores,
            score_temperature=patch_score_temperature,
            random_share=patch_random_share,
            per_objective_patch_scores=per_obj_patch_scores,
            patch_region_assignments=patch_region_assignments,
        )
        repair = PatchRepair(
            constraint_type=patch_constraint_type,
            target_value=problem.target_constraint_value,
            patch_mappings=initial_conditions['patch_mappings'],
            pixel_tolerance=pixel_tolerance,
            patch_scores=patch_scores,
            score_temperature=patch_score_temperature,
            top_k=patch_repair_top_k,
            per_objective_patch_scores=per_obj_patch_scores if warm_seeding else None,
            ref_dirs=repair_ref_dirs,
            score_obj_indices=_score_obj_indices,
            patch_region_assignments=patch_region_assignments,
            capture_diag=capture_repair_diag, n_generations=n_generations,
            n_capture=n_capture_gens, diag_dir=diag_dir,
        ) if use_repair else None

        if verbose:
            print(
                "Using stochastic score-guided patch operators "
                f"({patch_constraint_type}, temp={patch_score_temperature}, "
                f"random_share={patch_random_share}, top_k={patch_repair_top_k})"
            )
    else:
        scores = build_repair_scores(initial_conditions, scenario_params)
        sampling_strategy = str(scenario_params.get('sampling_strategy', 'scattered')).lower()

        if sampling_strategy in ('region_grow', 'region_evolve'):
            # Region operators build contiguous, arbitrary-shape regions so the search
            # can reach clustered (non-scattered) plans where cost and spatial_clustering
            # actually vary. region_evolve additionally makes the SEARCH explore the
            # landscape (relocate/spawn/swap whole regions) instead of collapsing onto
            # one basin. See [[hv-stagnation-flat-objectives]].
            region_seeds = int(scenario_params.get('region_seeds', 25))
            region_seeds_min = scenario_params.get('region_seeds_min', None)
            region_random_share = float(scenario_params.get('region_random_share', 0.0))
            growth_bias = str(scenario_params.get('region_growth_bias', 'scored')).lower()
            region_edits = int(scenario_params.get('region_mutation_edits', 100))
            region_seed_grid = int(scenario_params.get('region_seed_grid', 16))

            # Minimum-patch-size constraint (price-of-contiguity sweep). When active,
            # cap the seed count so the initial regions are already >= S on average
            # (avg region size ~ budget / seeds), leaving little for the repair to fix.
            min_patch_size = int(getattr(problem, 'min_patch_size', 1))
            if min_patch_size > 1:
                max_seeds_for_S = max(1, problem.max_action_pixels // min_patch_size)
                region_seeds = min(region_seeds, max_seeds_for_S)
                if region_seeds_min is not None:
                    region_seeds_min = min(int(region_seeds_min), region_seeds)

            # 'scored' growth prefers high restoration score AND low cost: blend the
            # standardised base score with negated standardised per-pixel cost so
            # regions form in cheap, high-value areas (exposing the cost gradient).
            region_scores = np.asarray(scores, dtype=np.float64)
            cost2d = initial_conditions.get('implementation_cost')
            rmask = initial_conditions.get('restoration_eligible_mask')
            if growth_bias == 'scored' and cost2d is not None and rmask is not None:
                cpix = np.asarray(cost2d)[rmask].astype(np.float64)
                if cpix.size == region_scores.size:
                    def _z(a):
                        sd = a.std()
                        return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a)
                    region_scores = _z(region_scores) - _z(cpix)

            if sampling_strategy == 'region_evolve':
                # Seed regions SPREAD across the map (coverage), evolve them with
                # region-level moves, and recombine whole regions (not HUX).
                sampling = SpatialCoverageSampling(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    region_seeds=region_seeds, region_seeds_min=region_seeds_min,
                    growth_bias='neutral', seed_grid=region_seed_grid,
                )
                mutation = RegionEvolveMutation(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    n_edits=region_edits, growth_bias=growth_bias,
                    pixel_tolerance=pixel_tolerance,
                )
                crossover = RegionSwapCrossover(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    growth_bias=growth_bias, pixel_tolerance=pixel_tolerance,
                )
                if verbose:
                    print(f"Using region-evolve operators (seeds<={region_seeds}, "
                          f"grid={region_seed_grid}, bias={growth_bias})")
            else:
                sampling = RegionGrowingSampling(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    region_seeds=region_seeds, growth_bias=growth_bias,
                    region_seeds_min=region_seeds_min, random_share=region_random_share,
                )
                mutation = RegionGrowingMutation(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    n_edits=region_edits, growth_bias=growth_bias,
                    pixel_tolerance=pixel_tolerance,
                )
                if verbose:
                    print(f"Using region-growing operators (seeds={region_seeds}, "
                          f"bias={growth_bias}, edits={region_edits})")
            if not use_repair:
                repair = None
            else:
                # Region modes always use the contiguity-preserving repair, so every
                # min-patch-size level differs ONLY in the floor S. S=1 means "no size
                # floor" but still uses the same scored contiguous budget regrowth - so
                # the price-of-contiguity sweep is a controlled comparison rather than
                # having S=1 fall through to the scattered AdaptiveRepair (which does
                # different optimisation work and made S=1 non-comparable).
                repair = MinPatchSizeRepair(
                    initial_conditions, problem.max_action_pixels, min_patch_size,
                    scores=region_scores, pixel_tolerance=pixel_tolerance,
                    growth_bias=growth_bias,
                )
                if verbose:
                    print(f"Using MinPatchSizeRepair (S={min_patch_size}, "
                          f"seeds<={region_seeds})")
        else:
            sampling = AdaptiveSampling(initial_conditions, problem.max_action_pixels, scenario_params)
            # repair_scored=False makes the budget repair NEUTRAL: passing scores=None
            # routes AdaptiveRepair to _enforce_count_random (constraints-only, no score
            # bias), giving an operator-unbiased front. Default True preserves the
            # historical score-based repair for every other run.
            repair_scored = bool(scenario_params.get('repair_scored', True))
            repair = AdaptiveRepair(
                initial_conditions, problem.max_action_pixels, scenario_params,
                scores if repair_scored else None,
                capture_diag=capture_repair_diag, n_generations=n_generations,
                n_capture=n_capture_gens, diag_dir=diag_dir,
            ) if use_repair else None

            if verbose:
                burden_sharing = scenario_params.get('burden_sharing', 'no')
                clustering_strength = scenario_params.get('spatial_clustering', 0.0)
                repair_desc = "score-based repair" if repair_scored else "neutral (random) repair"
                strategy_desc = []
                if burden_sharing == 'yes':
                    strategy_desc.append("burden-sharing")
                if clustering_strength > 0.0:
                    strategy_desc.append(f"clustering({clustering_strength})")
                if not strategy_desc:
                    strategy_desc.append(f"random with {repair_desc}")
                print(f"Using adaptive operators: {', '.join(strategy_desc)}")

    return sampling, repair, mutation, crossover


def _build_algorithm(problem, sampling, repair, n_generations, n_partitions=8,
                     mutation_prob_var=None, mutation=None, crossover=None,
                     algorithm_type="nsga3", pop_size=None):
    """
    Construct the multi-objective algorithm and generation-based termination.

    algorithm_type:
      "nsga3" (default) - NSGA-III with structured reference directions
                          (Das-Dennis simplex lattice) instead of crowding
                          distance. Restores selection pressure in 3-objective
                          space where NSGA-II degenerates because nearly all
                          solutions end up on rank 0. Population size equals the
                          number of reference directions (n_partitions=12, 3 obj
                          -> 45 ref dirs -> pop 45).
      "nsga2"           - classic NSGA-II with explicit pop_size and crowding
                          distance. Appropriate for 2-objective problems where
                          the Pareto front is a curve and crowding distance
                          preserves diversity well. Uses pop_size directly.

    Returns
    -------
    tuple
        (algorithm, termination)
    """
    # Per-variable flip probability. Default keeps historical behaviour of ~200
    # expected flips per individual (prob_var = 200 / n_var); callers may override
    # (e.g. mutation-rate sensitivity sweeps).
    if mutation_prob_var is None:
        mutation_prob_var = 200.0 / problem.n_var
    # Region-growing runs supply a contiguity-preserving mutation; otherwise use
    # the default per-variable bitflip mutation.
    if mutation is None:
        mutation = InstrumentedBitflipMutation(prob=1.0, prob_var=mutation_prob_var)
    # region_evolve supplies a region-swap crossover; otherwise use HUX.
    if crossover is None:
        crossover = HUX()

    algorithm_type = str(algorithm_type).lower()
    if algorithm_type == "nsga2":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        if pop_size is None:
            raise ValueError("NSGA-II requires an explicit pop_size")
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
        )
    else:
        n_obj = problem.n_obj
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
        )
    termination = get_termination("n_gen", n_generations)
    return algorithm, termination


def _package_results(result, problem, initial_conditions, scenario_params, callback,
                     use_patch_approach, pop_size, n_generations, hv_patience,
                     hv_min_improvement, save_results, output_dir, verbose,
                     save_snapshots=False, run_label="", run_config=None,
                     r_export_parent=None):
    """
    Assemble the optimization results dict and optionally save to disk.

    Parameters
    ----------
    result : pymoo Result
        Raw result returned by ``minimize()``.
    problem : RestorationProblem or PatchRestorationProblem
    initial_conditions, scenario_params : dicts
    callback : ProgressCallback
    use_patch_approach : bool
    pop_size, n_generations, hv_patience, hv_min_improvement : run settings
    save_results : bool
    output_dir : str
    verbose : bool

    Returns
    -------
    dict
        optimization_results ready for downstream use.
    """
    if verbose:
        convergence_reason = "hypervolume plateau" if callback.hv_callback.converged else "generation limit"
        final_gen = len(callback.hv_callback.hv_history)
        print(f"\u2713 Optimization completed after {final_gen} generations ({convergence_reason})")
        if callback.hv_callback.hv_history:
            print(f"Final hypervolume: {callback.hv_callback.hv_history[-1]:.6f}")
        print(f"Found {len(result.F)} Pareto-optimal solutions out of {len(result.pop.get('F'))} evaluated solutions")

    initial_conditions_filtered = _filter_initial_conditions_for_return(initial_conditions)

    # ----- Full Population Data -----
    X_full = result.pop.get("X")
    F_full_norm = result.pop.get("F")
    F_full_raw = np.asarray([problem.evaluate_raw_objectives(xi) for xi in X_full], dtype=float)

    # ----- Non-dominated Data -----
    X_nd = result.X
    F_nd_norm = result.F
    
    # ----- Identify non-dominated solutions within the full population -----
    # Create a set of tuples for efficient lookup of non-dominated decision vectors
    nd_solutions_set = {tuple(row) for row in X_nd}
    is_nondominated = np.array([tuple(row) in nd_solutions_set for row in X_full])

    optimization_results = {
        'scenario_params': scenario_params,
        'objective_names': problem.objective_names,
        'objectives': F_full_raw,
        'objectives_raw': F_full_raw,
        'objectives_normalized': F_full_norm,
        'decisions': X_full,
        'is_nondominated': is_nondominated,
        'n_solutions': len(X_full),
        'n_nondominated_solutions': len(X_nd),
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
                'f_max_history': callback.hv_callback.f_max_history,
            },
            'hv_patience': hv_patience,
            'hv_min_improvement': hv_min_improvement,
            'sampling_method': 'adaptive',
            'repair_operators': ['score_based_repair', 'random_repair'],
            'objective_normalization': {
                'enabled': bool(problem.normalize_objectives),
                'scales': {k: float(v) for k, v in problem.objective_scales.items()},
            },
            'timestamp': datetime.now().isoformat(),
        },
        'initial_conditions': initial_conditions_filtered,
        'hv_callback': callback.hv_callback,
        'run_label': run_label,
        'run_config': run_config,
    }

    # Attach operator diagnostics from repair and mutation
    _algo = getattr(result, 'algorithm', None)
    _repair = getattr(_algo, 'repair', None) if _algo is not None else None
    if _repair is not None and hasattr(_repair, 'repair_log'):
        optimization_results['repair_diagnostics'] = _repair.repair_log
    _mutation = None
    if _algo is not None and hasattr(_algo, 'mating') and hasattr(_algo.mating, 'mutation'):
        _mutation = _algo.mating.mutation
    if _mutation is not None and hasattr(_mutation, 'flip_log'):
        optimization_results['mutation_diagnostics'] = _mutation.flip_log

    if use_patch_approach and 'patch_mappings' in initial_conditions:
        optimization_results['patch_mappings'] = initial_conditions['patch_mappings']

    # Assemble X_history .npz from batched snapshots
    if save_snapshots:
        hv_cb = callback.hv_callback
        hv_cb._flush_batch()  # flush any remaining buffer
        if hv_cb.batch_files:
            try:
                arrays = [np.load(p)['X'] for p in hv_cb.batch_files]
                X_history = np.concatenate(arrays, axis=0)  # (n_gens, pop_size, n_var)
                os.makedirs(os.path.join(output_dir, 'intermediate_results'), exist_ok=True)
                timestamp = datetime.now().strftime('%d%m_%H%M')
                x_history_path = os.path.join(
                    output_dir, 'intermediate_results', f'X_history_{timestamp}.npz'
                )
                np.savez_compressed(x_history_path, X=X_history)
                for p in hv_cb.batch_files:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                optimization_results['X_history_path'] = x_history_path
                if verbose:
                    print(f"\u2713 X_history saved: {X_history.shape} → {x_history_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not assemble X_history: {e}")

    if save_results:
        save_results_with_reports(optimization_results, output_dir=output_dir, verbose=verbose, run_label=run_label, r_export_parent=r_export_parent)

    return optimization_results


def _print_failure_diagnostics(result, problem, initial_conditions):
    """
    Print diagnostic information when optimization produces no Pareto solutions.
    """
    print("\u2717 Optimization failed - no solutions found")
    if result is None:
        print("  Reason: result is None")
    elif not hasattr(result, 'F'):
        print("  Reason: result has no F attribute")
    elif result.F is None:
        print("  Reason: result.F is None")
        if hasattr(result, 'pop') and result.pop is not None and len(result.pop) > 0:
            pop_G = result.pop.get("G")
            pop_X = result.pop.get("X")
            if pop_G is not None:
                n_feasible = np.sum(np.all(pop_G <= 0, axis=1))
                n_total = len(pop_G)
                
                pop_F = result.pop.get("F")
                if pop_F is not None:
                    # Perform non-dominated sort on the final population
                    nds = NonDominatedSorting()
                    fronts = nds.do(pop_F, only_non_dominated_front=False)
                    n_nondominated = len(fronts[0]) if fronts and len(fronts) > 0 else 0
                    print(f"  Final population had {n_total} solutions, with {n_nondominated} non-dominated and {n_feasible} feasible.")
                else:
                    print(f"  Final population: {n_total} solutions, {n_feasible} feasible (objectives not available for ND sort).")

                if n_feasible == 0 and pop_X is not None:
                    print(f"  Checking actual pixel counts for first 5 solutions:")
                    patch_mappings = initial_conditions.get('patch_mappings')
                    for i in range(min(5, len(pop_X))):
                        x_patches = pop_X[i]
                        pixels = 0
                        if patch_mappings is not None:
                            for j in range(problem.n_restoration_patches):
                                if x_patches[j] == 1:
                                    pixels += len(patch_mappings['restoration_patches']['patch_to_pixels'][j])
                            for j in range(problem.n_conversion_patches):
                                if x_patches[problem.n_restoration_patches + j] == 1:
                                    pixels += len(patch_mappings['conversion_patches']['patch_to_pixels'][j])
                        print(f"    Sol {i}: pixels={pixels}, G={pop_G[i]}")
    elif len(result.F) == 0:
        print("  Reason: result.F has length 0 (no feasible solutions)")
        if hasattr(result, 'pop') and result.pop is not None:
            pop_G = result.pop.get("G")
            if pop_G is not None:
                n_feasible = np.sum(np.all(pop_G <= 0, axis=1))
                print(f"  Population: {len(pop_G)} solutions, {n_feasible} feasible")
                if n_feasible == 0:
                    print(f"  Constraint violations (first 5):")
                    for i in range(min(5, len(pop_G))):
                        print(f"    Solution {i}: G={pop_G[i]}")


def run_optimization_instance(initial_conditions, scenario_params, pop_size=50,
                                     n_generations=100, save_results=True, verbose=True,
                                     skip_diagnostics=False, hv_patience=25,
                                     hv_min_improvement=1e-6, n_jobs=None, use_repair=True,
                                     random_seed=None, use_patch_approach=False, patch_size=100,
                                     patch_constraint_type='pixel_count', pixel_tolerance=0.05,
                                     output_dir=str(OUTPUT_DIR), save_snapshots=False,
                                     run_label="", run_config=None,
                                     n_partitions=8, warm_seeding=True,
                                     r_export_parent=None, mutation_prob_var=None,
                                     mutation_flip_count=None,
                                     capture_repair_diag=False, n_capture_gens=5,
                                     algorithm_type="nsga3"):
    """
    Run the multi-objective restoration optimization for a single scenario.

    Args:
        initial_conditions: Initial objective conditions
        scenario_params: Dict with scenario parameters
        pop_size: Ignored — NSGA-III population size is determined by n_partitions (kept for API compatibility)
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
            - 'pixel_count': Constrain total number of pixels (RECOMMENDED)
            - 'patch_count': Constrain number of patches
        pixel_tolerance: Tolerance for pixel_count constraint (default 0.05 = ±5%)
        output_dir: Directory to save results (default: current directory)
        n_partitions: Number of partitions for NSGA-III Das-Dennis reference directions.
            Controls population size: n_partitions=8 → 45 ref dirs (≈ pop of 45).
            Increase to explore more of the objective space at the cost of more evaluations.
        mutation_prob_var: Per-variable bitflip probability for the mutation operator.
            None (default) reproduces the historical rate of 200 / n_var (i.e. ~200
            expected flips per individual). Pass a float to override, e.g. for a
            mutation-rate sensitivity sweep.
        capture_repair_diag: If True (pixel mode / AdaptiveRepair only), dump paired
            pre-repair vs post-repair genotype matrices and their raw objectives at a
            few evenly-spaced generations to <output_dir>/repair_diagnostics/. Used to
            compare genotype (pairwise Hamming) and phenotype (objective spread)
            diversity before vs after repair. Analyse with
            Debugs_tests/repair_diversity_report.py.
        n_capture_gens: Number of evenly-spaced generations (including first and last)
            to capture when capture_repair_diag is True.
        algorithm_type: Which multi-objective algorithm to run.
            "nsga3" (default) - NSGA-III with Das-Dennis reference directions;
                population size is set by n_partitions (pop_size is ignored).
                Best for >=3 objectives.
            "nsga2" - classic NSGA-II with crowding distance and an explicit
                pop_size. Natural choice for 2-objective problems where the
                Pareto front is a curve.

    Returns:
        dict: Optimization results, or None if optimization failed.
    """
    # --- 1. Initialize patch approach if not already done ---
    if use_patch_approach and not initial_conditions.get('patch_approach_enabled', False):
        #if verbose:
            #print(f"Initializing patch approach with patch_size={patch_size}...")
        initial_conditions = initialize_patch_approach(initial_conditions, patch_size=patch_size)

    # --- 2. Print run header ---
    if verbose:
        approach_str = "PATCH-BASED" if use_patch_approach else "PIXEL-BASED"
        print(f"\n=== SINGLE SCENARIO OPTIMIZATION ({approach_str}) ===")
        print(f"Scenario parameters: {scenario_params}")
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_generations} (with HV early stopping: patience={hv_patience})")
        if use_patch_approach:
            print(f"Eligible patches: {initial_conditions['n_restoration_patches']} restoration + "
                  f"{initial_conditions['n_conversion_patches']} conversion")
        else:
            print(f"Eligible pixels: {initial_conditions['n_pixels']}")

    # --- 3. Create optimization problem ---
    if use_patch_approach:
        problem = PatchRestorationProblem(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            n_jobs=n_jobs,
            patch_constraint_type=patch_constraint_type,
            pixel_tolerance=pixel_tolerance,
        )
    else:
        problem = RestorationProblem(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            n_jobs=n_jobs,
        )

    # --- 4. Print problem details ---
    if verbose:
        print(f"\nOptimization setup details:")
        print(f"  Max action pixels allowed: {problem.max_action_pixels}")
        print(f"  Number of objectives: {len(problem.objective_names)} ({', '.join(problem.objective_names)})")
        sample_obj_str = "  Baseline objectives (no restoration): "
        for obj_name in problem.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
                val = np.sum(initial_conditions[obj_name])
                sample_obj_str += f"{obj_name}={val:.2e}, "
        print(sample_obj_str.rstrip(", "))
        if problem.max_action_pixels == 0:
            print("  WARNING: max_action_pixels is 0! No actions possible.")
        if not skip_diagnostics:
            from debug_utils import diagnose_optimization_setup
            from .logger_setup import setup_logger
            setup_logger()  # no-op if already configured by the entry-point script
            diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10)
            print("✓ Optimisation setup verified.")

    # --- 5. Build operators ---
    # Each run captures into its own subfolder so pixel/patch runs (different
    # genotype dimensions) and successive runs never mix in one directory.
    if capture_repair_diag:
        _diag_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _diag_mode = 'patch' if use_patch_approach else 'pixel'
        _diag_sub = f"{run_label + '_' if run_label else ''}{_diag_mode}_{_diag_stamp}"
        repair_diag_dir = os.path.join(output_dir, 'repair_diagnostics', _diag_sub)
    else:
        repair_diag_dir = None
    sampling, repair, mutation, crossover = _build_operators(
        initial_conditions, scenario_params, problem,
        use_patch_approach, use_repair, patch_constraint_type, pixel_tolerance, verbose,
        warm_seeding=warm_seeding,
        capture_repair_diag=capture_repair_diag, n_capture_gens=n_capture_gens,
        n_generations=n_generations, diag_dir=repair_diag_dir,
    )

    # --- 5b. Warm-start seeding (non-patch path) ---
    # The patch path seeds inside PatchAwareSampling; here we build objective-extreme
    # seeds (static per-pixel scores + single-objective pre-optimisation for
    # arrangement-dependent objectives) and inject them via a generic decorator so
    # the same wrapped sampler feeds both the HV ref point and the algorithm.
    if warm_seeding and not use_patch_approach:
        _sampling_strategy = str(scenario_params.get('sampling_strategy', 'scattered')).lower()
        seed_X = build_warm_start_seeds(
            problem, initial_conditions, scenario_params,
            sampling, repair, mutation, crossover,
            _sampling_strategy, random_seed, verbose=verbose,
        )
        if seed_X is not None:
            sampling = WarmStartSampling(sampling, seed_X)
            if verbose:
                print(f"  Warm-start: injecting {seed_X.shape[0]} objective-extreme "
                      f"seed(s) into the initial population.")

    # --- 6. Build HV reference point ---
    hv_warmup_samples = int(scenario_params.get("hv_warmup_samples", 40 if use_patch_approach else 200))
    if verbose:
        print(f"Building fixed HV ref point with {hv_warmup_samples} warm-up samples...")
    fixed_ref = build_fixed_ref_point(
        problem=problem, sampling=sampling,
        n_samples=hv_warmup_samples, margin=0.05, seed=42, verbose=verbose,
        n_jobs=n_jobs,
    )

    # --- 7. Build algorithm ---
    # mutation_flip_count (int) takes precedence over mutation_prob_var (float) when set.
    if mutation_flip_count is not None:
        mutation_prob_var = mutation_flip_count / problem.n_var
    algorithm, termination = _build_algorithm(problem, sampling, repair, n_generations, n_partitions,
                                              mutation_prob_var=mutation_prob_var, mutation=mutation,
                                              crossover=crossover, algorithm_type=algorithm_type,
                                              pop_size=pop_size)

    # --- 8. Initial sampling quality check (patch mode only) ---
    if use_patch_approach and verbose:
        test_sample = sampling._do(problem, 5)
        if hasattr(problem, 'patch_mappings'):
            patch_mappings = problem.patch_mappings
            pixel_counts = []
            for i in range(test_sample.shape[0]):
                x = test_sample[i]
                pixels = sum(
                    len(patch_mappings['restoration_patches']['patch_to_pixels'][j])
                    for j, s in enumerate(x[:problem.n_restoration_patches]) if s == 1
                ) + sum(
                    len(patch_mappings['conversion_patches']['patch_to_pixels'][j])
                    for j, s in enumerate(x[problem.n_restoration_patches:]) if s == 1
                )
                pixel_counts.append(pixels)
            target_min = int(problem.target_constraint_value * (1 - pixel_tolerance))
            target_max = int(problem.target_constraint_value * (1 + pixel_tolerance))
            in_range = len([p for p in pixel_counts if target_min <= p <= target_max])
            print(f"  Initial sampling check: {in_range}/5 within target [{target_min}, {target_max}]")

    # --- 9. Run optimization ---
    if verbose:
        print(f"Starting optimisation at {datetime.now():%H:%M}...")

    try:
        snap_dir = None
        if save_snapshots:
            snap_dir = os.path.join(output_dir, 'intermediate_results', '_x_snapshot_batches')
        callback = ProgressCallback(
            verbose=verbose,
            n_generations=n_generations,
            hv_patience=hv_patience,
            hv_min_improvement=hv_min_improvement,
            ref_point=fixed_ref,
            save_snapshots=save_snapshots,
            snapshot_dir=snap_dir,
        )
        result = minimize(
            problem, algorithm, termination,
            seed=random_seed, verbose=False, callback=callback,
        )

        if result is not None and hasattr(result, 'F') and result.F is not None and len(result.F) > 0:
            return _package_results(
                result, problem, initial_conditions, scenario_params, callback,
                use_patch_approach, pop_size, n_generations, hv_patience,
                hv_min_improvement, save_results, output_dir, verbose,
                save_snapshots=save_snapshots,
                run_label=run_label, run_config=run_config,
                r_export_parent=r_export_parent,
            )
        else:
            if verbose:
                _print_failure_diagnostics(result, problem, initial_conditions)
            return None

    except Exception as e:
        if verbose:
            print(f"✗ Error during optimization: {e}")
        return None


# --- Main ---

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

    from .run_scenarios import run_all_scenarios_optimization

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
            random_seed=random_seed,
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

    return run_optimization_instance(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params,
        pop_size=pop_size,
        n_generations=n_generations,
        save_results=save_results,
        verbose=verbose,
        skip_diagnostics=False,
    )