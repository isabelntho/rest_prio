"""
Restoration Optimization central code for initialising problems
"""

# --- Imports ---
import os
import numpy as np
from scipy import ndimage
from datetime import datetime

from pymoo.core.problem import ElementwiseProblem
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
from .paths import OUTPUTS
from .patch_approach import (
    create_patch_mappings,
    PatchRepair,
    PatchAwareSampling,
    aggregate_patch_scores_from_pixel_scores,
    aggregate_patch_pixel_sums,
    assign_patches_to_regions,
)
from .scenarios import sample_scenario_parameters
from time import time

# --- Weighting ---

def anomaly_improvement_weight(anomaly_values, shape='exponential', scale=1.0):
    """
    Compute improvement weights based on baseline anomaly values.
    
    Weight function w(a0) is monotonic, peaks at anomaly=0, and decreases as |anomaly| increases.
    
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
        # Gaussian decay: w(a) = exp(-|a|^2/(2*scale^2))
        weights = np.exp(-(abs_anomaly**2) / (2 * scale**2))
    else:
        raise ValueError(f"Unknown weight shape: {shape}. Use 'exponential' or 'gaussian'")
    gamma = 3.0
    weights = weights ** gamma
    weights = np.where(neg_mask, weights, 0.0)
    # Ensure weights are in [0,1] and handle any numerical issues
    weights = np.clip(weights, 0.0, 1.0)
    
    return weights

def _rank_scores(a):
    """Map values to evenly-spaced ranks in [0, 1]; all-constant input returns zeros.

    Distribution-free on purpose. Min-max normalisation leaves a skewed layer with most
    entries nearly tied at one end, and z-scoring fixes the spread but makes the
    effective selection pressure depend on the layer's skew. With ranks the spacing depends only on
    ORDER, so downstream temperature/weight knobs mean the same thing in every region
    and condition scenario.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


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

    if "implementation_cost" in initial_conditions:
        c = initial_conditions["implementation_cost"][elig].astype(np.float64)
        # The historical branch leaves cost negligible: measured on CH the cost term has
        # sd 6.5e-7 against the anomaly terms' 3.2e-3, i.e. 0.02%, so it is a tie-breaker
        # only and this "quality" ranking is effectively pure anomaly. repair_cost_weight
        # > 0 instead blends the two as RANKS, both in [0, 1], so the weight means the
        # same thing whatever the two layers' distributions look like and the result sits
        # in [-w, 1]. Default 0.0 keeps existing runs bit-identical.
        cost_weight = float(scenario_params.get("repair_cost_weight", 0.0))
        if cost_weight > 0.0:
            scores = _rank_scores(scores) - cost_weight * _rank_scores(c)
        else:
            scores = scores - 1e-6 * (c / (np.nanmean(c) + 1e-12))

    return np.asarray(scores, dtype=np.float64)


def build_per_objective_repair_scores(initial_conditions, scenario_params):
    """Per-objective pixel scores for direction-aware patch repair.

    Returns a dict with keys 'abiotic', 'biotic', 'cost', 'restoration_benefit',
    'spatial_clustering', and optionally 'landscape_context' and
    'restoration_potential'.  Each value is a 1-D float64 array over eligible
    pixels, normalized to [0, 1].  Higher score always means "prefer this
    pixel for the corresponding objective"
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

    # Rows for the two objectives that previously had none. Both are static proxies -
    # restoration_benefit's spillover and spatial_clustering's arrangement dependence
    # cannot be expressed per pixel. Only mapped to objectives when
    # scenario_params['direction_aware_scores'] is on (see _map_objectives_to_pixel_scores).
    scores["restoration_benefit"] = np.clip(0.5 * (wa + wb), 0.0, 1.0)

    # Contiguity potential: share of the 4 orthogonal neighbours that are eligible.
    nbr = np.zeros(elig.shape, dtype=np.float64)
    nbr[:, :-1] += elig[:, 1:]
    nbr[:, 1:] += elig[:, :-1]
    nbr[:-1, :] += elig[1:, :]
    nbr[1:, :] += elig[:-1, :]
    scores["spatial_clustering"] = nbr[elig] / 4.0

    if "landscape_context_1d" in initial_conditions:
        ctx = initial_conditions["landscape_context_1d"].astype(np.float64)
        # Favour good-condition surroundings -> higher ctx gets the higher repair score.
        ctx_min, ctx_max = np.nanmin(ctx), np.nanmax(ctx)
        span = ctx_max - ctx_min
        if span > 1e-12:
            scores["landscape_context"] = (ctx - ctx_min) / span
        else:
            scores["landscape_context"] = np.full(len(ctx), 0.5, dtype=np.float64)

    if "restoration_potential_1d" in initial_conditions:
        rp = initial_conditions["restoration_potential_1d"].astype(np.float64)
        # Invert: lower restoration_potential (more degraded) -> higher repair score
        rp_min, rp_max = np.nanmin(rp), np.nanmax(rp)
        span = rp_max - rp_min
        if span > 1e-12:
            scores["restoration_potential"] = 1.0 - (rp - rp_min) / span
        else:
            scores["restoration_potential"] = np.full(len(rp), 0.5, dtype=np.float64)

    return scores


def _map_objectives_to_pixel_scores(problem, initial_conditions, scenario_params):
    """Map run objectives to their per-pixel score arrays.

    Returns (per_obj_pixel_scores, score_keys, score_obj_indices):
      per_obj_pixel_scores : dict from build_per_objective_repair_scores.
      score_keys           : score-dict keys for objectives that HAVE a per-pixel
                             score, row-aligned with...
      score_obj_indices    : ...the objective's column index in the n_obj weight
                             vector.
    spatial_clustering (configuration-level) and restoration_benefit
    (arrangement-dependent via spillover) have no exact per-pixel score. That turned out to be actively harmful for the
    common 3-objective set restoration_benefit / spatial_clustering / cost: it leaves
    exactly one score row, and PatchRepair normalises `w * row` to [0, 1], which is the
    SAME vector for every reference direction - direction-aware repair silently becomes
    an identity, and every individual is repaired toward one ranking. Setting
    scenario_params['direction_aware_scores'] maps both to the static proxies built in
    build_per_objective_repair_scores instead. Default off: it changes which extremes
    PatchAwareSampling seeds, so existing runs stay reproducible.
    """
    per_obj_pixel_scores = build_per_objective_repair_scores(initial_conditions, scenario_params)
    obj_to_score_key = {
        'abiotic_anomaly':       'abiotic',
        'biotic_anomaly':        'biotic',
        'implementation_cost':   'cost',
        'landscape_context':     'landscape_context',
        'restoration_potential': 'restoration_potential',
    }
    if scenario_params.get('direction_aware_scores', False):
        obj_to_score_key['restoration_benefit'] = 'restoration_benefit'
        obj_to_score_key['spatial_clustering'] = 'spatial_clustering'
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

    Defines patches over both restoration- and conversion-eligible areas, so
    decisions are made per patch while objectives stay pixel-level.
    """
    print(f"\nInitializing patch-based approach (patch size: {patch_size}x{patch_size} pixels)...")

    patch_mappings = create_patch_mappings(initial_conditions, patch_size=patch_size)

    initial_conditions['patch_mappings'] = patch_mappings
    initial_conditions['patch_approach_enabled'] = True
    initial_conditions['patch_size'] = patch_size
    initial_conditions['n_restoration_patches'] = patch_mappings['restoration_patches']['n_patches']
    initial_conditions['n_conversion_patches'] = patch_mappings['conversion_patches']['n_patches']

    return initial_conditions

# --- Restoration Effect Functions ---

def restoration_effect(restore_vars, initial_conditions, effect_params=None):
    """
    Define what happens when restoration is selected.
    This function calculates the effect of restoration actions on neighboring cells.

    Args:
        restore_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial objective values
        effect_params: Parameters controlling restoration effects (dict)

    Returns:
        dict: Updated objective values after restoration effects
    """
    assert effect_params is not None

    # Ensure neighbor_radius is integer for array indexing
    effect_params['neighbor_radius'] = int(round(effect_params['neighbor_radius']))

    shape = initial_conditions['shape']
    restoration_eligible_mask = initial_conditions['restoration_eligible_mask']
    restoration_eligible_indices = initial_conditions['restoration_eligible_indices']

    # Create 2D mask from 1D decision variable
    restoration_mask_2d = np.zeros(shape, dtype=bool)

    if np.any(restore_vars):
        # Convert restoration eligible indices with restoration decisions back to 2D coordinates
        restored_indices = restoration_eligible_indices[restore_vars == 1]
        rows, cols = np.divmod(restored_indices, shape[1])
        restoration_mask_2d[rows, cols] = True

    # Initialize updated conditions
    updated_conditions = {}

    # Process only objectives that are available in initial_conditions AND are
    # actual optimisation objectives. Keys loaded purely as data dependencies for
    # computed objectives  are listed in _dependency_keys.
    _dep_only = initial_conditions.get('_dependency_keys', set())
    # restoration_benefit reads the UPDATED abiotic/biotic anomalies (incl. spillover)
    # at evaluation time, so those two must be processed even when they are loaded as
    # data-only dependencies.
    _benefit_deps = ({'abiotic_anomaly', 'biotic_anomaly'}
                     if initial_conditions.get('restoration_benefit_enabled', False) else set())
    available_anomaly_objectives = [obj for obj in ['abiotic_anomaly', 'biotic_anomaly']
                                   if obj in initial_conditions and (obj not in _dep_only or obj in _benefit_deps)]
    
    # abiotic_anomaly and biotic_anomaly share the same action_mask
    # (restoration_mask_2d) and the same radius/kernel, so the neighbor
    # dilation geometry is identical across both objectives - cache it here
    # rather than recomputing ndimage.binary_dilation twice per evaluation.
    # Must stay a call-local variable: every evaluation has a different
    # restoration_mask_2d, so this must never persist across calls.
    _restoration_neighbor_mask = None

    for objective in available_anomaly_objectives:
        original_values = initial_conditions[objective].copy()
        updated_values = original_values.copy()
        
        # Restoration affects abiotic and biotic anomalies
        action_mask = restoration_mask_2d
        improvement_key = f'{objective.split("_")[0]}_effect'
        
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
                is_restoration_mask = action_mask is restoration_mask_2d
                if is_restoration_mask and _restoration_neighbor_mask is not None:
                    neighbor_mask = _restoration_neighbor_mask
                else:
                    # Create kernel for neighbor effects
                    radius = effect_params['neighbor_radius']
                    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                    kernel = (x*x + y*y) <= radius*radius

                    if effect_params.get('spillover_to_restored', False):
                        # Restored cells ALSO gain from other restored cells in range. The
                        # kernel centre is removed so a cell is not its own neighbour, and
                        # action cells are kept (they add spillover on top of their direct
                        # effect below). Still a coverage effect: one gain per cell.
                        kernel = kernel.copy()
                        kernel[radius, radius] = False
                        neighbor_mask = ndimage.binary_dilation(action_mask, structure=kernel)
                    else:
                        # Apply dilation to find neighbor cells
                        neighbor_mask = ndimage.binary_dilation(action_mask, structure=kernel)
                        neighbor_mask = neighbor_mask & ~action_mask  # Exclude direct action cells
                    if is_restoration_mask:
                        _restoration_neighbor_mask = neighbor_mask

                neighbor_improvement = improvement * effect_params['neighbor_effect_decay']

                neighbor_baseline_anomalies = original_values[neighbor_mask]
                neighbor_weights = anomaly_improvement_weight(
                    neighbor_baseline_anomalies, shape=weight_shape, scale=weight_scale
                )

                # Apply weighted neighbor improvement
                weighted_neighbor_improvements = neighbor_improvement * neighbor_weights
                if effect_params.get('spillover_to_restored', False):
                    # Build on updated_values so restored cells keep their direct effect.
                    updated_values[neighbor_mask] = (
                        updated_values[neighbor_mask] + weighted_neighbor_improvements
                    )
                else:
                    updated_values[neighbor_mask] = (
                        original_values[neighbor_mask] + weighted_neighbor_improvements
                    )
        
        # Only apply changes to restoration-eligible pixels.
        # This prevents affecting NaN->0 pixels outside the study area
        updated_values = np.where(restoration_eligible_mask, updated_values, original_values)

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
        updated_conditions['implementation_cost'] = total_cost
    
    return updated_conditions

# --- Optimization Problems ---

class RestorationProblem(ElementwiseProblem):
    """
    Multi-objective restoration optimization problem.
    """
    
    def __init__(self, initial_conditions, scenario_params, pixel_tolerance=None):
        """
        Initialize the optimization problem.

        Args:
            initial_conditions: Dict with initial objective states
            scenario_params: Dict with scenario parameters (e.g., max_restoration_fraction, effect_params)
            pixel_tolerance: fractional slack on the budget constraint. MUST equal the
                tolerance the repair operator uses (see _evaluate). None keeps whatever
                a subclass already set, else 0.0 (exact count).
        """
        if pixel_tolerance is not None:
            self.pixel_tolerance = float(pixel_tolerance)
        elif not hasattr(self, 'pixel_tolerance'):
            self.pixel_tolerance = 0.0
        self.initial_conditions = initial_conditions
        self.scenario_params = scenario_params

        # Extract effect parameters from scenario_params
        abiotic_effect = scenario_params.get('abiotic_effect', 0.01)
        biotic_effect = scenario_params.get('biotic_effect', 0.01)
        self.effect_params = {
            'abiotic_effect': abiotic_effect,
            'biotic_effect': biotic_effect,
            # Neighbour spillover: restoration improves un-restored eligible neighbours
            # within neighbor_radius, decayed by neighbor_effect_decay.
            'neighbor_radius': int(scenario_params.get('neighbor_radius', 3)),
            'neighbor_effect_decay': float(scenario_params.get('neighbor_effect_decay', 0.2)),
            # Opt-in: restored cells also gain spillover from other restored cells in range
            # (rewards clustering). False = the un-restored-neighbours-only model above.
            'spillover_to_restored': bool(scenario_params.get('spillover_to_restored', False)),
        }

        # restoration_potential formulation (Axis 2 - how the restoration target is
        # operationalised): 'sum' / 'threshold' / 'shortfall'. rp_threshold is the
        # target's cutoff/reference. See evaluate_raw_objectives below for what each does.
        self.rp_formulation = str(scenario_params.get('rp_formulation', 'sum')).lower()
        if self.rp_formulation not in ('sum', 'threshold', 'shortfall'):
            raise ValueError(
                f"Unknown rp_formulation: {self.rp_formulation!r}. "
                "Use 'sum', 'threshold' or 'shortfall'.")
        self.rp_threshold = float(scenario_params.get('rp_threshold', 0.0))

        # spatial_clustering metric: 'adjacency' / 'components' / 'largest_patch' /
        # 'inter_patch_adjacency' (needs the patch approach). See evaluate_raw_objectives
        # below for what each measures. The edge-count metrics reward CONTACT and can be
        # driven a long way by scattered plans whose patches merely touch in pairs;
        # 'largest_patch' (LPI) rewards CONSOLIDATION instead.
        self.clustering_metric = str(scenario_params.get('clustering_metric', 'adjacency')).lower()
        # First-order per-pixel condition gain from restoration, applied to the combined
        # (abiotic + biotic) restoration_potential score used by the 'threshold' and
        # 'shortfall' formulations. This is the NOMINAL gain: the engine's actual
        # per-pixel improvement in restoration_effect is scaled by
        # anomaly_improvement_weight, but both formulations use this constant.
        self.rp_improvement = 0.5 * (abiotic_effect + biotic_effect)

        # Determine which objectives are available. Order here IS the objective
        # order. A '*_enabled' key is a flag; any other key must be present in
        # initial_conditions and not loaded data-only (see _dependency_keys).
        _candidates = [
            ('abiotic_anomaly',       'abiotic_anomaly'),
            ('biotic_anomaly',        'biotic_anomaly'),
            ('landscape_context',     'landscape_context_1d'),
            ('restoration_potential', 'restoration_potential_1d'),
            ('restoration_benefit',   'restoration_benefit_enabled'),
            ('spatial_clustering',    'spatial_clustering_enabled'),
            ('restored_area',         'restored_area_enabled'),
            ('implementation_cost',   'implementation_cost'),
            ('es_future_val',         'es_future_val_1d'),
            ('es_future_robustness',  'es_future_robustness_1d'),
        ]
        _dep_only = set(initial_conditions.get('_dependency_keys', ()))

        def _available(key):
            if key.endswith('_enabled'):
                return bool(initial_conditions.get(key, False))
            return key in initial_conditions and key not in _dep_only

        self.objective_names = [name for name, key in _candidates if _available(key)]

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

        # Objective scaling reference:
        #   'theoretical' (default) - the landscape-wide maxima in
        #       _compute_normalization_denominators.
        #   'attainable'  - the largest |raw| value a budget-feasible plan actually
        #       reaches (_compute_attainable_scales).
        # A theoretical reference can sit orders of magnitude above anything the budget
        # can buy, which compresses that objective toward zero in normalised space and
        # hands dominance to the other objectives. Measured on CH (5% budget,
        # abiotic/biotic effect 0.01): restoration_benefit spanned 1.5e-4 between its
        # greedy extremes against 1.7e-1 for spatial_clustering - 1168x narrower, i.e.
        # decided at the level of float noise. Must run after max_action_pixels and the
        # pixel counts above, which the probe evaluations need.
        self.objective_scaling = str(
            scenario_params.get('objective_scaling', 'theoretical')).lower()
        if self.objective_scaling not in ('theoretical', 'attainable'):
            raise ValueError(
                f"Unknown objective_scaling: {self.objective_scaling!r}. "
                "Use 'theoretical' or 'attainable'.")
        if self.normalize_objectives and self.objective_scaling == 'attainable':
            self.objective_scales = self._compute_attainable_scales()


        # Binary decision variables: 0 = no action, 1 = action
        # First n_restoration_pixels elements = restoration decisions
        # Next n_conversion_pixels elements = conversion decisions
        total_decision_vars = n_restoration_pixels + n_conversion_pixels

        self.constraint_log = []

        # Initialize the problem
        super().__init__(
            n_var=total_decision_vars, n_obj=n_objectives,
            n_constr=(2 if self.min_patch_size > 1 else 1),
            xl=0, xu=1, type_var=int, elementwise=True,
        )

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
            elif obj_name == 'landscape_context':
                ctx = self.initial_conditions['landscape_context_1d']
                # Scale = sum of absolute values over all restoration-eligible pixels
                scale = float(np.nansum(np.abs(ctx)))
            elif obj_name == 'restoration_potential':
                rp = self.initial_conditions['restoration_potential_1d']
                _rp_form = getattr(self, 'rp_formulation', 'sum')
                if _rp_form == 'threshold':
                    # Count-based objective: scale by the number of eligible pixels
                    # so the normalised value falls in [-1, 0].
                    scale = float(len(rp))
                elif _rp_form == 'shortfall':
                    # Gain-based objective: scale by the total closable shortfall over
                    # ALL eligible pixels (the value of restoring everything), so the
                    # normalised value falls in [-1, 0]. Same "max attainable" logic as
                    # the threshold form's pixel count.
                    _ref = getattr(self, 'rp_threshold', 0.0)
                    _imp = getattr(self, 'rp_improvement', 0.0)
                    scale = float(np.nansum(np.minimum(rp + _imp, _ref)
                                            - np.minimum(rp, _ref)))
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
            elif obj_name == 'restored_area':
                # Theoretical ceiling: every restoration-eligible pixel restored. Under
                # 'pixel_count' this objective is constant at max_action_pixels and
                # would be pointless; it exists for 'cost_budget', where area varies
                # and this is the natural scale. Additive (a plain pixel sum), so the
                # standard greedy probes in _compute_attainable_scales are true extremes
                # for it - no compact-probe special case needed, unlike spatial_clustering.
                scale = float(self.initial_conditions['n_restoration_pixels'])
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
                _metric = getattr(self, 'clustering_metric', 'adjacency')
                if _metric == 'largest_patch':
                    # Already a share of the selected area, so it is its own scale.
                    scale = 1.0
                elif _metric == 'components':
                    # Worst case: every selected pixel its own component -> ~max_pix
                    # clusters. Scale by the pixel budget so normalised in ~[0, 1].
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

    def _compact_probe_order(self, budget):
        """Pixel indices of one contiguous grown plan, or None if it cannot be built.

        Grown from a single seed by a random frontier walk, so the result is a compact
        blob rather than a scattered selection - the arrangement analogue of the greedy
        cost/anomaly orders. Returned as an index array so the caller can treat it like
        any other probe order.
        """
        try:
            nbr, _rows, _cols = build_restoration_neighbor_table(self.initial_conditions)
            n_rest = int(self.n_restoration_pixels)
            selected = grow_region_plan(
                nbr, n_rest, int(budget), 1,
                np.zeros(n_rest, dtype=np.float64), 'neutral',
                np.random.default_rng(0))
            idx = np.flatnonzero(selected)
            return idx if idx.size > 0 else None
        except Exception as exc:
            print(f"  (compact probe unavailable, arrangement objectives keep a "
                  f"pixel-greedy reference: {exc})")
            return None

    def _compute_attainable_scales(self):
        """Scale each objective by the largest |raw| value a feasible plan reaches.

        Evaluates budget-feasible probe plans - greedy cheapest, greedy dearest, greedy
        highest baseline anomaly, one random, and (when an arrangement-dependent
        objective is present) one spatially grown compact plan - and takes the
        per-objective maximum |raw|. Objectives whose probes are all ~0 keep their
        theoretical scale.

        The greedy orders are genuine extremes for ADDITIVE objectives (cost, anomaly
        sums). They are NOT extremes for arrangement-dependent ones (spatial_clustering,
        and restoration_benefit via spillover), because a pixel-greedy selection never
        tries to clump. Without the compact probe the clustering denominator is just
        another un-clustered plan, which makes any "fraction of attainable" figure
        self-referential: the CH run reported 99.6% of its probe scale while reaching
        only ~45% of what a compact arrangement can structurally reach.

        Probes select pixels freely, so in patch mode the reference is mildly
        optimistic against what whole-patch decisions can reach; it is a per-objective
        constant either way, so dominance is unaffected. Conversion pixels are left at
        zero, so objectives driven by conversion keep their theoretical scale.

        constraint_type='cost_budget' (PatchRestorationProblem only): a FIXED PIXEL
        COUNT probe would badly underestimate objectives like restored_area, whose
        whole point is to vary with how cheaply the budget can be spent - e.g. on Bern,
        plans reaching 3x the pixel_count-equivalent area for the same spend are routine
        (see [[ch-3obj-front-is-one-dimensional]] Stage 3). Each probe order instead
        fills by CUMULATIVE COST up to target_constraint_value, exactly mirroring what
        the real fill loop (PatchAwareSampling/_enforce_budget) does.
        """
        eps = 1e-12
        n_rest = int(self.n_restoration_pixels)
        budget = min(int(self.max_action_pixels), n_rest)
        scales = dict(self.objective_scales)
        if budget <= 0:
            return scales

        ic = self.initial_conditions
        mask = ic['restoration_eligible_mask']

        constraint_type = getattr(self, 'patch_constraint_type', 'pixel_count')
        cost_budget_mode = constraint_type == 'cost_budget'
        if cost_budget_mode:
            cost_for_probe = np.nan_to_num(
                getattr(self, '_cost_restore_1d', None)
                if getattr(self, '_cost_restore_1d', None) is not None
                else np.asarray(ic['implementation_cost'], dtype=float)[mask])
            target_cost = float(self.target_constraint_value)

        def _fill(order):
            """Prefix of `order` to select: by cumulative cost under cost_budget,
            else the fixed pixel `budget` (unchanged default behaviour)."""
            if not cost_budget_mode:
                return order[:budget]
            csum = np.cumsum(cost_for_probe[order])
            k = min(int(np.searchsorted(csum, target_cost, side='right')) + 1, len(order))
            return order[:k]

        orders = []
        if 'implementation_cost' in ic:
            c = np.asarray(ic['implementation_cost'])[mask].astype(float)
            asc = np.argsort(c, kind='stable')
            orders.append(asc)          # cheapest
            orders.append(asc[::-1])    # dearest
        if 'abiotic_anomaly' in ic and 'biotic_anomaly' in ic:
            gain = (np.abs(np.asarray(ic['abiotic_anomaly'])[mask])
                    + np.abs(np.asarray(ic['biotic_anomaly'])[mask]))
            orders.append(np.argsort(-gain, kind='stable'))
        orders.append(np.random.default_rng(0).permutation(n_rest))

        # One contiguous grown plan, so arrangement-dependent objectives get a probe
        # that actually clumps. Skipped when no such objective is in play, since the
        # frontier walk is the slowest probe. Still grown to the PIXEL budget even
        # under cost_budget (grow_region_plan has no cost-aware stopping rule) - an
        # approximation already flagged as "mildly optimistic" above; only affects
        # spatial_clustering/restoration_benefit's scale precision, not restored_area's.
        if {'spatial_clustering', 'restoration_benefit'}.intersection(self.objective_names):
            compact = self._compact_probe_order(budget)
            if compact is not None:
                orders.append(compact)

        best = np.zeros(len(self.objective_names), dtype=float)
        for order in orders:
            x = np.zeros(n_rest + int(self.n_conversion_pixels), dtype=int)
            x[_fill(order)] = 1
            # Explicit unbound call: self.evaluate_raw_objectives would dispatch to
            # PatchRestorationProblem's override, which reads x as patch decisions.
            raw = RestorationProblem.evaluate_raw_objectives(self, x)
            best = np.maximum(best, np.nan_to_num(np.abs(np.asarray(raw, dtype=float))))

        for i, obj_name in enumerate(self.objective_names):
            if best[i] > eps:
                scales[obj_name] = float(best[i])

        print(f"Objective scaling: attainable ({len(orders)} probe plans, "
              f"budget {budget} px)")
        for obj_name in self.objective_names:
            print(f"  {obj_name:22s} {self.objective_scales.get(obj_name, 1.0):.6g}"
                  f" -> {scales[obj_name]:.6g}")
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

        updated_conditions = restoration_effect(x_restore, self.initial_conditions, self.effect_params)

        raw_objectives = []
        for obj_name in self.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                base = self.initial_conditions[obj_name]
                mask = self.initial_conditions["restoration_eligible_mask"]
                obj_value = -np.sum((updated_conditions[obj_name] - base)[mask])
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
                elif self.rp_formulation == 'shortfall':
                    # Shortfall-closure formulation: sum the reduction in the gap between
                    # condition and the reference level (rp_threshold), crediting
                    # improvement only up to the reference and nothing beyond it:
                    #   closure = min(rp + improvement, ref) - min(rp, ref)
                    # Pixels already at/above the reference contribute nothing; pixels
                    # more than `improvement` below it contribute the full improvement;
                    # pixels in between contribute the part that closes the gap.
                    # Unselected eligible pixels contribute 0, so this is equivalently
                    # the sum over ALL eligible pixels. Negated so the minimiser
                    # maximises total closure.
                    sel_rp = rp[sel]
                    closed = (np.minimum(sel_rp + self.rp_improvement, self.rp_threshold)
                              - np.minimum(sel_rp, self.rp_threshold))
                    obj_value = -float(np.sum(closed))
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
                if self.clustering_metric == 'largest_patch':
                    # Largest Patch Index: share of the selected area sitting in its
                    # biggest 4-connected component. Negated so the minimiser maximises
                    # it. Unlike the edge-count metrics this measures CONSOLIDATION
                    # rather than contact: the CH run drove inter_patch_adjacency 1.6x
                    # across its front while LPI stayed at 2.8-5.0%, i.e. many 3x3
                    # patches touching in pairs, scattered nationwide.
                    lab, n_comp = ndimage.label(sel_mask)
                    n_sel = int(sel_mask.sum())
                    if n_comp > 0 and n_sel > 0:
                        biggest = int(np.bincount(lab.ravel())[1:].max())
                        obj_value = -float(biggest) / float(n_sel)
                    else:
                        obj_value = 0.0
                elif self.clustering_metric == 'components':
                    # Number of disconnected clusters (4-connectivity). Fewer = more
                    # clumped. Minimised directly. Insensitive to cluster size/shape
                    # -- measures pure fragmentation.
                    # FALSIFIED 2026-09-18 (Stage 4.2, CH): the older claim that this
                    # "need not track cost the way edge count does" was untested and is
                    # WRONG - optimising it directly gave corr(clustering,cost)=+0.928
                    # (PC1 97.3%), a MUCH stronger cost coupling than
                    # inter_patch_adjacency's +0.194 (PC1 70.0%) under the same fixed
                    # operators, with no meaningful consolidation gain (LPI similar
                    # range). See [[ch-3obj-front-is-one-dimensional]].
                    _, n_comp = ndimage.label(sel_mask)
                    obj_value = float(n_comp)
                elif self.clustering_metric == 'inter_patch_adjacency':
                    # Inter-patch adjacency: orthogonal shared edges between selected
                    # pixels that lie in different patches. Excludes the internal
                    # edges guaranteed inside each fully-selected patch. Negated so
                    # the minimiser maximises inter-patch contact. Requires the patch approach.
                    pm = self.initial_conditions.get('patch_mappings')
                    patch_grid = (pm['restoration_patches']['patch_grid']
                                  if pm is not None else None)
                    if patch_grid is None:
                        horiz = np.count_nonzero(sel_mask[:, :-1] & sel_mask[:, 1:])
                        vert = np.count_nonzero(sel_mask[:-1, :] & sel_mask[1:, :])
                    else:
                        # Bitwise & binds tighter than !=, so parenthesise the patch-id comparison.
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
                    # selected pixels sharing an edge. Negated so the pymoo minimiser maximises it.
                    horiz = np.count_nonzero(sel_mask[:, :-1] & sel_mask[:, 1:])
                    vert = np.count_nonzero(sel_mask[:-1, :] & sel_mask[1:, :])
                    obj_value = -float(horiz + vert)
            elif obj_name == 'es_future_val':
                # Sum of per-pixel ES performance over selected restoration pixels.
                # Higher = greater total future ES gain -> maximise (negate for pymoo minimisation).
                esv = self.initial_conditions['es_future_val_1d']
                obj_value = -float(np.sum(esv[x_restore == 1]))
            elif obj_name == 'es_future_robustness':
                # Sum of per-pixel ES instability over selected restoration pixels.
                # Lower = less total undesirable deviation = more robust -> minimise directly.
                esr = self.initial_conditions['es_future_robustness_1d']
                obj_value = float(np.sum(esr[x_restore == 1]))
            elif obj_name == 'restored_area':
                # Maximise total restored pixels -> negate for pymoo minimisation.
                # Constant under 'pixel_count' (always max_action_pixels); meaningful
                # only when the constraint lets area vary (patch_constraint_type=
                # 'cost_budget'), where it is the natural counterpart to cost.
                obj_value = -float(np.sum(x_restore))
            elif obj_name == 'implementation_cost':
                obj_value = updated_conditions[obj_name]
            else:
                raise ValueError(f"Unknown objective: {obj_name}")

            raw_objectives.append(float(obj_value))

        return raw_objectives

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a restoration plan. x is [restoration decisions over the
        n_restoration_pixels eligible pixels | unused trailing slots], already
        repaired by sampling/repair.
        """
        x_restore = x[:self.n_restoration_pixels]
        x_convert = x[self.n_restoration_pixels:self.n_restoration_pixels + self.n_conversion_pixels]

        # Conversion is unsupported: always hold these decisions at zero so they
        # consume no budget.
        x_convert[:] = 0

        n_total_actions = np.sum(x_restore)

        # x here is always a pixel vector (PatchRestorationProblem._evaluate converts
        # patches to pixels before calling super()._evaluate). Must call the
        # unbound RestorationProblem method explicitly: self.evaluate_raw_objectives
        # would dispatch to PatchRestorationProblem's override when self is a patch
        # problem, which would wrongly reinterpret this pixel vector as patches.
        raw_objectives = RestorationProblem.evaluate_raw_objectives(self, x)
        objectives = self._normalize_objective_vector(raw_objectives)
        out["F"] = objectives
        out["F_raw"] = raw_objectives
        
        # Constraint: total number of pixels with actions (restore + convert), feasible
        # anywhere inside the band the repair enforces: [int(k*(1-tol)), int(k*(1+tol))]
        # (enforce_budget_contiguous). NOTE: this used to be abs(n - k), an EXACT count,
        # while the repair only guarantees the band. Plans the repair left inside the band
        # but off exactly k were therefore infeasible and lost to constraint-domination,
        # while HVCallback counts them, so population hypervolume could fall between
        # generations. Keep pixel_tolerance identical to the repair operators' tolerance.
        k = self.max_action_pixels
        lo = int(k * (1 - self.pixel_tolerance))
        hi = int(k * (1 + self.pixel_tolerance))
        out["G"] = [float(max(0, lo - n_total_actions, n_total_actions - hi))]

        # Minimum-patch-size constraint (price-of-contiguity sweep). Feasible when
        # every 4-connected component of selected restoration pixels has >= S pixels;
        # violation = max(0, S - smallest_component). MinPatchSizeRepair guarantees
        # this, so out["G"][1] audits the repair (should stay 0).
        if self.min_patch_size > 1:
            out["G"].append(self._min_patch_size_violation(x_restore))

        # Log constraint violations
        _ctypes = ['budget', 'min_patch_size']
        for i, g_val in enumerate(out["G"]):
            if g_val > 0:
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
    - 'cost_budget': Fixed total implementation cost (variable pixel/patch count and
      area). The counterpart to 'pixel_count' for testing whether area-vs-value
      trade-offs are an artefact of the fixed-area constraint (see
      [[ch-3obj-front-is-one-dimensional]]). Pairs naturally with the 'restored_area'
      objective, which is otherwise constant under 'pixel_count'.
    """
    
    def __init__(self, initial_conditions, scenario_params,
                 patch_constraint_type='pixel_count', pixel_tolerance=0.05):
        """
        pixel_tolerance is the fractional slack on the 'pixel_count' constraint
        (0.05 = +/-5%); _evaluate widens it by 50% to absorb whole-patch
        discretisation that repair cannot remove.
        """
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

        # Store constraint configuration NOW, before super().__init__(): the base
        # class's __init__ calls _compute_attainable_scales() (when objective_scaling=
        # 'attainable'), which needs self.patch_constraint_type / self._cost_restore_1d
        # / self.target_constraint_value already set to probe a cost_budget run
        # correctly (see that method's cost_budget_mode branch) - setting them AFTER
        # super().__init__() would make attainable scaling silently probe a fixed pixel
        # count instead, understating any objective (e.g. restored_area) that varies
        # with how cheaply the budget is spent. Everything needed here (cost arrays,
        # target_constraint_value for 'cost_budget') comes directly from
        # initial_conditions/scenario_params, not from anything the base class computes.
        self.patch_constraint_type = patch_constraint_type
        self.pixel_tolerance = pixel_tolerance
        # Conversion is unsupported: conversion patches change no objective, so they
        # must not count toward the budget either. The operators (PatchAwareSampling /
        # PatchRepair) never select them; this makes the constraint agree even for a
        # plan that did arrive with conversion bits set.
        self._exclude_conversion = True

        # Per-pixel cost arrays for the 'cost_budget' constraint, cached once here so
        # _evaluate does not re-mask the full raster every call. Indexed the same way
        # as x_restore/x_convert (boolean-mask indexing preserves row-major flatten
        # order, matching restoration_eligible_indices/conversion_eligible_indices).
        self._cost_restore_1d = None
        self._cost_convert_1d = None
        if patch_constraint_type == 'cost_budget':
            if 'implementation_cost' not in initial_conditions:
                raise ValueError(
                    "patch_constraint_type='cost_budget' requires 'implementation_cost' "
                    "in initial_conditions (i.e. 'cost' must be one of the loaded "
                    "objectives).")
            cost2d = np.asarray(initial_conditions['implementation_cost'], dtype=np.float64)
            self._cost_restore_1d = np.nan_to_num(
                cost2d[initial_conditions['restoration_eligible_mask']], nan=0.0)
            conv_mask = initial_conditions.get('conversion_eligible_mask')
            self._cost_convert_1d = (
                np.nan_to_num(cost2d[conv_mask], nan=0.0) if conv_mask is not None
                else np.zeros(int(initial_conditions['n_conversion_pixels']), dtype=np.float64))
            # Mirrors max_action_pixels = max_restoration_fraction * n_restoration_pixels
            # (computed independently below by the base class) - same fraction, applied
            # to total RESTORATION-eligible cost instead of pixel count.
            max_restoration_fraction = float(scenario_params.get('max_restoration_fraction', 0.05))
            self.target_constraint_value = float(
                max_restoration_fraction * np.sum(self._cost_restore_1d))
        elif patch_constraint_type not in ('pixel_count', 'patch_count'):
            raise ValueError(
                f"Unknown patch_constraint_type: {patch_constraint_type!r}. "
                "Use 'pixel_count', 'patch_count', or 'cost_budget'.")

        # Initialize parent class with pixel-level information
        # This sets up all the objective functions and constraints
        # We'll override n_var after parent initialization
        super().__init__(initial_conditions, scenario_params)

        # Override decision variable dimensions to use patches instead of pixels
        # Decision vector structure: [restoration_patches, conversion_patches]
        self.n_var = self.n_restoration_patches + self.n_conversion_patches

        # Calculate target values for the constraint types that need base-class state
        # (max_action_pixels, set inside super().__init__() just called). 'cost_budget'
        # was already computed above - nothing here needs redoing for it.
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
        # 'cost_budget' target_constraint_value was already computed before
        # super().__init__() above (see the comment there for why); unknown types were
        # already rejected there too.

        print(f"Patch-based problem initialized:")
        print(f"  Decision variables: {self.n_var} patches "
              f"({self.n_restoration_patches} restoration + {self.n_conversion_patches} conversion)")


    def _evaluate(self, x_patches, out, *args, **kwargs):
        """
        Evaluate a patch-level plan: [restoration patches | conversion patches].
        Expanded to pixels, then scored by the parent's pixel-level _evaluate.
        """
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
        if self._exclude_conversion:
            x_convert_pixels = np.zeros_like(x_convert_pixels)

        x_pixels = np.concatenate([x_restore_pixels, x_convert_pixels])
        super()._evaluate(x_pixels, out, *args, **kwargs)

        # Override the parent's budget constraint with the patch-mode one.
        if self.patch_constraint_type == 'patch_count':
            n_patches_used = np.sum(x_restore_patches) + np.sum(x_convert_patches)
            constraint_value = abs(n_patches_used - self.target_constraint_value)

        elif self.patch_constraint_type == 'pixel_count':
            n_restore_pixels = np.sum(x_restore_pixels)
            n_convert_pixels = np.sum(x_convert_pixels)
            n_pixels_used = n_restore_pixels + n_convert_pixels

            # Wider than the repair's tolerance: whole-patch granularity leaves
            # discretisation error that repair cannot remove.
            evaluation_tolerance = self.pixel_tolerance * 1.5


            min_pixels = int(self.target_constraint_value * (1 - evaluation_tolerance))
            max_pixels = int(self.target_constraint_value * (1 + evaluation_tolerance))
            
            # Only the first few violations are printed below.
            self._constraint_debug_count = getattr(self, '_constraint_debug_count', 0) + 1

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

        elif self.patch_constraint_type == 'cost_budget':
            # Mirrors the 'pixel_count' branch above, tracking total COST instead of
            # pixel count. Same widened tolerance rationale: whole-patch granularity
            # leaves discretisation error PatchRepair's tolerance cannot fully remove.
            cost_used = (float(np.dot(x_restore_pixels, self._cost_restore_1d))
                         + float(np.dot(x_convert_pixels, self._cost_convert_1d)))
            evaluation_tolerance = self.pixel_tolerance * 1.5
            min_cost = self.target_constraint_value * (1 - evaluation_tolerance)
            max_cost = self.target_constraint_value * (1 + evaluation_tolerance)

            if min_cost <= cost_used <= max_cost:
                constraint_value = 0.0
            else:
                constraint_value = min(abs(cost_used - min_cost), abs(cost_used - max_cost))

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

def build_fixed_ref_point(problem, sampling, n_samples=200, margin=0.05, seed=42, verbose=False):
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
    is observed for a specified number of generations. Hypervolume is computed on the
    feasible plans only; per-generation feasible share and min/max restored pixels are logged.
    """

    def __init__(self, patience=15, min_improvement=1e-6, verbose=True, ref_point=None,
                 snapshot_generations=None):
        """
        patience: generations without a >= min_improvement RELATIVE HV gain before
        declaring convergence. ref_point is required (fixed across the run so HV is
        comparable between generations); see build_fixed_ref_point.
        snapshot_generations: optional iterable of algorithm.n_gen values to snapshot
        X for (e.g. {1, 10, 25, 50, 100}). None snapshots every generation.
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
        # Feasibility tracking. HV is computed on FEASIBLE plans only (constraint-
        # domination discards the rest, so counting them made HV fall between generations).
        self.feasible_share_history = []   # fraction of the population with CV <= 0
        self.n_action_min_history = []     # min/max restored pixels per generation
        self.n_action_max_history = []     # (pixel mode only; None-free, empty in patch mode)

        # X snapshot buffering (populated only when snapshot_dir is set)
        self.snapshot_dir = None       # set by ProgressCallback when save_snapshots=True
        self.snapshot_generations = (
            set(snapshot_generations) if snapshot_generations is not None else None
        )  # None = every generation; otherwise only these algorithm.n_gen values
        self.X_batch = []              # in-memory buffer of int8 arrays
        self.batch_files = []          # paths of flushed batch .npz files
        self.batch_size = 10           # flush every N generations
        self._n_generations_est = 100  # updated by ProgressCallback for memory estimate
        self._x_memory_warned = False  # print estimate only once
    
    def __call__(self, algorithm):
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

                feasible = self._feasible_mask(algorithm, F)
                self.feasible_share_history.append(float(feasible.mean()))
                self._log_action_counts(algorithm)

                # Capture full population X snapshot (all generations, or only
                # snapshot_generations if that was given).
                if self.snapshot_dir is not None:
                    if not self._x_memory_warned and algorithm.n_gen == 1:
                        X_pop_est = algorithm.pop.get("X")
                        if X_pop_est is not None:
                            n_snap = (
                                len(self.snapshot_generations)
                                if self.snapshot_generations is not None
                                else self._n_generations_est
                            )
                            batch_mb = self.batch_size * X_pop_est.shape[0] * X_pop_est.shape[1] / 1e6
                            total_mb = n_snap * X_pop_est.shape[0] * X_pop_est.shape[1] / 1e6
                            print(f"   X snapshots: {X_pop_est.shape[1]} vars x {X_pop_est.shape[0]} pop, "
                                  f"{n_snap} generation(s) targeted, "
                                  f"batch RAM ~{batch_mb:.0f} MB, est. total on disk ~{total_mb:.0f} MB")
                            if total_mb > 1000:
                                print(f"   Warning: estimated X_history size exceeds 1 GB - consider "
                                      f"narrowing snapshot_generations")
                            self._x_memory_warned = True

                    if self.snapshot_generations is None or algorithm.n_gen in self.snapshot_generations:
                        X_pop = algorithm.pop.get("X")
                        if X_pop is not None:
                            self.X_batch.append(X_pop.astype(np.int8))
                            if len(self.X_batch) >= self.batch_size:
                                self._flush_batch()

                # Calculate hypervolume
                try:
                    # Create reference point (worst case for each objective)
                    if self.ref_point is None:
                        raise ValueError("HVCallback requires a fixed ref_point")

                    hv_indicator = HV(ref_point=self.ref_point)
                    current_hv = float(hv_indicator(F[feasible])) if feasible.any() else 0.0
                    
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
                # F is None or empty - keep hv_history length-aligned
                if self.verbose:
                    print(f"   Warning: HVCallback skipped at gen {algorithm.n_gen}: pop F unavailable")
                self.hv_history.append(float('nan'))
        else:
            # pop unavailable - keep hv_history length-aligned
            if self.verbose:
                print(f"   Warning: HVCallback skipped at gen {getattr(algorithm, 'n_gen', '?')}: pop unavailable")
            self.hv_history.append(float('nan'))

    @staticmethod
    def _feasible_mask(algorithm, F):
        """Boolean mask of plans with no constraint violation (all True if unconstrained)."""
        CV = algorithm.pop.get("CV")
        if CV is None:
            return np.ones(len(F), dtype=bool)
        return np.asarray(CV, dtype=float).reshape(len(F), -1).max(axis=1) <= 0

    def _log_action_counts(self, algorithm):
        """Record min/max restored pixels in the population (pixel mode only)."""
        problem = getattr(algorithm, 'problem', None)
        if problem is None or hasattr(problem, 'n_restoration_patches'):
            return
        X = algorithm.pop.get("X")
        if X is None:
            return
        n = X[:, :problem.n_restoration_pixels].sum(axis=1)
        self.n_action_min_history.append(int(n.min()))
        self.n_action_max_history.append(int(n.max()))

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
    Reports generation progress and delegates hypervolume tracking to
    HVCallback.
    """

    def __init__(self, verbose=True, n_generations=100,
                 hv_patience=15, hv_min_improvement=1e-6, ref_point=None,
                 save_snapshots=False, snapshot_dir=None, snapshot_generations=None):
        """
        hv_* and ref_point are forwarded to HVCallback. save_snapshots captures the
        full population X into snapshot_dir as batched .npz files - every generation,
        or only snapshot_generations (an iterable of algorithm.n_gen values) if given.
        """
        self.verbose = verbose
        self.n_generations = n_generations
        self.start_time = None
        self.hv_callback = HVCallback(
            patience=hv_patience,
            min_improvement=hv_min_improvement,
            verbose=verbose,
            ref_point=ref_point,
            snapshot_generations=snapshot_generations,
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

            feas = self.hv_callback.feasible_share_history
            feas_info = f" - Feasible: {feas[-1]:.0%}" if feas else ""
            print(f"   Generation {gen}/{self.n_generations} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed/60:.1f}min - ETA: {eta/60:.1f}min{violation_info}{feas_info}")

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
        (sampling, repair) - repair is None when use_repair is False.
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
        # aggregate_patch_scores_from_pixel_scores min-max normalises to [0, 1], and the
        # underlying score is heavily skewed, so most patches end up nearly tied near one
        # end and only the tail is distinguishable. 'rank' respaces them evenly, which
        # both spreads the middle of the distribution and makes the knob below exact:
        # with ranks uniform on [0, 1], _safe_softmax's (x - max)/T gives the best patch
        # exp(1/T) times the worst patch's draw probability, independent of the layer.
        # Default 'none' changes no existing run.
        transform = str(scenario_params.get('patch_score_transform', 'none')).lower()
        if transform == 'rank':
            patch_scores = _rank_scores(patch_scores)
        elif transform != 'none':
            raise ValueError(
                f"Unknown patch_score_transform: {transform!r}. Use 'none' or 'rank'.")

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
        # Objectives WITHOUT a meaningful per-patch score (e.g. spatial_clustering) 
        # are intentionally absent - no warm-seed or direction-aware repair possible.
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

        # 'cost_budget': every fill/repair loop must track total COST, not pixel count.
        # None (default) for 'pixel_count'/'patch_count' - PatchAwareSampling/PatchRepair
        # then fall back to pixel counts exactly as before this constraint type existed.
        budget_weights = None
        # No conversion objective: conversion patches are inert, so keep them out of the
        # draw and out of the budget (the constraint in PatchRestorationProblem agrees).
        exclude_conversion = bool(getattr(problem, '_exclude_conversion', False))
        if patch_constraint_type == 'cost_budget':
            budget_weights = aggregate_patch_pixel_sums(
                patch_mappings=initial_conditions['patch_mappings'],
                restoration_pixel_values=problem._cost_restore_1d,
                conversion_pixel_values=(np.zeros_like(problem._cost_convert_1d)
                                         if exclude_conversion else problem._cost_convert_1d),
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
            budget_weights=budget_weights,
            exclude_conversion=exclude_conversion,
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
            budget_weights=budget_weights,
            exclude_conversion=exclude_conversion,
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
            # can reach clustered plans where cost and spatial_clustering
            # actually vary. region_evolve additionally makes the SEARCH explore the
            # landscape (relocate/spawn/swap whole regions) instead of collapsing onto
            # one basin. See [[hv-stagnation-flat-objectives]].
            region_seeds = int(scenario_params.get('region_seeds', 25))
            region_seeds_min = scenario_params.get('region_seeds_min', None)
            region_random_share = float(scenario_params.get('region_random_share', 0.0))
            growth_bias = str(scenario_params.get('region_growth_bias', 'scored')).lower()
            region_edits = int(scenario_params.get('region_mutation_edits', 100))
            region_seed_grid = int(scenario_params.get('region_seed_grid', 16))
            # Stochastic ('scored') pixel ordering temperature. 0 = the historical
            # deterministic argsort, which makes independent individuals growing in the
            # same area converge on the IDENTICAL pixel set (the blocky, uniform-value
            # selection-frequency map). Only region_evolve opts in below; region_grow
            # keeps 0.0 so its runs stay reproducible against earlier results.
            score_temperature = 0.0

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
                score_temperature = float(scenario_params.get('region_score_temperature', 1.0))
                mutation = RegionEvolveMutation(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    n_edits=region_edits, growth_bias=growth_bias,
                    pixel_tolerance=pixel_tolerance,
                    score_temperature=score_temperature,
                )
                crossover = RegionSwapCrossover(
                    initial_conditions, problem.max_action_pixels, region_scores,
                    growth_bias=growth_bias, pixel_tolerance=pixel_tolerance,
                    score_temperature=score_temperature,
                )
                if verbose:
                    print(f"Using region-evolve operators (seeds<={region_seeds}, "
                          f"grid={region_seed_grid}, bias={growth_bias}, "
                          f"score_temp={score_temperature})")
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
                # Crossover choice. The default (HUX, supplied downstream by
                # _build_algorithm when crossover stays None) swaps individual pixels, so
                # it shatters the contiguous regions the sampler just built. 'region_swap' 
                # recombines whole components instead,the same operator region_evolve uses. 
                # Default keeps the historical behaviour so existing runs are unchanged.
                region_crossover = str(scenario_params.get('region_crossover', 'hux')).lower()
                if region_crossover == 'region_swap':
                    crossover = RegionSwapCrossover(
                        initial_conditions, problem.max_action_pixels, region_scores,
                        growth_bias=growth_bias, pixel_tolerance=pixel_tolerance,
                    )
                if verbose:
                    print(f"Using region-growing operators (seeds={region_seeds}, "
                          f"bias={growth_bias}, edits={region_edits}, "
                          f"crossover={region_crossover})")
            if not use_repair:
                repair = None
            else:
                # Region modes always use the contiguity-preserving repair, so every
                # min-patch-size level differs ONLY in the floor S. S=1 means "no size
                # floor" but still uses the same scored contiguous budget regrowth.
                repair = MinPatchSizeRepair(
                    initial_conditions, problem.max_action_pixels, min_patch_size,
                    scores=region_scores, pixel_tolerance=pixel_tolerance,
                    growth_bias=growth_bias,
                    # 0.0 on the region_grow path, so only region_evolve is affected.
                    score_temperature=score_temperature,
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
      "nsga3" (default) - NSGA-III - best for 3-objective space. Population size 
                          equals the number of reference directions (n_partitions=12, 
                          3 obj -> 45 ref dirs -> pop 45).
      "nsga2"           - classic NSGA-II with explicit pop_size and crowding
                          distance. Appropriate for 2-objective problems.

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
    Assemble the optimization results dict from a pymoo Result and optionally save
    it to disk. Returns the dict, ready for downstream use.
    """
    if verbose:
        convergence_reason = "hypervolume plateau" if callback.hv_callback.converged else "generation limit"
        final_gen = len(callback.hv_callback.hv_history)
        print(f"OK Optimization completed after {final_gen} generations ({convergence_reason})")
        if callback.hv_callback.hv_history:
            print(f"Final hypervolume: {callback.hv_callback.hv_history[-1]:.6f}")
        # The front is reported below, once it has been recomputed from the full
        # population (result.F alone undercounts it for NSGA-III - see there).

    initial_conditions_filtered = _filter_initial_conditions_for_return(initial_conditions)

    # ----- Full Population Data -----
    X_full = result.pop.get("X")
    F_full_norm = result.pop.get("F")
    F_full_raw = np.asarray([problem.evaluate_raw_objectives(xi) for xi in X_full], dtype=float)

    # Guard against the run-time (F) / post-hoc (F_raw) objective mismatch that
    # motivated calling RestorationProblem.evaluate_raw_objectives explicitly in
    # _evaluate: if F and F_raw were ever computed on different selections, their
    # ratio per objective would not be constant across the population.
    _valid = np.abs(F_full_norm) >= 1e-15
    for _oi, _oname in enumerate(problem.objective_names):
        _mask = _valid[:, _oi]
        if not np.any(_mask):
            continue
        _ratio = F_full_raw[_mask, _oi] / F_full_norm[_mask, _oi]
        _rmin, _rmax = float(np.min(_ratio)), float(np.max(_ratio))
        if not np.isclose(_rmin, _rmax, rtol=1e-6, atol=0.0):
            raise RuntimeError(
                f"Objective '{_oname}': raw/normalized ratio is not constant across "
                f"the population (min={_rmin!r}, max={_rmax!r}). This indicates F "
                "(run-time, normalized) and F_raw (post-hoc, via evaluate_raw_objectives) "
                "were computed on different decision vectors."
            )

    # ----- Non-dominated front over the WHOLE final population -----
    # NOT result.X. For NSGA-III, pymoo's result.X/algorithm.opt is
    #   pop[intersect(fronts[0], closest)]   (nsga3.py ReferenceDirectionSurvival._do)
    # i.e. the front-0 member CLOSEST to each occupied reference direction - one
    # representative per niche, deliberately curated for spread. Whenever a niche holds
    # more than one front-0 point (routine), that set is strictly smaller than the real
    # non-dominated front, so using it here would silently mislabel genuinely
    # non-dominated plans as dominated. NSGA-II's opt is plain pop[rank==0] and does not
    # have this property, but sorting explicitly is correct for both.
    _nd_idx = NonDominatedSorting().do(F_full_raw, only_non_dominated_front=True)
    _nd_idx = np.sort(np.asarray(_nd_idx, dtype=int))

    # result.X must be a SUBSET of that front: a curated representative is still a real
    # non-dominated point, and removing points from a set can only reveal more
    # non-dominated members, never fewer. A violation means F (run-time, normalized) and
    # F_raw (post-hoc) really were computed on different decision vectors - the failure
    # this guard exists to catch.
    _nd_X = {tuple(row) for row in X_full[_nd_idx]}
    _missing = [row for row in np.asarray(result.X) if tuple(row) not in _nd_X]
    if _missing:
        raise RuntimeError(
            f"{len(_missing)} of pymoo's {len(result.X)} reported optimal solutions are "
            f"NOT in the non-dominated front recomputed from F_raw ({len(_nd_idx)} "
            "members). This indicates F (run-time, normalized) and F_raw (post-hoc, via "
            "evaluate_raw_objectives) were computed on different decision vectors."
        )

    # ----- Non-dominated Data -----
    X_nd = X_full[_nd_idx]

    is_nondominated = np.zeros(len(X_full), dtype=bool)
    is_nondominated[_nd_idx] = True

    if verbose:
        _extra = ("" if len(X_nd) == len(result.X)
                  else f" (pymoo reported {len(result.X)} reference-direction representatives)")
        print(f"Found {len(X_nd)} Pareto-optimal solutions out of {len(X_full)} "
              f"evaluated solutions{_extra}")

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
                'feasible_share_history': callback.hv_callback.feasible_share_history,
                'n_action_min_history': callback.hv_callback.n_action_min_history,
                'n_action_max_history': callback.hv_callback.n_action_max_history,
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
                    print(f"OK X_history saved: {X_history.shape} -> {x_history_path}")
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
    print("FAIL Optimization failed - no solutions found")
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
                                     skip_diagnostics=False, hv_patience=15,
                                     hv_min_improvement=1e-6, use_repair=True,
                                     random_seed=None, use_patch_approach=False, patch_size=100,
                                     patch_constraint_type='pixel_count', pixel_tolerance=0.05,
                                     output_dir=str(OUTPUTS), save_snapshots=False,
                                     snapshot_generations=None,
                                     run_label="", run_config=None,
                                     n_partitions=8, warm_seeding=True,
                                     r_export_parent=None, mutation_prob_var=None,
                                     mutation_flip_count=None,
                                     capture_repair_diag=False, n_capture_gens=5,
                                     algorithm_type="nsga3", extra_seed_X=None):
    """
    Run the multi-objective restoration optimization for a single scenario.
    Returns the results dict, or None if the optimization produced no front.

    Only the non-obvious arguments are documented here; scenario_params keys are
    documented at their point of use (see RestorationProblem.__init__).

    extra_seed_X: optional (n_seed, n_var) int array of externally supplied genotypes injected into the initial population via WarmStartSampling. Independent of
        `warm_seeding` (that flag controls a different, automatic per-objective seeding
        mechanism). None (default) changes nothing.

    algorithm_type: "nsga3" (default, best for >=3 objectives) sizes the population
        from n_partitions and IGNORES pop_size; "nsga2" (2-objective fronts) uses
        pop_size directly. n_partitions=8 -> 45 Das-Dennis ref dirs -> pop 45.
    patch_constraint_type: 'pixel_count' (recommended) or 'patch_count'.
    mutation_prob_var / mutation_flip_count: per-variable bitflip probability, or the
        expected flip COUNT (converted to a probability; takes precedence). None keeps
        the historical 200 / n_var.
    capture_repair_diag / n_capture_gens: pixel mode + AdaptiveRepair only. Dumps paired
        pre/post-repair genotypes and raw objectives at n_capture_gens evenly-spaced
        generations to <output_dir>/repair_diagnostics/, for comparing genotype and phenotype diversity across repair. 
    snapshot_generations: with save_snapshots=True, an iterable of generation numbers
        to snapshot X for.
    """
    # --- 1. Initialize patch approach if not already done ---
    if use_patch_approach and not initial_conditions.get('patch_approach_enabled', False):
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
            patch_constraint_type=patch_constraint_type,
            pixel_tolerance=pixel_tolerance,
        )
    else:
        problem = RestorationProblem(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            pixel_tolerance=pixel_tolerance,
        )

    # --- 4. Print problem details ---
    if verbose:
        print(f"\nOptimization setup details:")
        print(f"  Max action pixels allowed: {problem.max_action_pixels}")
        print(f"  Number of objectives: {len(problem.objective_names)} ({', '.join(problem.objective_names)})")
        sample_obj_str = "  Baseline objectives (no restoration): "
        for obj_name in problem.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                val = np.sum(initial_conditions[obj_name])
                sample_obj_str += f"{obj_name}={val:.2e}, "
        print(sample_obj_str.rstrip(", "))
        if not skip_diagnostics:
            from .debug_utils import diagnose_optimization_setup
            from .logger_setup import setup_logger
            setup_logger()  # no-op if already configured by the entry-point script
            diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10)
            print("[OK] Optimisation setup verified.")

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

    # --- 5c. Externally supplied seed genotypes (opt-in; independent of warm_seeding above,
    # works in either mode). WarmStartSampling composes fine if both are active.
    if extra_seed_X is not None:
        sampling = WarmStartSampling(sampling, extra_seed_X)
        if verbose:
            print(f"  Seeding {len(extra_seed_X)} externally supplied genotype(s) into the "
                  "initial population.")

    # --- 6. Build HV reference point ---
    hv_warmup_samples = int(scenario_params.get("hv_warmup_samples", 40 if use_patch_approach else 200))
    if verbose:
        print(f"Building fixed HV ref point with {hv_warmup_samples} warm-up samples...")
    fixed_ref = build_fixed_ref_point(
        problem=problem, sampling=sampling,
        n_samples=hv_warmup_samples, margin=0.05, seed=42, verbose=verbose,
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
            snapshot_generations=snapshot_generations,
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
            print(f"ERROR during optimization: {e}")
        return None

# --- Main ---

def main(workspace_dir=".", scenario='all', objectives=None, n_samples_per_param=3, 
         pop_size=50, n_generations=100, save_results=True, verbose=True, random_seed=42,
         sample_fraction=None, sample_seed=42, ecosystem='all', lulc_path=None):
    """
    Load data and run one scenario, or every sampled scenario when scenario == "all".

    scenario: "all", or an integer index into sample_scenario_parameters().
    objectives: list of objective names (None = all available); see all_objectives in
        data_loader.py for the catalogue.
    sample_fraction / sample_seed: spatial subsampling of eligible pixels (None = all).
    """

    from .archive.run_scenarios import run_all_scenarios_optimization

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