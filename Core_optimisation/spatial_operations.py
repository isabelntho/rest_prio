"""
Spatial operations and custom operators for restoration optimization
==================================================================

This module contains:
1. Spatial algorithms (clustering, burden sharing)
2. Custom sampling operators for generating initial populations
3. Custom repair operators for constraint enforcement
4. Utility functions for spatial analysis
"""

import os
import heapq
import numpy as np
from scipy import ndimage
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.operators.mutation.bitflip import BitflipMutation
import rasterio
from scipy.ndimage import generic_filter

# =============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_sn_dens(raster_path, focal_classes, radius_m=300):
    with rasterio.open(raster_path) as src:
        lu = src.read(1)
        profile = src.profile
        res = src.res[0]

    focal = np.where(np.isin(lu, focal_classes), 1, 0).astype(float)
    focal[lu == src.nodata] = np.nan

    radius_px = int(radius_m / res)
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    footprint = (x**2 + y**2) <= radius_px**2

    def prop_focal(values):
        valid = ~np.isnan(values)
        return np.nansum(values) / np.sum(valid) if np.sum(valid) > 0 else np.nan

    sn_dens = generic_filter(
        focal,
        function=prop_focal,
        footprint=footprint,
        mode="constant",
        cval=np.nan
    )

    return sn_dens, profile

from scipy.ndimage import convolve

def compute_sn_dens_array(lu, nodata, res, focal_classes, radius_m=300):
    focal = np.where(np.isin(lu, focal_classes), 1.0, 0.0)
    valid = np.ones(lu.shape, dtype=np.float32)

    if nodata is not None:
        nodata_mask = (lu == nodata)
        focal = focal.astype(np.float32)
        focal[nodata_mask] = 0.0
        valid[nodata_mask] = 0.0
    else:
        focal = focal.astype(np.float32)

    radius_px = int(radius_m / res)
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    K = ((x**2 + y**2) <= radius_px**2).astype(np.float32)

    sum_focal = convolve(focal, K, mode="constant", cval=0.0)
    n_valid = convolve(valid, K, mode="constant", cval=0.0)

    dens = np.divide(sum_focal, n_valid, out=np.full_like(sum_focal, np.nan), where=(n_valid > 0))
    return dens


def compute_connectivity_gain_array(lu, nodata, res, focal_classes, radius_m=100):
    """
    Precompute the per-pixel connectivity gain from converting each pixel to semi-natural habitat.

    Connectivity at pixel i is defined as the *amount* of semi-natural (focal) habitat within
    a circular neighbourhood of radius R — i.e. the raw count of focal pixels, not the
    proportion.  Converting pixel j to semi-natural adds j to the focal set, so every pixel i
    within R of j gains +1 in connectivity.  Summing that gain over all such i gives the total
    connectivity gain attributable to converting j:

        connectivity_gain[j] = sum_{i: dist(i,j) <= R} focal[i]
                              = convolve(focal_mask, K)[j]

    This means j gains more when there is already substantial focal habitat in its
    neighbourhood — converting pixels that are embedded in the existing habitat matrix is
    rewarded over converting isolated pixels surrounded by agriculture.

    Note: the previous formulation used 1/n_valid(i) weighting, which normalised by kernel
    size rather than habitat presence, yielding near-uniform gain across most pixels.  This
    version makes gain depend on habitat amount, not kernel geometry.

    Args:
        lu: 2D numpy array of LULC codes
        nodata: LULC nodata value (excluded from focal count)
        res: Pixel resolution in metres
        focal_classes: List of LULC codes considered semi-natural
        radius_m: Neighbourhood radius in metres (default 100m = 1 pixel at 100m resolution)

    Returns:
        numpy.ndarray (float32): Per-pixel precomputed connectivity gain, same shape as lu.
            Nodata pixels have value 0.0.
    """
    focal = np.where(np.isin(lu, focal_classes), 1.0, 0.0).astype(np.float32)
    if nodata is not None:
        focal[lu == nodata] = 0.0

    radius_px = max(1, int(radius_m / res))
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    K = ((x ** 2 + y ** 2) <= radius_px ** 2).astype(np.float32)

    gain = convolve(focal, K, mode="constant", cval=0.0).astype(np.float32)

    # Zero out nodata pixels — they cannot be converted
    if nodata is not None:
        gain[lu == nodata] = 0.0

    return gain


# https://docs.scipy.org/doc/scipy/reference/ndimage.html may be faster

# fclass = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
#     52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]

# lu_path = "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif"
# =============================================================================
# BURDEN SHARING AND CLUSTERING ALGORITHMS
# =============================================================================


def build_region_assignments_cache(initial_conditions):
    """
    Build (or retrieve) the per-pixel region-assignment cache in ``initial_conditions``.

    Maps each restoration-eligible pixel (indexed by position within
    ``initial_conditions['eligible_indices']``) to its admin region ID.  The
    result is stored in-place as ``_region_assignments_cache`` and
    ``_non_empty_regions_cache`` so subsequent calls are free.

    Does nothing when ``admin_data`` is absent.
    """
    admin_data = initial_conditions.get('admin_data')
    if admin_data is None or '_region_assignments_cache' in initial_conditions:
        return

    from rasterio.features import rasterize

    shape = initial_conditions['shape']
    eligible_indices = initial_conditions['eligible_indices']
    transform = initial_conditions['transform']

    region_assignments = np.full(len(eligible_indices), -1, dtype=int)

    for i, region in enumerate(admin_data['unique_regions']):
        region_geom = admin_data['gdf'][admin_data['gdf'][admin_data['region_column']] == region]
        region_mask = rasterize(
            region_geom.geometry,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
        ).astype(bool)
        eligible_in_region = region_mask.flatten()[eligible_indices]
        region_assignments[eligible_in_region] = i

    non_empty_regions = []
    for region_id in range(admin_data['n_regions']):
        region_pixels = np.where(region_assignments == region_id)[0]
        if len(region_pixels) > 0:
            non_empty_regions.append((region_id, region_pixels))

    initial_conditions['_region_assignments_cache'] = region_assignments
    initial_conditions['_non_empty_regions_cache'] = non_empty_regions


def apply_burden_sharing(decision_vars, initial_conditions, seed=None, exact_count=None):
    """
    Apply burden sharing to ensure equal restoration across admin regions.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial conditions including admin data
        seed: Random seed for reproducibility (default: None)
        exact_count: If provided, enforces exact total count after burden sharing
        
    Returns:
        numpy.array: Modified decision variables with burden sharing applied
    """
    admin_data = initial_conditions.get('admin_data')
    
    if admin_data is None:
        result = decision_vars.copy()
        if exact_count is not None:
            result = _enforce_exact_pixel_count(result, exact_count)
        return result
    
    total_restore = exact_count if exact_count is not None else np.sum(decision_vars)
    
    if total_restore == 0:
        return decision_vars.copy()
    
    build_region_assignments_cache(initial_conditions)
    region_assignments = initial_conditions['_region_assignments_cache']
    non_empty_regions = initial_conditions['_non_empty_regions_cache']
    
    # Calculate target restoration per region (equal sharing)
    n_regions = admin_data['n_regions']
    restore_per_region = total_restore // n_regions
    extra_restores = total_restore % n_regions
    
    # Apply burden sharing
    new_decision_vars = np.zeros_like(decision_vars)
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Only process non-empty regions
    for region_id, region_pixels in non_empty_regions:
        # Determine restoration target for this region
        target = restore_per_region + (1 if region_id < extra_restores else 0)
        target = min(target, len(region_pixels))  # Can't restore more pixels than available
        
        if target > 0:
            # Select pixels to restore in this region - use permutation to avoid int32 overflow
            if target == len(region_pixels):
                selected_pixels = region_pixels
            else:
                # Use permutation instead of choice to avoid int32 overflow
                shuffled = np.random.permutation(len(region_pixels))[:target]
                selected_pixels = region_pixels[shuffled]
            new_decision_vars[selected_pixels] = 1
    
    # Enforce exact count if specified
    if exact_count is not None:
        new_decision_vars = _enforce_exact_pixel_count(new_decision_vars, exact_count)
    
    return new_decision_vars


def apply_spatial_clustering(decision_vars, initial_conditions, clustering_strength=0.0, exact_count=None):
    """
    Apply spatial clustering to decision variables to promote spatially coherent restoration.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial conditions including shape and eligible indices
        clustering_strength: Degree of clustering (0.0=no change, 1.0=maximum clustering)
        exact_count: If provided, enforces exact total count after clustering
        
    Returns:
        numpy.array: Modified decision variables with spatial clustering applied
    """
    if clustering_strength <= 0.0 or not np.any(decision_vars):
        result = decision_vars.copy()
        if exact_count is not None:
            result = _enforce_exact_pixel_count(result, exact_count)
        return result
    
    from scipy import ndimage
    
    shape = initial_conditions['shape']
    eligible_indices = initial_conditions['eligible_indices']
    target_count = np.sum(decision_vars)
    
    # Convert 1D to 2D
    restoration_2d = np.zeros(shape, dtype=bool)
    restore_indices = eligible_indices[decision_vars == 1]
    rows, cols = np.divmod(restore_indices, shape[1])
    restoration_2d[rows, cols] = True
    
    # Apply morphological closing with kernel size based on clustering strength
    kernel_size = int(1 + clustering_strength * 3)
    kernel = ndimage.generate_binary_structure(2, 1)  # Simple cross kernel
    for _ in range(kernel_size):
        kernel = ndimage.binary_dilation(kernel, structure=ndimage.generate_binary_structure(2, 1))
    
    # Close gaps and smooth with Gaussian
    clustered_2d = ndimage.binary_closing(restoration_2d, structure=kernel)
    
    # Convert to probability map and smooth
    prob_map = clustered_2d.astype(float)
    sigma = clustering_strength * 1.5
    prob_map = ndimage.gaussian_filter(prob_map, sigma=sigma)
    
    # Select top probabilities from eligible pixels only
    eligible_probs = prob_map.flatten()[eligible_indices]
    if target_count > 0 and len(eligible_indices) >= target_count:
        top_indices = np.argpartition(eligible_probs, -target_count)[-target_count:]
        new_decision_vars = np.zeros_like(decision_vars)
        new_decision_vars[top_indices] = 1
    else:
        new_decision_vars = decision_vars.copy()
    
    # Enforce exact count if specified
    if exact_count is not None:
        new_decision_vars = _enforce_exact_pixel_count(new_decision_vars, exact_count)
    
    return new_decision_vars


def _enforce_exact_pixel_count(decision_vars, target_count, seed=None):
    """
    Helper function to enforce exact pixel count constraint.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        target_count: Exact number of pixels to restore
        seed: Random seed for reproducibility
        
    Returns:
        numpy.array: Modified decision variables with exact count
    """
    if seed is not None:
        np.random.seed(seed)
    
    current_count = np.sum(decision_vars)
    result = decision_vars.copy()
    
    if current_count == target_count:
        return result
    
    if current_count < target_count:
        # Need to add pixels
        available_indices = np.where(result == 0)[0]
        n_to_add = target_count - current_count
        if len(available_indices) >= n_to_add:
            selected = np.random.choice(available_indices, n_to_add, replace=False)
            result[selected] = 1
    else:
        # Need to remove pixels
        active_indices = np.where(result == 1)[0]
        n_to_remove = current_count - target_count
        if len(active_indices) >= n_to_remove:
            selected = np.random.choice(active_indices, n_to_remove, replace=False)
            result[selected] = 0
    
    return result

# =============================================================================
# CUSTOM SAMPLING OPERATOR
# =============================================================================

class AdaptiveSampling(Sampling):
    """
    Unified sampling that handles all restoration patterns based on scenario parameters.
    Consolidates CustomBinaryRandomSampling, ClusteredSampling, BurdenSharingSampling, and CombinedSampling.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels, scenario_params=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
        
        # Extract parameters from scenario_params
        if scenario_params is None:
            scenario_params = {}
        self.burden_sharing = scenario_params.get('burden_sharing', 'no') == 'yes'
        self.clustering_strength = scenario_params.get('spatial_clustering', 0.0)
        self.enable_conversions = (
            'landscape_anomaly' in initial_conditions
            or 'connectivity_gain_1d' in initial_conditions
        )
    
    def _do(self, problem, n_samples, **kwargs):
        n_decision_vars = problem.n_var  # n_restoration_pixels + n_conversion_pixels
        n_restoration_pixels = problem.n_restoration_pixels
        n_conversion_pixels = problem.n_conversion_pixels
        X = np.zeros((n_samples, n_decision_vars), dtype=int)
        
        #print(f"DEBUG AdaptiveSampling: Creating {n_samples} samples")
        #print(f"  burden_sharing={self.burden_sharing}, clustering_strength={self.clustering_strength}")
        
        for i in range(n_samples):
            x = np.zeros(n_decision_vars, dtype=int)
            
            # Distribute actions between restoration and conversion
            n_total_actions = self.max_restored_pixels
            if n_total_actions > 0:
                if self.enable_conversions:
                    # Split actions randomly between restoration and conversion
                    max_restore = min(n_total_actions, n_restoration_pixels)
                    max_convert = min(n_total_actions, n_conversion_pixels)
                    
                    n_restore = np.random.randint(0, max_restore + 1)
                    n_convert = min(n_total_actions - n_restore, max_convert)
                else:
                    # Only restoration actions
                    n_restore = min(n_total_actions, n_restoration_pixels)
                    n_convert = 0
                
                # Set restoration actions (first part of decision vector)
                if n_restore > 0:
                    restore_indices = np.random.permutation(n_restoration_pixels)[:n_restore]
                    x[restore_indices] = 1
                    
                    # Apply burden sharing if enabled
                    if self.burden_sharing:
                        seed = np.random.randint(0, 2**31 - 1)
                        x[:n_restoration_pixels] = apply_burden_sharing(
                            x[:n_restoration_pixels], self.initial_conditions, 
                            seed=seed, exact_count=n_restore
                        )
                    
                    # Apply clustering if enabled
                    if self.clustering_strength > 0:
                        x[:n_restoration_pixels] = apply_spatial_clustering(
                            x[:n_restoration_pixels], self.initial_conditions, 
                            self.clustering_strength, exact_count=n_restore
                        )
                
                # Set conversion actions (second part of decision vector)
                if n_convert > 0:
                    convert_indices = np.random.permutation(n_conversion_pixels)[:n_convert]
                    x[n_restoration_pixels + convert_indices] = 1
            
            X[i] = x
            
            # Debug first few samples
            if i < 3:
                restore_count = np.sum(x[:n_restoration_pixels])
                convert_count = np.sum(x[n_restoration_pixels:])
                #print(f"  Sample {i}: {restore_count} restore + {convert_count} convert = {restore_count + convert_count} total")
        
        #print(f"DEBUG AdaptiveSampling: Completed {n_samples} samples")
        return X


class WarmStartSampling(Sampling):
    """Decorate a base sampler by injecting precomputed warm-seed individuals.

    Strategy-agnostic: wraps any Sampling operator and places the fixed seed
    genotypes in the LAST rows of the initial population, filling the remaining
    rows via the base sampler. Mirrors the patch path, which reserves the tail
    of the population for objective-extreme seeds (see PatchAwareSampling).

    seed_X : int array (n_seed, n_var) or None. When None, or when the requested
    n_samples is not larger than n_seed, the base sampler is used unchanged so at
    least one stochastic individual always remains.
    """

    def __init__(self, base_sampling, seed_X):
        super().__init__()
        self.base_sampling = base_sampling
        self.seed_X = (
            np.asarray(seed_X, dtype=int) if seed_X is not None and len(seed_X) > 0
            else None
        )

    def _do(self, problem, n_samples, **kwargs):
        n_seed = 0 if self.seed_X is None else self.seed_X.shape[0]
        if n_seed == 0 or n_samples <= n_seed:
            return self.base_sampling._do(problem, n_samples, **kwargs)
        X = self.base_sampling._do(problem, n_samples - n_seed, **kwargs)
        return np.vstack([np.asarray(X, dtype=int), self.seed_X])


# =============================================================================
# CUSTOM REPAIR OPERATOR
# =============================================================================

class AdaptiveRepair(Repair):
    """
    Unified repair that handles all constraint enforcement strategies.
    Consolidates previous ClusteringRepair, BurdenSharingRepair, CombinedRepair, ScoreCountRepair, and ExactCountRepair.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels, scenario_params=None, scores=None,
                 capture_diag=False, n_generations=None, n_capture=5, diag_dir=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
        self.scores = scores
        self.rng = np.random.RandomState(None)  # Can be seeded later if needed

        # Extract parameters from scenario_params
        if scenario_params is None:
            scenario_params = {}
        self.burden_sharing = scenario_params.get('burden_sharing', 'no') == 'yes'
        self.clustering_strength = scenario_params.get('spatial_clustering', 0.0)
        self.adjacency_beta = float(scenario_params.get('adjacency_beta', 0.25))  # Tunable adjacency weight

        # Initialize repair logging
        self.call_log = []  # Store repair events: {'generation': int, 'type': str, 'individuals_repaired': int}
        self.total_calls = 0
        self.repair_log = []  # Per-generation bit-diff diagnostics

        # Pre/post-repair diversity capture (opt-in diagnostic).
        # At a few evenly-spaced generations, dump the paired pre-repair and
        # post-repair genotype matrices plus their raw objectives to .npz so
        # genotype (pairwise Hamming) and phenotype (objective spread) diversity
        # can be compared before vs after repair, offline.
        self.capture_diag = bool(capture_diag)
        self.diag_dir = diag_dir
        if self.capture_diag and n_generations:
            gens = np.linspace(1, int(n_generations), int(n_capture)).round().astype(int)
            self._capture_gens = set(int(g) for g in np.unique(gens))
            if self.diag_dir is not None:
                os.makedirs(self.diag_dir, exist_ok=True)
        else:
            self._capture_gens = set()
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        X_in = X.copy()  # snapshot before any repair, for bit-diff

        score_repairs = 0
        random_repairs = 0
        individuals_repaired = 0
        
        # Track repair call
        self.total_calls += 1
        generation = kwargs.get('generation', self.total_calls)
        
        for i in range(len(X)):
            try:
                x = X[i].copy()
                n_pixels = problem.n_pixels
                
                # Allow some flexibility in total count to maintain diversity
                # Only repair solutions that are significantly over/under target
                current_total = np.sum(x[:n_pixels]) + np.sum(x[n_pixels:])
                target = self.max_restored_pixels
                
                # Only repair if count is >10% off target (preserve diversity for small differences)
                tolerance = max(target * 0.1, 5)  # 10% tolerance or at least 5 pixels
                
                if abs(current_total - target) > tolerance:
                    individuals_repaired += 1
                    # Use score-based repair with adjacency-aware dynamic scoring
                    if self.scores is not None:
                        x = self._enforce_count_with_scores(x, problem)
                        score_repairs += 1
                    else:
                        x = self._enforce_count_random(x, problem)  
                        random_repairs += 1
                else:
                    # Don't repair - let solution keep its natural diversity
                    pass
                
                # Apply spatial constraints if enabled and we have restoration pixels
                if np.sum(x[:n_pixels]) > 0:
                    if self.burden_sharing:
                        seed = hash(tuple(x)) % (2**31 - 1)
                        x[:n_pixels] = apply_burden_sharing(
                            x[:n_pixels], self.initial_conditions, 
                            seed=seed, exact_count=None  # Remove exact count to preserve diversity
                        )
                    
                    if self.clustering_strength > 0:
                        x[:n_pixels] = apply_spatial_clustering(
                            x[:n_pixels], self.initial_conditions, 
                            self.clustering_strength, exact_count=None  # Remove exact count to preserve diversity
                        )
                        
                    # No final enforcement - let solutions maintain natural diversity
                
                X_repaired[i] = x
                
            except Exception as e:
                print(f"ERROR in AdaptiveRepair._do() for individual {i}: {e}")
                import traceback
                traceback.print_exc()
                # Return original individual on error
                X_repaired[i] = X[i]
        
        # Log repair event
        repair_type = 'score_based' if score_repairs > 0 else 'random' if random_repairs > 0 else 'none'
        if individuals_repaired > 0:
            self.call_log.append({
                'generation': generation,
                'type': repair_type,
                'individuals_repaired': individuals_repaired,
                'score_repairs': score_repairs,
                'random_repairs': random_repairs
            })

        # Record bit-diff diagnostics
        diffs = np.sum(X_in != X_repaired, axis=1)  # (pop_size,)
        self.repair_log.append({
            'generation': generation,
            'mean_bits_changed': float(np.mean(diffs)),
            'std_bits_changed': float(np.std(diffs)),
            'max_bits_changed': int(np.max(diffs)),
            'fraction_repaired': float(individuals_repaired / max(len(X), 1)),
        })

        # Pre/post-repair diversity snapshot at selected generations.
        # X_in is the population after crossover+mutation but before repair;
        # X_repaired is the post-repair population. Raw objectives are computed
        # for both (evaluate_raw_objectives is side-effect-free). Only a handful
        # of generations pay this extra pop_size evaluations.
        if self.capture_diag and generation in self._capture_gens and self.diag_dir is not None:
            try:
                F_pre = np.asarray([problem.evaluate_raw_objectives(xi) for xi in X_in], dtype=float)
                F_post = np.asarray([problem.evaluate_raw_objectives(xi) for xi in X_repaired], dtype=float)
                out_path = os.path.join(self.diag_dir, f"repair_diag_gen{generation:04d}.npz")
                np.savez_compressed(
                    out_path,
                    X_pre=X_in.astype(np.int8),
                    X_post=X_repaired.astype(np.int8),
                    F_pre=F_pre,
                    F_post=F_post,
                    generation=generation,
                    n_pixels=problem.n_pixels,
                )
            except Exception as e:
                print(f"WARNING: repair diversity capture failed at gen {generation}: {e}")

        return X_repaired
    
    def _compute_adjacency_scores(self, x_restore):
        """
        Compute 4-neighbourhood adjacency scores for all pixels in the restoration eligible set.
        Returns the count of selected neighbours for each pixel.
        
        Args:
            x_restore: Binary array indicating selected restoration pixels
            
        Returns:
            adjacency_scores: Array with adjacency count (0-4) for each pixel
        """
        shape = self.initial_conditions["shape"]
        n_pixels = len(x_restore)
        adjacency_scores = np.zeros(n_pixels, dtype=np.float64)
        
        # Get indices of currently selected pixels
        selected_indices = np.where(x_restore == 1)[0]
        
        if len(selected_indices) == 0:
            return adjacency_scores
        
        # Convert selected indices to 2D coordinates
        selected_rows, selected_cols = np.divmod(selected_indices, shape[1])
        
        # Create a 2D map of selected pixels for fast lookup
        selected_map = np.zeros(shape, dtype=bool)
        selected_map[selected_rows, selected_cols] = True
        
        # Get all restoration-eligible indices
        elig_indices = self.initial_conditions.get("restoration_eligible_indices", np.arange(n_pixels))
        rows, cols = np.divmod(elig_indices, shape[1])
        
        # Compute 4-neighbourhood adjacency for each pixel
        # 4-neighbourhood: up, down, left, right
        for i in range(n_pixels):
            r, c = rows[i], cols[i]
            adj_count = 0
            
            # Check 4 neighbours
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    if selected_map[nr, nc]:
                        adj_count += 1
            
            adjacency_scores[i] = float(adj_count)
        
        return adjacency_scores
    
    def _enforce_count_with_scores(self, x, problem):
        """
        Enforce exact count using dynamic scores that combine base scores with adjacency.
        
        Dynamic score = base_score + beta * adjacency_score
        where adjacency_score = number of selected neighbours in 4-neighbourhood.
        """
        try:
            n_pixels = problem.n_pixels
            x_restore = x[:n_pixels].copy()
            x_convert = x[n_pixels:].copy()
            
            # Count total actions (restore + convert)
            cur_restore = int(np.sum(x_restore))
            cur_convert = int(np.sum(x_convert))
            cur_total = cur_restore + cur_convert
            k = self.max_restored_pixels

            if cur_total < k:
                need = k - cur_total
                # Compute dynamic scores based on current state and adjacency
                adjacency_scores = self._compute_adjacency_scores(x_restore)
                dynamic_scores = self.scores + self.adjacency_beta * adjacency_scores
                
                # Prioritize restoration actions (since convert is forced to 0)
                zeros_restore = np.where(x_restore == 0)[0]
                if zeros_restore.size > 0 and need > 0:
                    # Add best available restoration zeros by dynamic score
                    add_indices = zeros_restore[np.argsort(-dynamic_scores[zeros_restore])][:need]
                    x_restore[add_indices] = 1

            elif cur_total > k:
                drop = cur_total - k
                # Compute dynamic scores based on current state and adjacency
                adjacency_scores = self._compute_adjacency_scores(x_restore)
                dynamic_scores = self.scores + self.adjacency_beta * adjacency_scores
                
                # Remove from restoration actions first (since convert should be 0)
                ones_restore = np.where(x_restore == 1)[0]
                if ones_restore.size > 0 and drop > 0:
                    # Remove worst available restoration ones by dynamic score
                    rem_indices = ones_restore[np.argsort(dynamic_scores[ones_restore])][:drop]
                    x_restore[rem_indices] = 0

            # Update both restoration and conversion parts of the decision vector
            x[:n_pixels] = x_restore
            x[n_pixels:] = x_convert
            return x
        except Exception as e:
            print(f"ERROR in _enforce_count_with_scores: {e}")
            print(f"  x.shape: {x.shape}, n_pixels: {problem.n_pixels}, scores.shape: {self.scores.shape if self.scores is not None else None}")
            import traceback
            traceback.print_exc()
            return x  # Return original on error
    
    def _enforce_count_random(self, x, problem):
        """Enforce exact count using random selection for which pixels to add/remove."""
        n_pixels = problem.n_pixels
        x_restore = x[:n_pixels].copy()
        x_convert = x[n_pixels:].copy()
        
        # Count total actions (restore + convert)
        current_restore = np.sum(x_restore)
        current_convert = np.sum(x_convert)
        current_total = current_restore + current_convert
        
        if current_total != self.max_restored_pixels:
            if current_total < self.max_restored_pixels:
                # Need to add actions - prioritize restoration for now
                available_restore = np.where(x_restore == 0)[0]
                n_to_add = self.max_restored_pixels - current_total
                if len(available_restore) >= n_to_add:
                    selected = self.rng.choice(available_restore, n_to_add, replace=False)
                    x_restore[selected] = 1
                else:
                    # Add all available restoration pixels
                    if len(available_restore) > 0:
                        x_restore[available_restore] = 1
            else:
                # Need to remove actions - prioritize removing restoration for now  
                active_restore = np.where(x_restore == 1)[0]
                n_to_remove = current_total - self.max_restored_pixels
                if len(active_restore) >= n_to_remove:
                    selected = self.rng.choice(active_restore, n_to_remove, replace=False)
                    x_restore[selected] = 0
                else:
                    # Remove all restoration pixels
                    if len(active_restore) > 0:
                        x_restore[active_restore] = 0
        
        # Update both restoration and conversion parts of the decision vector
        x[:n_pixels] = x_restore
        x[n_pixels:] = x_convert
        return x


# =============================================================================
# INSTRUMENTED MUTATION OPERATOR
# =============================================================================

class InstrumentedBitflipMutation(BitflipMutation):
    """
    BitflipMutation wrapper that records how many bits it flips each generation.

    Attributes
    ----------
    flip_log : list of dict
        One entry per generation::

            {
              'generation': int,
              'mean_raw_flips': float,   # avg bits flipped before repair
              'std_raw_flips': float,
              'total_raw_flips': int,
            }

    Usage
    -----
    mutation = InstrumentedBitflipMutation(prob=1/n_var)
    # After optimization:
    flip_log = mutation.flip_log
    """

    def __init__(self, prob=0.1, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.flip_log = []
        self._current_gen = 0

    def _do(self, problem, X, **kwargs):
        X_before = X.copy()

        # NOTE: pymoo's BitflipMutation._do flips with `Xp[flip] = ~X[flip]`, which is
        # bitwise NOT. That only behaves as a logical flip on boolean arrays. Our
        # decision vector is int 0/1 (type_var=int, xl=0, xu=1), so `~0 = -1` and
        # `~1 = -2`, silently corrupting the genotype and making mutation directional
        # (a flipped 0 becomes -1, still "unselected", so it is a no-op). We do the
        # flip explicitly instead. `1 - (X == 1)` always yields {0,1} and self-heals
        # any stray non-binary genes that may already be present.
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        X_after = X.copy()
        flip = np.random.random(X.shape) < prob_var
        X_after[flip] = (1 - (X_after[flip] == 1)).astype(X_after.dtype)

        flips_per_ind = np.sum(X_before != X_after, axis=1)  # (pop_size,)
        # Use _current_gen + 1 so the label matches the 1-based generation counter
        # used by the repair operators.  ProgressCallback own sync overwrites
        # _current_gen before each call, so we must NOT increment here.
        self.flip_log.append({
            'generation': self._current_gen + 1,
            'mean_raw_flips': float(np.mean(flips_per_ind)),
            'std_raw_flips': float(np.std(flips_per_ind)),
            'total_raw_flips': int(np.sum(flips_per_ind)),
        })
        return X_after


# =============================================================================
# REGION-GROWING OPERATORS (contiguous, arbitrary-shape restoration regions)
# =============================================================================
#
# Motivation: the pixel-level sampler/mutation only reach spatially SCATTERED
# plans. On that manifold cost is near-constant (a ~22k/437k random subset
# averages to ~mean cost) and spatial_clustering stays frozen, so the 3-objective
# problem collapses to ~single-objective and HV freezes. Concentrating the budget
# in contiguous regions instead swings total cost by ~14x and gives clustering
# real range. These operators build and preserve contiguous regions of ANY shape
# (not square patches), so the search can reach that part of the space.
#
# Shared machinery: a precomputed 4-neighbour table over the restoration-eligible
# pixels (local index -> up to 4 eligible neighbour local indices, -1 for none),
# built once from restoration_eligible_indices + shape. O(1) neighbour lookup, no
# per-pixel Python scans.


def build_restoration_neighbor_table(initial_conditions):
    """Precompute the 4-neighbourhood over restoration-eligible pixels.

    Returns (nbr, rows, cols) where:
      nbr  : int32 array (n_rest, 4) of eligible-neighbour LOCAL indices (-1 if the
             orthogonal neighbour is off-grid or not restoration-eligible),
      rows : int32 array (n_rest,) raster row of each local pixel,
      cols : int32 array (n_rest,) raster col of each local pixel.

    Cached on initial_conditions under '_restoration_nbr_table' so repeated
    operator builds (e.g. across a sweep) pay for it once.
    """
    cache = initial_conditions.get('_restoration_nbr_table')
    if cache is not None:
        return cache

    shape = initial_conditions['shape']
    H, W = int(shape[0]), int(shape[1])
    elig_idx = np.asarray(initial_conditions['restoration_eligible_indices'], dtype=np.int64)
    n_rest = elig_idx.size
    rows = (elig_idx // W).astype(np.int32)
    cols = (elig_idx % W).astype(np.int32)

    # Grid of local indices (-1 where not restoration-eligible) for O(1) lookup.
    elig_grid = np.full((H, W), -1, dtype=np.int32)
    elig_grid[rows, cols] = np.arange(n_rest, dtype=np.int32)

    nbr = np.full((n_rest, 4), -1, dtype=np.int32)
    for d, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        rr = rows.astype(np.int64) + dr
        cc = cols.astype(np.int64) + dc
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        nbr[valid, d] = elig_grid[rr[valid], cc[valid]]

    cache = (nbr, rows, cols)
    initial_conditions['_restoration_nbr_table'] = cache
    return cache


def grow_region_plan(nbr, n_rest, target_k, n_seeds, scores, mode, rng):
    """Build one contiguous-region restoration plan (boolean array of length n_rest).

    Grows regions from n_seeds seeds by a score- or random-ordered frontier walk
    over the 4-neighbour table until target_k pixels are selected. mode='scored'
    grows toward high-score pixels (heap keyed by -score); mode='neutral' grows in
    random order (heap keyed by a random draw), giving arbitrary-shaped but
    location-neutral regions. Re-seeds from a random unselected pixel if a region
    saturates before the budget is met.
    """
    selected = np.zeros(n_rest, dtype=bool)
    in_heap = np.zeros(n_rest, dtype=bool)
    heap = []  # (key, local_idx); smaller key popped first

    def _key(idx):
        return -float(scores[idx]) if mode == 'scored' else float(rng.random())

    def _push_neighbours(idx):
        for d in range(4):
            nb = nbr[idx, d]
            if nb >= 0 and not selected[nb] and not in_heap[nb]:
                heapq.heappush(heap, (_key(nb), int(nb)))
                in_heap[nb] = True

    def _add_seed():
        # Pick a seed among currently-unselected pixels.
        if mode == 'scored':
            # Weighted by score so seeds start in good areas; cheap top-biased draw.
            avail = np.where(~selected)[0]
            if avail.size == 0:
                return False
            w = scores[avail].astype(np.float64)
            w = w - w.min()
            s = w.sum()
            p = (w / s) if s > 0 else None
            seed = int(rng.choice(avail, p=p))
        else:
            seed = int(rng.integers(n_rest))
            if selected[seed]:
                avail = np.where(~selected)[0]
                if avail.size == 0:
                    return False
                seed = int(rng.choice(avail))
        selected[seed] = True
        _push_neighbours(seed)
        return True

    count = 0
    for _ in range(max(1, int(n_seeds))):
        if count >= target_k:
            break
        if _add_seed():
            count += 1

    while count < target_k:
        if not heap:
            if not _add_seed():
                break
            count += 1
            continue
        _, idx = heapq.heappop(heap)
        in_heap[idx] = False
        if selected[idx]:
            continue
        selected[idx] = True
        count += 1
        _push_neighbours(idx)

    return selected


class RegionGrowingSampling(Sampling):
    """Initial population of contiguous, arbitrary-shape restoration regions.

    Each individual is grown from `region_seeds` seeds to exactly the pixel budget
    (problem.max_action_pixels). growth_bias='scored' steers toward high-score
    (and, via the score vector passed in, low-cost) areas; 'neutral' grows in
    random order. The conversion block (if any) is left at zero, matching the
    scattered pixel-mode sampler for the restoration-only objective set.
    """

    def __init__(self, initial_conditions, max_restored_pixels, scores,
                 region_seeds=25, growth_bias='scored',
                 region_seeds_min=None, random_share=0.0):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = int(max_restored_pixels)
        self.scores = np.asarray(scores, dtype=np.float64) if scores is not None else None
        self.region_seeds = int(region_seeds)
        # Per-individual seed-count variety: when region_seeds_min < region_seeds,
        # each individual draws its own S in [min, max], so the population spans
        # single-blob to many-cluster plans (the number-of-regions DOF). Default
        # (min == max) reproduces a fixed S.
        self.region_seeds_min = int(region_seeds_min) if region_seeds_min is not None else int(region_seeds)
        # Fraction of individuals grown with NEUTRAL (uniform) placement instead of
        # scored, to inject diversity along the cost/value axis and avoid every
        # individual converging on the same high-value regions. 0 = all scored.
        self.random_share = float(np.clip(random_share, 0.0, 1.0))
        self.growth_bias = str(growth_bias).lower()
        self.nbr, self.rows, self.cols = build_restoration_neighbor_table(initial_conditions)

    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        n_rest = problem.n_restoration_pixels
        k = min(self.max_restored_pixels, n_rest)
        scores = self.scores if self.scores is not None else np.zeros(n_rest)
        base_scored = (self.growth_bias == 'scored' and self.scores is not None)
        s_lo = max(1, min(self.region_seeds_min, self.region_seeds))
        s_hi = max(s_lo, self.region_seeds)
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

        X = np.zeros((n_samples, n_var), dtype=int)
        for i in range(n_samples):
            s_i = int(rng.integers(s_lo, s_hi + 1))
            # Per-individual mode: some individuals grown neutrally for diversity.
            mode_i = 'scored' if (base_scored and rng.random() >= self.random_share) else 'neutral'
            sel = grow_region_plan(self.nbr, n_rest, k, s_i, scores, mode_i, rng)
            X[i, :n_rest] = sel.astype(int)
        return X


class RegionGrowingMutation(Mutation):
    """Contiguity-preserving mutation: grow at the frontier, peel at the boundary.

    Instead of scattering isolated bit flips (which erode clusters), each mutated
    individual GROWS by adding ~m unselected frontier pixels (selected-adjacent)
    and PEELS ~m boundary pixels (selected pixels touching an unselected/ off-grid
    neighbour), keeping the count near the budget. 'scored' adds high-score / peels
    low-score; 'neutral' does both at random. Operates on the restoration block
    only; the conversion block is passed through unchanged.
    """

    def __init__(self, initial_conditions, max_restored_pixels, scores,
                 n_edits=100, growth_bias='scored', pixel_tolerance=0.05, prob=1.0):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = int(max_restored_pixels)
        self.scores = np.asarray(scores, dtype=np.float64) if scores is not None else None
        self.n_edits = int(n_edits)
        self.growth_bias = str(growth_bias).lower()
        self.pixel_tolerance = float(pixel_tolerance)
        self.prob = float(prob)
        self.nbr, self.rows, self.cols = build_restoration_neighbor_table(initial_conditions)
        self.flip_log = []
        self._current_gen = 0

    def _boundary_and_frontier(self, sel):
        """Return (frontier_unsel_idx, boundary_sel_idx) for a boolean selection.

        frontier_unsel: unselected eligible pixels with >=1 selected neighbour.
        boundary_sel:   selected pixels with >=1 unselected or off-grid neighbour.
        """
        nbr = self.nbr
        valid = nbr >= 0                       # (n_rest, 4)
        nb_clip = np.where(valid, nbr, 0)
        nb_sel = valid & sel[nb_clip]          # neighbour is selected
        any_nb_sel = nb_sel.any(axis=1)
        nb_unsel = (valid & ~sel[nb_clip]) | (~valid)  # neighbour unselected or off-grid
        any_nb_unsel = nb_unsel.any(axis=1)
        frontier_unsel = np.where((~sel) & any_nb_sel)[0]
        boundary_sel = np.where(sel & any_nb_unsel)[0]
        return frontier_unsel, boundary_sel

    def _pick(self, pool, m, prefer_high, rng):
        """Pick up to m indices from pool: by score (prefer_high/low) or random."""
        if pool.size == 0 or m <= 0:
            return pool[:0]
        m = int(min(m, pool.size))
        if self.growth_bias == 'scored' and self.scores is not None:
            s = self.scores[pool]
            order = np.argsort(-s if prefer_high else s)
            return pool[order[:m]]
        return pool[rng.choice(pool.size, size=m, replace=False)]

    def _do(self, problem, X, **kwargs):
        n_rest = problem.n_restoration_pixels
        k = self.max_restored_pixels
        lo = int(k * (1 - self.pixel_tolerance))
        hi = int(k * (1 + self.pixel_tolerance))
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        Xp = X.copy()
        edits_per_ind = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            if rng.random() > self.prob:
                continue
            sel = Xp[i, :n_rest].astype(bool)
            cur = int(sel.sum())
            if cur == 0:
                continue
            frontier, boundary = self._boundary_and_frontier(sel)
            # Nudge the net count toward the budget while staying contiguity-aware.
            grow_m = self.n_edits + max(0, lo - cur)
            peel_m = self.n_edits + max(0, cur - hi)
            add = self._pick(frontier, grow_m, prefer_high=True, rng=rng)
            rem = self._pick(boundary, peel_m, prefer_high=False, rng=rng)
            if add.size:
                sel[add] = True
            if rem.size:
                sel[rem] = False
            # Safety: never let a mutation push outside the tolerance band.
            cur2 = int(sel.sum())
            if cur2 > hi:
                on = np.where(sel)[0]
                drop = self._pick(on, cur2 - hi, prefer_high=False, rng=rng)
                sel[drop] = False
            elif cur2 < lo:
                off = np.where(~sel)[0]
                addmore = self._pick(off, lo - cur2, prefer_high=True, rng=rng)
                sel[addmore] = True
            Xp[i, :n_rest] = sel.astype(Xp.dtype)
            edits_per_ind[i] = int(np.sum(X[i, :n_rest] != Xp[i, :n_rest]))

        self.flip_log.append({
            'generation': self._current_gen + 1,
            'mean_raw_flips': float(np.mean(edits_per_ind)),
            'std_raw_flips': float(np.std(edits_per_ind)),
            'total_raw_flips': int(np.sum(edits_per_ind)),
        })
        return Xp


# =============================================================================
# REGION-EVOLVE OPERATORS (spatially-exploring search inside NSGA-III)
# =============================================================================
#
# Goal: let the search EXPLORE the landscape (not collapse onto one basin) while
# keeping region size free. A plan is the pixel selection; its "regions" are the
# connected components of that selection (scipy.ndimage.label, 4-connectivity).
# Operators act at the region level:
#   - SpatialCoverageSampling seeds regions spread across the whole map,
#   - RegionEvolveMutation relocates / grows / shrinks / spawns / deletes regions,
#   - RegionSwapCrossover recombines WHOLE regions between parents (not HUX, which
#     shatters contiguity).
# All reuse build_restoration_neighbor_table (O(1) 4-neighbour lookup).


def grow_regions_from_seeds(nbr, n_rest, target_k, seeds, scores, mode, rng,
                            avoid=None, base=None):
    """Grow a contiguous selection of target_k pixels from the given seed indices.

    Adds only currently-unselected, non-avoided eligible pixels via a frontier walk
    (heap keyed by -score when mode='scored', else random). Returns a boolean array
    of the NEWLY grown pixels (length n_rest). `avoid` (bool array) marks pixels that
    must not be added (e.g. pixels used by regions being kept). `base` optionally
    marks already-selected pixels so the frontier does not re-add them.
    """
    grown = np.zeros(n_rest, dtype=bool)
    in_heap = np.zeros(n_rest, dtype=bool)
    blocked = np.zeros(n_rest, dtype=bool)
    if avoid is not None:
        blocked |= avoid
    if base is not None:
        blocked |= base
    heap = []

    def key(i):
        return -float(scores[i]) if mode == 'scored' else float(rng.random())

    def push_nbrs(idx):
        row = nbr[idx]
        for d in range(4):
            nb = row[d]
            if nb >= 0 and not grown[nb] and not in_heap[nb] and not blocked[nb]:
                heapq.heappush(heap, (key(int(nb)), int(nb)))
                in_heap[nb] = True

    count = 0
    for s in seeds:
        s = int(s)
        if count >= target_k:
            break
        if grown[s]:
            continue
        if blocked[s]:
            # already-selected seed (base): don't re-add, but seed the frontier from it
            if base is not None and base[s]:
                push_nbrs(s)
            continue
        grown[s] = True
        count += 1
        push_nbrs(s)

    while count < target_k:
        if not heap:
            allowed = np.where(~grown & ~blocked)[0]
            if allowed.size == 0:
                break
            s = int(rng.choice(allowed))
            grown[s] = True
            count += 1
            push_nbrs(s)
            continue
        _, idx = heapq.heappop(heap)
        in_heap[idx] = False
        if grown[idx] or blocked[idx]:
            continue
        grown[idx] = True
        count += 1
        push_nbrs(idx)

    return grown


def _label_components(sel, shape, rows, cols):
    """Return list of local-index arrays, one per connected component (4-conn)."""
    sidx = np.where(sel)[0]
    if sidx.size == 0:
        return []
    m2 = np.zeros(shape, dtype=bool)
    m2[rows[sidx], cols[sidx]] = True
    lab, nc = ndimage.label(m2)
    ids = lab[rows[sidx], cols[sidx]]
    return [sidx[ids == c] for c in range(1, nc + 1)]


def enforce_budget_contiguous(sel, nbr, shape, rows, cols, k, tol, scores, mode, rng):
    """Bring a selection to within [k*(1-tol), k*(1+tol)] WITHOUT fragmenting regions.

    Over budget: drop whole smallest components first (keeps the rest intact), then
    peel boundary pixels only if a single large component still overshoots. Under
    budget: grow contiguously outward from the current selection. Mutates sel in place.
    """
    lo = int(k * (1 - tol))
    hi = int(k * (1 + tol))
    cur = int(sel.sum())
    if lo <= cur <= hi:
        return
    n_rest = sel.size
    if cur > hi:
        for c in sorted(_label_components(sel, shape, rows, cols), key=len):
            if cur <= hi:
                break
            if cur - c.size >= lo:
                sel[c] = False
                cur -= int(c.size)
        if cur > hi:
            valid = nbr >= 0
            nb_clip = np.where(valid, nbr, 0)
            nb_unsel = (valid & ~sel[nb_clip]) | (~valid)
            boundary = np.where(sel & nb_unsel.any(axis=1))[0]
            if boundary.size:
                order = boundary[np.argsort(scores[boundary])]
                sel[order[:cur - hi]] = False
    elif cur < lo:
        seeds = np.where(sel)[0]
        add = grow_regions_from_seeds(nbr, n_rest, k - cur, seeds, scores, mode, rng,
                                      base=sel.copy())
        sel |= add


class MinPatchSizeRepair(Repair):
    """Enforce the budget AND a minimum patch (component) size S, contiguity-preserving.

    Implements the minimum-patch-size constraint of the "price of contiguity" sweep:
    a feasible plan has every 4-connected component of its selected restoration pixels
    of size >= min_patch_size. Per individual:
      1. bring the restoration selection to the budget tolerance band WITHOUT
         fragmenting (enforce_budget_contiguous), then
      2. iteratively remove any component smaller than S and regrow the freed budget
         by attaching to the surviving components (enforce_budget_contiguous grows from
         the survivors' frontier, so no new sub-S fragments are spawned).
    Guarantees min component size >= S while keeping the pixel count within tolerance.
    S <= 1 reduces to plain contiguous budget enforcement (feature off). The conversion
    block is passed through unchanged.
    """

    def __init__(self, initial_conditions, max_restored_pixels, min_patch_size,
                 scores=None, pixel_tolerance=0.05, growth_bias='scored', max_iters=8):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = int(max_restored_pixels)
        self.min_patch_size = int(min_patch_size)
        self.scores = np.asarray(scores, dtype=np.float64) if scores is not None else None
        self.pixel_tolerance = float(pixel_tolerance)
        self.growth_bias = str(growth_bias).lower()
        self.max_iters = int(max_iters)
        self.nbr, self.rows, self.cols = build_restoration_neighbor_table(initial_conditions)
        self.shape = tuple(initial_conditions['shape'])
        self.call_log = []  # per-generation {min_comp_before, min_comp_after, iters}

    def _repair_one(self, sel, k, scores, mode, rng):
        tol = self.pixel_tolerance
        # 1. contiguous budget enforcement (also seeds a region if sel is empty).
        enforce_budget_contiguous(sel, self.nbr, self.shape, self.rows, self.cols,
                                  k, tol, scores, mode, rng)
        S = self.min_patch_size
        if S <= 1:
            return 0
        # 2. remove sub-S components and regrow onto survivors until none remain.
        iters = 0
        for _ in range(self.max_iters):
            comps = _label_components(sel, self.shape, self.rows, self.cols)
            small = [c for c in comps if c.size < S]
            if not small:
                break
            for c in small:
                sel[c] = False
            enforce_budget_contiguous(sel, self.nbr, self.shape, self.rows, self.cols,
                                      k, tol, scores, mode, rng)
            iters += 1
        return iters

    def _do(self, problem, X, **kwargs):
        n_rest = problem.n_restoration_pixels
        k = min(self.max_restored_pixels, n_rest)
        scores = self.scores if self.scores is not None else np.zeros(n_rest)
        mode = 'scored' if (self.growth_bias == 'scored' and self.scores is not None) else 'neutral'
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        Xp = X.copy()
        audit = self.min_patch_size > 1   # component labeling is only needed for the audit
        min_before, min_after, tot_iters = [], [], 0
        for i in range(len(X)):
            sel = Xp[i, :n_rest].astype(bool)
            if audit:
                pre = _label_components(sel, self.shape, self.rows, self.cols)
                min_before.append(min((c.size for c in pre), default=0))
            tot_iters += self._repair_one(sel, k, scores, mode, rng)
            if audit:
                post = _label_components(sel, self.shape, self.rows, self.cols)
                min_after.append(min((c.size for c in post), default=0))
            Xp[i, :n_rest] = sel.astype(Xp.dtype)
        self.call_log.append({
            'generation': kwargs.get('generation', len(self.call_log) + 1),
            'min_comp_before': int(np.min(min_before)) if min_before else 0,
            'min_comp_after': int(np.min(min_after)) if min_after else 0,
            'total_iters': int(tot_iters),
        })
        return Xp


class SpatialCoverageSampling(Sampling):
    """Initial population whose regions are SPREAD across the whole map.

    Seeds are drawn from many distinct cells of a coarse spatial grid (so starting
    plans cover north/south/west, not just the high-score basin), then grown to the
    pixel budget. Per-individual seed count varies for diversity in the number of
    regions. growth_bias 'neutral' grows location-neutrally (keeps coverage).
    """

    def __init__(self, initial_conditions, max_restored_pixels, scores,
                 region_seeds=25, region_seeds_min=None, growth_bias='neutral',
                 seed_grid=16):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = int(max_restored_pixels)
        self.scores = np.asarray(scores, dtype=np.float64) if scores is not None else None
        self.region_seeds = int(region_seeds)
        self.region_seeds_min = int(region_seeds_min) if region_seeds_min is not None else 1
        self.growth_bias = str(growth_bias).lower()
        self.seed_grid = int(seed_grid)
        self.nbr, self.rows, self.cols = build_restoration_neighbor_table(initial_conditions)
        H, W = initial_conditions['shape']
        gr = (self.rows.astype(np.int64) * self.seed_grid // max(H, 1))
        gc = (self.cols.astype(np.int64) * self.seed_grid // max(W, 1))
        self._cell = (gr * self.seed_grid + gc).astype(np.int64)
        self._cells = np.unique(self._cell)
        order = np.argsort(self._cell, kind='stable')
        self._cell_sorted = self._cell[order]
        self._idx_sorted = np.arange(self._cell.size)[order]

    def _seed_from_cell(self, cell, rng):
        lo = np.searchsorted(self._cell_sorted, cell, side='left')
        hi = np.searchsorted(self._cell_sorted, cell, side='right')
        if hi <= lo:
            return None
        return int(self._idx_sorted[rng.integers(lo, hi)])

    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        n_rest = problem.n_restoration_pixels
        k = min(self.max_restored_pixels, n_rest)
        scores = self.scores if self.scores is not None else np.zeros(n_rest)
        mode = 'scored' if (self.growth_bias == 'scored' and self.scores is not None) else 'neutral'
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

        X = np.zeros((n_samples, n_var), dtype=int)
        for i in range(n_samples):
            s_i = int(rng.integers(max(1, self.region_seeds_min), self.region_seeds + 1))
            cells = rng.choice(self._cells, size=min(s_i, self._cells.size), replace=False)
            seeds = [self._seed_from_cell(c, rng) for c in cells]
            seeds = [s for s in seeds if s is not None]
            if not seeds:
                seeds = [int(rng.integers(n_rest))]
            grown = grow_regions_from_seeds(self.nbr, n_rest, k, seeds, scores, mode, rng)
            X[i, :n_rest] = grown.astype(int)
        return X


class RegionEvolveMutation(RegionGrowingMutation):
    """Region-level mutation: relocate / grow / shrink / spawn / delete regions.

    Reuses RegionGrowingMutation's neighbour table, frontier/boundary helper and
    budget-safe _pick. Each mutated individual applies one region-level move, then is
    trimmed/grown back to the budget tolerance. RELOCATE (move a whole region to a new
    part of the map) is the key spatial-exploration move; SPAWN/DELETE vary the number
    of regions; GROW/SHRINK vary region size.
    """

    def __init__(self, initial_conditions, max_restored_pixels, scores,
                 move_probs=None, **kw):
        super().__init__(initial_conditions, max_restored_pixels, scores, **kw)
        self._shape = initial_conditions['shape']
        self.move_probs = move_probs or {
            'relocate': 0.40, 'spawn': 0.20, 'delete': 0.10,
            'grow': 0.15, 'shrink': 0.15,
        }
        self._moves = list(self.move_probs.keys())
        self._mp = np.array([self.move_probs[m] for m in self._moves], dtype=float)
        self._mp = self._mp / self._mp.sum()

    def _components(self, sel):
        sidx = np.where(sel)[0]
        if sidx.size == 0:
            return []
        m2 = np.zeros(self._shape, dtype=bool)
        m2[self.rows[sidx], self.cols[sidx]] = True
        lab, nc = ndimage.label(m2)
        ids = lab[self.rows[sidx], self.cols[sidx]]
        return [sidx[ids == c] for c in range(1, nc + 1)]

    def _random_unselected(self, sel, rng):
        for _ in range(20):
            i = int(rng.integers(sel.size))
            if not sel[i]:
                return i
        un = np.where(~sel)[0]
        return int(rng.choice(un)) if un.size else None

    def _enforce_budget(self, sel, rng):
        mode = 'scored' if self.growth_bias == 'scored' else 'neutral'
        scores = self.scores if self.scores is not None else np.zeros(sel.size)
        enforce_budget_contiguous(sel, self.nbr, self._shape, self.rows, self.cols,
                                  self.max_restored_pixels, self.pixel_tolerance,
                                  scores, mode, rng)

    def _do(self, problem, X, **kwargs):
        n_rest = problem.n_restoration_pixels
        k = self.max_restored_pixels
        scores = self.scores if self.scores is not None else np.zeros(n_rest)
        mode = 'scored' if self.growth_bias == 'scored' else 'neutral'
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        Xp = X.copy()
        edits = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            if rng.random() > self.prob:
                continue
            sel = Xp[i, :n_rest].astype(bool)
            if sel.sum() == 0:
                continue
            move = self._moves[int(rng.choice(len(self._moves), p=self._mp))]
            comps = self._components(sel)

            if move == 'relocate' and comps:
                c = comps[int(rng.integers(len(comps)))]
                size = int(c.size)
                sel[c] = False
                anchor = self._random_unselected(sel, rng)
                if anchor is not None:
                    grown = grow_regions_from_seeds(self.nbr, n_rest, size, [anchor],
                                                    scores, mode, rng, avoid=sel)
                    sel |= grown
            elif move == 'spawn':
                anchor = self._random_unselected(sel, rng)
                if anchor is not None:
                    s = max(1, int(0.05 * k))
                    grown = grow_regions_from_seeds(self.nbr, n_rest, s, [anchor],
                                                    scores, mode, rng, avoid=sel)
                    sel |= grown
            elif move == 'delete' and len(comps) > 1:
                c = comps[int(rng.integers(len(comps)))]
                sel[c] = False
            elif move == 'grow' and comps:
                c = comps[int(rng.integers(len(comps)))]
                grown = grow_regions_from_seeds(self.nbr, n_rest, int(c.size) + self.n_edits,
                                                list(c), scores, mode, rng, base=sel)
                sel |= grown
            elif move == 'shrink' and comps:
                c = comps[int(rng.integers(len(comps)))]
                cmask = np.zeros(n_rest, dtype=bool)
                cmask[c] = True
                _, boundary = self._boundary_and_frontier(sel)
                bnd = boundary[cmask[boundary]] if boundary.size else boundary
                rem = self._pick(bnd, self.n_edits, prefer_high=False, rng=rng)
                sel[rem] = False

            self._enforce_budget(sel, rng)
            Xp[i, :n_rest] = sel.astype(Xp.dtype)
            edits[i] = int(np.sum(X[i, :n_rest] != Xp[i, :n_rest]))

        self.flip_log.append({
            'generation': self._current_gen + 1,
            'mean_raw_flips': float(np.mean(edits)),
            'std_raw_flips': float(np.std(edits)),
            'total_raw_flips': int(np.sum(edits)),
        })
        return Xp


class RegionSwapCrossover(Crossover):
    """Recombine WHOLE regions between two parents, preserving contiguity.

    Each child takes a random subset of parent A's connected components plus a random
    subset of parent B's, then is trimmed/grown to the pixel budget. Unlike HUX (which
    swaps individual pixels and shatters clumps), this keeps regions intact so the
    search can recombine good spatial pieces.
    """

    def __init__(self, initial_conditions, max_restored_pixels, scores,
                 growth_bias='scored', pixel_tolerance=0.05, **kw):
        super().__init__(2, 2, **kw)
        self.max_restored_pixels = int(max_restored_pixels)
        self.scores = np.asarray(scores, dtype=np.float64) if scores is not None else None
        self.growth_bias = str(growth_bias).lower()
        self.pixel_tolerance = float(pixel_tolerance)
        self._shape = initial_conditions['shape']
        self.nbr, self.rows, self.cols = build_restoration_neighbor_table(initial_conditions)

    def _components(self, sel):
        sidx = np.where(sel)[0]
        if sidx.size == 0:
            return []
        m2 = np.zeros(self._shape, dtype=bool)
        m2[self.rows[sidx], self.cols[sidx]] = True
        lab, nc = ndimage.label(m2)
        ids = lab[self.rows[sidx], self.cols[sidx]]
        return [sidx[ids == c] for c in range(1, nc + 1)]

    def _trim_grow(self, sel, rng):
        mode = 'scored' if self.growth_bias == 'scored' else 'neutral'
        scores = self.scores if self.scores is not None else np.zeros(sel.size)
        enforce_budget_contiguous(sel, self.nbr, self._shape, self.rows, self.cols,
                                  self.max_restored_pixels, self.pixel_tolerance,
                                  scores, mode, rng)
        return sel

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        n_rest = problem.n_restoration_pixels
        k = self.max_restored_pixels
        hi = int(k * (1 + self.pixel_tolerance))
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        Q = np.empty_like(X)
        for m in range(n_matings):
            pa = X[0, m, :n_rest].astype(bool)
            pb = X[1, m, :n_rest].astype(bool)
            comps = self._components(pa) + self._components(pb)
            for off in range(2):
                child = np.zeros(n_rest, dtype=bool)
                if comps:
                    # Assemble WHOLE regions in random order up to the budget, so the
                    # child is a union of intact parent regions near the target size
                    # (few components, minimal growth needed).
                    order = rng.permutation(len(comps))
                    tot = 0
                    for j in order:
                        c = comps[j]
                        if tot >= k:
                            break
                        if tot + c.size <= hi:
                            child[c] = True
                            tot += int(c.size)
                if not child.any():
                    child = (pa if off == 0 else pb).copy()
                self._trim_grow(child, rng)
                Q[off, m, :n_rest] = child.astype(X.dtype)
                if n_var > n_rest:
                    Q[off, m, n_rest:] = X[off, m, n_rest:]
        return Q