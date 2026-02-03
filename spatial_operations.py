"""
Spatial operations and custom operators for restoration optimization
==================================================================

This module contains:
1. Spatial algorithms (clustering, burden sharing)
2. Custom sampling operators for generating initial populations
3. Custom repair operators for constraint enforcement
4. Utility functions for spatial analysis
"""

import numpy as np
from scipy import ndimage
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling

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

# fclass = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
#     52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]

# lu_path = "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif"
# lu_path = "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/AS72_2018.tif"
# sn_dens, profile = compute_sn_dens(lu_path, focal_classes=fclass, radius_m = 300)
# # save sn_dens raster
# output_path = "sn_dens72.tif"
# with rasterio.open(
#     output_path,
#     'w',
#     driver='GTiff',
#     height=sn_dens.shape[0],
#     width=sn_dens.shape[1],
#     count=1,
#     dtype=sn_dens.dtype,
#     crs=profile['crs'],
#     transform=profile['transform'],
#     nodata=np.nan
# ) as dst:
#     dst.write(sn_dens, 1)

#def compute_forest_dens(raster_path, forest_classes, radius_m=300):
    #place holder


# =============================================================================
# BURDEN SHARING AND CLUSTERING ALGORITHMS
# =============================================================================


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
    
    # Check if we already have cached region assignments
    if '_region_assignments_cache' not in initial_conditions:
        #print("DEBUG: Building region assignments cache (first time only)...")
        shape = initial_conditions['shape']
        eligible_indices = initial_conditions['eligible_indices']
        
        # Create raster mask for each admin region
        from rasterio.features import rasterize
        
        transform = initial_conditions['transform']
        crs = initial_conditions['crs']
        
        # Create region assignment for eligible pixels
        region_assignments = np.full(len(eligible_indices), -1, dtype=int)
        
        # For each region, determine which eligible pixels belong to it
        for i, region in enumerate(admin_data['unique_regions']):
            region_geom = admin_data['gdf'][admin_data['gdf'][admin_data['region_column']] == region]
            
            # Rasterize this region
            region_mask = rasterize(
                region_geom.geometry,
                out_shape=shape,
                transform=transform,
                fill=0,
                default_value=1
            ).astype(bool)
            
            # Find eligible pixels in this region
            eligible_in_region = region_mask.flatten()[eligible_indices]
            region_assignments[eligible_in_region] = i
        
        # Cache the assignments and create a list of non-empty regions
        non_empty_regions = []
        for region_id in range(admin_data['n_regions']):
            region_pixels = np.where(region_assignments == region_id)[0]
            if len(region_pixels) > 0:
                non_empty_regions.append((region_id, region_pixels))
        
        initial_conditions['_region_assignments_cache'] = region_assignments
        initial_conditions['_non_empty_regions_cache'] = non_empty_regions
        #print(f"DEBUG: Cached {len(non_empty_regions)} non-empty regions out of {admin_data['n_regions']} total")
    else:
        # Use cached data
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
        self.enable_conversions = 'landscape_anomaly' in initial_conditions
    
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


# =============================================================================
# CUSTOM REPAIR OPERATOR
# =============================================================================

class AdaptiveRepair(Repair):
    """
    Unified repair that handles all constraint enforcement strategies.
    Consolidates previous ClusteringRepair, BurdenSharingRepair, CombinedRepair, ScoreCountRepair, and ExactCountRepair.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels, scenario_params=None, scores=None):
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
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        score_repairs = 0
        random_repairs = 0
        
        for i in range(len(X)):
            try:
                x = X[i].copy()
                n_pixels = problem.n_pixels
                
                # Allow some flexibility in total count to maintain diversity
                # Only repair solutions that are significantly over/under target
                current_total = np.sum(x[:n_pixels]) + np.sum(x[n_pixels:])
                target = self.max_restored_pixels
                
                # Only repair if count is >20% off target (preserve diversity for small differences)
                tolerance = max(target * 0.2, 100)  # 20% tolerance or at least 100 pixels
                
                if abs(current_total - target) > tolerance:
                    if self.scores is not None and np.random.random() < 0.3:
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
        
        return X_repaired
    
    def _enforce_count_with_scores(self, x, problem):
        """Enforce exact count using scores to prioritize which pixels to add/remove."""
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
                # Prioritize restoration actions for now (since convert is forced to 0)
                zeros_restore = np.where(x_restore == 0)[0]
                if zeros_restore.size > 0 and need > 0:
                    # add best available restoration zeros
                    add = zeros_restore[np.argsort(-self.scores[zeros_restore])][:need]
                    x_restore[add] = 1

            elif cur_total > k:
                drop = cur_total - k
                # Remove from restoration actions first (since convert should be 0)
                ones_restore = np.where(x_restore == 1)[0]
                if ones_restore.size > 0 and drop > 0:
                    # remove worst available restoration ones
                    rem = ones_restore[np.argsort(self.scores[ones_restore])][:drop]
                    x_restore[rem] = 0

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