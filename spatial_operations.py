"""
Spatial operations for restoration optimization
==============================================

This module contains functions and classes for spatial clustering, burden sharing,
and related repair operators for the restoration optimization problem.
"""

import numpy as np
from scipy import ndimage
from pymoo.core.repair import Repair

import rasterio
import numpy as np
from scipy.ndimage import generic_filter

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

#def compute_forest_dens(raster_path, forest_classes, radius_m=300):
    #place holder


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
        print("DEBUG: Building region assignments cache (first time only)...")
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
        print(f"DEBUG: Cached {len(non_empty_regions)} non-empty regions out of {admin_data['n_regions']} total")
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
# REPAIR OPERATORS
# =============================================================================

class ClusteringRepair(Repair):
    """
    Repair operator that maintains spatial clustering after crossover/mutation.
    """
    
    def __init__(self, initial_conditions, clustering_strength=0.5, max_restored_pixels=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.clustering_strength = clustering_strength
        self.max_restored_pixels = max_restored_pixels
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i]
            if np.sum(x) > 0 and self.clustering_strength > 0:
                x = apply_spatial_clustering(
                    x, 
                    self.initial_conditions, 
                    self.clustering_strength,
                    exact_count=self.max_restored_pixels
                )
                # Additional enforcement if needed
                if self.max_restored_pixels is not None:
                    x = _enforce_exact_pixel_count(x, self.max_restored_pixels)
            X_repaired[i] = x
        
        return X_repaired


class BurdenSharingRepair(Repair):
    """
    Repair operator that maintains burden sharing after crossover/mutation.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i]
            if np.sum(x) > 0:
                seed = hash(tuple(x)) % (2**31 - 1)  # Use int32 safe range
                x = apply_burden_sharing(x, self.initial_conditions, seed=seed, exact_count=self.max_restored_pixels)
                # Additional enforcement if needed
                if self.max_restored_pixels is not None:
                    x = _enforce_exact_pixel_count(x, self.max_restored_pixels)
            X_repaired[i] = x
        
        return X_repaired


class CombinedRepair(Repair):
    """
    Repair operator that maintains both burden sharing and clustering with exact count.
    """
    
    def __init__(self, initial_conditions, clustering_strength=0.5, max_restored_pixels=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.clustering_strength = clustering_strength
        self.max_restored_pixels = max_restored_pixels
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i]
            if np.sum(x) > 0:
                # Apply burden sharing
                seed = hash(tuple(x)) % (2**31 - 1)  # Use int32 safe range
                x = apply_burden_sharing(x, self.initial_conditions, seed=seed, exact_count=self.max_restored_pixels)
                
                # Apply clustering
                if self.clustering_strength > 0:
                    x = apply_spatial_clustering(
                        x, 
                        self.initial_conditions, 
                        self.clustering_strength,
                        exact_count=self.max_restored_pixels
                    )
                
                # Final enforcement if needed
                if self.max_restored_pixels is not None:
                    x = _enforce_exact_pixel_count(x, self.max_restored_pixels)
            X_repaired[i] = x
        
        return X_repaired