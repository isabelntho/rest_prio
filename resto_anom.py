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
import pickle
import json
from datetime import datetime

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation

# =============================================================================
# SCENARIO PARAMETERS
# =============================================================================

def define_scenario_parameters():
    """
    Define lists of parameter values for scenario elements
    Returns:
        dict: Dictionary with parameter names as keys and lists of values
    """
    scenario_parameters = {
        'max_restoration_fraction': [0.15, 0.25, 0.35, 0.50], # Proportion of study area to restore
        'spatial_clustering': [0.6, 1.0], # Degree of spatial clustering (0=random, 1=highly clustered) (could add in 0.3)
        'burden_sharing': ['no', 'yes'] # Equal sharing across admin regions (no/yes)
    }
    
    return scenario_parameters

def get_scenario_combinations():
    """
    Get all possible combinations of scenario parameters.
    
    Returns:
        list: List of parameter combinations (dicts)
    """
    from itertools import product
    
    params = define_scenario_parameters()
    param_names = list(params.keys())
    param_values = list(params.values())
    
    # Generate all combinations
    combinations = []
    for combo in product(*param_values):
        combo_dict = dict(zip(param_names, combo))
        combinations.append(combo_dict)
    
    return combinations

def get_scenario_by_index(scenario_index):
    """
    Get a specific scenario by index.
    
    Args:
        scenario_index: Index of the scenario (0-based)
        
    Returns:
        dict: Parameter combination for the scenario
    """
    combinations = get_scenario_combinations()
    
    if 0 <= scenario_index < len(combinations):
        return combinations[scenario_index]
    else:
        raise ValueError(f"Scenario index {scenario_index} out of range. Available: 0-{len(combinations)-1}")

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_admin_regions(workspace_dir, region='Bern'):
    """
    Load administrative regions from shapefile for burden sharing.
    
    Args:
        workspace_dir: Directory containing admin shapefile
        region: Region to optimize for ('Bern' or 'CH')
                - 'Bern': Filters to Bern canton, uses district-level admin regions
                - 'CH': Uses all cantons for burden sharing
        
    Returns:
        dict: Admin regions data with region_map and region_counts
    """
    import geopandas as gpd
    
    admin_file = 'Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp'
    kanton_file = 'Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp'

    if not os.path.exists(kanton_file):
        print(f"Warning: Kanton shapefile not found at {kanton_file}")
        return None
    
    try:
        if region == 'CH':
            # Use cantons for burden sharing across Switzerland
            gdf = gpd.read_file(kanton_file)
            region_col = 'NAME'
            unique_regions = gdf[region_col].unique()
            
            print(f"✓ Loaded {len(unique_regions)} cantons for CH-wide optimization")
            
            return {
                'gdf': gdf,
                'region_column': region_col,
                'unique_regions': unique_regions,
                'n_regions': len(unique_regions)
            }
            
        elif region == 'Bern':
            # Filter to Bern canton and use district-level admin regions
            if not os.path.exists(admin_file):
                print(f"Warning: Admin shapefile not found at {admin_file}")
                return None
            
            # Load kanton shapefile and filter to Bern
            kanton_gdf = gpd.read_file(kanton_file)
            bern_gdf = kanton_gdf[kanton_gdf['NAME'] == 'Bern'].copy()
            
            if len(bern_gdf) == 0:
                print(f"Warning: No canton named 'Bern' found in {kanton_file}")
                return None
            
            print(f"✓ Filtered to Bern canton")
            
            # Load admin shapefile and crop/mask to Bern
            gdf = gpd.read_file(admin_file)
            gdf_bern = gpd.clip(gdf, bern_gdf)
            
            region_col = 'NAME'     
            unique_regions = gdf_bern[region_col].unique()
            
            print(f"✓ Cropped admin regions to Bern: {len(unique_regions)} regions")
            
            return {
                'gdf': gdf_bern,
                'region_column': region_col,
                'unique_regions': unique_regions,
                'n_regions': len(unique_regions)
            }
        else:
            print(f"Warning: Unknown region '{region}'. Use 'Bern' or 'CH'")
            return None
        
    except Exception as e:
        print(f"Error loading admin shapefile: {e}")
        return None

def load_initial_conditions(workspace_dir, objectives=None, region='Bern'):
    """
    Args:
        workspace_dir: Directory containing input data files (.tif)
        objectives: List of objectives to load (e.g., ['abiotic', 'biotic', 'landscape', 'cost'])
                   If None, loads all available objectives
        region: Region to optimize for ('Bern' or 'CH'), passed to load_admin_regions
    Returns:
        dict: Initial conditions for specified objectives
    """
    # Define all possible objectives and their file mappings
    all_objectives = {
        'abiotic': 'abiotic_condition_anomaly.tif',
        'biotic': 'biotic_condition_anomaly.tif', 
        'landscape': 'landscape_condition_anomaly.tif',
        'cost': 'implementation_cost.tif',
        'population_proximity': 'population_proximity.tif'
    }
    
    # Use all objectives if none specified
    if objectives is None:
        objectives = list(all_objectives.keys())
    
    print(f"Loading initial conditions for objectives: {objectives}")
    
    # Build file paths for selected objectives
    data_files = {}
    for obj in objectives:
        if obj in all_objectives:
            filename = all_objectives[obj]
            if obj == 'cost':
                data_files['implementation_cost'] = os.path.join(workspace_dir, filename)
            else:
                data_files[f'{obj}_anomaly'] = os.path.join(workspace_dir, filename)
        else:
            raise ValueError(f"Unknown objective: {obj}. Available: {list(all_objectives.keys())}")
    
    initial_conditions = {}
    
    # Check if all required files exist first
    missing_files = []
    for objective, file_path in data_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"  - {objective}: {file_path}")
    
    if missing_files:
        print(f"\n✗ ERROR: Required data files not found:")
        print(f"  {workspace_dir}")
        raise FileNotFoundError(f"Missing {len(missing_files)} required data file(s).")
    
    # Load all data files and track NaN locations
    nan_masks = {}  # Store NaN locations before replacement
    
    for objective, file_path in data_files.items():
        with rio.open(file_path) as src:
            data = src.read(1)
            if 'crs' not in initial_conditions:  # First file loaded
                initial_conditions['crs'] = src.crs
                initial_conditions['transform'] = src.transform
                initial_conditions['shape'] = data.shape
        
            # Track NaN locations BEFORE replacement
            nan_mask = np.isnan(data)
            nan_masks[objective] = nan_mask
            nan_count = np.sum(nan_mask)
            total_pixels = data.size
            
            if nan_count > 0:
                print(f"✓ Loaded {objective}: {data.shape} ({nan_count}/{total_pixels} = {100*nan_count/total_pixels:.1f}% NaN)")
                # Replace NaN with 0 so np.sum() works correctly in objective calculations
                data = np.nan_to_num(data, nan=0.0)
                print(f"  → Replaced NaN with 0 (pixels excluded via eligible_mask)")
            else:
                print(f"✓ Loaded {objective}: {data.shape}")
                
            initial_conditions[objective] = data
    
    # Create eligibility mask - exclude pixels that were NaN in ANY objective
    shape = initial_conditions['shape']
    eligible_mask = np.ones(shape, dtype=bool)
    
    for obj_name, nan_mask in nan_masks.items():
        if np.any(nan_mask):
            eligible_mask = eligible_mask & ~nan_mask
            nan_excluded = np.sum(nan_mask)
            print(f"  Masking {nan_excluded} NaN pixels from {obj_name}")

    # You might want to exclude certain areas, e.g.:
    # eligible_mask = (initial_conditions['implementation_cost'] < 8000) & \
    #                 (initial_conditions['abiotic_anomaly'] > 0.1)
    
    initial_conditions['eligible_mask'] = eligible_mask
    eligible_indices = np.where(eligible_mask.flatten())[0]
    initial_conditions['eligible_indices'] = eligible_indices
    initial_conditions['n_pixels'] = len(eligible_indices)
    
    total_pixels = shape[0] * shape[1]
    print(f"✓ Eligibility mask created: {initial_conditions['n_pixels']}/{total_pixels} eligible pixels ({100*initial_conditions['n_pixels']/total_pixels:.1f}% of raster)")
    
    # Load admin regions for burden sharing
    admin_data = load_admin_regions(workspace_dir, region=region)
    initial_conditions['admin_data'] = admin_data
    
    print(f"✓ Data preparation complete: {initial_conditions['n_pixels']} eligible pixels")
    
    return initial_conditions

# =============================================================================
# RESTORATION EFFECT FUNCTION
# =============================================================================

def restoration_effect(decision_vars, initial_conditions, effect_params=None):
    """
    Define what happens when restoration is selected (decision_var = 1).
    This function calculates the effect of restoration on neighboring cells.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial objective values
        effect_params: Parameters controlling restoration effects
        
    Returns:
        dict: Updated objective values after restoration effects
    """
    if effect_params is None:
        # Modify these values to customize restoration effects
        effect_params = {
            'abiotic_improvement': 0.01,      # Reduction in abiotic anomaly
            'biotic_improvement': 0.01,       # Reduction in biotic anomaly  
            'landscape_improvement': 0.01,    # Reduction in landscape anomaly
            'neighbor_radius': 1,            # Effect radius in cells
            'neighbor_effect_decay': 0.5     # Effect strength for neighbors
        }
    
    shape = initial_conditions['shape']
    eligible_mask = initial_conditions['eligible_mask']
    eligible_indices = initial_conditions['eligible_indices']
    
    # Create 2D restoration mask from 1D decision variables
    restoration_mask_2d = np.zeros(shape, dtype=bool)
    if np.any(decision_vars):
        # Convert eligible indices with restoration back to 2D coordinates
        restoration_eligible_indices = eligible_indices[decision_vars == 1]
        rows, cols = np.divmod(restoration_eligible_indices, shape[1])
        restoration_mask_2d[rows, cols] = True
    
    # Initialize updated conditions
    updated_conditions = {}
    
    # Process only objectives that are available in initial_conditions
    available_anomaly_objectives = [obj for obj in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly'] 
                                   if obj in initial_conditions]
    
    for objective in available_anomaly_objectives:
        original_values = initial_conditions[objective].copy()
        updated_values = original_values.copy()
        
        if np.any(restoration_mask_2d):
            # Direct effect on restored cells
            improvement = effect_params[f'{objective.split("_")[0]}_improvement']
            
            # Apply improvement:
            updated_values[restoration_mask_2d] = (
                original_values[restoration_mask_2d] + improvement
            )
            
            # Neighbor effects
            if effect_params['neighbor_radius'] > 0:
                # Create kernel for neighbor effects
                radius = effect_params['neighbor_radius']
                y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                kernel = (x*x + y*y) <= radius*radius
                
                # Apply dilation to find neighbor cells
                neighbor_mask = ndimage.binary_dilation(restoration_mask_2d, structure=kernel)
                neighbor_mask = neighbor_mask & ~restoration_mask_2d  # Exclude direct restoration cells
                
                # Apply reduced improvement to neighbors
                neighbor_improvement = improvement * effect_params['neighbor_effect_decay']
                updated_values[neighbor_mask] = (
                    original_values[neighbor_mask] + neighbor_improvement
                )
        
        # Only apply changes to eligible pixels; restore original values elsewhere
        # This prevents affecting NaN→0 pixels outside the study area
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
            total_cost = np.sum(base_cost[restoration_mask_2d])
        updated_conditions['implementation_cost'] = total_cost
    
    # Population proximity (only if proximity objective is being used)
    # Proximity is a fixed spatial property - no restoration improvement effect
    if 'population_proximity' in initial_conditions:
        proximity_map = initial_conditions['population_proximity'].copy()
        avg_distance = 0.0
        if np.any(restoration_mask_2d):
            avg_distance = np.mean(proximity_map[restoration_mask_2d])
        updated_conditions['population_proximity'] = avg_distance
    
    return updated_conditions

# =============================================================================
# SPATIAL CLUSTERING FUNCTIONS
# =============================================================================

def apply_burden_sharing(decision_vars, initial_conditions, seed=None):
    """
    Apply burden sharing to ensure equal restoration across admin regions.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial conditions including admin data
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        numpy.array: Modified decision variables with burden sharing applied
    """
    admin_data = initial_conditions.get('admin_data')
    
    if admin_data is None:
        print("Warning: No admin data available for burden sharing")
        return decision_vars.copy()
    
    shape = initial_conditions['shape']
    eligible_indices = initial_conditions['eligible_indices']
    total_restore = np.sum(decision_vars)
    
    if total_restore == 0:
        return decision_vars.copy()
    
    # Create raster mask for each admin region
    # This is a simplified implementation - you may need to adjust based on your shapefile
    from rasterio.features import rasterize
    
    transform = initial_conditions['transform']
    crs = initial_conditions['crs']
    
    # Convert eligible indices to 2D coordinates
    rows, cols = np.divmod(eligible_indices, shape[1])
    
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
    
    # Calculate target restoration per region (equal sharing)
    n_regions = admin_data['n_regions']
    restore_per_region = total_restore // n_regions
    extra_restores = total_restore % n_regions
    
    # Apply burden sharing
    new_decision_vars = np.zeros_like(decision_vars)
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    for region_id in range(n_regions):
        region_pixels = np.where(region_assignments == region_id)[0]
        
        if len(region_pixels) == 0:
            continue
            
        # Determine restoration target for this region
        target = restore_per_region + (1 if region_id < extra_restores else 0)
        target = min(target, len(region_pixels))  # Can't restore more pixels than available
        
        if target > 0:
            # Select pixels to restore in this region (deterministic if seed provided)
            selected_pixels = np.random.choice(region_pixels, target, replace=False)
            new_decision_vars[selected_pixels] = 1
    
    return new_decision_vars

def apply_spatial_clustering(decision_vars, initial_conditions, clustering_strength=0.0):
    """
    Apply spatial clustering to decision variables to promote spatially coherent restoration.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial conditions including shape and eligible indices
        clustering_strength: Degree of clustering (0.0=no change, 1.0=maximum clustering)
        
    Returns:
        numpy.array: Modified decision variables with spatial clustering applied
    """
    if clustering_strength <= 0.0 or not np.any(decision_vars):
        return decision_vars.copy()
    
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
    
    return new_decision_vars

# =============================================================================
# CUSTOM SAMPLING AND REPAIR OPERATORS
# =============================================================================

class ClusteredSampling(Sampling):
    """
    Custom sampling that generates spatially clustered restoration patterns.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels, clustering_strength=0.5):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
        self.clustering_strength = clustering_strength
    
    def _do(self, problem, n_samples, **kwargs):
        n_pixels = problem.n_var
        X = np.zeros((n_samples, n_pixels), dtype=int)
        
        for i in range(n_samples):
            # Generate a random solution with target number of pixels
            x = np.zeros(n_pixels, dtype=int)
            
            # Randomly select target number of pixels
            n_restore = np.random.randint(0, self.max_restored_pixels + 1)
            if n_restore > 0:
                restore_indices = np.random.choice(n_pixels, n_restore, replace=False)
                x[restore_indices] = 1
                
                # Apply spatial clustering to generate clustered pattern
                if self.clustering_strength > 0:
                    x = apply_spatial_clustering(
                        x, 
                        self.initial_conditions, 
                        self.clustering_strength
                    )
            
            X[i] = x
        
        return X


class BurdenSharingSampling(Sampling):
    """
    Custom sampling that generates solutions with equal restoration across regions.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
    
    def _do(self, problem, n_samples, **kwargs):
        n_pixels = problem.n_var
        X = np.zeros((n_samples, n_pixels), dtype=int)
        
        for i in range(n_samples):
            # Generate a random solution with target number of pixels
            x = np.zeros(n_pixels, dtype=int)
            
            n_restore = np.random.randint(0, self.max_restored_pixels + 1)
            if n_restore > 0:
                restore_indices = np.random.choice(n_pixels, n_restore, replace=False)
                x[restore_indices] = 1
                
                # Apply burden sharing to distribute equally across regions
                seed = np.random.randint(0, 2**32)
                x = apply_burden_sharing(x, self.initial_conditions, seed=seed)
            
            X[i] = x
        
        return X


class CombinedSampling(Sampling):
    """
    Custom sampling that combines burden sharing and spatial clustering.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels, clustering_strength=0.5):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
        self.clustering_strength = clustering_strength
    
    def _do(self, problem, n_samples, **kwargs):
        n_pixels = problem.n_var
        X = np.zeros((n_samples, n_pixels), dtype=int)
        
        for i in range(n_samples):
            x = np.zeros(n_pixels, dtype=int)
            
            n_restore = np.random.randint(0, self.max_restored_pixels + 1)
            if n_restore > 0:
                restore_indices = np.random.choice(n_pixels, n_restore, replace=False)
                x[restore_indices] = 1
                
                # Apply burden sharing first
                seed = np.random.randint(0, 2**32)
                x = apply_burden_sharing(x, self.initial_conditions, seed=seed)
                
                # Then apply spatial clustering
                if self.clustering_strength > 0:
                    x = apply_spatial_clustering(
                        x, 
                        self.initial_conditions, 
                        self.clustering_strength
                    )
            
            X[i] = x
        
        return X


class ClusteringRepair(Repair):
    """
    Repair operator that maintains spatial clustering after crossover/mutation.
    """
    
    def __init__(self, initial_conditions, clustering_strength=0.5):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.clustering_strength = clustering_strength
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i]
            if np.sum(x) > 0 and self.clustering_strength > 0:
                x = apply_spatial_clustering(
                    x, 
                    self.initial_conditions, 
                    self.clustering_strength
                )
            X_repaired[i] = x
        
        return X_repaired


class BurdenSharingRepair(Repair):
    """
    Repair operator that maintains burden sharing after crossover/mutation.
    """
    
    def __init__(self, initial_conditions):
        super().__init__()
        self.initial_conditions = initial_conditions
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i]
            if np.sum(x) > 0:
                seed = hash(tuple(x)) % (2**32)
                x = apply_burden_sharing(x, self.initial_conditions, seed=seed)
            X_repaired[i] = x
        
        return X_repaired


class CombinedRepair(Repair):
    """
    Repair operator that maintains both burden sharing and clustering.
    """
    
    def __init__(self, initial_conditions, clustering_strength=0.5):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.clustering_strength = clustering_strength
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i]
            if np.sum(x) > 0:
                # Apply burden sharing
                seed = hash(tuple(x)) % (2**32)
                x = apply_burden_sharing(x, self.initial_conditions, seed=seed)
                
                # Apply clustering
                if self.clustering_strength > 0:
                    x = apply_spatial_clustering(
                        x, 
                        self.initial_conditions, 
                        self.clustering_strength
                    )
            X_repaired[i] = x
        
        return X_repaired


def create_parameter_illustration_maps(initial_conditions, save_path=None):
    """
    Create maps illustrating different parameter options.
    
    Args:
        initial_conditions: Dict with initial conditions - load with load_initial_conditions()
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    shape = initial_conditions['shape']
    eligible_indices = initial_conditions['eligible_indices']
    
    # Generate sample restoration pattern
    np.random.seed(42)  # For reproducible illustrations
    
    # Row 1: Different restoration fractions (medium clustering = 0.3)
    fractions = [0.15, 0.25, 0.35, 0.50]
    clustering = 0.3
    
    # Row 2: Different clustering levels (medium fraction = 0.25)  
    clusterings = [0.0, 0.3, 0.6, 1.0]
    fraction = 0.25
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Varying restoration fraction
    for i, frac in enumerate(fractions):
        n_restore = int(frac * len(eligible_indices))
        decision_vars = np.zeros(len(eligible_indices))
        if n_restore > 0:
            decision_vars[np.random.choice(len(eligible_indices), n_restore, replace=False)] = 1
        
        # Apply clustering
        clustered_vars = apply_spatial_clustering(decision_vars, initial_conditions, clustering)
        
        # Convert to 2D for visualization
        restore_map = np.zeros(shape)
        restore_indices = eligible_indices[clustered_vars == 1]
        rows, cols = np.divmod(restore_indices, shape[1])
        restore_map[rows, cols] = 1
        
        axes[0, i].imshow(restore_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, i].set_title(f'Restore {frac*100:.0f}%\n(Clustering=0.3)')
        axes[0, i].axis('off')
    
    # Row 2: Varying clustering
    n_restore = int(fraction * len(eligible_indices))
    base_decision_vars = np.zeros(len(eligible_indices))
    if n_restore > 0:
        base_decision_vars[np.random.choice(len(eligible_indices), n_restore, replace=False)] = 1
    
    for i, clust in enumerate(clusterings):
        clustered_vars = apply_spatial_clustering(base_decision_vars, initial_conditions, clust)
        
        # Convert to 2D for visualization
        restore_map = np.zeros(shape)
        restore_indices = eligible_indices[clustered_vars == 1]
        rows, cols = np.divmod(restore_indices, shape[1])
        restore_map[rows, cols] = 1
        
        axes[1, i].imshow(restore_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, i].set_title(f'Clustering={clust}\n(Restore 25%)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter illustration saved to: {save_path}")
    
    plt.show()
    return fig

# Add this function before the RestorationProblem class (around line 750)

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
    problem = RestorationProblem(initial_conditions, scenario_params)
    
    print(f"\nPROBLEM SETUP:")
    print(f"   - Decision variables: {problem.n_var} (eligible pixels)")
    print(f"   - Objectives: {problem.n_obj} ({problem.objective_names})")
    print(f"   - Constraints: {problem.n_constr}")
    print(f"   - Max restored pixels: {problem.max_restored_pixels}")
    print(f"   - Restoration fraction: {scenario_params['max_restoration_fraction']*100:.1f}%")
    
    # Check baseline objectives
    print(f"\nBASELINE OBJECTIVES (no restoration):")
    x_none = np.zeros(problem.n_var, dtype=int)
    out_none = {}
    problem._evaluate(x_none, out_none)
    
    for i, obj_name in enumerate(problem.objective_names):
        if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
            # Display actual anomaly sum (reverse the negative)
            actual_sum = -out_none['F'][i]
            print(f"   - {obj_name}: sum={actual_sum:.4e} (objective={out_none['F'][i]:.4e})")
        else:
            print(f"   - {obj_name}: {out_none['F'][i]:.4e}")
    print(f"   - Constraint violation: {out_none['G'][0]:.4e} (should be ≤ 0)")
    
    # Test sample solutions
    #print(f"\n3. SAMPLE SOLUTIONS (random restoration patterns):")
    
    feasible_count = 0
    infeasible_count = 0
    
    for i in range(n_samples):
        # Generate random solution at max allowed restoration
        x = np.zeros(problem.n_var, dtype=int)
        n_restore = problem.max_restored_pixels
        if n_restore > 0 and n_restore <= problem.n_var:
            restore_indices = np.random.choice(problem.n_var, n_restore, replace=False)
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
    
    # Check if objectives can improve
    #print(f"\n4. OBJECTIVE IMPROVEMENT POTENTIAL:")
    
    if feasible_count > 0:
        print(f"   ✓ Feasible solutions exist")
        print(f"   ✓ Objectives show improvement with restoration")
    else:
        print(f"   ✗ WARNING: No feasible solutions found in {n_samples} samples!")
        print(f"   → Check if max_restored_pixels constraint is too restrictive")
    
    # Check for common issues
    #print(f"\n5. POTENTIAL ISSUES:")
    issues = []
    
    if problem.max_restored_pixels == 0:
        issues.append("   ✗ Max restored pixels is 0 - no restoration possible!")
    
    if problem.max_restored_pixels > problem.n_var:
        issues.append(f"   ✗ Max restored pixels ({problem.max_restored_pixels}) > available pixels ({problem.n_var})")
    
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
        n_restore = int(frac * problem.n_var)
        if n_restore > 0 and n_restore <= problem.n_var:
            x = np.zeros(problem.n_var, dtype=int)
            restore_indices = np.random.choice(problem.n_var, n_restore, replace=False)
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
            mask = initial_conditions['eligible_mask']
            data1 = initial_conditions[anomaly_objs[i]][mask]
            data2 = initial_conditions[anomaly_objs[j]][mask]
            corr = np.corrcoef(data1, data2)[0, 1]
            print(f"   {anomaly_objs[i]} vs {anomaly_objs[j]}: {corr:.4f}")

    # Add this section in diagnose_optimization_setup() after line 897 (after spatial correlations, before "END DIAGNOSTICS")
    
        # Check if improvement pushes values above threshold
        print(f"\nANOMALY VALUE DISTRIBUTIONS (threshold diagnostic):")
        print(f"   Testing with {problem.max_restored_pixels} restored pixels...")
        
        # Test with max restoration
        x_test = np.zeros(problem.n_var, dtype=int)
        if problem.max_restored_pixels > 0:
            restore_indices = np.random.choice(problem.n_var, problem.max_restored_pixels, replace=False)
            x_test[restore_indices] = 1
        
        # Get updated conditions after restoration
        updated = restoration_effect(x_test, initial_conditions)
        
        eligible_mask = initial_conditions['eligible_mask']
        
        for obj_name in anomaly_objs:
            before = initial_conditions[obj_name][eligible_mask]
            after = updated[obj_name][eligible_mask]
            
            # For restored pixels only
            restored_2d = np.zeros(initial_conditions['shape'], dtype=bool)
            rows, cols = np.divmod(initial_conditions['eligible_indices'][x_test == 1], initial_conditions['shape'][1])
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
            print(f"         → Improvement adds {np.mean(after_restored - before_restored):.3f} on average")    

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
    
    def __init__(self, initial_conditions, scenario_params):
        """
        Initialize the optimization problem.
        
        Args:
            initial_conditions: Dict with initial objective states
            scenario_params: Dict with scenario parameters (e.g., max_restoration_fraction)
        """
        self.initial_conditions = initial_conditions
        self.scenario_params = scenario_params
        
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
        
        n_pixels = initial_conditions['n_pixels']
        max_restoration_fraction = scenario_params['max_restoration_fraction']
        self.max_restored_pixels = int(max_restoration_fraction * n_pixels)
        
        # Binary decision variables: 0 = no restoration, 1 = restoration
        super().__init__(
            n_var=n_pixels,           # One decision per eligible pixel
            n_obj=n_objectives,       # Variable number of objectives
            n_constr=1,              # Constraint on total restoration area
            xl=0,                    # Lower bound: no restoration
            xu=1,                    # Upper bound: restoration
            type_var=int             # Integer (binary) variables
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution (restoration plan).
        
        Args:
            x: Decision variables (binary array) - already clustered/burden-shared by sampling/repair
            out: Output dictionary for objectives and constraints
        """
        # Solution x is already properly structured from custom sampling/repair operators
        # No post-processing needed - just evaluate as-is
        n_restored = np.sum(x)
        
        # Apply restoration effects directly to decision variables
        updated_conditions = restoration_effect(x, self.initial_conditions)
        
        # Calculate objective values (all to minimize) - only for available objectives
        objectives = []
        
        for obj_name in self.objective_names:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
                # Anomaly objectives: MINIMIZE negative sum (equivalent to MAXIMIZE sum)
                # Higher anomalies are better, so we want to maximize the sum
                # NSGA-II minimizes, so we minimize the negative
                threshold = 0.0  # or different per objective
                pixels_above = np.sum(updated_conditions[obj_name] > threshold)
                obj_value = -pixels_above  # Maximize count above threshold
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
        
        # Constraint: limit total restoration area
        out["G"] = [n_restored - self.max_restored_pixels]  # Violation if positive

# =============================================================================
# OPTIMIZATION EXECUTION
# =============================================================================

def run_single_scenario_optimization(initial_conditions, scenario_params, pop_size=50, 
                                   n_generations=100, save_results=True, verbose=True):
    """
    Run the multi-objective restoration optimization for a single scenario.
    
    Args:
        initial_conditions: Initial objective conditions
        scenario_params: Dict with scenario parameters
        pop_size: Population size for NSGA-II
        n_generations: Number of optimization generations
        save_results: Whether to save results to files
        verbose: Print progress information
        
    Returns:
        dict: Optimization results
    """
    if verbose:
        print(f"\n=== SINGLE SCENARIO OPTIMIZATION ===")
        print(f"Scenario parameters: {scenario_params}")
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_generations}")
        print(f"Max restoration: {scenario_params['max_restoration_fraction']*100:.1f}% of eligible area")
        print(f"Eligible pixels: {initial_conditions['n_pixels']}")
        
        # Show which objectives are being used
        problem_temp = RestorationProblem(initial_conditions, scenario_params)
        print(f"Objectives: {problem_temp.objective_names} ({len(problem_temp.objective_names)} total)")
    
    # Create optimization problem
    problem = RestorationProblem(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params
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
        print("\nRunning diagnostics to check optimization setup...")
        diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10)
    
    # Create custom sampling and repair operators based on scenario parameters
    burden_sharing = scenario_params.get('burden_sharing', 'no')
    clustering_strength = scenario_params.get('spatial_clustering', 0.0)
    
    # Select appropriate sampling strategy
    if burden_sharing == 'yes' and clustering_strength > 0.0:
        sampling = CombinedSampling(
            initial_conditions, 
            problem.max_restored_pixels, 
            clustering_strength
        )
        repair = CombinedRepair(initial_conditions, clustering_strength)
        if verbose:
            print(f"  Using combined burden-sharing + clustering (strength={clustering_strength})")
    elif burden_sharing == 'yes':
        sampling = BurdenSharingSampling(initial_conditions, problem.max_restored_pixels)
        repair = BurdenSharingRepair(initial_conditions)
        if verbose:
            print(f"  Using burden-sharing sampling/repair")
    elif clustering_strength > 0.0:
        sampling = ClusteredSampling(
            initial_conditions, 
            problem.max_restored_pixels, 
            clustering_strength
        )
        repair = ClusteringRepair(initial_conditions, clustering_strength)
        if verbose:
            print(f"  Using clustered sampling/repair (strength={clustering_strength})")
    else:
        sampling = BinaryRandomSampling()
        repair = None
        if verbose:
            print(f"  Using standard binary random sampling")
    
    # Create optimization algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,                # Custom or standard sampling
        crossover=HUX(),                  # Half-uniform crossover
        mutation=BitflipMutation(),       # Bit-flip mutation
        repair=repair                     # Custom repair to maintain properties
    )
    
    # Set termination criteria
    termination = get_termination("n_gen", n_generations)
    
    # Progress callback
    class ProgressCallback:
        def __init__(self, verbose=True):
            self.verbose = verbose
            self.start_time = None
            
        def __call__(self, algorithm):
            if self.start_time is None:
                self.start_time = datetime.now()
            
            gen = algorithm.n_gen
            max_gen = algorithm.termination.n_max_gen
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.verbose and gen % 10 == 0:
                progress = (gen / max_gen) * 100
                eta = (elapsed / gen) * (max_gen - gen) if gen > 0 else 0
                print(f"   Generation {gen}/{max_gen} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed/60:.1f}min - ETA: {eta/60:.1f}min")
    
    # Run optimization
    if verbose:
        print("Starting optimization...")
    
    try:
        callback = ProgressCallback(verbose=verbose)
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=42,                # For reproducible results
            verbose=False,
            callback=callback
        )
        
        # Debug: Check what's in the result
        if verbose:
            print(f"\nDEBUG - Result object:")
            print(f"  result is None: {result is None}")
            if result is not None:
                print(f"  hasattr(result, 'F'): {hasattr(result, 'F')}")
                print(f"  hasattr(result, 'X'): {hasattr(result, 'X')}")
                print(f"  hasattr(result, 'pop'): {hasattr(result, 'pop')}")
                
                # Check final population
                if hasattr(result, 'pop') and result.pop is not None:
                    print(f"\n  Final population size: {len(result.pop)}")
                    if len(result.pop) > 0:
                        pop_F = result.pop.get("F")
                        if pop_F is not None:
                            print(f"  Population objectives shape: {pop_F.shape}")
                            print(f"  Sample objectives (first 5 solutions):")
                            for i in range(min(5, len(pop_F))):
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
                
                if hasattr(result, 'F'):
                    print(f"\n  result.F (Pareto front):")
                    print(f"    is None: {result.F is None}")
                    if result.F is not None:
                        print(f"    shape: {result.F.shape}")
                        print(f"    length: {len(result.F)}")
                if hasattr(result, 'X'):
                    print(f"  result.X is None: {result.X is None}")
                    if result.X is not None:
                        print(f"    shape: {result.X.shape}")
        
        if result is not None and hasattr(result, 'F') and result.F is not None and len(result.F) > 0:
            if verbose:
                print(f"✓ Optimization completed: {len(result.F)} Pareto-optimal solutions found")
            
            # Prepare results
            optimization_results = {
                'scenario_params': scenario_params,
                'objectives': result.F,           # Objective values
                'decisions': result.X,            # Decision variables (restoration plans)
                'n_solutions': len(result.F),
                'problem_info': {
                    'n_pixels': initial_conditions['n_pixels']
                },
                'algorithm_info': {
                    'pop_size': pop_size,
                    'n_generations': n_generations,
                    'timestamp': datetime.now().isoformat()
                },
                'initial_conditions': initial_conditions
            }
            
            # Save results if requested
            if save_results:
                save_scenario_results(optimization_results, verbose=verbose)
            
            return optimization_results
            
        else:
            if verbose:
                print("✗ Optimization failed - no solutions found")
            return None
            
    except Exception as e:
        if verbose:
            print(f"✗ Error during optimization: {e}")
        return None

def run_all_scenarios_optimization(initial_conditions, pop_size=50, n_generations=100, 
                                 save_results=True, verbose=True):
    """
    Run optimization for all scenario combinations.
    
    Args:
        initial_conditions: Initial objective conditions
        pop_size: Population size for NSGA-II
        n_generations: Number of optimization generations
        save_results: Whether to save results to files
        verbose: Print progress information
        
    Returns:
        dict: Combined results from all scenarios
    """
    scenario_combinations = get_scenario_combinations()
    
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
        
        scenario_results = run_single_scenario_optimization(
            initial_conditions=initial_conditions,
            scenario_params=scenario_params,
            pop_size=pop_size,
            n_generations=n_generations,
            save_results=save_results,
            verbose=verbose
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
    combined_results = {
        'scenarios': all_results,
        'n_scenarios_run': len(all_results),
        'n_scenarios_total': len(scenario_combinations),
        'scenario_parameters': define_scenario_parameters(),
        'algorithm_info': {
            'pop_size': pop_size,
            'n_generations': n_generations,
            'timestamp': datetime.now().isoformat()
        },
        'initial_conditions': initial_conditions
    }
    
    # Save combined results
    if save_results and all_results:
        save_combined_results(combined_results, verbose=verbose)
    
    if verbose:
        print(f"\n=== ALL SCENARIOS COMPLETE ===")
        print(f"Successfully completed: {len(all_results)}/{len(scenario_combinations)} scenarios")
    
    return combined_results

# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_scenario_results(results, output_dir=".", verbose=True):
    """
    Save single scenario optimization results to files.
    
    Args:
        results: Results dictionary from single scenario optimization
        output_dir: Directory to save files
        verbose: Print save status
        
    Returns:
        tuple: (pickle_filename, summary_filename)
    """
    if verbose:
        print("\nSaving scenario results...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename with scenario parameters
    scenario_str = '_'.join([f"{k}{v}" for k, v in results['scenario_params'].items()])
    
    # Save complete results (pickle format)
    results_filename = os.path.join(output_dir, f"results_{timestamp}.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary (JSON format)
    summary_filename = os.path.join(output_dir, f"summary_{timestamp}.json")
    
    objectives = results['objectives']
    
    # Get objective names from the problem
    problem_temp = RestorationProblem(results['initial_conditions'], results['scenario_params'])
    objective_names = problem_temp.objective_names
    
    summary = {
        'scenario_params': results['scenario_params'],
        'optimization_info': {
            'n_solutions': results['n_solutions'],
            'n_pixels': results['problem_info']['n_pixels'],
            'timestamp': results['algorithm_info']['timestamp'],
            'objectives_used': objective_names
        },
        'objective_statistics': {}
    }
    
    # Add statistics for each objective that was used
    for i, obj_name in enumerate(objective_names):
        summary['objective_statistics'][obj_name] = {
            'min': float(objectives[:, i].min()),
            'max': float(objectives[:, i].max()),
            'mean': float(objectives[:, i].mean())
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"✓ Complete results saved to: {results_filename}")
        print(f"✓ Summary saved to: {summary_filename}")
    
    return results_filename, summary_filename

def save_combined_results(combined_results, output_dir=".", verbose=True):
    """
    Save combined multi-scenario results.
    
    Args:
        combined_results: Combined results from all scenarios
        output_dir: Directory to save files
        verbose: Print save status
    """
    if verbose:
        print(f"\nSaving combined scenario results...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save complete combined results
    results_filename = os.path.join(output_dir, f"restoration_all_scenarios_{timestamp}.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump(combined_results, f)
    
    # Create combined summary
    summary_filename = os.path.join(output_dir, f"restoration_all_scenarios_summary_{timestamp}.json")
    
    summary = {
        'combined_info': {
            'n_scenarios_run': combined_results['n_scenarios_run'],
            'n_scenarios_total': combined_results['n_scenarios_total'],
            'scenario_parameters': combined_results['scenario_parameters'],
            'timestamp': combined_results['algorithm_info']['timestamp']
        },
        'scenarios': {}
    }
    
    # Add summary for each scenario
    for scenario_id, scenario_results in combined_results['scenarios'].items():
        objectives = scenario_results['objectives']
        
        # Get objective names from this scenario
        problem_temp = RestorationProblem(scenario_results['initial_conditions'], scenario_results['scenario_params'])
        objective_names = problem_temp.objective_names
        
        objective_ranges = {}
        for i, obj_name in enumerate(objective_names):
            objective_ranges[obj_name] = [float(objectives[:, i].min()), float(objectives[:, i].max())]
        
        summary['scenarios'][f'scenario_{scenario_id}'] = {
            'params': scenario_results['scenario_params'],
            'n_solutions': scenario_results['n_solutions'],
            'objectives_used': objective_names,
            'objective_ranges': objective_ranges
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"✓ Combined results saved to: {results_filename}")
        print(f"✓ Combined summary saved to: {summary_filename}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main(workspace_dir=".", scenario='all', objectives=None, pop_size=50, n_generations=100, 
         save_results=True, verbose=True):
    """
    Main execution function for restoration optimization.
    
    Args:
        workspace_dir: Directory containing input data
        scenario: Scenario to run ('all' for all scenarios, or integer index for specific scenario)
        objectives: List of objectives to use (e.g., ['abiotic', 'biotic', 'landscape'])
                   Available: 'abiotic', 'biotic', 'landscape', 'cost'
                   If None, uses all available objectives
        pop_size: Population size for optimization
        n_generations: Number of optimization generations
        save_results: Whether to save results
        verbose: Print progress information
        
    Returns:
        dict: Optimization results
    """
    
    # Step 1: Load initial conditions
    if verbose:
        print("=== RESTORATION OPTIMIZATION WORKFLOW ===")
        if objectives is not None:
            print(f"Using objectives: {objectives}")
    
    initial_conditions = load_initial_conditions(workspace_dir, objectives=objectives)
    
    # Step 2: Display available scenarios
    if verbose:
        scenario_combinations = get_scenario_combinations()
        print(f"\nAvailable scenarios ({len(scenario_combinations)} total):")
        #for i, params in enumerate(scenario_combinations):
        #    print(f"  Scenario {i}: {params}")
    
    # Step 3: Run optimization based on scenario selection
    if scenario == 'all':
        # Run all scenarios
        results = run_all_scenarios_optimization(
            initial_conditions=initial_conditions,
            pop_size=pop_size,
            n_generations=n_generations,
            save_results=save_results,
            verbose=verbose
        )
    else:
        # Run single scenario by index
        try:
            scenario_params = get_scenario_by_index(scenario)
            results = run_single_scenario_optimization(
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
    
    if results is not None and verbose:
        print(f"\n=== OPTIMIZATION COMPLETE ===")
        
        if scenario == 'all':
            print(f"Completed {results['n_scenarios_run']}/{results['n_scenarios_total']} scenarios")
            for scenario_id, scenario_results in results['scenarios'].items():
                params = scenario_results['scenario_params']
                n_solutions = scenario_results['n_solutions']
                print(f"  Scenario {scenario_id} ({params}): {n_solutions} solutions")
        else:
            print(f"Found {results['n_solutions']} Pareto-optimal restoration plans")
            print(f"Scenario parameters: {results['scenario_params']}")
            
            objectives = results['objectives']
            print(f"\nObjective ranges:")
            
            # Get objective names from the problem
            problem_temp = RestorationProblem(results['initial_conditions'], results['scenario_params'])
            obj_display_names = {
                'abiotic_anomaly': 'Abiotic anomaly',
                'biotic_anomaly': 'Biotic anomaly', 
                'landscape_anomaly': 'Landscape anomaly',
                'implementation_cost': 'Implementation cost'
            }
            
            for i, obj_name in enumerate(problem_temp.objective_names):
                display_name = obj_display_names.get(obj_name, obj_name)
                print(f"  {display_name}: {objectives[:, i].min():.2f} - {objectives[:, i].max():.2f}")
    
    return results

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Example usage:

    
    # Or run with specific objectives only:
    results = main(
         workspace_dir=".",
         scenario=0,
         objectives=['abiotic', 'biotic', 'landscape'],
         pop_size=50,
         n_generations=100,
         save_results=True,
         verbose=True
     )
    

    
    if results is not None:
        print("\n✓ Optimization completed successfully!")
        print("Check the generated .pkl and .json files for detailed results.")
    else:
        print("\n✗ Optimization failed.")