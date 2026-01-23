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

# Import spatial operations from separate module
from spatial_operations import apply_burden_sharing, apply_spatial_clustering, _enforce_exact_pixel_count
from spatial_operations import ClusteringRepair, BurdenSharingRepair, CombinedRepair

# =============================================================================
# ECOSYSTEM DEFINITIONS AND LULC HANDLING
# =============================================================================

# Define ecosystem types and their corresponding LULC codes
ECOSYSTEM_TYPES = {
    'forest': [12, 13],
    'agricultural': [15], 
    'grassland': [16, 17]
}
#focal_classes = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
#    52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]

def load_lulc_raster(workspace_dir=None, lulc_path=None, region='CH', 
                     target_bounds=None, target_shape=None, target_transform=None, target_crs=None):
    """
    Load LULC raster for ecosystem masking.
    
    Args:
        workspace_dir: Directory to look for LULC file (optional)
        lulc_path: Explicit path to LULC file (optional)
        region: Region to crop LULC to ('Bern', 'CH', etc.). 'CH' loads full extent.
        target_bounds: Target bounds to crop/resample LULC to match reference data
        target_shape: Target shape to resample LULC to
        target_transform: Target transform for the output LULC
        target_crs: Target CRS for the output LULC
        
    Returns:
        numpy.ndarray: LULC raster data (processed to match target specifications)
        dict: Rasterio metadata (crs, transform, etc.)
    """
    # Default LULC paths to try
    default_paths = [
        "W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif",
        "LULC_2018_agg.tif"  # Local file in workspace
    ]
    
    if lulc_path:
        paths_to_try = [lulc_path]
    elif workspace_dir:
        paths_to_try = [os.path.join(workspace_dir, "LULC_2018_agg.tif")] + default_paths
    else:
        paths_to_try = default_paths
    
    # Load the LULC raster
    lulc_data = None
    lulc_meta = None
    original_path = None
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with rio.open(path) as src:
                    lulc_data = src.read(1)
                    lulc_meta = {
                        'crs': src.crs,
                        'transform': src.transform,
                        'shape': lulc_data.shape,
                        'bounds': src.bounds
                    }
                    original_path = path
                print(f"✓ Loaded LULC raster: {path}")
                print(f"  Original shape: {lulc_data.shape}, unique values: {len(np.unique(lulc_data[~np.isnan(lulc_data)]))}")
                break
            except Exception as e:
                print(f"✗ Error loading LULC from {path}: {e}")
                continue
    
    if lulc_data is None:
        raise FileNotFoundError(f"LULC raster not found. Tried: {paths_to_try}")
    
    # If target parameters are provided, crop/resample LULC to match reference data
    if target_bounds is not None and target_shape is not None:
        try:
            from rasterio.warp import reproject, Resampling
            from rasterio.windows import from_bounds
            import rasterio.mask
            
            print(f"🔄 Processing LULC to match reference data extent...")
            
            with rio.open(original_path) as src:
                # Check if we need to reproject
                if target_crs and src.crs != target_crs:
                    print(f"  Reprojecting from {src.crs} to {target_crs}")
                    
                    # Create output array with target specifications
                    processed_data = np.empty(target_shape, dtype=src.dtypes[0])
                    
                    # Reproject to match target
                    reproject(
                        source=rio.band(src, 1),
                        destination=processed_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest  # Use nearest neighbor for categorical data
                    )
                    
                else:
                    # Same CRS - just crop to bounds
                    try:
                        # Calculate window for the target bounds
                        window = from_bounds(*target_bounds, src.transform)
                        
                        # Read the windowed data
                        processed_data = src.read(1, window=window)
                        
                        # If the windowed data doesn't match target shape, resample
                        if processed_data.shape != target_shape:
                            print(f"  Resampling from {processed_data.shape} to {target_shape}")
                            
                            # Create temporary in-memory dataset for resampling
                            temp_transform = rio.windows.transform(window, src.transform)
                            temp_data = np.empty(target_shape, dtype=src.dtypes[0])
                            
                            reproject(
                                source=processed_data,
                                destination=temp_data,
                                src_transform=temp_transform,
                                src_crs=src.crs,
                                dst_transform=target_transform,
                                dst_crs=target_crs or src.crs,
                                resampling=Resampling.nearest
                            )
                            
                            processed_data = temp_data
                            
                    except Exception as window_error:
                        print(f"  Window-based cropping failed: {window_error}")
                        print(f"  Falling back to full reprojection...")
                        
                        # Fallback: reproject the entire raster
                        processed_data = np.empty(target_shape, dtype=src.dtypes[0])
                        
                        reproject(
                            source=rio.band(src, 1),
                            destination=processed_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=target_crs or src.crs,
                            resampling=Resampling.nearest
                        )
            
            # Update metadata to match target
            lulc_meta.update({
                'crs': target_crs or lulc_meta['crs'],
                'transform': target_transform,
                'shape': processed_data.shape,
                'bounds': target_bounds
            })
            
            print(f"✓ Processed LULC to match reference: {processed_data.shape}")
            print(f"  Processed unique values: {len(np.unique(processed_data[~np.isnan(processed_data)]))}")
            
            return processed_data, lulc_meta
            
        except Exception as e:
            print(f"Warning: Failed to process LULC to match reference data: {e}")
            print("Using original LULC - ecosystem masking may not work correctly")
            return lulc_data, lulc_meta
    
    else:
        print(f"✓ Using original LULC extent (no target specified)")
        return lulc_data, lulc_meta

def create_ecosystem_mask(lulc_data, ecosystem_type):
    """
    Create a boolean mask for a specific ecosystem type.
    
    Args:
        lulc_data: LULC raster data
        ecosystem_type: Type of ecosystem ('forest', 'agricultural', 'grassland', or 'all')
        
    Returns:
        numpy.ndarray: Boolean mask (True for pixels belonging to the ecosystem)
    """
    if ecosystem_type == 'all':
        # Include all defined ecosystem types
        all_codes = []
        for codes in ECOSYSTEM_TYPES.values():
            all_codes.extend(codes)
        mask = np.isin(lulc_data, all_codes)
    elif ecosystem_type in ECOSYSTEM_TYPES:
        lulc_codes = ECOSYSTEM_TYPES[ecosystem_type]
        mask = np.isin(lulc_data, lulc_codes)
    else:
        raise ValueError(f"Unknown ecosystem type: {ecosystem_type}. Available: {list(ECOSYSTEM_TYPES.keys()) + ['all']}")
    
    print(f"✓ Created {ecosystem_type} ecosystem mask: {np.sum(mask)}/{mask.size} pixels ({100*np.sum(mask)/mask.size:.1f}%)")
    return mask

# =============================================================================
# WEIGHTING FUNCTIONS
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

# PATCH 1 (add near your other helpers, e.g. close to anomaly_improvement_weight)

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
        'landscape_effect': [0.01],#(0.005, 0.02), # Reduction in landscape anomaly (min, max)
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
# DATA PREPARATION
# =============================================================================

def get_region_reference(region='Bern'):
    """
    Get the reference raster information for a specific region.
    This defines the expected CRS, transform, shape, and bounds for data validation.
    
    Args:
        region: Region name ('Bern' or 'CH')
        
    Returns:
        dict: Reference raster information or None to use first loaded raster
    """
    # Define region-specific reference information
    # These should match your expected data specifications for each region
    
    if region == 'Bern':
        # Bern canton extent and projection
        # Update these values based on your actual Bern data specifications
        return {
            'crs': 'EPSG:2056',  # Swiss coordinate system LV95
            'expected_bounds': (2556200, 1130600, 2677700, 1243700),  # Approximate Bern extent
            'expected_shape': None,  # Will be determined from first raster if None
            'description': 'Bern canton reference'
        }
    elif region == 'CH':
        # Switzerland-wide extent and projection  
        return {
            'crs': 'EPSG:2056',  # Swiss coordinate system LV95
            'expected_bounds': (2480000, 1070000, 2834000, 1300000),  # Approximate CH extent
            'expected_shape': (2300, 3600),  # Will be determined from first raster if None
            'description': 'Switzerland reference'
        }
    else:
        # Unknown region - use first loaded raster as reference
        return None

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
    
    try:
        if region == 'CH':
            # Use cantons for burden sharing across Switzerland
            if not os.path.exists(kanton_file):
                print(os.getcwd())
                print(os.path.abspath(kanton_file))
                print(f"Warning: Kanton shapefile not found at {kanton_file}")
                return None
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

def load_initial_conditions(workspace_dir, objectives=None, region='Bern', ecosystem='all', 
                            sample_fraction=None, sample_seed=42, lulc_path=None):
    """
    Args:
        workspace_dir: Directory containing input data files (.tif)
        objectives: List of objectives to load (e.g., ['abiotic', 'biotic', 'landscape', 'cost'])
                   If None, loads all available objectives
        region: Region to optimize for ('Bern' or 'CH'), passed to load_admin_regions
        ecosystem: Ecosystem type to focus on ('forest', 'agricultural', 'grassland', or 'all')
                  Data will be masked to only include pixels from this ecosystem type
        sample_fraction: Fraction of eligible pixels to use (e.g., 0.25 for 25% sample)
                        If None, uses all eligible pixels
        sample_seed: Random seed for reproducible sampling (default: 42)
        lulc_path: Path to LULC raster file. If None, uses default paths
    Returns:
        dict: Initial conditions for specified objectives and ecosystem
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
    
    # Get region-specific reference information
    region_ref = get_region_reference(region)
    
    # Load all data files and track NaN locations
    nan_masks = {}  # Store NaN locations before replacement
    ref = None
    first_raster = True
    
    for objective, file_path in data_files.items():
        with rio.open(file_path) as src:
            data = src.read(1)
            
            if first_raster:
                # Initialize reference on first raster
                ref = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'shape': data.shape,
                    'bounds': src.bounds,
                }
                
                # Apply region-specific validation if available
                if region_ref is not None:
                    # Check if the loaded data matches region expectations
                    if str(src.crs) != region_ref['crs']:
                        print(f"Warning: Raster CRS {src.crs} doesn't match expected {region_ref['crs']} for region {region}")
                    
                    if region_ref['expected_bounds'] is not None:
                        expected = region_ref['expected_bounds']
                        actual = src.bounds
                        if not (abs(actual.left - expected[0]) < 1000 and 
                                abs(actual.bottom - expected[1]) < 1000 and
                                abs(actual.right - expected[2]) < 1000 and
                                abs(actual.top - expected[3]) < 1000):
                            print(f"Warning: Raster bounds {actual} significantly differ from expected {expected} for region {region}")
                    
                    print(f"✓ Using {region_ref['description']} as validation reference")
                else:
                    print(f"✓ Using first raster as validation reference for region {region}")
                
                initial_conditions['crs'] = ref['crs']
                initial_conditions['transform'] = ref['transform']
                initial_conditions['shape'] = ref['shape']
                initial_conditions['region'] = region
                first_raster = False
            else:
                if src.crs != ref['crs']:
                    raise ValueError(
                        f"CRS mismatch for {objective}. "
                        f"Expected {ref['crs']}, got {src.crs} ({file_path})"
                    )
                if src.transform != ref['transform']:
                    raise ValueError(
                        f"Transform mismatch for {objective}. "
                        f"Expected {ref['transform']}, got {src.transform} ({file_path})"
                    )
                if data.shape != ref['shape']:
                    raise ValueError(
                        f"Shape mismatch for {objective}. "
                        f"Expected {ref['shape']}, got {data.shape} ({file_path})"
                    )
                if src.bounds != ref['bounds']:
                    raise ValueError(
                        f"Extent mismatch for {objective}. "
                        f"Expected {ref['bounds']}, got {src.bounds} ({file_path})"
                    )
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
    
    # Load LULC raster and create ecosystem mask
    print(f"\\n--- Loading LULC data for ecosystem: {ecosystem} ---")
    try:
        # Load LULC and crop/resample to match the reference data
        lulc_data, lulc_meta = load_lulc_raster(workspace_dir, lulc_path, region, 
                                               target_bounds=ref['bounds'], 
                                               target_shape=ref['shape'],
                                               target_transform=ref['transform'],
                                               target_crs=ref['crs'])
        
        # Validate LULC raster compatibility with reference
        if lulc_data.shape != ref['shape']:
            raise ValueError(
                f"LULC shape mismatch after processing. Expected {ref['shape']}, got {lulc_data.shape}"
            )
        
        # Create ecosystem mask
        ecosystem_mask = create_ecosystem_mask(lulc_data, ecosystem)
        
        # Store LULC info
        initial_conditions['lulc_data'] = lulc_data
        initial_conditions['lulc_meta'] = lulc_meta
        initial_conditions['ecosystem'] = ecosystem
        initial_conditions['ecosystem_mask'] = ecosystem_mask
        
    except Exception as e:
        print(f"✗ Error loading LULC data: {e}")
        if ecosystem != 'all':
            print("Falling back to 'all' ecosystem (no masking)")
            ecosystem = 'all'
            ecosystem_mask = np.ones(ref['shape'], dtype=bool)
        else:
            raise e

    # Create eligibility mask - exclude pixels that were NaN in ANY objective
    shape = initial_conditions['shape']
    eligible_mask = np.ones(shape, dtype=bool)
    
    for obj_name, nan_mask in nan_masks.items():
        if np.any(nan_mask):
            eligible_mask = eligible_mask & ~nan_mask
            nan_excluded = np.sum(nan_mask)
            print(f"  Masking {nan_excluded} NaN pixels from {obj_name}")

    # Apply ecosystem mask to eligible pixels
    if 'ecosystem_mask' in initial_conditions:
        pre_ecosystem_count = np.sum(eligible_mask)
        eligible_mask = eligible_mask & initial_conditions['ecosystem_mask']
        post_ecosystem_count = np.sum(eligible_mask)
        ecosystem_excluded = pre_ecosystem_count - post_ecosystem_count
        print(f"  Ecosystem masking excluded {ecosystem_excluded} pixels")
        print(f"  Final eligible pixels for {ecosystem}: {post_ecosystem_count}")

    # You might want to exclude certain areas, e.g.:
    # eligible_mask = (initial_conditions['implementation_cost'] < 8000) & \\
    #                 (initial_conditions['abiotic_anomaly'] > 0.1)
    
    initial_conditions['eligible_mask'] = eligible_mask
    eligible_indices = np.where(eligible_mask.flatten())[0]
    
    # Apply spatial sampling if requested
    if sample_fraction is not None and 0 < sample_fraction < 1:
        np.random.seed(sample_seed)  # For reproducible sampling
        n_total_eligible = len(eligible_indices)
        n_sample_target = int(sample_fraction * n_total_eligible)
        
        if n_sample_target > 0:
            # Create spatially continuous sample using rectangular region
            rows, cols = shape
            
            # Convert eligible indices to 2D coordinates
            eligible_rows, eligible_cols = np.unravel_index(eligible_indices, shape)
            min_row, max_row = np.min(eligible_rows), np.max(eligible_rows)
            min_col, max_col = np.min(eligible_cols), np.max(eligible_cols)
            
            # Calculate rectangle dimensions to achieve target sample size
            eligible_area = (max_row - min_row + 1) * (max_col - min_col + 1)
            scale_factor = np.sqrt(sample_fraction * eligible_area / ((max_row - min_row + 1) * (max_col - min_col + 1)))
            
            rect_height = max(1, int(scale_factor * (max_row - min_row + 1)))
            rect_width = max(1, int(scale_factor * (max_col - min_col + 1)))
            
            # Randomly position rectangle within eligible bounds
            max_start_row = max(min_row, max_row - rect_height + 1)
            max_start_col = max(min_col, max_col - rect_width + 1)
            
            start_row = np.random.randint(min_row, max_start_row + 1)
            start_col = np.random.randint(min_col, max_start_col + 1)
            
            end_row = min(start_row + rect_height, max_row + 1)
            end_col = min(start_col + rect_width, max_col + 1)
            
            # Create mask for rectangular region
            sample_mask = np.zeros(shape, dtype=bool)
            sample_mask[start_row:end_row, start_col:end_col] = True
            
            # Combine with eligible mask to get spatially continuous sample
            combined_mask = eligible_mask & sample_mask
            sampled_indices = np.where(combined_mask.flatten())[0]
            
            if len(sampled_indices) > 0:
                eligible_indices = sampled_indices
                eligible_mask = combined_mask
                
                print(f"✓ Applied spatially continuous sampling:")
                print(f"  Rectangle: ({start_row}:{end_row}, {start_col}:{end_col})")
                print(f"  Sample: {len(sampled_indices)}/{n_total_eligible} pixels ({len(sampled_indices)/n_total_eligible*100:.1f}%)")
            else:
                print(f"✗ Warning: No eligible pixels in sample region, using all eligible pixels")
        else:
            print(f"✗ Warning: Sample size too small ({n_sample_target} pixels), using all eligible pixels")
    
    initial_conditions['eligible_mask'] = eligible_mask
    initial_conditions['eligible_indices'] = eligible_indices
    initial_conditions['n_pixels'] = len(eligible_indices)
    initial_conditions['sample_info'] = {
        'sample_fraction': sample_fraction,
        'sample_seed': sample_seed,
        'is_sampled': sample_fraction is not None and 0 < sample_fraction < 1
    }
    
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
            'landscape_effect': 0.01,     # Improvement for landscape anomaly
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
    landscape_effect = effect_params.get('landscape_effect', 0.01)
    effect_params['abiotic_improvement'] = abiotic_effect
    effect_params['biotic_improvement'] = biotic_effect
    effect_params['landscape_improvement'] = landscape_effect
    
    # Ensure neighbor_radius is integer for array indexing
    effect_params['neighbor_radius'] = int(round(effect_params['neighbor_radius']))
    
    shape = initial_conditions['shape']
    eligible_mask = initial_conditions['eligible_mask']
    eligible_indices = initial_conditions['eligible_indices']
    
    # Create 2D masks from 1D decision variables
    restoration_mask_2d = np.zeros(shape, dtype=bool)
    conversion_mask_2d = np.zeros(shape, dtype=bool)
    
    if np.any(restore_vars):
        # Convert eligible indices with restoration back to 2D coordinates
        restoration_eligible_indices = eligible_indices[restore_vars == 1]
        rows, cols = np.divmod(restoration_eligible_indices, shape[1])
        restoration_mask_2d[rows, cols] = True
    
    if np.any(convert_vars):
        # Convert eligible indices with conversion back to 2D coordinates
        conversion_eligible_indices = eligible_indices[convert_vars == 1]
        rows, cols = np.divmod(conversion_eligible_indices, shape[1])
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
            action_mask = conversion_mask_2d
            improvement_key = f'{objective.split("_")[0]}_improvement'
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
# CUSTOM SAMPLING AND REPAIR OPERATORS
# =============================================================================

class CustomBinaryRandomSampling(Sampling):
    """
    Custom binary random sampling that properly handles our expanded decision vector.
    Only generates restoration decisions, leaves conversion decisions as zeros.
    """
    
    def __init__(self, initial_conditions, max_restored_pixels):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.max_restored_pixels = max_restored_pixels
    
    def _do(self, problem, n_samples, **kwargs):
        n_decision_vars = problem.n_var  # 2*n_pixels (restore + convert)
        n_pixels = problem.n_pixels      # actual number of eligible pixels
        X = np.zeros((n_samples, n_decision_vars), dtype=int)
        
        for i in range(n_samples):
            x = np.zeros(n_decision_vars, dtype=int)
            
            # Only work with restoration decisions (first n_pixels elements)
            n_restore = self.max_restored_pixels
            if n_restore > 0:
                # Use permutation to avoid int32 overflow with large pixel indices
                restore_indices = np.random.permutation(n_pixels)[:n_restore]
                x[restore_indices] = 1
            
            X[i] = x
        
        return X

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
        n_decision_vars = problem.n_var  # 2*n_pixels (restore + convert)
        n_pixels = problem.n_pixels      # actual number of eligible pixels
        X = np.zeros((n_samples, n_decision_vars), dtype=int)
        
        for i in range(n_samples):
            # Generate a random solution with target number of pixels
            x = np.zeros(n_decision_vars, dtype=int)
            
            # Only work with restoration decisions (first n_pixels elements)
            n_restore = self.max_restored_pixels
            if n_restore > 0:
                # Use permutation to avoid int32 overflow with large pixel indices
                restore_indices = np.random.permutation(n_pixels)[:n_restore]
                x[restore_indices] = 1
                
                # Apply spatial clustering to generate clustered pattern
                if self.clustering_strength > 0:
                    # Apply clustering only to restoration decisions
                    x_restore = apply_spatial_clustering(
                        x[:n_pixels], 
                        self.initial_conditions, 
                        self.clustering_strength,
                        exact_count=self.max_restored_pixels
                    )
                    x[:n_pixels] = x_restore
            
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
        print(f"DEBUG: BurdenSharingSampling._do called with n_samples={n_samples}, n_var={problem.n_var}, max_restored_pixels={self.max_restored_pixels}")
        n_decision_vars = problem.n_var  # 2*n_pixels (restore + convert)
        n_pixels = problem.n_pixels      # actual number of eligible pixels
        X = np.zeros((n_samples, n_decision_vars), dtype=int)
        
        for i in range(n_samples):
            print(f"DEBUG: BurdenSharingSampling sample {i}/{n_samples}")
            # Generate a random solution with target number of pixels
            x = np.zeros(n_decision_vars, dtype=int)
            
            try:
                print(f"DEBUG: Generating random restore count (max={self.max_restored_pixels})")
                n_restore = np.random.randint(0, self.max_restored_pixels + 1)
                print(f"DEBUG: Generated n_restore={n_restore}")
                
                if n_restore > 0:
                    print(f"DEBUG: Using permutation to select {n_restore} pixels from {n_pixels}")
                    # Use permutation to avoid int32 overflow with large pixel indices
                    restore_indices = np.random.permutation(n_pixels)[:n_restore]
                    print(f"DEBUG: Generated restore_indices, setting pixels")
                    # Only set restoration decisions (first n_pixels elements)
                    x[:n_pixels][restore_indices] = 1
                    print(f"DEBUG: Set pixels, calling apply_burden_sharing")
                    
                    # Apply burden sharing to distribute equally across regions (only to restoration part)
                    seed = np.random.randint(0, 2**31 - 1)  # Use int32 safe range
                    x_restore = apply_burden_sharing(x[:n_pixels], self.initial_conditions, seed=seed, exact_count=self.max_restored_pixels)
                    x[:n_pixels] = x_restore  # Update restoration part of decision vector
                    print(f"DEBUG: apply_burden_sharing completed")
                
            except Exception as e:
                print(f"DEBUG: Error in BurdenSharingSampling sample {i}: {e}")
                import traceback
                traceback.print_exc()
                # Return zeros for this sample if there's an error
                x = np.zeros(n_decision_vars, dtype=int)
            
            X[i] = x
        
        print(f"DEBUG: BurdenSharingSampling._do completed")
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
        n_decision_vars = problem.n_var  # 2*n_pixels (restore + convert)
        n_pixels = problem.n_pixels      # actual number of eligible pixels
        X = np.zeros((n_samples, n_decision_vars), dtype=int)
        
        for i in range(n_samples):
            x = np.zeros(n_decision_vars, dtype=int)
            
            # Only work with restoration decisions (first n_pixels elements)
            x_restore = x[:n_pixels]
            
            n_restore = self.max_restored_pixels  # Use exact count for initial sampling
            if n_restore > 0:
                # Use permutation to avoid int32 overflow with large pixel indices
                restore_indices = np.random.permutation(n_pixels)[:n_restore]
                x_restore[restore_indices] = 1
                
                # Apply burden sharing first (only to restoration decisions)
                seed = np.random.randint(0, 2**31 - 1)  # Use int32 safe range
                x_restore = apply_burden_sharing(x_restore, self.initial_conditions, seed=seed, exact_count=self.max_restored_pixels)
                x[:n_pixels] = x_restore  # Update the full decision vector
                
                # Then apply spatial clustering
                if self.clustering_strength > 0:
                    x_restore_clustered = apply_spatial_clustering(
                        x[:n_pixels], 
                        self.initial_conditions, 
                        self.clustering_strength,
                        exact_count=self.max_restored_pixels
                    )
                    x[:n_pixels] = x_restore_clustered
            
            X[i] = x
        
        return X

class ScoreCountRepair(Repair):
    """
    Enforces EXACT number of restored pixels, but chooses which pixels to add or remove
    based on a per pixel score, rather than at random.
    """
    def __init__(self, max_restored_pixels, scores):
        super().__init__()
        self.max_restored_pixels = int(max_restored_pixels)
        self.scores = np.asarray(scores, dtype=np.float64)

    def _do(self, problem, X, **kwargs):
        Xr = np.zeros_like(X)

        for i in range(len(X)):
            x = X[i].copy()
            # Work with both restoration and conversion decisions
            n_pixels = problem.n_pixels
            x_restore = x[:n_pixels]
            x_convert = x[n_pixels:]
            
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
            Xr[i] = x

        return Xr

class ExactCountRepair(Repair):
    """
    Repair operator that enforces exact count constraint after genetic operations.
    """
    
    def __init__(self, max_restored_pixels, seed=None):
        super().__init__()
        self.max_restored_pixels = max_restored_pixels
        self.rng = np.random.RandomState(seed)
    
    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(len(X)):
            x = X[i].copy()
            # Work with both restoration and conversion decisions  
            n_pixels = problem.n_pixels
            x_restore = x[:n_pixels]
            x_convert = x[n_pixels:]
            
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
                        # Could add conversion pixels here in future
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
                        # Could remove conversion pixels here in future
            
            # Update both restoration and conversion parts of the decision vector
            x[:n_pixels] = x_restore
            x[n_pixels:] = x_convert
            
            #n_changed = np.sum(x != X[i])
            #print("repair changed bits:", int(n_changed), "count before after:", int(current_total), int(self.max_restored_pixels))

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
            decision_vars[np.random.permutation(len(eligible_indices))[:n_restore]] = 1
        
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
        base_decision_vars[np.random.permutation(len(eligible_indices))[:n_restore]] = 1
    
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
    
    # Check if objectives can improve
    #print(f"\n4. OBJECTIVE IMPROVEMENT POTENTIAL:")
    
    if feasible_count > 0:
        print(f"   ✓ Feasible solutions exist, Objectives show improvement with restoration")
    else:
        print(f"   ✗ WARNING: No feasible solutions found in {n_samples} samples!")
        print(f"   → Check if max_restored_pixels constraint is too restrictive")
    
    # Check for common issues
    #print(f"\n5. POTENTIAL ISSUES:")
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
            mask = initial_conditions['eligible_mask']
            data1 = initial_conditions[anomaly_objs[i]][mask]
            data2 = initial_conditions[anomaly_objs[j]][mask]
            corr = np.corrcoef(data1, data2)[0, 1]
            print(f"   {anomaly_objs[i]} vs {anomaly_objs[j]}: {corr:.4f}")

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
        #print(f"         → Improvement adds {np.mean(after_restored - before_restored):.3f} on average")

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
            scenario_params: Dict with scenario parameters (e.g., max_restoration_fraction, effect_params)
        """
        self.initial_conditions = initial_conditions
        self.scenario_params = scenario_params
        
        # Extract effect parameters from scenario_params (excluding fixed spatial parameters)
        abiotic_effect = scenario_params.get('abiotic_effect', 0.01)
        biotic_effect = scenario_params.get('biotic_effect', 0.01)
        landscape_effect = scenario_params.get('landscape_effect', 0.01)
        self.effect_params = {
            'abiotic_effect': abiotic_effect,
            'biotic_effect': biotic_effect,
            'landscape_effect': landscape_effect,
            'abiotic_improvement': abiotic_effect,
            'biotic_improvement': biotic_effect,
            'landscape_improvement': landscape_effect,
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
        
        n_pixels = initial_conditions['n_pixels']
        max_restoration_fraction = scenario_params['max_restoration_fraction']
        self.max_restored_pixels = int(max_restoration_fraction * n_pixels)
        self.n_pixels = n_pixels  # Store for splitting decision vector
        
        # Binary decision variables: 0 = no action, 1 = action
        # First n_pixels elements = restoration decisions
        # Second n_pixels elements = conversion decisions (forced to 0 for now)
        super().__init__(
            n_var=2*n_pixels,        # Two decisions per eligible pixel: restore and convert
            n_obj=n_objectives,      # Variable number of objectives
            n_constr=1,              # Constraint on total restoration area
            xl=0,                    # Lower bound: no action
            xu=1,                    # Upper bound: action
            type_var=int             # Integer (binary) variables
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution (restoration plan).
        
        Args:
            x: Decision variables (binary array) - already clustered/burden-shared by sampling/repair
                First n_pixels elements: restoration decisions
                Second n_pixels elements: conversion decisions
            out: Output dictionary for objectives and constraints
        """
        # Split decision vector into restore and convert actions
        x_restore = x[:self.n_pixels]
        x_convert = x[self.n_pixels:] 
        
        # Force convert actions to zero for now (future development)
        x_convert[:] = 0
        
        # Use only restoration decisions for current logic
        n_restored = np.sum(x_restore)
        n_converted = np.sum(x_convert)
        n_total_actions = n_restored + n_converted
        
        # Apply restoration effects to both restoration and conversion decisions
        updated_conditions = restoration_effect(x_restore, x_convert, self.initial_conditions, self.effect_params)
        
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
                #vals = updated_conditions[obj_name][self.initial_conditions["eligible_mask"]]
                #pixels_above = np.sum(vals > 0.0)

                # smooth tie breaker, only counts how far above 0 you are
                #margin = np.sum(np.maximum(0.0, vals))

                # small weight so the count still dominates
                #obj_value = -pixels_above - 1e-6 * margin

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

def run_single_scenario_optimization(initial_conditions, scenario_params, pop_size=50, 
                                   n_generations=100, save_results=True, verbose=True, skip_diagnostics=False):
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
        
    Returns:
        dict: Optimization results
    """
    if verbose:
        print(f"\n=== SINGLE SCENARIO OPTIMIZATION ===")
        print(f"Scenario parameters: {scenario_params}")
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_generations}")
        #print(f"Max restoration: {scenario_params['max_restoration_fraction']*100:.1f}% of eligible area")
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
        
        # Only run diagnostics if not skipped
        if not skip_diagnostics:
            #print("\nRunning diagnostics to check optimization setup...")
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
        repair = CombinedRepair(initial_conditions, clustering_strength, problem.max_restored_pixels)
        if verbose:
            print(f"  Using combined burden-sharing + clustering ({clustering_strength})")
    elif burden_sharing == 'yes':
        sampling = BurdenSharingSampling(initial_conditions, problem.max_restored_pixels)
        repair = BurdenSharingRepair(initial_conditions, problem.max_restored_pixels)
        if verbose:
            print(f"  Using burden-sharing sampling/repair")
    elif clustering_strength > 0.0:
        sampling = ClusteredSampling(
            initial_conditions, 
            problem.max_restored_pixels, 
            clustering_strength
        )
        repair = ClusteringRepair(initial_conditions, clustering_strength, problem.max_restored_pixels)
        if verbose:
            print(f"  Using clustered sampling/repair (strength={clustering_strength})")
    else:
        sampling = CustomBinaryRandomSampling(initial_conditions, problem.max_restored_pixels)
        #repair = ExactCountRepair(problem.max_restored_pixels)
        scores = build_repair_scores(initial_conditions, scenario_params)
        repair = ScoreCountRepair(problem.max_restored_pixels, scores)
        if verbose:
            print(f"  Using custom binary random sampling with score-based repair")
    
    # Create optimization algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,                # Custom or standard sampling
        crossover=HUX(),                  # Half-uniform crossover
        mutation=BitflipMutation(prob=0.05),       # Bit-flip mutation
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
                #print(f"  hasattr(result, 'F'): {hasattr(result, 'F')}")
                #print(f"  hasattr(result, 'X'): {hasattr(result, 'X')}")
                #print(f"  hasattr(result, 'pop'): {hasattr(result, 'pop')}")
                
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
                
                #if hasattr(result, 'F'):
                    #print(f"\n  result.F (Pareto front):")
                    #print(f"    is None: {result.F is None}")
                    #if result.F is not None:
                        #print(f"    shape: {result.F.shape}")
                        #print(f"    length: {len(result.F)}")
                #if hasattr(result, 'X'):
                    #print(f"  result.X is None: {result.X is None}")
                    #if result.X is not None:
                        #print(f"    shape: {result.X.shape}")
        
        if result is not None and hasattr(result, 'F') and result.F is not None and len(result.F) > 0:
            if verbose:
                print(f"✓ Optimization completed: {len(result.F)} Pareto-optimal solutions found")
            
            # Prepare results
            # Create a copy of initial_conditions without the geodataframe to avoid large pickle files
            initial_conditions_filtered = initial_conditions.copy()
            if 'admin_data' in initial_conditions_filtered and initial_conditions_filtered['admin_data'] is not None:
                admin_data_filtered = initial_conditions_filtered['admin_data'].copy()
                # Remove the geodataframe but keep other admin data
                if 'gdf' in admin_data_filtered:
                    del admin_data_filtered['gdf']
                initial_conditions_filtered['admin_data'] = admin_data_filtered
            
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
                    'timestamp': datetime.now().isoformat()
                },
                'initial_conditions': initial_conditions_filtered
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
# RESULTS SAVING
# =============================================================================

def save_parameter_summary(output_dir=".", n_samples_per_param=3, random_seed=42, verbose=True):
    """
    Save a summary of parameter ranges and sampled values.
    
    Args:
        output_dir: Directory to save files
        n_samples_per_param: Number of samples per continuous parameter
        random_seed: Random seed used for sampling
        verbose: Print save status
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get parameter ranges and sampled values
    param_ranges = define_scenario_parameters()
    sampled_combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
    
    # Create parameter summary
    param_summary = {
        'sampling_info': {
            'n_samples_per_param': n_samples_per_param,
            'random_seed': random_seed,
            'total_scenarios': len(sampled_combinations),
            'timestamp': datetime.now().isoformat()
        },
        'parameter_ranges': {},
        'sampled_values': {},
        'scenarios': []
    }
    
    # Process parameter ranges
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            param_summary['parameter_ranges'][param_name] = {
                'type': 'continuous',
                'min': param_range[0],
                'max': param_range[1]
            }
        else:
            param_summary['parameter_ranges'][param_name] = {
                'type': 'categorical',
                'values': param_range
            }
    
    # Extract unique sampled values for each parameter
    for param_name in param_ranges.keys():
        values = [combo[param_name] for combo in sampled_combinations]
        unique_values = sorted(list(set(values)))
        param_summary['sampled_values'][param_name] = unique_values
    
    # Add all scenario combinations
    for i, combo in enumerate(sampled_combinations):
        param_summary['scenarios'].append({
            'scenario_id': i,
            'parameters': combo
        })
    
    # Save to JSON
    summary_filename = os.path.join(output_dir, f"parameter_summary_{timestamp}.json")
    with open(summary_filename, 'w') as f:
        json.dump(param_summary, f, indent=2)
    
    if verbose:
        print(f"✓ Parameter summary saved to: {summary_filename}")
        print(f"   Total scenarios: {len(sampled_combinations)}")
        print(f"   Continuous parameters: {sum(1 for r in param_ranges.values() if isinstance(r, tuple))}")
        print(f"   Categorical parameters: {sum(1 for r in param_ranges.values() if not isinstance(r, tuple))}")
    
    return summary_filename

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
    
    timestamp = datetime.now().strftime('%d%m')
    
    # Get ecosystem information for filename
    ecosystem = results.get('initial_conditions', {}).get('ecosystem', 'all')
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    # Save complete results (pickle format)
    x = 1
    while True:
        results_filename = os.path.join(
            output_dir,
            f"results{ecosystem_suffix}_{timestamp}_{x}.pkl"
        )
        if not os.path.exists(results_filename):
            break
        x += 1
    print(f"  Saving to: {results_filename}")
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary (JSON format)
    summary_filename = os.path.join(output_dir, f"summary{ecosystem_suffix}_{timestamp}.json")
    
    objectives = results['objectives']
    
    # Get objective names from the problem
    problem_temp = RestorationProblem(results['initial_conditions'], results['scenario_params'])
    objective_names = problem_temp.objective_names
    
    summary = {
        'ecosystem': ecosystem,
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
    
    timestamp = datetime.now().strftime('%d%m')
    
    # Get ecosystem information from first scenario (should be same for all)
    first_scenario_results = next(iter(combined_results['scenarios'].values()))
    ecosystem = first_scenario_results.get('initial_conditions', {}).get('ecosystem', 'all')
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    x = 1
    while True:
        results_filename = os.path.join(output_dir, f"results_{timestamp}_{x}.pkl")
        summary_filename = os.path.join(output_dir, f"results_sc_{timestamp}_{x}.json")
        if not os.path.exists(results_filename):
            break
        x += 1
    print(f"  Saving combined results to: {results_filename}")
    with open(results_filename, 'wb') as f:
        pickle.dump(combined_results, f)
    
    # Convert parameter ranges to JSON-serializable format
    param_ranges = combined_results.get('scenario_parameters', {})
    json_param_ranges = {}
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            json_param_ranges[param_name] = {'type': 'continuous', 'min': param_range[0], 'max': param_range[1]}
        else:
            json_param_ranges[param_name] = {'type': 'categorical', 'values': param_range}
    
    summary = {
        'ecosystem': ecosystem,
        'combined_info': {
            'n_scenarios_run': combined_results['n_scenarios_run'],
            'n_scenarios_total': combined_results['n_scenarios_total'],
            'parameter_ranges': json_param_ranges,
            'sampling_info': {
                'n_samples_per_param': combined_results.get('n_samples_per_param', 3),
                'random_seed': combined_results.get('random_seed', 42)
            },
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
    ECOSYSTEM_TO_RUN = "grassland" #'all'  # Change this to 'forest', 'agricultural', 'grassland', or 'all'
    
    # Available regions: 'Bern', 'CH'
    # This affects the reference raster used for data validation
    REGION = 'Bern'  # Change to 'CH' for Switzerland-wide optimization
    
    print(f"\\n=== RESTORATION OPTIMIZATION FOR {ECOSYSTEM_TO_RUN.upper()} ECOSYSTEM ===")
    print(f"Available ecosystem types: {list(ECOSYSTEM_TYPES.keys())} or 'all'")
    print(f"Using region: {REGION} (affects data validation reference)")
    
    # =========================================================================
    # IMPLEMENTATION 1: CUSTOM PARAMETERS
    # =========================================================================
    # Define your own parameter values and run optimization
    
    custom_scenario_params = {
         'max_restoration_fraction': 0.05,  # Restore 30% of eligible area
         'spatial_clustering': 0,         # High spatial clustering
         'burden_sharing': 'no',           # Equal sharing across regions
         'abiotic_effect': 0.01,      # Effect on abiotic anomaly
         'biotic_effect': 0.01,       # Effect on biotic anomaly
         'landscape_effect': 0.01,    # Effect on landscape anomaly
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
        verbose=True
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
        print("Check the generated .pkl and .json files for detailed results.")
    else:
        print("\n✗ Optimization failed.")