"""
Data Loading and Preprocessing Module
====================================

Handles all data loading, LULC processing, and initial conditions setup
for restoration optimization.

Functions:
- load_lulc_raster(): Load and process LULC raster data
- create_ecosystem_mask(): Create boolean masks for ecosystem types
- get_region_reference(): Get reference raster information for regions
- load_admin_regions(): Load administrative regions for burden sharing
- load_initial_conditions(): Load and prepare all initial data conditions
"""

import os
import numpy as np
import tempfile
import rasterio as rio
import geopandas as gpd
from spatial_operations import compute_sn_dens

# =============================================================================
# ECOSYSTEM DEFINITIONS AND CONSTANTS
# =============================================================================

# Define ecosystem types and their corresponding LULC codes (for ecosystem masking)
ECOSYSTEM_TYPES = {
    'forest': [12, 13],
    'agricultural': [15], 
    'grassland': [16, 17]
}

# Define focal classes for landscape density calculations (separate from ecosystem LULC codes)
# These should correspond to the landscape LULC dataset classes
FOCAL_CLASSES = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]

# =============================================================================
# LULC LOADING AND PROCESSING
# =============================================================================

def load_lulc_raster(workspace_dir=None, lulc_path=None, region='CH', 
                     target_bounds=None, target_shape=None, target_transform=None, target_crs=None,
                     lulc_type='ecosystem'):
    """
    Load LULC raster for ecosystem masking or landscape density calculations.
    
    Args:
        workspace_dir: Directory to look for LULC file (optional)
        lulc_path: Explicit path to LULC file (optional)
        region: Region to crop LULC to ('Bern', 'CH', etc.). 'CH' loads full extent.
        target_bounds: Target bounds to crop/resample LULC to match reference data
        target_shape: Target shape to resample LULC to
        target_transform: Target transform for the output LULC
        target_crs: Target CRS for the output LULC
        lulc_type: Type of LULC to load ('ecosystem' or 'landscape')
        
    Returns:
        numpy.ndarray: LULC raster data (processed to match target specifications)
        dict: Rasterio metadata (crs, transform, etc.)
        str: Original file path used
    """
    # Default LULC paths to try based on type
    if lulc_type == 'ecosystem':
        default_filenames = ["LULC_2018_agg.tif"]
        default_paths = [
            "W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018.tif",
            "W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif"
        ]
    elif lulc_type == 'landscape':
        default_filenames = ["AS72_2018.tif"]
        default_paths = [
            "W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/AS72_2018.tif"
        ]
    else:
        raise ValueError(f"Unknown lulc_type: {lulc_type}. Use 'ecosystem' or 'landscape'")
    
    # Add local workspace files
    if workspace_dir:
        default_paths = [os.path.join(workspace_dir, fn) for fn in default_filenames] + default_paths
    else:
        default_paths = default_filenames + default_paths
    
    if lulc_path:
        paths_to_try = [lulc_path]
    elif workspace_dir:
        paths_to_try = [os.path.join(workspace_dir, "AS72_2018.tif")] + default_paths
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
                print(f"✓ Loaded {lulc_type} LULC raster: {path}")
                break
            except Exception as e:
                print(f"✗ Error loading LULC from {path}: {e}")
                continue
    
    if lulc_data is None:
        raise FileNotFoundError(f"{lulc_type.title()} LULC raster not found. Tried: {paths_to_try}")
    
    # If target parameters are provided, crop/resample LULC to match reference data
    if target_bounds is not None and target_shape is not None:
        try:
            from rasterio.warp import reproject, Resampling
            from rasterio.windows import from_bounds
            
            #print(f"🔄 Processing LULC to match reference data extent...")
            
            with rio.open(original_path) as src:
                # Check if we need to reproject
                if target_crs and src.crs != target_crs:
                    #print(f"  Reprojecting from {src.crs} to {target_crs}")
                    
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
                            #print(f"  Resampling from {processed_data.shape} to {target_shape}")
                            
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
                       #print(f"  Window-based cropping failed: {window_error}")
                       #print(f"  Falling back to full reprojection...")
                        
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
            
            #print(f"✓ Processed LULC to match reference: {processed_data.shape}")
            #print(f"  Processed unique values: {len(np.unique(processed_data[~np.isnan(processed_data)]))}")
            
            return processed_data, lulc_meta, original_path
            
        except Exception as e:
            #print(f"Warning: Failed to process LULC to match reference data: {e}")
            #print("Using original LULC - ecosystem masking may not work correctly")
            return lulc_data, lulc_meta, original_path
    
    else:
        #print(f"✓ Using original LULC extent (no target specified)")
        return lulc_data, lulc_meta, original_path


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
    
    #print(f"✓ Created {ecosystem_type} ecosystem mask: {np.sum(mask)}/{mask.size} pixels ({100*np.sum(mask)/mask.size:.1f}%)")
    return mask

# =============================================================================
# REGION REFERENCE AND ADMIN DATA
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

# =============================================================================
# MAIN DATA LOADING FUNCTION
# =============================================================================

def load_initial_conditions(workspace_dir, objectives=None, region='Bern', ecosystem='all', 
                            sample_fraction=None, sample_seed=42, ecosystem_lulc_path=None, landscape_lulc_path=None):
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
        ecosystem_lulc_path: Path to LULC raster file for ecosystem masking. If None, uses default paths
        landscape_lulc_path: Path to LULC raster file for landscape calculations. If None, uses default paths
    Returns:
        dict: Initial conditions for specified objectives and ecosystem
    """
    # Define all possible objectives and their file mappings
    all_objectives = {
        'abiotic': 'abiotic_condition_anomaly.tif',
        'biotic': 'biotic_condition_anomaly.tif', 
        'landscape': 'sn_dens.tif',
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
                    #if str(src.crs) != region_ref['crs']:
                        #print(f"Warning: Raster CRS {src.crs} doesn't match expected {region_ref['crs']} for region {region}")
                    
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
                # Skip strict validation for landscape_anomaly since it has special resampling handling
                if objective != 'landscape_anomaly':
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
                else:
                    # For landscape_anomaly, just log the difference - will be handled later with resampling
                    print(f"  Note: landscape_anomaly has different specs - will resample to match reference")
            # Track NaN locations BEFORE replacement
            nan_mask = np.isnan(data)
            nan_masks[objective] = nan_mask
            nan_count = np.sum(nan_mask)
            total_pixels = data.size
            
            if nan_count > 0:
                print(f"✓ Loaded {objective}: {data.shape} ({nan_count}/{total_pixels} = {100*nan_count/total_pixels:.1f}% NaN)")
                # Replace NaN with 0 so np.sum() works correctly in objective calculations
                data = np.nan_to_num(data, nan=0.0)
            else:
                print(f"✓ Loaded {objective}: {data.shape}")
                
            initial_conditions[objective] = data
    
    # Load LULC rasters - separate datasets for ecosystem masking and landscape calculations
    print(f"\n--- Loading LULC data for ecosystem: {ecosystem} ---")
    try:
        # Load ecosystem LULC for masking
        ecosystem_lulc_data, ecosystem_lulc_meta, ecosystem_lulc_path = load_lulc_raster(
            workspace_dir, ecosystem_lulc_path, region, 
            target_bounds=ref['bounds'], 
            target_shape=ref['shape'],
            target_transform=ref['transform'],
            target_crs=ref['crs'],
            lulc_type='ecosystem'
        )
        
        # Load landscape LULC for density calculations
        landscape_lulc_data, landscape_lulc_meta, landscape_lulc_path = load_lulc_raster(
            workspace_dir, landscape_lulc_path, region,
            target_bounds=ref['bounds'],
            target_shape=ref['shape'], 
            target_transform=ref['transform'],
            target_crs=ref['crs'],
            lulc_type='landscape'
        )
        
        # Validate LULC raster compatibility with reference
        if ecosystem_lulc_data.shape != ref['shape']:
            raise ValueError(
                f"Ecosystem LULC shape mismatch after processing. Expected {ref['shape']}, got {ecosystem_lulc_data.shape}"
            )
        if landscape_lulc_data.shape != ref['shape']:
            raise ValueError(
                f"Landscape LULC shape mismatch after processing. Expected {ref['shape']}, got {landscape_lulc_data.shape}"
            )
        
        # Create ecosystem mask using ecosystem LULC
        ecosystem_mask = create_ecosystem_mask(ecosystem_lulc_data, ecosystem)
        
        # Store both LULC datasets
        initial_conditions['ecosystem_lulc_data'] = ecosystem_lulc_data
        initial_conditions['ecosystem_lulc_meta'] = ecosystem_lulc_meta
        initial_conditions['landscape_lulc_data'] = landscape_lulc_data
        initial_conditions['landscape_lulc_meta'] = landscape_lulc_meta
        initial_conditions['ecosystem'] = ecosystem
        initial_conditions['ecosystem_mask'] = ecosystem_mask
        
        # Calculate landscape anomaly if requested in objectives
        if 'landscape' in objectives:
            landscape_filename = all_objectives['landscape']
            landscape_file_path = os.path.join(workspace_dir, landscape_filename)
            
            if os.path.exists(landscape_file_path):
                # Load pre-computed landscape density from file
                print(f"Loading pre-computed landscape density from: {landscape_file_path}")
                
                with rio.open(landscape_file_path) as src:
                    landscape_data = src.read(1)
                    
                    # Check compatibility and resample if needed
                    needs_resampling = False
                    if landscape_data.shape != ref['shape']:
                        #print(f"  Landscape shape mismatch: {landscape_data.shape} vs {ref['shape']} - will resample")
                        needs_resampling = True
                    if src.crs != ref['crs']:
                        #print(f"  Landscape CRS mismatch: {src.crs} vs {ref['crs']} - will reproject")
                        needs_resampling = True
                    if src.transform != ref['transform']:
                        #print(f"  Landscape transform mismatch - will resample")
                        needs_resampling = True
                    
                    if needs_resampling:
                        # Resample landscape to match reference data
                        try:
                            from rasterio.warp import reproject, Resampling
                            
                            print(f"🔄 Resampling landscape to match reference data...")
                            
                            # Create output array with target specifications
                            resampled_data = np.empty(ref['shape'], dtype=landscape_data.dtype)
                            
                            # Reproject to match reference
                            reproject(
                                source=rio.band(src, 1),
                                destination=resampled_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref['transform'],
                                dst_crs=ref['crs'],
                                resampling=Resampling.bilinear  # Use bilinear for continuous density data
                            )
                            
                            landscape_data = resampled_data
                            print(f"✓ Resampled landscape to {landscape_data.shape}")
                            
                        except Exception as resample_error:
                            raise ValueError(
                                f"Failed to resample landscape file to match reference data: {resample_error}\n"
                                f"Landscape: shape={landscape_data.shape}, crs={src.crs}, transform={src.transform}\n"
                                f"Reference: shape={ref['shape']}, crs={ref['crs']}, transform={ref['transform']}"
                            )
                
                # Handle NaN values and convert density to anomaly
                nan_mask_landscape = np.isnan(landscape_data)
                landscape_density = np.nan_to_num(landscape_data, nan=0.0)
                
                # Convert density to anomaly (higher density = lower anomaly)  
                landscape_anomaly = 1.0 - landscape_density
                
                print(f"✓ Loaded landscape density and converted to anomaly: {landscape_anomaly.shape}")
                
            else:
                # Fallback to calculation using compute_sn_dens
                print(f"Landscape file not found ({landscape_file_path}), computing from LULC data...")
                
                # Define focal classes for landscape density calculation
                focal_classes = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                    52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]  # Forest and grassland classes
                
                # Store information needed for efficient landscape recalculation
                initial_conditions['landscape_focal_classes'] = focal_classes
                initial_conditions['landscape_lulc_original_path'] = landscape_lulc_path
                
                # Use the landscape LULC file path directly
                landscape_density, _ = compute_sn_dens(landscape_lulc_path, focal_classes, radius_m=300)
                
                # Convert density to anomaly (higher density = lower anomaly)
                landscape_anomaly = 1.0 - landscape_density
                
                # Handle NaN values
                nan_mask_landscape = np.isnan(landscape_anomaly)
                landscape_anomaly = np.nan_to_num(landscape_anomaly, nan=0.0)
                
                print(f"✓ Calculated landscape_anomaly: {landscape_anomaly.shape}")
            
            # Store in initial conditions (common for both methods)
            initial_conditions['landscape_anomaly'] = landscape_anomaly
            nan_masks['landscape_anomaly'] = nan_mask_landscape
            
            # Always store focal classes and landscape LULC path for landscape recalculation with conversions
            initial_conditions['landscape_focal_classes'] = [12, 13, 16, 17]  # Default focal classes
            initial_conditions['landscape_lulc_original_path'] = landscape_lulc_path
            
            nan_count = np.sum(nan_mask_landscape)
            total_pixels = landscape_anomaly.size
            print(f"  ({nan_count}/{total_pixels} = {100*nan_count/total_pixels:.1f}% NaN)")
            if nan_count > 0:
                print(f"  → Replaced NaN with 0 (pixels excluded via eligible_mask)")
        
    except Exception as e:
        print(f"✗ Error loading LULC data: {e}")
        if ecosystem != 'all':
            print("Falling back to 'all' ecosystem (no masking)")
            ecosystem = 'all'
            ecosystem_mask = np.ones(ref['shape'], dtype=bool)
        else:
            raise e

    # Create separate eligibility masks for restoration and conversion
    shape = initial_conditions['shape']
    base_eligible_mask = np.ones(shape, dtype=bool)
    
    # Exclude pixels that were NaN in ANY objective from both restoration and conversion
    for obj_name, nan_mask in nan_masks.items():
        if np.any(nan_mask):
            base_eligible_mask = base_eligible_mask & ~nan_mask
            nan_excluded = np.sum(nan_mask)
            #print(f"  Masking {nan_excluded} NaN pixels from {obj_name}")
    print(f"  Masked NaN pixels from objective layers")
    # RESTORATION ELIGIBLE MASK: Apply ecosystem mask to eligible pixels (current method)
    restoration_eligible_mask = base_eligible_mask.copy()
    if 'ecosystem_mask' in initial_conditions:
        pre_ecosystem_count = np.sum(restoration_eligible_mask)
        restoration_eligible_mask = restoration_eligible_mask & initial_conditions['ecosystem_mask']
        post_ecosystem_count = np.sum(restoration_eligible_mask)
        ecosystem_excluded = pre_ecosystem_count - post_ecosystem_count
        #print(f"  Ecosystem masking for restoration excluded {ecosystem_excluded} pixels")
        print(f"  Final restoration eligible pixels for {ecosystem}: {post_ecosystem_count}")
    
    # CONVERSION ELIGIBLE MASK: Landscape LULC pixels NOT in focal_classes
    conversion_eligible_mask = base_eligible_mask.copy()
    if 'landscape_lulc_data' in initial_conditions:
        landscape_lulc = initial_conditions['landscape_lulc_data']
        # Define focal classes for landscape density calculations (should match global focal_classes)
        focal_classes = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                        52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]
        # Additional LULC values that must never be included
        exclude_classes = np.concatenate([
            np.arange(1, 22),          # 1–21
            np.array([61, 62, 63])
        ])

        # Final focal classes after exclusion
        focal_classes = np.setdiff1d(focal_classes, exclude_classes)
        # Get pixels that are NOT in focal_classes
        not_focal_mask = ~np.isin(landscape_lulc, focal_classes)
        pre_conversion_count = np.sum(conversion_eligible_mask)
        conversion_eligible_mask = conversion_eligible_mask & not_focal_mask
        post_conversion_count = np.sum(conversion_eligible_mask)
        focal_excluded = pre_conversion_count - post_conversion_count
        print(f"  Focal class masking for conversion excluded {focal_excluded} pixels")
        print(f"  Final conversion eligible pixels (non-focal): {post_conversion_count}")
    else:
        print(f"  Warning: No landscape LULC data found, using base mask for conversion")
    
    # For backward compatibility, use restoration eligible mask as the main eligible_mask
    eligible_mask = restoration_eligible_mask

    # You might want to exclude certain areas, e.g.:
    # eligible_mask = (initial_conditions['implementation_cost'] < 8000) & \\
    #                 (initial_conditions['abiotic_anomaly'] > 0.1)
    
    # Apply spatial sampling if requested
    eligible_indices = np.where(eligible_mask.flatten())[0]
    
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
            
            # Apply sampling to BOTH restoration and conversion masks
            restoration_sampled_mask = restoration_eligible_mask & sample_mask
            conversion_sampled_mask = conversion_eligible_mask & sample_mask
            
            restoration_sampled_indices = np.where(restoration_sampled_mask.flatten())[0]
            conversion_sampled_indices = np.where(conversion_sampled_mask.flatten())[0]
            
            if len(restoration_sampled_indices) > 0:
                # Update all masks to use the sampled region
                restoration_eligible_mask = restoration_sampled_mask
                conversion_eligible_mask = conversion_sampled_mask
                eligible_mask = restoration_sampled_mask  # For backward compatibility
                eligible_indices = restoration_sampled_indices
                
                print(f"✓ Applied spatially continuous sampling to both restoration and conversion:")
                print(f"  Rectangle: ({start_row}:{end_row}, {start_col}:{end_col})")
                print(f"  Restoration sample: {len(restoration_sampled_indices)} pixels")
                print(f"  Conversion sample: {len(conversion_sampled_indices)} pixels")
            else:
                print(f"✗ Warning: No eligible pixels in sample region, using all eligible pixels")
        else:
            print(f"✗ Warning: Sample size too small ({n_sample_target} pixels), using all eligible pixels")
    
    # Store both eligible masks and their indices
    initial_conditions['eligible_mask'] = eligible_mask  # For backward compatibility (restoration mask)
    initial_conditions['restoration_eligible_mask'] = restoration_eligible_mask
    initial_conditions['conversion_eligible_mask'] = conversion_eligible_mask
    
    restoration_eligible_indices = np.where(restoration_eligible_mask.flatten())[0]
    conversion_eligible_indices = np.where(conversion_eligible_mask.flatten())[0]
    
    initial_conditions['eligible_indices'] = eligible_indices  # For backward compatibility (restoration indices)
    initial_conditions['restoration_eligible_indices'] = restoration_eligible_indices
    initial_conditions['conversion_eligible_indices'] = conversion_eligible_indices
    
    initial_conditions['n_pixels'] = len(eligible_indices)  # For backward compatibility (restoration count)
    initial_conditions['n_restoration_pixels'] = len(restoration_eligible_indices)
    initial_conditions['n_conversion_pixels'] = len(conversion_eligible_indices)
    
    initial_conditions['sample_info'] = {
        'sample_fraction': sample_fraction,
        'sample_seed': sample_seed,
        'is_sampled': sample_fraction is not None and 0 < sample_fraction < 1
    }
    
    total_pixels = shape[0] * shape[1]
    print(f"✓ Eligibility masks created:")
    print(f"  Restoration: {initial_conditions['n_restoration_pixels']}/{total_pixels} eligible pixels ({100*initial_conditions['n_restoration_pixels']/total_pixels:.1f}% of raster)")
    print(f"  Conversion: {initial_conditions['n_conversion_pixels']}/{total_pixels} eligible pixels ({100*initial_conditions['n_conversion_pixels']/total_pixels:.1f}% of raster)")
    
    # Load admin regions for burden sharing
    admin_data = load_admin_regions(workspace_dir, region=region)
    initial_conditions['admin_data'] = admin_data
    
    print(f"✓ Data preparation complete: {initial_conditions['n_restoration_pixels']} restoration + {initial_conditions['n_conversion_pixels']} conversion eligible pixels")
    
    return initial_conditions