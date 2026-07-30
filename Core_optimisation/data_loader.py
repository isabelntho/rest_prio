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

import logging
import os
import numpy as np
import tempfile
import rasterio as rio
import geopandas as gpd
from .spatial_operations import compute_sn_dens, compute_sn_dens_array, compute_connectivity_gain_array
from .paths import DATA_DIR

logger = logging.getLogger("resto_prio")

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
                logger.info(f"Loaded {lulc_type} LULC raster: {path}")
                break
            except Exception as e:
                logger.error(f"Error loading LULC from {path}: {e}")
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
    elif ecosystem_type == 'fg':
        # Combined forest and grassland
        fg_codes = ECOSYSTEM_TYPES['forest'] + ECOSYSTEM_TYPES['grassland']
        mask = np.isin(lulc_data, fg_codes)
    elif ecosystem_type in ECOSYSTEM_TYPES:
        lulc_codes = ECOSYSTEM_TYPES[ecosystem_type]
        mask = np.isin(lulc_data, lulc_codes)
    else:
        raise ValueError(f"Unknown ecosystem type: {ecosystem_type}. Available: {list(ECOSYSTEM_TYPES.keys()) + ['all', 'fg']}")
    
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
                logger.warning(f"Kanton shapefile not found at {kanton_file} (cwd={os.getcwd()}, abs={os.path.abspath(kanton_file)})")
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
                logger.warning(f"Admin shapefile not found at {admin_file}")
                return None
            
            # Load kanton shapefile and filter to Bern
            kanton_gdf = gpd.read_file(kanton_file)
            bern_gdf = kanton_gdf[kanton_gdf['NAME'] == 'Bern'].copy()
            
            if len(bern_gdf) == 0:
                logger.warning(f"No canton named 'Bern' found in {kanton_file}")
                return None
            
            #print(f"✓ Filtered to Bern canton")
            
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
            logger.warning(f"Unknown region '{region}'. Use 'Bern' or 'CH'")
            return None
        
    except Exception as e:
        logger.error(f"Error loading admin shapefile: {e}")
        return None


# =============================================================================
# PLANNING UNIT LOADING
# =============================================================================

def load_planning_units(initial_conditions, mode='grid', unit_size_px=20,
                        workspace_dir=None, region='Bern', admin_data=None,
                        max_unit_pixels=None):
    """
    Build planning unit mappings from either a regular pixel grid or admin boundaries.

    Planning units are coarse decision entities (much larger than patches).  When
    a unit is selected during optimisation ALL restoration-eligible AND conversion-
    eligible pixels within it are activated.

    Parameters
    ----------
    initial_conditions : dict
        Standard initial_conditions dict (must be fully populated by
        ``load_initial_conditions``).
    mode : {'grid', 'admin'}
        'grid'  – regular non-overlapping blocks of ``unit_size_px × unit_size_px`` pixels.
        'admin' – one unit per non-empty administrative boundary polygon.
    unit_size_px : int
        Side-length in pixels for grid mode (e.g. 20 → 20×20 px = 2 km at 100 m resolution).
        Ignored in admin mode.
    workspace_dir : str, optional
        Workspace directory; required for admin mode when ``admin_data`` is not
        already provided.
    region : str
        Region identifier passed to ``load_admin_regions`` when loading admin data
        automatically ('Bern' or 'CH').  Ignored when ``admin_data`` is supplied.
    admin_data : dict, optional
        Pre-loaded admin data dict (from ``load_admin_regions``).  When supplied,
        ``workspace_dir`` and ``region`` are not used.
    max_unit_pixels : int, optional
        Admin mode only.  Units with more eligible pixels than this threshold are
        split into two halves along their longer axis.  None = no splitting.

    Returns
    -------
    dict
        Planning unit mapping with keys:
        - 'n_units'                     : int
        - 'unit_ids'                    : list[int]
        - 'unit_names'                  : list[str]
        - 'unit_mode'                   : 'grid' or 'admin'
        - 'unit_size_px'                : int or None
        - 'unit_to_restoration_pixels'  : dict[int, np.ndarray]  (eligible index space)
        - 'unit_to_conversion_pixels'   : dict[int, np.ndarray]  (eligible index space)
        - 'unit_pixel_counts'           : np.ndarray  (restoration + conversion per unit)
        - 'unit_restoration_counts'     : np.ndarray
        - 'unit_conversion_counts'      : np.ndarray
    """
    shape = initial_conditions['shape']
    restoration_eligible_indices = initial_conditions['restoration_eligible_indices']
    conversion_eligible_indices  = initial_conditions['conversion_eligible_indices']

    # Build fast lookup: global pixel index → position in eligible array
    rest_global_to_elig = {int(g): e for e, g in enumerate(restoration_eligible_indices)}
    conv_global_to_elig = {int(g): e for e, g in enumerate(conversion_eligible_indices)}

    if mode == 'grid':
        unit_mappings = _define_grid_units(
            shape, unit_size_px, rest_global_to_elig, conv_global_to_elig
        )
        logger.info(
            f"Grid planning units: {unit_mappings['n_units']} units "
            f"({unit_size_px}×{unit_size_px} px each)"
        )

    elif mode == 'admin':
        # Load admin data if not supplied
        if admin_data is None:
            if workspace_dir is None:
                raise ValueError("load_planning_units admin mode: supply admin_data or workspace_dir")
            admin_data = load_admin_regions(workspace_dir, region)
            if admin_data is None:
                raise RuntimeError(
                    f"load_planning_units: could not load admin regions for region='{region}'. "
                    "Check that the admin shapefiles are accessible."
                )
        unit_mappings = _define_admin_units(
            shape, initial_conditions['transform'], admin_data,
            rest_global_to_elig, conv_global_to_elig, max_unit_pixels
        )
        logger.info(
            f"Admin planning units: {unit_mappings['n_units']} units "
            f"from {admin_data['n_regions']} regions"
        )
    else:
        raise ValueError(f"load_planning_units: unknown mode '{mode}'. Use 'grid' or 'admin'.")

    return unit_mappings


def _define_grid_units(shape, unit_size_px, rest_global_to_elig, conv_global_to_elig):
    """Build planning units from a regular pixel grid."""
    n_rows, n_cols = shape
    n_unit_rows = int(np.ceil(n_rows / unit_size_px))
    n_unit_cols = int(np.ceil(n_cols / unit_size_px))

    unit_to_rest = {}
    unit_to_conv = {}
    unit_names   = []
    uid = 0

    for i in range(n_unit_rows):
        for j in range(n_unit_cols):
            r0, r1 = i * unit_size_px, min((i + 1) * unit_size_px, n_rows)
            c0, c1 = j * unit_size_px, min((j + 1) * unit_size_px, n_cols)

            rows_idx = np.arange(r0, r1)
            cols_idx = np.arange(c0, c1)
            rr, cc   = np.meshgrid(rows_idx, cols_idx, indexing='ij')
            global_px = rr.ravel() * n_cols + cc.ravel()

            re = np.array([rest_global_to_elig[g] for g in global_px if g in rest_global_to_elig],
                          dtype=np.int64)
            ce = np.array([conv_global_to_elig[g] for g in global_px if g in conv_global_to_elig],
                          dtype=np.int64)

            if re.size == 0 and ce.size == 0:
                continue

            unit_to_rest[uid] = re
            unit_to_conv[uid] = ce
            unit_names.append(f"grid_{i}_{j}")
            uid += 1

    return _assemble_unit_dict(uid, unit_names, unit_to_rest, unit_to_conv, 'grid', unit_size_px)


def _define_admin_units(shape, transform, admin_data,
                        rest_global_to_elig, conv_global_to_elig,
                        max_unit_pixels):
    """Build planning units from administrative boundary polygons."""
    from rasterio.features import rasterize as rio_rasterize

    gdf        = admin_data['gdf']
    region_col = admin_data['region_column']
    n_cols     = shape[1]

    unit_to_rest = {}
    unit_to_conv = {}
    unit_names   = []
    uid = 0

    for region_name in admin_data['unique_regions']:
        region_geom = gdf[gdf[region_col] == region_name]
        region_mask = rio_rasterize(
            region_geom.geometry,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        ).astype(bool)

        global_flat = np.where(region_mask.flatten())[0]
        re = np.array([rest_global_to_elig[g] for g in global_flat if g in rest_global_to_elig],
                      dtype=np.int64)
        ce = np.array([conv_global_to_elig[g] for g in global_flat if g in conv_global_to_elig],
                      dtype=np.int64)

        if re.size == 0 and ce.size == 0:
            continue

        total_px = re.size + ce.size

        if max_unit_pixels is not None and total_px > max_unit_pixels:
            # Split along the longer axis of the bounding box
            rows_in, cols_in = np.where(region_mask)
            r_range = int(rows_in.max() - rows_in.min()) + 1
            c_range = int(cols_in.max() - cols_in.min()) + 1
            flat_idx = np.arange(region_mask.size)
            if r_range >= c_range:
                mid = int(rows_in.min()) + r_range // 2
                splits_mask = [
                    region_mask.flatten() & (flat_idx // n_cols < mid),
                    region_mask.flatten() & (flat_idx // n_cols >= mid),
                ]
            else:
                mid = int(cols_in.min()) + c_range // 2
                splits_mask = [
                    region_mask.flatten() & (flat_idx % n_cols < mid),
                    region_mask.flatten() & (flat_idx % n_cols >= mid),
                ]
            for part_i, pmask in enumerate(splits_mask):
                gpix = np.where(pmask)[0]
                re_p = np.array([rest_global_to_elig[g] for g in gpix if g in rest_global_to_elig],
                                dtype=np.int64)
                ce_p = np.array([conv_global_to_elig[g] for g in gpix if g in conv_global_to_elig],
                                dtype=np.int64)
                if re_p.size == 0 and ce_p.size == 0:
                    continue
                unit_to_rest[uid] = re_p
                unit_to_conv[uid] = ce_p
                unit_names.append(f"{region_name}_part{part_i}")
                uid += 1
        else:
            unit_to_rest[uid] = re
            unit_to_conv[uid] = ce
            unit_names.append(str(region_name))
            uid += 1

    return _assemble_unit_dict(uid, unit_names, unit_to_rest, unit_to_conv, 'admin', None)


def _assemble_unit_dict(n_units, unit_names, unit_to_rest, unit_to_conv, mode, unit_size_px):
    """Assemble the canonical planning unit mapping dict."""
    rest_counts = np.array([len(unit_to_rest.get(i, [])) for i in range(n_units)], dtype=np.int64)
    conv_counts = np.array([len(unit_to_conv.get(i, [])) for i in range(n_units)], dtype=np.int64)
    return {
        'n_units'                    : n_units,
        'unit_ids'                   : list(range(n_units)),
        'unit_names'                 : unit_names,
        'unit_mode'                  : mode,
        'unit_size_px'               : unit_size_px,
        'unit_to_restoration_pixels' : unit_to_rest,
        'unit_to_conversion_pixels'  : unit_to_conv,
        'unit_pixel_counts'          : rest_counts + conv_counts,
        'unit_restoration_counts'    : rest_counts,
        'unit_conversion_counts'     : conv_counts,
    }


# =============================================================================
# MAIN DATA LOADING FUNCTION
# =============================================================================

def load_initial_conditions(workspace_dir, objectives=None, region='Bern', ecosystem='all', 
                            sample_fraction=None, sample_seed=42, ecosystem_lulc_path=None, landscape_lulc_path=None,
                            aggregation_factor=None, condition_scenario='global_all'):
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
        aggregation_factor: Integer block-aggregation factor applied to all arrays before optimization
                           (e.g., 2 halves resolution in each dimension, reducing pixels by ~4x).
                           Objective arrays are block-averaged; eligibility masks use any-eligible logic.
                           None or 1 disables aggregation.
        condition_scenario: Tag identifying which pre-computed anomaly rasters to load from
                           data/anomaly_scenarios/. E.g. 'global_all', 'global_drop_smd',
                           'upper_q75_all'. Defaults to 'global_all'.
    Returns:
        dict: Initial conditions for specified objectives and ecosystem
    """
    # Define all possible objectives and their file mappings.
    # A value of None means the objective is computed in-memory (no file required).
    # Abiotic and biotic anomaly rasters live in a region-specific subfolder of
    # data/: the whole-Switzerland run (region='CH') writes to data/CH_wide/,
    # every other region uses data/anomaly_scenarios/. The scenario tag selects
    # the file within that folder.
    anomaly_dir = 'CH_wide' if region == 'CH' else 'anomaly_scenarios'
    all_objectives = {
        'abiotic': str(DATA_DIR / anomaly_dir / f'abiotic_{condition_scenario}.tif'),
        'biotic':  str(DATA_DIR / anomaly_dir / f'biotic_{condition_scenario}.tif'),
        'landscape': str(DATA_DIR / 'sn_dens.tif'),
        'connectivity': None,  # Computed in-memory from landscape LULC; no file required
        # landscape_context / restoration_potential are ALWAYS computed in-memory from the
        # scenario's abiotic/biotic rasters (None → computed below), never loaded from a
        # pre-computed .tif.
        #
        # Previously 'global_all' loaded data/{landscape_context,restoration_potential}.tif
        # while every other condition_scenario recomputed in-memory. That put the baseline
        # cell on a different construction route than the LOO / q75 cells: if the saved
        # rasters were not byte-identical to the in-memory computation, a file-vs-recompute
        # discrepancy would be misattributed to the construction / scaling factors in the
        # Block 3 factorial (the 'all' vs 'drop_*' and global vs q75 contrasts). Forcing
        # in-memory for ALL scenarios guarantees the only thing differing between cells is
        # the formulation under test.
        'landscape_context': None,
        'restoration_potential': None,
        # restoration_benefit: spatially-explicit benefit. Unlike the static
        # restoration_potential, it is the total abiotic+biotic anomaly improvement
        # (incl. neighbour spillover) achieved by a plan, so it depends on WHICH pixels
        # AND their arrangement. Computed at evaluation time in RestorationProblem from
        # the abiotic/biotic anomaly rasters (auto-loaded as dependencies below).
        'restoration_benefit': None,
        # spatial_clustering has no underlying raster: it is a configuration-dependent
        # compactness metric computed from the selected-pixel geometry at evaluation time
        # (see RestorationProblem.evaluate_raw_objectives). None → treated as a computed
        # objective; only an enable flag is set in initial_conditions.
        'spatial_clustering': None,
        # Cost layer is region-specific: the whole-Switzerland run (region='CH')
        # loads data/CH_wide/cost_combined.tif; every other region uses the
        # corrected Bern layer at the data/ root.
        'cost': str(DATA_DIR / 'CH_wide' / 'cost_combined.tif') if region == 'CH'
                else str(DATA_DIR / 'implementation_cost_corrected.tif'),
        'population_proximity': str(DATA_DIR / 'population_proximity.tif'),
        'es_future_val': 'robustness/blce-robustness-data-archive/Mean_sum_of_change_ES.tif',
        'es_future_robustness': 'robustness/blce-robustness-data-archive/Undesirable_deviation_sum_of_change_ES.tif',
    }
    
    # Use all objectives if none specified
    if objectives is None:
        objectives = list(all_objectives.keys())
    
    logger.info(f"Loading initial conditions for objectives: {objectives}")
    
    # Build file paths for selected objectives
    data_files = {}
    computed_objectives = set()  # Objectives that are computed from other data, not loaded from file
    for obj in objectives:
        if obj in all_objectives:
            filename = all_objectives[obj]
            if filename is None:
                # Computed objective — no file to load; handled after LULC is loaded
                computed_objectives.add(obj)
            elif obj == 'cost':
                data_files['implementation_cost'] = os.path.join(workspace_dir, filename)
            elif obj in ('landscape_context', 'restoration_potential', 'es_future_val', 'es_future_robustness'):
                data_files[obj] = os.path.join(workspace_dir, filename)
            else:
                data_files[f'{obj}_anomaly'] = os.path.join(workspace_dir, filename)
        else:
            raise ValueError(f"Unknown objective: {obj}. Available: {list(all_objectives.keys())}")

    # Auto-load abiotic/biotic as data dependencies for computed objectives that require them,
    # even when they are not standalone optimisation objectives.
    _needs_abiotic_biotic = {'landscape_context', 'restoration_potential', 'restoration_benefit'}
    _dependency_keys = set()  # keys loaded as data-only, not as objectives
    if computed_objectives & _needs_abiotic_biotic:
        for _dep, _dep_key in [('abiotic', 'abiotic_anomaly'), ('biotic', 'biotic_anomaly')]:
            if _dep_key not in data_files and _dep not in objectives:
                _dep_path = os.path.join(workspace_dir, all_objectives[_dep])
                if os.path.exists(_dep_path):
                    data_files[_dep_key] = _dep_path
                    _dependency_keys.add(_dep_key)
                    logger.info(f"Auto-loading {_dep_key} as dependency for computed objectives")

    initial_conditions = {}
    
    # Check if all required files exist first
    missing_files = []
    for objective, file_path in data_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"  - {objective}: {file_path}")
    
    if missing_files:
        logger.error(f"Required data files not found in {workspace_dir}: {missing_files}")
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
                            logger.warning(f"Raster bounds {actual} significantly differ from expected {expected} for region {region}")
                
                initial_conditions['crs'] = ref['crs']
                initial_conditions['transform'] = ref['transform']
                initial_conditions['shape'] = ref['shape']
                initial_conditions['region'] = region
                first_raster = False
            else:
                # Objectives that may have a slightly different extent and are resampled below.
                _resampled_objectives = {'landscape_anomaly', 'es_future_val', 'es_future_robustness'}
                if objective not in _resampled_objectives:
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
                    if data.shape != ref['shape'] or src.transform != ref['transform']:
                        logger.info(
                            f"{objective} has different specs "
                            f"(shape {data.shape} vs {ref['shape']}) — will resample to match reference"
                        )
            # Track NaN locations BEFORE replacement
            nan_mask = np.isnan(data)
            nan_masks[objective] = nan_mask
            nan_count = np.sum(nan_mask)
            total_pixels = data.size
            
            if nan_count > 0:
                logger.info(f"Loaded {objective}: {data.shape} ({nan_count}/{total_pixels} = {100*nan_count/total_pixels:.1f}% NaN)")
                # Replace NaN with 0 so np.sum() works correctly in objective calculations
                data = np.nan_to_num(data, nan=0.0)
            else:
                logger.info(f"Loaded {objective}: {data.shape}")

            # Resample ES future rasters to the reference grid if their shape/transform differs.
            if objective in ('es_future_val', 'es_future_robustness') and data.shape != ref['shape']:
                try:
                    from rasterio.warp import reproject, Resampling as _Resampling
                    _resampled = np.empty(ref['shape'], dtype=np.float64)
                    reproject(
                        source=rio.band(src, 1),
                        destination=_resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref['transform'],
                        dst_crs=ref['crs'],
                        resampling=_Resampling.bilinear,
                    )
                    nan_masks[objective] = np.isnan(_resampled)
                    data = np.nan_to_num(_resampled, nan=0.0)
                    logger.info(
                        f"Resampled {objective} to {data.shape} "
                        f"({int(nan_masks[objective].sum())} NaN pixels masked from eligibility)"
                    )
                except Exception as _resamp_err:
                    raise ValueError(
                        f"Failed to resample {objective} to reference grid: {_resamp_err}\n"
                        f"Source: shape={data.shape}, crs={src.crs}, transform={src.transform}\n"
                        f"Reference: shape={ref['shape']}, crs={ref['crs']}, transform={ref['transform']}"
                    )

            initial_conditions[objective] = data
    
    # Load LULC rasters - separate datasets for ecosystem masking and landscape calculations
    logger.info(f"Loading LULC data for ecosystem: {ecosystem}")
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
                logger.info(f"Loading pre-computed landscape density from: {landscape_file_path}")
                
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
                            
                            logger.info("Resampling landscape to match reference data...")
                            
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
                            logger.info(f"Resampled landscape to {landscape_data.shape}")
                            
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
                
                logger.info(f"Loaded landscape density and converted to anomaly: {landscape_anomaly.shape}")
                
            else:
                # Fallback to calculation using compute_sn_dens
                logger.info(f"Landscape file not found ({landscape_file_path}), computing from LULC data...")
                
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
                
                logger.info(f"Calculated landscape_anomaly: {landscape_anomaly.shape}")
            
            # Store in initial conditions (common for both methods)
            initial_conditions['landscape_anomaly'] = landscape_anomaly
            nan_masks['landscape_anomaly'] = nan_mask_landscape
            
            # Always store focal classes and landscape LULC path for landscape recalculation with conversions
            initial_conditions['landscape_focal_classes'] = [12, 13, 16, 17]  # Default focal classes
            initial_conditions['landscape_lulc_original_path'] = landscape_lulc_path
            
            nan_count = np.sum(nan_mask_landscape)
            total_pixels = landscape_anomaly.size
            logger.info(f"landscape_anomaly NaN: {nan_count}/{total_pixels} = {100*nan_count/total_pixels:.1f}%")
            if nan_count > 0:
                logger.info("Replaced landscape NaN with 0 (pixels excluded via eligible_mask)")
        
    except Exception as e:
        logger.error(f"Error loading LULC data: {e}")
        if ecosystem != 'all':
            logger.warning("Falling back to 'all' ecosystem (no masking)")
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
    #print(f"  Masked NaN pixels from objective layers")
    # RESTORATION ELIGIBLE MASK: Apply ecosystem mask to eligible pixels (current method)
    restoration_eligible_mask = base_eligible_mask.copy()
    if 'ecosystem_mask' in initial_conditions:
        pre_ecosystem_count = np.sum(restoration_eligible_mask)
        restoration_eligible_mask = restoration_eligible_mask & initial_conditions['ecosystem_mask']
        post_ecosystem_count = np.sum(restoration_eligible_mask)
        ecosystem_excluded = pre_ecosystem_count - post_ecosystem_count
        #print(f"  Ecosystem masking for restoration excluded {ecosystem_excluded} pixels")
   
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
    else:
        logger.warning("No landscape LULC data found, using base mask for conversion")
    
    # For backward compatibility, use restoration eligible mask as the main eligible_mask
    eligible_mask = restoration_eligible_mask

    # You might want to exclude certain areas, e.g.:
    # eligible_mask = (initial_conditions['implementation_cost'] < 8000) & \\
    #                 (initial_conditions['abiotic_anomaly'] > 0.1)

    # -------------------------------------------------------------------------
    # Spatial aggregation (block coarsening)
    # -------------------------------------------------------------------------
    if aggregation_factor is not None and aggregation_factor > 1:
        f = int(aggregation_factor)
        h, w = initial_conditions['shape']
        # Trim to dimensions divisible by f (crop bottom/right)
        h_trim = (h // f) * f
        w_trim = (w // f) * f

        def _block_mean(arr, nan_mask):
            """Block-average a float array, respecting original NaN locations."""
            a = arr[:h_trim, :w_trim].astype(float)
            # Restore NaN before averaging to avoid zero-fill bias
            a[nan_mask[:h_trim, :w_trim]] = np.nan
            a = a.reshape(h_trim // f, f, w_trim // f, f)
            result = np.nanmean(a, axis=(1, 3))
            return result

        def _block_any(mask):
            """Coarsen a boolean mask using any-eligible logic."""
            m = mask[:h_trim, :w_trim].reshape(h_trim // f, f, w_trim // f, f)
            return m.any(axis=(1, 3))

        # Aggregate objective arrays
        obj_keys = list(nan_masks.keys())  # e.g. abiotic_anomaly, implementation_cost, ...
        for key in obj_keys:
            if key in initial_conditions:
                initial_conditions[key] = _block_mean(initial_conditions[key], nan_masks[key])
                # Update nan_mask to coarse grid
                nan_masks[key] = np.isnan(initial_conditions[key])
                # Re-zero NaN so downstream sum() calls work
                initial_conditions[key] = np.nan_to_num(initial_conditions[key], nan=0.0)

        # Aggregate boolean masks
        restoration_eligible_mask = _block_any(restoration_eligible_mask)
        conversion_eligible_mask  = _block_any(conversion_eligible_mask)
        eligible_mask             = _block_any(eligible_mask)
        if 'ecosystem_mask' in initial_conditions:
            initial_conditions['ecosystem_mask'] = _block_any(initial_conditions['ecosystem_mask'])

        # Update shape and geotransform
        new_shape = (h_trim // f, w_trim // f)
        initial_conditions['shape'] = new_shape
        shape = new_shape
        t = initial_conditions['transform']
        from rasterio.transform import Affine
        initial_conditions['transform'] = Affine(t.a * f, t.b, t.c, t.d, t.e * f, t.f)

        logger.info(f"Applied {f}x spatial aggregation: {(h, w)} → {new_shape} "
                    f"({new_shape[0]*new_shape[1]} pixels)")

        # Save aggregated objective rasters to data/ for inspection
        coarse_crs = initial_conditions['crs']
        coarse_transform = initial_conditions['transform']
        for key, orig_path in data_files.items():
            if key not in initial_conditions:
                continue
            arr = initial_conditions[key]
            stem, ext = os.path.splitext(os.path.basename(orig_path))
            out_path = os.path.join(os.path.dirname(orig_path), f"{stem}_agg{f}{ext}")
            with rio.open(
                out_path, 'w',
                driver='GTiff',
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=arr.dtype,
                crs=coarse_crs,
                transform=coarse_transform,
            ) as dst:
                out_arr = arr.copy().astype(float)
                # Restore nodata where nan_mask indicates original NaN
                if key in nan_masks:
                    coarse_nan = nan_masks[key]
                    out_arr[coarse_nan] = np.nan
                dst.write(out_arr, 1)
            logger.info(f"Saved aggregated raster: {out_path}")

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
                
                logger.info(f"Applied spatially continuous sampling: rectangle ({start_row}:{end_row}, {start_col}:{end_col}), "
                            f"{len(restoration_sampled_indices)} restoration + {len(conversion_sampled_indices)} conversion pixels")
            else:
                logger.warning("No eligible pixels in sample region, using all eligible pixels")
        else:
            logger.warning(f"Sample size too small ({n_sample_target} pixels), using all eligible pixels")
    
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

    # Compute connectivity_gain if requested.
    # Done here — after conversion_eligible_indices is finalised — so the 1D slice aligns correctly.
    if 'connectivity' in computed_objectives:
        if 'landscape_lulc_data' in initial_conditions:
            lulc_meta = initial_conditions['landscape_lulc_meta']
            lulc_data = initial_conditions['landscape_lulc_data']
            lulc_nodata = lulc_meta.get('nodata')
            lulc_res = abs(lulc_meta['transform'][0])  # pixel size in metres
            focal_classes_conn = initial_conditions.get(
                'landscape_focal_classes',
                FOCAL_CLASSES
            )
            logger.info("Computing per-pixel connectivity gain (radius=100m) …")
            gain_2d = compute_connectivity_gain_array(
                lulc_data, lulc_nodata, lulc_res, focal_classes_conn, radius_m=100
            )
            initial_conditions['connectivity_gain'] = gain_2d
            # 1D slice indexed identically to x_convert (conversion-eligible pixels)
            flat_gain = gain_2d.flatten()
            initial_conditions['connectivity_gain_1d'] = flat_gain[conversion_eligible_indices]
            logger.info(f"connectivity_gain computed: shape={gain_2d.shape}, "
                        f"1d slice length={len(initial_conditions['connectivity_gain_1d'])}")
        else:
            logger.warning("'connectivity' objective requested but landscape_lulc_data not available; skipping.")

    # Compute landscape_context if requested.
    # HYBRID structural/condition context (range ~[0, 1]; HIGHER = more supportive):
    #     ctx = (1 - w) * sn_dens  +  w * good_neighbour_fraction
    #   sn_dens                = focal proportion of semi-natural habitat within 300 m
    #                            (structural connectivity, from the landscape LULC; this is
    #                            INDEPENDENT of the abiotic/biotic condition field).
    #   good_neighbour_fraction = proportion of eligible neighbours within 500 m that are in
    #                            good condition (abiotic > 0 AND biotic > 0); excludes self.
    #   w = LC_CONDITION_WEIGHT = how much surrounding CONDITION modulates the otherwise
    #                            structural context.
    #
    # Rationale: a pure condition context is near-collinear with restoration_potential (both
    # derive from the abiotic/biotic anomaly), which collapses the trade-off front. Anchoring
    # on the structural sn_dens layer keeps landscape_context an independent objective
    # (prototype pixel-level r with restoration_potential ≈ 0.28 for sn_dens vs ≈ 0.52 for
    # condition-only; w=0.25 ≈ 0.37), while still letting surrounding condition contribute.
    # See DEVELOPMENT_TRACKER 2026-06-24.  Anomaly convention (restoration_effect): positive
    # = good, negative = degraded.  To change the structural/condition balance, edit
    # LC_CONDITION_WEIGHT below.
    LC_CONDITION_WEIGHT = 0.25
    if 'landscape_context' in computed_objectives:
        _has_abiotic = 'abiotic_anomaly' in initial_conditions
        _has_biotic  = 'biotic_anomaly'  in initial_conditions
        _has_lulc    = 'landscape_lulc_data' in initial_conditions
        if _has_lulc or _has_abiotic or _has_biotic:
            from scipy.ndimage import uniform_filter
            _w = min(max(float(LC_CONDITION_WEIGHT), 0.0), 1.0)
            _radius_px   = 5  # 500 m at 100 m resolution (condition focal window)
            _kernel_size = 2 * _radius_px + 1  # 11 × 11 box approximation
            _elig_2d     = initial_conditions['restoration_eligible_mask']
            _elig_float  = _elig_2d.astype(np.float64)

            # --- structural component: sn_dens = proportion of semi-natural habitat (300 m) ---
            _sn_dens = None
            if _has_lulc:
                _lu   = initial_conditions['landscape_lulc_data']
                _meta = initial_conditions.get('landscape_lulc_meta', {}) or {}
                try:
                    _res = abs(float(_meta['transform'][0]))
                except Exception:
                    _res = 100.0
                _nodata = _meta.get('nodata') if isinstance(_meta, dict) else None
                # Forest + grassland classes in the landscape LULC encoding (matches the
                # existing landscape_anomaly / sn_dens machinery).
                _focal_classes = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                  56, 57, 58, 59, 60, 64, 65, 66, 67]
                _sn_dens = compute_sn_dens_array(_lu, _nodata, _res, _focal_classes, radius_m=300)

            # --- condition component: fraction of eligible neighbours in good condition ---
            _cond = None
            if _has_abiotic or _has_biotic:
                _cnt_focal = uniform_filter(_elig_float, size=_kernel_size, mode='constant') * (_kernel_size ** 2)
                _cnt_neigh = np.maximum(_cnt_focal - _elig_float, 0.0)  # exclude self
                _good_condition = _elig_2d.copy()
                if _has_abiotic:
                    _good_condition = _good_condition & (initial_conditions['abiotic_anomaly'] > 0)
                if _has_biotic:
                    _good_condition = _good_condition & (initial_conditions['biotic_anomaly'] > 0)
                _good_float = _good_condition.astype(np.float64)
                _good_focal = uniform_filter(_good_float, size=_kernel_size, mode='constant') * (_kernel_size ** 2)
                _good_neigh = np.maximum(_good_focal - _good_float, 0.0)  # exclude self
                with np.errstate(invalid='ignore', divide='ignore'):
                    _cond = np.where(_cnt_neigh > 0, _good_neigh / _cnt_neigh, 0.0)

            # --- blend (degrade gracefully if a component is unavailable) ---
            if _sn_dens is not None and _cond is not None:
                _ctx_2d = (1.0 - _w) * np.nan_to_num(_sn_dens, nan=0.0) + _w * _cond
                _label = f"hybrid: (1-{_w:.2f})*sn_dens[300m] + {_w:.2f}*good_neighbour_fraction"
            elif _sn_dens is not None:
                _ctx_2d = np.nan_to_num(_sn_dens, nan=0.0)
                _label = "structural sn_dens[300m] only (no abiotic/biotic available)"
            else:
                _ctx_2d = _cond
                _label = "condition good_neighbour_fraction only (no landscape LULC available)"

            initial_conditions['landscape_context'] = _ctx_2d
            flat_ctx = _ctx_2d.flatten()
            initial_conditions['landscape_context_1d'] = flat_ctx[restoration_eligible_indices].astype(np.float64)
            logger.info(
                f"landscape_context computed ({_label}): "
                f"1d length={len(initial_conditions['landscape_context_1d'])}, "
                f"mean={initial_conditions['landscape_context_1d'].mean():.4f}"
            )
        else:
            logger.warning("'landscape_context' objective requested but neither landscape LULC nor abiotic/biotic is available; skipping.")

    # Build 1D slice when landscape_context was loaded from a pre-computed file.
    if 'landscape_context' in initial_conditions and 'landscape_context_1d' not in initial_conditions:
        flat_ctx = initial_conditions['landscape_context'].flatten()
        initial_conditions['landscape_context_1d'] = flat_ctx[restoration_eligible_indices].astype(np.float64)
        logger.info(
            f"landscape_context loaded from file: "
            f"1d length={len(initial_conditions['landscape_context_1d'])}, "
            f"mean={initial_conditions['landscape_context_1d'].mean():.4f}"
        )

    # Compute restoration_potential if requested.
    # For each restoration-eligible pixel: equal-weight mean of abiotic and biotic baseline
    # anomaly values at that pixel (static — no neighbourhood, no restoration effect).
    # Lower value → pixel more degraded on both dimensions → higher restoration potential.
    # Minimised directly: sum over selected pixels; more negative = selecting more degraded pixels.
    if 'restoration_potential' in computed_objectives:
        _rp_has_abiotic = 'abiotic_anomaly' in initial_conditions
        _rp_has_biotic  = 'biotic_anomaly'  in initial_conditions
        if _rp_has_abiotic and _rp_has_biotic:
            _rp_2d = (initial_conditions['abiotic_anomaly'].astype(np.float64) +
                      initial_conditions['biotic_anomaly'].astype(np.float64)) / 2.0
            _components = 2
        elif _rp_has_abiotic:
            logger.warning("'restoration_potential' objective: biotic_anomaly not available; using abiotic only.")
            _rp_2d = initial_conditions['abiotic_anomaly'].astype(np.float64)
            _components = 1
        elif _rp_has_biotic:
            logger.warning("'restoration_potential' objective: abiotic_anomaly not available; using biotic only.")
            _rp_2d = initial_conditions['biotic_anomaly'].astype(np.float64)
            _components = 1
        else:
            logger.warning("'restoration_potential' objective requested but neither abiotic_anomaly nor biotic_anomaly is available; skipping.")
            _rp_2d = None

        if _rp_2d is not None:
            initial_conditions['restoration_potential'] = _rp_2d
            _flat_rp = _rp_2d.flatten()
            initial_conditions['restoration_potential_1d'] = _flat_rp[restoration_eligible_indices].astype(np.float64)
            logger.info(
                f"restoration_potential computed (components={_components}): "
                f"1d length={len(initial_conditions['restoration_potential_1d'])}, "
                f"mean={initial_conditions['restoration_potential_1d'].mean():.4f}"
            )

    # Build 1D slice when restoration_potential was loaded from a pre-computed file.
    if 'restoration_potential' in initial_conditions and 'restoration_potential_1d' not in initial_conditions:
        _flat_rp = initial_conditions['restoration_potential'].flatten()
        initial_conditions['restoration_potential_1d'] = _flat_rp[restoration_eligible_indices].astype(np.float64)
        logger.info(
            f"restoration_potential loaded from file: "
            f"1d length={len(initial_conditions['restoration_potential_1d'])}, "
            f"mean={initial_conditions['restoration_potential_1d'].mean():.4f}"
        )

    # Build 1D slices for ES future objectives loaded from robustness raster files.
    # NaN pixels (outside BLCE study area) are excluded from restoration_eligible_indices
    # via the nan_masks mechanism, so only valid-data pixels appear in the 1D slice.
    if 'es_future_val' in initial_conditions and 'es_future_val_1d' not in initial_conditions:
        _flat_esv = np.nan_to_num(initial_conditions['es_future_val'].flatten(), nan=0.0)
        initial_conditions['es_future_val_1d'] = _flat_esv[restoration_eligible_indices].astype(np.float64)
        _nan_in_elig = int(np.sum(initial_conditions['es_future_val_1d'] == 0.0))
        logger.info(
            f"es_future_val loaded from file: "
            f"1d length={len(initial_conditions['es_future_val_1d'])}, "
            f"mean={initial_conditions['es_future_val_1d'].mean():.4f}, "
            f"zero/NaN-filled in eligible area={_nan_in_elig}"
        )

    if 'es_future_robustness' in initial_conditions and 'es_future_robustness_1d' not in initial_conditions:
        _flat_esr = np.nan_to_num(initial_conditions['es_future_robustness'].flatten(), nan=0.0)
        initial_conditions['es_future_robustness_1d'] = _flat_esr[restoration_eligible_indices].astype(np.float64)
        _nan_in_elig_r = int(np.sum(initial_conditions['es_future_robustness_1d'] == 0.0))
        logger.info(
            f"es_future_robustness loaded from file: "
            f"1d length={len(initial_conditions['es_future_robustness_1d'])}, "
            f"mean={initial_conditions['es_future_robustness_1d'].mean():.4f}, "
            f"zero/NaN-filled in eligible area={_nan_in_elig_r}"
        )

    # spatial_clustering is computed from the selected-pixel geometry at evaluation
    # time, so there is no per-pixel array to build here — only a flag telling
    # RestorationProblem to register it as an objective.
    if 'spatial_clustering' in computed_objectives:
        initial_conditions['spatial_clustering_enabled'] = True
        logger.info("spatial_clustering objective enabled (configuration-dependent compactness)")

    # restoration_benefit is likewise computed at evaluation time (from the updated
    # abiotic/biotic anomalies incl. neighbour spillover), so only a flag is set here.
    # The abiotic/biotic anomaly rasters it needs were auto-loaded as dependencies.
    if 'restoration_benefit' in computed_objectives:
        initial_conditions['restoration_benefit_enabled'] = True
        logger.info("restoration_benefit objective enabled (spillover-aware abiotic+biotic improvement)")

    initial_conditions['sample_info'] = {
        'sample_fraction': sample_fraction,
        'sample_seed': sample_seed,
        'is_sampled': sample_fraction is not None and 0 < sample_fraction < 1
    }

    # Record which keys were loaded as data-only dependencies (not optimisation objectives).
    # RestorationProblem uses this to skip registering them as objectives.
    initial_conditions['_dependency_keys'] = _dependency_keys
    
    total_pixels = shape[0] * shape[1]
    #print(f"✓ Eligibility masks created:")
    #print(f"  Restoration: {initial_conditions['n_restoration_pixels']}/{total_pixels} eligible pixels ({100*initial_conditions['n_restoration_pixels']/total_pixels:.1f}% of raster)")
    #print(f"  Conversion: {initial_conditions['n_conversion_pixels']}/{total_pixels} eligible pixels ({100*initial_conditions['n_conversion_pixels']/total_pixels:.1f}% of raster)")
    
    # Load admin regions for burden sharing
    admin_data = load_admin_regions(workspace_dir, region=region)
    initial_conditions['admin_data'] = admin_data
    
    logger.info(f"Data preparation complete: {initial_conditions['n_restoration_pixels']} restoration + {initial_conditions['n_conversion_pixels']} conversion eligible pixels")
    
    return initial_conditions