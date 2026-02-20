"""
Speed test script for compute_sn_dens vs compute_sn_dens_array functions
"""

import numpy as np
import time
import tempfile
import os
import rasterio as rio
from spatial_operations import compute_sn_dens, compute_sn_dens_array
from data_loader import load_lulc_raster

def test_sn_dens_speed(ecosystem='agricultural', region='Bern', radius_m=300, n_iterations=3):
    """
    Test speed comparison between compute_sn_dens and compute_sn_dens_array
    
    Args:
        ecosystem: Ecosystem type to test with
        region: Region to load data for  
        radius_m: Radius for landscape calculation
        n_iterations: Number of iterations to average timing
    """
    print(f"=== TESTING SN_DENS SPEED ===")
    print(f"Ecosystem: {ecosystem}, Region: {region}, Radius: {radius_m}m")
    print(f"Running {n_iterations} iterations each...")
    
    # Load LULC data
    print("\nLoading LULC data...")
    try:
        result = load_lulc_raster(".", region=region)
        print(f"DEBUG: load_lulc_raster returned: {type(result)}")
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            # load_lulc_raster returns (lulc_data, lulc_meta, original_path)
            lulc_data, lulc_meta, lulc_path = result
            print(f"DEBUG: Tuple contents - data: {type(lulc_data)}, meta: {type(lulc_meta)}, path: {type(lulc_path)}")
                        
        elif isinstance(result, dict):
            # If it returns a dict, extract the needed components
            lulc_data = result.get('data') or result.get('lulc_data')
            lulc_path = result.get('path') or result.get('lulc_path')
            lulc_meta = result.get('meta') or result.get('lulc_meta')
        else:
            raise ValueError(f"Unexpected return type from load_lulc_raster: {type(result)}")
        
        # Use hardcoded focal classes
        focal_classes = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                        52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67] 

        print(f"✓ Loaded LULC: {lulc_data.shape}, focal classes: {len(focal_classes)}")
        print(f"  Resolution: {lulc_meta['transform'][0]:.1f}m")
        print(f"  Focal classes: {focal_classes[:5]}..." if len(focal_classes) > 5 else f"  Focal classes: {focal_classes}")
    except Exception as e:
        print(f"✗ Failed to load LULC data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract key parameters for array-based function
    res = abs(lulc_meta['transform'][0])  # Pixel resolution
    nodata = lulc_meta.get('nodata', None)
    
    # Test 1: compute_sn_dens (file-based)
    print(f"\n1. Testing compute_sn_dens (file-based)...")
    file_times = []
    
    for i in range(n_iterations):
        print(f"   Iteration {i+1}/{n_iterations}...", end=' ')
        start_time = time.time()
        
        try:
            sn_dens_file, profile = compute_sn_dens(lulc_path, focal_classes, radius_m=radius_m)
            end_time = time.time()
            iteration_time = end_time - start_time
            file_times.append(iteration_time)
            print(f"{iteration_time:.3f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            file_times.append(float('inf'))
    
    # Test 2: compute_sn_dens_array (array-based)
    print(f"\n2. Testing compute_sn_dens_array (array-based)...")
    array_times = []
    
    for i in range(n_iterations):
        print(f"   Iteration {i+1}/{n_iterations}...", end=' ')
        start_time = time.time()
        
        try:
            sn_dens_array = compute_sn_dens_array(lulc_data, nodata, res, focal_classes, radius_m=radius_m)
            end_time = time.time()
            iteration_time = end_time - start_time
            array_times.append(iteration_time)
            print(f"{iteration_time:.3f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            array_times.append(float('inf'))
    
    # Calculate statistics
    valid_file_times = [t for t in file_times if t != float('inf')]
    valid_array_times = [t for t in array_times if t != float('inf')]
    
    if valid_file_times and valid_array_times:
        file_avg = np.mean(valid_file_times)
        file_std = np.std(valid_file_times)
        array_avg = np.mean(valid_array_times)
        array_std = np.std(valid_array_times)
        
        print(f"\n=== RESULTS ===")
        print(f"compute_sn_dens (file):      {file_avg:.3f}s ± {file_std:.3f}s")
        print(f"compute_sn_dens_array:       {array_avg:.3f}s ± {array_std:.3f}s")
        print(f"Speedup (array vs file):     {file_avg/array_avg:.2f}x")
        
        # Check if results are similar
        try:
            if 'sn_dens_file' in locals() and 'sn_dens_array' in locals():
                # Compare last iteration results
                max_diff = np.nanmax(np.abs(sn_dens_file - sn_dens_array))
                mean_diff = np.nanmean(np.abs(sn_dens_file - sn_dens_array))
                print(f"\nResult similarity:")
                print(f"Max difference:              {max_diff:.6f}")
                print(f"Mean absolute difference:    {mean_diff:.6f}")
                if max_diff < 1e-10:
                    print("✓ Results are essentially identical")
                elif max_diff < 1e-6:
                    print("✓ Results are very similar")
                elif max_diff < 1e-3:
                    print("⚠ Results have small differences")
                else:
                    print("✗ Results have significant differences")
        except:
            print("Could not compare results")
    
    else:
        print(f"\n✗ Some tests failed - check errors above")


if __name__ == "__main__":
    # Test with real data
    test_sn_dens_speed(ecosystem='agricultural', region='Bern', radius_m=300, n_iterations=3)
    
    print(f"\n=== SPEED TEST COMPLETE ===")