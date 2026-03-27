"""
LEGACY FUNCTIONS — removed from spatial_operations.py
======================================================
These functions were removed from spatial_operations.py because they are
currently unused (see NOTE docstrings). They are preserved here in case they
become useful again in the future.

Do NOT import from this file in production code.
"""

import numpy as np


def merge_overlapping_regions(regions):
    """
    Merge overlapping rectangular regions to minimize redundant processing.
    
    NOTE: This function is currently unused as we switched to full landscape 
    recalculation, but kept for potential future selective processing.
    """
    if not regions:
        return []
    
    # Convert set to list and sort
    region_list = list(regions)
    region_list.sort()
    
    merged = []
    current = region_list[0]
    
    for next_region in region_list[1:]:
        # Check if regions overlap
        r_min1, r_max1, c_min1, c_max1 = current
        r_min2, r_max2, c_min2, c_max2 = next_region
        
        # Check for overlap
        if (r_max1 >= r_min2 and r_min1 <= r_max2 and 
            c_max1 >= c_min2 and c_min1 <= c_max2):
            # Merge regions
            current = (
                min(r_min1, r_min2),
                max(r_max1, r_max2),
                min(c_min1, c_min2),
                max(c_max1, c_max2)
            )
        else:
            # No overlap, add current to merged list
            merged.append(current)
            current = next_region
    
    merged.append(current)
    return merged


def approximate_region_landscape_change(conversions, region_shape, radius_px):
    """
    Approximate landscape change within a region using mathematical models.
    No compute_sn_dens needed - pure mathematical approximation.
    
    NOTE: This function is currently unused as we switched to full landscape 
    recalculation for accuracy, but kept for potential future use.
    """
    from scipy.ndimage import convolve

    # Create conversion mask for this region
    conversion_mask = np.zeros(region_shape, dtype=np.float32)
    for row, col in conversions:
        if 0 <= row < region_shape[0] and 0 <= col < region_shape[1]:
            conversion_mask[row, col] = 1.0
    
    # Create circular influence kernel
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    kernel = ((x**2 + y**2) <= radius_px**2).astype(np.float32)
    
    # Distance-weighted kernel (closer pixels have more influence)
    distances = np.sqrt(x**2 + y**2)
    distances[distances == 0] = 1  # Avoid division by zero
    kernel = kernel / (1 + distances * 0.1)  # Gradual distance decay
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Apply convolution to estimate landscape density improvement
    density_improvement = convolve(conversion_mask, kernel, mode='constant', cval=0.0)
    
    # Convert density improvement to anomaly change
    # Density increase = anomaly decrease (improvement)
    anomaly_improvement = -density_improvement
    
    # Scale the effect based on conversion effectiveness
    # This parameter can be tuned based on validation against full calculations
    effect_strength = 0.15  # Adjust based on your data
    
    return effect_strength * anomaly_improvement
