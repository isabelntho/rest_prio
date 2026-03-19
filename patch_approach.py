"""
Patch-based restoration optimization approach
==============================================

This module implements a patch-based decision framework where decisions 
are made at the patch level (groups of pixels) but objectives are still 
calculated at the pixel level for accuracy.

OVERVIEW
--------
- Decision vector: Patches (groups of pixels, e.g., 100x100)
- Objective calculation: Still at pixel level (maintains accuracy)
- Benefits: Smaller decision space, faster optimization for large areas
- Patches defined on fixed non-overlapping grid

USAGE IN resto_anom.py
----------------------
Simply set use_patch_approach=True in run_single_scenario_optimization():
   
   results = run_single_scenario_optimization(
       initial_conditions=initial_conditions,
       scenario_params=scenario_params,
       use_patch_approach=True,  # Enable patch mode
       patch_size=10  # Optional: patch size in pixels (default=100)
   )

Or when using run_settings dict:

   run_settings = {
       'pop_size': 50,
       'n_generations': 100,
       'use_patch_approach': True,  # Enable patch mode
       'patch_size': 100 # Optional: default is 100
   }

PATCH STRUCTURE
---------------
- Patches are defined on a fixe non-overlapping grid
- Grid boundaries determined by patch_size
- Each pixel belongs to exactly one patch
- Patch #42 always contains the same pixels across all optimization runs
- Only which patches are SELECTED varies between iterations

IMPLEMENTATION
--------------
The PatchRestorationProblem class (in resto_anom.py):
  - Inherits from RestorationProblem
  - Converts patch decisions → pixel decisions internally
  - Evaluates using existing pixel-level objective functions
  - Returns patch-level constraint (number of patches used)

Created: February 2026
"""

import numpy as np
from typing import Dict, Tuple, List

# =============================================================================
# PYMOO IMPORTS
# =============================================================================

from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair


def _safe_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    t = max(float(temperature), 1e-9)
    z = (x - np.max(x)) / t
    exp_z = np.exp(np.clip(z, -60.0, 60.0))
    s = np.sum(exp_z)
    if s <= 0 or not np.isfinite(s):
        return np.full_like(exp_z, 1.0 / len(exp_z), dtype=np.float64)
    return exp_z / s


def _sample_without_replacement_weighted(candidates: np.ndarray,
                                         weights: np.ndarray,
                                         k: int) -> np.ndarray:
    """Sample up to k unique indices from candidates using positive weights."""
    if k <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)

    selected = []
    available = candidates.copy()
    w = np.asarray(weights, dtype=np.float64).copy()

    n_draws = min(k, len(available))
    for _ in range(n_draws):
        w = np.clip(w, 0.0, None)
        if np.sum(w) <= 0:
            probs = np.full(len(available), 1.0 / len(available))
        else:
            probs = w / np.sum(w)

        pick_pos = np.random.choice(len(available), p=probs)
        selected.append(int(available[pick_pos]))

        available = np.delete(available, pick_pos)
        w = np.delete(w, pick_pos)
        if len(available) == 0:
            break

    return np.asarray(selected, dtype=int)


def aggregate_patch_scores_from_pixel_scores(
    patch_mappings: Dict,
    restoration_pixel_scores: np.ndarray = None,
    conversion_pixel_scores: np.ndarray = None,
    mode: str = 'mean'
) -> np.ndarray:
    """
    Aggregate pixel-level scores into one score per patch.

    Args:
        patch_mappings: Patch mapping dictionary.
        restoration_pixel_scores: Score per restoration-eligible pixel.
        conversion_pixel_scores: Score per conversion-eligible pixel.
        mode: Aggregation strategy ('mean' or 'quantile90').

    Returns:
        np.ndarray: Patch scores for concatenated [restoration_patches, conversion_patches].
    """
    def _agg(vals: np.ndarray) -> float:
        if vals.size == 0:
            return 0.0
        if mode == 'quantile90':
            return float(np.nanquantile(vals, 0.9))
        return float(np.nanmean(vals))

    restoration_patches = patch_mappings['restoration_patches']
    conversion_patches = patch_mappings['conversion_patches']

    n_rest = restoration_patches['n_patches']
    n_conv = conversion_patches['n_patches']

    rest_scores = np.zeros(n_rest, dtype=np.float64)
    conv_scores = np.zeros(n_conv, dtype=np.float64)

    if restoration_pixel_scores is not None:
        r_scores = np.asarray(restoration_pixel_scores, dtype=np.float64)
        for pid in range(n_rest):
            pix = np.asarray(restoration_patches['patch_to_pixels'][pid], dtype=int)
            pix = pix[(pix >= 0) & (pix < len(r_scores))]
            if pix.size > 0:
                rest_scores[pid] = _agg(r_scores[pix])

    if conversion_pixel_scores is not None:
        c_scores = np.asarray(conversion_pixel_scores, dtype=np.float64)
        for pid in range(n_conv):
            pix = np.asarray(conversion_patches['patch_to_pixels'][pid], dtype=int)
            pix = pix[(pix >= 0) & (pix < len(c_scores))]
            if pix.size > 0:
                conv_scores[pid] = _agg(c_scores[pix])

    all_scores = np.concatenate([rest_scores, conv_scores]) if n_conv > 0 else rest_scores

    # Normalize to [0, 1] for stable use with stochastic operators.
    if all_scores.size > 0:
        lo = np.nanmin(all_scores)
        hi = np.nanmax(all_scores)
        span = hi - lo
        if np.isfinite(span) and span > 1e-12:
            all_scores = (all_scores - lo) / span
        else:
            all_scores = np.ones_like(all_scores) * 0.5

    return np.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# PATCH DEFINITION AND MAPPING
# =============================================================================

def define_patches(shape: Tuple[int, int], 
                   patch_size: int = 10,
                   eligible_mask: np.ndarray = None) -> Dict:
    """Create rectangular patch grid. Returns dict with patch mappings and metadata."""
    n_rows, n_cols = shape
    
    # Calculate number of patches in each dimension
    n_patch_rows = int(np.ceil(n_rows / patch_size))
    n_patch_cols = int(np.ceil(n_cols / patch_size))
    
    # Create patch grid (2D array mapping each pixel to a patch ID)
    patch_grid = np.zeros(shape, dtype=int)
    patch_id = 0
    patch_to_pixels = {}
    patch_info = []
    
    for i in range(n_patch_rows):
        for j in range(n_patch_cols):
            # Define patch boundaries
            row_start = i * patch_size
            row_end = min((i + 1) * patch_size, n_rows)
            col_start = j * patch_size
            col_end = min((j + 1) * patch_size, n_cols)
            
            # Get pixel indices in this patch
            rows = np.arange(row_start, row_end)
            cols = np.arange(col_start, col_end)
            row_coords, col_coords = np.meshgrid(rows, cols, indexing='ij')
            
            # Convert 2D coordinates to 1D indices
            pixel_indices_1d = row_coords.ravel() * n_cols + col_coords.ravel()
            
            # If eligible_mask provided, filter to only eligible pixels
            if eligible_mask is not None:
                # Get 2D mask for this patch
                patch_mask = eligible_mask[row_start:row_end, col_start:col_end].ravel()
                pixel_indices_1d = pixel_indices_1d[patch_mask]
            
            # Only include patch if it has at least one eligible pixel
            if len(pixel_indices_1d) > 0:
                patch_to_pixels[patch_id] = pixel_indices_1d
                patch_grid[row_start:row_end, col_start:col_end] = patch_id
                
                # Store patch metadata
                patch_info.append({
                    'patch_id': patch_id,
                    'row_start': row_start,
                    'row_end': row_end,
                    'col_start': col_start,
                    'col_end': col_end,
                    'n_pixels': len(pixel_indices_1d),
                    'n_rows': row_end - row_start,
                    'n_cols': col_end - col_start
                })
                
                patch_id += 1
    
    return {
        'n_patches': patch_id,
        'patch_indices': list(range(patch_id)),
        'patch_to_pixels': patch_to_pixels,
        'patch_grid': patch_grid,
        'patch_size': patch_size,
        'patch_info': patch_info,
        'shape': shape
    }


def convert_patch_decisions_to_pixels(patch_decisions: np.ndarray,
                                      patch_mapping: Dict,
                                      n_pixels: int) -> np.ndarray:
    """Convert patch-level decisions to pixel-level binary array."""
    pixel_decisions = np.zeros(n_pixels, dtype=int)
    patch_to_pixels = patch_mapping['patch_to_pixels']

    # Gather all selected patch pixels at once (much faster for many tiny patches).
    selected_patches = np.where(patch_decisions == 1)[0]
    if selected_patches.size == 0:
        return pixel_decisions

    chunks = []
    for patch_id in selected_patches:
        if patch_id in patch_to_pixels:
            pix = np.atleast_1d(patch_to_pixels[patch_id])
            if pix.size > 0:
                chunks.append(pix)

    if not chunks:
        return pixel_decisions

    all_indices = np.concatenate(chunks)
    valid_indices = all_indices[(all_indices >= 0) & (all_indices < n_pixels)]
    if valid_indices.size > 0:
        pixel_decisions[valid_indices] = 1
    
    return pixel_decisions


def map_global_to_eligible_indices(global_pixel_indices: np.ndarray,
                                   eligible_indices: np.ndarray,
                                   global_to_eligible: Dict[int, int] = None) -> np.ndarray:
    """Map global pixel indices to eligible pixel index space."""
    if global_to_eligible is None:
        global_to_eligible = {g: e for e, g in enumerate(eligible_indices)}
    return np.array([global_to_eligible[idx] for idx in global_pixel_indices 
                    if idx in global_to_eligible])


def create_patch_mappings_for_restoration_and_conversion(initial_conditions: Dict,
                                                         patch_size: int = 10) -> Dict:
    """Create separate patch systems for restoration and conversion areas."""
    shape = initial_conditions['shape']
    
    # Create patches for restoration-eligible areas
    restoration_patches = define_patches(
        shape=shape,
        patch_size=patch_size,
        eligible_mask=initial_conditions['restoration_eligible_mask']
    )
    
    # Map global pixel indices to restoration-eligible indices
    restoration_eligible_indices = initial_conditions['restoration_eligible_indices']
    restoration_global_to_eligible = {
        g: e for e, g in enumerate(restoration_eligible_indices)
    }
    for patch_id, global_indices in restoration_patches['patch_to_pixels'].items():
        eligible_indices = map_global_to_eligible_indices(
            global_indices,
            restoration_eligible_indices,
            global_to_eligible=restoration_global_to_eligible,
        )
        restoration_patches['patch_to_pixels'][patch_id] = eligible_indices
    
    # Create patches for conversion-eligible areas
    conversion_patches = define_patches(
        shape=shape,
        patch_size=patch_size,
        eligible_mask=initial_conditions['conversion_eligible_mask']
    )
    
    # Map global pixel indices to conversion-eligible indices
    conversion_eligible_indices = initial_conditions['conversion_eligible_indices']
    conversion_global_to_eligible = {
        g: e for e, g in enumerate(conversion_eligible_indices)
    }
    for patch_id, global_indices in conversion_patches['patch_to_pixels'].items():
        eligible_indices = map_global_to_eligible_indices(
            global_indices,
            conversion_eligible_indices,
            global_to_eligible=conversion_global_to_eligible,
        )
        conversion_patches['patch_to_pixels'][patch_id] = eligible_indices
    
    return {
        'restoration_patches': restoration_patches,
        'conversion_patches': conversion_patches,
        'patch_size': patch_size
    }


# =============================================================================
# CONSTRAINT HELPERS FOR PATCH APPROACH
# =============================================================================

def enforce_patch_count(patch_decisions: np.ndarray, 
                       target_count: int,
                       patch_mapping: Dict = None,
                       scores: np.ndarray = None) -> np.ndarray:
    """Enforce exact patch count by adding/removing patches."""
    current_count = np.sum(patch_decisions)
    
    if current_count == target_count:
        return patch_decisions.copy()
    
    result = patch_decisions.copy()
    
    if current_count < target_count:
        # Need to add patches
        n_to_add = target_count - current_count
        available = np.where(patch_decisions == 0)[0]
        
        if len(available) == 0:
            return result  # Can't add more
        
        n_to_add = min(n_to_add, len(available))
        
        if scores is not None and len(scores) == len(patch_decisions):
            # Select patches with highest scores
            available_scores = scores[available]
            add_indices = available[np.argsort(-available_scores)[:n_to_add]]
        else:
            # Random selection
            add_indices = np.random.choice(available, size=n_to_add, replace=False)
        
        result[add_indices] = 1
        
    else:
        # Need to remove patches
        n_to_remove = current_count - target_count
        active = np.where(patch_decisions == 1)[0]
        
        if len(active) == 0:
            return result  # Nothing to remove
            
        n_to_remove = min(n_to_remove, len(active))
        
        if scores is not None and len(scores) == len(patch_decisions):
            # Remove patches with lowest scores
            active_scores = scores[active]
            remove_indices = active[np.argsort(active_scores)[:n_to_remove]]
        else:
            # Random removal
            remove_indices = np.random.choice(active, size=n_to_remove, replace=False)
        
        result[remove_indices] = 0
    
    return result


# =============================================================================
# PYMOO OPERATORS FOR PATCH APPROACH
# =============================================================================

class PatchAwareSampling(Sampling):
    """Sampling that creates initial solutions near target pixel count."""
    
    def __init__(self, patch_mappings, target_pixels, pixel_tolerance=0.05,
                 patch_scores=None, score_temperature=0.25, random_share=0.15):
        super().__init__()
        self.patch_mappings = patch_mappings
        self.target_pixels = target_pixels
        self.pixel_tolerance = pixel_tolerance
        self.patch_scores = patch_scores
        self.score_temperature = float(score_temperature)
        self.random_share = float(np.clip(random_share, 0.0, 1.0))
        
        # Precompute pixels per patch for both restoration and conversion
        restoration_patches = patch_mappings['restoration_patches']
        conversion_patches = patch_mappings['conversion_patches']
        
        self.restoration_pixels_per_patch = np.array([
            len(restoration_patches['patch_to_pixels'][i])
            for i in range(restoration_patches['n_patches'])
        ])
        
        self.conversion_pixels_per_patch = np.array([
            len(conversion_patches['patch_to_pixels'][i])
            for i in range(conversion_patches['n_patches'])
        ])
        
        self.n_restoration_patches = restoration_patches['n_patches']
        self.n_conversion_patches = conversion_patches['n_patches']

        all_pixels = np.concatenate([self.restoration_pixels_per_patch,
                                     self.conversion_pixels_per_patch]).astype(np.float64)
        self._min_patch_pixels = max(float(np.min(all_pixels[all_pixels > 0])) if np.any(all_pixels > 0) else 1.0, 1.0)

        # Precompute static blended weights once; per-iteration renormalization is cheap.
        if self.patch_scores is not None and len(self.patch_scores) == len(all_pixels):
            score_part = _safe_softmax(np.asarray(self.patch_scores, dtype=np.float64),
                                       temperature=self.score_temperature)
            size_part = np.clip(all_pixels, 0.0, None)
            if np.sum(size_part) <= 0:
                size_part = np.full_like(size_part, 1.0 / len(size_part), dtype=np.float64)
            else:
                size_part = size_part / np.sum(size_part)
            self._base_weights = np.clip((1.0 - self.random_share) * score_part + self.random_share * size_part,
                                         1e-12, None)
        else:
            self._base_weights = np.clip(all_pixels, 1e-12, None)

    def _candidate_weights(self, candidates: np.ndarray, all_pixels: np.ndarray) -> np.ndarray:
        """Blend quality-driven and size-driven weights to avoid deterministic collapse."""
        if candidates.size == 0:
            return np.array([], dtype=np.float64)

        return np.clip(self._base_weights[candidates], 1e-12, None)
        
    def _do(self, problem, n_samples, **kwargs):
        """Generate solutions targeting center of tolerance range."""
        n_patches = self.n_restoration_patches + self.n_conversion_patches
        X = np.zeros((n_samples, n_patches), dtype=int)
        
        # Combine all patch pixel counts
        all_pixels = np.concatenate([self.restoration_pixels_per_patch,
                                     self.conversion_pixels_per_patch])
        
        target_center = self.target_pixels
        target_min = int(self.target_pixels * (1 - self.pixel_tolerance))
        target_max = int(self.target_pixels * (1 + self.pixel_tolerance))
        
        for i in range(n_samples):
            # Score-guided stochastic construction to preserve diversity.
            active = np.zeros(n_patches, dtype=bool)
            current = 0
            safety = 0
            max_steps = int(max(target_max / self._min_patch_pixels * 3.0, 64))

            while current < target_min and safety < max_steps:
                candidates = np.where(~active)[0]
                if candidates.size == 0:
                    break

                weights = self._candidate_weights(candidates, all_pixels)
                idx = int(_sample_without_replacement_weighted(candidates, weights, 1)[0])
                active[idx] = True
                current += int(all_pixels[idx])
                safety += 1

            # Optional extra picks up to target_max for variability.
            while current < target_center and safety < max_steps:
                candidates = np.where(~active)[0]
                if candidates.size == 0:
                    break

                # Only consider candidates that keep us within upper tolerance.
                feasible = candidates[current + all_pixels[candidates] <= target_max]
                if feasible.size == 0:
                    break

                # Stochastic stopping close to center to avoid rigid filling.
                if current >= target_min and np.random.random() < 0.35:
                    break

                weights = self._candidate_weights(feasible, all_pixels)
                idx = int(_sample_without_replacement_weighted(feasible, weights, 1)[0])
                active[idx] = True
                current += int(all_pixels[idx])
                safety += 1

            X[i, active] = 1
        
        return X


class PatchRepair(Repair):
    """Repair operator enforcing patch_count or pixel_count constraints."""
    
    def __init__(self, constraint_type='pixel_count', target_value=None, 
                 patch_mappings=None, pixel_tolerance=0.05,
                 patch_scores=None, score_temperature=0.2, top_k=12):
        super().__init__()
        self.constraint_type = constraint_type
        self.target_value = target_value
        self.patch_mappings = patch_mappings
        self.pixel_tolerance = pixel_tolerance
        self.patch_scores = patch_scores
        self.score_temperature = float(score_temperature)
        self.top_k = int(max(2, top_k))
        
        # Precompute pixels per patch for efficiency
        if patch_mappings is not None:
            self.restoration_patch_pixel_counts = {
                pid: len(pixels) 
                for pid, pixels in patch_mappings['restoration_patches']['patch_to_pixels'].items()
            }
            self.conversion_patch_pixel_counts = {
                pid: len(pixels)
                for pid, pixels in patch_mappings['conversion_patches']['patch_to_pixels'].items()
            }
    
    def _do(self, problem, X, **kwargs):
        import numpy as np
        
        if not hasattr(problem, 'n_restoration_patches'):
            # Fallback for non-patch problems
            return X
        
        n_restoration_patches = problem.n_restoration_patches
        
        # X is 2D array: (population_size, n_var)
        for i in range(len(X)):
            if self.constraint_type == 'patch_count':
                X[i] = enforce_patch_count(X[i], self.target_value)
            
            elif self.constraint_type == 'pixel_count':
                X[i] = self._enforce_pixel_count(
                    X[i], 
                    n_restoration_patches,
                    self.target_value
                )
        
        return X
    
    def _enforce_pixel_count(self, patch_decisions, n_restoration_patches, target_pixels):
        """Enforce pixel count by adding/removing patches within tolerance."""
        result = patch_decisions.copy()
        min_pix = int(target_pixels * (1 - self.pixel_tolerance))
        max_pix = int(target_pixels * (1 + self.pixel_tolerance))

        # Build per-patch pixel vector once.
        all_patch_sizes = np.concatenate([
            np.array([self.restoration_patch_pixel_counts.get(i, 0) for i in range(n_restoration_patches)], dtype=np.int64),
            np.array([self.conversion_patch_pixel_counts.get(i, 0) for i in range(len(result) - n_restoration_patches)], dtype=np.int64),
        ])

        current = int(np.dot(result.astype(np.int64), all_patch_sizes))
        
        if min_pix <= current <= max_pix:
            return result

        score_vec = None
        if self.patch_scores is not None and len(self.patch_scores) == len(result):
            score_vec = np.asarray(self.patch_scores, dtype=np.float64)

        min_patch = max(int(np.min(all_patch_sizes[all_patch_sizes > 0])) if np.any(all_patch_sizes > 0) else 1, 1)
        max_steps = int(max(abs(current - target_pixels) / min_patch * 3.0, 32))
        step = 0
        
        if current < min_pix:
            while current < min_pix and step < max_steps:
                remaining = np.where(result == 0)[0]
                if remaining.size == 0:
                    break

                if score_vec is not None:
                    order = remaining[np.argsort(-score_vec[remaining])]
                else:
                    order = remaining[np.argsort(-all_patch_sizes[remaining])]

                pool_idx = order[:min(self.top_k, len(order))]
                if pool_idx.size == 0:
                    break

                if score_vec is not None:
                    pool_scores = score_vec[pool_idx]
                else:
                    pool_scores = all_patch_sizes[pool_idx].astype(np.float64)

                probs = _safe_softmax(pool_scores, temperature=self.score_temperature)
                chosen = int(_sample_without_replacement_weighted(pool_idx, probs, 1)[0])

                result[chosen] = 1
                current += int(all_patch_sizes[chosen])
                step += 1
        else:
            while current > max_pix and step < max_steps:
                active = np.where(result == 1)[0]
                if active.size == 0:
                    break

                if score_vec is not None:
                    order = active[np.argsort(score_vec[active])]
                else:
                    order = active[np.argsort(all_patch_sizes[active])]

                pool_idx = order[:min(self.top_k, len(order))]
                if pool_idx.size == 0:
                    break

                if score_vec is not None:
                    pool_scores = -score_vec[pool_idx]
                else:
                    pool_scores = -all_patch_sizes[pool_idx].astype(np.float64)

                probs = _safe_softmax(pool_scores, temperature=self.score_temperature)
                chosen = int(_sample_without_replacement_weighted(pool_idx, probs, 1)[0])

                result[chosen] = 0
                current -= int(all_patch_sizes[chosen])
                step += 1
        
        return result


def convert_patch_results_to_pixel_decisions(results_dict, patch_mappings):
    """Convert patch-based results to pixel-level decisions for visualization."""
    patch_decisions = results_dict.get('decisions')
    if patch_decisions is None:
        patch_decisions = results_dict.get('X')
    
    if patch_decisions is None:
        raise ValueError("No decisions found in results")
    
    # Extract n_patches from nested structure
    n_restoration_patches = patch_mappings['restoration_patches']['n_patches']
    n_conversion_patches = patch_mappings['conversion_patches']['n_patches']
    
    # Get pixel counts from problem_info if available
    if 'problem_info' in results_dict:
        n_restoration_pixels = results_dict['problem_info']['n_restoration_pixels']
        n_conversion_pixels = results_dict['problem_info']['n_conversion_pixels']
        n_pixels = n_restoration_pixels + n_conversion_pixels
    else:
        # Fallback: count from initial_conditions if available
        if 'initial_conditions' in results_dict:
            n_restoration_pixels = results_dict['initial_conditions']['n_restoration_pixels']
            n_conversion_pixels = results_dict['initial_conditions']['n_conversion_pixels']
            n_pixels = n_restoration_pixels + n_conversion_pixels
        else:
            raise ValueError("Cannot determine pixel counts from results")
    
    # Convert all solutions
    pixel_decisions_list = []
    for solution in patch_decisions:
        # Split patch decisions into restoration and conversion
        x_restore_patches = solution[:n_restoration_patches]
        x_convert_patches = solution[n_restoration_patches:]
        
        # Convert each type separately
        x_restore_pixels = convert_patch_decisions_to_pixels(
            x_restore_patches,
            patch_mappings['restoration_patches'],
            n_restoration_pixels
        )
        
        x_convert_pixels = convert_patch_decisions_to_pixels(
            x_convert_patches,
            patch_mappings['conversion_patches'],
            n_conversion_pixels
        )
        
        # Combine into single pixel decision vector
        pixel_decision = np.concatenate([x_restore_pixels, x_convert_pixels])
        pixel_decisions_list.append(pixel_decision)
    
    # Update results
    results_dict['decisions'] = np.array(pixel_decisions_list)
    results_dict['decisions_patches'] = patch_decisions
    results_dict['is_patch_based'] = True
    
    return results_dict