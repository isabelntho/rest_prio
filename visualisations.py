"""
Visualisations for restoration optimization results
Load results from pickle file and create visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import ListedColormap
from patch_approach import (
    convert_patch_decisions_to_pixels,
    create_patch_mappings,
)
from utils import pickle_load as _pickle_load_with_numpy_compat


def _convert_patch_decisions_to_pixel_matrix(decisions, initial_conditions, results=None):
    """Convert patch-level decision matrix to pixel-level [restoration, conversion] matrix.

    Returns None when conversion is not possible.
    """
    decisions = np.asarray(decisions)
    if decisions.ndim != 2:
        return None

    n_var = decisions.shape[1]
    n_rest = len(initial_conditions.get("restoration_eligible_indices", []))
    n_conv = len(initial_conditions.get("conversion_eligible_indices", []))

    # Already in pixel space
    if n_var in (n_rest, n_rest + n_conv, 2 * len(initial_conditions.get("eligible_indices", []))):
        return decisions

    patch_mappings = None
    if results is not None:
        patch_mappings = results.get("patch_mappings")
    if patch_mappings is None:
        patch_mappings = initial_conditions.get("patch_mappings")

    # If patch mappings are missing, try to infer a patch size that matches n_var.
    if patch_mappings is None:
        for ps in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
            try:
                candidate = create_patch_mappings(initial_conditions, patch_size=ps)
                n_rest_p = candidate["restoration_patches"]["n_patches"]
                n_conv_p = candidate["conversion_patches"]["n_patches"]
                if (n_rest_p + n_conv_p) == n_var:
                    patch_mappings = candidate
                    break
            except Exception:
                continue

    if patch_mappings is None:
        return None

    n_rest_patches = patch_mappings["restoration_patches"]["n_patches"]
    n_conv_patches = patch_mappings["conversion_patches"]["n_patches"]
    if (n_rest_patches + n_conv_patches) != n_var:
        return None

    pixel_decisions = np.zeros((decisions.shape[0], n_rest + n_conv), dtype=np.int8)
    for i in range(decisions.shape[0]):
        x_patch = decisions[i]
        x_restore_p = x_patch[:n_rest_patches]
        x_convert_p = x_patch[n_rest_patches:]

        x_restore = convert_patch_decisions_to_pixels(
            x_restore_p,
            patch_mappings["restoration_patches"],
            n_rest,
        )
        x_convert = convert_patch_decisions_to_pixels(
            x_convert_p,
            patch_mappings["conversion_patches"],
            n_conv,
        )
        pixel_decisions[i, :n_rest] = x_restore
        pixel_decisions[i, n_rest:] = x_convert

    return pixel_decisions


def plot_eligible_pixels(initial_conditions, save_path=None, figsize=(12, 10), pad=0, title=None):
    """
    Plot eligible pixels for restoration, conversion, and both actions.
    
    Args:
        initial_conditions: Dict from load_initial_conditions containing eligible masks
        save_path: Optional path to save the figure
        figsize: Figure size tuple (width, height)
        pad: Padding around eligible region for cropping
        title: Custom title for the plot
        
    Returns:
        matplotlib figure and axis objects, plus the eligibility map
    """
    # Get the separate eligible masks
    restoration_mask = initial_conditions['restoration_eligible_mask']
    conversion_mask = initial_conditions['conversion_eligible_mask']
    shape = initial_conditions['shape']
    
    # Create eligibility map with 4 categories:
    eligibility_map = np.zeros(shape, dtype=int)
    
    # Set restoration-only pixels (1)
    restoration_only = restoration_mask & ~conversion_mask
    eligibility_map[restoration_only] = 1
    
    # Set conversion-only pixels (2)
    conversion_only = conversion_mask & ~restoration_mask
    eligibility_map[conversion_only] = 2
    
    # Set pixels eligible for both (3)
    both_eligible = restoration_mask & conversion_mask
    eligibility_map[both_eligible] = 3
    
    # Count pixels in each category
    n_restoration_only = np.sum(restoration_only)
    n_conversion_only = np.sum(conversion_only)
    n_both = np.sum(both_eligible)
    n_total_restoration = np.sum(restoration_mask)
    n_total_conversion = np.sum(conversion_mask)
    
    print(f"Eligibility Summary:")
    print(f"  Restoration only: {n_restoration_only:,} pixels")
    print(f"  Conversion only: {n_conversion_only:,} pixels") 
    print(f"  Both actions: {n_both:,} pixels")
    print(f"  Total restoration eligible: {n_total_restoration:,} pixels")
    print(f"  Total conversion eligible: {n_total_conversion:,} pixels")
    
    # Debug: Check if the issue is with the calculation
    print(f"Debug checks:")
    print(f"  conversion_mask max value: {conversion_mask.max()}")
    print(f"  conversion_mask sum: {conversion_mask.sum()}")
    print(f"  conversion_only max value: {conversion_only.max()}")
    print(f"  conversion_only sum: {conversion_only.sum()}")
    
    # Check what values are actually in the eligibility map
    unique_values, counts = np.unique(eligibility_map, return_counts=True)
    print(f"Eligibility map values and counts:")
    for val, count in zip(unique_values, counts):
        category_name = ["Not eligible", "Restoration only", "Conversion only", "Both eligible"][int(val)]
        print(f"  Value {val} ({category_name}): {count:,} pixels")
    
    # Crop to region of interest using combined eligible pixels for bounds
    combined_eligible_indices = np.where((restoration_mask | conversion_mask).flatten())[0]
    eligibility_cropped = crop_to_eligible(eligibility_map, combined_eligible_indices, shape, pad=pad)
    
    # Debug: Check what values are in the cropped map
    unique_cropped, counts_cropped = np.unique(eligibility_cropped, return_counts=True)
    print(f"Cropped eligibility map values and counts:")
    for val, count in zip(unique_cropped, counts_cropped):
        if val == 0: cat = "Not eligible"
        elif val == 1: cat = "Restoration only" 
        elif val == 2: cat = "Conversion only"
        elif val == 3: cat = "Both eligible"
        else: cat = f"Unknown ({val})"
        print(f"  Value {val} ({cat}): {count:,} pixels")
    
    # Create custom colormap and labels
    # Since we have no conversion-only pixels (value 2), we need to handle this carefully
    colors = [
        '#F0F0F0',  # Light grey for not eligible (value 0)
        '#2E8B57',  # Sea green for restoration only (value 1)
        '#2E8B57',  # Duplicate green for unused value 2 (to prevent interpolation artifacts)
        '#FFD700'   # Gold for both eligible (value 3)
    ]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors)
    
    # Use discrete boundaries to prevent interpolation
    boundaries = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(boundaries, cmap.N)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(eligibility_cropped, cmap=cmap, norm=norm)
    
    ax.axis("off")
    
    # Set title
    if title is None:
        title = f"Eligible Pixels for Restoration and Conversion Actions"
    ax.set_title(title, fontsize=14, pad=20)
    
    # Create custom legend
    from matplotlib.patches import Patch
    
    # Calculate total eligible pixels correctly
    total_eligible_pixels = n_restoration_only + n_conversion_only + n_both
    n_not_eligible = shape[0] * shape[1] - total_eligible_pixels
    
    legend_elements = [
        Patch(facecolor=colors[0], label=f'Not eligible ({n_not_eligible:,} pixels)'),
        Patch(facecolor=colors[1], label=f'Restoration only ({n_restoration_only:,} pixels)'),
        Patch(facecolor=colors[3], label=f'Restoration and Conversion ({n_both:,} pixels)')
    ]
    # Note: Omitting "Conversion only" since n_conversion_only = 0
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved eligibility map to: {save_path}")
    
    plt.show()
    
    return fig, ax, eligibility_map


def plot_objectives_correlation_matrix(initial_conditions, save_path=None, figsize=(8, 6)):
    """
    Plot correlation matrix for all objectives in initial conditions. 
    
    Args:
        initial_conditions: Dict with objective arrays (from load_initial_conditions)
        save_path: Optional path to save the figure
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib figure object
    """
    # Identify available objectives
    objective_keys = []
    potential_objectives = ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly', 
                          'implementation_cost', 'population_proximity']
    
    for key in potential_objectives:
        if key in initial_conditions:
            objective_keys.append(key)
    
    if len(objective_keys) < 2:
        print(f"Warning: Only {len(objective_keys)} objective(s) found. Need at least 2 for correlation matrix.")
        return None
    
    # Extract data for eligible pixels only
    eligible_mask = initial_conditions['eligible_mask']
    data = {}
    
    for key in objective_keys:
        values = initial_conditions[key][eligible_mask]
        # Remove any remaining NaN values
        if np.any(np.isnan(values)):
            print(f"Warning: NaN values found in {key}, excluding from correlation")
            continue
        data[key] = values
    
    if len(data) < 2:
        print("Warning: Less than 2 valid objectives after NaN removal.")
        return None
    
    # Create DataFrame and compute correlation matrix
    df = pd.DataFrame(data)
    corr_matrix = df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns], rotation=45)
    ax.set_yticklabels([idx.replace('_', ' ').title() for idx in corr_matrix.index])
    
    # Add correlation values as text
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                         fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    # Set title
    n_pixels = len(list(data.values())[0])
    ax.set_title(f'Objectives Correlation Matrix\n({n_pixels:,} eligible pixels)', 
                fontsize=12, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to: {save_path}")
    
    return fig


def load_results(pkl_path, scenario_id=None, for_plotting_all=False):
    """
    Load optimization results from pickle file.
    
    Args:
        pkl_path: Path to .pkl file containing optimization results
        scenario_id: For multi-scenario files, specify which scenario to load (default: None)
                    - None: Load scenario 0 for multi-scenario files, or the single scenario
                    - int: Load specific scenario by ID
                    - 'all': Load all scenarios (returns list of scenario results)
        for_plotting_all (bool): Internal flag to prevent filtering when the full
                                 population is needed for a plot.
        
    Returns:
        dict or list: 
            - If single scenario or specific scenario_id: Results dictionary
            - If scenario_id='all': List of results dictionaries for all scenarios
    """
    raw_results = _pickle_load_with_numpy_compat(pkl_path)

    # Detect file structure
    if 'scenarios' in raw_results:
        # Multi-scenario file
        if scenario_id == 'all':
            # Load all scenarios
            all_results = []
            print(f"Multi-scenario file detected. Loading all {len(raw_results['scenarios'])} scenarios.")
            
            for sid in sorted(raw_results['scenarios'].keys()):
                scenario_results = raw_results['scenarios'][sid].copy()
                
                # Add metadata to each scenario
                scenario_results['multi_scenario_info'] = {
                    'total_scenarios': raw_results['n_scenarios_total'],
                    'scenarios_run': raw_results['n_scenarios_run'],
                    'current_scenario_id': sid,
                    'all_scenario_ids': list(raw_results['scenarios'].keys())
                }
                
                # Process objectives for this scenario
                scenario_results = _process_objectives(scenario_results)
                all_results.append(scenario_results)
            
            print(f"Loaded {len(all_results)} scenarios successfully.")
            return all_results
        
        else:
            # Load single scenario
            if scenario_id is None:
                scenario_id = 0  # Default to first scenario
                print(f"Multi-scenario file detected. Loading scenario {scenario_id} (default).")
                print(f"Available scenarios: {list(raw_results['scenarios'].keys())}")
            
            if scenario_id not in raw_results['scenarios']:
                available = list(raw_results['scenarios'].keys())
                raise ValueError(f"Scenario {scenario_id} not found. Available scenarios: {available}")
            
            # Extract the specific scenario
            results = raw_results['scenarios'][scenario_id].copy()
            
            # Add some metadata from the parent structure
            results['multi_scenario_info'] = {
                'total_scenarios': raw_results['n_scenarios_total'],
                'scenarios_run': raw_results['n_scenarios_run'],
                'current_scenario_id': scenario_id,
                'all_scenario_ids': list(raw_results['scenarios'].keys())
            }
            
            print(f"Loaded scenario {scenario_id} with parameters: {results['scenario_params']}")
    
    else:
        # Single scenario file
        if scenario_id == 'all':
            print("Single scenario file detected. Returning as single-item list.")
            results = raw_results
            results = _process_objectives(results)
            return [results]  # Return as list for consistency
        
        results = raw_results
        print("Single scenario file detected.")
        if scenario_id is not None and scenario_id != 'all':
            print(f"Warning: scenario_id={scenario_id} specified but file contains only one scenario")

    # Process objectives for single scenario result
    results = _process_objectives(results)
    
    # If we need the full population for plotting, we're done.
    if for_plotting_all:
        return results

    # Filter for non-dominated solutions if the flag exists
    if 'is_nondominated' in results:
        print("Filtering for non-dominated solutions.")
        
        # Store the full population before filtering
        results['full_population'] = {
            'objectives': results.get('objectives'),
            'decisions': results.get('decisions'),
            'objectives_normalized': results.get('objectives_normalized'),
        }
        
        non_dominated_mask = results['is_nondominated']
        
        # Filter the main keys
        results['objectives'] = results['objectives'][non_dominated_mask]
        results['decisions'] = results['decisions'][non_dominated_mask]
        if 'objectives_normalized' in results and results['objectives_normalized'] is not None:
            results['objectives_normalized'] = results['objectives_normalized'][non_dominated_mask]
        
        # Update solution counts
        results['n_solutions'] = len(results['objectives'])
        results['n_nondominated_solutions'] = results['n_solutions']
        
        print(f"Found {results['n_solutions']} non-dominated solutions.")

    return results


def _process_objectives(results):
    """
    Helper function to process objectives (convert anomaly objectives to percentages).
    
    Args:
        results: Results dictionary containing objectives and problem info
        
    Returns:
        dict: Results with processed objectives
    """
    # Debug: Check what keys are available
    print(f"Available keys: {list(results.keys())}")
    if 'objectives' not in results:
        print(f"Error: 'objectives' key not found in results.")
        print(f"Available keys: {list(results.keys())}")
        raise KeyError("'objectives' key missing from results dictionary")
    
    anomaly_names = ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']
    F = results['objectives'].astype(float)
    n_pixels = results['problem_info']['n_pixels']
   # obj_names = results['objective_names']   # list in same order as columns ADD BACK IN later

    #for j, name in enumerate(obj_names):
    #    if name in anomaly_names:
    #        F[:, j] = (-F[:, j] / n_pixels) * 100.0

    results['objectives'] = F
    return results

import numpy as np
import matplotlib.pyplot as plt

def decision_to_map_2d(decision_vec, initial_conditions, fill_value=np.nan, action_type='combined'):
    """
    Convert a decision vector on eligible pixels to a 2D raster map.
    decision_vec: shape (n_eligible,) or (2*n_eligible,) or (n_restoration + n_conversion,) or (n_pixels_if_you_stored_full,)
    Returns a 2D array with 1 for selected eligible pixels, 0 for non selected eligible pixels.
    Non eligible pixels are fill_value.
    
    Args:
        decision_vec: Decision vector containing restoration/conversion decisions
        initial_conditions: Initial conditions dictionary
        fill_value: Value to use for non-eligible pixels
        action_type: Which actions to plot ('combined', 'restoration', 'conversion')
                    'combined': Plot both restoration and conversion (they shouldn't overlap)
                    'restoration': Plot only restoration decisions (first part of vector)
                    'conversion': Plot only conversion decisions (second part of vector)
    """
    shape = initial_conditions["shape"]
    
    # Get indices for both restoration and conversion eligible pixels
    restoration_indices = initial_conditions.get("restoration_eligible_indices", 
                                                 initial_conditions["eligible_indices"])  # Fallback to old format
    conversion_indices = initial_conditions.get("conversion_eligible_indices", [])
    
    n_restoration = len(restoration_indices)
    n_conversion = len(conversion_indices)
    n_total_separate = n_restoration + n_conversion
    
    # Legacy support
    eligible_indices = initial_conditions["eligible_indices"]
    n_eligible = len(eligible_indices)

    m = np.full(shape, fill_value, dtype=float)
    decision_vec = np.asarray(decision_vec).astype(float)

    if decision_vec.size == n_eligible:
        # Old format: single decision per eligible pixel
        m.flat[eligible_indices] = decision_vec
    elif decision_vec.size == 2 * n_eligible:
        # Old format: [restoration_decisions, conversion_decisions] on same pixels
        restore_decisions = decision_vec[:n_eligible]
        convert_decisions = decision_vec[n_eligible:]
        
        if action_type == 'restoration':
            m.flat[eligible_indices] = restore_decisions
        elif action_type == 'conversion':
            m.flat[eligible_indices] = convert_decisions
        elif action_type == 'combined':
            # Combine both types (they shouldn't overlap, so sum should be safe)
            combined_decisions = restore_decisions + convert_decisions
            m.flat[eligible_indices] = combined_decisions
        else:
            raise ValueError(f"action_type must be 'combined', 'restoration', or 'conversion', got {action_type}")
    elif decision_vec.size == n_total_separate and n_conversion > 0:
        # New format: [restoration_decisions, conversion_decisions] on separate pixel sets
        restore_decisions = decision_vec[:n_restoration]
        convert_decisions = decision_vec[n_restoration:]
        
        if action_type == 'restoration':
            m.flat[restoration_indices] = restore_decisions
        elif action_type == 'conversion':
            m.flat[conversion_indices] = convert_decisions
        elif action_type == 'combined':
            # Plot both restoration and conversion on their respective pixels
            m.flat[restoration_indices] = restore_decisions
            m.flat[conversion_indices] = convert_decisions
        else:
            raise ValueError(f"action_type must be 'combined', 'restoration', or 'conversion', got {action_type}")
    elif decision_vec.size == np.prod(shape):
        # Full raster format
        m = decision_vec.reshape(shape)
    else:
        raise ValueError(
            f"decision_vec length {decision_vec.size} does not match any expected format:\n"
            f"  - Legacy eligible: {n_eligible}\n"
            f"  - Legacy 2*eligible: {2*n_eligible}\n" 
            f"  - Separate sets (restoration + conversion): {n_total_separate} ({n_restoration} + {n_conversion})\n"
            f"  - Full raster: {np.prod(shape)}"
        )

    return m

def plot_example_solution(
    pkl_path,
    scenario_id=0,
    solution_id=None,
    choose="best_sum",   # "best_sum", "best_first", "random", "index"
    title=None,
    cmap="Greys",
    show_eligible=True,
    pad=0,
    save_path=None,
    seed=1,
    action_type='combined',  # 'combined', 'restoration', 'conversion'
):
    """
    Plot a single Pareto solution as a spatial map.
    """

    # Load scenario
    results = load_results(pkl_path, scenario_id=scenario_id)
    decisions = results["decisions"]               # (n_solutions, n_eligible)
    objectives = results["objectives"]             # (n_solutions, n_obj)
    ic = results["initial_conditions"]
    eligible_indices = ic["eligible_indices"]
    shape = ic["shape"]

    n_solutions = decisions.shape[0]

    # Choose solution
    if choose == "index":
        if solution_id is None:
            raise ValueError("choose='index' requires solution_id.")
        sol_idx = int(solution_id)

    elif choose == "best_sum":
        sol_idx = int(np.argmin(np.sum(objectives, axis=1)))

    elif choose == "best_first":
        sol_idx = int(np.argmin(objectives[:, 0]))

    elif choose == "random":
        rng = np.random.default_rng(seed)
        sol_idx = int(rng.integers(0, n_solutions))

    else:
        raise ValueError("choose must be one of: 'best_sum', 'best_first', 'random', 'index'.")

    x = decisions[sol_idx]

    # Build map
    if show_eligible:
        m2d = decision_to_map_2d(x, ic, fill_value=-1.0, action_type=action_type)  # -1 for non eligible
    else:
        m2d = decision_to_map_2d(x, ic, fill_value=np.nan, action_type=action_type)

    m_plot = crop_to_eligible(m2d, eligible_indices, shape, pad=pad)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if show_eligible:
        im = ax.imshow(m_plot, cmap=cmap, vmin=-1, vmax=1)
    else:
        im = ax.imshow(m_plot, cmap=cmap, vmin=0, vmax=1)

    ax.axis("off")

    if title is None:
        # Count actions by type for more informative title
        if x.size == 2 * len(eligible_indices):
            n_restore = int(np.sum(x[:len(eligible_indices)]))
            n_convert = int(np.sum(x[len(eligible_indices):]))
            title = f"Scenario {scenario_id}, solution {sol_idx} ({choose}), restore: {n_restore}, convert: {n_convert}"
        else:
            title = f"Scenario {scenario_id}, solution {sol_idx} ({choose}), selected pixels: {int(np.sum(x))}"
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax, m2d, sol_idx


import numpy as np
import pandas as pd

def selection_freq_baseline_summary(
    raw_results,
    pkl_path,
    scenario_id=None,
    objectives=("abiotic_anomaly", "biotic_anomaly", "landscape_anomaly","implementation_cost"),
    core_thr=0.8,
    rare_thr=0.2,
    include_params=True,
    action_type='combined'
):
    """
    One function for either:
      - a single scenario (scenario_id provided), or
      - all scenarios (scenario_id is None)

    Returns:
      summary_df: tidy table, one row per (scenario, objective)
      arrays: dict keyed by scenario_id with optional large arrays:
              arrays[sid]["sel_freq"], arrays[sid]["a0"][obj]
              (only computed for requested objectives)
              
    Args:
        action_type: Which actions to analyze ('combined', 'restoration', 'conversion')
    """

    scenarios = raw_results["scenarios"]
    ic = raw_results["initial_conditions"]
    elig_idx = ic["eligible_indices"]
    n_eligible = len(elig_idx)
    
    # Get separate restoration and conversion indices for new format
    restoration_indices = ic.get("restoration_eligible_indices", elig_idx)  # Fallback to old format
    conversion_indices = ic.get("conversion_eligible_indices", [])
    n_restoration = len(restoration_indices)
    n_conversion = len(conversion_indices)
    n_total_separate = n_restoration + n_conversion

    # Helper: baseline anomalies at eligible pixels
    def baseline_at_eligible(arr2d, elig_idx):
        return arr2d.ravel()[elig_idx]

    # Decide which scenarios to process
    if scenario_id is None:
        sids = list(scenarios.keys())
    else:
        if scenario_id not in scenarios:
            raise KeyError(f"scenario_id {scenario_id} not found. Available: {list(scenarios.keys())[:10]} ...")
        sids = [scenario_id]

    rows = []
    arrays = {}

    for sid in sids:
        sc = scenarios[sid]
        X = sc.get("decisions", None)
        if X is None or len(X) == 0:
            continue

        # Handle expanded decision vectors (restoration + conversion)
        if X.shape[1] == 2 * n_eligible:
            # Old format: [restoration_decisions, conversion_decisions] on same pixels
            if action_type == 'restoration':
                X_analysis = X[:, :n_eligible]
            elif action_type == 'conversion': 
                X_analysis = X[:, n_eligible:]
            elif action_type == 'combined':
                # Combine restoration and conversion decisions
                X_restoration = X[:, :n_eligible]
                X_conversion = X[:, n_eligible:]
                X_analysis = X_restoration + X_conversion  # Element-wise sum
            else:
                raise ValueError(f"action_type must be 'combined', 'restoration', or 'conversion', got {action_type}")
        elif X.shape[1] == n_total_separate and n_conversion > 0:
            # New format: [restoration_decisions, conversion_decisions] on separate pixel sets
            X_restoration = X[:, :n_restoration]
            X_conversion = X[:, n_restoration:]
            
            if action_type == 'restoration':
                X_analysis = X_restoration
                # Use restoration eligible indices for analysis
                elig_idx_analysis = restoration_indices
            elif action_type == 'conversion':
                X_analysis = X_conversion
                # Use conversion eligible indices for analysis 
                elig_idx_analysis = conversion_indices
            elif action_type == 'combined':
                # For combined, we need to map both to the full eligible space
                # This is complex so for now just use restoration data as primary
                X_analysis = X_restoration
                elig_idx_analysis = restoration_indices
                print(f"Warning: Combined analysis with separate pixel sets using restoration data only")
            else:
                raise ValueError(f"action_type must be 'combined', 'restoration', or 'conversion', got {action_type}")
        elif X.shape[1] == n_eligible:
            # Legacy format - single decision per pixel
            X_analysis = X
            elig_idx_analysis = elig_idx
        else:
            raise ValueError(f"Unexpected decision matrix shape: {X.shape}, expected (*, {n_eligible}) or (*, {2*n_eligible}) or (*, {n_total_separate})")

        sel_freq = X_analysis.mean(axis=0)
        core = sel_freq >= core_thr
        rare = sel_freq <= rare_thr

        params = sc.get("scenario_params", {}) or {}
        arrays[sid] = {"sel_freq": sel_freq, "a0": {}}

        for obj in objectives:
            if obj not in ic:
                continue

            vals = baseline_at_eligible(ic[obj], elig_idx)
            arrays[sid]["a0"][obj] = vals

            # correlation guard
            if np.std(vals) == 0 or np.std(sel_freq) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(vals, sel_freq)[0, 1]

            row = {
                "scenario_id": sid,
                "objective": obj,
                "action_type": action_type,
                "corr_baseline_vs_sel_freq": float(corr) if np.isfinite(corr) else None,
                "mean_baseline_core": float(np.mean(vals[core])) if np.any(core) else None,
                "mean_baseline_rare": float(np.mean(vals[rare])) if np.any(rare) else None,
                "q10_core": float(np.quantile(vals[core], 0.10)) if np.any(core) else None,
                "q90_core": float(np.quantile(vals[core], 0.90)) if np.any(core) else None,
                "q10_rare": float(np.quantile(vals[rare], 0.10)) if np.any(rare) else None,
                "q90_rare": float(np.quantile(vals[rare], 0.90)) if np.any(rare) else None,
                "n_core": int(np.sum(core)),
                "n_rare": int(np.sum(rare)),
                "n_pixels": int(len(sel_freq)),
                "n_solutions": int(X.shape[0]),
            }

            if include_params:
                row.update({
                    "max_restoration_fraction": params.get("max_restoration_fraction"),
                    "spatial_clustering": params.get("spatial_clustering"),
                    "burden_sharing": params.get("burden_sharing"),
                    "abiotic_effect": params.get("abiotic_effect"),
                    "biotic_effect": params.get("biotic_effect"),
                    "landscape_effect": params.get("landscape_effect"),
                })

            rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df["core_minus_rare_mean"] = summary_df["mean_baseline_core"] - summary_df["mean_baseline_rare"]
    csv_path = pkl_path.replace('.pkl', f'_selection_corr_{action_type}.csv')
    pdf = pd.DataFrame(rows)
    pdf.to_csv(csv_path, index=False)

    return summary_df, arrays


def export_scenarios_to_csv(pkl_path, csv_path=None, solution_method='all_pareto'):
    """
    Export multi-scenario results to CSV with one row per Pareto optimal solution.
    
    Args:
        pkl_path: Path to .pkl file containing multi-scenario results
        csv_path: Path for output CSV file (if None, uses pkl_path with .csv extension)
        solution_method: How to export solutions:
                        'all_pareto' - one row per Pareto optimal solution (default)
                        'best_sum' - one row per scenario with solution having minimum sum of objectives
                        'best_first' - one row per scenario with solution having minimum first objective
    
    Returns:
        pandas.DataFrame: DataFrame with solution data
    """
    import pandas as pd
    import os
    
    # Load all scenarios
    all_scenarios = load_results(pkl_path, scenario_id='all')
    
    if csv_path is None:
        csv_path = pkl_path.replace('.pkl', '_solutions.csv')
    
    # Collect data for each solution
    solution_data = []
    
    for scenario_idx, results in enumerate(all_scenarios):
        params = results['scenario_params']
        objectives = results['objectives']
        decisions = results['decisions']
        obj_names = results.get('objective_names', [f'objective_{i}' for i in range(objectives.shape[1])])
        
        if solution_method == 'all_pareto':
            # Export all Pareto optimal solutions
            for sol_idx in range(objectives.shape[0]):
                # Create row data for this solution
                row_data = {
                    'scenario_id': scenario_idx,
                    'solution_id': sol_idx,
                    'n_pixels_restored': int(np.sum(decisions[sol_idx]))
                }
                
                # Add objective values
                for i, obj_name in enumerate(obj_names):
                    row_data[obj_name] = objectives[sol_idx, i]
                
                # Add parameter values (same for all solutions in scenario)
                for param_name, param_value in params.items():
                    row_data[param_name] = param_value
                
                solution_data.append(row_data)
                
        elif solution_method in ['best_sum', 'best_first']:
            # Export only one solution per scenario
            if solution_method == 'best_sum':
                best_idx = np.argmin(np.sum(objectives, axis=1))
            else:  # best_first
                best_idx = np.argmin(objectives[:, 0])
            
            row_data = {
                'scenario_id': scenario_idx,
                'solution_id': best_idx,
                'n_pixels_restored': int(np.sum(decisions[best_idx])),
                'total_pareto_solutions': objectives.shape[0]
            }
            
            # Add objective values
            for i, obj_name in enumerate(obj_names):
                row_data[obj_name] = objectives[best_idx, i]
            
            # Add parameter values
            for param_name, param_value in params.items():
                row_data[param_name] = param_value
            
            solution_data.append(row_data)
        
        else:
            raise ValueError(f"Unknown solution_method: {solution_method}. Use 'all_pareto', 'best_sum', or 'best_first'")
    
    # Create DataFrame
    df = pd.DataFrame(solution_data)
    
    # Reorder columns: identifiers, objectives, parameters, metadata
    id_columns = ['scenario_id', 'solution_id']
    obj_columns = [col for col in df.columns if col in obj_names]
    param_columns = [col for col in df.columns if col in params.keys()]
    meta_columns = [col for col in df.columns if col not in id_columns + obj_columns + param_columns]
    
    column_order = id_columns + obj_columns + param_columns + meta_columns
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    total_solutions = len(df)
    n_scenarios = df['scenario_id'].nunique()
    
    print(f"✓ Exported {total_solutions} solutions from {n_scenarios} scenarios to: {csv_path}")
    print(f"  Method: {solution_method}")
    if solution_method == 'all_pareto':
        avg_solutions = total_solutions / n_scenarios
        print(f"  Average solutions per scenario: {avg_solutions:.1f}")
    print(f"  Columns: {list(df.columns)}")
    
    return df

def crop_to_eligible(m, eligible_indices, shape, pad=0):
    rows, cols = np.unravel_index(eligible_indices, shape)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()

    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(shape[0] - 1, r1 + pad)
    c1 = min(shape[1] - 1, c1 + pad)

    return m[r0:r1 + 1, c0:c1 + 1]

def create_selection_frequency_map(pkl_path, save_path=None, cmap='YlOrRd', title=None, raster_save_path=None, show_eligible=True, action_type='combined'):
    """
    Create a map showing the % with which each pixel is selected across Pareto-optimal solutions.
    
    Args:
        pkl_path: Path to pickle file with optimization results
        save_path: Optional path to save the figure (e.g., 'selection_frequency.png')
        cmap: Colormap to use for frequency (default: 'YlOrRd')
        title: Custom title for the plot
        show_eligible: If True, show eligible-but-not-selected pixels in grey (default: True)
        action_type: Which actions to plot ('combined', 'restoration', 'conversion')
        
    Returns:
        tuple: (fig, ax, frequency_map_2d) - figure, axes, and the 2D frequency map
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # Load results
    results = load_results(pkl_path)
    
    # Extract key information
    decisions = results['decisions']  # May be pixel-space or patch-space
    initial_conditions = results['initial_conditions']
    shape = initial_conditions['shape']
    eligible_indices = initial_conditions['eligible_indices']
    eligible_mask = initial_conditions['eligible_mask']
    n_solutions = results['n_solutions']
    n_eligible = len(eligible_indices)
    
    # Get separate restoration and conversion indices for new format
    restoration_indices = initial_conditions.get("restoration_eligible_indices", eligible_indices)  # Fallback to old format
    conversion_indices = initial_conditions.get("conversion_eligible_indices", [])
    n_restoration = len(restoration_indices)
    n_conversion = len(conversion_indices)
    n_total_separate = n_restoration + n_conversion
    
    print(f"Loaded {n_solutions} Pareto-optimal solutions")
    print(f"Raster shape: {shape}")
    print(f"Eligible pixels: {n_eligible}")
    # Convert patch-space decisions to pixel-space when needed.
    decisions_converted = _convert_patch_decisions_to_pixel_matrix(
        decisions,
        initial_conditions,
        results=results,
    )
    if decisions_converted is not None:
        decisions = decisions_converted

    print(f"Decision vector shape: {decisions.shape}")
    
    # Calculate selection frequency for each eligible pixel
    if decisions.shape[1] == 2 * n_eligible:
        # Old format with restoration and conversion decisions on same pixels
        if action_type == 'restoration':
            selection_frequency = np.sum(decisions[:, :n_eligible], axis=0) / n_solutions * 100
            print(f"Calculating restoration frequency")
        elif action_type == 'conversion':
            selection_frequency = np.sum(decisions[:, n_eligible:], axis=0) / n_solutions * 100
            print(f"Calculating conversion frequency")
        elif action_type == 'combined':
            # Combine both types of decisions
            restore_freq = decisions[:, :n_eligible]
            convert_freq = decisions[:, n_eligible:]
            combined_decisions = restore_freq + convert_freq  # Element-wise sum
            selection_frequency = np.sum(combined_decisions, axis=0) / n_solutions * 100
            print(f"Calculating combined restoration + conversion frequency")
        else:
            raise ValueError(f"action_type must be 'combined', 'restoration', or 'conversion', got {action_type}")
        # Use legacy eligible indices for mapping
        mapping_indices = eligible_indices
    elif decisions.shape[1] == n_total_separate and n_conversion > 0:
        # New format: [restoration_decisions, conversion_decisions] on separate pixel sets
        if action_type == 'restoration':
            selection_frequency = np.sum(decisions[:, :n_restoration], axis=0) / n_solutions * 100
            mapping_indices = restoration_indices
            print(f"Calculating restoration frequency for separate pixel sets")
        elif action_type == 'conversion':
            selection_frequency = np.sum(decisions[:, n_restoration:], axis=0) / n_solutions * 100
            mapping_indices = conversion_indices
            print(f"Calculating conversion frequency for separate pixel sets")
        elif action_type == 'combined':
            # For combined, create frequency map showing both restoration and conversion.
            # Keep non-action-eligible pixels as NaN so they can be rendered in grey when show_eligible=True.
            selection_frequency_map = np.full(shape, np.nan, dtype=float)
            action_eligible_indices = np.unique(
                np.concatenate([np.asarray(restoration_indices), np.asarray(conversion_indices)])
            )
            selection_frequency_map.flat[action_eligible_indices] = 0.0

            # Map restoration frequencies
            restore_freq = np.sum(decisions[:, :n_restoration], axis=0) / n_solutions * 100
            selection_frequency_map.flat[restoration_indices] = restore_freq

            # Add conversion frequencies (they should not overlap)
            convert_freq = np.sum(decisions[:, n_restoration:], axis=0) / n_solutions * 100
            selection_frequency_map.flat[conversion_indices] = convert_freq

            # For the return format, we'll use the map directly
            selection_frequency = None  # Will use selection_frequency_map instead
            mapping_indices = None
            print(f"Calculating combined restoration + conversion frequency for separate pixel sets")
        else:
            raise ValueError(f"action_type must be 'combined', 'restoration', or 'conversion', got {action_type}")
    elif decisions.shape[1] == n_eligible:
        # Old format with single decision per pixel
        selection_frequency = np.sum(decisions, axis=0) / n_solutions * 100
        mapping_indices = eligible_indices
        print(f"Using legacy single-decision format")
    else:
        raise ValueError(f"Unexpected decision matrix shape: {decisions.shape}, expected ({n_solutions}, {n_eligible}) or ({n_solutions}, {2*n_eligible}) or ({n_solutions}, {n_total_separate})")
    
    # Create frequency map
    if selection_frequency is not None:
        # Standard approach: map selection frequencies to their pixel locations
        print(f"Selection frequency range: {selection_frequency.min():.1f}% - {selection_frequency.max():.1f}%")
        
        # Count pixels never selected
        never_selected = np.sum(selection_frequency == 0)
        print(f"Eligible pixels never selected: {never_selected} ({100*never_selected/len(mapping_indices):.1f}%)")
        
        # Create 2D map with special handling for eligible-but-not-selected
        if show_eligible:
            # Use -1 for non-eligible, 0-100 for frequencies
            frequency_map_2d = np.full(shape, -1.0)  # -1 for non-eligible (will show as grey)
            frequency_map_2d.flat[mapping_indices] = selection_frequency
        else:
            # Original behavior: NaN for non-eligible
            frequency_map_2d = np.full(shape, np.nan)
            frequency_map_2d.flat[mapping_indices] = selection_frequency
    else:
        # Combined approach for separate pixel sets: already have the full map
        frequency_map_2d = selection_frequency_map
        valid_pixels = frequency_map_2d[~np.isnan(frequency_map_2d)]
        if len(valid_pixels) > 0:
            print(f"Selection frequency range: {valid_pixels.min():.1f}% - {valid_pixels.max():.1f}%")
            never_selected = np.sum(valid_pixels == 0)
            print(f"Eligible pixels never selected: {never_selected} ({100*never_selected/len(valid_pixels):.1f}%)")
        else:
            print("No valid frequency data found")
            
        # Handle show_eligible option for combined map
        if show_eligible:
            # Replace NaN with -1 to show non-eligible areas
            frequency_map_2d = np.where(np.isnan(frequency_map_2d), -1.0, frequency_map_2d)

    #Save raster
    if raster_save_path is not None:
        if 'transform' not in initial_conditions or 'crs' not in initial_conditions:
                raise KeyError(
                    "To write a raster, initial_conditions must contain either "
                    "'raster_profile' or both 'transform' and 'crs'."
               )
        transform = initial_conditions['transform']
        crs = initial_conditions['crs']
        nodata = initial_conditions.get('nodata', -1.0 if show_eligible else -9999.0)
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': crs,
            'transform': transform,
            'nodata': float(nodata),
            'compress': 'deflate',
            'tiled': True
        }

        out = frequency_map_2d.astype('float32', copy=False)
        if not show_eligible:
            out = np.where(np.isfinite(out), out, float(nodata)).astype('float32', copy=False)
        else:
           # keep -1 as nodata by default, unless a different nodata was provided
           if float(nodata) != -1.0:
               out = np.where(out == -1.0, float(nodata), out).astype('float32', copy=False)
        import rasterio
        with rasterio.open(raster_save_path, 'w', **profile) as dst:
            dst.write(out, 1)
        print(f"✓ Selection frequency raster saved to: {raster_save_path}")


    frequency_map_plot = crop_to_eligible(frequency_map_2d, eligible_indices, shape, pad=0)
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    if show_eligible:
        # Create custom colormap: grey for non-eligible, then frequency colors
        base_cmap = plt.cm.get_cmap(cmap)
        colors = ['#CCCCCC']  # Grey for non-eligible pixels
        # Add colors from the main colormap for 0-100%
        colors.extend([base_cmap(i) for i in np.linspace(0, 1, 256)])
        custom_cmap = ListedColormap(colors)
        
        # Plot with custom range: -1 to 100
        im = ax.imshow(frequency_map_plot , cmap=custom_cmap, vmin=-1, vmax=100)
        
        # Add colorbar only for 0-100 range (skip the grey)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('RFOP (%)', rotation=270, labelpad=20, fontsize=12)

    else:
        # Standard plot with NaN
        im = ax.imshow(frequency_map_plot, cmap=cmap, vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('RFOP (%)', rotation=270, labelpad=20, fontsize=12)
    
    ax.axis('off')
    subtitle_text = f"Eligible pixels never selected: {never_selected} ({100*never_selected/len(eligible_indices):.1f}%)"
    action_title = f" ({action_type.title()} Actions)" if action_type != 'combined' else " (Combined Actions)"
    plot_title = f"{title or 'Selection Frequency'}{action_title}\n{subtitle_text}"
    plt.title(plot_title)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Selection frequency map saved to: {save_path}")
    
    plt.show()
    
    return fig, ax, frequency_map_2d
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _param_matches(stored, target, rtol=1e-9, atol=1e-12):
    if stored is None:
        return False
    if isinstance(stored, (int, float, np.integer, np.floating)) and isinstance(
        target, (int, float, np.integer, np.floating)
    ):
        return np.isclose(float(stored), float(target), rtol=rtol, atol=atol)
    return stored == target

def create_param_controlled_frequency_maps_multiscenario(
    pkl_path,
    param_names,
    fixed_values=dict(),
    scenario_filter=None,
    save_path=None,
    cmap="YlOrRd",
    show_eligible=True,
    suptitle=None,
):
    """
    2 by 2 figure, each panel fixes one scenario parameter to a chosen value.
    Each panel shows pooled relative selection frequency across all scenarios that match.

    Args:
        pkl_path: path to a multi scenario pickle
        param_names: list of exactly 4 parameter names, keys in results['scenario_params']
        fixed_values: dict {param_name: value} for each of the 4 parameters
        scenario_filter: optional function f(scenario_results) -> bool, extra filtering
        save_path: optional path to save figure
        cmap: matplotlib colormap name
        show_eligible: if True, non eligible pixels are grey (value -1)
        suptitle: optional figure title

    Returns:
        fig, axes, maps_by_param
    """
    for p in param_names:
        if p not in fixed_values:
            raise ValueError(f"fixed_values is missing a value for {p}.")

    all_results = load_results(pkl_path, scenario_id="all")

    if scenario_filter is not None:
        all_results = [r for r in all_results if scenario_filter(r)]

    if len(all_results) == 0:
        raise ValueError("No scenarios remain after filtering.")
    # Reference geometry, eligibility
    ic0 = all_results[0]["initial_conditions"]
    shape = ic0["shape"]
    eligible_indices = ic0["eligible_indices"]

    # Validate consistency across scenarios
    for r in all_results[1:]:
        ic = r["initial_conditions"]
        if ic["shape"] != shape or not np.array_equal(ic["eligible_indices"], eligible_indices):
            raise ValueError("All scenarios must share the same shape and eligible_indices to pool maps.")

    # Map builder
    def to_map_2d(selection_frequency_percent):
        if show_eligible:
            m = np.full(shape, -1.0)
            m.flat[eligible_indices] = selection_frequency_percent
            return m
        m = np.full(shape, np.nan)
        m.flat[eligible_indices] = selection_frequency_percent
        return m
    
    def crop_to_eligible(m, eligible_indices, shape, pad=0):
        rows, cols = np.unravel_index(eligible_indices, shape)
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        r0 = max(0, r0 - pad)
        c0 = max(0, c0 - pad)
        r1 = min(shape[0] - 1, r1 + pad)
        c1 = min(shape[1] - 1, c1 + pad)

        return m[r0:r1 + 1, c0:c1 + 1]
    # Colormap with grey for non eligible pixels
    custom_cmap = None
    if show_eligible:
        base_cmap = plt.cm.get_cmap(cmap)
        colors = ["#CCCCCC"]
        colors.extend([base_cmap(i) for i in np.linspace(0, 1, 256)])
        custom_cmap = ListedColormap(colors)

    maps_by_param = {}
    meta_by_param = {}

    for vary_param in param_names:
        # For this parameter, we want to see variation while keeping others fixed
        # Find scenarios where all OTHER parameters match fixed values
        matching = []
        for r in all_results:
            params = r["scenario_params"]
            # Check that all parameters EXCEPT vary_param match fixed values
            matches_all_others = True
            for p in param_names:
                if p == vary_param:
                    continue  # Skip the parameter we're varying
                if not _param_matches(params.get(p), fixed_values[p]):
                    matches_all_others = False
                    break
            
            if matches_all_others:
                matching.append(r)
        
        if len(matching) == 0:
            maps_by_param[vary_param] = to_map_2d(np.zeros(len(eligible_indices)))
            meta_by_param[vary_param] = {"n_scenarios": 0, "n_solutions": 0, "param_values": []}
            continue

        decisions_list = []
        param_values_used = []
        for r in matching:
            decisions_list.append(r["decisions"])
            param_values_used.append(r["scenario_params"].get(vary_param))

        decisions_concat = np.vstack(decisions_list)  # shape (sum_solutions, n_pixels)
        selection_frequency = np.sum(decisions_concat, axis=0) / decisions_concat.shape[0] * 100.0

        maps_by_param[vary_param] = to_map_2d(selection_frequency)
        meta_by_param[vary_param] = {
            "n_scenarios": len(matching),
            "n_solutions": decisions_concat.shape[0],
            "param_values": sorted(list(set(param_values_used)))  # Unique values of the varying parameter
        }

    # Plot 2 by 2 with one shared colorbar
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    ims = []
    for k, vary_param in enumerate(param_names):
        ax = axes[k]
        m = maps_by_param[vary_param]
        m_plot = crop_to_eligible(m, eligible_indices, shape, pad=0)

        if show_eligible:
            im = ax.imshow(m_plot, cmap=custom_cmap, vmin=-1, vmax=100)
        else:
            im = ax.imshow(m_plot, cmap=cmap, vmin=0, vmax=100)

        ax.axis("off")
        
        # Create title showing which parameter varies and what others are fixed to
        other_params = [p for p in param_names if p != vary_param]
        fixed_str = ", ".join([f"{p}={fixed_values[p]}" for p in other_params])
        param_vals = meta_by_param[vary_param]['param_values']
        param_range = f"{min(param_vals):.3f}-{max(param_vals):.3f}" if param_vals else "none"
        
        ax.set_title(
            f"Varying {vary_param} ({param_range}), {meta_by_param[vary_param]['n_solutions']} solutions",
            fontsize=10
        )
        ims.append(im)

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # reserve margin
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("RFOP (%)", rotation=270, labelpad=20, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, axes, maps_by_param


def create_effect_summary_visualization(pkl_path, save_path="effect_summary.png"):
    """
    Show the EFFECTS of parameters rather than all individual maps.
    
    Args:
        pkl_path: Path to pickle file with combined results
        save_path: Where to save the visualization
    """
    all_scenarios = load_results(pkl_path, scenario_id='all')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract effect metrics from all scenarios
    effects_data = []
    for scenario_idx, results in enumerate(all_scenarios):
        params = results['scenario_params']
        objectives = results['objectives']
        
        # Calculate effect metrics (e.g., best solution per scenario)
        best_solution_idx = np.argmin(np.sum(objectives, axis=1))  # Or use other criteria
        effects_data.append({
            'restoration_fraction': params['max_restoration_fraction'],
            'spatial_clustering': params['spatial_clustering'],
            'burden_sharing': params['burden_sharing'],
            'total_anomaly_reduction': np.sum(objectives[best_solution_idx][:3]),  # Sum of 3 anomalies
            #'cost_efficiency': objectives[best_solution_idx][3] / np.sum(objectives[best_solution_idx][:3]),
            'scenario_id': scenario_idx
        })
    
    df = pd.DataFrame(effects_data)
    
    # 1. Restoration Fraction Effect (heatmap)
    pivot_frac = df.pivot_table(values='total_anomaly_reduction', 
                               index='spatial_clustering', 
                               columns='restoration_fraction')
    sns.heatmap(pivot_frac, annot=True, fmt='.0f', ax=axes[0,0], cmap='RdYlBu')
    axes[0,0].set_title('Anomaly Reduction by\nRestoration Fraction & Clustering')
    
    # 2. Burden Sharing Effect (box plots)
    df_burden = df.melt(id_vars=['burden_sharing'], 
                       value_vars=['total_anomaly_reduction'])#, 'cost_efficiency')
    sns.boxplot(data=df_burden, x='burden_sharing', y='value', hue='variable', ax=axes[0,1])
    axes[0,1].set_title('Burden Sharing Effects')
    
    # 3. Cost-Effectiveness Scatter
    for burden in ['no', 'yes']:
        subset = df[df['burden_sharing'] == burden]
        axes[0,2].scatter(subset['cost_efficiency'], subset['total_anomaly_reduction'], 
                         label=f'Burden sharing: {burden}', alpha=0.7)
    axes[0,2].set_xlabel('Cost Efficiency')
    axes[0,2].set_ylabel('Total Anomaly Reduction')
    axes[0,2].set_title('Cost vs Effectiveness')
    axes[0,2].legend()
    
    # 4-6. Parameter interaction effects
    # ... additional effect visualizations
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_pareto_front(
    pkl_path,
    save_path=None,
    figsize=(10, 8),
    alpha=0.7,
    show_all_solutions=False,
    scenario_id=None,
):
    """
    Plot Pareto front from optimization results.

    Args:
        pkl_path (str): Path to the pickle file containing results.
        save_path (str, optional): Path to save the figure. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        alpha (float, optional): Transparency of the points. Defaults to 0.7.
        show_all_solutions (bool, optional): If True, plots all solutions from the
            final population and highlights the non-dominated ones. Defaults to False.
        scenario_id (int, optional): The ID of the scenario to load from a
            multi-scenario result file. Defaults to the first available scenario.
    """
    results = load_results(pkl_path, scenario_id=scenario_id)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    plot_title = "Pareto Front"
    obj_names = results.get("objective_names", ["Objective 1", "Objective 2", "Objective 3"])

    if show_all_solutions:
        # When showing all, we need the full unfiltered data
        results = load_results(pkl_path, scenario_id=scenario_id, for_plotting_all=True)
        all_objectives = results.get("objectives")
        is_nondominated = results.get("is_nondominated")

        if all_objectives is None:
            print(f"Warning: No objectives found in {pkl_path}")
            return None
        
        # Use the third objective for color if available
        if all_objectives.shape[1] > 2:
            colors = all_objectives[:, 2]
            c_label = obj_names[2].replace("_", " ").title()
        else:
            colors = "grey"
            c_label = ""

        # Plot all points first
        sc = ax.scatter(
            all_objectives[:, 0],
            all_objectives[:, 1],
            c=colors,
            cmap="viridis",
            s=60,
            alpha=alpha,
            label=f"All Solutions ({len(all_objectives)})"
        )

        # Then, overlay the red outline on the non-dominated points
        if is_nondominated is not None and np.any(is_nondominated):
            ax.scatter(
                all_objectives[is_nondominated, 0],
                all_objectives[is_nondominated, 1],
                s=60, # Match size
                facecolors='none',
                edgecolors='red',
                linewidths=1.5,
                label=f'Non-dominated ({np.sum(is_nondominated)})'
            )
        
        if all_objectives.shape[1] > 2:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(c_label)
        
        ax.legend()
        plot_title = "All Final Solutions"

    else:
        # Original behavior: load pre-filtered non-dominated front
        results = load_results(pkl_path, scenario_id=scenario_id)
        objectives = results.get("objectives")
        if objectives is None:
            print(f"Warning: No objectives found in {pkl_path}")
            return None
            
        if objectives.shape[1] > 2:
            colors = objectives[:, 2]
            c_label = obj_names[2].replace("_", " ").title()
        else:
            colors = "tab:blue"
            c_label = ""

        sc = ax.scatter(
            objectives[:, 0],
            objectives[:, 1],
            c=colors,
            cmap="viridis",
            s=60,
            alpha=alpha,
        )
        
        if objectives.shape[1] > 2:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(c_label)

    ax.set_xlabel(obj_names[0].replace("_", " ").title())
    ax.set_ylabel(obj_names[1].replace("_", " ").title())
    ax.set_title(plot_title)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved pareto front to: {save_path}")

    return fig

def plot_parallel_coordinates(pkl_path, save_path=None, figsize=(12, 7), alpha_background=1):
    """
    Plot parallel coordinates showing all objectives.
    Each objective is a vertical axis, each solution is a line.
    Dominated solutions are drawn in grey, non-dominated solutions in black,
    and the best solution for each objective is highlighted in a distinct color.
    
    Args:
        pkl_path: Path to pickle file with optimization results
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        alpha_background: Transparency for dominated (grey) solutions
        
    Returns:
        fig: matplotlib figure
    """
    # Load full population so we can distinguish dominated from non-dominated
    results = load_results(pkl_path, for_plotting_all=True)
    objectives = results['objectives']
    is_nondominated = results.get('is_nondominated')
    n_solutions, n_objectives = objectives.shape

    # Fall back: treat all solutions as non-dominated if flag is absent
    if is_nondominated is None:
        is_nondominated = np.ones(n_solutions, dtype=bool)

    n_nondominated = int(np.sum(is_nondominated))
    n_dominated = n_solutions - n_nondominated

    # Get objective names
    from resto_anom import RestorationProblem
    problem = RestorationProblem(results['initial_conditions'], results['scenario_params'])
    
    # Clean up names for display
    display_names = {'abiotic_anomaly': 'Abiotic\nAnomaly', 'biotic_anomaly': 'Biotic\nAnomaly',
                     'landscape_anomaly': 'Landscape\nAnomaly', 'implementation_cost': 'Implementation\nCost',
                     'population_proximity': 'Population\nProximity'}
    labels = [display_names.get(name, name) for name in problem.objective_names]
    
    print(f"Plotting parallel coordinates: {n_solutions} solutions total "
          f"({n_nondominated} non-dominated, {n_dominated} dominated)")
    
    # Normalize objectives to [0, 1] for visualization using the full population range
    objectives_norm = np.zeros_like(objectives)
    for i in range(n_objectives):
        obj_min, obj_max = objectives[:, i].min(), objectives[:, i].max()
        if obj_max > obj_min:
            objectives_norm[:, i] = (objectives[:, i] - obj_min) / (obj_max - obj_min)
        else:
            objectives_norm[:, i] = 0.5
    
    # Find best solution for each objective among non-dominated solutions only
    best_solutions = {}
    highlight_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # Distinct colors
    nd_indices = np.where(is_nondominated)[0]
    for i in range(n_objectives):
        best_nd_pos = np.argmin(objectives[nd_indices, i])
        best_idx = int(nd_indices[best_nd_pos])
        best_solutions[i] = {'idx': best_idx, 'color': highlight_colors[i % len(highlight_colors)], 'label': labels[i]}
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_positions = np.arange(n_objectives)

    # Layer 1: dominated solutions in grey (background)
    dominated_plotted = False
    for sol_idx in range(n_solutions):
        if not is_nondominated[sol_idx]:
            ax.plot(x_positions, objectives_norm[sol_idx, :],
                    color='grey', alpha=alpha_background, linewidth=0.8, zorder=1,
                    label='Dominated' if not dominated_plotted else '_nolegend_')
            dominated_plotted = True

    # Layer 2: non-dominated solutions in black
    nd_plotted = False
    best_indices = {info['idx'] for info in best_solutions.values()}
    for sol_idx in nd_indices:
        if sol_idx not in best_indices:
            ax.plot(x_positions, objectives_norm[sol_idx, :],
                    color='black', alpha=0.9, linewidth=0.9, zorder=2,
                    label='Non-dominated' if not nd_plotted else '_nolegend_')
            nd_plotted = True

    # Layer 3: best solutions with distinct colors (foreground)
    plotted_indices = set()
    for obj_idx, best_info in best_solutions.items():
        sol_idx = best_info['idx']
        if sol_idx not in plotted_indices:
            ax.plot(x_positions, objectives_norm[sol_idx, :],
                    color=best_info['color'], alpha=0.9, linewidth=2.5,
                    label=f"Best for {best_info['label'].replace(chr(10), ' ')}", zorder=3)
            plotted_indices.add(sol_idx)
    
    # Format axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Normalized Objective Value\n(0 = best, 1 = worst)', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f'Parallel Coordinates Plot ({n_nondominated} non-dominated / {n_solutions} total solutions)',
        fontsize=14, pad=15
    )
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Parallel coordinates plot saved to: {save_path}")


# =============================================================================
# PATCH SIZE COMPARISON GRIDS
# =============================================================================
# Functions for generating side-by-side comparison plots across multiple
# patch sizes, given a list of pre-loaded run results (run_data).
#
# run_data format:
#   [{"patch_size": int, "seed": int, "initial_conditions": dict,
#     "decisions": ndarray, "F": ndarray}, ...]
# =============================================================================

def _extract_decisions(results):
    """Get decision matrix from results dict with fallback key names."""
    if results is None:
        return None
    if "decisions" in results:
        return np.asarray(results["decisions"])
    if "X" in results:
        return np.asarray(results["X"])
    return None


def _extract_objectives(results):
    """Get objective matrix from results dict with fallback key names."""
    if results is None:
        return None
    if "objectives" in results:
        return np.asarray(results["objectives"])
    if "F" in results:
        return np.asarray(results["F"])
    return None


def _build_patch_comparison_frequency_map(decisions, initial_conditions):
    """Build raster of restoration selection frequency (% across Pareto solutions).

    Handles both pixel-space and patch-space decision vectors.
    """
    shape = initial_conditions["shape"]
    n_rest = int(initial_conditions["n_restoration_pixels"])
    n_conv = int(initial_conditions.get("n_conversion_pixels", 0))
    rest_indices = initial_conditions["restoration_eligible_indices"]

    freq_map = np.full(shape, np.nan, dtype=np.float64)
    if decisions is None or decisions.size == 0:
        return freq_map

    decisions = np.asarray(decisions)
    n_var = decisions.shape[1]

    if n_var == (n_rest + n_conv):
        rest_decisions = decisions[:, :n_rest]
        freq = np.mean(rest_decisions, axis=0) * 100.0
    elif n_var == n_rest:
        freq = np.mean(decisions, axis=0) * 100.0
    else:
        patch_mappings = initial_conditions.get("patch_mappings")
        if patch_mappings is None:
            raise ValueError(
                f"Decision vector length does not match pixel space and patch_mappings are missing. "
                f"Got n_var={n_var}, expected {n_rest + n_conv}."
            )
        n_rest_patches = int(patch_mappings["restoration_patches"]["n_patches"])
        n_conv_patches = int(patch_mappings["conversion_patches"]["n_patches"])
        expected_patch_var = n_rest_patches + n_conv_patches
        if n_var != expected_patch_var:
            raise ValueError(
                f"Decision vector length matches neither pixel nor patch space. "
                f"Got n_var={n_var}, expected pixel={n_rest + n_conv}, patch={expected_patch_var}."
            )
        rest_pixel_solutions = np.zeros((decisions.shape[0], n_rest), dtype=np.int8)
        for i in range(decisions.shape[0]):
            rest_pixel_solutions[i] = convert_patch_decisions_to_pixels(
                decisions[i, :n_rest_patches],
                patch_mappings["restoration_patches"],
                n_rest,
            )
        freq = np.mean(rest_pixel_solutions, axis=0) * 100.0

    if len(freq) != len(rest_indices):
        raise ValueError(
            f"Selection frequency length does not match restoration index length: "
            f"freq={len(freq)}, indices={len(rest_indices)}"
        )
    flat = freq_map.ravel()
    flat[rest_indices] = freq
    return flat.reshape(shape)


def _plot_patch_comparison_parallel_coords(ax, F):
    """Simple parallel coordinates plot for a single run."""
    if F is None or len(F) == 0:
        ax.text(0.5, 0.5, "No objective data", ha="center", va="center")
        ax.set_axis_off()
        return

    vals = np.column_stack([-F[:, 0], -F[:, 1], F[:, 2]])
    labels = ["Abiotic improve", "Biotic improve", "Cost"]
    mins = np.nanmin(vals, axis=0)
    maxs = np.nanmax(vals, axis=0)
    spans = np.where((maxs - mins) <= 1e-12, 1.0, maxs - mins)
    vals_n = (vals - mins) / spans

    x = np.arange(vals_n.shape[1])
    cost_norm = vals_n[:, 2]
    cmap = plt.cm.viridis
    for i in range(vals_n.shape[0]):
        ax.plot(x, vals_n[i], color=cmap(cost_norm[i]), alpha=0.25, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates")
    ax.grid(alpha=0.3)


def _plot_parallel_coordinates_with_global_scale(ax, F, mins, maxs):
    """Parallel coordinates normalized with shared limits across patch sizes."""
    if F is None or len(F) == 0:
        ax.text(0.5, 0.5, "No objective data", ha="center", va="center")
        ax.set_axis_off()
        return

    vals = np.column_stack([-F[:, 0], -F[:, 1], F[:, 2]])
    labels = ["Abiotic improve", "Biotic improve", "Cost"]
    spans = np.where((maxs - mins) <= 1e-12, 1.0, maxs - mins)
    vals_n = (vals - mins) / spans

    x = np.arange(vals_n.shape[1])
    cost_norm = vals_n[:, 2]
    cmap = plt.cm.viridis
    for i in range(vals_n.shape[0]):
        ax.plot(x, vals_n[i], color=cmap(cost_norm[i]), alpha=0.25, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value")
    ax.grid(alpha=0.3)


def create_selection_frequency_grid(run_data, output_dir):
    """Create one 2x2 selection-frequency grid across patch sizes.

    Parameters
    ----------
    run_data : list of dict
        Each dict must have keys: ``patch_size``, ``decisions``,
        ``initial_conditions``.
    output_dir : str
        Directory where the PNG is saved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    im = None

    for ax, item in zip(axes, run_data):
        freq_map = _build_patch_comparison_frequency_map(
            item["decisions"], item["initial_conditions"]
        )
        im = ax.imshow(freq_map, cmap="viridis", vmin=0, vmax=100)
        ax.set_title(f"Patch size {item['patch_size']}x{item['patch_size']}")
        ax.set_axis_off()

    for ax in axes[len(run_data):]:
        ax.set_axis_off()

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("% selected")
    fig.suptitle("Selection frequency across patch sizes", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(output_dir, "selection_frequency_grid.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_pareto_front_grid(run_data, output_dir):
    """Create one 2x2 Pareto-front grid across patch sizes.

    Parameters
    ----------
    run_data : list of dict
        Each dict must have keys: ``patch_size``, ``F``.
    output_dir : str
        Directory where the PNG is saved.
    """
    all_F = [item["F"] for item in run_data if item["F"] is not None and len(item["F"]) > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    if all_F:
        F_all = np.vstack(all_F)
        x_all = -F_all[:, 0]
        y_all = -F_all[:, 1]
        cmin, cmax = np.nanmin(F_all[:, 2]), np.nanmax(F_all[:, 2])
        xlim = (np.nanmin(x_all), np.nanmax(x_all))
        ylim = (np.nanmin(y_all), np.nanmax(y_all))
    else:
        cmin, cmax = 0.0, 1.0
        xlim = ylim = (0.0, 1.0)

    sc = None
    for ax, item in zip(axes, run_data):
        F = item["F"]
        if F is not None and len(F) > 0:
            sc = ax.scatter(-F[:, 0], -F[:, 1], c=F[:, 2], cmap="plasma", s=25,
                            alpha=0.85, vmin=cmin, vmax=cmax)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel("Abiotic improvement")
            ax.set_ylabel("Biotic improvement")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Pareto data", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(f"Patch size {item['patch_size']}x{item['patch_size']}")

    for ax in axes[len(run_data):]:
        ax.set_axis_off()

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Cost")

    fig.suptitle("Pareto front across patch sizes", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(output_dir, "pareto_front_grid.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_parallel_coordinates_grid(run_data, output_dir):
    """Create one 2x2 parallel-coordinates grid across patch sizes.

    Parameters
    ----------
    run_data : list of dict
        Each dict must have keys: ``patch_size``, ``F``.
    output_dir : str
        Directory where the PNG is saved.
    """
    all_vals = [
        np.column_stack([-item["F"][:, 0], -item["F"][:, 1], item["F"][:, 2]])
        for item in run_data
        if item["F"] is not None and len(item["F"]) > 0
    ]
    if all_vals:
        vals_all = np.vstack(all_vals)
        mins = np.nanmin(vals_all, axis=0)
        maxs = np.nanmax(vals_all, axis=0)
    else:
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([1.0, 1.0, 1.0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, item in zip(axes, run_data):
        _plot_parallel_coordinates_with_global_scale(ax, item["F"], mins, maxs)
        ax.set_title(f"Patch size {item['patch_size']}x{item['patch_size']}")

    for ax in axes[len(run_data):]:
        ax.set_axis_off()

    fig.suptitle("Parallel coordinates across patch sizes", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(output_dir, "parallel_coordinates_grid.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_patch_size_comparison_grids(run_data, output_dir):
    """Generate all three 2x2 comparison grid plots for a patch-size sweep.

    Creates selection-frequency, Pareto-front, and parallel-coordinates grids
    side-by-side across patch sizes and saves them to ``output_dir``.

    Parameters
    ----------
    run_data : list of dict
        List of run result dicts.  Each dict must have keys: ``patch_size``,
        ``seed``, ``initial_conditions``, ``decisions``, ``F``.
    output_dir : str
        Directory where the three PNG files are saved.

    Returns
    -------
    list of str
        Paths to the three saved PNG files.
    """
    p1 = create_selection_frequency_grid(run_data, output_dir)
    p2 = create_pareto_front_grid(run_data, output_dir)
    p3 = create_parallel_coordinates_grid(run_data, output_dir)
    return [p1, p2, p3]

    
    plt.show()
    return fig


def show_key_spatial_contrasts(pkl_path, save_path="spatial_contrasts.png"):
    """
    Show only 4-6 representative maps that highlight key differences.
    
    Args:
        pkl_path: Path to pickle file with combined results
        save_path: Where to save the visualization
    """
    all_scenarios = load_results(pkl_path, scenario_id='all')
    
    # Select representative scenarios that show clear contrasts
    key_scenarios = {
        'Low restoration, no clustering, no burden sharing': (0.15, 0.0, 'no'),
        'High restoration, high clustering, no burden sharing': (0.50, 1.0, 'no'), 
        'Medium restoration, no clustering, burden sharing': (0.25, 0.0, 'yes'),
        'Medium restoration, high clustering, burden sharing': (0.25, 1.0, 'yes')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (title, (frac, clust, burden)) in enumerate(key_scenarios.items()):
        # Find matching scenario
        for scenario_idx, results in enumerate(all_scenarios):
            params = results['scenario_params']
            if (params['max_restoration_fraction'] == frac and 
                params['spatial_clustering'] == clust and 
                params['burden_sharing'] == burden):
                
                # Get best solution and convert to spatial map
                objectives = results['objectives']
                decisions = results['decisions']
                best_idx = np.argmin(np.sum(objectives, axis=1))
                
                # Create spatial visualization (simplified)
                restoration_map = decision_to_map_2d(
                    decisions[best_idx],
                    results['initial_conditions'],
                    fill_value=np.nan,
                    action_type='combined',
                )
                
                axes[i].imshow(restoration_map, cmap='RdYlBu_r')
                axes[i].set_title(title)
                axes[i].axis('off')
                break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

### Performance ranking table (which scenarios work best for what)

def quantify_parameter_effects(pkl_path, save_path=None, figsize=(10, 6)):
    """
    Quantify how much each parameter affects objectives (statistical summary).
    
    Args:
        pkl_path: Path to pickle file with combined results
        save_path: Optional path to save visualization
        figsize: Figure size for plots
        
    Returns:
        DataFrame with effect sizes and significance tests
    """
    all_scenarios = load_results(pkl_path, scenario_id='all')
    
    # Extract all solutions from all scenarios
    all_solutions = []
    for scenario_idx, results in enumerate(all_scenarios):
        params = results['scenario_params']
        objectives = results['objectives']
        
        for obj_values in objectives:
            all_solutions.append({
                'restoration_fraction': params['max_restoration_fraction'],
                'spatial_clustering': params['spatial_clustering'], 
                'burden_sharing': params['burden_sharing'],
                'abiotic_anomaly': obj_values[0],
                'biotic_anomaly': obj_values[1],
                'landscape_anomaly': obj_values[2],
                'implementation_cost': obj_values[3],
                'total_anomaly': sum(obj_values[:3])
            })
    
    df = pd.DataFrame(all_solutions)
    
    # Calculate effect sizes for each parameter
    effects_summary = {}
    
    for param in ['restoration_fraction', 'spatial_clustering', 'burden_sharing']:
        for objective in ['total_anomaly', 'implementation_cost']:
            if param == 'burden_sharing':
                # Categorical comparison
                no_burden = df[df[param] == 'no'][objective]
                yes_burden = df[df[param] == 'yes'][objective]
                effect_size = abs(yes_burden.mean() - no_burden.mean()) / no_burden.std()
                p_value = stats.ttest_ind(no_burden, yes_burden)[1]
            else:
                # Continuous correlation
                correlation = df[param].corr(df[objective])
                effect_size = abs(correlation)
                p_value = stats.pearsonr(df[param], df[objective])[1]
            
            effects_summary[f'{param}_on_{objective}'] = {
                'effect_size': effect_size,
                'p_value': p_value,
                'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            }
    
    return pd.DataFrame(effects_summary).T
