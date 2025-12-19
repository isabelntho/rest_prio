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
        'spatial_clustering': [0.0, 0.3, 0.6, 1.0], # Degree of spatial clustering (0=random, 1=highly clustered)
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
# DATA PREPARATION SECTION
# =============================================================================

def load_admin_regions(workspace_dir):
    """
    Load administrative regions from shapefile for burden sharing.
    
    Args:
        workspace_dir: Directory containing admin shapefile
        
    Returns:
        dict: Admin regions data with region_map and region_counts
    """
    import geopandas as gpd
    
    admin_file = 'Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp'
    
    if not os.path.exists(admin_file):
        print(f"Warning: Admin shapefile not found at {admin_file}")
        return None
    
    try:
        # Load admin shapefile
        gdf = gpd.read_file(admin_file)
        region_col = 'NAME'     
        unique_regions = gdf[region_col].unique()
        
        return {
            'gdf': gdf,
            'region_column': region_col,
            'unique_regions': unique_regions,
            'n_regions': len(unique_regions)
        }
        
    except Exception as e:
        print(f"Error loading admin shapefile: {e}")
        return None

def load_initial_conditions(workspace_dir):
    """
    Args:
        workspace_dir: Directory containing input data files (.tif)
    Returns:
        dict: Initial conditions for all objectives
    """
    print("Loading initial objective conditions...")
    
    # Define file paths - modify these to match your data structure
    data_files = {
        'abiotic_anomaly': os.path.join(workspace_dir, 'abiotic_condition_anomaly.tif'),
        'biotic_anomaly': os.path.join(workspace_dir, 'biotic_condition_anomaly.tif'),
        'landscape_anomaly': os.path.join(workspace_dir, 'landscape_condition_anomaly.tif'),
        'implementation_cost': os.path.join(workspace_dir, 'implementation_cost.tif')
    }
    
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
    
    # Load all data files
    for objective, file_path in data_files.items():
        with rio.open(file_path) as src:
            data = src.read(1)
            # Get spatial reference for later use
            if objective == 'abiotic_anomaly':  # Use first file for reference
                initial_conditions['crs'] = src.crs
                initial_conditions['transform'] = src.transform
                initial_conditions['shape'] = data.shape
            initial_conditions[objective] = data
            print(f"✓ Loaded {objective}: {data.shape}")
    
    # Create eligibility mask
    shape = initial_conditions['shape']
    eligible_mask = np.ones(shape, dtype=bool)  # All cells eligible for now
    
    # You might want to exclude certain areas, e.g.:
    # eligible_mask = (initial_conditions['implementation_cost'] < 8000) & \
    #                 (initial_conditions['abiotic_anomaly'] > 0.1)
    
    initial_conditions['eligible_mask'] = eligible_mask
    eligible_indices = np.where(eligible_mask.flatten())[0]
    initial_conditions['eligible_indices'] = eligible_indices
    initial_conditions['n_pixels'] = len(eligible_indices)
    
    # Load admin regions for burden sharing
    admin_data = load_admin_regions(workspace_dir)
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
    
    To customize restoration effects, modify the effect_params values below or 
    pass a custom effect_params dictionary when calling this function.
    
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
            'abiotic_improvement': 0.3,      # Reduction in abiotic anomaly
            'biotic_improvement': 0.4,       # Reduction in biotic anomaly  
            'landscape_improvement': 0.5,    # Reduction in landscape anomaly
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
    
    for objective in ['abiotic_anomaly', 'biotic_anomaly', 'landscape_anomaly']:
        original_values = initial_conditions[objective].copy()
        updated_values = original_values.copy()
        
        if np.any(restoration_mask_2d):
            # Direct effect on restored cells
            improvement = effect_params[f'{objective.split("_")[0]}_improvement']
            updated_values[restoration_mask_2d] = np.maximum(0, 
                original_values[restoration_mask_2d] - improvement)
            
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
                updated_values[neighbor_mask] = np.maximum(0,
                    original_values[neighbor_mask] - neighbor_improvement)
        
        updated_conditions[objective] = updated_values
    
    # Implementation cost (only for restored cells)
    base_cost = initial_conditions['implementation_cost'].copy()
    total_cost = 0
    if np.any(restoration_mask_2d):
        total_cost = np.sum(base_cost[restoration_mask_2d])
    
    updated_conditions['implementation_cost'] = total_cost
    
    return updated_conditions

# =============================================================================
# SPATIAL CLUSTERING FUNCTIONS
# =============================================================================

def apply_burden_sharing(decision_vars, initial_conditions):
    """
    Apply burden sharing to ensure equal restoration across admin regions.
    
    Args:
        decision_vars: Binary array (0/1) for restoration decisions
        initial_conditions: Dict with initial conditions including admin data
        
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
    
    for region_id in range(n_regions):
        region_pixels = np.where(region_assignments == region_id)[0]
        
        if len(region_pixels) == 0:
            continue
            
        # Determine restoration target for this region
        target = restore_per_region + (1 if region_id < extra_restores else 0)
        target = min(target, len(region_pixels))  # Can't restore more pixels than available
        
        if target > 0:
            # Select pixels to restore in this region (random selection for now)
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

# =============================================================================
# OPTIMIZATION PROBLEM DEFINITION
# =============================================================================

class RestorationProblem(ElementwiseProblem):
    """
    Four-objective restoration optimization problem.
    
    Objectives (all to minimize):
    1. Abiotic condition anomaly
    2. Biotic condition anomaly
    3. Landscape condition anomaly
    4. Implementation cost
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
        
        n_pixels = initial_conditions['n_pixels']
        max_restoration_fraction = scenario_params['max_restoration_fraction']
        self.max_restored_pixels = int(max_restoration_fraction * n_pixels)
        
        # Binary decision variables: 0 = no restoration, 1 = restoration
        super().__init__(
            n_var=n_pixels,           # One decision per eligible pixel
            n_obj=4,                  # Four objectives
            n_constr=1,              # Constraint on total restoration area
            xl=0,                    # Lower bound: no restoration
            xu=1,                    # Upper bound: restoration
            type_var=int             # Integer (binary) variables
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution (restoration plan).
        
        Args:
            x: Decision variables (binary array)
            out: Output dictionary for objectives and constraints
        """
        # Apply burden sharing if specified in scenario parameters
        burden_sharing = self.scenario_params.get('burden_sharing', 'no')
        if burden_sharing == 'yes':
            x_processed = apply_burden_sharing(x, self.initial_conditions)
        else:
            x_processed = x
        
        # Apply spatial clustering if specified in scenario parameters
        clustering_strength = self.scenario_params.get('spatial_clustering', 0.0)
        if clustering_strength > 0.0:
            x_clustered = apply_spatial_clustering(x_processed, self.initial_conditions, clustering_strength)
        else:
            x_clustered = x_processed
        
        # Apply restoration effects using clustered decision variables
        updated_conditions = restoration_effect(x_clustered, self.initial_conditions)
        
        # Calculate objective values (all to minimize)
        objectives = []
        
        # 1. Abiotic condition anomaly (sum of remaining anomalies)
        abiotic_objective = np.sum(updated_conditions['abiotic_anomaly'])
        objectives.append(abiotic_objective)
        
        # 2. Biotic condition anomaly (sum of remaining anomalies)
        biotic_objective = np.sum(updated_conditions['biotic_anomaly'])
        objectives.append(biotic_objective)
        
        # 3. Landscape condition anomaly (sum of remaining anomalies)
        landscape_objective = np.sum(updated_conditions['landscape_anomaly'])
        objectives.append(landscape_objective)
        
        # 4. Implementation cost (total cost)
        cost_objective = updated_conditions['implementation_cost']
        objectives.append(cost_objective)
        
        out["F"] = objectives
        
        # Constraint: limit total restoration area (use clustered variables for constraint)
        clustering_strength = self.scenario_params.get('spatial_clustering', 0.0)
        if clustering_strength > 0.0:
            n_restored = np.sum(x_clustered)
        else:
            n_restored = np.sum(x)
        out["G"] = [n_restored - self.max_restored_pixels]  # Violation if positive

# =============================================================================
# OPTIMIZATION EXECUTION
# =============================================================================

def run_single_scenario_optimization(initial_conditions, scenario_params, pop_size=50, 
                                   n_generations=100, save_results=True, verbose=True):
    """
    Run the four-objective restoration optimization for a single scenario.
    
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
    
    # Create optimization problem
    problem = RestorationProblem(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params
    )
    
    # Create optimization algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),  # Binary sampling for 0/1 variables
        crossover=HUX(),                  # Half-uniform crossover
        mutation=BitflipMutation()        # Bit-flip mutation
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
        
        if result is not None and hasattr(result, 'F') and result.F is not None:
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
    results_filename = os.path.join(output_dir, f"restoration_scenario_{scenario_str}_{timestamp}.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary (JSON format)
    summary_filename = os.path.join(output_dir, f"restoration_scenario_{scenario_str}_summary_{timestamp}.json")
    
    objectives = results['objectives']
    summary = {
        'scenario_params': results['scenario_params'],
        'optimization_info': {
            'n_solutions': results['n_solutions'],
            'n_pixels': results['problem_info']['n_pixels'],
            'timestamp': results['algorithm_info']['timestamp']
        },
        'objective_statistics': {
            'abiotic_anomaly': {
                'min': float(objectives[:, 0].min()),
                'max': float(objectives[:, 0].max()),
                'mean': float(objectives[:, 0].mean())
            },
            'biotic_anomaly': {
                'min': float(objectives[:, 1].min()),
                'max': float(objectives[:, 1].max()),
                'mean': float(objectives[:, 1].mean())
            },
            'landscape_anomaly': {
                'min': float(objectives[:, 2].min()),
                'max': float(objectives[:, 2].max()),
                'mean': float(objectives[:, 2].mean())
            },
            'implementation_cost': {
                'min': float(objectives[:, 3].min()),
                'max': float(objectives[:, 3].max()),
                'mean': float(objectives[:, 3].mean())
            }
        }
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
        summary['scenarios'][f'scenario_{scenario_id}'] = {
            'params': scenario_results['scenario_params'],
            'n_solutions': scenario_results['n_solutions'],
            'objective_ranges': {
                'abiotic_anomaly': [float(objectives[:, 0].min()), float(objectives[:, 0].max())],
                'biotic_anomaly': [float(objectives[:, 1].min()), float(objectives[:, 1].max())],
                'landscape_anomaly': [float(objectives[:, 2].min()), float(objectives[:, 2].max())],
                'implementation_cost': [float(objectives[:, 3].min()), float(objectives[:, 3].max())]
            }
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"✓ Combined results saved to: {results_filename}")
        print(f"✓ Combined summary saved to: {summary_filename}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main(workspace_dir=".", scenario='all', pop_size=50, n_generations=100, 
         save_results=True, verbose=True):
    """
    Main execution function for restoration optimization.
    
    Args:
        workspace_dir: Directory containing input data
        scenario: Scenario to run ('all' for all scenarios, or integer index for specific scenario)
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
    
    initial_conditions = load_initial_conditions(workspace_dir)
    
    # Step 2: Display available scenarios
    if verbose:
        scenario_combinations = get_scenario_combinations()
        print(f"\nAvailable scenarios ({len(scenario_combinations)} total):")
        for i, params in enumerate(scenario_combinations):
            print(f"  Scenario {i}: {params}")
    
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
            obj_names = ['Abiotic anomaly', 'Biotic anomaly', 'Landscape anomaly', 'Implementation cost']
            for i, name in enumerate(obj_names):
                print(f"  {name}: {objectives[:, i].min():.2f} - {objectives[:, i].max():.2f}")
    
    return results

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Example usage:
    
    # Run all scenarios (now includes burden sharing variants)
    results = main(
        workspace_dir=".",          # Current directory - change to your data path  
        scenario='all',             # Run all scenarios ('all' or integer index for specific scenario)
        pop_size=30,               # Population size
        n_generations=50,          # Number of generations
        save_results=True,         # Save results to files
        verbose=True               # Print progress
    )
    
    # Or run a single scenario by index:
    # For burden sharing scenarios, ensure admin.shp exists in workspace_dir
    # results = main(
    #     workspace_dir=".",
    #     scenario=0,               # Run scenario 0 (first scenario)
    #     pop_size=30,
    #     n_generations=50,
    #     save_results=True,
    #     verbose=True
    # )
    
    if results is not None:
        print("\n✓ Optimization completed successfully!")
        print("Check the generated .pkl and .json files for detailed results.")
    else:
        print("\n✗ Optimization failed.")