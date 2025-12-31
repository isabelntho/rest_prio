"""
Visualisations for restoration optimization results
Load results from pickle file and create visualizations
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_results(pkl_path):
    """
    Load optimization results from pickle file.
    
    Args:
        pkl_path: Path to .pkl file containing optimization results
        
    Returns:
        dict: Results dictionary with scenario_params, objectives, decisions, etc.
    """
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results


def create_selection_frequency_map(pkl_path, save_path=None, cmap='YlOrRd', title=None, show_eligible=True):
    """
    Create a map showing the % with which each pixel is selected across Pareto-optimal solutions.
    
    Args:
        pkl_path: Path to pickle file with optimization results
        save_path: Optional path to save the figure (e.g., 'selection_frequency.png')
        cmap: Colormap to use for frequency (default: 'YlOrRd')
        title: Custom title for the plot
        show_eligible: If True, show eligible-but-not-selected pixels in grey (default: True)
        
    Returns:
        tuple: (fig, ax, frequency_map_2d) - figure, axes, and the 2D frequency map
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # Load results
    results = load_results(pkl_path)
    
    # Extract key information
    decisions = results['decisions']  # Shape: (n_solutions, n_pixels)
    initial_conditions = results['initial_conditions']
    shape = initial_conditions['shape']
    eligible_indices = initial_conditions['eligible_indices']
    eligible_mask = initial_conditions['eligible_mask']
    n_solutions = results['n_solutions']
    
    print(f"Loaded {n_solutions} Pareto-optimal solutions")
    print(f"Raster shape: {shape}")
    print(f"Eligible pixels: {len(eligible_indices)}")
    
    # Calculate selection frequency for each eligible pixel
    selection_frequency = np.sum(decisions, axis=0) / n_solutions * 100  # Convert to percentage
    
    print(f"Selection frequency range: {selection_frequency.min():.1f}% - {selection_frequency.max():.1f}%")
    
    # Count pixels never selected
    never_selected = np.sum(selection_frequency == 0)
    print(f"Eligible pixels never selected: {never_selected} ({100*never_selected/len(eligible_indices):.1f}%)")
    
    # Create 2D map with special handling for eligible-but-not-selected
    if show_eligible:
        # Use -1 for non-eligible, 0-100 for frequencies
        frequency_map_2d = np.full(shape, -1.0)  # -1 for non-eligible (will show as grey)
        frequency_map_2d.flat[eligible_indices] = selection_frequency
    else:
        # Original behavior: NaN for non-eligible
        frequency_map_2d = np.full(shape, np.nan)
        frequency_map_2d.flat[eligible_indices] = selection_frequency
    
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
        im = ax.imshow(frequency_map_2d, cmap=custom_cmap, vmin=-1, vmax=100)
        
        # Add colorbar only for 0-100 range (skip the grey)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Selection Frequency (%)', rotation=270, labelpad=20, fontsize=12)

    else:
        # Standard plot with NaN
        im = ax.imshow(frequency_map_2d, cmap=cmap, vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Selection Frequency (%)', rotation=270, labelpad=20, fontsize=12)
    
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Selection frequency map saved to: {save_path}")
    
    plt.show()
    
    return fig, ax, frequency_map_2d

# Parameter effect heatmaps (what you get for changing parameters)

def create_effect_summary_visualization(pkl_path, save_path="effect_summary.png"):
    """
    Show the EFFECTS of parameters rather than all individual maps.
    
    Args:
        pkl_path: Path to pickle file with combined results
        save_path: Where to save the visualization
    """
    combined_results = load_results(pkl_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract effect metrics from all scenarios
    effects_data = []
    for scenario_id, results in combined_results['scenarios'].items():
        params = results['scenario_params']
        objectives = results['objectives']
        
        # Calculate effect metrics (e.g., best solution per scenario)
        best_solution_idx = np.argmin(np.sum(objectives, axis=1))  # Or use other criteria
        effects_data.append({
            'restoration_fraction': params['max_restoration_fraction'],
            'spatial_clustering': params['spatial_clustering'],
            'burden_sharing': params['burden_sharing'],
            'total_anomaly_reduction': np.sum(objectives[best_solution_idx][:3]),  # Sum of 3 anomalies
            'cost_efficiency': objectives[best_solution_idx][3] / np.sum(objectives[best_solution_idx][:3]),
            'scenario_id': scenario_id
        })
    
    df = pd.DataFrame(effects_data)
    
    # 1. Restoration Fraction Effect (heatmap)
    pivot_frac = df.pivot_table(values='total_anomaly_reduction', 
                               index='spatial_clustering', 
                               columns='max_restoration_fraction')
    sns.heatmap(pivot_frac, annot=True, fmt='.0f', ax=axes[0,0], cmap='RdYlBu')
    axes[0,0].set_title('Anomaly Reduction by\nRestoration Fraction & Clustering')
    
    # 2. Burden Sharing Effect (box plots)
    df_burden = df.melt(id_vars=['burden_sharing'], 
                       value_vars=['total_anomaly_reduction', 'cost_efficiency'])
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


def plot_pareto_front(pkl_path, save_path=None, figsize=(10, 8), alpha=0.7):
    """
    Plot the Pareto front from optimization results.
    """
    # Load results
    results = load_results(pkl_path)
    objectives = results['objectives']
    n_solutions, n_objectives = objectives.shape
    
    # Get objective names
    from resto_anom import RestorationProblem
    problem = RestorationProblem(results['initial_conditions'], results['scenario_params'])
    
    # Clean up names for display
    display_names = {'abiotic_anomaly': 'Abiotic Anomaly', 'biotic_anomaly': 'Biotic Anomaly',
                     'landscape_anomaly': 'Landscape Anomaly', 'implementation_cost': 'Implementation Cost'}
    labels = [display_names.get(name, name) for name in problem.objective_names]
    
    print(f"Plotting Pareto front: {n_solutions} solutions, {n_objectives} objectives")
    
    # Setup scatter plot parameters
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    scatter_params = {'alpha': alpha, 'edgecolors': 'black', 'linewidth': 0.5}
    
    # Add color if 3+ objectives
    if n_objectives >= 3:
        scatter_params.update({'c': objectives[:, 2], 'cmap': 'viridis'})
    
    # Add size if 4+ objectives
    if n_objectives >= 4:
        sizes = objectives[:, 3]
        size_range = sizes.max() - sizes.min()
        scatter_params['s'] = 50 + 450 * (sizes - sizes.min()) / size_range if size_range > 0 else 100
    else:
        scatter_params['s'] = 100
    
    # Create scatter plot
    scatter = ax.scatter(objectives[:, 0], objectives[:, 1], **scatter_params)
    ax.set_xlabel(labels[0], fontsize=12)
    ax.set_ylabel(labels[1], fontsize=12)
    ax.set_title(f'Pareto Front ({n_solutions} solutions)', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for 3rd objective
    if n_objectives >= 3:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(labels[2], rotation=270, labelpad=20, fontsize=11)
    
    # Add size legend for 4th objective
    if n_objectives >= 4:
        sizes = objectives[:, 3]
        size_samples = [sizes.min(), (sizes.min() + sizes.max()) / 2, sizes.max()]
        legend_elements = [plt.scatter([], [], s=s, c='grey', alpha=0.6, edgecolors='black', 
                                      linewidth=0.5, label=f'{val:.2e}')
                          for val, s in zip(size_samples, [50, 275, 500])]
        ax.legend(handles=legend_elements, title=labels[3], loc='upper right',
                 framealpha=0.9, fontsize=9, title_fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pareto front plot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_parallel_coordinates(pkl_path, save_path=None, figsize=(12, 7), alpha_background=0.15):
    """
    Plot parallel coordinates showing all objectives.
    Each objective is a vertical axis, each solution is a line.
    Highlights the best solution for each objective with a distinct color.
    
    Args:
        pkl_path: Path to pickle file with optimization results
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        alpha_background: Transparency for non-highlighted solutions
        
    Returns:
        fig: matplotlib figure
    """
    # Load results
    results = load_results(pkl_path)
    objectives = results['objectives']
    n_solutions, n_objectives = objectives.shape
    
    # Get objective names
    from resto_anom import RestorationProblem
    problem = RestorationProblem(results['initial_conditions'], results['scenario_params'])
    
    # Clean up names for display
    display_names = {'abiotic_anomaly': 'Abiotic\nAnomaly', 'biotic_anomaly': 'Biotic\nAnomaly',
                     'landscape_anomaly': 'Landscape\nAnomaly', 'implementation_cost': 'Implementation\nCost',
                     'population_proximity': 'Population\nProximity'}
    labels = [display_names.get(name, name) for name in problem.objective_names]
    
    print(f"Plotting parallel coordinates: {n_solutions} solutions, {n_objectives} objectives")
    
    # Normalize objectives to [0, 1] for visualization
    objectives_norm = np.zeros_like(objectives)
    for i in range(n_objectives):
        obj_min, obj_max = objectives[:, i].min(), objectives[:, i].max()
        if obj_max > obj_min:
            objectives_norm[:, i] = (objectives[:, i] - obj_min) / (obj_max - obj_min)
        else:
            objectives_norm[:, i] = 0.5
    
    # Find best solution for each objective (minimum value = best)
    best_solutions = {}
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # Distinct colors
    for i in range(n_objectives):
        best_idx = np.argmin(objectives[:, i])
        best_solutions[i] = {'idx': best_idx, 'color': colors[i % len(colors)], 'label': labels[i]}
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot all solutions in grey (background)
    x_positions = np.arange(n_objectives)
    for sol_idx in range(n_solutions):
        ax.plot(x_positions, objectives_norm[sol_idx, :], 
               color='grey', alpha=alpha_background, linewidth=0.8, zorder=1)
    
    # Plot best solutions with distinct colors (foreground)
    plotted_indices = set()  # Track which solutions we've already highlighted
    for obj_idx, best_info in best_solutions.items():
        sol_idx = best_info['idx']
        if sol_idx not in plotted_indices:
            ax.plot(x_positions, objectives_norm[sol_idx, :], 
                   color=best_info['color'], alpha=0.9, linewidth=2.5, 
                   label=f"Best for {best_info['label'].replace(chr(10), ' ')}", zorder=2)
            plotted_indices.add(sol_idx)
    
    # Format axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Normalized Objective Value\n(0 = best, 1 = worst)', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Parallel Coordinates Plot ({n_solutions} Pareto solutions)', fontsize=14, pad=15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Parallel coordinates plot saved to: {save_path}")
    
    plt.show()
    return fig


def show_key_spatial_contrasts(pkl_path, save_path="spatial_contrasts.png"):
    """
    Show only 4-6 representative maps that highlight key differences.
    
    Args:
        pkl_path: Path to pickle file with combined results
        save_path: Where to save the visualization
    """
    combined_results = load_results(pkl_path)
    
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
        for scenario_id, results in combined_results['scenarios'].items():
            params = results['scenario_params']
            if (params['max_restoration_fraction'] == frac and 
                params['spatial_clustering'] == clust and 
                params['burden_sharing'] == burden):
                
                # Get best solution and convert to spatial map
                objectives = results['objectives']
                decisions = results['decisions']
                best_idx = np.argmin(np.sum(objectives, axis=1))
                
                # Create spatial visualization (simplified)
                restoration_map = create_spatial_map_from_decision(
                    decisions[best_idx], 
                    results['initial_conditions']
                )
                
                axes[i].imshow(restoration_map, cmap='RdYlBu_r')
                axes[i].set_title(title)
                axes[i].axis('off')
                break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

### Performance ranking table (which scenarios work best for what)

def quantify_parameter_effects(pkl_path):
    """
    Quantify how much each parameter affects objectives (statistical summary).
    
    Args:
        pkl_path: Path to pickle file with combined results
        
    Returns:
        DataFrame with effect sizes and significance tests
    """
    combined_results = load_results(pkl_path)
    
    # Extract all solutions from all scenarios
    all_solutions = []
    for scenario_id, results in combined_results['scenarios'].items():
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
