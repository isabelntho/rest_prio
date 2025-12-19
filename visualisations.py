#Visualisations
#Parameter effect heatmaps (what you get for changing parameters)

def create_effect_summary_visualization(combined_results, save_path="effect_summary.png"):
    """
    Show the EFFECTS of parameters rather than all individual maps.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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

# 4-6 representative spatial patterns (extreme cases only)

def show_key_spatial_contrasts(combined_results, save_path="spatial_contrasts.png"):
    """
    Show only 4-6 representative maps that highlight key differences.
    """
    import matplotlib.pyplot as plt
    
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

def quantify_parameter_effects(combined_results):
    """
    Quantify how much each parameter affects objectives (statistical summary).
    """
    import pandas as pd
    from scipy import stats
    
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