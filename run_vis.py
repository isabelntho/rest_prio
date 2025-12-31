from visualisations import (
    load_results,
    create_selection_frequency_map,
    plot_pareto_front,
    plot_parallel_coordinates,
    create_effect_summary_visualization,
    show_key_spatial_contrasts,
    quantify_parameter_effects
)

# Specify your pickle file path
pkl_file = "results_20251229_164034.pkl" 

fig, ax, freq_map = create_selection_frequency_map(
    pkl_path=pkl_file,
    save_path="selection_frequency_map.png",
    cmap='YlOrRd',  # Yellow to Red colormap
    title="Pixel Selection Frequency"
)


plot_pareto_front(
    pkl_path=pkl_file,
    save_path="pareto_front.png",
    figsize=(10, 8),
    alpha=0.7
)

plot_parallel_coordinates(
    pkl_path=pkl_file,
    save_path="parallel_coordinates.png",
    figsize=(12, 7),
    alpha_background=0.15
)

results = load_results(pkl_file)
print("\nResults loaded:")
print(f"  Scenario params: {results['scenario_params']}")
print(f"  Number of solutions: {results['n_solutions']}")
print(f"  Objectives shape: {results['objectives'].shape}")
print(f"  Decisions shape: {results['decisions'].shape}")

# Example 5: Access specific solution
solution_idx = 0  # First Pareto-optimal solution
print(f"\nSolution {solution_idx}:")
print(f"  Objectives: {results['objectives'][solution_idx]}")
print(f"  Pixels restored: {results['decisions'][solution_idx].sum()}")

