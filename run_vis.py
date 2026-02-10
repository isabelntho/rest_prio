import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# Import HVCallback to enable pickle loading
from resto_anom import HVCallback

from visualisations import (
    load_results,
    create_selection_frequency_map,
    plot_pareto_front,
    plot_parallel_coordinates,
    create_effect_summary_visualization,
    create_param_controlled_frequency_maps_multiscenario,
    show_key_spatial_contrasts,
    quantify_parameter_effects, 
    export_scenarios_to_csv,
    selection_freq_baseline_summary,
    plot_objectives_correlation_matrix,
    plot_example_solution, 
    plot_eligible_pixels
)
from data_loader import load_initial_conditions

def run_basic_analysis(pkl_path):
    """Run basic analysis and print summary statistics."""
    print("="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    with open(pkl_path, 'rb') as f:
        raw_results = pickle.load(f)
    
    results = load_results(pkl_path)
    F = results["objectives"]
    print(f"Number of solutions: {F.shape[0]}")
    print(f"Unique objective vectors: {np.unique(F, axis=0).shape[0]}")
    print(f"Objective std: {F.std(axis=0)}")
    
    return raw_results, results

def load_setup_data():
    """Load initial conditions and setup parameters."""
    try:
        initial_conditions = load_initial_conditions(".", objectives=['abiotic', 'biotic', 'landscape', 'cost'])
    except Exception as e:
        print(f"Warning: Could not load initial conditions: {e}")
        initial_conditions = None
    
    param_names = ["abiotic_effect", "biotic_effect", "max_restoration_fraction", "spatial_clustering", "burden_sharing"]
    fixed_values = {
        "biotic_effect": 0.02,
        "abiotic_effect": 0.02,
        "max_restoration_fraction": 0.3,
        "spatial_clustering": 0.3,
        "burden_sharing": "no"
    }
    
    return initial_conditions, param_names, fixed_values

def run_visualizations(pkl_file, visualizations="all"):
    """Run specified visualizations.
    
    Args:
        pkl_file: Name of pickle file (without extension)
        visualizations: 'all' or list of specific viz names
    """
    pkl_path = os.path.join(os.getcwd(), f"results_{pkl_file}.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return
    
    print(f"📊 Running visualizations for: {pkl_file}")
    os.makedirs("figs", exist_ok=True)
    
    # Load data and setup
    raw_results, results = run_basic_analysis(pkl_path)
    initial_conditions, param_names, fixed_values = load_setup_data()
    
    # Define available visualizations
    viz_funcs = {
        "correlation": lambda: plot_objectives_correlation_matrix(initial_conditions, save_path="objectives_correlation.png") if initial_conditions else print("Skipping correlation"),
        "eligible_pixels": lambda: plot_eligible_pixels(initial_conditions, save_path='eligible_pixels.png') if initial_conditions else print("Skipping eligible pixels"),
        "frequency_map": lambda: create_selection_frequency_map(
            pkl_path=pkl_path,
            save_path=f"figs/RFOP_{pkl_file}.png",
            cmap='YlOrRd',
            title="Relative frequency of occurrence in the Pareto Front (RFOP)"
        ),
        "example_solution": lambda: plot_example_solution(pkl_path=pkl_path, choose="random"),
        "param_controlled": lambda: create_param_controlled_frequency_maps_multiscenario(
            pkl_path=pkl_path, param_names=param_names, fixed_values=fixed_values,
            save_path=f"figs/RFOP_paramd_{pkl_file}.png"
        ),
        "export_scenarios": lambda: export_scenarios_to_csv(pkl_path=pkl_path, csv_path=f"scenarios_{pkl_file}.csv"),
        "baseline_summary": lambda: selection_freq_baseline_summary(raw_results, pkl_path),
        "effect_summary": lambda: create_effect_summary_visualization(pkl_path=pkl_path, save_path=f"figs/effect_summary_{pkl_file}.png"),
        "spatial_contrasts": lambda: show_key_spatial_contrasts(pkl_path=pkl_path, save_path=f"figs/key_spatial_contrasts_{pkl_file}.png"),
        "parameter_effects": lambda: quantify_parameter_effects(pkl_path=pkl_path, save_path=f"figs/parameter_effects_{pkl_file}.png", figsize=(10, 6)),
        "pareto_front": lambda: plot_pareto_front(pkl_path=pkl_path, save_path=f"figs/pareto_front_{pkl_file}.png", figsize=(10, 8), alpha=0.7),
        "parallel_coords": lambda: plot_parallel_coordinates(pkl_path=pkl_path, save_path=f"figs/parallel_coords_{pkl_file}.png", figsize=(12, 7), alpha_background=0.15)
    }
    
    # Determine which to run
    if visualizations == "all":
        viz_to_run = list(viz_funcs.keys())
    elif isinstance(visualizations, str):
        viz_to_run = [visualizations]
    else:
        viz_to_run = visualizations
    
    # Execute visualizations
    for viz_name in viz_to_run:
        if viz_name not in viz_funcs:
            print(f"❌ Unknown: {viz_name}. Available: {list(viz_funcs.keys())}")
            continue
        print(f"🎨 Running: {viz_name}")
        try:
            viz_funcs[viz_name]()
            print(f"   ✅ Done")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"🏁 Complete for {pkl_file}")

# Main execution
if __name__ == "__main__":
    # Configuration
    pkl_file = "agricultural_0902_1"  # Change this to your file
    
    # Run all visualizations
    #run_visualizations(pkl_file, "all")
    
    # Or run specific ones (uncomment to use):
    #run_visualizations(pkl_file, ["frequency_map", "example_solution"])
    # run_visualizations(pkl_file, "pareto_front")
    #run_visualizations(pkl_file, "eligible_pixels")
    run_visualizations(pkl_file, "frequency_map")
    # Available options:
    # "correlation", "eligible_pixels", "frequency_map", "example_solution", 
    # "param_controlled", "export_scenarios", "baseline_summary", "effect_summary",
    # "spatial_contrasts", "parameter_effects", "pareto_front", "parallel_coords"

