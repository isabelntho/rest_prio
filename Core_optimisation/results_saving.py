"""
Results Saving Utilities
========================
Functions for saving optimization results, parameter summaries, and combined scenario results.
Includes comprehensive reporting for sharing setup details and debugging with colleagues.

Created: January 2026
"""

import os
import json
import pickle
import subprocess
import numpy as np
from datetime import datetime
from .paths import OUTPUT_DIR, MULTISEED_DIR
try:
    import scipy.stats
except ImportError:
    pass


def _get_git_hash():
    """Return short git commit hash for the current repo, 'no_git' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "no_git"
    except Exception:
        return "no_git"


def _append_to_registry(entry, output_dir=str(OUTPUT_DIR)):
    """Append a single-line JSON entry to run_registry.jsonl in output_dir.
    Write failures are silently ignored so they never break a run.
    """
    registry_path = os.path.join(output_dir, "run_registry.jsonl")
    try:
        with open(registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

def create_optimization_report(results, output_dir=str(OUTPUT_DIR), verbose=True, run_label="", run_timestamp=None):
    """
    Create a comprehensive optimization report including problem definition,
    algorithm details, and evolution tracking for sharing with colleagues.
    
    Args:
        results: Results dictionary from optimization
        output_dir: Directory to save report
        verbose: Print save status
        run_label: Human-readable label for this run
        run_timestamp: Timestamp string shared across all files of this run (YYYYMMDD_HHMM).
                       Generated fresh if not provided.
        
    Returns:
        str: Path to the generated report file
    """
    if verbose:
        print("\nCreating comprehensive optimization report...")

    os.makedirs(os.path.join(output_dir, "optimisation_reports"), exist_ok=True)
    
    timestamp = run_timestamp or datetime.now().strftime('%Y%m%d_%H%M')
    run_label_slug = run_label.replace(" ", "_") if run_label else ""
    label_suffix = f"_{run_label_slug}" if run_label_slug else ""
    ecosystem = results.get('scenario_params', {}).get(
        'ecosystem_label',
        results.get('initial_conditions', {}).get('ecosystem', 'all')
    )
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    report_filename = os.path.join(output_dir, f"optimisation_reports/opt_{ecosystem_suffix}_{timestamp}{label_suffix}.json")
    
    # Extract key information from results
    initial_conditions = results.get('initial_conditions', {})
    scenario_params = results.get('scenario_params', {})
    algorithm_info = results.get('algorithm_info', {})
    problem_info = results.get('problem_info', {})
    objectives = results.get('objectives', np.array([]))
    objectives_normalized = results.get('objectives_normalized', np.array([]))
    objective_names = results.get('objective_names', [])
    hv_callback = results.get('hv_callback', None)
    
    # Build comprehensive report
    report = {
        "report_metadata": {
            "created_timestamp": datetime.now().isoformat(),
            "report_type": "optimization_setup_and_results",
            "ecosystem": ecosystem,
            "run_label": run_label,
            "git_hash": _get_git_hash(),
            "optimization_timestamp": algorithm_info.get('timestamp', 'unknown')
        },
        
        # ===== PROBLEM DEFINITION =====
        
        "problem_definition": {
            "problem_type": "multi_objective_restoration_optimization",
            "optimization_framework": "pymoo_nsga2",
            
            "objectives": {
                "count": len(objective_names),
                "names": objective_names,
                "formulations": {},
                "baseline_values": {},
                "final_ranges": {}
            },
            
            "decision_variables": {
                "type": "binary_restoration_decisions",
                "total_pixels": initial_conditions.get('n_pixels', 0),
                "restoration_eligible_pixels": initial_conditions.get('n_restoration_pixels', 0),
                "conversion_eligible_pixels": initial_conditions.get('n_conversion_pixels', 0),
                "bounds": {
                    "min_restored_pixels": 0,
                    "max_restored_pixels": problem_info.get('max_restored_pixels', 0),
                    "budget_constraint": scenario_params.get('max_restoration_fraction', 0.0)
                }
            },
            
            "constraints": {
                "budget_constraint": {
                    "type": "restoration_budget_limit", 
                    "max_fraction": scenario_params.get('max_restoration_fraction', 0.0),
                    "max_pixels": problem_info.get('max_restored_pixels', 0)
                },
                "spatial_constraints": {
                    "burden_sharing": scenario_params.get('burden_sharing', False),
                    "clustering_strength": scenario_params.get('clustering_strength', 0.0)
                }
            }
        },
        
        # ===== ALGORITHM CONFIGURATION =====
        "algorithm_configuration": {
            "algorithm_name": "NSGA-II",
            "population_size": algorithm_info.get('pop_size', 50),
            "max_generations": algorithm_info.get('n_generations', 100),
            "actual_generations": algorithm_info.get('actual_generations', 0),
            
            "constraint_handling": {
                "method": "adaptive_repair",
                "repair_operators": algorithm_info.get('repair_operators', []),
                "repair_strategy": "score_based_with_random_fallback",
                "tolerance": "budget_based_tolerance"
            },
            
            "termination": {
                "max_generations": algorithm_info.get('n_generations', 100),
                "hypervolume_stopping": {
                    "enabled": True,
                    "patience": algorithm_info.get('hv_patience', 15),
                    "min_improvement": algorithm_info.get('hv_min_improvement', 1e-6),
                    "reference_point": algorithm_info.get('hv_reference_point', None)
                },
                "actual_reason": algorithm_info.get('termination_reason', 'unknown')
            },
            "objective_normalization": algorithm_info.get('objective_normalization', {
                "enabled": False,
                "scales": {}
            })
        },
        
        # ===== SCENARIO PARAMETERS =====
        "scenario_parameters": {
            "restoration_effects": {
                "abiotic_effect": scenario_params.get('abiotic_effect', 0.01),
                "biotic_effect": scenario_params.get('biotic_effect', 0.01),
                "neighbor_radius": 1,
                "neighbor_effect_decay": 0.5
            },
            "spatial_parameters": {
                "burden_sharing": scenario_params.get('burden_sharing', False),
                "clustering_strength": scenario_params.get('clustering_strength', 0.0)
            },
            "economic_parameters": {
                "restoration_cost_multiplier": scenario_params.get('restoration_cost_multiplier', 1.0),
                "conversion_cost_multiplier": scenario_params.get('conversion_cost_multiplier', 1.0)
            }
        },
        
        # ===== OPTIMIZATION EVOLUTION =====
        "optimization_evolution": {},
        
        # ===== FINAL RESULTS SUMMARY =====
        "results_summary": {
            "pareto_solutions": {
                "count": len(objectives) if len(objectives.shape) > 1 else 0,
                "hypervolume": hv_callback.best_hv if hv_callback and hasattr(hv_callback, 'best_hv') else None
            },
            "objective_statistics": {},
            "objective_statistics_normalized": {},
            "convergence_metrics": {},
            #"decision_patterns": results.get('decision_analysis', {})
        },
        # ===== POPULATION DIAGNOSTICS =====
        "population_diagnostics": {}
    }
    
    # Fill in objective formulations and baseline values
    for i, obj_name in enumerate(objective_names):
        if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
            report["problem_definition"]["objectives"]["formulations"][obj_name] = {
                "formula": "minimize(-pixel_count_above_threshold)",
                "description": f"Maximize count of pixels above threshold (higher {obj_name.split('_')[0]} is better)",
                "direction": "minimize_negative_count"
            }
        elif obj_name == 'landscape_anomaly':
            report["problem_definition"]["objectives"]["formulations"][obj_name] = {
                "formula": "minimize(sum(landscape_anomaly_values))", 
                "description": "Minimize total landscape impact (lower landscape anomaly is better)",
                "direction": "minimize_sum"
            }
        elif obj_name == 'implementation_cost':
            report["problem_definition"]["objectives"]["formulations"][obj_name] = {
                "formula": "minimize(total_implementation_cost)",
                "description": "Minimize total cost of restoration actions",
                "direction": "minimize"
            }
        elif obj_name == 'population_proximity':
            report["problem_definition"]["objectives"]["formulations"][obj_name] = {
                "formula": "minimize(average_distance_to_population)",
                "description": "Minimize average distance to population centers", 
                "direction": "minimize"
            }
        
        # Add baseline values
        if obj_name in initial_conditions:
            if obj_name in ['abiotic_anomaly', 'biotic_anomaly']:
                # These objectives use improvement over eligible pixels only
                # Baseline = -sum(initial_anomaly - initial_anomaly) over eligible = 0
                # But to be consistent with objective calculation:
                baseline_array = initial_conditions[obj_name]
                mask = initial_conditions["restoration_eligible_mask"]
                # For reporting purposes, show the sum of initial anomalies over eligible pixels
                baseline_val = float(np.sum(baseline_array[mask]))
                
            elif obj_name == 'landscape_anomaly':
                # Landscape objective sums entire array (global objective)
                baseline_array = initial_conditions[obj_name]
                if hasattr(baseline_array, 'sum'):
                    baseline_val = float(baseline_array.sum())
                else:
                    baseline_val = float(baseline_array)
                    
            elif obj_name == 'implementation_cost':
                # Implementation cost baseline is 0 (no cost when no actions taken)
                baseline_val = 0.0
                
            else:
                # Handle other potential objectives
                baseline_value = initial_conditions[obj_name]
                if hasattr(baseline_value, 'item') and hasattr(baseline_value, 'size') and baseline_value.size == 1:
                    # It's a numpy scalar with single element
                    baseline_val = float(baseline_value.item())
                elif hasattr(baseline_value, '__len__') and len(baseline_value) == 1:
                    # It's a length-1 array
                    baseline_val = float(baseline_value[0])
                elif hasattr(baseline_value, 'sum') and hasattr(baseline_value, 'shape'):
                    # It's a numpy array - sum it up
                    baseline_val = float(baseline_value.sum())
                else:
                    # It's already a scalar or convert directly
                    try:
                        baseline_val = float(baseline_value)
                    except (TypeError, ValueError):
                        baseline_val = 0.0  # fallback
                        
            report["problem_definition"]["objectives"]["baseline_values"][obj_name] = baseline_val
        
        # Add final ranges
        if len(objectives) > 0 and len(objectives.shape) > 1:
            obj_values = objectives[:, i]
            # Safely convert numpy values to Python floats
            try:
                min_val = float(obj_values.min())
                max_val = float(obj_values.max()) 
                mean_val = float(obj_values.mean())
                std_val = float(obj_values.std())
            except (TypeError, ValueError):
                # Fallback if conversion fails
                min_val = max_val = mean_val = std_val = 0.0
                
            report["problem_definition"]["objectives"]["final_ranges"][obj_name] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "unique_count": len(np.unique(obj_values))
            }
    
    # Add hypervolume evolution if available
    if hv_callback and hasattr(hv_callback, 'hv_history'):
        hv_history = [float(hv) for hv in hv_callback.hv_history]
        report["optimization_evolution"]["hypervolume_history"] = hv_history
        
        # Calculate convergence metrics
        if len(hv_history) > 1:
            hv_improvement = hv_history[-1] - hv_history[0] if hv_history[0] > 0 else 0
            report["results_summary"]["convergence_metrics"]["hypervolume_improvement"] = float(hv_improvement)
            report["results_summary"]["convergence_metrics"]["generations_to_convergence"] = len(hv_history)
            
            # Find generation of best improvement
            if len(hv_history) > 5:
                hv_diffs = np.diff(hv_history)
                best_improvement_gen = int(np.argmax(hv_diffs)) + 1 if len(hv_diffs) > 0 else 0
                report["results_summary"]["convergence_metrics"]["best_improvement_generation"] = best_improvement_gen
    
    # Add final statistics to results summary
    if len(objectives) > 0 and len(objectives.shape) > 1:
        for i, obj_name in enumerate(objective_names):
            obj_values = objectives[:, i]
            # Safely convert all numpy values to Python floats
            try:
                min_val = float(obj_values.min())
                max_val = float(obj_values.max())
                mean_val = float(obj_values.mean())
                std_val = float(obj_values.std())
                range_span = max_val - min_val
            except (TypeError, ValueError):
                # Fallback if any conversion fails
                min_val = max_val = mean_val = std_val = range_span = 0.0
                
            report["results_summary"]["objective_statistics"][obj_name] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "range_span": range_span
            }

    # Add normalized objective statistics if available
    if len(objectives_normalized) > 0 and len(objectives_normalized.shape) > 1:
        for i, obj_name in enumerate(objective_names):
            obj_values = objectives_normalized[:, i]
            try:
                min_val = float(obj_values.min())
                max_val = float(obj_values.max())
                mean_val = float(obj_values.mean())
                std_val = float(obj_values.std())
                range_span = max_val - min_val
            except (TypeError, ValueError):
                min_val = max_val = mean_val = std_val = range_span = 0.0

            report["results_summary"]["objective_statistics_normalized"][obj_name] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "range_span": range_span
            }
    
    # ===== FULL POPULATION DIAGNOSTICS =====
    # Add full population diagnostics if available
    full_population = results.get('full_population', None)
    if full_population is not None and 'objectives' in full_population:
        pop_objectives = full_population['objectives']
        pop_decisions = full_population['decisions']
        
        # Population-wide objective variance and statistics
        population_stats = {}
        for i, obj_name in enumerate(objective_names):
            obj_vals = pop_objectives[:, i]
            population_stats[obj_name] = {
                "population_min": float(np.min(obj_vals)),
                "population_max": float(np.max(obj_vals)),
                "population_mean": float(np.mean(obj_vals)),
                "population_std": float(np.std(obj_vals)),
                "population_variance": float(np.var(obj_vals)),
                "population_median": float(np.median(obj_vals)),
                "population_q25": float(np.percentile(obj_vals, 25)),
                "population_q75": float(np.percentile(obj_vals, 75)),
                "population_skewness": float(scipy.stats.skew(obj_vals)) if 'scipy' in globals() and hasattr(scipy, 'stats') else None,
                "population_kurtosis": float(scipy.stats.kurtosis(obj_vals)) if 'scipy' in globals() and hasattr(scipy, 'stats') else None
            }
        
        report["population_diagnostics"]["population_statistics"] = population_stats
        
        # Correlation matrix among objectives
        if pop_objectives.shape[1] > 1:
            correlation_matrix = np.corrcoef(pop_objectives.T)
            # Convert to JSON-serializable format
            correlation_data = {}
            for i in range(len(objective_names)):
                correlation_data[objective_names[i]] = {}
                for j in range(len(objective_names)):
                    correlation_data[objective_names[i]][objective_names[j]] = float(correlation_matrix[i, j])
            
            report["population_diagnostics"]["objective_correlations"] = correlation_data
        
        # Distribution parameters for each objective
        distribution_analysis = {}
        for i, obj_name in enumerate(objective_names):
            obj_vals = pop_objectives[:, i]
            
            # Histogram data (bins and counts)
            hist_counts, hist_bins = np.histogram(obj_vals, bins=20)
            
            distribution_analysis[obj_name] = {
                "histogram_bins": hist_bins.tolist(),
                "histogram_counts": hist_counts.tolist(),
                "distribution_range": float(np.max(obj_vals) - np.min(obj_vals)),
                "coefficient_of_variation": float(np.std(obj_vals) / np.abs(np.mean(obj_vals))) if np.mean(obj_vals) != 0 else 0.0
            }
        
        report["population_diagnostics"]["distribution_analysis"] = distribution_analysis
        
        # LULC transition sensitivity analysis
        sensitivity_analysis = {}
        
        # For each LULC transition type, calculate sensitivity
        n_restoration = results.get('initial_conditions', {}).get('n_restoration_pixels', 0)
        n_conversion = results.get('initial_conditions', {}).get('n_conversion_pixels', 0)
        
        if n_restoration > 0:
            # Restoration sensitivity: average objective change per restored pixel
            restoration_sensitivities = {}
            for i, obj_name in enumerate(objective_names):
                # Find solutions with restoration and calculate sensitivity
                restoration_counts = np.sum(pop_decisions[:, :n_restoration], axis=1)
                obj_vals = pop_objectives[:, i]
                
                # Only include solutions with some restoration
                valid_mask = restoration_counts > 0
                if np.any(valid_mask):
                    valid_counts = restoration_counts[valid_mask]
                    valid_objs = obj_vals[valid_mask]
                    
                    # Calculate sensitivity as objective change per restored pixel
                    if len(valid_counts) > 1:
                        sensitivity = np.corrcoef(valid_counts, valid_objs)[0, 1] if len(valid_counts) > 1 else 0.0
                        restoration_sensitivities[obj_name] = {
                            "correlation_with_restoration_count": float(sensitivity),
                            "average_change_per_pixel": float(np.mean(valid_objs / valid_counts)) if np.all(valid_counts > 0) else 0.0
                        }
            
            sensitivity_analysis["restoration_sensitivity"] = restoration_sensitivities
        
        if n_conversion > 0:
            # Conversion sensitivity: average objective change per converted pixel
            conversion_sensitivities = {}
            for i, obj_name in enumerate(objective_names):
                # Find solutions with conversion and calculate sensitivity
                conversion_counts = np.sum(pop_decisions[:, n_restoration:n_restoration+n_conversion], axis=1)
                obj_vals = pop_objectives[:, i]
                
                # Only include solutions with some conversion
                valid_mask = conversion_counts > 0
                if np.any(valid_mask):
                    valid_counts = conversion_counts[valid_mask]
                    valid_objs = obj_vals[valid_mask]
                    
                    # Calculate sensitivity as objective change per converted pixel
                    if len(valid_counts) > 1:
                        sensitivity = np.corrcoef(valid_counts, valid_objs)[0, 1] if len(valid_counts) > 1 else 0.0
                        conversion_sensitivities[obj_name] = {
                            "correlation_with_conversion_count": float(sensitivity),
                            "average_change_per_pixel": float(np.mean(valid_objs / valid_counts)) if np.all(valid_counts > 0) else 0.0
                        }
            
            sensitivity_analysis["conversion_sensitivity"] = conversion_sensitivities
        
        report["population_diagnostics"]["lulc_transition_sensitivity"] = sensitivity_analysis
    # ===== END OF DIAGNOSTIC ADDITIONS =====

    # Save report
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    if verbose:
        print(f"✓ Comprehensive optimization report saved to: {report_filename}")
        print(f"   Report includes: problem definition, algorithm config, evolution tracking")
        print(f"   Objectives: {len(objective_names)} ({', '.join(objective_names)})")
        print(f"   Pareto solutions: {len(objectives) if len(objectives.shape) > 1 else 0}")
        if hv_callback:
            if hasattr(hv_callback, 'best_hv') and hv_callback.best_hv > 0:
                print(f"   Final hypervolume: {hv_callback.best_hv:.6f}")
            elif hasattr(hv_callback, 'hv_history') and len(hv_callback.hv_history) > 0:
                print(f"   Final hypervolume: {hv_callback.hv_history[-1]:.6f}")
            else:
                print("   Hypervolume: tracking enabled")
    
    return report_filename

def create_evolution_tracking_report(results, output_dir=str(OUTPUT_DIR), verbose=True, run_label="", run_timestamp=None):
    """
    Create a detailed report tracking optimization evolution over generations.
    Useful for understanding convergence patterns and algorithm behavior.
    
    Args:
        results: Results dictionary from optimization
        output_dir: Directory to save report
        verbose: Print save status
        run_label: Human-readable label for this run
        run_timestamp: Timestamp string shared across all files of this run (YYYYMMDD_HHMM).
                       Generated fresh if not provided.
        
    Returns:
        str: Path to the generated evolution report file
    """
    if verbose:
        print("\nCreating evolution tracking report...")

    os.makedirs(os.path.join(output_dir, "evolution_reports"), exist_ok=True)
    
    timestamp = run_timestamp or datetime.now().strftime('%Y%m%d_%H%M')
    run_label_slug = run_label.replace(" ", "_") if run_label else ""
    label_suffix = f"_{run_label_slug}" if run_label_slug else ""
    
    # Get ecosystem information for filename
    ecosystem = results.get('scenario_params', {}).get(
        'ecosystem_label',
        results.get('initial_conditions', {}).get('ecosystem', 'all')
    )
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    evolution_filename = os.path.join(
        output_dir,
        f"evolution_reports/evo_{ecosystem_suffix}_{timestamp}{label_suffix}.json"
    )
    
    # Extract evolution data
    hv_callback = results.get('hv_callback', None)
    algorithm_info = results.get('algorithm_info', {})
    objective_names = results.get('objective_names', [])
    
    evolution_report = {
        "evolution_metadata": {
            "created_timestamp": datetime.now().isoformat(),
            "report_type": "optimization_evolution_tracking",
            "total_generations": algorithm_info.get('actual_generations', 0),
            "population_size": algorithm_info.get('pop_size', 50)
        },
        
        "hypervolume_evolution": {
            "enabled": hv_callback is not None,
            "history": [],
            "convergence_analysis": {},
            "stopping_criteria": {
                "patience": algorithm_info.get('hv_patience', 15),
                "min_improvement": algorithm_info.get('hv_min_improvement', 1e-6),
                "reference_point": algorithm_info.get('hv_reference_point', None)
            }
        },
        
        "generation_statistics": {},
        
        "convergence_metrics": {
            "hypervolume_based": {},
            "objective_based": {},
            "diversity_based": {}
        },
        
        "optimization_phases": []
    }
    
    # Process hypervolume history if available
    if hv_callback and hasattr(hv_callback, 'hv_history'):
        hv_history = [float(hv) for hv in hv_callback.hv_history]
        evolution_report["hypervolume_evolution"]["history"] = hv_history
        
        # Analyze hypervolume progression
        if len(hv_history) > 1:
            hv_improvements = np.diff(hv_history)
            
            # Overall convergence analysis with safe conversions
            try:
                total_imp = hv_history[-1] - hv_history[0] if hv_history[0] > 0 else 0.0
                rel_imp = (hv_history[-1] - hv_history[0]) / hv_history[0] if hv_history[0] > 0 else 0.0
                max_imp = float(np.max(hv_improvements)) if len(hv_improvements) > 0 else 0.0
                gen_imp = int(np.sum(hv_improvements > 1e-10))
                gen_plat = int(np.sum(np.abs(hv_improvements) < 1e-10))
                avg_rate = float(np.mean(hv_improvements)) if len(hv_improvements) > 0 else 0.0
                
                evolution_report["hypervolume_evolution"]["convergence_analysis"] = {
                    "total_improvement": total_imp,
                    "relative_improvement": rel_imp,
                    "max_single_improvement": max_imp,
                    "generations_with_improvement": gen_imp,
                    "generations_with_plateau": gen_plat,
                    "average_improvement_rate": avg_rate
                }
            except (TypeError, ValueError):
                evolution_report["hypervolume_evolution"]["convergence_analysis"] = {
                    "total_improvement": 0.0,
                    "relative_improvement": 0.0,
                    "max_single_improvement": 0.0,
                    "generations_with_improvement": 0,
                    "generations_with_plateau": 0,
                    "average_improvement_rate": 0.0
                }
            
            # Identify optimization phases
            phases = []
            current_phase = {"type": "exploration", "start_gen": 0, "improvement": 0.0}
            
            for i, improvement in enumerate(hv_improvements):
                if improvement > algorithm_info.get('hv_min_improvement', 1e-6):
                    # Significant improvement - exploration/search phase
                    if current_phase["type"] != "exploration":
                        phases.append(current_phase)
                        current_phase = {"type": "exploration", "start_gen": i, "improvement": 0.0}
                    current_phase["improvement"] += improvement
                else:
                    # Minimal improvement - exploitation/convergence phase  
                    if current_phase["type"] != "convergence":
                        phases.append(current_phase)
                        current_phase = {"type": "convergence", "start_gen": i, "improvement": 0.0}
                    current_phase["improvement"] += improvement
            
            # Add final phase
            current_phase["end_gen"] = len(hv_improvements)
            current_phase["duration"] = current_phase["end_gen"] - current_phase["start_gen"]
            phases.append(current_phase)
            
            evolution_report["optimization_phases"] = phases
            
            # Detailed convergence metrics
            evolution_report["convergence_metrics"]["hypervolume_based"] = {
                "stability_analysis": {
                    "final_10_gen_std": float(np.std(hv_history[-10:])) if len(hv_history) >= 10 else 0.0,
                    "final_5_gen_improvement": float(hv_history[-1] - hv_history[-6]) if len(hv_history) >= 6 else 0.0,
                }
            }
    
    # Process population statistics if available
    pop_stats = algorithm_info.get('population_statistics', {})
    if pop_stats:
        evolution_report["population_evolution"] = {
            "enabled": True,
            "f_mean_history": pop_stats.get('f_mean_history', []),
            "f_std_history": pop_stats.get('f_std_history', []),
            "f_min_history": pop_stats.get('f_min_history', []),
            "f_max_history": pop_stats.get('f_max_history', []),
            "objectives": objective_names
        }
        
        # Compute convergence metrics based on population statistics
        f_mean_history = pop_stats.get('f_mean_history', [])
        if len(f_mean_history) > 0 and len(objective_names) > 0:
            try:
                # Convert to numpy array for easier analysis
                f_mean_array = np.array(f_mean_history)  # shape: (n_generations, n_objectives)
                
                # Convergence analysis for each objective
                objective_convergence = {}
                for i, obj_name in enumerate(objective_names):
                    if f_mean_array.shape[1] > i:
                        obj_history = f_mean_array[:, i]
                        objective_convergence[obj_name] = {
                            "initial_mean": float(obj_history[0]) if len(obj_history) > 0 else 0.0,
                            "final_mean": float(obj_history[-1]) if len(obj_history) > 0 else 0.0,
                            "total_change": float(obj_history[-1] - obj_history[0]) if len(obj_history) > 0 else 0.0,
                            "final_5_gen_std": float(np.std(obj_history[-5:])) if len(obj_history) >= 5 else 0.0
                        }
                
                evolution_report["convergence_metrics"]["objective_based"] = objective_convergence
                
            except Exception as e:
                if verbose:
                    print(f"   Warning: Could not process population statistics: {e}")
    
    # Add generation-by-generation statistics if available
    pop_stats = algorithm_info.get('population_statistics', {})
    for gen in range(algorithm_info.get('actual_generations', 0)):
        generation_stats = {
            "generation": gen,
            "hypervolume": hv_history[gen] if hv_callback and len(hv_history) > gen else None
        }
        
        # Add population statistics for this generation if available
        if pop_stats:
            f_mean_history = pop_stats.get('f_mean_history', [])
            f_std_history = pop_stats.get('f_std_history', [])
            f_min_history = pop_stats.get('f_min_history', [])
            f_max_history = pop_stats.get('f_max_history', [])
            
            if len(f_mean_history) > gen:
                generation_stats["f_mean"] = f_mean_history[gen]
            if len(f_std_history) > gen:
                generation_stats["f_std"] = f_std_history[gen]
            if len(f_min_history) > gen:
                generation_stats["f_min"] = f_min_history[gen]
            if len(f_max_history) > gen:
                generation_stats["f_max"] = f_max_history[gen]
        
        evolution_report["generation_statistics"][f"generation_{gen}"] = generation_stats
    
    # Save evolution report
    with open(evolution_filename, 'w') as f:
        json.dump(evolution_report, f, indent=2)
    
    if verbose:
        print(f"✓ Evolution tracking report saved to: {evolution_filename}")
        if hv_callback:
            hv_count = len(hv_callback.hv_history) if hasattr(hv_callback, 'hv_history') else 0
            print(f"   Tracked {hv_count} generations of hypervolume evolution")
            if len(evolution_report["optimization_phases"]) > 0:
                print(f"   Identified {len(evolution_report['optimization_phases'])} optimization phases")
        
        pop_stats = algorithm_info.get('population_statistics', {})
        if pop_stats:
            f_mean_history = pop_stats.get('f_mean_history', [])
    
    return evolution_filename

def save_results_with_reports(problem_or_results, res=None, initial_conditions=None, scenario_params=None,
                                        pop_size=None, n_generations=None, total_time=None, hv_value=None,
                                        ecosystem='unknown', region='unknown', experiment_id=None, random_seed=None,
                                        output_dir=str(OUTPUT_DIR), verbose=True, include_reports=True, include_debug=False,
                                        run_label="", r_export_parent=None):
    """
     Save optimization results with comprehensive reporting.

     Supports two calling styles:
     1) New style (preferred):
         save_results_with_reports(results_dict, output_dir=str(OUTPUT_DIR), verbose=True)
     2) Legacy style:
         save_results_with_reports(problem, res, initial_conditions, scenario_params,
                                            pop_size, n_generations, total_time, hv_value, ...)
    
    Args:
        problem_or_results: Problem instance (legacy) or results dict (new style).
        res: Result object from pymoo (legacy style only).
        initial_conditions: Initial conditions dictionary (legacy style only).
        scenario_params: Scenario parameters dictionary (legacy style only).
        pop_size: Population size (legacy style only).
        n_generations: Number of generations (legacy style only).
        total_time: Total execution time (legacy style only).
        hv_value: Final hypervolume (legacy style only).
        ecosystem: Name of the ecosystem.
        region: Name of the region.
        experiment_id: Unique ID for the experiment run.
        random_seed: The random seed used.
        output_dir: Directory to write output files.
        verbose: Print save status.
        include_reports: Generate comprehensive optimization report (new style only).
        include_debug: Generate debug report if available (new style only).
        
    Returns:
        dict: Dictionary with paths to all generated files.
    """
    # New-style call path used by run_single_scenario_optimization.
    if isinstance(problem_or_results, dict) and res is None:
        return save_scenario_results(
            results=problem_or_results,
            output_dir=output_dir,
            verbose=verbose,
            include_reports=include_reports,
            include_debug=include_debug,
            run_label=run_label,
            r_export_parent=r_export_parent,
        )

    problem = problem_or_results

    # Create a comprehensive results dictionary
    results_data = {
        'problem': problem,
        'result_object': res,
        'initial_conditions': initial_conditions,
        'scenario_params': scenario_params,
        'algorithm_info': {
            'pop_size': pop_size,
            'n_generations': n_generations,
            'execution_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'random_seed': random_seed,
            'experiment_id': experiment_id
        },
        'hypervolume': hv_value,
        'objective_names': problem.objective_names,
        'objectives': res.F,
        'objectives_normalized': res.F,
        'decisions': res.X,
        'n_solutions': len(res.F) if res.F is not None else 0
    }
    
    # Use experiment_id for file naming if available
    if experiment_id:
        base_filename = f"{experiment_id}"
        output_subdir = os.path.join(str(MULTISEED_DIR), experiment_id)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"results_{ecosystem}_{region}_{timestamp}"
        output_subdir = str(MULTISEED_DIR)

    os.makedirs(output_subdir, exist_ok=True)

    # Save results using the new base filename
    pickle_path = os.path.join(output_subdir, f"{base_filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results_data, f)

    # Create and save a JSON summary
    summary = {
        'experiment_id': experiment_id,
        'random_seed': random_seed,
        'ecosystem': ecosystem,
        'region': region,
        'scenario_params': scenario_params,
        'optimization_info': {
            'pop_size': pop_size,
            'n_generations': n_generations,
            'execution_time': total_time,
            'n_solutions': results_data['n_solutions'],
            'hypervolume': hv_value
        },
        'objective_names': results_data['objective_names'],
        'objective_stats': {}
    }

    if results_data['n_solutions'] > 0:
        for i, name in enumerate(results_data['objective_names']):
            values = results_data['objectives'][:, i]
            summary['objective_stats'][name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

    summary_path = os.path.join(output_subdir, f"summary_{base_filename}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✓ Results saved to {output_subdir}")

    return {
        'pickle_file': pickle_path,
        'summary_file': summary_path,
        'results_df': None # This was added in resto_anom, so we keep it for compatibility
    }

def save_parameter_summary(output_dir=str(OUTPUT_DIR), n_samples_per_param=3, random_seed=42, verbose=True):
    """
    Save a summary of parameter ranges and sampled values.
    
    Args:
        output_dir: Directory to save files
        n_samples_per_param: Number of samples per continuous parameter
        random_seed: Random seed used for sampling
        verbose: Print save status
    """
    from .resto_anom import define_scenario_parameters, sample_scenario_parameters
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Get parameter ranges and sampled values
    param_ranges = define_scenario_parameters()
    sampled_combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
    
    # Create parameter summary
    param_summary = {
        'sampling_info': {
            'n_samples_per_param': n_samples_per_param,
            'random_seed': random_seed,
            'total_scenarios': len(sampled_combinations),
            'timestamp': datetime.now().isoformat()
        },
        'parameter_ranges': {},
        'sampled_values': {},
        'scenarios': []
    }
    
    # Process parameter ranges
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            param_summary['parameter_ranges'][param_name] = {
                'type': 'continuous',
                'min': param_range[0],
                'max': param_range[1]
            }
        else:
            param_summary['parameter_ranges'][param_name] = {
                'type': 'categorical',
                'values': param_range
            }
    
    # Extract unique sampled values for each parameter
    for param_name in param_ranges.keys():
        values = [combo[param_name] for combo in sampled_combinations]
        unique_values = sorted(list(set(values)))
        param_summary['sampled_values'][param_name] = unique_values
    
    # Add all scenario combinations
    for i, combo in enumerate(sampled_combinations):
        param_summary['scenarios'].append({
            'scenario_id': i,
            'parameters': combo
        })
    
    summary_filename = os.path.join(output_dir, f"parameter_summary_{timestamp}.json")
    
    with open(summary_filename, 'w') as f:
        json.dump(param_summary, f, indent=2)
    
    if verbose:
        print(f"✓ Parameter summary saved to: {summary_filename}")
        print(f"   Total scenarios: {len(sampled_combinations)}")
        print(f"   Continuous parameters: {sum(1 for r in param_ranges.values() if isinstance(r, tuple))}")
        print(f"   Categorical parameters: {sum(1 for r in param_ranges.values() if not isinstance(r, tuple))}")
    
    return summary_filename

def save_scenario_results(results, output_dir=str(OUTPUT_DIR), verbose=True, include_reports=True, include_debug=False, run_label="", r_export_parent=None):
    """
    Save single scenario optimization results to files with optional comprehensive reporting.
    
    Args:
        results: Results dictionary from single scenario optimization
        output_dir: Directory to save files
        verbose: Print save status
        include_reports: Whether to generate comprehensive optimization report
        include_debug: Whether to generate detailed debug report
        
    Returns:
        dict: Dictionary with paths to all generated files
    """
    if verbose:
        print("\nSaving scenario results...")

    os.makedirs(os.path.join(output_dir, "results_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summary_files"), exist_ok=True)
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    run_label_slug = run_label.replace(" ", "_") if run_label else ""
    label_suffix = f"_{run_label_slug}" if run_label_slug else ""
    
    # Get ecosystem information for filename
    ecosystem = results.get('scenario_params', {}).get(
        'ecosystem_label',
        results.get('initial_conditions', {}).get('ecosystem', 'all')
    )
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    generated_files = {}
    
    # Save complete results (pickle format)
    results_filename = os.path.join(
        output_dir,
        f"results_files/res{ecosystem_suffix}_{run_timestamp}{label_suffix}.pkl"
    )
    print(f"  Saving to: {results_filename}")
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    generated_files['pickle_file'] = results_filename
    
    # Save summary (JSON format)
    summary_filename = os.path.join(output_dir, f"summary_files/summary{ecosystem_suffix}_{run_timestamp}{label_suffix}.json")
    
    objectives = results['objectives']
    
    # Get objective names from the problem (avoid circular import)
    # We can get them from the results dict directly
    objective_names = results['objective_names']
    
    summary = {
        'run_label': run_label,
        'ecosystem': ecosystem,
        'scenario_params': results['scenario_params'],
        'optimization_info': {
            'n_solutions': results['n_solutions'],
            'n_pixels': results['problem_info']['n_pixels'],
            'timestamp': results['algorithm_info']['timestamp'],
            'objectives_used': objective_names
        },
        'objective_statistics': {}
    }
    
    # Add statistics for each objective that was used
    for i, obj_name in enumerate(objective_names):
        summary['objective_statistics'][obj_name] = {
            'min': float(objectives[:, i].min()),
            'max': float(objectives[:, i].max()),
            'mean': float(objectives[:, i].mean())
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    generated_files['summary_file'] = summary_filename
    
    # Generate comprehensive optimization report if requested
    if include_reports:
        try:
            report_filename = create_optimization_report(
                results, output_dir, verbose=False,
                run_label=run_label, run_timestamp=run_timestamp,
            )
            generated_files['optimization_report'] = report_filename
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate optimization report: {e}")

        try:
            evolution_filename = create_evolution_tracking_report(
                results, output_dir, verbose=False,
                run_label=run_label, run_timestamp=run_timestamp,
            )
            generated_files['evolution_report'] = evolution_filename
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate evolution report: {e}")
    
    # Generate debug report if requested
    if include_debug:
        debug_report_fn = globals().get('create_debug_report')
        if callable(debug_report_fn):
            try:
                debug_filename = debug_report_fn(results, output_dir, verbose=False)
                generated_files['debug_report'] = debug_filename
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate debug report: {e}")
        elif verbose:
            print("  Warning: create_debug_report is not available; skipping debug report")
    
    if verbose:
        print(f"✓ Complete results saved to: {results_filename}")
        print(f"✓ Summary saved to: {summary_filename}")
        if include_reports and 'optimization_report' in generated_files:
            print(f"✓ Optimization report saved to: {os.path.basename(generated_files['optimization_report'])}")
        if include_reports and 'evolution_report' in generated_files:
            print(f"✓ Evolution report saved to: {os.path.basename(generated_files['evolution_report'])}")
        if include_debug and 'debug_report' in generated_files:
            print(f"✓ Debug report saved to: {os.path.basename(generated_files['debug_report'])}")

    # Append a compact entry to the project-level run registry
    algorithm_info = results.get('algorithm_info', {})
    registry_entry = {
        "run_id": f"{run_timestamp}{label_suffix}_{ecosystem}",
        "run_label": run_label,
        "timestamp": run_timestamp,
        "git_hash": _get_git_hash(),
        "ecosystem": ecosystem,
        "objectives": results.get('objective_names', []),
        "algorithm": {
            "pop_size": algorithm_info.get('pop_size'),
            "n_generations": algorithm_info.get('n_generations'),
            "actual_generations": algorithm_info.get('actual_generations'),
            "random_seed": algorithm_info.get('random_seed'),
            "use_patch_approach": results.get('problem_info', {}).get('is_patch_based'),
        },
        "scenario_params": results.get('scenario_params', {}),
        "files": generated_files,
    }
    _append_to_registry(registry_entry, output_dir=output_dir)

    # ── Auto-export to R-readable format ──────────────────────────────────────
    # If r_export_parent is given (grid mode), write to r_export_parent/run_label/
    # with no timestamp — the parent already carries the grid-level timestamp.
    # Otherwise (single run) write to r_inputs/{timestamp}_{run_label}/.
    try:
        from export_to_r import export_results as _export_to_r
        _base = run_label_slug if run_label_slug else registry_entry["run_id"]
        if r_export_parent is not None:
            r_out_dir = os.path.join(r_export_parent, _base)
        else:
            r_out_dir = os.path.join(output_dir, "r_inputs", f"{run_timestamp}_{_base}")
        _export_to_r(results_filename, output_dir=r_out_dir, nondom_pixels_only=True)
        generated_files["r_export_dir"] = r_out_dir
        if verbose:
            print(f"✓ R export → {os.path.relpath(r_out_dir, output_dir)}/")
    except Exception as e:
        if verbose:
            print(f"  Warning: R export failed: {e}")

    return generated_files

def save_combined_results(combined_results, output_dir=str(OUTPUT_DIR), verbose=True, run_label=""):
    """
    Save combined multi-scenario results.
    
    Args:
        combined_results: Combined results from all scenarios
        output_dir: Directory to save files
        verbose: Print save status
    """
    if verbose:
        print(f"\nSaving combined scenario results...")

    os.makedirs(os.path.join(output_dir, "results_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summary_files"), exist_ok=True)
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    run_label_slug = run_label.replace(" ", "_") if run_label else ""
    label_suffix = f"_{run_label_slug}" if run_label_slug else ""
    
    # Get ecosystem information from first scenario (should be same for all)
    first_scenario_results = next(iter(combined_results['scenarios'].values()))
    ecosystem = first_scenario_results.get('scenario_params', {}).get(
        'ecosystem_label',
        first_scenario_results.get('initial_conditions', {}).get('ecosystem', 'all')
    )
    ecosystem_suffix = f"_{ecosystem}" if ecosystem != 'all' else ""
    
    results_filename = os.path.join(output_dir, f"results_files/results{ecosystem_suffix}_{run_timestamp}{label_suffix}.pkl")
    summary_filename = os.path.join(output_dir, f"summary_files/summary{ecosystem_suffix}_{run_timestamp}{label_suffix}.json")
    print(f"  Saving combined results to: {results_filename}")
    with open(results_filename, 'wb') as f:
        pickle.dump(combined_results, f)
    
    # Convert parameter ranges to JSON-serializable format
    param_ranges = combined_results.get('scenario_parameters', {})
    json_param_ranges = {}
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            json_param_ranges[param_name] = {'type': 'continuous', 'min': param_range[0], 'max': param_range[1]}
        else:
            json_param_ranges[param_name] = {'type': 'categorical', 'values': param_range}
    
    summary = {
        'ecosystem': ecosystem,
        'combined_info': {
            'n_scenarios_run': combined_results['n_scenarios_run'],
            'n_scenarios_total': combined_results['n_scenarios_total'],
            'parameter_ranges': json_param_ranges,
            'sampling_info': {
                'n_samples_per_param': combined_results.get('n_samples_per_param', 3),
                'random_seed': combined_results.get('random_seed', 42)
            },
            'timestamp': combined_results['algorithm_info']['timestamp']
        },
        'scenarios': {}
    }
    
    # Add summary for each scenario
    for scenario_id, scenario_results in combined_results['scenarios'].items():
        objectives = scenario_results['objectives']
        
        # Get objective names from this scenario (avoid circular import)
        objective_names = scenario_results['objective_names']
        
        objective_ranges = {}
        for i, obj_name in enumerate(objective_names):
            objective_ranges[obj_name] = [float(objectives[:, i].min()), float(objectives[:, i].max())]
        
        summary['scenarios'][f'scenario_{scenario_id}'] = {
            'params': scenario_results['scenario_params'],
            'n_solutions': scenario_results['n_solutions'],
            'objectives_used': objective_names,
            'objective_ranges': objective_ranges
        }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"✓ Combined results saved to: {results_filename}")
        print(f"✓ Combined summary saved to: {summary_filename}")

    # Append a compact entry to the project-level run registry
    algorithm_info = combined_results.get('algorithm_info', {})
    registry_entry = {
        "run_id": f"{run_timestamp}{label_suffix}_{ecosystem}_combined",
        "run_label": run_label,
        "timestamp": run_timestamp,
        "git_hash": _get_git_hash(),
        "ecosystem": ecosystem,
        "n_scenarios_run": combined_results.get('n_scenarios_run'),
        "algorithm": {
            "pop_size": algorithm_info.get('pop_size'),
            "n_generations": algorithm_info.get('n_generations'),
            "n_samples_per_param": combined_results.get('n_samples_per_param'),
            "random_seed": combined_results.get('random_seed'),
        },
        "files": {
            "pickle_file": results_filename,
            "summary_file": summary_filename,
        },
    }
    _append_to_registry(registry_entry, output_dir=output_dir)