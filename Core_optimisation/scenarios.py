# =============================================================================
# SCENARIO PARAMETERS
# =============================================================================

def define_scenario_parameters():
    """
    Define continuous ranges for scenario parameters
    Returns:
        dict: Dictionary with parameter names as keys and (min, max) tuples for continuous ranges
              or lists for categorical parameters
    """
    scenario_parameters = {
        'max_restoration_fraction': [0.1, 0.3],#(0.10, 0.50), # Proportion of study area to restore (min, max)
        'spatial_clustering': [0, 0.3],#, 1.0), # Degree of spatial clustering (0=random, 1=highly clustered)
        'burden_sharing': ['no'], # Equal sharing across admin regions (categorical)
        # Separate improvement parameters for each anomaly type
        'abiotic_effect': (0.005, 0.02), # Reduction in abiotic anomaly (min, max)
        'biotic_effect': (0.005, 0.02), # Reduction in biotic anomaly (min, max)
        #'landscape_effect': [0.01],#(0.005, 0.02), # Reduction in landscape anomaly (min, max)
        # Anomaly-dependent improvement weighting
        'anomaly_weight_shape': ['exponential'], # Shape of weighting function
        'anomaly_weight_scale': [1.0], # Scale parameter for weight decay
    }
    
    return scenario_parameters

def sample_scenario_parameters(n_samples_per_param=3, random_seed=42):
    """
    Sample values from continuous parameter ranges and combine with categorical parameters.
    
    Args:
        n_samples_per_param: Number of samples to draw from each continuous parameter
        random_seed: Random seed for reproducible sampling
        
    Returns:
        list: List of parameter combinations (dicts)
    """
    from itertools import product
    import numpy as np
    
    # Set random seed for reproducible sampling
    np.random.seed(random_seed)
    
    params = define_scenario_parameters()
    sampled_values = {}
    
    # Sample from continuous ranges or use categorical values
    for param_name, param_range in params.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Continuous parameter - sample uniformly between min and max
            min_val, max_val = param_range
            samples = np.linspace(min_val, max_val, n_samples_per_param)
            sampled_values[param_name] = samples.tolist()
        else:
            # Categorical parameter - use all values
            sampled_values[param_name] = param_range
    
    # Generate all combinations of sampled values
    param_names = list(sampled_values.keys())
    param_value_lists = list(sampled_values.values())
    
    combinations = []
    for combo in product(*param_value_lists):
        combo_dict = dict(zip(param_names, combo))
        combinations.append(combo_dict)
    
    return combinations

def get_scenario_combinations():
    """
    Get all possible combinations of scenario parameters (legacy function).
    Now uses sampling approach with 3 values per continuous parameter.
    
    Returns:
        list: List of parameter combinations (dicts)
    """
    return sample_scenario_parameters(n_samples_per_param=3)

def get_scenario_by_index(scenario_index, n_samples_per_param=3, random_seed=42):
    """
    Get a specific scenario by index using sampled parameters.
    
    Args:
        scenario_index: Index of the scenario (0-based)
        n_samples_per_param: Number of samples per continuous parameter
        random_seed: Random seed for reproducible sampling
        
    Returns:
        dict: Parameter combination for the scenario
    """
    combinations = sample_scenario_parameters(n_samples_per_param, random_seed)
    
    if 0 <= scenario_index < len(combinations):
        return combinations[scenario_index]
    else:
        raise ValueError(f"Scenario index {scenario_index} out of range. Available: 0-{len(combinations)-1}")

def expand_scenarios(n_samples_per_param=3, random_seed=42):
    """
    Returns the scenario combinations list in the same format as sample_scenario_parameters.
    """
    return sample_scenario_parameters(n_samples_per_param=n_samples_per_param, random_seed=random_seed)
