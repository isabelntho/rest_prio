"""Three-objective NSGA-III configuration - process-parallel factorial.

Lifted verbatim from run_custom_parallel.py. Objectives are
restoration_potential x spatial_clustering x cost. Note this uses the STATIC
per-pixel restoration_potential, not the arrangement-dependent
restoration_benefit that nsga3_3obj and nsga2_2obj use.

    python -m Core_optimisation.run nsga3_factorial

This config is also the one that runs the patch representation
(USE_PATCH_APPROACH = True) and warm seeding.

ALGORITHM, HV_PATIENCE, MUTATION_FLIP_COUNT and CAPTURE_REPAIR_DIAG are
deliberately absent: run_custom_parallel.py never passed them, so the engine
defaults (nsga3; patience 15; 200 flips; no repair diagnostics) apply. See
_defaults.py.
"""

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
RUN_LABEL = "iEMSs_fullfact_clustobj"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "condition_grid" sweeps _condition_tags x SEEDS
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
#   "factorial"      fully-crossed design (iEMSs Block 4)
SCENARIO_MODE = "factorial"

# Number of worker PROCESSES for the grid modes. 1 = sequential in-process
# fallback. Bound by available RAM (each concurrent run holds its own rasters).
GRID_WORKERS = 4

# Condition scenario tag - selects pre-computed anomaly rasters from data/anomaly_scenarios/
CONDITION_SCENARIO = "global_all"

# Seeds used by the grid modes.
SEEDS = [101, 102, 103, 104, 105]  # 5 seed replicates (iEMSs run matrix, Block 0)
# Custom mode runs once at RANDOM_SEED, as run_custom_parallel.py did.
CUSTOM_MULTISEED = False

# Condition tag set swept by condition_grid mode (was a local in
# run_custom_parallel._run_condition_grid).
_condition_tags = [
    "global_all",
    "global_drop_smd", "global_drop_sbd", "global_drop_soc",
    "global_drop_uzl", "global_drop_cdi", "global_drop_swf_h",
    "global_drop_swf_t", "global_drop_ndvi",
    "upper_q75_all",
]

OBJECTIVES = ["restoration_potential", "spatial_clustering", "cost"]  # iEMSs headline objectives
# Available names + what each means: see all_objectives in data_loader.py.
# NB: the "spatial_clustering" objective is distinct from the custom_scenario_params
# 'spatial_clustering' knob below, which only soft-biases sampling.

SAMPLE_FRACTION = None
SAMPLE_SEED = 42
POP_SIZE = 92
N_GENERATIONS = 100
RANDOM_SEED = 100
N_SAMPLES_PER_PARAM = 3
N_PARTITIONS = 12
WARM_SEEDING = True

# Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
custom_scenario_params = {
    "max_restoration_fraction": 0.05,
    "spatial_clustering": 0,
    "biotic_effect": 0.01,
    "abiotic_effect": 0.01,
    "normalize_objectives": True,
    "patch_score_temperature": 2.0,
    "patch_repair_top_k": 100,
    "burden_sharing": "no",
    # restoration_potential formulation: "sum" (default) / "threshold" / "shortfall".
    # rp_threshold is the "good state" cutoff / reference level for the latter two.
    # Full semantics: RestorationProblem.__init__ and .evaluate_raw_objectives (optimization_engine.py).
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    # spatial_clustering metric (only used when that objective is in OBJECTIVES):
    # "adjacency" (default) / "components" / "inter_patch_adjacency" (needs patch approach).
    # See RestorationProblem.__init__ (optimization_engine.py) for what each measures.
    "clustering_metric": "adjacency",
}

# Named policy scenarios for SCENARIO_MODE == "policy_grid".
POLICY_VARIANTS = {
    "policy_baseline":         {},
    "policy_ambitious":        {"max_restoration_fraction": 0.10},
    "policy_very_ambitious":   {"max_restoration_fraction": 0.20},
    "policy_burden_shared":    {"burden_sharing": "yes"},
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10,
                                "burden_sharing": "yes"},
}

# Benchmark condition scenarios run in policy_grid mode with baseline params x SEEDS.
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# -- Factorial design (SCENARIO_MODE == "factorial") - iEMSs Block 4 ----------
FACTORIAL_FORMS = ["sum", "threshold", "shortfall"]
FACTORIAL_SCALINGS = ["global", "upper_q75"]
FACTORIAL_CONSTRUCTIONS = [
    "all",
    "drop_sbd",
    "drop_soc",
    "drop_ndvi",
    # "drop_<highest_leverage_1>",   # <- fill from Block 2 R3c Jaccard
    # "drop_<highest_leverage_2>",   # <- fill from Block 2 R3c Jaccard
]
FACTORIAL_POLICIES = {
    "status_quo":    {},
    "ambitious":     {"max_restoration_fraction": 0.10},
    "burden_shared": {"burden_sharing": "yes"},
}

# Patch approach settings
USE_PATCH_APPROACH = True
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.05 #previously 0.15

# Spatial aggregation factor (None/1 disables).
AGGREGATION_FACTOR = None

# Save per-generation population snapshots for animation.
SAVE_SNAPSHOTS = False
# Restrict snapshots to these algorithm.n_gen values (1-indexed) instead of every
# generation, e.g. (1, 11, 26, 51, 101) for gens 0/10/25/50/100. None = every
# generation (X_history can reach tens of GB at this run's pop/pixel count).
SNAPSHOT_GENERATIONS = None

# Profile the run with cProfile.
PROFILE = False
PROFILE_TOP_N = 30
