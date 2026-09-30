"""Three-objective NSGA-III configuration - sequential.

Lifted verbatim from run_custom.py. Headline objectives are
restoration_benefit x spatial_clustering x cost under NSGA-III (population sized
from N_PARTITIONS reference directions, so POP_SIZE is advisory).

    python -m Core_optimisation.run nsga3_3obj

GRID_WORKERS = 1 keeps this sequential, matching the original script's in-process
loops. Raise it to fan the grid modes out across processes.

ALGORITHM and HV_PATIENCE are deliberately absent: run_custom.py never passed
algorithm_type or hv_patience, so the engine defaults (nsga3; patience 15, early
stopping ON) apply. See _defaults.py.
"""

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
# Used in output filenames and the run_registry.jsonl log.
RUN_LABEL = "3obj_with_benefitcorr_pixscattered"

# Region used for validation reference in load_initial_conditions
REGION = "Bern"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "condition_grid" sweeps _condition_tags x SEEDS
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
#   "factorial"      fully-crossed design over FACTORIAL_FORMS x FACTORIAL_SCALINGS x
#                    FACTORIAL_CONSTRUCTIONS x FACTORIAL_POLICIES x SEEDS (iEMSs Block 4)
SCENARIO_MODE = "custom"

# Condition scenario tag - selects pre-computed anomaly rasters from data/anomaly_scenarios/
# Available tags:
#   global_all | global_drop_smd | global_drop_sbd | global_drop_soc |
#   global_drop_uzl | global_drop_tsd | global_drop_can | global_drop_cdi |
#   global_drop_swf_h | global_drop_swf_t | global_drop_lai | global_drop_ndvi |
#   upper_q75_all
CONDITION_SCENARIO = "global_all"

# Seeds used by the grid modes.
#   List[int] - runs each scenario/variant once per seed; labels: <name>_seed<n>
#   None       - runs each scenario/variant once using RANDOM_SEED; labels: <name>
# (run_custom.py wrote this as the scalar 101, which crashed the grid modes at
# len(SEEDS); a one-element list is the same single seed and actually runs.)
SEEDS = [101]
# Custom mode runs once at RANDOM_SEED, as run_custom.py did.
CUSTOM_MULTISEED = False

# Agricultural-focus LOO set (iEMSs Blocks 1+2), swept by condition_grid mode.
# LOO is restricted to indicators present in the agricultural EC set (setup.r):
# smd/sbd/soc (abiotic) + uzl/cdi/swf_h/swf_t/ndvi (biotic). tsd/can/lai are
# forest-only - dropping them is a no-op over agricultural pixels - so they are
# excluded. upper_q75_all is the q75 benchmark for R3.
_condition_tags = [
    "global_all",
    "global_drop_smd", "global_drop_sbd", "global_drop_soc",
    "global_drop_uzl", "global_drop_cdi", "global_drop_swf_h",
    "global_drop_swf_t", "global_drop_ndvi",
    "upper_q75_all",
]

OBJECTIVES = ["restoration_benefit", "spatial_clustering", "cost"]  # iEMSs headline objectives
# Available names + what each means: see all_objectives in data_loader.py.
# NB: the "spatial_clustering" objective is distinct from the custom_scenario_params
# 'spatial_clustering' knob below, which only soft-biases sampling.
SAMPLE_FRACTION = None
SAMPLE_SEED = 42
POP_SIZE = 92 #previously 50?
N_GENERATIONS = 150
RANDOM_SEED = 101 #42
N_SAMPLES_PER_PARAM = 3
N_PARTITIONS = 12#for 3 objectives 12 # for 4 objectives 6
WARM_SEEDING = False
# Expected number of bitflips per individual per generation (k in prob_var = k / n_var).
# None keeps the historical default of 200. Increase for more exploration, decrease to
# converge faster (at the risk of premature convergence).
MUTATION_FLIP_COUNT = None  # e.g. 100, 200, 400
#previously n partitions 12, pop size 50

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
    # Full semantics: RestorationProblem.__init__ and .evaluate_raw_objectives (resto_anom.py).
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    # spatial_clustering metric (only used when that objective is in OBJECTIVES):
    # "adjacency" (default) / "components" / "inter_patch_adjacency" (needs patch approach).
    # See RestorationProblem.__init__ (resto_anom.py) for what each measures.
    "clustering_metric": "inter_patch_adjacency",
    # Pixel-mode sampling strategy (ignored when USE_PATCH_APPROACH=True):
    # "scattered" (default, scattered plans) / "region_grow" (contiguous regions) /
    # "region_evolve" (regions that relocate/spawn/recombine as wholes, avoiding one
    # basin). See _build_operators (resto_anom.py) for the operator wiring.
    "sampling_strategy": "scattered",
    # Minimum-patch-size constraint (price-of-contiguity). Every 4-connected component
    # of selected pixels must be >= min_patch_size pixels, enforced by MinPatchSizeRepair
    # on the region_grow path. 1 = OFF (no size floor). Matches nsga2_2obj.
    "min_patch_size": 2,
    "region_seeds": 25,               # max seed regions per individual (varies per individual down to region_seeds_min)
    "region_seeds_min": 5,            # min seed regions per individual (region_evolve seeding)
    "region_seed_grid": 16,           # coarse grid (NxN) that spreads initial seeds across the map (region_evolve)
    "region_growth_bias": "scored",   # "scored" (toward high-value/low-cost) or "neutral" (random)
    "region_mutation_edits": 100,     # grow/shrink edit size per mutated individual
}

# Named policy scenarios for SCENARIO_MODE == "policy_grid".
# Each entry is a dict of parameter overrides applied on top of custom_scenario_params.
# An empty dict {} means "no overrides" - mirrors the condition baseline run.
POLICY_VARIANTS = {
    "policy_baseline":         {},                                                  # mirrors condition baseline
    "policy_ambitious":        {"max_restoration_fraction": 0.10},                 # double the budget
    "policy_very_ambitious":   {"max_restoration_fraction": 0.20},                 # 4x the budget
    "policy_burden_shared":    {"burden_sharing": "yes"},                          # equal burden across regions
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10,
                                "burden_sharing": "yes"},                          # ambitious + burden shared
}

# Benchmark condition scenarios run in policy_grid mode with baseline params x SEEDS.
# Labels: <tag>_seed<n>  (or <tag> when SEEDS is None).
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# -- Factorial design (SCENARIO_MODE == "factorial") - iEMSs Block 4 ----------
# Fully-crossed design over the decision-making + model formulation factors.
# Every cell is the product (form x scaling x construction x policy) x SEEDS.
#
#   FACTORIAL_FORMS         objective target form -> scenario_params['rp_formulation']
#                           {"sum", "threshold", "shortfall"}
#                           (Axis 2: what counts as success)
#   FACTORIAL_SCALINGS      condition reference benchmark -> raster tag PREFIX.
#                           "global" = anomaly, "upper_q75" = q75 (Axis: scaling).
#   FACTORIAL_CONSTRUCTIONS condition-indicator construction -> raster tag SUFFIX.
#                           "all" = full indicator set; "drop_<ind>" / reduced builds.
#   FACTORIAL_POLICIES      policy/governance lever -> overrides on custom_scenario_params.
#
# scaling x construction together select the condition_scenario raster tag,
# built as f"{scaling}_{construction}" (e.g. "global_all", "upper_q75_drop_smd"),
# which must match the .tif files written by ec_anomalies.r and resolved in
# data_loader.load_initial_conditions.
#
# Each cell's factor levels are written to its run_config as flat factor_* keys
# (factor_form / factor_scaling / factor_construction / factor_policy) so they
# survive export_to_r -> metadata.json and are recoverable for variance
# partitioning in R (load_run_factorial / compute_rfop_variance_partition).
FACTORIAL_FORMS = ["sum", "threshold", "shortfall"]
FACTORIAL_SCALINGS = ["global", "upper_q75"]
# Construction axis = 3 levels: "all" + the 2 HIGHEST-LEVERAGE agricultural drops.
# Which 2 comes from Block 2's R3c Jaccard, so RUN BLOCKS 1+2 FIRST, then fill the
# two drop_<ind> slots below.
#   Available drops: smd sbd soc uzl cdi swf_h swf_t ndvi
FACTORIAL_CONSTRUCTIONS = [
    "all",
    "drop_smd",   # <- fill from Block 2 R3c Jaccard
    #"drop_<highest_leverage_2>",   # <- fill from Block 2 R3c Jaccard
]
FACTORIAL_POLICIES = {
    "status_quo":    {},                                  # baseline budget, no burden sharing
    "ambitious":     {"max_restoration_fraction": 0.10},  # larger target area
    "burden_shared": {"burden_sharing": "yes"},           # equal burden across regions
}

# Patch approach settings
USE_PATCH_APPROACH = False
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.05#.15

# Spatial aggregation: block-coarsen all input rasters by this integer factor before optimisation.
# 2 = halve resolution in each dimension (~4x fewer pixels). Set to None or 1 to disable.
AGGREGATION_FACTOR = None

# Set to True to save per-generation population snapshots for animation
# Output: intermediate_results/X_history_{timestamp}.npz  shape=(n_gens, pop_size, n_var) int8
SAVE_SNAPSHOTS = False
# Restrict snapshots to these algorithm.n_gen values (1-indexed) instead of every
# generation, e.g. (1, 11, 26, 51, 101) for gens 0/10/25/50/100. None = every
# generation (X_history can reach tens of GB at this run's pop/pixel count).
SNAPSHOT_GENERATIONS = None

# Set to True to capture paired pre-repair vs post-repair genotype/phenotype
# diversity at a few evenly-spaced generations (pixel mode / AdaptiveRepair only).
# Output: <output_dir>/repair_diagnostics/repair_diag_gen{NNNN}.npz
#   X_pre, X_post (int8 genotypes), F_pre, F_post (raw objectives), n_pixels.
# Analyse offline with Debugs_tests/repair_diversity_report.py.
CAPTURE_REPAIR_DIAG = True
N_CAPTURE_GENS = 5

# Sequential, matching run_custom.py's in-process loops. Raise to fan the grid
# modes out across worker processes.
GRID_WORKERS = 1

# Set to True to profile the run with cProfile and print the top 30 hotspots afterwards.
# Results are also written to logs/profile_<run_label>.txt
PROFILE = False
PROFILE_TOP_N = 30
