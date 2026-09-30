"""Two-objective NSGA-II configuration - the reported baseline.

Lifted verbatim from run_custom_nsga2.py. TWO objectives (restoration_benefit +
cost) under NSGA-II: restoration_benefit vs cost is the real headline trade-off
once spatial_clustering is dropped (clustering shadows cost). With two
objectives the Pareto front is a curve, so NSGA-II's crowding-distance diversity
works well and NSGA-III is unnecessary.

    python -m Core_optimisation.run nsga2_2obj

This is the config paper2/_methods.qmd AST-parses for the reported parameters -
keep every value a plain top-level literal. See _defaults.py.
"""

# Algorithm selector passed to run_optimization_instance. "nsga2" uses NSGA-II
# with an explicit POP_SIZE (below).
ALGORITHM = "nsga2"

# Available ecosystem run modes:
# - 'forest'/'agricultural'/'grassland': run one filtered ecosystem
# - 'fg': run one optimisation using forest + grassland pixels
# - 'all': run three separate optimisations (one per ecosystem)
# - 'combined': run one optimisation without ecosystem filtering
ECOSYSTEM_TO_RUN = "combined"

# Short human-readable label describing what this run is testing.
# Used in output filenames and the run_registry.jsonl log.
RUN_LABEL = "CH_patch_rerun"

# Region used for validation reference in load_initial_conditions
REGION = "CH"

# Choose scenario mode:
#   "custom"         runs exactly one scenario using custom_scenario_params
#   "condition_grid" sweeps the condition tag list x SEEDS (currently the
#                    indicator-weighting set - see CONDITION_GRID_FAMILY)
#   "policy_grid"    runs each entry in POLICY_VARIANTS once against CONDITION_SCENARIO
#   "factorial"      fully-crossed design over FACTORIAL_FORMS x FACTORIAL_SCALINGS x
#                    FACTORIAL_CONSTRUCTIONS x FACTORIAL_POLICIES x SEEDS (iEMSs Block 4)
SCENARIO_MODE = "custom"

# Which family of tags "condition_grid" mode sweeps.
#   "weighting_vertex"  the original 13 rule-based vertices: w_flat, w_cat and the 11
#                       focal vertices. These are LOCAL - they span L1 0 to 0.150 of a
#                       weight simplex that reaches 0.65.
#   "weighting_simplex" the extended campaign: w_flat (re-run so this campaign carries
#                       its own reference cell and seed floor) plus 12 explicit Dirichlet
#                       vectors at L1 0.20 / 0.35 / 0.50, four per band. Rasters come from
#                       the GENERATE_SIMPLEX_WEIGHT_SCENARIOS block of data/ec_anomalies.r,
#                       driven by data/diagnostics/weight_vectors_simplex.csv.
# Both are 13 tags x 5 seeds = 65 runs, so the two campaigns are directly comparable.
CONDITION_GRID_FAMILY = "weighting_simplex"
# L1 bands of the extended campaign, as the integer percentages used in the tag grammar
# global_w_d{band}{a-d}. Must match what weight_simplex_screen.py --emit-vectors wrote.
SIMPLEX_BANDS = ("20", "35", "50")
SIMPLEX_DRAWS = "abcd"

# Condition scenario tag - selects pre-computed anomaly rasters from
# data/anomaly_scenarios/ (or data/CH_wide/ when REGION == "CH"). All tags are
# produced by data/ec_anomalies.r and follow the "{scaling}_{construction}" grammar
# that the downstream R analysis parses.
# Available tags:
#   benchmark scaling:  global_all | upper_q25_all | upper_q50_all | upper_q75_all |
#                       upper_q90_all | zones_all
#                       global .. q90 form ONE ORDERED series (global = all pixels =
#                       the q0 end): how ambitious is the reference condition.
#   indicator LOO:      global_drop_smd | global_drop_sbd | global_drop_soc |
#                       global_drop_uzl | global_drop_tsd | global_drop_can |
#                       global_drop_cdi | global_drop_swf_h | global_drop_swf_t |
#                       global_drop_lai | global_drop_ndvi
#                       (plus upper_q75_drop_* for the agricultural indicators)
#   indicator weighting: global_w_flat | global_w_cat | global_w_<indicator>
#                       Weighting tags write the SAME all-indicator weighted
#                       composite to both the abiotic_ and biotic_ rasters, so the
#                       objective's 1:1 category sum carries the weight vector.
CONDITION_SCENARIO = "global_all"

# Seeds used by the grid modes, and (with CUSTOM_MULTISEED) by custom mode.
#   List[int] - runs each scenario/variant once per seed; labels: <name>_seed<n>
#   None       - runs each scenario/variant once using RANDOM_SEED; labels: <name>
SEEDS = [101, 102, 103]
# Custom mode replicates the single run across SEEDS (was _MULTISEED).
CUSTOM_MULTISEED = True

# TWO-objective formulation: restoration_benefit (spatially-explicit ecological
# benefit) vs cost. "cost" maps to implementation_cost; "restoration_benefit" is the
# total abiotic+biotic anomaly improvement INCLUDING neighbour spillover, so it depends
# on WHICH pixels AND their arrangement (unlike the static restoration_potential).
# NSGA-II handles the 2-objective front.
OBJECTIVES = ["restoration_benefit", "cost"]
# Available names + what each means: see all_objectives in data_loader.py.
SAMPLE_FRACTION = None
SAMPLE_SEED = 42
# NSGA-II uses POP_SIZE directly (unlike NSGA-III, which sizes the population
# from N_PARTITIONS). N_PARTITIONS is left defined for API compatibility but is
# unused when ALGORITHM == "nsga2".
POP_SIZE = 100
N_GENERATIONS = 150
# Worker PROCESSES for the grid modes (parallelises whole optimisation instances:
# one task per condition tag in condition_grid, one task per (tag, seed) cell in
# factorial). 1 = sequential in-process fallback. Bound by RAM, not cores - each
# concurrent run holds its own ~1.3M-pixel rasters.
GRID_WORKERS = 4
RANDOM_SEED = 101
N_SAMPLES_PER_PARAM = 3
N_PARTITIONS = 12  # unused for NSGA-II; kept for run_optimization_instance API
WARM_SEEDING = False
# Expected number of bitflips per individual per generation (k in prob_var = k / n_var).
# None keeps the historical default of 200.
MUTATION_FLIP_COUNT = None  # e.g. 100, 200, 400
# Hypervolume early-stopping patience forwarded to run_optimization_instance.
#   None              keep its default (15) - early stopping ON.
#   N_GENERATIONS + 1 early stopping OFF, so every run gets the SAME generation
#                     budget. Use this whenever runs are being compared: a run that
#                     stops at generation 20 has a smaller front than one that ran
#                     to 100 for reasons that have nothing to do with the scenario.
HV_PATIENCE = 15

# Custom single scenario parameters (only used when SCENARIO_MODE == "custom")
custom_scenario_params = {
    "max_restoration_fraction": 0.05,
    "spatial_clustering": 0,
    "biotic_effect": 0.01,
    "abiotic_effect": 0.01,
    "normalize_objectives": True,
    "burden_sharing": "no",
    # restoration_potential formulation: "sum" (default) / "threshold" / "shortfall".
    # Inert here since restoration_potential is not in OBJECTIVES above; see
    # RestorationProblem.__init__ (resto_anom.py) for what each level means.
    "rp_formulation": "sum",
    "rp_threshold": 0.0,
    # Pixel-mode sampling strategy for the 2-objective potential-vs-cost run:
    #   "scattered"   = AdaptiveSampling + bitflip: the true UNCONSTRAINED front
    #                   (cherry-pick anywhere) - the baseline the contiguity sweep
    #                   is compared against. Produces very scattered plans.
    #   "region_grow" = contiguous regions; pair with min_patch_size to impose the
    #                   minimum-patch-size constraint (the "price of contiguity" sweep).
    "sampling_strategy": "scattered",
    # Neutral repair: with "scattered", sampling (AdaptiveSampling) and mutation
    # (bitflip) are already unbiased; repair_scored=False passes scores=None to
    # AdaptiveRepair so it only enforces the budget (constraints-only, no score bias).
    # Gives an operator-unbiased front on restoration_benefit. Set True (or drop) to
    # restore score-based repair.
    "repair_scored": False,
    # Minimum-patch-size constraint (price-of-contiguity sweep axis). Every
    # 4-connected component of selected pixels must be >= min_patch_size pixels,
    # enforced by MinPatchSizeRepair. 1 = OFF (region_grow with no size floor).
    # Ignored on the "scattered" path (no contiguity constraint). The sweep is driven
    # by Debugs_tests/contiguity_price_sweep.py, which overrides this across levels.
    "min_patch_size": 2,
    # Region operator knobs (used by region_grow / the min-patch-size repair).
    "region_seeds": 800,             # max seed regions per individual (avg region ~ budget/seeds)
    "region_seeds_min": 300,          # min seed regions per individual
    "region_growth_bias": "scored", # "scored" (high value / low cost) or "neutral" (random)
    "region_random_share": 0.5,     # fraction of region seeds placed at random (vs scored)
    "region_mutation_edits": 30,   # grow/shrink edit size per mutated individual
    "region_seed_grid": 40,
    # Stochastic pixel-ordering temperature for the 'scored' region operators
    # (inert when region_growth_bias is "neutral"). 0 = the old deterministic
    # argsort, which made independent individuals growing in the same area
    # converge on the IDENTICAL pixel set - the blocky, uniform-value RFOP map.
    # Higher = more random but still score-preferring (Gumbel-top-k).
    # Sweep at pop 50 x 100 gens, seed 101: T=0 plateau_frac 0.45 / 66.5k pixels
    # selected; T=0.5 0.26 / 86.8k; T=1.0 0.15 / 110.3k. Front quality moved the
    # other way (common-ref HV -16.8% at T=0.5, -22.7% at T=1.0), but T=1.0 was
    # the only arm still climbing at gen 100, so the deficit may be slower
    # convergence rather than a worse reachable front.
    "region_score_temperature": 1.0,
    # Warm-start pre-optimisation budget (only used when WARM_SEEDING is True and
    # use_patch_approach is False). Objectives without a per-pixel score
    # (restoration_benefit, spatial_clustering) get a short single-objective GA
    # seed; these set that GA's population and generations. Keep small - one run
    # per such objective happens before every optimisation run.
    "warm_seed_preopt_pop": 40,
    "warm_seed_preopt_gens": 30,
}

# Named policy scenarios for SCENARIO_MODE == "policy_grid".
POLICY_VARIANTS = {
    "policy_baseline":         {},                                                  # mirrors condition baseline
    "policy_ambitious":        {"max_restoration_fraction": 0.10},                 # double the budget
    "policy_very_ambitious":   {"max_restoration_fraction": 0.20},                 # 4x the budget
    "policy_burden_shared":    {"burden_sharing": "yes"},                          # equal burden across regions
    "policy_ambitious_burden": {"max_restoration_fraction": 0.10,
                                "burden_sharing": "yes"},                          # ambitious + burden shared
}

# Benchmark condition scenarios run in policy_grid mode with baseline params x SEEDS.
BENCHMARK_SCENARIOS = ["upper_q75_all"]

# -- Factorial design (SCENARIO_MODE == "factorial") - iEMSs Block 4 --
# CURRENT CAMPAIGN: the benchmark-quantile sweep. Only the SCALING axis varies;
# form / construction / policy are single-level, so the grid is 5 scalings x 5 seeds
# = 25 runs.
#
# FACTORIAL_FORMS is deliberately ONE level. rp_formulation is read only inside the
# restoration_potential objective (resto_anom.py), and OBJECTIVES above is
# ["restoration_benefit", "cost"] - so every form level would produce an IDENTICAL
# run and merely triple the grid. Restore ["sum", "threshold", "shortfall"] only if
# restoration_potential is put back into OBJECTIVES.
FACTORIAL_FORMS = ["sum"]
# The benchmark axis, as one ordered series (global = all pixels = the q0 end):
# how ambitious is the reference condition each indicator is z-scored against.
# Read as a dose-response - plan overlap vs quantile - NOT via hypervolume: each
# level rescales the objective differently, so HV is not comparable across them.
#
# CURRENTLY RESTRICTED TO "zones" - the zone-wise benchmark campaign. global /
# q25 / q50 / q75 are already in outputs/run_registry.jsonl at the full 12
# constructions x 5 seeds (253 / 60 / 60 / 249 runs), so re-listing them here
# would recompute ~240 finished runs. Restore the full list only to rebuild the
# quantile series from scratch:
#   FACTORIAL_SCALINGS = ["global", "upper_q25", "upper_q50", "upper_q75"]
#
# NOTE: "zones" is NOT part of the ordered quantile series above - it is a
# different KIND of reference (each production-region x altitude zone is centred
# on its own mean) rather than a more/less ambitious level of the same one. It
# belongs on the benchmark axis as a separate categorical level, so do not plot
# it inside the global..q90 dose-response.
FACTORIAL_SCALINGS = ["upper_q75", "zones"]
# Construction axis: all-indicator composite + the full 11-indicator leave-one-out.
# Every scaling level above carries this SAME set of drop_* rasters, so the
# scaling x construction design is balanced. (upper_q90 is excluded from
# FACTORIAL_SCALINGS for that reason - it is all-only - and because the benefit
# objective is nearly flat there: best plan / random plan is 1.13x at q90.)
FACTORIAL_CONSTRUCTIONS = [
    "all",
    #"drop_smd", "drop_sbd", "drop_soc",                    # abiotic
    #"drop_uzl", "drop_cdi", "drop_swf_h", "drop_swf_t",    # biotic, ag/grassland
    #"drop_ndvi",
    #"drop_tsd", "drop_can", "drop_lai",                    # biotic, forest only
]
FACTORIAL_POLICIES = {
    "status_quo":    {},                                  # baseline budget, no burden sharing
}

# Patch approach settings. The contiguity work uses the PIXEL representation
# (region operators + min-patch-size constraint), so the patch approach is off;
# set True only to run the fixed-2x2-grain patch variant instead.
USE_PATCH_APPROACH = True
PATCH_SIZE = 2
PATCH_CONSTRAINT_TYPE = 'pixel_count'
PIXEL_TOLERANCE = 0.01 #0.05

# Spatial aggregation: block-coarsen all input rasters by this integer factor before optimisation.
AGGREGATION_FACTOR = None

# Set to True to save per-generation population snapshots for animation
SAVE_SNAPSHOTS = False
# Restrict snapshots to these algorithm.n_gen values (1-indexed) instead of every
# generation, e.g. (1, 11, 26, 51, 101) for gens 0/10/25/50/100. None = every
# generation (X_history can reach tens of GB at this run's pop/pixel count).
SNAPSHOT_GENERATIONS = (1, 11, 26, 51, 101, 151)

# Set to True to capture paired pre-repair vs post-repair genotype/phenotype diversity.
CAPTURE_REPAIR_DIAG = True
N_CAPTURE_GENS = 5

# Set to True to profile the run with cProfile and print the top 30 hotspots afterwards.
PROFILE = False
PROFILE_TOP_N = 30

# Distinguishes this config's log file from the 3-objective ones.
LOG_SUFFIX = "_nsga2"
