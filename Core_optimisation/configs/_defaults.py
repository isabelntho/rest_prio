"""The run-configuration contract: every knob a preset may set, and its default.

A preset (the other modules in this package) is a flat block of module-level
constant assignments. run.py overlays a preset onto DEFAULTS and raises on any
name that is not a key here, so a typo'd constant fails loudly instead of
silently doing nothing.

WHY THE FORMAT IS CONSTRAINED
-----------------------------
paper2/_methods.qmd extracts the reported parameters by STATICALLY AST-PARSING
the run config (no code executed), reading top-level ast.Assign nodes whose
values are literals. So a preset must be:

  - plain module-level assignments of LITERAL values,
  - using these exact names,
  - with no imports and no computed values,

with one exception the parser already handles specially: HV_PATIENCE written as
the BinOp `N_GENERATIONS + 1`.

For the same reason a preset carries its FULL constant block rather than only
the values that differ from DEFAULTS: an elided knob is invisible to the AST
parse, which would then silently fall back to the qmd's own hardcoded default -
which may not be the default here. DEFAULTS exists to document the contract,
validate names, and fill knobs a preset genuinely never defines.

SENTINELS
---------
Two defaults of None mean "omit the kwarg entirely" rather than "pass None", and
that distinction is load-bearing:

  HV_PATIENCE     None -> omit, so run_optimization_instance's own default (15,
                  hypervolume early stopping ON) applies. Reproduces the
                  conditional splat in grid_parallel._invoke.
  _condition_tags None -> derive the tag list from CONDITION_GRID_FAMILY rather
                  than taking an explicit list from the preset.
"""

DEFAULTS = {
    # -- algorithm --------------------------------------------------------
    # "nsga2" uses POP_SIZE directly; "nsga3" sizes the population from
    # N_PARTITIONS via das-dennis reference directions.
    "ALGORITHM": "nsga3",
    "POP_SIZE": 92,
    "N_GENERATIONS": 100,
    "N_PARTITIONS": 12,
    "RANDOM_SEED": 101,
    "WARM_SEEDING": False,
    # Expected bitflips per individual per generation (k in prob_var = k/n_var).
    # None keeps the engine default of 200.
    "MUTATION_FLIP_COUNT": None,
    # None = engine default (15, early stopping ON). N_GENERATIONS + 1 disables
    # early stopping so every run in a grid gets the same generation budget.
    "HV_PATIENCE": None,

    # -- what to run ------------------------------------------------------
    # 'forest'/'agricultural'/'grassland' = one filtered ecosystem; 'fg' =
    # forest + grassland pixels; 'all' = three separate runs; 'combined' = one
    # run with no ecosystem filtering.
    "ECOSYSTEM_TO_RUN": "combined",
    "RUN_LABEL": "unlabelled",
    "REGION": "Bern",
    # "custom" | "condition_grid" | "policy_grid" | "factorial".
    # NB: the legacy "all" mode is not supported here - see run.py.
    "SCENARIO_MODE": "custom",
    "OBJECTIVES": ["restoration_benefit", "cost"],
    "CONDITION_SCENARIO": "global_all",
    # int or list of ints; normalised to a list. None = run once at RANDOM_SEED.
    "SEEDS": None,
    # In "custom" mode, replicate the single run once per seed in SEEDS
    # (labels <RUN_LABEL>_seed<n>). False = one run at RANDOM_SEED. The grid
    # modes always iterate SEEDS regardless of this flag.
    "CUSTOM_MULTISEED": False,

    # -- data loading -----------------------------------------------------
    "SAMPLE_FRACTION": None,
    "SAMPLE_SEED": 42,
    "AGGREGATION_FACTOR": None,
    "N_SAMPLES_PER_PARAM": 3,

    # -- scenario parameters ----------------------------------------------
    "custom_scenario_params": {},
    "POLICY_VARIANTS": {},
    "BENCHMARK_SCENARIOS": [],

    # -- factorial design -------------------------------------------------
    "FACTORIAL_FORMS": ["sum"],
    "FACTORIAL_SCALINGS": ["global"],
    "FACTORIAL_CONSTRUCTIONS": ["all"],
    "FACTORIAL_POLICIES": {"status_quo": {}},

    # -- condition grid ---------------------------------------------------
    # Explicit tag list. None = derive from CONDITION_GRID_FAMILY.
    "_condition_tags": None,
    # "weighting_vertex" | "weighting_simplex"; only read when _condition_tags
    # is None.
    "CONDITION_GRID_FAMILY": "weighting_vertex",
    "SIMPLEX_BANDS": ("20", "35", "50"),
    "SIMPLEX_DRAWS": "abcd",

    # -- representation ---------------------------------------------------
    "USE_PATCH_APPROACH": False,
    "PATCH_SIZE": 2,
    "PATCH_CONSTRAINT_TYPE": "pixel_count",
    "PIXEL_TOLERANCE": 0.05,

    # -- diagnostics / execution ------------------------------------------
    # Worker PROCESSES for the grid modes and multi-seed custom mode.
    # 1 = sequential in-process fallback. Bound by RAM, not cores: each
    # concurrent run holds its own ~1.3M-pixel rasters.
    "GRID_WORKERS": 1,
    "SAVE_SNAPSHOTS": False,
    # With SAVE_SNAPSHOTS True: an iterable of algorithm.n_gen values (1-indexed)
    # to snapshot X for, e.g. (1, 11, 26, 51, 101) for gens 0/10/25/50/100 - keeps
    # X_history small. None snapshots EVERY generation (can be tens of GB).
    "SNAPSHOT_GENERATIONS": None,
    "CAPTURE_REPAIR_DIAG": False,
    "N_CAPTURE_GENS": 5,
    "PROFILE": False,
    "PROFILE_TOP_N": 30,
    # Appended to the log filename, to tell concurrent configs apart.
    "LOG_SUFFIX": "",
}

# Engine defaults for the kwargs run_optimization_instance fills in when a
# caller omits them. The parity harness uses this to compare EFFECTIVE kwargs,
# so "omitted" and "passed explicitly at its default" compare equal.
ENGINE_DEFAULTS = {
    "algorithm_type": "nsga3",
    "mutation_flip_count": None,
    "capture_repair_diag": False,
    "n_capture_gens": 5,
    "hv_patience": 15,
    "hv_min_improvement": 1e-6,
    "skip_diagnostics": False,
    "use_repair": True,
    "save_snapshots": False,
    "snapshot_generations": None,
    "warm_seeding": True,
    "n_partitions": 8,
    "patch_size": 100,
    "patch_constraint_type": "pixel_count",
    "pixel_tolerance": 0.05,
    "use_patch_approach": False,
    "r_export_parent": None,
    "run_label": "",
    "run_config": None,
}
