# Development Tracker

Record of what each core script does, changes made with copilot over time

### Script Relationships (Diagram)

<pre class="mermaid">
flowchart TD
    DL[data_loader.py]
    SP[spatial_operations.py]
    PA[patch_approach.py]
    SC[scenarios.py]
    RS[results_saving.py]
    RA[resto_anom.py]
    UT[utils.py]

    RPE[run_patch_size_experiments.py]
    RMS[run_multi_seed_optimization.py]

    VIS[visualisations.py]
    RV[run_vis.py]
    ASA[analyze_spatial_agreement.py]
    SD[seeds.py]

    OUT[(results_*.pkl / summary_*.json / reports)]
    POUT[(patch_tests* / multi_seed_results)]

    DL --> RA
    SP --> RA
    PA --> RA
    SC --> RA
    RA --> RS
    RS --> OUT

    RPE --> RA
    RMS --> RA
    RPE --> POUT
    RMS --> POUT

    UT --> VIS
    UT --> ASA
    UT --> SD
    OUT --> VIS
    POUT --> VIS
    VIS --> RV
    VIS --> ASA
</pre>

<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
mermaid.initialize({ startOnLoad: true });
</script>

## Scripts and Contributions

### Core Optimization Pipeline
- resto_anom.py: Main optimization engine. Defines objectives and constraints, initializes patch mode, runs NSGA-II, and orchestrates saving/reporting. 

#### data_loader.py
Loads rasters and preprocessing inputs; builds masks and initial_conditions for optimization.
- load_initial_conditions: builds the main initial_conditions object consumed by optimization.
- load_lulc_raster: loads and aligns LULC rasters to analysis extent/grid.
- create_ecosystem_mask: builds ecosystem masks that define where restoration optimization is run.
- load_admin_regions: loads/administers region geometry used in burden-sharing logic.
- get_region_reference: provides regional raster reference (bounds, transform, shape).
- ECOSYSTEM_TYPES, FOCAL_CLASSES: shared constants that drive masking and landscape focal classes.

#### patch_approach.py
Patch-based decision framework. Builds patch mappings, converts patch decisions to pixel decisions, and provides patch-aware sampling and repair operators.
- create_patch_mappings: creates restoration/conversion patch systems and index mappings.
- PatchAwareSampling: patch-level sampler that constructs feasible candidates near target pixels.
- PatchRepair: patch-level repair enforcing patch/pixel-count constraints.
- aggregate_patch_scores_from_pixel_scores: aggregates pixel merit into patch scores for score-guided operators.

#### spatial_operations.py
Spatial helper functions and custom operators for clustering, burden sharing, and landscape density calculations.
- apply_burden_sharing: redistributes selected pixels across admin units under burden-sharing settings.
- apply_spatial_clustering: applies clustering pressure to make selections more spatially coherent.
- _enforce_exact_pixel_count: hard repair utility to hit exact target counts.
- AdaptiveSampling: pixel-based custom sampling operator for initialization.
- AdaptiveRepair: pixel-based custom repair operator used during evolution.
- compute_sn_dens, compute_sn_dens_array: neighborhood-density functions used for landscape condition calculations.

Note: `merge_overlapping_regions` and `approximate_region_landscape_change` were removed from this file (unused) and placed in `old_debugging/spatial_operations_legacy.py` for reference.

#### utils.py
Shared utilities consumed across the pipeline.
- pickle_load: loads `.pkl` result files with compatibility handling for missing classes and numpy version differences. Used by `visualisations.py`, `analyze_spatial_agreement.py`, and `seeds.py`.

#### scenarios.py
Scenario parameter definitions and scenario expansion/sampling helpers.
- define_scenario_parameters: defines scenario parameter space and defaults.
- sample_scenario_parameters: stochastic sampling of parameter values.
- expand_scenarios: expands sampled parameters into scenario combinations for batch optimization.

#### results_saving.py
Writes result files, summaries, and reports from optimization runs.
- save_results_with_reports: high-level save path that writes outputs plus report artifacts.
- save_scenario_results: writes per-scenario result bundles.
- save_combined_results: writes aggregated multi-scenario outputs.
- save_parameter_summary: writes parameter-space summary metadata for runs.

### Experiment Runners
- run_patch_size_experiments.py: Batch runner for patch-size sensitivity experiments across seeds/ecosystems.
- run_multi_seed_optimization.py: Batch runner for seed robustness analysis for one ecosystem setup.

### Analysis and Visualization
- visualisations.py: Main plotting and analysis utilities (selection frequency maps, Pareto plots, parallel coordinates, patch-size comparison grids, etc.).
- analyze_spatial_agreement.py: Computes and compares spatial agreement metrics between runs/approaches.
- seeds.py: Multi-seed analysis — loads results, calculates hypervolume, detects duplicates, plots seed comparisons.
- run_vis.py: Convenience script to generate visual outputs from saved results.
- Bivariate_plot.r, ec_anomalies.r, topo_aggregation.r, setup.r: R scripts for exploratory analysis and figures.

### Legacy and Reference
- MDMO_Aug25.py: Original prototype for the optimization pipeline. Kept for reference only — do not run or import.
- old_debugging/: Archived debug and one-off analysis scripts (debug_simple.py, debug_spatial_agreement.py, simple_inspect.py, check_shapes.py, check_patch_sizes.py, spatial_operations_legacy.py).

### Data, Outputs, and Environment
- environment.yml: Python environment definition.
- patch_tests/, patch_tests_1803/, patch_tests_1903/, patch_tests_1903_2/: Patch experiment outputs.
- multi_seed_results/: Multi-seed output artifacts.
- results_*.pkl, summary_*.json, optimization_report_*.json, evolution_report_*.json: Saved optimization artifacts and reports.
- figs/, notebook_outputs/, optimization_results_analysis_files/: Generated plots and report assets.

## Work Log

### 2026-03-19

Improved the patch-based optimization
- previously mostly deterministic based on number of eligible pixels per patch, meaning there was very little diversity in potential solutions
- now uses score-guided stochastic sampling and repair. Patch size (number of eligible pixels) now has an explicit but blended influence in selection probabilities, so larger patches can be favored when useful without dominating every choice. 
- = better exploration of diverse feasible solutions.

#### Patch approach updates (primary)
- Added stable stochastic helper utilities in patch_approach.py:
	- _safe_softmax: numerically stable probability normalization with temperature control.
	- _sample_without_replacement_weighted: weighted unique sampling used by both sampling and repair.
- Added aggregate_patch_scores_from_pixel_scores in patch_approach.py to convert pixel-level merit into patch-level scores (mean or quantile90), then normalize for operator use.

#### Sampling strategy changes in patch_approach.py
- Reworked PatchAwareSampling from a more rigid fill behavior to a score-guided stochastic constructor.
- Added blended base weights combining:
	- score_part (patch merit),
	- size_part (eligible pixel count per patch),
	- random_share and score_temperature controls.
- Added stochastic stopping near target center and feasible-candidate filtering under target_max to reduce deterministic collapse.
- Added min-patch-based max_steps safeguards to avoid long loops in difficult feasibility cases.

#### Repair strategy changes in patch_approach.py
- Reworked PatchRepair._enforce_pixel_count to stochastic top-k add/remove pools rather than deterministic single-path edits.
- Add path (under target): choose from top-k promising inactive patches with softmax-based randomization.
- Remove path (over target): choose from top-k least-desirable active patches with softmax-based randomization.
- Kept hard tolerance enforcement while increasing diversity retention and reducing repeated identical repairs.

#### Optimization flow wiring in resto_anom.py
- Wired patch-score construction and injection into patch operators so sampling and repair can use the same patch-level merit signal.
- Exposed/tuned operator knobs passed from scenario/run settings, including score temperature, random share, and top-k behavior.
- Updated patch-mode setup so these controls are applied consistently during initialization and optimization execution.

#### Patch mapping and runtime performance
- Reduced patch mapping initialization overhead by reusing precomputed global-to-eligible lookup dictionaries.
- Reduced repair/sampling runtime costs via vectorized bookkeeping and bounded iteration counts.
- Tuned hypervolume warm-up behavior (sample-count controls and progress visibility) to shorten startup overhead in patch runs.

#### Secondary work (supporting)
- Updated visualization pipeline to call shared visualisation functions directly.
- Added compatibility handling for older/newer numpy pickle module paths.
- Added patch-space-to-pixel-space conversion in visualization functions to prevent decision-shape mismatches.

## Notes for Future Updates
- Add one dated section per work session.
- Keep entries short: what changed, why, and any downstream impact.
- If a change affects outputs, note which output folder or file pattern to inspect.

### 2026-03-20
- Implemented objective normalization in `resto_anom.py` to improve optimization stability, controlled by a `normalize_objectives` parameter.
- Refactored the `_evaluate` method and objective calculation logic in `resto_anom.py` for clarity and modularity.
- Streamlined the project by removing several unused scripts and output directories.

### 2026-03-27

Codebase simplification — reduced clutter and consolidated duplicated logic.

#### New: utils.py
- Created `utils.py` with a single `pickle_load()` function that consolidates the numpy-compatibility unpickler pattern previously copy-pasted into `visualisations.py`, `analyze_spatial_agreement.py`, and `seeds.py`.
- All three files now import `pickle_load` from `utils` instead of defining their own.

#### Archived scripts → old_debugging/
- Moved `debug_simple.py`, `debug_spatial_agreement.py`, `simple_inspect.py`, `check_shapes.py` to `old_debugging/` — one-off debugging scripts no longer needed in the active workspace.
- Moved `check_patch_sizes.py` to `old_debugging/` after consolidating its visualization logic into `visualisations.py` (see below).

#### spatial_operations.py — dead code removal
- Removed `merge_overlapping_regions()` and `approximate_region_landscape_change()` (both marked as unused) from `spatial_operations.py`.
- Both functions are preserved in `old_debugging/spatial_operations_legacy.py` in case they are needed later.
- Also removed the block of commented-out example raster-save code above them.

#### visualisations.py — patch-size comparison grids consolidated
- Added a `PATCH SIZE COMPARISON GRIDS` section with: `create_patch_size_comparison_grids()` (top-level entry point), `create_selection_frequency_grid()`, `create_pareto_front_grid()`, `create_parallel_coordinates_grid()`, and supporting helpers.
- These replace the equivalent functions that were previously duplicated in `check_patch_sizes.py`.

#### MDMO_Aug25.py — legacy marker
- Added a clearly visible `LEGACY REFERENCE — NOT USED IN PRODUCTION` header to `MDMO_Aug25.py`.
