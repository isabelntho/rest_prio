# Development Tracker

Purpose: keep a lightweight record of what each core script does and what changed over time.

## Quick Project Outline

### Core Optimization Pipeline
- resto_anom.py: Main optimization engine. Defines objectives and constraints, initializes patch mode, runs NSGA-II, and orchestrates saving/reporting.
- patch_approach.py: Patch-based decision framework. Builds patch mappings, converts patch decisions to pixel decisions, and provides patch-aware sampling and repair operators.
- spatial_operations.py: Spatial helper functions and custom operators for clustering, burden sharing, and landscape density calculations.
- data_loader.py: Loads rasters and preprocessing inputs; builds masks and initial_conditions for optimization.
- scenarios.py: Scenario parameter definitions and scenario expansion/sampling helpers.
- results_saving.py: Writes result files, summaries, and reports from optimization runs.

### Experiment Runners
- run_patch_size_experiments.py: Batch runner for patch-size sensitivity experiments across seeds/ecosystems.
- run_multi_seed_optimization.py: Batch runner for seed robustness analysis for one ecosystem setup.
- check_patch_sizes.py: Patch-size sweep runner plus visualization/grid generation for side-by-side comparison.

### Analysis and Visualization
- visualisations.py: Main plotting and analysis utilities (selection frequency maps, Pareto plots, parallel coordinates, etc.).
- analyze_spatial_agreement.py: Computes and compares spatial agreement metrics between runs/approaches.
- run_vis.py: Convenience script to generate visual outputs from saved results.
- Bivariate_plot.r, ec_anomalies.r, topo_aggregation.r, setup.r: R scripts for exploratory analysis and figures.

### Data, Outputs, and Environment
- environment.yml: Python environment definition.
- patch_tests/, patch_tests_1803/, patch_tests_1903/, patch_tests_1903_2/: Patch experiment outputs.
- multi_seed_results/: Multi-seed output artifacts.
- results_*.pkl, summary_*.json, optimization_report_*.json, evolution_report_*.json: Saved optimization artifacts and reports.
- figs/, notebook_outputs/, optimization_results_analysis_files/: Generated plots and report assets.

## Chat Work Log

### 2026-03-19

Improved the patch-based optimization so it explores more diverse and realistic solutions instead of repeatedly converging on very similar patch sets. Made it faster and more robust by improving patch mapping performance and ensuring patch-based results can be visualized correctly.

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
