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
    RNS[run_scenarios.py]
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
    SC --> RNS
    RA --> RS
    RA --> RNS
    RNS --> RS
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

### Internal Pipeline Detail

For detailed diagrams of how data flows *inside* `resto_anom.py` — including annotated array shapes, the patch vs. pixel operator branch, `restoration_effect()` maths, and the NSGA-II generation cycle — see **[PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md)**.

> **Maintainer note:** if you change `restoration_effect()`, the objective functions, operator classes (`AdaptiveSampling`, `AdaptiveRepair`, `PatchAwareSampling`, `PatchRepair`), or the run-settings passed to `run_optimization_instance()`, update `PIPELINE_DIAGRAM.md` to match.

## Scripts and Contributions

### Core Optimization Pipeline
- **resto_anom.py**: Optimization engine. Defines `RestorationProblem` / `PatchRestorationProblem`, all effect calculation helpers, operator builders (`_build_operators`, `_build_algorithm`), results packaging (`_package_results`), `HVCallback`, `ProgressCallback`, and the primary single-run entry point `run_optimization_instance`. Multi-scenario orchestration has been moved to `run_scenarios.py`.
- **run_scenarios.py**: Scenario orchestration layer. Handles looping over scenario combinations, assembling and saving combined multi-scenario results. Imports `run_optimization_instance` from `resto_anom.py`; `resto_anom.py` does not depend on this module.

#### run_scenarios.py
Scenario orchestration layer, extracted from `resto_anom.py`.
- `run_one`: runs a single optimization instance from a `run_settings` dict.
- `run_scenario_batch`: loops over a list of scenario parameter dicts, collecting results.
- `run_all_scenarios_optimization`: expands the full scenario space and runs a complete batch.
- `build_combined_results`: assembles the combined results dict from a completed batch.
- `finalise_combined_results`: saves combined results and parameter summary to disk.

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
- **run_patch_size_experiments.py**: Batch runner for patch-size sensitivity experiments across seeds/ecosystems. Imports `run_optimization_instance` from `resto_anom`.
- **run_multi_seed_optimization.py**: Batch runner for seed robustness analysis for one ecosystem setup. Imports `run_optimization_instance` from `resto_anom`.

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

### 2026-04-08

Diagnosis and targeted fixes for three optimization pipeline issues: all solutions appearing Pareto-optimal, salt-and-pepper spatial patterns in selected pixels, and mismatch between high-priority areas identified by weighted-sum analysis and those selected by the optimizer.

#### spatial_operations.py

- `AdaptiveRepair._do()`: reduced repair tolerance from 20% → 10% of target pixel count (minimum floor 100 → 5 pixels).
  - Previous behaviour: solutions within ±20% of budget were passed through unrepaired, meaning the effective budget varied widely across the population. A solution with 80% of budget pixels always dominates one with 120% on cost, making every member of the population trivially non-dominated regardless of ecological trade-offs. Tighter enforcement ensures all evaluated solutions operate at comparable budgets, so the Pareto front reflects genuine objective trade-offs.

#### resto_anom.py

- `restoration_effect()`: removed `anomaly_improvement_weight` from the neighbor cell calculation; replaced with a flat spillover improvement (`original_values[neighbor_mask] + neighbor_improvement`).
  - Previous behaviour: neighbor improvement was multiplied by `anomaly_improvement_weight` of each neighbor's baseline anomaly. Because neighbors of restored cells are often near-zero or positive anomaly, the weight approached 0 and the actual neighbor benefit was negligible. With no spatial continuity incentive in the objective function and `spatial_clustering=0`, the optimizer treated adjacent and distant pixels identically, producing salt-and-pepper solutions. Flat spillover gives the optimizer a genuine reason to favour spatially contiguous selections.
- `build_repair_scores()`: changed default `anomaly_weight_scale` fallback from `1.0` → `0.5`.
  - Previous behaviour: with `scale=1.0` and `gamma=3`, mildly degraded pixels (anomaly ≈ −0.1) received scores of ≈ 9 × 10⁻⁵, near-zero and essentially indistinguishable. Score-based repair and sampling had almost no spatial signal to guide selection across most of the landscape. Halving the scale spreads the score distribution over a broader range of degradation values without flattening it, giving the repair operator meaningful guidance for mildly degraded pixels. This default can be overridden via `scenario_params["anomaly_weight_scale"]`.

> **Reverted (same session):** The flat spillover change to `restoration_effect()` was reverted — see below.

#### Switch from NSGA-II to NSGA-III — resto_anom.py

Root cause identified: with 3 objectives and a population of 50, NSGA-II's fast non-dominated sort places virtually all solutions on rank 0 (expected behaviour for many-objective problems). This collapses selection pressure entirely to crowding distance, making the algorithm unable to distinguish genuinely better solutions from random ones — explaining both the salt-and-pepper spatial patterns and the mismatch with weighted-sum priority areas.

- Replaced `NSGA2` import with `NSGA3` from `pymoo.algorithms.moo.nsga3`; added `get_reference_directions` from `pymoo.util.ref_dirs`.
- `_build_algorithm()`: now accepts `n_partitions` parameter; generates Das-Dennis reference directions (`get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)`) and passes them to `NSGA3`. All other constructor arguments (`sampling`, `crossover=HUX()`, `mutation`, `repair`) unchanged. Population size is no longer set explicitly — NSGA-III determines it from the number of reference directions.
- `run_optimization_instance()`: added `n_partitions=8` parameter (default). With 3 objectives and `n_partitions=8`, 45 reference directions are generated → effective population of 45, close to the previous `POP_SIZE=50`. The `pop_size` parameter is retained for API compatibility but is ignored by NSGA-III. Docstring updated accordingly.

> **Note:** `PIPELINE_DIAGRAM.md` Diagram 3 references the NSGA-II generational loop — update to reflect NSGA-III and reference-direction-based selection.

#### Revert: restore anomaly-weighted neighbour logic in resto_anom.py

- `restoration_effect()`: reverted the flat spillover change from earlier this session; restored the original anomaly-weighted neighbour improvement (`neighbor_weights = anomaly_improvement_weight(neighbor_baseline_anomalies, ...)`; `updated_values[neighbor_mask] = original_values[neighbor_mask] + neighbor_improvement * neighbor_weights`).
  - Reason: flat spillover caused both objectives (abiotic and biotic) to move together on every restored pixel, because `updated_values` feeds both objectives identically. This collapsed objective independence and produced highly correlated solutions. The original anomaly-weighted logic differentiates neighbours by their own anomaly type and magnitude, preserving the trade-off structure that NSGA-III needs to spread solutions across reference directions. Now that NSGA-III is in place, selection pressure is restored and the weighted neighbour logic is no longer the bottleneck for spatial coherence.

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

#### resto_anom.py — internal refactoring
- Lifted `ProgressCallback` from a closure inside the optimization function to a module-level class, with explicit constructor parameters (`n_generations`, `hv_patience`, `hv_min_improvement`, `ref_point`) replacing closed-over variables.
- Extracted four private helpers from `run_single_scenario_optimization`: `_build_operators`, `_build_algorithm`, `_package_results`, `_print_failure_diagnostics`.
- Renamed `run_single_scenario_optimization` → `run_optimization_instance` to better reflect its purpose.
- Removed multi-scenario orchestration functions (`run_one`, `run_scenario_batch`, `run_all_scenarios_optimization`, `build_combined_results`, `finalise_combined_results`) — all moved to the new `run_scenarios.py`.
- Trimmed top-level imports to only what `resto_anom.py` itself uses.

#### New: run_scenarios.py
- Created to hold all scenario orchestration logic extracted from `resto_anom.py`.
- Contains: `run_one`, `run_scenario_batch`, `run_all_scenarios_optimization`, `build_combined_results`, `finalise_combined_results`.
- Imports `run_optimization_instance` and `_filter_initial_conditions_for_return` from `resto_anom`; `resto_anom` does not import from this module (circular-import-free design).

#### data_loader.py — ecosystem type fix
- Added `'fg'` as a valid `ecosystem_type` value in `create_ecosystem_mask`, combining forest (codes 12, 13) and grassland (codes 16, 17) into a single mask. Error message updated to list `'fg'` as a valid option.

#### Callers updated
- `run_multi_seed_optimization.py` and `run_patch_size_experiments.py` updated to import and call `run_optimization_instance` instead of the old `run_single_scenario_optimization`.

#### Further simplification of resto_anom.py
- Removed unused imports: `email.mime.base`, `pandas`, `rasterio`, `BinaryRandomSampling`, `NonDominatedSorting`, `apply_burden_sharing`, `apply_spatial_clustering`, `_enforce_exact_pixel_count`, `compute_sn_dens`, `perf_counter`, `psutil`, and several `data_loader` symbols.
- Removed dead instance variables from `RestorationProblem.__init__`: `_t0_wall`, `_eval_n`, `_slow_eval_seconds`, `_proc`.
- Removed dead timing/debug instrumentation from `_evaluate` (`_debug_count`, `t_eval0`, `mem0`, `t_split`, second `t0`).
- Removed `_mem_gb()` method (only used by dead instrumentation).
- Simplified 3-way `Pool()` if/elif/else to a single line.
- De-duplicated the two `super().__init__()` calls in `RestorationProblem.__init__` into a kwargs-dict pattern.
- Removed dead locals `abiotic_effect`, `biotic_effect`, `eligible_indices` from `restoration_effect`.
- Removed commented-out code block (`#weighted_improvements`, etc.) in `restoration_effect`.
- Removed redundant `import numpy as np` inside `recalculate_landscape_anomaly_with_conversions` and `build_fixed_ref_point`; removed unused `rng = np.random.default_rng(seed)` in `build_fixed_ref_point`.
- Moved `diagnose_optimization_setup` to new `debug_utils.py`; call site in `run_optimization_instance` uses a deferred import to avoid circular dependency.
- Extracted `if __name__ == "__main__":` block to new `run_custom.py` for standalone use.
- Replaced all 3-line `# ===...===` section banners with single-line `# --- Name ---` style.

#### New: debug_utils.py
- Contains `diagnose_optimization_setup` (moved from `resto_anom.py`), with the dead `if i < 3:` inner loop removed.
- Uses a deferred `from resto_anom import RestorationProblem` inside the function body to avoid circular imports.

#### New: run_custom.py
- Standalone script for interactive/custom single-scenario runs; replaces the `if __name__ == "__main__":` block that was in `resto_anom.py`.
- Edit configuration constants at the top and run directly: `python run_custom.py`.

#### Potential future simplification (not implemented)
- `conversion_mask()` in `resto_anom.py` is called from exactly one place (`restoration_effect`) with the `conversion_mask_2d` array already constructed at that call site. It could be inlined to eliminate the indirection, but was left as a separate function to preserve readability.

### 2026-03-30

Added per-generation population snapshot capture to support visualisation of how spatial decisions evolve across the NSGA-II run.

#### Design decisions
- Captures the full population decision matrix `X` (shape `(pop_size, n_var)`, stored as `int8`) at every generation.
- Streams snapshots to temporary batch `.npz` files every 10 generations to bound peak RAM usage (≤ `10 × pop_size × n_var × 1 byte` in memory at any time, ~172 MB for the current landscape at `n_var ≈ 362,600`).
- At the end of optimisation, all batch files are concatenated into a single compressed `intermediate_results/X_history_{timestamp}.npz` (shape `(n_gens, pop_size, n_var) int8`) and the temporary batch files deleted.
- Feature is opt-in: `save_snapshots=False` by default so existing callers are unaffected.
- Estimated total on-disk size for 100 generations: ~115 MB uncompressed → ~10–15 MB compressed.

#### Changes — resto_anom.py (only file modified)
- `HVCallback.__init__`: added `snapshot_dir`, `X_batch`, `batch_files`, `batch_size`, `_n_generations_est`, `_x_memory_warned` fields.
- `HVCallback.__call__`: after F-stats block, appends `X.astype(np.int8)` to buffer each generation; calls `_flush_batch()` when buffer reaches `batch_size`; prints one-time RAM/disk estimate at generation 1.
- `HVCallback._flush_batch`: new method — stacks buffer, saves `X_batch_NNNN.npz` under `snapshot_dir`, clears buffer.
- `ProgressCallback.__init__`: added `save_snapshots=False` and `snapshot_dir=None` parameters; when `save_snapshots=True`, creates the batch directory and activates snapshotting on `hv_callback`.
- `_package_results`: added `save_snapshots=False` parameter; when True, flushes remaining buffer, concatenates batch files, saves `intermediate_results/X_history_{timestamp}.npz`, removes batch files, and adds `'X_history_path'` to the results dict.
- `run_optimization_instance`: added `save_snapshots=False` parameter; creates the temporary batch directory path and wires `save_snapshots` through to `ProgressCallback` and `_package_results`.

#### Usage
```python
results = run_optimization_instance(
    initial_conditions=initial_conditions,
    scenario_params=scenario_params,
    save_snapshots=True,
    output_dir="./results_files",
)
# Load afterwards
import numpy as np
X = np.load(results['X_history_path'])['X']  # (n_gens, pop_size, n_var) int8
# Selection frequency map at generation g:
freq = X[g].sum(axis=0)  # how many solutions selected each variable
```