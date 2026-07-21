# Assumptions register

Consequential assumptions built into the model. Values are the current defaults; sources are
`../README.md` (sections 3, 5, 7), the design-decision entries in
`../Documentation/DEVELOPMENT_TRACKER.md`, and the code in `../Core_optimisation/`.

Confidence key: **low** = chosen for plausibility, not evidence; **medium** = supported by
internal reasoning/diagnostics but not external data; **structural** = a modelling-frame choice
whose alternative would change results qualitatively.

## Restoration-effect model

The effect model is an **assumed functional form, not calibrated against field data**. This is
the single most important caveat for interpreting any result. See `uncertainty_register.md`.

When pixel `i` is restored, its anomaly changes by
`delta[i] = effect_magnitude * w(anomaly_before[i])`, with weight
`w(a) = (1 - exp(-|a| / sigma))^gamma` and `w = 0` where `a >= 0` (undegraded cells).

| Assumption | Current value | Rationale | Confidence |
|---|---|---|---|
| Effect magnitude (abiotic) | `abiotic_effect = 0.01` | Placeholder scale for improvement per restored pixel | low |
| Effect magnitude (biotic) | `biotic_effect = 0.01` | Same as abiotic; no basis to differentiate yet | low |
| Weight shape - saturation | `sigma` (`anomaly_weight_scale`): 1.0 in the effect model, 0.5 in repair scoring | Spreads score over a broad range of degradation without flattening. NOTE: two code paths disagree - `restoration_effect()` defaults `anomaly_weight_scale` to 1.0 (`resto_anom.py`), while `build_repair_scores()` defaults to 0.5. No run script sets the key, so the effect model uses 1.0 and the repair scores use 0.5. The 0.5-everywhere intent (2026-04-08) is not fully wired. | low |
| Weight shape - exponent | `gamma = 3` | Concentrates improvement on more-degraded pixels | low |
| Improvement is degradation-weighted | `w(a)` form above | More-degraded pixels benefit more; undegraded pixels do not improve | structural |
| Neighbour spillover radius | `neighbour_radius = 3` pixels | Restoration benefits nearby pixels | low |
| Neighbour spillover magnitude | `effect_magnitude * 0.2` (decay 0.2) | Neighbours improve at reduced magnitude | low |
| Neighbour improvement is anomaly-weighted | uses `w()` per neighbour | Preserves abiotic/biotic trade-off structure; a flat-spillover alternative was tried and reverted 2026-04-08 because it collapsed objective independence | structural |
| Cost has no spillover | cost summed over restored pixels only | Cost is incurred only where action is taken | medium |

## Objectives and their construction

| Assumption | Current value | Rationale | Confidence |
|---|---|---|---|
| Objectives are additive over selected pixels | sum over restored/selected pixels | Tractable, interpretable aggregation | structural |
| Objectives treated as independent for trade-off | separate F1/F2/F3... | NSGA-III needs distinct objectives to spread solutions | structural |
| Objective normalisation | divide each objective by sum of abs baseline over eligible pixels | Makes objectives dimensionless, ~O(1), for solver stability (added 2026-03-20) | medium |
| restoration_potential definition | mean of abiotic+biotic baseline anomaly at the pixel | Lower = more degraded on both = higher potential | medium |
| landscape_context definition | hybrid: `0.75 * sn_dens[300 m semi-natural-habitat proportion] + 0.25 * good-neighbour-fraction[500 m, 11x11 box]`, range ~[0,1] | HIGHER = more supportive surroundings (structural anchor keeps it independent of restoration_potential; redefined 2026-06-24, see DEVELOPMENT_TRACKER). Objective negates ctx so the minimiser maximises it. | medium |
| connectivity_gain definition | focal-habitat within radius (circular kernel, ~100 m) | Rewards converting pixels embedded in existing habitat | medium |
| spatial_clustering definition | selected-pixel compactness: `adjacency` (count of orthogonal shared edges, default) or `components` (count of disconnected clusters) | Rewards clumped solutions; metric chosen via `clustering_metric` | medium |
| es_future_val / es_future_robustness (experimental) | sum of per-pixel ES performance / instability over selected pixels | Future-scenario ES gain (maximise) / instability (minimise); added 2026-05-19, not in settled set | low |

## Budget and decision space

| Assumption | Current value | Rationale | Confidence |
|---|---|---|---|
| Restoration budget | `max_restoration_fraction = 0.05` (5% of eligible pixels) | Policy-scale placeholder; swept in policy_grid (0.05 baseline / 0.10 / 0.20) | low |
| Budget tolerance window | `pixel_tolerance = 0.05` (+/-5%) for patch mode; constraint-evaluation window is `pixel_tolerance * 1.5` = +/-7.5% (`resto_anom.py`). Planning-unit runs use 0.20. | Allows discretisation slack at patch/unit granularity | medium |
| Repair stochasticity | top-k shortlist + softmax (`temperature`, `top_k`) | Preserves population diversity vs greedy repair; note tighter enforcement (10%) was used 2026-04-08 to make the Pareto front reflect genuine trade-offs, later relaxed with higher temperature/top_k | medium |
| Patch aggregation | 2x2 pixels (`patch_size = 2`) | Shrinks decision space; objectives still evaluated per pixel | medium |
| Planning-unit mode | grid blocks or admin polygons | Coarser decisions while preserving pixel-level objective accuracy | medium |

## Algorithm

| Assumption | Current value | Rationale | Confidence |
|---|---|---|---|
| Solver | NSGA-III (Das-Dennis ref dirs) | With 3 objectives NSGA-II collapsed nearly all solutions to rank 0; switched 2026-04-08 | medium |
| Reference directions / pop size | `n_partitions = 12` (run scripts) -> ~91 ref dirs / pop ~91 for 3 objectives (POP_SIZE is ignored; NSGA-III sets pop = n ref dirs). Ref-dir count scales with n_obj. The `_build_algorithm` default is 8, but runs pass 12. | Explore more of the objective space than the earlier 8-partition / 45-dir setting | medium |
| Mutation rate | ~200 bit-flips per individual per generation (`prob_var = 200/n_var`) | Higher exploration; raised from ~1 flip (2026-04-16), later to 200 | medium |
| Generations | 100 (with HV-based early stopping) | Practical run length; convergence checked via hypervolume | medium |
| Parallel evaluation | ThreadPool (numpy/scipy release the GIL) | Correct for this workload; avoids Windows spawn/pickle overhead (2026-04-29) | medium |

## Data

| Assumption | Current value | Rationale | Confidence |
|---|---|---|---|
| Common raster grid | all inputs co-registered, EPSG:2056 | Required for pixel-aligned objective evaluation | medium |
| Eligible pixels | LULC classes 12,13,16,17 with valid values in all objective rasters | Defines the decision space | medium |
| Cost raster | `implementation_cost_corrected.tif` | Earlier `implementation_cost.tif` was incorrectly zero for most grassland cells (fixed 2026-04-16) | medium |
| Anomaly sign convention | negative = degraded relative to reference | Basis for improvement calculation | medium |
