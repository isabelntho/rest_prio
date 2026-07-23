# Uncertainty register

Uncertainty is inherent to this model and is not fully resolvable at the current stage. This
register records what is uncertain, how it is currently handled, and the residual concern.
Claims about model output must remain proportional to this - in particular, the model's
predictive confidence about real restoration outcomes is **low**, because the effect model is
assumed rather than calibrated (see `assumptions.md`).

## 1. Parameter uncertainty

| Source | Description | How handled | Residual concern |
|---|---|---|---|
| Effect magnitudes | `abiotic_effect`, `biotic_effect` = 0.01 are placeholders | Held fixed; some values swept informally | Absolute improvement magnitudes are not meaningful in real units |
| Weight shape | `sigma` (1.0 in effect model / 0.5 in repair scoring), `gamma = 3` control which pixels benefit | Chosen from score-distribution diagnostics (2026-04-08), not data; the two code paths use different sigma (see `assumptions.md`) | Changes the spatial priority pattern; not externally justified |
| Spillover | `neighbour_radius = 3`, decay 0.2 | Fixed | Untested sensitivity of spatial coherence to these values |
| Budget | `max_restoration_fraction` | Swept: 0.05 / 0.10 / 0.20 in policy_grid | Optimal portfolios shift with budget; no single "correct" level |

## 2. Structural uncertainty

| Source | Description | How handled | Residual concern |
|---|---|---|---|
| Effect-model functional form | Saturating degradation-weighted response is one of many plausible forms | Single form used | A different form could reorder priorities; not compared |
| Neighbour weighting | Anomaly-weighted vs flat spillover | Flat spillover tried and reverted (collapsed objective independence, 2026-04-08); code uses anomaly-weighted | Choice is made for solver behaviour, not ecological evidence |
| landscape_context construction | Structural (sn_dens) vs condition (neighbour anomaly) vs hybrid | Redefined 2026-06-24 to a hybrid (0.75 structural + 0.25 condition) after a pure-condition version was near-collinear with restoration_potential and collapsed the front | Structural/condition weight (0.25) is a tuning choice, not ecologically calibrated |
| spatial_clustering metric | `adjacency` (shared edges) vs `components` (cluster count) vs `inter_patch_adjacency` (inter-patch shared edges only) | All available via `clustering_metric`; adjacency correlates with cost, components and inter_patch_adjacency test for decoupling | Metric choice changes the compactness-vs-cost trade-off |
| Objective aggregation | Additive, independent objectives | Structural choice | Ignores interactions/diminishing returns across selected pixels |
| Decision granularity | pixel vs 2x2 patch vs planning unit | Multiple modes available and compared (patch-size experiments) | Granularity affects achievable trade-offs and budget discretisation |
| Solver choice | NSGA-III vs alternatives | Switched from NSGA-II after rank-collapse diagnosis | Reference-direction/pop-size settings not exhaustively tuned |

## 3. Scenario uncertainty

| Source | Description | How handled | Residual concern |
|---|---|---|---|
| Condition benchmark | Anomalies relative to whole area (`global`) vs best-condition reference (`upper_q75`) | Both produced as scenario rasters; run-selectable | Priority areas differ by benchmark; neither is definitively "correct" |
| Indicator set (LOO) | Leave-one-out removal of each abiotic/biotic indicator | 12 LOO variants generated in `data/anomaly_scenarios/` | Sensitivity of results to individual indicators not yet summarised |
| Policy variants | Budget levels, burden-sharing on/off | policy_grid sweeps combinations | Trade-offs are policy-conditional |
| Directionality corrections | SBD/TSD directionality was corrected (2026-04-30) | Corrected in `ec_anomalies.r` | Results before the fix are not comparable |

## 4. Stochastic / algorithmic uncertainty

| Source | Description | How handled | Residual concern |
|---|---|---|---|
| Seed variance | Evolutionary search is stochastic | Multi-seed runs (`../outputs/multi_seed_results/`); reproducible per fixed seed | Only seed-controlled reproducibility, not bitwise determinism |
| Convergence | Run may stop before the true front | Hypervolume history + HV-based early stopping | No guarantee of global Pareto optimality |
| Equifinality | Many near-equivalent portfolios achieve similar objective values | Selection-frequency and spatial-agreement analysis across solutions/seeds | A single "recommended" map is not well-identified; report ensembles/frequencies instead |

## 5. Data uncertainty

| Source | Description | How handled | Residual concern |
|---|---|---|---|
| Input raster quality | Condition anomalies and cost derived upstream | Documented provenance in `provenance_manifest.json` | Upstream errors propagate (e.g. the earlier zero-cost grassland bug) |
| Cost raster | Corrected version now default | `implementation_cost_corrected.tif` | Cost model itself is approximate |
| CRS / alignment | All inputs EPSG:2056, co-registered | Enforced at load | Misaligned or bad-CRS inputs would silently distort objectives |

## Summary statement for downstream use

Treat model output as a **map of trade-offs and relative priorities under stated assumptions**,
not as a prediction of restoration outcomes or a single optimal plan. Where a decision depends
on the result, check its stability across seeds, benchmarks, and budget levels first.
