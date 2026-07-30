# Formulation experiments - session summary

Date: 2026-07-24. Scope: why the optimiser barely evolved, how different
formulation choices changed the results, and what changed in the code since the
last git commit (4ef0654).

Plain-language, facts only. Where a result depends on the formulation, that is
stated as a fact about the formulation - not as one setup being "correct".

---

## 1. What was held fixed across all runs

- Region: Bern. Ecosystem: all / combined.
- Eligible restoration pixels: 437,110. Budget: 5% = 21,855 pixels.
- Three objectives, all minimised, values normalised:
  restoration_potential (sum of per-pixel potential), spatial_clustering,
  implementation_cost.
- restoration_potential: lower (more negative) = more degraded land = higher potential.

## 2. The starting problem

### Why the metric was switched to inter_patch_adjacency

In the patch approach the plain adjacency metric (a shared-edge count) was strongly
correlated with cost. Each 2x2 patch already has 4 guaranteed internal shared edges,
so adjacency mostly counted the number of selected patches - which is the same thing
as the amount of land bought, i.e. cost. That made spatial_clustering little more
than a proxy for cost, so it was not really acting as an independent objective.

inter_patch_adjacency was introduced to try to decouple the two. It counts only
shared edges that cross a patch boundary and excludes the guaranteed within-patch
edges, so it measures how much separate patches touch each other rather than trivial
within-patch compactness. The aim was a clustering objective independent of cost.

Later finding (see 4E): the cost-clustering correlation is mostly a landscape
property - cheap land is spatially clumped (Moran's I of cost = 0.85) - not a metric
artifact. On top of that, at a fixed budget with all-or-nothing 2x2 patches the
number of selected patches is roughly fixed, so the guaranteed internal edges are
roughly constant and inter_patch_adjacency is just plain adjacency minus a near-
constant offset (nearly the same quantity, so nearly the same correlation with cost).
So no adjacency variant can decouple clustering from cost, which is why the switch
did not achieve its aim - the switched run still produced the cost-clustering
correlation (the L-shaped front below). The switch was reasonable given what was
known at the time; the later measurement shows it was aimed at the wrong cause.

Measured on the actual inter_patch_adjacency patch run (91 solutions,
Debugs_tests/adjacency_metric_equivalence.py): plain adjacency and inter_patch_adjacency
correlate at r = +0.99 (effectively the same axis); their correlations with cost are
near-identical (-0.94 and -0.94); and the internal-edge term the switch removed is
near-constant (CV 3.3%, tracking a near-constant patch count CV 2.8%). So the metric
switch moved along the same axis and left the cost-clustering correlation intact.
(Sign note: clustering and cost correlate negatively here - more-clustered plans are
cheaper - consistent with 4E: cheap land is spatially clumped.)

### What that produced (the first doubt)

When the metric was switched to inter_patch_adjacency, the Pareto front came out
looking wrong. Almost all solutions bunched at low restoration_potential (roughly 0
to 0.25 normalised) while spanning the full spatial_clustering range top to bottom;
only a few isolated points reached high restoration_potential. Implementation cost
(the colour) rose with clustering. So the front barely traded restoration_potential
at all - it mostly moved along clustering and cost at near-zero potential, an
L-shaped front that did not look like a real three-way trade-off. And despite the
intent, cost and clustering still moved together (see entry 3B).

That is what triggered the wider check. On top of it, runs showed almost no change
in hypervolume (HV) or objective values across generations. Goal for the session:
find the cause, and find a setup that explores spatially varied plans instead of
collapsing onto one area.

## 3. Formulation choices explored, and what each one produced

Each entry is: the choice -> the observed fact.

### A. Pixel genotype + scattered sampling (bitflip mutation)
- Plans come out spatially scattered.
- HV nearly flat; objective means barely move across generations.
- Cost coefficient of variation is near-flat. Measured cause: a large-N averaging
  effect that matches a random-selection null (see Debugs_tests/cost_cv_null.py),
  not active suppression.
- Separate bug found and fixed: the integer bitflip mutation used `~X`, which
  corrupted the genotype (0 -> -1, 1 -> -2). Fixed in InstrumentedBitflipMutation.

### B. Patch genotype (2x2 patches) + clustering_metric = inter_patch_adjacency
- Observed correlation across solutions: implementation_cost and spatial_clustering
  move together (highly correlated).

### C. Pixel genotype + region operators + clustering_metric = adjacency
Operators (new this session): seed contiguous regions spread across the map, then
relocate / spawn / grow / shrink them, and recombine whole regions. Every plan is
forced to be contiguous by construction.
- HV now climbs across generations, then plateaus (around generation ~120). The
  search does real work here, unlike setup A.
- Observed correlation across solutions: restoration_potential and spatial_clustering
  move together; implementation_cost looks independent.
- Late generations: spatial_clustering keeps improving while the restoration_potential
  mean drifts back the wrong way (worsens). The two are being traded against each other.
- RFOP map (frequency each pixel appears in the Pareto front): occurrence concentrates
  in a few basins (north-central and west); large parts of the canton are never selected.
- 250-generation run: 40 non-dominated solutions. Pareto plots show one dominant
  diagonal (potential and clustering together). Plan pairs that are close in objective
  space are only moderately similar in space (Pearson r = -0.63): objective diversity
  does not guarantee map diversity.

### D. Crossover on/off (operator diagnostic, setup C)
- Removing crossover made the run hang: pymoo's duplicate-elimination retry loop
  re-ran the expensive region mutation ~100x per generation. Setting
  eliminate_duplicates=False let it run.

### E. Direct measurement of the raw input layers (no operators, no representation)
Measured on the cost and restoration_potential layers the optimiser reads
(Debugs_tests/clustering_redundancy.py, 5% budget):
- Spatial autocorrelation (Moran's I, rook): implementation_cost = 0.85 (strongly
  clumped); restoration_potential = 0.53 (moderately clumped).
- Cheapest 5% of pixels: clustered (51.7% of max adjacency, 6,225 components),
  low potential, cheap.
- Most-degraded 5% of pixels: scattered (19.5% of max adjacency, 13,829 components),
  ~5x more expensive, high potential.
- Cherry-pick gap (how much objective value is lost when a plan is forced contiguous):
  restoration_potential loses 54%; implementation_cost loses 24%.

## 4. What the facts say about formulation-dependence

- The correlation between objectives changes with the formulation:
  - patch + inter_patch_adjacency: cost and clustering correlated.
  - pixel + region operators + adjacency: potential and clustering correlated.
- On the raw layers (formulation-free): cheap land is clumped, degraded (high-potential)
  land is scattered and more expensive.
- The region operators force every plan to be contiguous. This pins clustering high and
  can produce correlations that are not present in the raw layers.
- Across every setup, spatial_clustering is not an independent third axis: it overlaps
  with cost (raw data / patch runs) or with potential (region-operator runs). The
  effective problem is close to 2-D. This is consistent with HV plateauing quickly and
  the RFOP map concentrating in a few basins.
- The irreducible tension in the raw data is restoration_potential vs cost: the most
  valuable land is scattered and expensive; the cheap land is clumped and low-value.

### Does this invalidate the patch approach?

No - and that is the point. The L-shaped front was read as the patch approach
producing something broken; it is not. It is the honest shape of a problem where one
objective (spatial_clustering) is near-redundant with cost. Three things were tangled
together and should be kept apart:

- The patch REPRESENTATION (2x2 decision units) was never the fault. It stays a
  legitimate choice if a minimum restoration grain and lower dimensionality
  (~157k patch vars vs ~437k pixels) are wanted. Its real limitations are unrelated:
  a fixed 2x2 grain (no arbitrary region shapes/sizes) and partial boundary patches.
- The clustering OBJECTIVE is the actual issue, and it is representation-independent:
  it overlaps with cost (patch / raw data) or potential (region operators) in every
  setup, collapsing the problem toward 2-D.
- The L-front is not an artifact - it is what a near-redundant-objective front looks
  like. The patch run, by letting clustering vary freely, exposed the true landscape
  correlation; the region operators instead impose contiguity and hide it.

So the decision sits ABOVE patch-vs-pixel: fix the objective set first (drop
spatial_clustering as an objective; run restoration_potential vs cost, with contiguity
as a constraint if wanted). Only then does patch-vs-pixel reduce to a clean modelling
choice about grain and shape flexibility.

## 5. Outputs produced this session

Result files (outputs/results_files/):
- res_20260724_1130_region_grow_test_250.pkl - 250-gen region run, 40 non-dominated
  (source of the latest Pareto / RFOP figures). Note: RUN_LABEL says "region_grow" but
  the config used sampling_strategy = region_evolve.
- res_20260723_1748_region_evolve_spread.pkl - 30-gen region_evolve spread test.
- Matching R-export folders under outputs/r_inputs/ (same timestamps).

Figures generated (from the 250-gen run):
- HV and per-objective evolution over generations.
- Normalised Pareto scatter (potential vs clustering, coloured by cost).
- Raw pairwise objective scatter (non-dominated vs dominated).
- RFOP / selection-frequency map.
- Spatial similarity (Jaccard) vs objective-space distance.

Diagnostic / analysis scripts (Debugs_tests/):
- clustering_redundancy.py - Moran's I, cherry-pick gap, per-set cross-objective table
  (section 3E above).
- region_evolve_spread_test.py - runs region_evolve, spread metrics, RFOP map.
- instrument_operators.py - logs crossover behaviour.
- region_front_check.py - non-dominated front structure and correlations.
- operator_sweep.py - operator comparison.

## 6. Code changes since last commit (4ef0654)

### Optimisation code (this session's core work)

Core_optimisation/spatial_operations.py (+654 lines). New region operators and helpers:
- build_restoration_neighbor_table - precompute 4-neighbour table over eligible pixels.
- grow_region_plan, grow_regions_from_seeds - frontier heap-walk region growth.
- _label_components, enforce_budget_contiguous - contiguity-preserving budget enforcement.
- RegionGrowingSampling, RegionGrowingMutation - build/preserve contiguous regions.
- SpatialCoverageSampling - seed regions spread across a coarse grid.
- RegionEvolveMutation - relocate / spawn / delete / grow / shrink whole regions.
- RegionSwapCrossover - recombine whole parent regions (replaces HUX for this mode).

Core_optimisation/resto_anom.py (+129 lines). Wiring:
- Import the new operators.
- _build_operators: new `sampling_strategy` branch ('region_grow' / 'region_evolve');
  scored growth blends standardised score minus standardised per-pixel cost; now returns
  a 4-tuple (sampling, repair, mutation, crossover).
- _build_algorithm: accepts optional `mutation` and `crossover`; defaults stay
  InstrumentedBitflipMutation and HUX when not supplied.
- run_optimization_instance: threads mutation and crossover through.

Core_optimisation/run_custom.py (+23 lines). Config for the region runs:
- RUN_LABEL -> "region_grow_test_250".
- N_GENERATIONS 100 -> 250.
- WARM_SEEDING True -> False.
- USE_PATCH_APPROACH True -> False.
- Added sampling_strategy = "region_evolve" plus knobs: region_seeds 25,
  region_seeds_min 5, region_seed_grid 16, region_growth_bias "scored",
  region_mutation_edits 100.

export_to_r.py (+7 lines):
- run_config and initial_conditions now use `or {}` so a run saved without a run_config
  (e.g. a test script) does not crash the metadata export.

### Incidental changes (not part of the optimisation work)

data/ec_anomalies.r (+53 lines): adds a region toggle (KB vs CH). For region == "CH"
it swaps in the whole-Switzerland EC stack and mask. R data-prep, separate from the
optimiser. (Also removed two non-ASCII check-mark characters from cat() output.)

.gitignore (+26 lines): ignores several Debugs_tests scratch scripts (gen0_diversity.py,
repair_diversity_report.py, cost_cv_null.py, mutation_rate_sweep.py, check_crossover_off.py)
and the Quarto optimisation_approaches_overview.md/html and its libs folder.

### New untracked files
Debugs_tests/: clustering_redundancy.py, instrument_operators.py, operator_sweep.py,
region_evolve_spread_test.py, region_front_check.py.
