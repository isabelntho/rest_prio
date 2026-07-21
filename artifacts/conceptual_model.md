# Conceptual model

This is a thin summary. The authoritative problem specification is `../README.md` (sections 1-7)
and the internal data flow is documented in `../PIPELINE_DIAGRAM.md` (three diagrams: module
pipeline, array transformations inside `_evaluate()`/`restoration_effect()`, and the NSGA-III
generational loop).

## Purpose and scope

Identify where, within an eligible landscape, to invest a limited restoration budget so as to
best trade off ecological improvement against cost. Scope is Kanton Bern, forest + grassland
ecosystems, at raster resolution in EPSG:2056. The model is for trade-off exploration and
methods research, not operational site selection (see `model_card.md`).

## Entities and state

- Landscape: co-registered rasters of condition anomalies, cost, and eligibility.
- Decision unit: a pixel, a 2x2 patch, or a planning unit (grid block / admin polygon),
  depending on run mode. Only eligible units enter the decision vector.
- Solution: a binary vector selecting units for restoration (and, where enabled, conversion).

## Decision variables

Binary `x[i]`: 1 = unit `i` selected for restoration, 0 = not. Vector length = number of
eligible units in the active mode. Patch/unit decisions are expanded to pixel decisions before
objective evaluation, so objectives are always evaluated at pixel level (see PIPELINE_DIAGRAM
Diagram 2).

## Constraint

A budget targets a fraction of eligible pixels (default `max_restoration_fraction = 0.05`).
A repair operator enforces the budget after every variation step, within a tolerance window
(patch mode: +/-5%, evaluated at +/-7.5%; see `assumptions.md` for the current tolerance and
its status).

## Objectives

A selectable subset of objectives, all minimised (see `model_card.md` for the full current
list of up to ten, with production vs experimental/legacy status). A run may use the
abiotic/biotic split or the aggregate restoration_potential/landscape_context logics,
optionally with cost, connectivity, and spatial_clustering.

## Restoration-effect model (summary)

When a pixel is restored, its condition anomaly improves by an amount that depends on how
degraded it already is (a saturating, degradation-weighted response), with a reduced-magnitude
spillover to neighbours within a fixed radius. Cost has no spillover. The exact functional form
and parameter values, and their status as **assumed rather than calibrated**, are in
`assumptions.md`; the maths is in `../README.md` section 5 and `../PIPELINE_DIAGRAM.md` Diagram 2.

## Algorithm (summary)

NSGA-III with Das-Dennis reference directions, HUX crossover, bit-flip mutation, and a
score-guided repair operator enforcing the budget. See `../README.md` section 6 and
`../PIPELINE_DIAGRAM.md` Diagram 3 for parameters and the generational loop.
