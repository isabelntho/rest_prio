# Model card - Restoration prioritisation optimisation

## Overview

| Field | Value |
|---|---|
| Model name | Restoration prioritisation multi-objective optimiser (rest_prio) |
| Model type | Many-objective evolutionary optimisation (NSGA-III, pymoo >= 0.6.1.5) |
| Decision problem | Spatial selection of pixels/patches/planning units for ecological restoration |
| Study area | Kanton Bern, Switzerland (EPSG:2056, Swiss LV95) |
| Target ecosystem | Forest + grassland combined ("fg"; LULC classes 12, 13, 16, 17) |
| Status | Methods development - not validated against observed restoration outcomes |
| Version | Tracked by git; see `provenance_manifest.json` for the current commit |

For the full problem specification see `../README.md`. For internal data flow see
`../PIPELINE_DIAGRAM.md`.

## What the model does

Given co-registered raster inputs (condition anomalies, implementation cost, eligibility), the
model searches for restoration portfolios that trade off multiple objectives under a budget
constraint. The output is a set of non-dominated (Pareto) solutions, each a binary selection
over eligible units, plus the full final population and hypervolume history.

## Inputs

Per-pixel rasters, co-registered to a common extent/CRS/resolution (EPSG:2056):
- abiotic condition anomaly, biotic condition anomaly (signed floats; negative = degraded)
- implementation cost (per-pixel restoration cost)
- eligible-pixel mask (target LULC classes intersected with valid objective values)

Full input inventory is in `provenance_manifest.json`.

## Objectives (selectable subset per run)

All objectives are minimised (improvements are sign-flipped). A run selects a subset from the
set below. The code (`../Core_optimisation/resto_anom.py`, `RestorationProblem.__init__`)
currently exposes up to ten objectives; not all are equally mature.

Production (used in current run matrices):
- abiotic improvement, biotic improvement (maximise improvement over restored area)
- implementation cost (minimise)
- restoration_potential (favour the most degraded pixels; mean of abiotic+biotic baseline)
- landscape_context (favour pixels whose surroundings are already supportive; hybrid metric,
  see `assumptions.md`)
- connectivity_gain (favour conversions that increase focal-habitat connectivity, ~100 m)
- spatial_clustering (favour spatially compact selections; adjacency or components metric)

Experimental / legacy (available but not part of the settled objective set):
- es_future_val, es_future_robustness (ecosystem-service performance/robustness under future
  scenarios; added 2026-05-19)
- landscape_anomaly (legacy SN-density landscape objective; superseded by connectivity_gain)

## Decision granularity (three modes)

- pixel: one binary variable per eligible pixel
- patch: 2x2 pixel patches (default) to shrink the decision space; objectives still evaluated
  at pixel level
- planning unit: grid blocks or admin polygons; selecting a unit activates all its eligible
  pixels

## Intended use

- Exploring trade-offs between restoration objectives under budget and policy scenarios.
- Comparing prioritisation logics (e.g. contextually supported vs most-degraded selection).
- Methodological research on multi-objective restoration prioritisation.

## Out of scope / not intended for

- Operational site-level restoration decisions without expert review.
- Prediction of actual ecological recovery - the restoration-effect model is an assumed
  functional form, not a calibrated ecological model (see `assumptions.md`).
- Use outside the modelled study area, ecosystem types, or condition-anomaly definitions.

## Key limitations

- Effect-model parameters are assumed, not calibrated against field data.
- No external validation exists; evaluation is verification-only (see `evaluation_report.md`).
- Solutions exhibit equifinality: many near-equivalent portfolios (see `uncertainty_register.md`).
- Results depend on the condition-anomaly benchmark and indicator set chosen for a run.
