# Evaluation report

## The central distinction: verification vs validation

- **Verification** = "are we solving the problem right?" (the code and optimiser behave as
  intended). This project has verification evidence.
- **Validation** = "are we solving the right problem?" (model output corresponds to real
  restoration outcomes). This project has **no external validation**, because there is no
  observed-outcome dataset to validate against and the restoration-effect model is an assumed
  functional form (see `assumptions.md`). The model is best described as
  **optimisation-over-assumptions**, not prediction.

This distinction must be preserved in any reporting: performance metrics below speak to solver
quality, not ecological correctness.

## Evaluation aligned with model purpose

The purpose is trade-off exploration under a budget (see `model_card.md`). Evaluation therefore
focuses on whether the optimiser produces a well-formed, diverse, budget-feasible Pareto set,
and whether results are stable enough to interpret.

## Verification evidence available

| Check | What it shows | Where |
|---|---|---|
| Budget-constraint enforcement | All evaluated solutions sit at comparable budgets, so the Pareto front reflects genuine objective trade-offs rather than budget differences | Repair operator; diagnosed 2026-04-08 (rank-collapse fix) |
| Hypervolume convergence | HV history per run; early stopping when HV stops improving | `HVCallback` / `ProgressCallback`; per-run reports |
| Seed robustness | Repeated runs across seeds; hypervolume and duplicate detection | `../outputs/multi_seed_results/` |
| Spatial agreement | Overlap of selected areas across solutions/approaches | `../spatial_agreement_solutions_fg_2503_2.csv` |
| Selection frequency | How often each pixel is chosen across the solution set (equifinality) | selection-frequency outputs |
| Parallelism / performance | Parallelism-efficiency report; profiled bottleneck fixes | run reports; DEVELOPMENT_TRACKER (2026-04-29, 2026-05-05) |

## Calibration vs validation

No calibration against observed restoration outcomes has been performed. Parameter values are
set by plausibility and internal diagnostics, not fit to data (see `assumptions.md`). Therefore
none of the internal checks above should be reported as validation, and no calibration-derived
confidence should be attached to absolute objective values.

## Robustness / sensitivity status

- Seed sensitivity: covered by multi-seed runs.
- Scenario sensitivity: condition-benchmark and leave-one-out indicator scenarios exist
  (`data/anomaly_scenarios/`) and are run-selectable, but a consolidated sensitivity summary
  across them is not yet part of this report.
- Parameter sensitivity (effect-model parameters): not systematically quantified.
  (Note: an older `Documentation/sensitivity.qmd` exists but is out of date and is not a
  source for this report.)

## Limitations of the evaluation

- No ground-truth outcomes -> no accuracy, skill, or predictive-error metrics can be reported.
- Pareto optimality is approximate (heuristic search); convergence is monitored, not guaranteed.
- Equifinality means single-solution results are weakly identified; prefer frequency/ensemble
  summaries (see `uncertainty_register.md` section 4).

## Appropriate-use statement

Use the evaluation to judge that the optimiser is working and that priorities are reproducible
under fixed settings - not to claim the model predicts restoration success.
