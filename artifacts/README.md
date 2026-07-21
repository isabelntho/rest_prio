# Modeling artifacts

Reviewable artifacts for the restoration-prioritisation optimisation model.

These are **living documents**: they are created early and revised throughout the project
lifecycle as the model changes. They externalise assumptions, decisions, evidence, and
evaluation so the work can be inspected and reused. Downstream use of any statement here is
gated by its stated status/confidence - do not treat an assumption or result as settled if
it is flagged as provisional.

The artifacts are deliberately **thin**: they summarise and link to the authoritative source
documents rather than duplicating them, so content does not drift out of sync.

## Contents and sources

| Artifact | What it holds | Authoritative source(s) it links to |
|---|---|---|
| `model_card.md` | One-page model summary: type, inputs, objectives, intended use, limits | `../README.md` |
| `conceptual_model.md` | Purpose, scope, entities, decision variables, effect model | `../README.md`, `../PIPELINE_DIAGRAM.md` |
| `assumptions.md` | Consequential assumptions with rationale and confidence | `../README.md`, `../Documentation/DEVELOPMENT_TRACKER.md` |
| `uncertainty_register.md` | Parameter / structural / scenario / stochastic / data uncertainty | `../README.md`, condition + policy scenarios |
| `evaluation_report.md` | Verification vs validation status; what evidence exists | `../outputs/multi_seed_results/`, HV history, spatial-agreement CSV |
| `provenance_manifest.json` | Machine-readable data/code/environment provenance | `../pixi.lock`, `../renv.lock`, `../environment.yml`, `../data/` |

## Scope note

This is a single-researcher, methods-development modelling project. The artifact set is limited
to the deliverables that carry genuine review value at this stage. Deliverables aimed at other
model classes or lifecycle stages (agent-based-model specifications, participatory/stakeholder
processes, formal ethics review) are out of scope for the current work.
