# Pipeline Diagrams — `resto_anom.py` Internals

Three diagrams documenting how data flows through `resto_anom.py`: the overall module
pipeline, the per-solution array transformations inside `_evaluate()` / `restoration_effect()`,
and the NSGA-III generational loop.

**Maintainer note:** update these diagrams when changing `restoration_effect()`,
objectives, operator classes, or run-settings in `run_optimization_instance()`.
See also the script-relationship diagram in `DEVELOPMENT_TRACKER.md`.

---

## Diagram 1 — Module pipeline (`run_optimization_instance`)

<pre class="mermaid">
flowchart TD

    subgraph INPUTS["Inputs"]
        IC["initial_conditions dict<br/>abiotic_anomaly [H×W float]<br/>biotic_anomaly [H×W float]<br/>landscape_anomaly [H×W float]<br/>implementation_cost [H×W float]<br/>restoration_eligible_mask [H×W bool]<br/>conversion_eligible_mask [H×W bool]<br/>restoration_eligible_indices [1D int]<br/>n_restoration_pixels, n_conversion_pixels"]
        SP["scenario_params dict<br/>max_restoration_fraction<br/>abiotic_effect, biotic_effect<br/>anomaly_weight_shape / scale<br/>spatial_clustering, burden_sharing<br/>normalize_objectives"]
    end

    IC & SP --> PQ{use_patch_approach?}

    PQ -- YES --> IPP["initialize_patch_approach()<br/>create_patch_mappings()<br/>→ patch_mappings added to initial_conditions<br/>n_restoration_patches, n_conversion_patches"]
    IPP --> PROB_P["PatchRestorationProblem<br/>n_var = n_restoration_patches + n_conversion_patches"]

    PQ -- NO --> PROB_X["RestorationProblem<br/>n_var = n_restoration_pixels + n_conversion_pixels"]

    PROB_P & PROB_X --> SCALES["_compute_normalization_denominators()<br/>scale[obj] = sum of abs(baseline[eligible_mask])<br/>max_action_pixels = max_restoration_fraction × n_restoration_pixels"]

    SCALES --> BRS["build_repair_scores()<br/>per-pixel priority scores [n_restoration_pixels float]<br/>uses anomaly_improvement_weight() × abiotic/biotic_effect"]

    BRS --> OQ{use_patch_approach?}

    OQ -- NO --> PIXOPS["PIXEL OPERATORS<br/>AdaptiveSampling — spatial_operations.py<br/>— random select max_action_pixels from eligible<br/>— optional spatial clustering / burden-sharing<br/>AdaptiveRepair — spatial_operations.py<br/>— pixel-level score-based add/remove"]

    OQ -- YES --> PAGGREG["aggregate_patch_scores_from_pixel_scores()<br/>pixel scores → patch-level mean scores [n_patches float]"]
    PAGGREG --> PATCHOPS["PATCH OPERATORS<br/>PatchAwareSampling — patch_approach.py<br/>— score_temperature, random_share params<br/>PatchRepair — patch_approach.py<br/>— whole-patch add/remove by score<br/>— pixel_tolerance enforced"]

    PIXOPS & PATCHOPS --> REF["build_fixed_ref_point()<br/>warm-up: evaluate n_samples random solutions<br/>→ fixed HV reference point [n_obj float]"]

    REF --> ALGO["_build_algorithm()<br/>NSGA2(sampling, HUX crossover,<br/>BitFlip mutation prob=0.1, repair)<br/>+ get_termination('n_gen', n_generations)"]

    ALGO --> OPT["minimize() — NSGA-III loop<br/>(see Diagram 3)"]

    OPT --> PKG["_package_results()<br/>objectives_raw [n_sol × n_obj]<br/>objectives_normalized [n_sol × n_obj]<br/>decisions [n_sol × n_var binary]<br/>HV history, population statistics, algo_info"]

    PKG --> SQ{save_results?}
    SQ -- YES --> DISK["save_results_with_reports()<br/>→ .pkl + JSON evolution reports"]
    SQ -- NO --> ROUT
    DISK --> ROUT[("optimization_results dict")]
</pre>

---

## Diagram 2 — Array transformations: `_evaluate()` and `restoration_effect()`

<pre class="mermaid">
flowchart TD

    XVEC["Decision vector x<br/>patch mode: [n_restoration_patches + n_conversion_patches] binary<br/>pixel mode: [n_restoration_pixels + n_conversion_pixels] binary"]

    XVEC --> PMQ{patch mode?}
    PMQ -- YES --> P2PIX["convert_patch_decisions_to_pixels()<br/>x_patches [n_patches]<br/>→ x_pixels [n_restoration_pixels + n_conversion_pixels]<br/>(each selected patch expanded to its member pixels)"]
    PMQ -- NO --> SPLIT
    P2PIX --> SPLIT

    SPLIT["Split decision vector<br/>x_restore [n_restoration_pixels]<br/>x_convert  [n_conversion_pixels]"]

    SPLIT -- x_restore --> MAPR["Map restoration decisions to 2D<br/>restoration_eligible_indices[x_restore == 1]<br/>row, col = divmod(indices, W)<br/>→ restoration_mask_2d [H×W bool]"]

    SPLIT -- x_convert --> MAPC["Map conversion decisions to 2D<br/>conversion_eligible_indices[x_convert == 1]<br/>→ conversion_mask_2d [H×W bool]"]

    MAPR --> WGT["anomaly_improvement_weight(baseline_anomalies)<br/>exponential: w = (1 − exp(−|a| / scale)) ^ 3<br/>w = 0 if anomaly >= 0  (cell not degraded)<br/>→ weights [n_selected float, 0 to 1]"]

    WGT --> DIR["Direct effect on action cells<br/>updated[action_mask] = baseline + effect × weight"]

    NBR --> EMASK["Neighbour effect (flat spillover — 2026-04-08)<br/>binary_dilation(action_mask, circular kernel radius r)<br/>→ neighbour_mask  (action cells excluded)<br/>updated[neighbour_mask] += effect × decay<br/>(no anomaly weighting on neighbours: flat benefit regardless of neighbour anomaly value)"]

    NBR --> EMASK["Mask back to restoration_eligible_mask<br/>(changes outside eligible area reverted)"]

    MAPC --> LCOPY["lulc_data.copy() [H×W int]<br/>lulc_data[conversion_mask_2d] = focal_class_value"]
    LCOPY --> DENS["compute_sn_dens_array()<br/>sliding-window focal-class density, radius 300 m<br/>→ density [H×W float 0 to 1]"]
    DENS --> LANOM["landscape_anomaly = 1 − density  [H×W float]"]

    EMASK --> OAGG["Aggregate objectives over eligible masks"]
    LANOM --> OAGG

    OAGG --> OBJ_AB["abiotic / biotic objective<br/>− sum( (updated − baseline)[restoration_eligible_mask] )<br/>(negative anomaly change = improvement; sign flipped for minimisation)"]
    OAGG --> OBJ_L["landscape objective<br/>sum(L1 − L0) / (sum(L0) + eps)<br/>(relative change)"]
    OAGG --> OBJ_C["cost objective (if present)<br/>sum( implementation_cost[action_pixels] )"]

    OBJ_AB & OBJ_L & OBJ_C --> FNORM["_normalize_objective_vector()<br/>F_norm[i] = raw_obj[i] / scale[i]<br/>→ normalized objectives [n_obj float]"]

    FNORM --> CONSTR["Budget constraint<br/>G = | n_total_actions − max_action_pixels |<br/>target: G = 0  (repair operator enforces this before evaluation)"]

    CONSTR --> FOUT["out['F']     [n_obj normalized]<br/>out['F_raw'] [n_obj raw]<br/>out['G']     [1]"]
</pre>

---

## Diagram 3 — NSGA-III generational loop

<pre class="mermaid">
flowchart TD

    subgraph INIT["Initialisation"]
        SAMP["Sampling.do(problem, pop_size)<br/>PIXEL — AdaptiveSampling<br/>  random select max_action_pixels from eligible pixels<br/>  optional spatial clustering / burden-sharing<br/>PATCH — PatchAwareSampling<br/>  score-guided patch selection (score_temperature, random_share)<br/>  target pixel count enforced within tolerance<br/>→ population [n_ref_dirs × n_var binary]<br/>  (n_ref_dirs = 45 for n_partitions=8, 3 objectives)"]
        IREPAIR["Initial repair pass<br/>PIXEL — AdaptiveRepair: exact pixel count enforced by score<br/>PATCH — PatchRepair: whole-patch add/remove by score"]
        SAMP --> IREPAIR
    end

    IREPAIR --> EVAL0["Evaluate initial population<br/>_evaluate() per individual  (see Diagram 2)<br/>→ F [n_obj], G [1] for each solution"]

    EVAL0 --> GL["=== GENERATION LOOP ==="]

    GL --> SEL["NSGA-III reference-direction-based selection<br/>Das-Dennis structured ref dirs (n_partitions=8 → 45 dirs)<br/>association: each solution assigned to nearest ref dir<br/>→ parent pairs selected by ref-dir niche count"]

    SEL --> CROSS["HUX crossover<br/>swap complementary half-bits between parents<br/>→ offspring [pop_size × n_var binary]"]

    CROSS --> MUT["BitFlip mutation  (prob = 0.1 per bit)<br/>→ mutated offspring [pop_size × n_var binary]"]

    MUT --> REP["Repair operator<br/>PIXEL — AdaptiveRepair<br/>  count actions; add/remove individual pixels by priority score<br/>  until n_actions == max_action_pixels exactly<br/>PATCH — PatchRepair  (pixel_count or patch_count mode)<br/>  add/remove whole patches by score-guided sampling<br/>  eval tolerance = pixel_tolerance × 1.5  (allows discretisation slack)"]

    REP --> EVALO["Evaluate offspring<br/>_evaluate() per individual  (see Diagram 2)<br/>parallelised via Pool.starmap when n_jobs > 1<br/>→ F [n_obj], G [1]"]

    EVALO --> NSORT["NSGA-III survival selection<br/>merge parent + offspring  [2 × n_ref_dirs solutions]<br/>fast non-dominated sort → rank<br/>reference-direction niche preservation → prune to n_ref_dirs"]

    NSORT --> HVCB["ProgressCallback / HVCallback<br/>HV(Pareto front F, fixed ref_point) → hv_history<br/>track f_mean / f_std / f_min / f_max per generation<br/>no HV improvement for hv_patience generations → converged = True"]

    HVCB --> TQ{termination?}
    TQ -- "n_gen reached OR converged" --> EXTRACT["Extract Pareto front<br/>result.X [n_pareto × n_var binary]<br/>result.F [n_pareto × n_obj normalized]"]
    TQ -- continue --> GL

    EXTRACT --> RAWRE["Re-evaluate raw objectives<br/>evaluate_raw_objectives(xi) per Pareto solution<br/>→ objectives_raw [n_pareto × n_obj]"]

    RAWRE --> RET["Return to _package_results()  (see Diagram 1)"]
</pre>

<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
mermaid.initialize({ startOnLoad: true });
</script>
