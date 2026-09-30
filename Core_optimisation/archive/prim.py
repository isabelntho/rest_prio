"""Scenario discovery: which assumptions make a given restoration plan fail?

`uncertainty_analysis` scores plans against 85 named condition tags and ranks them by
worst-case regret. It cannot say WHICH combination of formulation choices breaks a plan,
because its uncertainty set is a handful of corners rather than a sampled space. This
module samples the space and runs PRIM (Friedman-Fisher bump hunting) per plan.

An ensemble member is 13 numbers: the benchmark quantile q, the abiotic block share beta
of the union weight vector, and 11 indicator weights. Composites are rebuilt in python
from the exported per-indicator z-scores (Core_optimisation/condition_composite), so no
new R runs are needed. The quantile axis works because a positive affine map preserves
quantile membership: R's reference set {x >= Q_p(x)} is the same set as {z >= Q_p(z)},
so z_q = (z_g - mean(z_g[S_p])) / sd(z_g[S_p]) and the global mean/sd cancel.

  pixi run python -m Core_optimisation.prim <stage>

  gate      Fixed point (q=0 is `global`), quantile axis vs the on-disk upper_q*
            rasters, and d_v at q=0/flat vs the engine-validated layer shard.
            Nothing downstream is meaningful until this passes.
  select    Representative plans: cost strata x within-front regret off the baseline
            front (broken apart spatially), plus a dominated high-cost/low-benefit
            plan and the single most robust plan by uncertainty_analysis's archive-
            wide (discrete-axis) criterion. These are the only plans ever scored.
  design    The LHC ensemble.
  score     Plan x scenario benefit, plus each scenario's own benefit scale.
  fail      Normalised regret = 1 - benefit / top-k achievable benefit, and the
            satisficing threshold on it. Raw benefit is NOT comparable across the
            ensemble - the achievable scale drifts 3.9x with the benchmark quantile -
            so a fixed benefit threshold would measure that rescaling, not the plan.
  prim      PRIM per plan; peeling trajectories, box selection, permutation null.
  compare   Cross-plan box comparison and figures.

The stages above ask which assumptions break ONE plan. The four below transpose the
problem - rows are plans, columns are plan CHARACTERISTICS - and ask what a robust map
looks like. Read `features`' marginal table before any box: half of max_regret variance
sits between condition tags, so a plan-shape box can be a provenance proxy. Inference is
clustered on the tag throughout (CLUSTER_BY), not on the plan and not on the run.

  features    Plan-level covariates for the whole archive: patch structure, compactness,
              dispersion, land-cover composition, cost, and indicator exposure. Gated on
              recomputing each plan's cost and benefit from the rasters.
  plan_lhc    The same plans' regret under the continuous LHC ensemble rather than the 61
              discrete tags, on a stratified subsample. Reports it both cost-matched
              (comparable with the discrete outcomes) and scale-normalised (`fail`'s
              convention); cost enters those two with OPPOSITE signs, so which one the
              box is fitted on is a real choice, not a formatting one.
  plan_prim   PRIM over plan characteristics, four outcomes x two feature sets, with a
              TAG-level null and bootstrap.
  plan_compare  Do the four outcomes agree, and what does the robust box look like on the
              ground.

  all       the above in order

The full 25-composite weighting gate lives in Debugs_tests/weight_simplex_screen.py,
which shares this module's composite code; `gate` here re-checks three of them.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.stats import binom, gamma, qmc

from Core_optimisation.paths import OUTPUTS, FIGS_DIR, ensure
from Core_optimisation.condition_composite import (
    ABIOTIC, BIOTIC, SCEN_DIR, build_matrices, composite_by_ecosystem,
    draw_to_eco_columns, ecosystem_masks, l1_from_flat, load_indicator_stack,
    quantile_prefix, read_raster, rescale_to_quantile, scheme_weight_columns,
)
from Core_optimisation.uncertainty_analysis import (
    anomaly_weight, evaluate, front_reference, jaccard, load_archive, load_cross,
    load_layers, nondominated_2d, plan_masks, regret_against, DISCRIM_MIN_SPAN,
    ROBUST_QUANTILE, VALIDATE_RTOL_COST, _jsonable,
    OUT_DIR as UNC_DIR,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ===========================================================================
# configuration
# ===========================================================================
# q = 0 recovers the `global` benchmark. Capped at 0.75, the highest quantile with
# on-disk rasters to gate against; above it the reference set thins to the point where
# R starts skipping indicators, which is a composition change rather than a move along
# the axis.
Q_RANGE = (0.0, 0.75)
BETA_RANGE = (0.10, 0.90)
# Reference-set floor, mirroring the degenerate-sd skip in ec_anomalies.r:517-527.
REF_SET_MIN_N = 5000

N_SCENARIOS = 1000
# Dirichlet concentration for the weight block. 1.0 is uniform over the simplex; the
# LHC columns go through gamma.ppf so the stratification survives the transform.
DIRICHLET_ALPHA = 1.0
DESIGN_SEED = 20260901

# q = 0 + flat union weights, which is exactly the on-disk global_w_flat tag.
BASELINE_TAG = "global_w_flat"

# Representative plans: COST_STRATA cost levels x {low, high} within-front regret.
#
# The split is the front's own median max_regret, NOT plan_summary's `is_robust`: that
# flag is the bottom quartile of the whole 14591-plan archive, and 52 of the 56 baseline
# front plans clear it, so it cannot separate them. `is_robust` is still reported.
COST_STRATA = 3
CANDIDATES_PER_STRATUM = 40
# Topped up with the most spatially distinct remaining front plans when a stratum is
# empty, so the count stays in the 5-8 the analysis is designed around.
N_PLANS_TARGET = 6
SELECT_SEED = 20260901

# Failure: performance below this percentile of the baseline front is unacceptable.
SATISFICE_PCTL = 25

PRIM_ALPHA = 0.05
PRIM_MASS_MIN = 0.05
PRIM_COVERAGE_MIN = 0.50
PRIM_BOOTSTRAP = 100
PRIM_PVALUE_MAX = 0.05
# Label permutations per plan. A 13-dimension peel down to a 5% box always finds SOME
# enrichment, so a box only means something if it beats what the same peel gets on noise.
PRIM_NULL_PERM = 20
PRIM_SEED = 20260901

GATE_QUANTILES = (0.25, 0.50, 0.75)
GATE_DROP_TAGS = (("upper_q50_drop_soc", 0.50, "soc"),
                  ("upper_q25_drop_ndvi", 0.25, "ndvi"))
# Signed z-scores crossing zero, so the absolute term carries the criterion - a relative
# one divides by ~0 near the benchmark condition. Same tolerance as weight_simplex_screen.
GATE_ATOL = 1e-5
GATE_RTOL = 1e-5
# The exported layers' own mean/sd, which is what makes q = 0 the global benchmark.
GATE_FIXED_POINT_TOL = 1e-6
# d_v at q=0/flat against the layer shard, which is itself validated against the engine.
# Measured 9e-8 of the layer's scale and 1e-8 on the sum, so both carry ~100x headroom.
GATE_RTOL_LAYER = 1e-5
GATE_RTOL_SUM = 1e-6
GATE_RTOL_BENEFIT = 1e-5

# --- plan-characteristic stages -------------------------------------------
# A covariate with fewer than this many distinct values cannot be peeled on. The budget
# is a hard equality constraint, so n_cells takes 4 values over the whole archive; this
# drops it (and anything else that turns out constant) by measurement rather than by
# assumption.
PLAN_FEATURE_MIN_DISTINCT = 5

# The archive's 14591 plans are not independent draws, so the effective sample size is
# the CLUSTER count and the null and bootstrap resample clusters rather than plans; the
# plan-level null is reported beside the clustered one to show how much that matters
# rather than asserting it.
#
# The cluster is the condition tag: 73 of them, 5 seeds and 74-432 plans each. Seeds of
# one tag are replicates of the same formulation, which is the pooling convention the
# rest of the analysis uses, so they belong inside one cluster and not beside it. The
# finer campaign|tag|seed run is kept as a column but is not the unit - it splits those
# replicates apart, and buys little: run identity explains 55.4% of max_regret variance
# against the tag's 51.4%, so all but ~4 points of the run effect IS the tag effect.
CLUSTER_BY = "native_tag"
CLUSTER_LABEL = "tag"
PLAN_NULL_PERM = 20
PLAN_BOOTSTRAP = 100
PLAN_SEED = 20260901

LHC_SUBSAMPLE_N = 2000
LHC_SUBSAMPLE_SEED = 20260901
# Worst case over 1000 scenarios is one scenario; p90 is the same statement with a
# replicate behind it. The max is recorded next to it either way.
LHC_REGRET_PCTL = 90

# Box-overlap reading aid, NOT a test. Two boxes selected independently from 14591 plans
# at ~20% mass overlap ~0.11 by chance, so 0.30 is "clearly more than chance" and nothing
# more; the printed Jaccard is the number to quote, and a value near the cut means the
# answer is genuinely intermediate rather than one side of a line.
PLAN_OVERLAP_MIN = 0.30

OUT_DIR = OUTPUTS / "uncertainty" / "prim"
FIG_DIR = FIGS_DIR / "uncertainty" / "prim"
REPORT_JSON = OUT_DIR / "report.json"


def report_update(stage, payload):
    """Merge one stage's headline numbers into report.json under its own key."""
    ensure(OUT_DIR)
    doc = {}
    if REPORT_JSON.exists():
        try:
            with open(REPORT_JSON, encoding="ascii") as f:
                doc = json.load(f)
        except (ValueError, OSError) as e:
            print(f"  (report.json unreadable, starting a fresh one: {e!r})")
    doc[stage] = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    doc["config"] = {
        "q_range": list(Q_RANGE), "beta_range": list(BETA_RANGE),
        "ref_set_min_n": REF_SET_MIN_N, "n_scenarios": N_SCENARIOS,
        "dirichlet_alpha": DIRICHLET_ALPHA, "design_seed": DESIGN_SEED,
        "baseline_tag": BASELINE_TAG, "cost_strata": COST_STRATA,
        "satisfice_pctl": SATISFICE_PCTL, "prim_alpha": PRIM_ALPHA,
        "prim_mass_min": PRIM_MASS_MIN, "prim_coverage_min": PRIM_COVERAGE_MIN,
        "prim_bootstrap": PRIM_BOOTSTRAP, "prim_pvalue_max": PRIM_PVALUE_MAX,
    }
    with open(REPORT_JSON, "w", encoding="ascii") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=_jsonable)
    print(f"  report  -> {REPORT_JSON} [{stage}]")


# ===========================================================================
# shared inputs
# ===========================================================================
def load_context():
    """Everything the composite path needs: layers, matrices, masks, engine constants."""
    L = load_layers([BASELINE_TAG])
    if BASELINE_TAG not in L["tags"]:
        raise SystemExit(f"no layer shard for {BASELINE_TAG} - run "
                         "`uncertainty_analysis extract` first.")
    shape = L["shape"]
    elig = np.asarray(L["elig"][BASELINE_TAG], bool)
    elig_idx = np.flatnonzero(elig.ravel())

    layers, footprints = load_indicator_stack("global")
    eco_masks = ecosystem_masks(shape)
    Z, P, codes, eco_of_pixel, ecos = build_matrices(
        layers, footprints, eco_masks, elig_idx)
    eco_share = np.array([(eco_of_pixel == ei).sum() for ei in range(len(ecos))], float)
    eco_share /= max(eco_share.sum(), 1.0)

    eff = L["eff"]
    return {
        "shape": shape, "elig_idx": elig_idx, "cost": L["cost"],
        "radius": int(L["radius"]), "decay": float(L["decay"]),
        "eff_total": float(eff["abiotic_effect"] + eff["biotic_effect"]),
        "d_baseline": np.asarray(L["d"][BASELINE_TAG], np.float64),
        "layers": layers, "footprints": footprints,
        "Z": Z, "P": P, "codes": codes, "eco_of_pixel": eco_of_pixel, "ecos": ecos,
        "eco_share": eco_share,
        "flat_col": scheme_weight_columns("flat", codes, ecos, layers),
        "prefix": quantile_prefix(layers),
    }


def scenario_layer(ctx, q, w_union, prefix=None):
    """Per-cell direct benefit d over the eligible cells, for one ensemble member."""
    M, S, skipped = rescale_to_quantile(
        ctx["prefix"] if prefix is None else prefix, q, ctx["codes"], ctx["ecos"],
        REF_SET_MIN_N)
    Wcol = draw_to_eco_columns(np.asarray(w_union, float), ctx["flat_col"])
    C = composite_by_ecosystem(ctx["Z"], ctx["P"], ctx["eco_of_pixel"], Wcol,
                               affine=(M, S))
    d = ctx["eff_total"] * anomaly_weight(np.nan_to_num(C, nan=0.0))
    return np.where(np.isfinite(C), d, 0.0), Wcol, skipped


def flat_union(n_codes):
    """The union weight vector that resolves to `flat` in every ecosystem."""
    return np.full(n_codes, 1.0 / n_codes)


# ===========================================================================
# stage: gate
# ===========================================================================
def _category_columns(cat_codes, codes, ecos, layers):
    """Equal weights on the available codes of one ECT category, per ecosystem.

    composite renormalises by the weights actually present, so equal non-zero weights
    reproduce calculate_ect_mean_anomaly's NA-aware mean.
    """
    Wcol = np.zeros((len(codes), len(ecos)))
    for ei, eco in enumerate(ecos):
        for ci, c in enumerate(codes):
            if c in cat_codes and (eco, c) in layers:
                Wcol[ci, ei] = 1.0
    return Wcol


def cmd_gate(argv=()):
    """Fixed point, quantile axis vs R, and d_v vs the engine-validated layer shard."""
    ctx = load_context()
    codes, ecos, layers = ctx["codes"], ctx["ecos"], ctx["layers"]
    elig_idx = ctx["elig_idx"]
    out = {}

    print("[A] fixed point: exported layers must have mean 0, sd 1")
    worst_m = worst_s = 0.0
    for (eco, code), (a, _s1, _s2) in ctx["prefix"].items():
        worst_m = max(worst_m, abs(float(a.mean())))
        worst_s = max(worst_s, abs(float(a.std(ddof=1)) - 1.0))
    print(f"    worst |mean| {worst_m:.2e}, worst |sd - 1| {worst_s:.2e} "
          f"(tol {GATE_FIXED_POINT_TOL:.0e})")
    if max(worst_m, worst_s) > GATE_FIXED_POINT_TOL:
        raise SystemExit("FAILED - q = 0 does not recover the global benchmark.")
    out["fixed_point"] = {"max_abs_mean": worst_m, "max_abs_sd_minus_1": worst_s}

    print("\n[B] quantile axis: rebuilt two-block composites vs the on-disk rasters")
    targets = [(f"upper_q{int(q * 100)}_all", q, None) for q in GATE_QUANTILES]
    targets += list(GATE_DROP_TAGS)
    rows, n_fail = [], 0
    for tag, q, drop in targets:
        M, S, skipped = rescale_to_quantile(ctx["prefix"], q, codes, ecos, REF_SET_MIN_N)
        for cat_name, cat_codes in (("abiotic", ABIOTIC), ("biotic", BIOTIC)):
            path = SCEN_DIR / f"{cat_name}_{tag}.tif"
            if not path.exists():
                print(f"    {tag:<22} {cat_name:<8} SKIPPED - {path.name} not on disk")
                continue
            keep = [c for c in cat_codes if c != drop]
            Wcol = _category_columns(keep, codes, ecos, layers)
            got = composite_by_ecosystem(ctx["Z"], ctx["P"], ctx["eco_of_pixel"], Wcol,
                                         affine=(M, S))
            ref = read_raster(path).ravel()[elig_idx]
            both = np.isfinite(ref) & np.isfinite(got)
            err = np.abs(got[both] - ref[both])
            tol = GATE_ATOL + GATE_RTOL * np.abs(ref[both])
            n_bad = int((err > tol).sum())
            n_ref, n_got = int(np.isfinite(ref).sum()), int(np.isfinite(got).sum())
            max_abs = float(err.max()) if err.size else np.nan
            ok = n_bad == 0 and n_ref == n_got
            n_fail += 0 if ok else 1
            print(f"    {'OK ' if ok else 'FAIL'} {tag:<22} {cat_name:<8} "
                  f"max abs err {max_abs:.2e}  {n_bad:,} px outside tol  "
                  f"(ref {n_ref:,} / rebuilt {n_got:,})"
                  + (f"  [{len(skipped)} indicator(s) skipped]" if skipped else ""))
            rows.append({"tag": tag, "category": cat_name, "q": q,
                         "n_compared": int(both.sum()), "n_ref_valid": n_ref,
                         "n_rebuilt_valid": n_got, "n_outside_tol": n_bad,
                         "max_abs_err": max_abs, "n_skipped": len(skipped)})
    if not rows:
        raise SystemExit("no upper_q* rasters found to validate the quantile axis against")
    if n_fail:
        raise SystemExit(f"\nFAILED on {n_fail} of {len(rows)} comparisons. The affine "
                         "quantile rebuild does not reproduce R - stop here.")
    print(f"    PASSED - worst {max(r['max_abs_err'] for r in rows):.2e} over "
          f"{len(rows)} comparisons")
    out["quantile_axis"] = rows

    print("\n[C] end-to-end: d_v at q=0/flat vs the engine-validated layer shard")
    d, _W, _sk = scenario_layer(ctx, 0.0, flat_union(len(codes)))
    ref = ctx["d_baseline"].ravel()[elig_idx]
    # Judged against the layer's own scale, not per-cell relative. d = eff*(1-exp(-|C|))^3
    # is ~|C|^3 near the composite's zero crossing, so a float32 wobble in C is amplified
    # 3x relative on cells whose d is already ~1e-11 - a per-cell relative test reports
    # those as 1e-2 errors on a rebuild that is exact everywhere that carries weight.
    scale_d = float(np.abs(ref).max())
    abs_err = float(np.abs(d - ref).max())
    sum_err = abs(float(d.sum() - ref.sum()) / max(abs(float(ref.sum())), 1e-30))
    print(f"    max abs error {abs_err:.2e} vs layer max {scale_d:.2e} "
          f"({abs_err / scale_d:.1e} of scale, tol {GATE_RTOL_LAYER:.0e})")
    print(f"    summed benefit error {sum_err:.2e} (tol {GATE_RTOL_SUM:.0e}) "
          "- this is the quantity a plan score is built from")
    if abs_err / scale_d > GATE_RTOL_LAYER or sum_err > GATE_RTOL_SUM:
        raise SystemExit("FAILED - the rebuilt baseline layer is not the shard's layer.")
    out["end_to_end"] = {"max_abs_err": abs_err, "layer_max": scale_d,
                         "max_abs_err_rel_to_scale": abs_err / scale_d,
                         "summed_benefit_rel_err": sum_err}

    print("\n  GATE PASSED")
    report_update("gate", out)


# ===========================================================================
# stage: select
# ===========================================================================
def _patch_stats(sel_flat, shape):
    """(n_patches, mean_patch) of one plan, 4-connectivity."""
    m = np.zeros(shape, bool)
    m.flat[np.asarray(sel_flat, np.int64)] = True
    lab, n = ndimage.label(m)
    if n == 0:
        return 0, 0.0
    return int(n), float(np.bincount(lab.ravel())[1:].mean())


def cmd_select(argv=()):
    """Representative plans: cost strata x within-front regret, plus two outliers."""
    B, C, _OV, tags = load_cross()
    if BASELINE_TAG not in tags:
        raise SystemExit(f"{BASELINE_TAG} is not a column of cross_raw.npz")
    b = B[:, tags.index(BASELINE_TAG)]
    front = np.flatnonzero(nondominated_2d(np.column_stack([b, C])))
    print(f"Baseline front under {BASELINE_TAG}: {front.size} of {len(C)} plans")

    # Archive-wide, not front-only: the two extras below draw from outside the front.
    ps_full = pd.read_csv(UNC_DIR / "plan_summary.csv",
                          usecols=["plan_id", "max_regret", "is_robust", "native_tag",
                                   "native_seed", "n_cells"]).set_index("plan_id")
    ps = ps_full.loc[front]
    reg = ps["max_regret"].to_numpy()
    robust_cut = np.median(reg)
    print(f"  max_regret on the front: {reg.min():.3f} to {reg.max():.3f}, "
          f"median {robust_cut:.3f}  (is_robust: {int(ps['is_robust'].sum())} of "
          f"{len(ps)}, too few False to stratify on)")
    rho = float(pd.Series(C[front]).corr(pd.Series(reg), method="spearman"))
    print(f"  cost vs max_regret on the front: Spearman {rho:.2f}"
          + ("  <- the two strata axes are NOT independent" if abs(rho) > 0.4 else ""))

    A = load_archive()
    shape = A_shape = tuple(int(v) for v in load_layers([BASELINE_TAG])["shape"])
    rng = np.random.default_rng(SELECT_SEED)
    picks, rows = [], []

    def add_pick(pid, cost_stratum, regret_band, label, note=""):
        picks.append(int(pid))
        n_patch, mean_patch = _patch_stats(A["plans"][pid], shape)
        rows.append({"plan_id": int(pid), "cost_stratum": cost_stratum,
                     "regret_band": regret_band, "cost": float(C[pid]),
                     "benefit_baseline": float(b[pid]),
                     "max_regret": float(ps_full.loc[pid, "max_regret"]),
                     "is_robust": bool(ps_full.loc[pid, "is_robust"]),
                     "native_tag": str(ps_full.loc[pid, "native_tag"]),
                     "native_seed": int(ps_full.loc[pid, "native_seed"]),
                     "n_cells": int(A["plans"][pid].size),
                     "n_patches": n_patch, "mean_patch": mean_patch, "note": note})
        print(f"  {label:<16} plan {pid:<6} cost {C[pid]:.0f}  benefit {b[pid]:.2f}  "
              f"max_regret {ps_full.loc[pid, 'max_regret']:.3f}  {n_patch} patches"
              + (f"  ({note})" if note else ""))

    def take(cand, stratum, band):
        """Pick the most spatially distinct front candidate and record it."""
        if cand.size > CANDIDATES_PER_STRATUM:
            cand = np.sort(rng.choice(cand, CANDIDATES_PER_STRATUM, replace=False))
        if not picks:
            best = cand[np.argsort(b[cand])[cand.size // 2]]
        else:
            dist = [min(1.0 - jaccard(A["plans"][c], A["plans"][p]) for p in picks)
                    for c in cand]
            best = cand[int(np.argmax(dist))]
        add_pick(best, stratum, band, f"cost{stratum}/{band}")

    edges = np.quantile(C[front], np.linspace(0, 1, COST_STRATA + 1))
    edges[-1] += 1.0
    for si in range(COST_STRATA):
        in_cost = (C[front] >= edges[si]) & (C[front] < edges[si + 1])
        for band, in_band in (("low", reg <= robust_cut), ("high", reg > robust_cut)):
            cand = front[in_cost & in_band]
            if cand.size == 0:
                print(f"  cost{si + 1}/{band}: empty")
                continue
            take(cand, si + 1, band)
    while len(picks) < N_PLANS_TARGET:
        rest = np.setdiff1d(front, np.asarray(picks))
        if rest.size == 0:
            break
        take(rest, 0, "topup")

    # Extra 1: a DOMINATED plan from the high-cost/low-benefit cloud, off the front
    # entirely - the front-and-strata picks above are all efficient by construction, so
    # this is the contrast case: does PRIM find a much larger, less informative box for
    # a plan nobody would actually choose?
    ref = front_reference(b[front], C[front])
    reg_all, _refb, _unc = regret_against(ref, b, C)
    cost_p75 = float(np.quantile(C[front], 0.75))
    cand = np.setdiff1d(np.flatnonzero(C >= cost_p75), np.asarray(picks))
    if cand.size:
        dom = int(cand[np.argmax(reg_all[cand])])
        add_pick(dom, -1, "dominated", "extra/dominated",
                 note=f"cost-matched regret {reg_all[dom]:.2f} at cost >= p75 of the "
                      f"front ({cost_p75:.0f})")
    else:
        print("  extra/dominated: no candidate at cost >= p75 of the front")

    # Extra 2: the single most robust plan by uncertainty_analysis's own archive-wide
    # criterion (bottom quartile of max_regret against the 85 discrete formulation
    # tags) - a DIFFERENT robustness notion than this module's continuous ensemble.
    # Excludes exact max_regret == 0.0: those are the cheapest is_robust plans and 0.0
    # is the value regret_against returns for a plan below every reference front's cost
    # floor ("uncovered"), which cross's own diagnostic flags as an artefact, not a
    # genuine zero-regret result.
    robust_pool = ps_full.index[ps_full["is_robust"] & (ps_full["max_regret"] > 0) &
                                ~ps_full.index.isin(picks)]
    if len(robust_pool):
        rid = int(ps_full.loc[robust_pool, "max_regret"].idxmin())
        add_pick(rid, -1, "archive_robust", "extra/archive_robust",
                 note=f"lowest nonzero archive-wide max_regret "
                      f"({ps_full.loc[rid, 'max_regret']:.4f}) among the bottom-"
                      f"quartile robust subset")
    else:
        print("  extra/archive_robust: no candidate available")

    df = pd.DataFrame(rows)
    pair = np.ones((len(picks), len(picks)))
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            pair[i, j] = pair[j, i] = jaccard(A["plans"][picks[i]], A["plans"][picks[j]])
    print(f"\n  pairwise Jaccard: min {pair[~np.eye(len(picks), dtype=bool)].min():.3f}, "
          f"median {np.median(pair[~np.eye(len(picks), dtype=bool)]):.3f}")

    ensure(OUT_DIR)
    df.to_csv(OUT_DIR / "representative_plans.csv", index=False)
    offsets = np.zeros(len(picks) + 1, np.int64)
    offsets[1:] = np.cumsum([A["plans"][p].size for p in picks])
    np.savez_compressed(OUT_DIR / "representative_plans.npz",
                        plan_id=np.asarray(picks, np.int64),
                        sel_flat=np.concatenate([A["plans"][p] for p in picks]),
                        sel_offsets=offsets, jaccard=pair,
                        cost=np.asarray([C[p] for p in picks]),
                        benefit_baseline=np.asarray([b[p] for p in picks]))
    print(f"  -> {OUT_DIR / 'representative_plans.csv'}")
    _select_figure(b, C, front, picks, A, shape)
    report_update("select", {
        "n_front": int(front.size), "n_picked": len(picks),
        "plan_ids": picks, "regret_median_cut": float(robust_cut),
        "n_front_is_robust": int(ps["is_robust"].sum()),
        "cost_regret_spearman": rho, "cost_edges": edges.tolist(),
        "jaccard_min": float(pair[~np.eye(len(picks), dtype=bool)].min()),
        "jaccard_median": float(np.median(pair[~np.eye(len(picks), dtype=bool)])),
        "plans": df.to_dict("records"),
    })


def _plan_thumbnail(ax, flat_idx, shape, pad=15):
    """Selected-cell mask cropped to its own bounding box, for a small inset map."""
    rows, cols = np.unravel_index(np.asarray(flat_idx, np.int64), shape)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, shape[1])
    m = np.zeros((r1 - r0, c1 - c0), bool)
    m[rows - r0, cols - c0] = True
    ax.imshow(m, cmap="Greens", vmin=0, vmax=1.4, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _select_figure(b, C, front, picks, A, shape):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ensure(FIG_DIR)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(-b, C, s=3, c="0.8", label="archive")
    ax.scatter(-b[front], C[front], s=8, c="tab:blue", label="baseline front")
    ax.scatter(-b[picks], C[picks], s=90, marker="D", facecolor="none",
               edgecolor="crimson", linewidth=1.6, label="representative")
    for i, p in enumerate(picks):
        ax.annotate(str(i), (-b[p], C[p]), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("restoration benefit")
    ax.set_ylabel("implementation cost")
    ax.set_title(f"Representative plans on the full uncertainty front")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "representative_plans_front.png", dpi=140)
    plt.close(fig)

    # Spatial context: what "cheap" vs "expensive" looks like on the ground, cropped to
    # each plan's own footprint so a 3-patch plan and a 500-patch plan are both legible.
    n = len(picks)
    t_cols = 4 if n > 6 else 3
    t_rows = int(np.ceil(n / t_cols))
    fig = plt.figure(figsize=(2.3 * t_cols, 2.6 * t_rows))
    gs = GridSpec(t_rows, t_cols, figure=fig, wspace=0.05, hspace=0.3)
    for i, p in enumerate(picks):
        r, c = divmod(i, t_cols)
        tax = fig.add_subplot(gs[r, c])
        _plan_thumbnail(tax, A["plans"][p], shape)
        tax.set_title(f"{i}: plan {p}", fontsize=7)
    fig.suptitle("Representative plans - spatial footprint", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "representative_plans_maps.png", dpi=140)
    plt.close(fig)
    print(f"  figures -> {FIG_DIR / 'representative_plans_front.png'}, "
          f"{FIG_DIR / 'representative_plans_maps.png'}")


# ===========================================================================
# stage: design
# ===========================================================================
def weight_columns():
    """Design column names, in the order PRIM sees them."""
    return ["q", "beta"] + [f"w_{c}" for c in list(ABIOTIC) + list(BIOTIC)]


def cmd_design(argv=()):
    """Latin hypercube over q, beta and the 11 indicator weights."""
    cols = weight_columns()
    n_ab, n_bi = len(ABIOTIC), len(BIOTIC)
    u = qmc.LatinHypercube(d=len(cols), seed=DESIGN_SEED).random(N_SCENARIOS)

    q = Q_RANGE[0] + u[:, 0] * (Q_RANGE[1] - Q_RANGE[0])
    beta = BETA_RANGE[0] + u[:, 1] * (BETA_RANGE[1] - BETA_RANGE[0])
    # Gamma inverse-CDF then normalise = a Dirichlet draw that keeps the LHC
    # stratification; a plain u / sum(u) would bias every draw toward flat.
    g = gamma.ppf(np.clip(u[:, 2:], 1e-12, 1 - 1e-12), a=DIRICHLET_ALPHA)
    w_ab = g[:, :n_ab] / g[:, :n_ab].sum(axis=1, keepdims=True) * beta[:, None]
    w_bi = g[:, n_ab:] / g[:, n_ab:].sum(axis=1, keepdims=True) * (1 - beta)[:, None]
    W = np.hstack([w_ab, w_bi])

    df = pd.DataFrame(np.column_stack([q, beta, W]), columns=cols)
    # Row 0 is the baseline: q = 0 and flat union weights, i.e. the global_w_flat tag.
    df.iloc[0] = [0.0, n_ab / (n_ab + n_bi)] + [1.0 / (n_ab + n_bi)] * (n_ab + n_bi)
    bad = float(np.abs(df[cols[2:]].sum(axis=1) - 1.0).max())
    if bad > 1e-9:
        raise SystemExit(f"weight vectors do not sum to 1 (worst {bad:.2e})")

    ensure(OUT_DIR)
    df.to_csv(OUT_DIR / "design.csv", index=False)
    print(f"{N_SCENARIOS} scenarios x {len(cols)} factors -> {OUT_DIR / 'design.csv'}")
    print(f"  q    [{q.min():.3f}, {q.max():.3f}]   beta [{beta.min():.3f}, "
          f"{beta.max():.3f}]   weight sums off by <= {bad:.1e}")
    report_update("design", {
        "n_scenarios": N_SCENARIOS, "factors": cols,
        "q_min": float(q.min()), "q_max": float(q.max()),
        "beta_min": float(beta.min()), "beta_max": float(beta.max()),
        "weight_sum_max_err": bad,
    })


# ===========================================================================
# stage: score
# ===========================================================================
def load_plans():
    """(plan_ids, [flat index arrays], cost, baseline benefit) from `select`."""
    p = OUT_DIR / "representative_plans.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the `select` stage first.")
    z = np.load(p, allow_pickle=True)
    off, flat = z["sel_offsets"], z["sel_flat"]
    plans = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return z["plan_id"], plans, z["cost"], z["benefit_baseline"]


def cmd_score(argv=()):
    """Score the representative plans under every ensemble member."""
    ctx = load_context()
    plan_ids, plans, cost, b_native = load_plans()
    design = pd.read_csv(OUT_DIR / "design.csv")
    cols = weight_columns()
    elig_idx, shape = ctx["elig_idx"], ctx["shape"]
    n_elig, n_plan, n_scen = elig_idx.size, len(plans), len(design)
    budget = int(np.median([p.size for p in plans]))

    # Flat raster index -> position among eligible cells. A spillover cell outside the
    # eligible set carries no benefit (d is zero there), so it simply drops out.
    pos = np.full(int(np.prod(shape)), -1, np.int64)
    pos[elig_idx] = np.arange(n_elig)
    rows, cols_i, vals = [], [], []
    for i, sel in enumerate(plans):
        _sel_m, nb_m = plan_masks(sel, shape, ctx["radius"])
        for idx, weight in ((np.asarray(sel, np.int64), 1.0),
                            (np.flatnonzero(nb_m.ravel()), ctx["decay"])):
            p = pos[idx]
            p = p[p >= 0]
            rows.append(np.full(p.size, i))
            cols_i.append(p)
            vals.append(np.full(p.size, weight))
    S = sparse.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols_i))),
                          shape=(n_plan, n_elig))
    print(f"{n_plan} plans x {n_scen} scenarios over {n_elig:,} eligible cells "
          f"(budget k = {budget:,})")

    Bm = np.zeros((n_plan, n_scen))
    scale = np.zeros(n_scen)
    diag = []
    t0 = time.perf_counter()
    for s in range(n_scen):
        row = design.iloc[s]
        w = row[cols[2:]].to_numpy(float)
        d, Wcol, skipped = scenario_layer(ctx, float(row["q"]), w)
        Bm[:, s] = -(S @ d)
        scale[s] = float(np.partition(d, -budget)[-budget:].sum())
        diag.append({"scenario": s, "q": float(row["q"]), "beta": float(row["beta"]),
                     "l1_from_flat": l1_from_flat(Wcol, ctx["flat_col"], ctx["eco_share"]),
                     "n_skipped": len(skipped), "scale": scale[s]})
        if (s + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"    {s + 1}/{n_scen}  ({el:.0f}s, eta "
                  f"{el / (s + 1) * (n_scen - s - 1):.0f}s)")

    # Scenario 0 is the baseline, so it must reproduce the stored global_w_flat scores.
    err = float(np.max(np.abs(Bm[:, 0] - b_native) / np.maximum(np.abs(b_native), 1e-30)))
    print(f"  self-check vs cross_raw {BASELINE_TAG}: {err:.2e} "
          f"(tol {GATE_RTOL_BENEFIT:.0e})")
    if err > GATE_RTOL_BENEFIT:
        raise SystemExit("Self-check FAILED - scenario 0 is not the baseline lens.")

    n_sk = sum(1 for r in diag if r["n_skipped"])
    print(f"  scenarios with a skipped indicator: {n_sk} of {n_scen}")
    print(f"  scenario scale: min {scale.min():.1f}, median {np.median(scale):.1f}, "
          f"max {scale.max():.1f}  ({scale.max() / scale.min():.1f}x spread)")
    # Discriminability, measured on the plans actually under analysis. A best-k vs
    # worst-k span over raw cells would read ~1.0 everywhere - the worst-k cells are the
    # ones scoring exactly zero - and so would measure nothing.
    perf = -Bm
    spread = (perf.max(axis=0) - perf.min(axis=0)) / np.maximum(perf.max(axis=0), 1e-30)
    n_sat = int((spread < DISCRIM_MIN_SPAN).sum())
    print(f"  plan spread (max-min)/max over the {n_plan} plans: min {spread.min():.3f}, "
          f"median {np.median(spread):.3f}, max {spread.max():.3f}")
    print(f"  scenarios where the objective barely separates them (< "
          f"{DISCRIM_MIN_SPAN}): {n_sat} of {n_scen} ({n_sat / n_scen:.0%})")

    ensure(OUT_DIR)
    np.savez_compressed(OUT_DIR / "scores.npz", B=Bm, scale=scale, spread=spread,
                        cost=cost, plan_id=plan_ids)
    pd.DataFrame(diag).assign(spread=spread).to_csv(
        OUT_DIR / "scenario_diagnostics.csv", index=False)
    print(f"  -> {OUT_DIR / 'scores.npz'}")
    report_update("score", {
        "n_plans": n_plan, "n_scenarios": n_scen, "n_eligible": int(n_elig),
        "budget_k": budget, "selfcheck_max_rel_err": err,
        "n_scenarios_with_skip": n_sk,
        "scale_min": float(scale.min()), "scale_median": float(np.median(scale)),
        "scale_max": float(scale.max()),
        "runtime_s": float(time.perf_counter() - t0),
    })


# ===========================================================================
# stage: fail
# ===========================================================================
def load_scores():
    p = OUT_DIR / "scores.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the `score` stage first.")
    z = np.load(p, allow_pickle=True)
    return z["B"], z["scale"], z["cost"], z["plan_id"]


def cmd_fail(argv=()):
    """Both failure definitions, plus the scale diagnostic that separates them."""
    Bm, scale, _cost, plan_ids = load_scores()
    B_all, C_all, _OV, tags = load_cross()
    b = B_all[:, tags.index(BASELINE_TAG)]
    front = np.flatnonzero(nondominated_2d(np.column_stack([b, C_all])))

    # Normalised regret: the share of the achievable benefit a plan forgoes under this
    # scenario. scale(v) is the top-k sum of d_v, so this divides out the scenario's own
    # benefit magnitude - which drifts 3.9x across the ensemble and would otherwise make
    # a fixed benefit threshold measure the objective's rescaling rather than the plan.
    perf = -Bm
    regret = np.clip(1.0 - perf / scale[None, :], 0.0, 1.0)

    # The threshold is still the satisficing standard: the regret the p25 plan of the
    # baseline front incurs under the BASELINE scenario, then required everywhere.
    thr_abs = float(np.percentile(-b[front], SATISFICE_PCTL))
    thr = 1.0 - thr_abs / float(scale[0])
    fail = regret > thr
    print(f"Satisficing standard: p{SATISFICE_PCTL} of the {front.size}-plan baseline "
          f"front = benefit {thr_abs:.3f}")
    print(f"Failure: normalised regret > {thr:.4f}  (that plan forgoes "
          f"{thr:.1%} of what scenario {BASELINE_TAG} makes achievable)")

    print("\n  per plan        fail rate    n_fail    median regret")
    rows = []
    for i, pid in enumerate(plan_ids):
        print(f"    plan {int(pid):<10} {fail[i].mean():>7.1%}  {int(fail[i].sum()):>7}"
              f"      {np.median(regret[i]):>10.3f}")
        rows.append({"plan_id": int(pid), "frac_fail": float(fail[i].mean()),
                     "n_fail": int(fail[i].sum()),
                     "median_regret": float(np.median(regret[i])),
                     "max_regret": float(regret[i].max())})
    degenerate = [r["plan_id"] for r in rows if r["frac_fail"] in (0.0, 1.0)]
    print(f"  degenerate (0% or 100% failure, no box): "
          f"{degenerate if degenerate else 'none'}")

    # Why regret and not a fixed benefit threshold: both effects below run the same way.
    diag = pd.read_csv(OUT_DIR / "scenario_diagnostics.csv")
    bins = pd.cut(diag["q"], 5)
    print("\n  why the objective needs normalising, by q")
    print(f"    {'q bin':<22}{'n':>5}{'scale':>9}{'spread':>8}{'perf':>9}"
          f"{'regret':>9}{'fail':>8}")
    scale_rows = []
    for iv, g in diag.groupby(bins, observed=True):
        idx = g.index.to_numpy()
        print(f"    {str(iv):<22}{len(g):>5}{g['scale'].median():>9.1f}"
              f"{g['spread'].median():>8.3f}{perf[:, idx].mean():>9.1f}"
              f"{np.median(regret[:, idx]):>9.3f}{fail[:, idx].mean():>8.1%}")
        scale_rows.append({"q_bin": str(iv), "n": int(len(g)),
                           "scale_median": float(g["scale"].median()),
                           "spread_median": float(g["spread"].median()),
                           "mean_performance": float(perf[:, idx].mean()),
                           "median_regret": float(np.median(regret[:, idx])),
                           "frac_fail": float(fail[:, idx].mean())})
    n_sat = int((diag["spread"] < DISCRIM_MIN_SPAN).sum())
    print(f"\n  raising q inflates raw benefit ({scale_rows[0]['mean_performance']:.0f}"
          f" -> {scale_rows[-1]['mean_performance']:.0f}) AND compresses the plans "
          f"({scale_rows[0]['spread_median']:.2f} -> {scale_rows[-1]['spread_median']:.2f}).")
    print(f"  Normalising removes the first; the second is a real loss of "
          f"discriminating power ({n_sat} of {len(diag)} scenarios separate the plans "
          f"by < {DISCRIM_MIN_SPAN}).")

    ensure(OUT_DIR)
    np.savez_compressed(OUT_DIR / "failure.npz", fail=fail, regret=regret, perf=perf,
                        plan_id=plan_ids)
    pd.DataFrame(rows).to_csv(OUT_DIR / "failure_rates.csv", index=False)
    report_update("fail", {
        "measure": "normalised regret = 1 - benefit / top-k achievable benefit",
        "satisficing_benefit": thr_abs, "regret_threshold": thr,
        "baseline_scale": float(scale[0]),
        "satisfice_pctl": SATISFICE_PCTL, "n_front": int(front.size),
        "overall_rate": float(fail.mean()),
        "per_plan": rows, "degenerate_plans": degenerate,
        "n_low_spread": n_sat, "discrim_min_span": DISCRIM_MIN_SPAN,
        "scale_by_q": scale_rows,
    })


# ===========================================================================
# stage: prim
# ===========================================================================
def prim_peel(X, y, alpha=PRIM_ALPHA, mass_min=PRIM_MASS_MIN):
    """Friedman-Fisher peeling. Returns the trajectory as a list of dicts."""
    n = X.shape[0]
    n_fail = float(y.sum())
    box = np.arange(n)
    lims = np.column_stack([X.min(axis=0), X.max(axis=0)])
    traj = [{"step": 0, "idx": box, "lims": lims.copy(), "mass": 1.0,
             "density": float(y.mean()), "coverage": 1.0}]
    step = 0
    while box.size / n > mass_min:
        best = None
        for d in range(X.shape[1]):
            v = X[box, d]
            for side in (0, 1):
                cut = np.quantile(v, alpha if side == 0 else 1 - alpha)
                keep = v > cut if side == 0 else v < cut
                if keep.all() or keep.sum() < 2:
                    continue
                dens = float(y[box[keep]].mean())
                if best is None or dens > best[0]:
                    best = (dens, d, side, cut, box[keep])
        if best is None:
            break
        dens, d, side, cut, box = best
        lims[d, side] = cut
        step += 1
        traj.append({"step": step, "idx": box, "lims": lims.copy(),
                     "mass": box.size / n, "density": dens,
                     "coverage": float(y[box].sum() / n_fail) if n_fail else 0.0})
    return traj


def prim_paste(X, y, idx, lims, alpha=PRIM_ALPHA):
    """Extend restricted edges outward while density improves."""
    lims = lims.copy()
    full_lo, full_hi = X.min(axis=0), X.max(axis=0)
    improved = True
    while improved:
        improved = False
        dens = float(y[idx].mean())
        for d in range(X.shape[1]):
            for side in (0, 1):
                edge = lims[d, side]
                if (side == 0 and edge <= full_lo[d]) or (side == 1 and edge >= full_hi[d]):
                    continue
                span = lims[d, 1] - lims[d, 0]
                trial = lims.copy()
                trial[d, side] = (max(full_lo[d], edge - alpha * span) if side == 0
                                  else min(full_hi[d], edge + alpha * span))
                new = np.flatnonzero(np.all((X >= trial[:, 0]) & (X <= trial[:, 1]), axis=1))
                if new.size > idx.size and float(y[new].mean()) > dens:
                    lims, idx, dens, improved = trial, new, float(y[new].mean()), True
    return idx, lims


def _restricted_dims(lims, full):
    """Indices of dimensions whose box limits are inside the sampled range."""
    return [d for d in range(len(lims))
            if lims[d, 0] > full[d, 0] + 1e-12 or lims[d, 1] < full[d, 1] - 1e-12]


def select_box(traj, coverage_min=PRIM_COVERAGE_MIN):
    """Densest box still covering `coverage_min` of the failures."""
    ok = [t for t in traj if t["coverage"] >= coverage_min and t["mass"] > 0]
    pool = ok if ok else traj
    return max(pool, key=lambda t: t["density"])


def quasi_p(X, y, idx, lims, dims):
    """Per-dimension quasi-p-value: is the restriction better than chance? (Bryant &
    Lempert) - the box density against the density of the box with that dim relaxed."""
    out = {}
    n_fail_box = int(y[idx].sum())
    for d in dims:
        relaxed = lims.copy()
        relaxed[d] = [X[:, d].min(), X[:, d].max()]
        rel_idx = np.flatnonzero(np.all((X >= relaxed[:, 0]) & (X <= relaxed[:, 1]), axis=1))
        p_rel = float(y[rel_idx].mean()) if rel_idx.size else 0.0
        out[d] = (1.0 if p_rel >= 1.0 else
                  float(binom.sf(n_fail_box - 1, idx.size, min(max(p_rel, 1e-12), 1.0))))
    return out


def _fit(X, y):
    """Peel, paste, select: returns (idx, lims, chosen trajectory entry, traj)."""
    traj = prim_peel(X, y)
    chosen = select_box(traj)
    idx, lims = prim_paste(X, y, chosen["idx"], chosen["lims"])
    return idx, lims, chosen, traj


def null_lift(X, y, rng, n_perm=PRIM_NULL_PERM):
    """Density lift PRIM reaches on permuted labels: the bar a real box must clear.

    Peeling 13 dimensions down to a 5% box will always find SOME enrichment, and the
    thinner the failure set the more it finds. Reported per plan because the bar depends
    on that plan's failure rate.
    """
    lifts = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        idx, _l, _c, _t = _fit(X, yp)
        lifts.append(float(yp[idx].mean() / max(yp.mean(), 1e-12)))
    return float(np.percentile(lifts, 95)), float(np.median(lifts))


def cmd_prim(argv=()):
    """PRIM per plan, for both failure definitions."""
    design = pd.read_csv(OUT_DIR / "design.csv")
    cols = weight_columns()
    X = design[cols].to_numpy(float)
    full = np.column_stack([X.min(axis=0), X.max(axis=0)])
    z = np.load(OUT_DIR / "failure.npz", allow_pickle=True)
    plan_ids = z["plan_id"]
    rng = np.random.default_rng(PRIM_SEED)

    print(f"  lift = box density / base rate; null = what the same peel reaches on "
          f"permuted labels ({PRIM_NULL_PERM} draws, p95).")
    print(f"  A box is only credible when lift clears its own null.\n")
    box_rows, traj_rows, dim_rows, membership = [], [], [], {}
    Y = np.asarray(z["fail"], float)
    for i, pid in enumerate(plan_ids):
        y = Y[i]
        if y.sum() == 0 or y.sum() == y.size:
            print(f"  plan {int(pid)}: degenerate ({y.mean():.0%}) - no box")
            continue
        idx, lims, chosen, traj = _fit(X, y)
        dims = _restricted_dims(lims, full)
        pvals = quasi_p(X, y, idx, lims, dims)
        lift = float(y[idx].mean() / y.mean())
        null95, null50 = null_lift(X, y, rng)
        credible = lift > null95
        membership[int(pid)] = idx
        box_rows.append({
            "plan_id": int(pid), "n_fail": int(y.sum()),
            "frac_fail": float(y.mean()), "mass": idx.size / len(X),
            "density": float(y[idx].mean()),
            "coverage": float(y[idx].sum() / y.sum()),
            "lift": lift, "null_lift_p95": null95, "null_lift_median": null50,
            "credible": credible, "n_restricted": len(dims),
            "restricted": ";".join(
                f"{cols[d]}=[{lims[d, 0]:.3f},{lims[d, 1]:.3f}]" for d in dims),
        })
        for t in traj:
            traj_rows.append({"plan_id": int(pid), "step": t["step"], "mass": t["mass"],
                              "density": t["density"], "coverage": t["coverage"],
                              "selected": t["step"] == chosen["step"]})
        for d in dims:
            dim_rows.append({"plan_id": int(pid), "dimension": cols[d],
                             "lo": lims[d, 0], "hi": lims[d, 1], "quasi_p": pvals[d],
                             "supported": pvals[d] <= PRIM_PVALUE_MAX})
        print(f"  plan {int(pid):<6} fail {y.mean():>5.1%}  n_fail {int(y.sum()):>3}  "
              f"density {y[idx].mean():.2f}  cov {y[idx].sum() / y.sum():.2f}  "
              f"lift {lift:.1f}x vs null {null95:.1f}x  "
              f"{'CREDIBLE' if credible else 'NOT ABOVE NULL'}")
        print("         dims: " + ", ".join(
            f"{cols[d]}{'*' if pvals[d] <= PRIM_PVALUE_MAX else ''}" for d in dims))

    # Bootstrap: how often does a dimension survive a resample AS A SUPPORTED one?
    # Counting "restricted at all" is useless here - a 60-step peel over 13 dimensions
    # touches nearly every dimension at least once, so that reads ~1.00 for everything.
    print(f"\n  bootstrap ({PRIM_BOOTSTRAP} resamples) - frequency a dimension is "
          f"restricted AND clears quasi-p <= {PRIM_PVALUE_MAX}")
    freq = {}
    for i, pid in enumerate(plan_ids):
        y = Y[i]
        if y.sum() == 0 or y.sum() == y.size:
            continue
        counts = np.zeros(len(cols))
        for _ in range(PRIM_BOOTSTRAP):
            take = rng.integers(0, len(X), len(X))
            Xb, yb = X[take], y[take]
            if yb.sum() < 2:
                continue
            bi, blims, _bc, _bt = _fit(Xb, yb)
            bdims = _restricted_dims(blims, full)
            bp = quasi_p(Xb, yb, bi, blims, bdims)
            for d in bdims:
                if bp[d] <= PRIM_PVALUE_MAX:
                    counts[d] += 1
        freq[int(pid)] = counts / PRIM_BOOTSTRAP
    for r in dim_rows:
        r["boot_freq"] = float(freq[r["plan_id"]][cols.index(r["dimension"])])
    for pid, c in sorted(freq.items()):
        top = [d for d in np.argsort(-c)[:5] if c[d] > 0.25]
        print(f"    plan {pid:<6} " +
              ("  ".join(f"{cols[d]} {c[d]:.2f}" for d in top) if top
               else "no dimension survives in >25% of resamples"))

    ensure(OUT_DIR)
    pd.DataFrame(box_rows).to_csv(OUT_DIR / "boxes.csv", index=False)
    pd.DataFrame(traj_rows).to_csv(OUT_DIR / "peel_trajectory.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(OUT_DIR / "dimension_support.csv", index=False)
    np.savez_compressed(OUT_DIR / "box_membership.npz",
                        **{f"plan{p}": v for p, v in membership.items()})
    print(f"  -> {OUT_DIR / 'boxes.csv'}")
    n_cred = sum(1 for r in box_rows if r["credible"])
    print(f"\n  {n_cred} of {len(box_rows)} boxes clear their permutation null.")
    report_update("prim", {
        "boxes": box_rows, "dimensions": dim_rows, "factors": cols,
        "n_boxes": len(box_rows), "n_credible": n_cred,
        "null_permutations": PRIM_NULL_PERM,
    })


# ===========================================================================
# stage: compare
# ===========================================================================
def cmd_compare(argv=()):
    """Do the plans fail under the same assumptions, or different ones?"""
    boxes = pd.read_csv(OUT_DIR / "boxes.csv")
    dims = pd.read_csv(OUT_DIR / "dimension_support.csv")
    mem = dict(np.load(OUT_DIR / "box_membership.npz", allow_pickle=True))
    design = pd.read_csv(OUT_DIR / "design.csv")
    cols = weight_columns()

    dropped = [int(p) for p, c in zip(boxes.plan_id, boxes.credible) if not c]
    ids = [int(p) for p, c in zip(boxes.plan_id, boxes.credible) if c]
    if dropped:
        print(f"  excluded, box does not clear its permutation null: {dropped}")
    if len(ids) < 2:
        raise SystemExit("fewer than two credible boxes - nothing to compare")

    print("\n  pairwise box overlap (Jaccard of in-box scenarios)")
    M = np.ones((len(ids), len(ids)))
    for a in range(len(ids)):
        for b_ in range(a + 1, len(ids)):
            M[a, b_] = M[b_, a] = jaccard(mem[f"plan{ids[a]}"], mem[f"plan{ids[b_]}"])
    print("        " + "".join(f"{i:>8}" for i in ids))
    for a, i in enumerate(ids):
        print(f"    {i:>4}" + "".join(f"{M[a, b_]:>8.2f}" for b_ in range(len(ids))))
    off = M[~np.eye(len(ids), dtype=bool)]
    print(f"    median {np.median(off):.2f}, min {off.min():.2f} -> "
          + ("the plans fail under LARGELY THE SAME assumptions"
             if np.median(off) > 0.5 else
             "the plans fail under DIFFERENT assumptions"))

    print("\n  which factors bind, by plan (bootstrap frequency in brackets)")
    bind_rows = []
    for pid in ids:
        sub = dims[(dims.plan_id == pid) & dims.supported].sort_values(
            "boot_freq", ascending=False)
        print(f"    plan {pid:<6} " + ("  ".join(
            f"{r.dimension}[{r.boot_freq:.2f}]" for r in sub.itertuples())
            or "none supported"))
        bind_rows.append({"plan_id": pid,
                          "supported_dims": list(sub["dimension"]),
                          "boot_freq": [float(v) for v in sub["boot_freq"]]})

    _compare_figures(boxes, dims, mem, design, cols, ids)
    report_update("compare", {
        "credible_plan_ids": ids, "excluded_plan_ids": dropped,
        "overlap_matrix": M.tolist(), "overlap_median": float(np.median(off)),
        "overlap_min": float(off.min()), "binding_factors": bind_rows,
        "boxes": boxes.to_dict("records"),
    })


def _compare_figures(boxes, dims, mem, design, cols, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure(FIG_DIR)
    X = design[cols].to_numpy(float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    boxes = boxes[boxes.plan_id.isin(ids)]
    order_ids = boxes.sort_values("frac_fail", ascending=False)["plan_id"].tolist()
    fail_of = dict(zip(boxes.plan_id, boxes.frac_fail))

    _fig_boxes(dims, ids, order_ids, fail_of, cols, lo, rng)
    _fig_scatter(dims, mem, order_ids, fail_of)
    _fig_heatmap(dims, ids, order_ids, fail_of, cols, lo, hi)
    print(f"\n  figures -> {FIG_DIR}")


def _fig_boxes(dims, ids, order_ids, fail_of, cols, lo, rng, fname="boxes.png",
               title="Vulnerability boxes - constrained factors only, tightest first",
               label_of=None):
    """One row per (plan, factor) actually constrained - unsupported factors dropped,
    tightest constraint first within each plan, so the signal is not buried.

    `fname`/`title`/`label_of` let the plan-characteristic stages reuse this unchanged;
    `dims.plan_id` is then a row-group code rather than a plan.
    """
    import matplotlib.pyplot as plt
    if label_of is None:
        label_of = lambda pid: f"plan {pid}\n{fail_of[pid]:.0%} fail"
    rows = []
    for pid in order_ids:
        d = dims[(dims.plan_id == pid) & dims.supported].copy()
        d["width"] = d.hi - d.lo
        for _, dr in d.sort_values("width").iterrows():
            rows.append((pid, dr))
    n = len(rows)
    # A plain f"C{cols.index(dim) % 10}" can collide once more than 10 of the 13
    # factors are in play (w_smd and w_ndvi both landed on green); remap colours over
    # only the factors that actually appear so each stays visually distinct.
    used = [f for f in cols if any(dr["dimension"] == f for _, dr in rows)]
    color_of = {f: f"C{i % 10}" for i, f in enumerate(used)}
    width = 7.5 + 0.09 * max(0, max(len(f) for f in used) - 9)
    fig, ax = plt.subplots(figsize=(width, 0.32 * n + 1.4))
    group_start = 0
    for pid in order_ids:
        n_here = sum(1 for p, _ in rows if p == pid)
        if n_here == 0:
            continue
        y_top = n - 1 - group_start
        for k in range(group_start, group_start + n_here):
            _pid, dr = rows[k]
            y = n - 1 - k
            c = cols.index(dr["dimension"])
            ax.plot([(dr["lo"] - lo[c]) / rng[c], (dr["hi"] - lo[c]) / rng[c]], [y, y],
                    lw=6, solid_capstyle="butt", color=color_of[dr["dimension"]])
            ax.annotate(f"{dr['dimension']}  [{dr['boot_freq']:.0%}]",
                        ((dr["hi"] - lo[c]) / rng[c], y), fontsize=7.5, xytext=(5, 0),
                        textcoords="offset points", va="center")
        y_mid = n - 1 - (group_start + (n_here - 1) / 2)
        ax.text(-0.20, y_mid, label_of(pid), fontsize=8, ha="right", va="center")
        if group_start > 0:
            ax.axhline(y_top + 0.5, color="0.85", lw=0.8)
        group_start += n_here
    ax.set_yticks([])
    # Right margin sized to the longest annotation rather than fixed: the formulation
    # factors are ~8 characters and fit under a flat 1.28, but the plan characteristics
    # reusing this run to 20+ and would be clipped off the axes.
    tail = 0.022 * max(len(f"{dr['dimension']}  [100%]") for _p, dr in rows)
    ax.set_xlim(-0.02, 1.02 + tail)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("retained factor range (normalized); [ ] = bootstrap support")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


def _fig_scatter(dims, mem, order_ids, fail_of, n_panels=3):
    """q vs L1-from-flat for the most informative plans, with the PRIM box overlaid.

    The box lives in 13-dim factor space; q's interval is the box's own exact bound,
    but L1 is a derived summary of the 11 weight dims, not a native PRIM axis, so its
    side of the rectangle is the empirical min/max of L1 among the box's own in-box
    scenarios - the tightest axis-aligned envelope this 2-D projection can show honestly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    diag = pd.read_csv(OUT_DIR / "scenario_diagnostics.csv")
    z = np.load(OUT_DIR / "failure.npz", allow_pickle=True)
    plan_id = [int(p) for p in z["plan_id"]]

    by_fail = sorted(order_ids, key=lambda p: fail_of[p])
    pids = list(dict.fromkeys(by_fail[-2:][::-1] + by_fail[:1]))[:n_panels]

    fig, axs = plt.subplots(1, len(pids), figsize=(4.3 * len(pids), 4.3), squeeze=False)
    for a, pid in enumerate(pids):
        ax = axs[0][a]
        f = np.asarray(z["fail"][plan_id.index(pid)], bool)
        ax.scatter(diag["q"][~f], diag["l1_from_flat"][~f], s=6, c="0.75", label="ok")
        ax.scatter(diag["q"][f], diag["l1_from_flat"][f], s=6, c="crimson", label="fail")
        idx = mem[f"plan{pid}"]
        qb = dims[(dims.plan_id == pid) & (dims.dimension == "q")]
        q0, q1 = ((float(qb.lo.iloc[0]), float(qb.hi.iloc[0])) if len(qb)
                  else (diag["q"].min(), diag["q"].max()))
        l0, l1 = diag["l1_from_flat"].iloc[idx].min(), diag["l1_from_flat"].iloc[idx].max()
        ax.add_patch(Rectangle((q0, l0), q1 - q0, l1 - l0, fill=False, ls="--",
                               ec="k", lw=1.3, label="PRIM box"))
        ax.set_title(f"plan {pid}  ({fail_of[pid]:.0%} fail)", fontsize=10)
        ax.set_xlabel("q")
        if a == 0:
            ax.set_ylabel("L1 distance from flat weights")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Failing scenarios cluster at low q, high weight departure", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "failure_scatter.png", dpi=140)
    plt.close(fig)


def _fig_heatmap(dims, ids, order_ids, fail_of, cols, lo, hi,
                 fname="vulnerability_heatmap.png",
                 title="Vulnerability factors across plans", label_of=None):
    """The cross-plan result: which factors are plan-invariant vulnerabilities.

    Colour = fraction of that factor's sampled range the box excludes (darker = tighter),
    which is the direction-agnostic, cross-factor-comparable quantity - the raw retained
    bound (e.g. "w_sbd < 0.09") is not comparable across factors with different ranges,
    so it is left out of the cell rather than shown alongside the colour it can conflict
    with; see dimension_support.csv for the raw bounds.
    """
    import matplotlib.pyplot as plt
    if label_of is None:
        label_of = lambda pid: f"plan {pid} ({fail_of[pid]:.0%})"
    sup = dims[dims.supported & dims.plan_id.isin(ids)]
    factors = (sup.groupby("dimension").size().sort_values(ascending=False).index.tolist())
    if not factors:
        print("  no factor is supported in any credible box - skipping the heatmap")
        return
    tight = np.full((len(order_ids), len(factors)), np.nan)
    for pi, pid in enumerate(order_ids):
        for fi, f in enumerate(factors):
            r = sup[(sup.plan_id == pid) & (sup.dimension == f)]
            if r.empty:
                continue
            c = cols.index(f)
            tight[pi, fi] = 1.0 - (float(r.hi.iloc[0]) - float(r.lo.iloc[0])) / \
                max(hi[c] - lo[c], 1e-9)

    # vmax is the OBSERVED max, not 1: PRIM's peel stops at PRIM_MASS_MIN (5% of the
    # scenarios), so a box practically never excludes a whole factor's range and
    # tightness never nears 1.0. Fixing vmax=1 would leave the colourbar's darkest third
    # empty - every cell reads noticeably lighter than the swatch it is meant to match -
    # so the printed bar's extremes are calibrated to what is actually on the grid.
    vmax = float(np.nanmax(tight))
    fig, ax = plt.subplots(figsize=(1.1 * len(factors) + 2, 0.6 * len(order_ids) + 1.5))
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(tight), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, rotation=45, ha="right")
    ax.set_yticks(range(len(order_ids)))
    ax.set_yticklabels([label_of(p) for p in order_ids])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="fraction of sampled range excluded", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


# ===========================================================================
# stage: features - plan-level covariates
# ===========================================================================
# The covariate blocks. zbar_<code> is appended at build time from the indicator codes
# actually on disk, so it is not listed here.
SPATIAL_FEATURES = ("n_patches", "largest_patch_share", "mean_patch_size", "perim_area",
                    "rgyr", "bbox_fill", "spillover_ratio")
COVER_FEATURES = ("share_forest", "share_agricultural", "share_grassland")
EXPOSURE_FEATURES = ("mean_d_baseline", "n_indicators_mean", "eff_n_indicators")
CORE_FEATURES = SPATIAL_FEATURES + ("cost",) + COVER_FEATURES + EXPOSURE_FEATURES

PLAN_FEATURES_CSV = OUT_DIR / "plan_features.csv"
PLAN_LHC_CSV = OUT_DIR / "plan_features_lhc.csv"
PLAN_MARGINALS_CSV = OUT_DIR / "plan_feature_marginals.csv"


def _plan_geometry(sel, shape, masks):
    """Spatial form of one plan: patch structure, compactness, dispersion, spillover."""
    sel_m, nb_m = masks
    lab, n = ndimage.label(sel_m)
    sizes = np.bincount(lab.ravel())[1:]
    if int(sizes.sum()) != sel.size:
        raise SystemExit(f"patch sizes sum to {int(sizes.sum())} but the plan has "
                         f"{sel.size} cells - the mask and the index disagree.")
    # 4-connected boundary: every selected/unselected transition along either axis.
    perim = float((sel_m[:, :-1] != sel_m[:, 1:]).sum() + (sel_m[:-1] != sel_m[1:]).sum())
    r, c = np.unravel_index(sel, shape)
    bbox = float((r.max() - r.min() + 1) * (c.max() - c.min() + 1))
    return {
        "n_patches": int(n),
        "largest_patch_share": float(sizes.max()) / sel.size,
        "mean_patch_size": float(sizes.mean()),
        "perim_area": perim / sel.size,
        "rgyr": float(np.sqrt(((r - r.mean()) ** 2 + (c - c.mean()) ** 2).mean())),
        "bbox_fill": sel.size / bbox,
        "spillover_ratio": float(nb_m.sum()) / sel.size,
    }


def _exposure(Zr, Pr):
    """(zbar, eff_n_indicators, n_indicators_mean) from one plan's rows of (Z, P).

    zbar is the presence-aware mean, matching composite's renormalisation: an indicator
    absent on a cell drops out rather than counting as a zero. eff_n is exp(Shannon) over
    the shares of mean |z|, so a plan leaning on one indicator reads near 1 and one spread
    evenly over all of them reads near the indicator count.
    """
    denom = np.maximum(Pr.sum(axis=0), 1.0)
    zbar = Zr.sum(axis=0) / denom
    mag = np.abs(Zr).sum(axis=0) / denom
    s = mag / max(float(mag.sum()), 1e-12)
    s = s[s > 0]
    eff = float(np.exp(-(s * np.log(s)).sum())) if s.size else 0.0
    return zbar, eff, float(Pr.sum(axis=1).mean())


def _marginals(F, feats, y="max_regret"):
    """Spearman of each feature against the outcome, split raw / within- / between-cluster.

    This IS the confound check. A feature whose association is entirely between-cluster is
    telling you which formulation found the plan, not what the map looks like.
    """
    cols = list(feats) + [y]
    g = F.groupby("cluster")
    mu = g[cols].mean()
    dev = F[cols] - g[cols].transform("mean")
    return pd.DataFrame([{
        "feature": f,
        "spearman_raw": float(F[f].corr(F[y], method="spearman")),
        "spearman_within_cluster": float(dev[f].corr(dev[y], method="spearman")),
        "spearman_between_cluster": float(mu[f].corr(mu[y], method="spearman")),
        "var_share_cluster": float(1.0 - dev[f].var() / max(float(F[f].var()), 1e-30)),
    } for f in feats])


def cmd_features(argv=()):
    """Plan-level covariates for the whole archive, gated against the rasters."""
    A = load_archive()
    L = load_layers([BASELINE_TAG])
    if BASELINE_TAG not in L["tags"]:
        raise SystemExit(f"no layer shard for {BASELINE_TAG} - run "
                         "`uncertainty_analysis extract` first.")
    shape, radius, decay = L["shape"], int(L["radius"]), float(L["decay"])
    d_base = np.asarray(L["d"][BASELINE_TAG], np.float64)
    d_flat = d_base.ravel()
    plans = [np.asarray(p, np.int64) for p in A["plans"]]
    n_plan = len(plans)

    # Indexed over the cells any plan actually selects, NOT the baseline eligible set:
    # the drop_* constructions are eligible on ~150 fewer cells, so plans born under them
    # select cells the baseline mask excludes and would silently lose their z-scores.
    union = np.unique(np.concatenate(plans))
    elig = np.asarray(L["elig"][BASELINE_TAG], bool).ravel()
    n_outside = int((~elig[union]).sum())
    print(f"{n_plan} plans over {union.size:,} distinct cells "
          f"({n_outside:,} outside the {BASELINE_TAG} eligible mask)")

    layers, footprints = load_indicator_stack("global")
    Z, P, codes, eco_of_pixel, ecos = build_matrices(
        layers, footprints, ecosystem_masks(shape), union)
    pos = np.full(int(np.prod(shape)), -1, np.int64)
    pos[union] = np.arange(union.size)
    print(f"  indicators: {len(codes)} codes, ecosystems: {', '.join(ecos)}")

    rows = []
    t0 = time.perf_counter()
    for i, sel in enumerate(plans):
        masks = plan_masks(sel, shape, radius)
        # evaluate() rather than a hand-rolled sum: it is the same arithmetic `cross`
        # used, which is what makes the two gates below exact rather than approximate.
        ben, cst = evaluate(sel, d_base, L["cost"], shape, radius, decay, masks=masks)
        r = pos[sel]
        zbar, eff_n, n_ind = _exposure(Z[r], P[r])
        eco = np.bincount(eco_of_pixel[r] + 1, minlength=len(ecos) + 1)[1:] / sel.size
        row = {"plan_id": i, **_plan_geometry(sel, shape, masks),
               "cost": cst, "benefit_baseline": ben,
               "mean_d_baseline": float(d_flat[sel].mean()),
               "n_indicators_mean": n_ind, "eff_n_indicators": eff_n}
        row.update({f"share_{e}": float(eco[k]) for k, e in enumerate(ecos)})
        row.update({f"zbar_{c}": float(zbar[k]) for k, c in enumerate(codes)})
        rows.append(row)
        if (i + 1) % 2000 == 0:
            el = time.perf_counter() - t0
            print(f"    {i + 1}/{n_plan}  ({el:.0f}s, eta "
                  f"{el / (i + 1) * (n_plan - i - 1):.0f}s)")
    F = pd.DataFrame(rows)

    print("\n[A] gate: recomputed cost and benefit vs what `cross` stored")
    ps = pd.read_csv(UNC_DIR / "plan_summary.csv",
                     usecols=["plan_id", "native_campaign", "native_tag", "native_seed",
                              "n_cells", "cost", "max_regret", "is_robust"])
    Bm, Cm, _OV, tags = load_cross()
    b_ref = Bm[:, tags.index(BASELINE_TAG)]
    c_err = float(np.max(np.abs(F["cost"].to_numpy() - Cm) / np.maximum(np.abs(Cm), 1e-30)))
    b_err = float(np.max(np.abs(F["benefit_baseline"].to_numpy() - b_ref)
                         / np.maximum(np.abs(b_ref), 1e-30)))
    print(f"    cost    max rel err {c_err:.2e}  (tol {VALIDATE_RTOL_COST:.0e})")
    print(f"    benefit max rel err {b_err:.2e}  (tol {GATE_RTOL_BENEFIT:.0e})")
    if c_err > VALIDATE_RTOL_COST or b_err > GATE_RTOL_BENEFIT:
        raise SystemExit("FAILED - the feature pipeline is not indexing the same plans "
                         "`cross` scored. Nothing downstream is meaningful.")
    share_cols = [f"share_{e}" for e in ecos]
    worst_share = float(F[share_cols].sum(axis=1).max())
    print(f"    land-cover shares sum to at most {worst_share:.6f}")
    if worst_share > 1.0 + 1e-9:
        raise SystemExit("FAILED - ecosystem shares exceed 1; the masks overlap.")

    F = F.merge(ps, on="plan_id", suffixes=("", "_summary"))
    # `run` is kept for provenance reporting; `cluster` is the inference unit.
    F["run"] = (F["native_campaign"] + "|" + F["native_tag"] + "|"
                + F["native_seed"].astype(str))
    F["cluster"] = F[CLUSTER_BY].astype(str)
    F["is_fragile"] = F["max_regret"] >= F["max_regret"].quantile(1.0 - ROBUST_QUANTILE)
    F["within_cluster_robust"] = (
        F["max_regret"] < F.groupby("cluster")["max_regret"].transform("median"))
    # regret_against returns 0.0 for a plan cheaper than every reference-front point
    # ("uncovered"), which is not robustness. Flagged here, excluded in plan_prim.
    F["uncovered_artefact"] = F["max_regret"] == 0.0
    n_unc = int(F["uncovered_artefact"].sum())
    print(f"    uncovered artefact (max_regret exactly 0.0): {n_unc} plan(s)")

    print("\n[B] degenerate covariates (fewer than "
          f"{PLAN_FEATURE_MIN_DISTINCT} distinct values)")
    cand = [c for c in CORE_FEATURES if c in F.columns] + \
           sorted(c for c in F.columns if c.startswith("zbar_"))
    dropped = [c for c in cand + ["n_cells"] if F[c].nunique() < PLAN_FEATURE_MIN_DISTINCT]
    kept = [c for c in cand if c not in dropped]
    print(f"    dropped: {', '.join(dropped) if dropped else 'none'}")
    print(f"    kept   : {len(kept)} covariates")
    if "n_cells" not in dropped:
        print("    NOTE: n_cells survived the screen - the budget is not the hard "
              "equality constraint this analysis assumes.")

    print(f"\n[C] marginal association with max_regret, split by the "
          f"{CLUSTER_LABEL} confound")
    marg = _marginals(F, kept).sort_values("spearman_raw", key=np.abs, ascending=False)

    def _var_share(col):
        dev = F["max_regret"] - F.groupby(col)["max_regret"].transform("mean")
        return float(1.0 - dev.var() / max(float(F["max_regret"].var()), 1e-30))

    cl_share, run_share = _var_share("cluster"), _var_share("run")
    print(f"    {'feature':<22}{'raw':>8}{'within':>8}{'between':>9}"
          f"{'var|' + CLUSTER_LABEL:>9}")
    for r in marg.itertuples():
        print(f"    {r.feature:<22}{r.spearman_raw:>8.3f}"
              f"{r.spearman_within_cluster:>8.3f}"
              f"{r.spearman_between_cluster:>9.3f}{r.var_share_cluster:>9.3f}")
    print(f"\n    {CLUSTER_LABEL} identity alone explains {cl_share:.1%} of max_regret "
          f"variance over {F['cluster'].nunique()} {CLUSTER_LABEL}s - the ceiling any "
          f"feature box works under.")
    print(f"    (the finer campaign|tag|seed run explains {run_share:.1%} over "
          f"{F['run'].nunique()} runs, so seed adds "
          f"{run_share - cl_share:.1%} beyond the {CLUSTER_LABEL}.)")

    ensure(OUT_DIR)
    F.to_csv(PLAN_FEATURES_CSV, index=False)
    marg.to_csv(PLAN_MARGINALS_CSV, index=False)
    print(f"  -> {PLAN_FEATURES_CSV}")
    report_update("features", {
        "n_plans": n_plan, "n_runs": int(F["run"].nunique()),
        "cluster_by": CLUSTER_BY, "n_clusters": int(F["cluster"].nunique()),
        "n_union_cells": int(union.size), "n_cells_outside_baseline_elig": n_outside,
        "cost_max_rel_err": c_err, "benefit_max_rel_err": b_err,
        "indicator_codes": list(codes), "ecosystems": list(ecos),
        "features_kept": kept, "features_dropped": dropped,
        "n_uncovered_artefact": n_unc,
        "max_regret_var_share_cluster": cl_share,
        "max_regret_var_share_run": run_share,
        "marginals": marg.to_dict("records"),
        "runtime_s": float(time.perf_counter() - t0),
    })


# ===========================================================================
# stage: plan_lhc - the same plans under the continuous ensemble
# ===========================================================================
def _load_features():
    if not PLAN_FEATURES_CSV.exists():
        raise FileNotFoundError(f"{PLAN_FEATURES_CSV} missing - run `features` first.")
    return pd.read_csv(PLAN_FEATURES_CSV)


def _stratified_subsample(F, n, seed):
    """Plan ids spread over cost decile x native tag, so this is not just the cheap tail."""
    rng = np.random.default_rng(seed)
    key = pd.qcut(F["cost"], 10, labels=False, duplicates="drop").astype(str) + \
        "|" + F["native_tag"].astype(str)
    groups = [np.asarray(v) for v in F.groupby(key).indices.values()]
    per = max(1, n // len(groups))
    take = np.concatenate([rng.choice(g, min(g.size, per), replace=False)
                           for g in groups])
    if take.size > n:
        take = rng.choice(take, n, replace=False)
    elif take.size < n:
        rest = np.setdiff1d(np.arange(len(F)), take)
        take = np.concatenate([take, rng.choice(rest, min(n - take.size, rest.size),
                                                replace=False)])
    print(f"  subsample: {take.size} plans over {len(groups)} cost x tag strata")
    return np.sort(F["plan_id"].to_numpy()[take])


def _plan_operator(plans, pos, shape, radius):
    """(core_idx, core_off, ring_idx, ring_off) over eligible-cell positions.

    Two flat index arrays rather than a scipy sparse operator: at ~95k nonzeros per plan
    a CSR would carry an explicit float weight per nonzero for what are only ever two
    distinct values, which is another 1.5 GB. A cell outside the eligible set carries no
    benefit and simply drops out, exactly as in `score`.
    """
    core, ring, c_off, r_off = [], [], [0], [0]
    for sel in plans:
        _sel_m, nb_m = plan_masks(sel, shape, radius)
        for src, acc, off in ((sel, core, c_off),
                              (np.flatnonzero(nb_m.ravel()), ring, r_off)):
            p = pos[np.asarray(src, np.int64)]
            p = p[p >= 0].astype(np.int32)
            if p.size == 0:
                raise SystemExit("a plan has an empty core or spillover ring - "
                                 "np.add.reduceat cannot express an empty segment.")
            acc.append(p)
            off.append(off[-1] + p.size)
    return (np.concatenate(core), np.asarray(c_off[:-1], np.int64),
            np.concatenate(ring), np.asarray(r_off[:-1], np.int64))


def cmd_plan_lhc(argv=()):
    """Score a stratified plan subsample under the LHC ensemble instead of the 61 tags."""
    F = _load_features()
    design = pd.read_csv(OUT_DIR / "design.csv")
    wcols = weight_columns()
    ctx = load_context()
    A = load_archive()
    pids = _stratified_subsample(F, LHC_SUBSAMPLE_N, LHC_SUBSAMPLE_SEED)
    plans = [np.asarray(A["plans"][p], np.int64) for p in pids]
    elig_idx, shape = ctx["elig_idx"], ctx["shape"]
    n_elig, n_plan, n_scen = elig_idx.size, len(plans), len(design)
    budget = int(np.median([p.size for p in plans]))

    pos = np.full(int(np.prod(shape)), -1, np.int64)
    pos[elig_idx] = np.arange(n_elig)
    t0 = time.perf_counter()
    core, c_off, ring, r_off = _plan_operator(plans, pos, shape, ctx["radius"])
    print(f"  operator: {core.size + ring.size:,} nonzeros "
          f"({(core.nbytes + ring.nbytes) / 1e6:.0f} MB) in {time.perf_counter() - t0:.0f}s")
    print(f"{n_plan} plans x {n_scen} scenarios over {n_elig:,} eligible cells "
          f"(budget k = {budget:,})")

    Bm = np.zeros((n_plan, n_scen))
    scale = np.zeros(n_scen)
    t0 = time.perf_counter()
    for s in range(n_scen):
        row = design.iloc[s]
        d, _Wcol, _sk = scenario_layer(ctx, float(row["q"]),
                                       row[wcols[2:]].to_numpy(float))
        Bm[:, s] = -(np.add.reduceat(d[core], c_off)
                     + ctx["decay"] * np.add.reduceat(d[ring], r_off))
        scale[s] = float(np.partition(d, -budget)[-budget:].sum())
        if (s + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"    {s + 1}/{n_scen}  ({el:.0f}s, eta "
                  f"{el / (s + 1) * (n_scen - s - 1):.0f}s)")

    # Scenario 0 is q=0 + flat weights, i.e. the baseline lens, so it must reproduce the
    # stored global_w_flat scores - the same self-check `score` makes.
    _B, _C, _OV, tags = load_cross()
    b_ref = _B[pids, tags.index(BASELINE_TAG)]
    err = float(np.max(np.abs(Bm[:, 0] - b_ref) / np.maximum(np.abs(b_ref), 1e-30)))
    print(f"  self-check vs cross_raw {BASELINE_TAG}: {err:.2e} "
          f"(tol {GATE_RTOL_BENEFIT:.0e})")
    if err > GATE_RTOL_BENEFIT:
        raise SystemExit("Self-check FAILED - scenario 0 is not the baseline lens.")

    # Two regret conventions, because they are NOT interchangeable once cost is a
    # covariate rather than a fixed budget:
    #   scale-normalised  1 - benefit / top-k achievable, exactly `fail`'s definition.
    #                     It does not control for cost, so an expensive plan that buys
    #                     more benefit scores well and cost reads -0.44 against it -
    #                     the opposite sign to the discrete set.
    #   cost-matched      against each scenario's own front over the scored plans, the
    #                     convention `classify` uses for the 61 tags. This is the one
    #                     the plan-characteristic boxes are fitted on, so the two
    #                     uncertainty sets are compared on the same footing.
    regret = np.clip(1.0 - (-Bm) / scale[None, :], 0.0, 1.0)
    cost = F.set_index("plan_id").loc[pids, "cost"].to_numpy(float)
    reg_cm = np.zeros_like(regret)
    n_unc = 0
    for s in range(n_scen):
        reg_cm[:, s], _rb, unc = regret_against(front_reference(Bm[:, s], cost),
                                                Bm[:, s], cost)
        n_unc += int(unc.sum())
    print(f"  cost-matched: {n_unc} plan-scenario cells uncovered (cheaper than every "
          f"front point, scored 0.0) of {n_plan * n_scen:,}")

    out = pd.DataFrame({
        "plan_id": pids,
        "lhc_regret_cm_p90": np.percentile(reg_cm, LHC_REGRET_PCTL, axis=1),
        "lhc_regret_cm_max": reg_cm.max(axis=1),
        "lhc_regret_p90": np.percentile(regret, LHC_REGRET_PCTL, axis=1),
        "lhc_regret_max": regret.max(axis=1),
        "lhc_regret_median": np.median(regret, axis=1),
    })
    mr = F.set_index("plan_id").loc[pids, "max_regret"].to_numpy()
    rho = {c: float(pd.Series(out[c].to_numpy()).corr(pd.Series(mr), method="spearman"))
           for c in ("lhc_regret_cm_p90", "lhc_regret_p90")}
    print(f"  vs the discrete-set max_regret: cost-matched Spearman "
          f"{rho['lhc_regret_cm_p90']:.3f}, scale-normalised "
          f"{rho['lhc_regret_p90']:.3f}")
    print("  " + ("the two uncertainty sets rank plans alike"
                  if rho["lhc_regret_cm_p90"] > 0.7 else
                  "the two uncertainty sets DISAGREE on the ranking even once cost is "
                  "matched - the sampled space is not the 61 corners"))

    ensure(OUT_DIR)
    np.savez_compressed(OUT_DIR / "plan_lhc_scores.npz", B=Bm, scale=scale,
                        regret=regret, regret_cost_matched=reg_cm, plan_id=pids)
    out.to_csv(PLAN_LHC_CSV, index=False)
    print(f"  -> {PLAN_LHC_CSV}")
    report_update("plan_lhc", {
        "n_plans": n_plan, "n_scenarios": n_scen, "subsample_seed": LHC_SUBSAMPLE_SEED,
        "regret_pctl": LHC_REGRET_PCTL, "selfcheck_max_rel_err": err,
        "spearman_vs_discrete_max_regret": rho,
        "n_uncovered_cells": n_unc,
        "regret_cm_p90_median": float(np.median(out["lhc_regret_cm_p90"])),
        "regret_cm_p90_min": float(out["lhc_regret_cm_p90"].min()),
        "regret_cm_p90_max": float(out["lhc_regret_cm_p90"].max()),
        "regret_p90_median": float(np.median(out["lhc_regret_p90"])),
        "runtime_s": float(time.perf_counter() - t0),
    })


# ===========================================================================
# stage: plan_prim - PRIM over plan characteristics
# ===========================================================================
def _cluster_groups(clusters):
    """Row indices of each cluster, in a fixed order."""
    return [np.flatnonzero(clusters == c) for c in pd.unique(clusters)]


def _mem_key(outcome):
    """npz-safe name for an outcome: a `/` becomes a directory entry inside the zip."""
    return outcome.replace("/", "__")


def null_lift_clustered(X, y, groups, rng, n_perm=PLAN_NULL_PERM):
    """Density lift PRIM reaches once the cluster/feature link is broken.

    The plan-level null of `null_lift` treats 14591 plans as independent when they come
    from 73 condition tags, so it sets a bar far below what clustered data produces by
    chance. Here the cluster list is shuffled and each cluster's plans are relabelled
    from the label pool of the cluster it was mapped to, which preserves plans-per-
    cluster and the within-cluster label mix while destroying the association a box
    would claim.
    """
    lifts = []
    for _ in range(n_perm):
        order = rng.permutation(len(groups))
        yp = np.empty_like(y)
        for j, g in enumerate(groups):
            pool = y[groups[order[j]]]
            yp[g] = pool[rng.integers(0, pool.size, g.size)]
        if yp.sum() == 0 or yp.sum() == yp.size:
            continue
        idx, _l, _c, _t = _fit(X, yp)
        lifts.append(float(yp[idx].mean() / max(yp.mean(), 1e-12)))
    if not lifts:
        return np.nan, np.nan
    return float(np.percentile(lifts, 95)), float(np.median(lifts))


def boot_freq_clustered(X, y, groups, full, n_cols, rng, n_boot=PLAN_BOOTSTRAP):
    """Frequency a dimension is restricted AND supported under a CLUSTER-level resample.

    Resampling plans would resample the same condition tag ~200 times over and report
    near-certainty for whatever that tag happens to look like; resampling tags is the
    honest unit.
    """
    counts = np.zeros(n_cols)
    done = 0
    for _ in range(n_boot):
        take = np.concatenate([groups[p] for p in
                               rng.integers(0, len(groups), len(groups))])
        Xb, yb = X[take], y[take]
        if yb.sum() < 2 or yb.sum() == yb.size:
            continue
        bi, blims, _bc, _bt = _fit(Xb, yb)
        bdims = _restricted_dims(blims, full)
        bp = quasi_p(Xb, yb, bi, blims, bdims)
        for d in bdims:
            if bp[d] <= PRIM_PVALUE_MAX:
                counts[d] += 1
        done += 1
    return counts / max(done, 1)


def _box_provenance(F, idx, top=5):
    """Which clusters and native tags a box is drawn from - the artefact check."""
    sub = F.iloc[idx]
    arch = F["native_tag"].value_counts(normalize=True)
    box = sub["native_tag"].value_counts(normalize=True)
    return {
        "n_clusters_in_box": int(sub["cluster"].nunique()),
        "n_clusters_total": int(F["cluster"].nunique()),
        "n_runs_in_box": int(sub["run"].nunique()),
        "top_tag_share": float(box.iloc[0]) if len(box) else 0.0,
        "top_tags": [{"native_tag": str(t), "share_box": float(s),
                      "share_archive": float(arch[t]),
                      "enrichment": float(s / arch[t]) if arch[t] else np.nan}
                     for t, s in box.head(top).items()],
    }


def plan_feature_sets(F):
    """(core, full) covariate lists after re-applying the degenerate-value screen."""
    cand = [c for c in CORE_FEATURES if c in F.columns]
    zb = sorted(c for c in F.columns if c.startswith("zbar_"))
    core = [c for c in cand if F[c].nunique() >= PLAN_FEATURE_MIN_DISTINCT]
    extra = [c for c in zb if F[c].nunique() >= PLAN_FEATURE_MIN_DISTINCT]
    return core, core + extra


def plan_outcomes(F, lhc):
    """(id, y, row mask, question) for each outcome a box is fitted on."""
    ok = ~F["uncovered_artefact"].to_numpy()
    out = [
        ("robust", F["is_robust"].to_numpy(bool), ok,
         "which plans stay good under every variant"),
        ("fragile", F["is_fragile"].to_numpy(bool), ok,
         "which plans break - the classic PRIM direction"),
        ("within_cluster", F["within_cluster_robust"].to_numpy(bool), ok,
         "given the formulation that found it, which map shape survives"),
    ]
    if lhc is not None:
        # The cost-matched column: the discrete outcomes above are cost-matched too, so
        # a Jaccard between their boxes compares uncertainty sets rather than conventions.
        col = ("lhc_regret_cm_p90" if "lhc_regret_cm_p90" in lhc.columns
               else "lhc_regret_p90")
        j = F["plan_id"].map({int(p): v for p, v in
                              zip(lhc["plan_id"], lhc[col])}).to_numpy(float)
        has = np.isfinite(j) & ok
        thr = float(np.quantile(j[has], 1.0 - ROBUST_QUANTILE))
        y = np.zeros(len(F), bool)
        y[has] = j[has] >= thr
        out.append(("lhc", y, has,
                    "fragile under the continuous ensemble, not the 61 corners"))
    return out


def cmd_plan_prim(argv=()):
    """PRIM over plan characteristics: four outcomes x two feature sets, run-clustered."""
    F = _load_features()
    lhc = pd.read_csv(PLAN_LHC_CSV) if PLAN_LHC_CSV.exists() else None
    if lhc is None:
        print(f"  ({PLAN_LHC_CSV.name} absent - skipping the `lhc` outcome)")
    core_cols, full_cols = plan_feature_sets(F)
    clusters = F["cluster"].to_numpy()
    rng = np.random.default_rng(PLAN_SEED)

    print(f"  {len(F)} plans, {F['cluster'].nunique()} {CLUSTER_LABEL}s "
          f"(clustering on {CLUSTER_BY}). Feature sets: core {len(core_cols)}, "
          f"full {len(full_cols)}.")
    print(f"  lift = box density / base rate. A box must clear the {CLUSTER_LABEL.upper()}"
          "-level null; the plan-level\n  null is printed beside it to show what ignoring "
          "the clustering would have claimed.\n")

    box_rows, dim_rows, traj_rows, membership, prov = [], [], [], {}, {}
    for set_name, cols in (("core", core_cols), ("full", full_cols)):
        for out_name, y_all, mask, question in plan_outcomes(F, lhc):
            key = f"{set_name}/{out_name}"
            sub = np.flatnonzero(mask)
            X = F[cols].to_numpy(float)[sub]
            y = np.asarray(y_all[sub], float)
            if y.sum() == 0 or y.sum() == y.size:
                print(f"  {key:<20} degenerate ({y.mean():.0%}) - no box")
                continue
            full = np.column_stack([X.min(axis=0), X.max(axis=0)])
            groups = _cluster_groups(clusters[sub])
            idx, lims, chosen, traj = _fit(X, y)
            dims = _restricted_dims(lims, full)
            pvals = quasi_p(X, y, idx, lims, dims)
            lift = float(y[idx].mean() / y.mean())
            null95, null50 = null_lift_clustered(X, y, groups, rng)
            plain95, _p50 = null_lift(X, y, rng)
            credible = bool(lift > null95)
            membership[_mem_key(key)] = sub[idx]
            prov[key] = _box_provenance(F.iloc[sub].reset_index(drop=True), idx)
            print(f"  {key:<22} base {y.mean():>5.1%}  density {y[idx].mean():.2f}  "
                  f"cov {y[idx].sum() / y.sum():.2f}  mass {idx.size / len(X):.2f}  "
                  f"lift {lift:.1f}x vs {CLUSTER_LABEL}-null {null95:.1f}x (plan-null "
                  f"{plain95:.1f}x)  {'CREDIBLE' if credible else 'NOT ABOVE NULL'}")
            print(f"       {prov[key]['n_clusters_in_box']} of "
                  f"{prov[key]['n_clusters_total']} {CLUSTER_LABEL}s represented; "
                  f"top native_tag {prov[key]['top_tag_share']:.0%} of the box")

            freq = (boot_freq_clustered(X, y, groups, full, len(cols), rng)
                    if credible else np.full(len(cols), np.nan))
            for d in dims:
                dim_rows.append({"outcome": key, "feature_set": set_name,
                                 "dimension": cols[d], "lo": lims[d, 0],
                                 "hi": lims[d, 1], "quasi_p": pvals[d],
                                 "supported": pvals[d] <= PRIM_PVALUE_MAX,
                                 "boot_freq": float(freq[d])})
            for t in traj:
                traj_rows.append({"outcome": key, "step": t["step"], "mass": t["mass"],
                                  "density": t["density"], "coverage": t["coverage"],
                                  "selected": t["step"] == chosen["step"]})
            box_rows.append({
                "outcome": key, "feature_set": set_name, "outcome_id": out_name,
                "question": question, "n_rows": int(len(X)), "n_clusters": len(groups),
                "base_rate": float(y.mean()), "mass": idx.size / len(X),
                "density": float(y[idx].mean()),
                "coverage": float(y[idx].sum() / y.sum()), "lift": lift,
                "null_lift_p95": null95, "null_lift_median": null50,
                "plain_null_lift_p95": plain95, "credible": credible,
                "n_restricted": len(dims),
                "n_clusters_in_box": prov[key]["n_clusters_in_box"],
                "n_runs_in_box": prov[key]["n_runs_in_box"],
                "top_tag_share": prov[key]["top_tag_share"],
                "restricted": ";".join(
                    f"{cols[d]}=[{lims[d, 0]:.4g},{lims[d, 1]:.4g}]" for d in dims),
            })
            sup = [cols[d] for d in dims if pvals[d] <= PRIM_PVALUE_MAX]
            print(f"       supported: {', '.join(sup) if sup else 'none'}\n")

    ensure(OUT_DIR)
    pd.DataFrame(box_rows).to_csv(OUT_DIR / "plan_boxes.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(OUT_DIR / "plan_dimension_support.csv", index=False)
    pd.DataFrame(traj_rows).to_csv(OUT_DIR / "plan_peel_trajectory.csv", index=False)
    np.savez_compressed(OUT_DIR / "plan_box_membership.npz", **membership)
    n_cred = sum(1 for r in box_rows if r["credible"])
    print(f"  {n_cred} of {len(box_rows)} boxes clear the {CLUSTER_LABEL}-level null.")
    print(f"  -> {OUT_DIR / 'plan_boxes.csv'}")
    report_update("plan_prim", {
        "cluster_by": CLUSTER_BY, "n_clusters": int(F["cluster"].nunique()),
        "core_features": core_cols, "full_features": full_cols,
        "n_boxes": len(box_rows), "n_credible": n_cred,
        "null_permutations": PLAN_NULL_PERM, "bootstrap": PLAN_BOOTSTRAP,
        "boxes": box_rows, "dimensions": dim_rows, "provenance": prov,
    })


# ===========================================================================
# stage: plan_compare
# ===========================================================================
def cmd_plan_compare(argv=()):
    """Do the four outcomes describe the same plans, and what does the box look like?"""
    boxes = pd.read_csv(OUT_DIR / "plan_boxes.csv")
    dims = pd.read_csv(OUT_DIR / "plan_dimension_support.csv")
    mem = dict(np.load(OUT_DIR / "plan_box_membership.npz", allow_pickle=True))
    F = _load_features()
    lhc = pd.read_csv(PLAN_LHC_CSV) if PLAN_LHC_CSV.exists() else None

    dropped = list(boxes.loc[~boxes.credible, "outcome"])
    ids = list(boxes.loc[boxes.credible, "outcome"])
    if dropped:
        print(f"  excluded, box does not clear the {CLUSTER_LABEL}-level null: {dropped}")
    if len(ids) < 2:
        raise SystemExit("fewer than two credible boxes - nothing to compare")

    # Populations are a property of the outcome, not the feature set, and are rebuilt
    # rather than stored: `lhc` is fitted on the 2000-plan subsample and the rest on all
    # 14591, so every cross-outcome comparison has to be restricted to the rows BOTH
    # boxes could have selected.
    pops = {}
    for s in ("core", "full"):
        for oid, _y, m, _q in plan_outcomes(F, lhc):
            pops[f"{s}/{oid}"] = np.flatnonzero(m)

    def overlap(a, b):
        """(Jaccard, containment, chance containment) on the shared population."""
        common = np.intersect1d(pops[a], pops[b])
        A = np.intersect1d(mem[_mem_key(a)], common)
        B = np.intersect1d(mem[_mem_key(b)], common)
        inter = np.intersect1d(A, B).size
        small, big = (A, B) if A.size <= B.size else (B, A)
        return (jaccard(A, B),
                inter / max(small.size, 1),
                big.size / max(common.size, 1))

    print("\n  pairwise overlap of the in-box PLAN sets, on each pair's shared population")
    M = np.ones((len(ids), len(ids)))
    for a in range(len(ids)):
        for b_ in range(a + 1, len(ids)):
            M[a, b_] = M[b_, a] = overlap(ids[a], ids[b_])[0]
    w = max(len(i) for i in ids) + 2
    print(" " * w + "".join(f"{i:>22}" for i in ids))
    for a, i in enumerate(ids):
        print(f"{i:<{w}}" + "".join(f"{M[a, b_]:>22.2f}" for b_ in range(len(ids))))

    # Jaccard is capped by the mass gap between two boxes - a 13%-mass box nested inside
    # a 40%-mass one reads 0.33 at PERFECT agreement - so the verdicts below are read off
    # CONTAINMENT (what share of the smaller box the larger one holds) against the
    # containment a box of that size would get by chance, which is the larger box's own
    # mass. PLAN_OVERLAP_MIN still applies to the Jaccard where the masses are alike.
    print("\n  the two comparisons that matter, read off containment vs chance")
    verdicts = {}
    for s in ("core", "full"):
        for a, b, high, low in (
                ("robust", "within_cluster",
                 f"the archive-wide box survives with the {CLUSTER_LABEL} effect "
                 "removed, so it is a map-shape result",
                 "the archive-wide box is largely carried by provenance - state the "
                 "plan-shape result as the within_cluster one"),
                ("fragile", "lhc",
                 "the 61 discrete corners stand in for the sampled space",
                 "the discrete corners and the sampled ensemble pick out different "
                 "fragile plans")):
            ka, kb = f"{s}/{a}", f"{s}/{b}"
            if ka not in ids or kb not in ids:
                continue
            j, cont, chance = overlap(ka, kb)
            verdicts[f"{s}: {a} vs {b}"] = {"jaccard": j, "containment": cont,
                                            "chance_containment": chance}
            print(f"    [{s}] {a} vs {b}: Jaccard {j:.2f}, containment {cont:.2f} "
                  f"vs {chance:.2f} by chance ({cont / max(chance, 1e-9):.1f}x) -> "
                  + (high if cont > 2 * chance else low))

    print("\n  supported features, by outcome (bootstrap support in brackets)")
    bind = []
    for oid in ids:
        sub = dims[(dims.outcome == oid) & dims.supported].sort_values(
            "boot_freq", ascending=False)
        print(f"    {oid:<20} " + ("  ".join(
            f"{r.dimension}[{r.boot_freq:.2f}]" for r in sub.itertuples())
            or "none supported"))
        bind.append({"outcome": oid, "features": list(sub["dimension"]),
                     "boot_freq": [float(v) for v in sub["boot_freq"]]})

    _plan_compare_figures(boxes, dims, mem, F, ids)
    report_update("plan_compare", {
        "credible": ids, "excluded": dropped,
        "overlap_matrix": M.tolist(), "overlap_labels": ids,
        "verdicts": verdicts, "binding_features": bind,
    })


def _plan_compare_figures(boxes, dims, mem, F, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure(FIG_DIR)

    # _fig_boxes/_fig_heatmap group on a `plan_id` column; here the group is an outcome,
    # so the names are carried through their label_of hook rather than in the codes.
    code = {o: i for i, o in enumerate(ids)}
    d = dims[dims.outcome.isin(ids)].copy()
    d["plan_id"] = d["outcome"].map(code)
    base = dict(zip(boxes.outcome, boxes.base_rate))
    order = list(boxes[boxes.credible].sort_values("lift", ascending=False)["outcome"])
    order_codes = [code[o] for o in order]
    name = {v: k for k, v in code.items()}
    fail_of = {code[o]: base[o] for o in ids}

    # Bounds are normalised against the ARCHIVE-wide feature range so the outcomes are
    # on one scale; the `lhc` box was peeled inside its subsample's own range, which the
    # stratified draw keeps close to it but does not make identical.
    cols = sorted(set(dims["dimension"]))
    lo = F[cols].min().to_numpy()
    hi = F[cols].max().to_numpy()
    rng = np.where(hi > lo, hi - lo, 1.0)
    _fig_boxes(d, order_codes, order_codes, fail_of, cols, lo, rng,
               fname="plan_boxes.png",
               title="Robustness boxes over plan characteristics",
               label_of=lambda c: f"{name[c]}\n{fail_of[c]:.0%} base")
    _fig_heatmap(d, order_codes, order_codes, fail_of, cols, lo, hi,
                 fname="plan_feature_heatmap.png",
                 title="Which plan characteristics bind, across outcomes",
                 label_of=lambda c: f"{name[c]} ({fail_of[c]:.0%})")

    # The marginal decomposition next to the box: a bar whose raw and between-cluster
    # heights match and whose within-cluster height is ~0 is a provenance effect.
    marg = pd.read_csv(PLAN_MARGINALS_CSV).sort_values(
        "spearman_raw", key=np.abs, ascending=False)
    y = np.arange(len(marg))
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(marg) + 1.6))
    for off, col, lab, c in (
            (-0.27, "spearman_raw", "raw", "tab:blue"),
            (0.0, "spearman_within_cluster", f"within {CLUSTER_LABEL}", "tab:green"),
            (0.27, "spearman_between_cluster", f"between {CLUSTER_LABEL}", "tab:orange")):
        ax.barh(y + off, marg[col], height=0.26, color=c, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(marg["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Spearman with max_regret")
    ax.set_title(f"Feature association, split by the {CLUSTER_LABEL} confound")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plan_feature_marginals.png", dpi=140)
    plt.close(fig)

    _fig_box_maps(mem, F, ids)
    print(f"\n  figures -> {FIG_DIR}")


def _selection_frequency(plans, idx, n_cells, chunk=500):
    """Per-cell selection frequency, accumulated in chunks.

    uncertainty_analysis._frequency concatenates every plan first, which is 2.4 GB for
    the out-of-box complement here (14k plans x 22k cells); the counts are the same.
    """
    counts = np.zeros(n_cells, np.int64)
    for s in range(0, len(idx), chunk):
        block = np.concatenate([plans[i] for i in idx[s:s + chunk]]).astype(np.int64)
        counts += np.bincount(block, minlength=n_cells)
    return counts.astype(float) / max(len(idx), 1)


def _fig_box_maps(mem, F, ids):
    """What the box looks like on the ground: in-box vs out-of-box selection frequency."""
    import matplotlib.pyplot as plt
    A = load_archive()
    L = load_layers([BASELINE_TAG])
    shape = L["shape"]
    n_cells = int(np.prod(shape))
    pick = [o for o in ids if o.endswith("/robust")] or ids[:1]
    oid = pick[0]
    inbox = F["plan_id"].to_numpy()[mem[_mem_key(oid)]]
    outbox = np.setdiff1d(F["plan_id"].to_numpy(), inbox)
    f_in = _selection_frequency(A["plans"], inbox, n_cells).reshape(shape)
    f_out = _selection_frequency(A["plans"], outbox, n_cells).reshape(shape)

    # The frequency panels keep an absolute 0-1 scale - that is the quantity. The
    # difference panel is scaled to its own observed extreme instead: fixing it at +/-1
    # leaves the colourbar's outer half empty and the real gap reads as no gap at all.
    dif = f_in - f_out
    dlim = float(np.abs(dif).max())
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, img, ttl, cm, lim in (
            (axs[0], f_in, f"in box ({inbox.size} plans)", "Greens", (0, 1)),
            (axs[1], f_out, f"outside ({outbox.size} plans)", "Greens", (0, 1)),
            (axs[2], dif, f"difference (+/-{dlim:.2f})", "RdBu_r", (-dlim, dlim))):
        im = ax.imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"Selection frequency, {oid} box", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plan_box_maps.png", dpi=140)
    plt.close(fig)


# ===========================================================================
# main
# ===========================================================================
STAGES = {"gate": cmd_gate, "select": cmd_select, "design": cmd_design,
          "score": cmd_score, "fail": cmd_fail, "prim": cmd_prim,
          "compare": cmd_compare, "features": cmd_features, "plan_lhc": cmd_plan_lhc,
          "plan_prim": cmd_plan_prim, "plan_compare": cmd_plan_compare}


def cmd_all(argv=()):
    for name, fn in STAGES.items():
        print(f"\n{'=' * 74}\n{name.upper()}\n{'=' * 74}")
        fn(argv)


def main(argv):
    if not argv or argv[0] not in {**STAGES, "all": cmd_all}:
        raise SystemExit(f"usage: python -m Core_optimisation.prim "
                         f"<{'|'.join(list(STAGES) + ['all'])}>")
    ({**STAGES, "all": cmd_all}[argv[0]])(argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
