"""Section 3.7 - decision (distributive allocation) uncertainty as archive-wide regret.

Resolving how restoration effort is shared between Bern's municipalities is a value
choice, not a fact. This module takes four distributive rules, and for EVERY pooled
plan works out the pixel set that rule forces in (and the free-allocation pixels it
displaces), re-scores the reallocated plan under the 85 condition variants of section
3.5, and expresses the benefit shortfall as a regret - normalised by the generic
plan-pair benefit gap under the same formulation (`discriminability.csv:front_span`,
the normalisation validated at 278-plan scale in Debugs_tests/equity_vs_generic_gap.py).
A bottom-25%-worst-case robust subset is then defined per rule, matching section 3.5.

Rules (per-municipality target share of a plan's K pixels, water-filled and capped at
each municipality's eligible capacity):
  equal        K / M each
  area         proportional to municipal polygon area
  eligible     proportional to eligible/restorable pixels
  degradation  proportional to restoration need = max(0, -mean condition anomaly)

No GA re-runs. Builds on the frozen section-3.5 caches (archive, layer shards,
cross_raw.npz, discriminability.csv) and the cached municipality raster.

  pixi run python -m Core_optimisation.allocation_regret <stage>

A rule moves ~82% of a plan's pixels, shattering a region-grown plan across ~433
municipalities. Under the FULL model that scattering raises benefit - spillover credits
un-restored neighbours, so a scattered plan outscores a compact one of the same size
(the contiguity price, uncertainty_analysis.cmd_discrim) - and the reallocated plan beats
its own free plan for every plan and every rule. Two ways round it, both computed:
Nor is the plan's own free allocation a valid baseline even without spillover: a
region-grown plan holds low-value cells kept for contiguity, the rule drops exactly those
and refills with each municipality's best, so the rule raises direct benefit too. The
comparison must therefore be against a MATCHED free plan - same size, same dropped cells,
refilled greedily but without the municipal constraint - which is the F vs F_free contrast
validated at 278-plan scale. `direct` runs that contrast without the spillover term and is
the headline; `matched` repeats it under the full model as a check.

  muni     Per-municipality area / eligible / anomaly, and the rule weight vectors.
  direct   Direct-only benefit of the rule plan and its matched free plan. The headline.
  matched  The same pair under the full effect model. Slow; optional, check only.
  regret   Normalised regret, worst-case over variants, robust subset per rule,
           cross-rule stability, overlap with the model- (and scenario-) robust subsets,
           and the `matched` check if it has been run.
  all      the above in order
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
import pandas as pd

from Core_optimisation.paths import OUTPUTS, DATA_DIR, ensure
from Core_optimisation.regret_common import (
    model_robust_mask, overlap_stats, write_report,
)
from Core_optimisation.uncertainty_analysis import (
    load_archive, load_layers, plan_masks, jaccard, ROBUST_QUANTILE,
)

# ===========================================================================
# configuration
# ===========================================================================
RULES = ["equal", "area", "eligible", "degradation"]
D_REF_TAG = "global_all"                   # least-damage ranking for forced-in picks
SHP_DIR = "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps"

UNC_DIR = OUTPUTS / "uncertainty"
OUT_DIR = OUTPUTS / "allocation_regret"
REPORT_JSON = OUT_DIR / "report.json"
MUNI_CACHE = UNC_DIR / "muni_bern_ids.npy"
TARGETS_NPZ = OUT_DIR / "muni_targets.npz"
MUNI_CSV = OUT_DIR / "muni_stats.csv"
DIRECT_NPZ = OUT_DIR / "alloc_direct.npz"
MATCHED_NPZ = OUT_DIR / "alloc_matched.npz"
ANOM_ABIOTIC = DATA_DIR / "anomaly_scenarios" / "abiotic_global_all.tif"
ANOM_BIOTIC = DATA_DIR / "anomaly_scenarios" / "biotic_global_all.tif"


def report_update(stage, payload):
    """Merge one stage's headline numbers into report.json under its own key."""
    write_report(REPORT_JSON, stage, payload,
                 {"rules": list(RULES), "d_ref_tag": D_REF_TAG,
                  "robust_quantile": ROBUST_QUANTILE})


# ===========================================================================
# municipality raster + allocation targets
# ===========================================================================
def build_muni_raster(shape, transform_affine, crs):
    """Bern-canton municipality id per pixel (0 = none). Cached to MUNI_CACHE.

    Copied from Debugs_tests/equity_forced_regret.py so this module has no
    Debugs_tests import; the cache is the one that script already wrote.
    """
    if MUNI_CACHE.exists():
        return np.load(MUNI_CACHE)
    import geopandas as gpd
    from rasterio.features import rasterize

    bern = gpd.read_file(f"{SHP_DIR}/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp")
    bern = bern[bern["NAME"] == "Bern"].copy()
    gdf = gpd.clip(gpd.read_file(f"{SHP_DIR}/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp"), bern)
    gdf = gdf.to_crs(str(crs))
    names = sorted(gdf["NAME"].unique())
    muni = np.zeros(shape, dtype=np.int32)
    for i, nm in enumerate(names, start=1):
        m = rasterize(gdf[gdf["NAME"] == nm].geometry, out_shape=shape,
                      transform=transform_affine, fill=0, default_value=1, dtype=np.uint8)
        muni[m.astype(bool)] = i
    ensure(MUNI_CACHE.parent)
    np.save(MUNI_CACHE, muni)
    return muni


def weighted_target(caps, w, K):
    """Integer per-municipality targets summing to K, target_m propto w_m, capped at caps_m.

    Water-fill a level `lam` so sum(min(caps, lam*w)) = K, floor, then hand the shortfall
    to the municipalities with the largest fractional part (as equity_forced_regret.equal_target).
    """
    caps = caps.astype(float)
    w = np.asarray(w, float)
    w = np.where(w > 0, w, 0.0)
    if not w.any():
        w = np.ones_like(w)
    hi = caps.max() / w[w > 0].min() + 1.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(caps, mid * w).sum() < K:
            lo = mid
        else:
            hi = mid
    t = np.minimum(caps, hi * w)
    ti = np.floor(t).astype(np.int64)
    short = int(K - ti.sum())
    if short > 0:
        head = ti < caps.astype(np.int64)
        order = np.argsort(np.where(head, -(t - ti), np.inf))
        ti[order[:short]] += 1
    return np.clip(ti, 0, caps.astype(np.int64))


def cmd_muni(argv=()):
    """Per-municipality area / eligible-land / degradation and the rule weight vectors."""
    import rasterio
    from rasterio.transform import Affine

    L = load_layers(tags=[D_REF_TAG])
    shape = L["shape"]
    n_cells = int(np.prod(shape))
    transform = Affine(*[float(x) for x in np.asarray(L["transform"], float)[:6]])
    muni = build_muni_raster(shape, transform, L["crs"]).ravel()

    with rasterio.open(UNC_DIR / "freq_full.tif") as src:
        elig_flat = np.flatnonzero(~np.isnan(src.read(1).ravel()))
    with rasterio.open(ANOM_ABIOTIC) as src:
        ab = src.read(1).ravel()
    with rasterio.open(ANOM_BIOTIC) as src:
        bi = src.read(1).ravel()
    anom = 0.5 * (ab.astype(np.float64) + bi.astype(np.float64))   # more negative = more degraded

    E = elig_flat[muni[elig_flat] > 0]
    mu_e = muni[E]
    ids = np.unique(mu_e)
    id_of = np.searchsorted(ids, mu_e)
    area_px = np.bincount(muni[muni > 0], minlength=int(ids.max()) + 1)[ids].astype(np.int64)
    elig_px = np.bincount(id_of, minlength=ids.size).astype(np.int64)
    anom_sum = np.bincount(id_of, weights=anom[E], minlength=ids.size)
    mean_anom = anom_sum / np.maximum(elig_px, 1)

    weights = {"equal": np.ones(ids.size),
               "area": area_px.astype(float),
               "eligible": elig_px.astype(float),
               "degradation": np.maximum(0.0, -mean_anom)}

    # A representative target vector (modal plan size) for reporting only; `score`
    # recomputes it per plan because the pixel budget varies by a few cells.
    K0 = int(np.median(np.diff(np.load(UNC_DIR / "archive.npz")["sel_offsets"])))
    targets0 = {r: weighted_target(elig_px.astype(float), weights[r], K0) for r in RULES}

    ensure(OUT_DIR)
    np.savez(TARGETS_NPZ, ids=ids, E=E.astype(np.int64), mu_e=mu_e.astype(np.int64),
             caps=elig_px, area_px=area_px, mean_anom=mean_anom, K0=np.int64(K0),
             **{f"w_{r}": weights[r] for r in RULES},
             **{f"target0_{r}": targets0[r] for r in RULES})
    pd.DataFrame(dict(muni_id=ids, area_px=area_px, elig_px=elig_px, mean_anom=mean_anom,
                      **{f"target_{r}": targets0[r] for r in RULES})).to_csv(MUNI_CSV, index=False)
    print(f"  {ids.size} municipalities with eligible land; K0 {K0}")
    for r in RULES:
        t = targets0[r]
        print(f"    {r:<12} target median {int(np.median(t)):5d}, "
              f"min {int(t.min()):4d}, max {int(t.max()):5d}, "
              f"municipalities at cap {int((t >= elig_px).sum())}")
    print(f"  -> {TARGETS_NPZ}\n  -> {MUNI_CSV}")
    report_update("muni", {
        "n_municipalities": int(ids.size), "n_eligible_with_muni": int(E.size),
        "budget_pixels": K0,
        "eligible_px": {"median": float(np.median(elig_px)), "min": int(elig_px.min()),
                        "max": int(elig_px.max())},
        "mean_anomaly": {"median": float(np.median(mean_anom)), "min": float(mean_anom.min()),
                         "max": float(mean_anom.max())},
        "target_at_cap": {r: int((targets0[r] >= elig_px).sum()) for r in RULES},
    })


# ===========================================================================
# stage: score
# ===========================================================================
def _muni_index(d_ref):
    """Everything the swap needs about municipalities, built once from the `muni` stage.

    `E` holds the eligible cells that fall in a municipality; every other array here is
    indexed into `E` rather than into the raster.
    """
    z = np.load(TARGETS_NPZ, allow_pickle=True)
    ids, E, mu_e = z["ids"], z["E"].astype(np.int64), z["mu_e"].astype(np.int64)
    dref_E = d_ref[E]
    pos_E = np.full(d_ref.size, -1, np.int64)
    pos_E[E] = np.arange(E.size)
    id_of_E = np.searchsorted(ids, mu_e)
    blocks = [np.flatnonzero(id_of_E == k) for k in range(ids.size)]
    return {
        "ids": ids, "E": E, "pos_E": pos_E, "id_of_E": id_of_E, "dref_E": dref_E,
        "caps": z["caps"].astype(float), "n_cells": d_ref.size, "blocks": blocks,
        # all eligible cells, best d_ref first: where the matched free plan refills from
        "E_desc": np.argsort(-dref_E, kind="stable"),
        # each municipality's eligible cells, best d_ref first: the order forced-in
        # pixels are taken in, so it is worth sorting once rather than per plan.
        "blocks_desc": [b[np.argsort(-dref_E[b], kind="stable")] for b in blocks],
        "weights": {r: z[f"w_{r}"] for r in RULES},
    }


def _swap(M, sel_mask_E, id_of_sel, target):
    """Forced-in (F) and displaced (D) E-index arrays for one plan under one rule target.

    Under-served municipalities take their best unselected cells; over-served ones give
    up their worst held cells, so the rule is met the least damaging way available.
    """
    n_m = np.bincount(id_of_sel, minlength=M["ids"].size)
    F, D = [], []
    for k in np.flatnonzero(np.maximum(0, target - n_m)):
        cand = M["blocks_desc"][k]
        F.append(cand[~sel_mask_E[cand]][:target[k] - n_m[k]])
    for k in np.flatnonzero(np.maximum(0, n_m - target)):
        held = M["blocks"][k][sel_mask_E[M["blocks"][k]]]
        D.append(held[np.argsort(M["dref_E"][held], kind="stable")][:n_m[k] - target[k]])
    return (np.concatenate(F) if F else np.empty(0, np.int64),
            np.concatenate(D) if D else np.empty(0, np.int64))


def _reallocate(plans, M, rule):
    """Every plan rewritten to meet `rule`'s municipal targets.

    Returns (rule plans, matched free plans, forced-in counts). Both keep the same cells
    and add the same number; they differ only in WHERE the additions may come from - the
    rule plan takes each under-served municipality's best, the matched plan the best
    d_ref cells anywhere. Neither may reuse a displaced cell, so both genuinely relocate.
    That is the only fair contrast: measured against the plan's own free allocation the
    rule looks better, because it trades away cells the region-growing kept for
    contiguity rather than for benefit (equity_forced_regret.py makes the same point).

    The two are similar but not identical in spatial form - the best d_ref cells are
    themselves dispersed, though less evenly than one-per-municipality - so the full
    model still flatters whichever is more scattered. `direct` therefore carries the
    headline and `matched` only checks it.
    """
    rule_plans, free_plans = [], []
    n_forced = np.zeros(len(plans), np.int64)
    targets = {}          # depend only on the plan's pixel count, of which there are ~7
    for i, pl in enumerate(plans):
        e = M["pos_E"][pl]
        sel_in_E = e[e >= 0]
        sel_mask_E = np.zeros(M["E"].size, bool)
        sel_mask_E[sel_in_E] = True
        if pl.size not in targets:
            targets[pl.size] = weighted_target(M["caps"], M["weights"][rule], int(pl.size))
        F, D = _swap(M, sel_mask_E, M["id_of_E"][sel_in_E], targets[pl.size])
        drop = np.zeros(M["n_cells"], bool)
        drop[M["E"][D]] = True
        kept = pl[~drop[pl]]
        rule_plans.append(np.concatenate([kept, M["E"][F]]).astype(np.int64))
        free_pick = M["E_desc"][~sel_mask_E[M["E_desc"]]][:F.size]
        free_plans.append(np.concatenate([kept, M["E"][free_pick]]).astype(np.int64))
        n_forced[i] = F.size
    return rule_plans, free_plans, n_forced


def _municipalities_used(plans, M):
    """How many municipalities each free plan actually puts restoration in."""
    n = np.zeros(len(plans), np.int64)
    for i, pl in enumerate(plans):
        e = M["pos_E"][pl]
        n[i] = np.count_nonzero(np.bincount(M["id_of_E"][e[e >= 0]],
                                            minlength=M["ids"].size))
    return n


def _full_benefit(plans, DT, shape, radius, decay):
    """(n_plans, n_tags) negated benefit, identical to uncertainty_analysis.cross_evaluate.

    DT is (n_cells, n_tags) C-contiguous rather than its transpose, so a plan gathers
    CONTIGUOUS rows - about twice as fast for this access pattern. `radius <= 0` gives
    the direct term only (no dilation).
    """
    B = np.zeros((len(plans), DT.shape[1]))
    t0 = time.perf_counter()
    for i, flat in enumerate(plans):
        sel_idx = np.asarray(flat, np.int64)
        B[i] = -DT[sel_idx].sum(axis=0, dtype=np.float64)
        if radius > 0:
            _, nb_m = plan_masks(sel_idx, shape, radius)
            B[i] -= decay * DT[np.flatnonzero(nb_m.ravel())].sum(axis=0, dtype=np.float64)
        if (i + 1) % 3000 == 0:
            el = time.perf_counter() - t0
            print(f"    {i + 1}/{len(plans)} ({el:.0f}s, "
                  f"eta {el / (i + 1) * (len(plans) - i - 1):.0f}s)")
    return B


def _score(argv, spillover):
    """Shared body of `direct` and `matched`: benefit of each rule plan and its matched
    free plan, without (direct) or with (matched) the spillover term."""
    A, L = load_archive(), load_layers()
    tags = list(L["tags"])
    d_ref = np.asarray(L["d"][D_REF_TAG], np.float64).ravel()   # shards store 2-D rasters
    DT = np.ascontiguousarray(
        np.stack([np.asarray(L["d"][t], np.float32).ravel() for t in tags], axis=1))
    M = _muni_index(d_ref)
    plans = A["plans"]
    radius = L["radius"] if spillover else 0

    b_own = _full_benefit(plans, DT, L["shape"], radius, L["decay"])
    b_rule = np.zeros((len(plans), len(RULES), len(tags)))
    b_matched = np.zeros_like(b_rule)
    n_forced = np.zeros((len(plans), len(RULES)), np.int64)
    for ri, r in enumerate(RULES):
        print(f"\n  rule '{r}': reallocating and scoring")
        rule_plans, free_plans, n_forced[:, ri] = _reallocate(plans, M, r)
        b_rule[:, ri, :] = _full_benefit(rule_plans, DT, L["shape"], radius, L["decay"])
        b_matched[:, ri, :] = _full_benefit(free_plans, DT, L["shape"], radius, L["decay"])
        print(f"  {r:<12} worse than OWN free {(b_rule[:, ri] > b_own).mean():6.1%}"
              f"   worse than MATCHED free {(b_rule[:, ri] > b_matched[:, ri]).mean():6.1%}")
    return tags, b_own, b_rule, b_matched, n_forced, _municipalities_used(plans, M), M["ids"].size


def cmd_direct(argv=()):
    """Direct-only benefit of the rule plan and its matched free plan - the headline.

    The plan's OWN free allocation is not a valid baseline even with spillover gone: a
    region-grown plan carries low-value cells kept for contiguity, the rule drops exactly
    those and refills with each municipality's best, so the rule RAISES direct benefit
    (`share_worse_than_own_free` shows how rarely it does not). The MATCHED free plan
    drops the same cells and refills the same number greedily but without the municipal
    constraint - the F vs F_free contrast validated at 278-plan scale.
    """
    tags, b_own, b_rule, b_matched, n_forced, muni_used, n_muni = _score(argv, spillover=False)
    ensure(OUT_DIR)
    np.savez(DIRECT_NPZ, bd_rule=b_rule.astype(np.float32),
             bd_matched=b_matched.astype(np.float32), n_forced=n_forced,
             muni_used_free=muni_used, tags=np.array(tags), rules=np.array(RULES),
             n_municipalities=np.int64(n_muni))
    print(f"  -> {DIRECT_NPZ}")
    report_update("direct", {
        "n_plans": len(b_own), "n_variants": len(tags), "n_municipalities": int(n_muni),
        "muni_used_free_median": float(np.median(muni_used)),
        "muni_zero_free_median": float(n_muni - np.median(muni_used)),
        "n_forced_median": {r: float(np.median(n_forced[:, ri])) for ri, r in enumerate(RULES)},
        "share_worse_than_own_free": {r: float((b_rule[:, ri] > b_own).mean())
                                      for ri, r in enumerate(RULES)},
        "share_worse_than_matched": {r: float((b_rule[:, ri] > b_matched[:, ri]).mean())
                                     for ri, r in enumerate(RULES)},
    })


def cmd_matched(argv=()):
    """The same comparison under the FULL effect model, to check `direct`. Slow."""
    tags, _b_own, b_rule, b_matched, _nf, _mu, _nm = _score(argv, spillover=True)
    ensure(OUT_DIR)
    np.savez(MATCHED_NPZ, b_rule=b_rule.astype(np.float32),
             b_matched=b_matched.astype(np.float32), tags=np.array(tags),
             rules=np.array(RULES))
    print(f"\n  -> {MATCHED_NPZ}")
    report_update("matched", {"n_plans": b_rule.shape[0], "n_variants": len(tags)})


# ===========================================================================
# stage: regret
# ===========================================================================
def cmd_regret(argv=()):
    """Normalised regret, robust subset per rule, cross-rule stability, and transfer."""
    zd = np.load(DIRECT_NPZ, allow_pickle=True)
    tags = [str(t) for t in zd["tags"]]
    rules = [str(r) for r in zd["rules"]]
    n_forced = zd["n_forced"]
    muni_used_free = zd["muni_used_free"]
    n_muni = int(zd["n_municipalities"])
    bd_rule = zd["bd_rule"].astype(np.float64)             # (n_plans, n_rules, n_tags)
    bd_matched = zd["bd_matched"].astype(np.float64)       # the valid baseline
    n_plans = bd_rule.shape[0]

    disc = pd.read_csv(UNC_DIR / "discriminability.csv").set_index("variant")
    span = disc["front_span"].reindex(tags).to_numpy()
    sat = disc["saturated"].reindex(tags).astype(bool).to_numpy()
    live = ~sat
    assert np.isfinite(span[live]).all(), "a live variant has no front_span in discriminability.csv"
    print(f"  {int(live.sum())}/{len(tags)} variants used ({int(sat.sum())} saturated)")

    def worst_case(b_rule, b_base):
        """Worst-case normalised shortfall of each plan x rule over the live variants.

        Benefit is negated (more negative = better), so b_rule > b_base is a shortfall.
        Dividing by the variant's generic plan-pair gap puts every variant's shortfall on
        one scale (equity_vs_generic_gap.py's normalisation, validated at 278-plan scale).
        """
        short = np.maximum(0.0, (b_rule - b_base) / np.abs(b_base))
        r = short / span[None, None, :]
        r[:, :, sat] = np.nan
        return np.nanmax(r, axis=2)                        # (n_plans, n_rules)

    # Headline: direct benefit, rule plan against its contiguity-matched free plan
    # (see cmd_direct on why the plan's own free allocation is not a valid baseline).
    dmax = worst_case(bd_rule, bd_matched)
    mrob = model_robust_mask(UNC_DIR / "plan_summary.csv", n_plans)

    srob_path = OUTPUTS / "scenario_regret" / "robust_plan_ids.npy"
    srob = None
    if srob_path.exists():
        srob = np.zeros(n_plans, bool)
        srob[np.load(srob_path)] = True

    ensure(OUT_DIR)
    robust = {}
    per_rule = {}
    for ri, r in enumerate(rules):
        cutoff = float(np.quantile(dmax[:, ri], ROBUST_QUANTILE))
        rob = dmax[:, ri] <= cutoff
        robust[r] = rob
        np.save(OUT_DIR / f"robust_plan_ids_{r}.npy", np.flatnonzero(rob).astype(np.int64))
        # how many live variants the rule actually costs benefit under, per plan, and
        # any variant under which it never does (the "exception variant" of the 278-plan
        # equal-share result, checked here across the whole archive).
        n_worse = ((bd_rule[:, ri, :] > bd_matched[:, ri, :]) & live[None, :]).sum(axis=1)
        never_worse = ((bd_rule[:, ri, :] <= bd_matched[:, ri, :]) | sat[None, :]).all(axis=0)
        per_rule[r] = {
            "dmax_median": float(np.median(dmax[:, ri])),
            "dmax_p25": float(np.percentile(dmax[:, ri], 25)),
            "dmax_p75": float(np.percentile(dmax[:, ri], 75)),
            "dmax_p90": float(np.quantile(dmax[:, ri], 0.9)),
            "dmax_max": float(dmax[:, ri].max()),
            "cutoff": cutoff, "n_robust": int(rob.sum()),
            "worse_variant_count_median": float(np.median(n_worse)),
            "worse_variant_count_max": int(n_worse.max()),
            "always_better_variants": [tags[j] for j in np.flatnonzero(never_worse)],
            "forced_pixels_median": float(np.median(n_forced[:, ri])),
            "model_robust_overlap": overlap_stats(rob, mrob, n_plans),
        }
        if srob is not None:
            per_rule[r]["scenario_robust_overlap"] = overlap_stats(rob, srob, n_plans)

    # --- cross-rule stability --------------------------------------------------
    core = np.logical_and.reduce([robust[r] for r in rules])
    pair_jac = {}
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            pair_jac[f"{a}|{b}"] = jaccard(np.flatnonzero(robust[a]), np.flatnonzero(robust[b]))
    jvals = list(pair_jac.values())

    synth = {
        "n_core": int(core.sum()),
        "core_vs_model": overlap_stats(core, mrob, n_plans),
        "pairwise_jaccard": pair_jac,
        "pairwise_jaccard_min": float(min(jvals)), "pairwise_jaccard_median": float(np.median(jvals)),
    }
    if srob is not None:
        synth["core_vs_scenario"] = overlap_stats(core, srob, n_plans)

    # --- check: same contrast under the FULL effect model (spillover included), to show
    # the direct-only headline is not an artefact of dropping the spillover term.
    check = None
    if MATCHED_NPZ.exists():
        zm = np.load(MATCHED_NPZ, allow_pickle=True)
        dmax_m = worst_case(zm["b_rule"].astype(np.float64), zm["b_matched"].astype(np.float64))
        rob_m = {r: dmax_m[:, ri] <= np.quantile(dmax_m[:, ri], ROBUST_QUANTILE)
                 for ri, r in enumerate(rules)}
        order_direct = [rules[i] for i in np.argsort(np.median(dmax, axis=0))]
        order_matched = [rules[i] for i in np.argsort(np.median(dmax_m, axis=0))]
        check = {
            "dmax_median": {r: float(np.median(dmax_m[:, ri])) for ri, r in enumerate(rules)},
            "rule_order_direct": order_direct, "rule_order_matched": order_matched,
            "rule_order_agrees": order_direct == order_matched,
            "subset_jaccard": {r: jaccard(np.flatnonzero(robust[r]), np.flatnonzero(rob_m[r]))
                               for r in rules},
        }
        print(f"  CHECK (full model vs contiguity-matched free plan): rule order "
              f"{'AGREES' if check['rule_order_agrees'] else 'DIFFERS'}; subset Jaccard "
              + ", ".join(f"{r}={check['subset_jaccard'][r]:.2f}" for r in rules))
    else:
        print(f"  (no {MATCHED_NPZ.name} - run the `matched` stage for the full-model check)")

    print(f"  robust subset size by rule: "
          + ", ".join(f"{r}={per_rule[r]['n_robust']}" for r in rules))
    print(f"  cross-rule core: {int(core.sum())} plans; pairwise Jaccard "
          f"{min(jvals):.2f}-{max(jvals):.2f} (median {np.median(jvals):.2f})")
    for r in rules:
        o = per_rule[r]["model_robust_overlap"]
        print(f"    {r:<12} dmax median {per_rule[r]['dmax_median']:.4f}  "
              f"robust {per_rule[r]['n_robust']}  model-overlap J={o['jaccard']:.2f} "
              f"({o['ratio_vs_chance']:.2f}x)")

    report_update("regret", {
        "n_plans": n_plans, "n_variants": len(tags), "n_variants_live": int(live.sum()),
        "benefit_metric": "direct_only",
        "rules": rules, "per_rule": per_rule, "cross_rule": synth,
        "full_model_check": check,
        "n_model_robust": int(mrob.sum()),
        "n_scenario_robust": (int(srob.sum()) if srob is not None else None),
        "n_municipalities": n_muni,
        "muni_zero_free_median": float(n_muni - np.median(muni_used_free)),
    })


# ===========================================================================
# main
# ===========================================================================
STAGES = {"muni": cmd_muni, "direct": cmd_direct, "matched": cmd_matched,
          "regret": cmd_regret}


def cmd_all(argv=()):
    for name, fn in STAGES.items():
        print(f"\n{'=' * 74}\n{name.upper()}\n{'=' * 74}")
        fn(argv)


def main(argv):
    if not argv or argv[0] not in {**STAGES, "all": cmd_all}:
        raise SystemExit("usage: python -m Core_optimisation.allocation_regret "
                         f"<{'|'.join(list(STAGES) + ['all'])}>")
    ({**STAGES, "all": cmd_all}[argv[0]])(argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
