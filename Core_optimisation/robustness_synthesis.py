"""Section 3.8 - does robustness transfer between uncertainty sources?

The three legs each produce an archive-wide, regret-based, bottom-25%-worst-case robust
subset on the same 14,591-plan archive:

  model      uncertainty_analysis.py   plan_summary.csv:is_robust
  scenario   scenario_regret.py        scenario_regret/robust_plan_ids.npy
  decision   allocation_regret.py      allocation_regret/robust_plan_ids_<rule>.npy
                                       (+ the across-rule core = their intersection)

This module reports every pairwise and three-way overlap - size, Jaccard, ratio vs
chance, hypergeometric enrichment p - and the transfer shares a decision-maker cares
about: of the model-robust plans, how many also survive the other sources.

It also characterises the SPATIAL nature of the robust subsets (the `spatial` block):
each plan's own configuration (patch count, radius of gyration), whether the subsets
restore in the same places as the model-robust map (against a random-subset null,
since every plan is region-grown on one landscape and shares a core), what kind of
land the cross-source-robust plans over-select relative to the whole archive, and
whether the cross-source-robust core's land-cover mix departs from a size-matched
frequency-ranked slice of the archive (`spatial.land_cover`).

  pixi run python -m Core_optimisation.robustness_synthesis
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import itertools

import numpy as np
import pandas as pd
from scipy import ndimage

from Core_optimisation.paths import OUTPUTS, ensure, DATA_DIR
from Core_optimisation.archive.regret_common import (
    model_robust_mask, overlap_stats, write_report,
)
from Core_optimisation.uncertainty_analysis import (
    jaccard, load_archive, load_layers, CORE_FREQ,
)
from Core_optimisation.archive.scenario_regret import _incidence, _subset_frequency, _load_masks

UNC_DIR = OUTPUTS / "uncertainty"
SCEN_DIR = OUTPUTS / "scenario_regret"
ALLOC_DIR = OUTPUTS / "allocation_regret"
OUT_DIR = OUTPUTS / "robustness_synthesis"
REPORT_JSON = OUT_DIR / "report.json"
GEOMETRY_NPZ = OUT_DIR / "plan_geometry.npz"

D_REF_TAG = "global_all"
SPATIAL_NULL_DRAWS = 200
SPATIAL_NULL_SEED = 20260909
# the geometry null only resamples a cached per-plan vector, so it can afford more draws
GEOMETRY_NULL_DRAWS = 2000

LULC_PATH = DATA_DIR / "ecosystem_lulc_masked.tif"
# data_loader ECOSYSTEM_TYPES for ecosystem='all'; the reclass mirrors
# Documentation/_plot_functions.R:reclass_lulc so the supplement's model-robust-core
# table and the cross-source-robust table (spatial.land_cover) share one class scheme.
LULC_RECLASS = {12: 0, 13: 0, 15: 1, 16: 2, 17: 2}
LULC_CLASSES = ("Forest", "Agricultural land", "Pastures & grasslands")


def _triple(a, b, c, n):
    inter = int((a & b & c).sum())
    exp = a.sum() * b.sum() * c.sum() / n ** 2
    return {"intersection": inter, "expected_by_chance": float(exp),
            "ratio_vs_chance": inter / max(exp, 1e-9),
            "share_of_a": inter / max(int(a.sum()), 1)}


def _q(v):
    """p25 / median / p75 / p90 of a 1-D array, as plain floats.

    p90 is the upper whisker of the box plots in paper2 (@fig-unc-synth-geometry),
    matching the model- and scenario-regret figures.
    """
    return {"p25": float(np.percentile(v, 25)),
            "median": float(np.percentile(v, 50)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90))}


def _median_null(v, mask, n, rng, draws=GEOMETRY_NULL_DRAWS):
    """Subset median of a per-plan quantity against a random same-size-subset null.

    Without this the geometry leg is a quantile eyeball against the whole archive, which
    cannot say whether a subset median of 3 patches against the archive's 5 is structure
    or the spread a subset that size carries anyway.
    """
    k = int(mask.sum())
    obs = float(np.median(v[mask]))
    null = np.array([np.median(v[rng.permutation(n)[:k]]) for _ in range(draws)])
    # add-one estimator: a permutation p is bounded below by 1/(draws+1), never 0
    p = 2.0 * (min(int((null <= obs).sum()), int((null >= obs).sum())) + 1) / (draws + 1)
    return {"null_median": float(np.median(null)),
            "null_lo": float(np.quantile(null, 0.025)),
            "null_hi": float(np.quantile(null, 0.975)),
            "null_draws": int(draws),
            "p_value": float(min(p, 1.0))}


def _freq_calibration(plans, X, f_ref, f_full, k, shape, rng):
    """Does the whole-map form of the frequency statistic have any range left?

    Both forms are scored on subsets displaced BY CONSTRUCTION - the k plans with the most
    extreme centroids, and the k with the fewest/most patches. On the whole-map form these
    land inside the random-subset null, so an observed r near the null is not evidence that
    a subset restores in typical places; the deviation form is what separates them.
    """
    w = shape[1]
    cx = np.array([(np.asarray(p, np.int64) % w).mean() for p in plans])
    cy = np.array([(np.asarray(p, np.int64) // w).mean() for p in plans])
    d_ref = f_ref - f_full
    out = {}
    for name, order in (("west", np.argsort(cx)), ("east", np.argsort(cx)[::-1]),
                        ("north", np.argsort(cy)), ("south", np.argsort(cy)[::-1])):
        f = _subset_frequency(X, order[:k])
        out[name + "most_centroid"] = {
            "r": float(np.corrcoef(f_ref, f)[0, 1]),
            "deviation_r": float(np.corrcoef(d_ref, f - f_full)[0, 1])}
    null = np.array([np.corrcoef(f_ref, _subset_frequency(X, rng.permutation(len(plans))[:k]))[0, 1]
                     for _ in range(SPATIAL_NULL_DRAWS)])
    out["random_null"] = {"median": float(np.median(null)), "min": float(null.min()),
                          "p95": float(np.quantile(null, 0.95))}
    out["subset_size"] = int(k)
    return out


def _plan_geometry(plans, shape):
    """Per-plan patch count (ndimage.label, 4-connectivity) and radius of gyration (km)."""
    w = shape[1]
    npatch = np.empty(len(plans), np.int32)
    rg_km = np.empty(len(plans), np.float32)
    board = np.zeros(shape, bool)
    for i, p in enumerate(plans):
        idx = np.asarray(p, np.int64)
        board.flat[idx] = True
        npatch[i] = ndimage.label(board)[1]
        board.flat[idx] = False
        r = idx // w
        c = idx % w
        rc = r - r.mean()
        cc = c - c.mean()
        rg_km[i] = np.sqrt((rc * rc + cc * cc).mean()) * 0.1   # 100 m pixels -> km
        if i % 3000 == 0:
            print(f"    geometry {i}/{len(plans)}")
    return npatch, rg_km


def _load_or_build_geometry(plans, shape):
    n = len(plans)
    if GEOMETRY_NPZ.exists():
        z = np.load(GEOMETRY_NPZ)
        if len(z["n_patches"]) == n:
            return z["n_patches"], z["rg_km"]
        print("  (plan_geometry.npz stale, rebuilding)")
    npatch, rg_km = _plan_geometry(plans, shape)
    ensure(OUT_DIR)
    np.savez(GEOMETRY_NPZ, n_patches=npatch, rg_km=rg_km)
    print(f"  geometry -> {GEOMETRY_NPZ}")
    return npatch, rg_km


def _triple_land_cover(X, elig, shape, triple_mask, f_full):
    """Land-cover composition of the cross-source-robust (triple) core vs a size-matched
    slice of the archive ranked by raw selection frequency.

    The same test uncertainty_analysis runs for the model-robust core (paper2 supplement),
    re-pointed at the all-three-sources-robust subset so section 4.6's composition leg is
    triple-scoped. `chisq_vs_matched` near 1 means the triple core adds no land-cover
    character beyond what raw selection frequency already gives at that size.
    """
    import rasterio
    from scipy.stats import chi2_contingency

    with rasterio.open(LULC_PATH) as src:
        codes = src.read(1)
    if codes.shape != tuple(shape):
        raise SystemExit(
            f"{LULC_PATH.name} is {codes.shape}, model grid is {tuple(shape)} - "
            "the two are not co-registered")
    codes = codes.ravel()[elig]
    cls = np.full(codes.shape, -1, np.int8)
    for code, k in LULC_RECLASS.items():
        cls[codes == code] = k

    def counts(m):
        return np.array([int(((cls == k) & m).sum()) for k in range(len(LULC_CLASSES))],
                        float)

    def pct(c):
        s = c.sum()
        return (c / s).tolist() if s else [float("nan")] * len(c)

    c_elig = counts(np.ones(elig.size, bool))
    core = _subset_frequency(X, triple_mask) >= CORE_FREQ
    n_core = int(core.sum())
    out = {"classes": list(LULC_CLASSES), "core_freq": float(CORE_FREQ),
           "n_triple_core": n_core, "eligible_pct": pct(c_elig)}
    if n_core == 0:
        out.update(triple_core_pct=None, matched_full_pct=None,
                   enrichment_vs_eligible=None, chisq_vs_matched=None)
        return out

    matched = np.zeros(elig.size, bool)
    matched[np.argsort(-f_full, kind="stable")[:n_core]] = True
    c_core, c_match = counts(core), counts(matched)
    tbl = np.vstack([c_core, c_match])
    tbl = tbl[:, tbl.sum(axis=0) > 0]
    chi2, p, dof, _ = chi2_contingency(tbl)
    p_core, p_match, p_elig = pct(c_core), pct(c_match), pct(c_elig)
    # With ~17k cells per group even a sub-percentage-point shift in composition is
    # "significant", so the headline is the EFFECT size: Cramer's V (0 = identical
    # composition, 1 = disjoint) and the total-variation distance between the two
    # class-share vectors. p is kept for completeness but is large-N-dominated.
    gap = np.abs(np.array(p_core) - np.array(p_match))
    cramers_v = float(np.sqrt(chi2 / tbl.sum() / (min(tbl.shape) - 1)))
    out.update(
        triple_core_pct=p_core,
        matched_full_pct=p_match,
        enrichment_vs_eligible=[a / b if b else float("nan")
                                for a, b in zip(p_core, p_elig)],
        chisq_vs_matched={"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
                          "cramers_v": cramers_v,
                          "tv_distance": float(0.5 * gap.sum()),
                          "max_class_share_gap": float(gap.max())})
    return out


def _spatial_block(subsets, n):
    """geometry / freq_agreement / concentration for the four robust subsets."""
    arch = load_archive()
    plans = arch["plans"]
    L = load_layers([D_REF_TAG])
    shape = L["shape"]

    # 1. each plan's own configuration, summarised per subset against the whole archive
    npatch, rg_km = _load_or_build_geometry(plans, shape)
    named = {"archive": np.ones(n, bool), "model": subsets["model"],
             "scenario": subsets["scenario"], "decision:core": subsets["decision:core"],
             "triple": subsets["triple"]}
    g_rng = np.random.default_rng(SPATIAL_NULL_SEED)
    geometry = {}
    for k, m in named.items():
        entry = {"n_patches": _q(npatch[m]), "rg_km": _q(rg_km[m])}
        if k != "archive":
            entry["n_patches"].update(_median_null(npatch, m, n, g_rng))
            entry["rg_km"].update(_median_null(rg_km, m, n, g_rng))
        geometry[k] = entry

    # 2. do the subsets restore in the same PLACES as the model-robust map? Correlate
    #    their selection-frequency maps, but read that against a null: any two large
    #    archive subsets already look alike because every plan shares a landscape core.
    #
    #    Reported in two forms. `r` is the whole map, kept as the published statistic;
    #    `deviation_r` correlates f_subset - f_archive instead, so the core every plan is
    #    region-grown onto drops out of both maps and what is left is displacement. The
    #    whole-map form is compressed into ~0.94-0.97 by that shared core (see
    #    `calibration`), which the deviation form is not. `overlap_with_model` is the
    #    caveat on the deviation form: a subset drawn from inside the model-robust set
    #    agrees with its map partly by membership, not by geography.
    elig, n_elig, survive = _load_masks()
    pos = np.full(int(elig.max()) + 1, -1, np.int64)
    pos[elig] = np.arange(n_elig)
    X = _incidence(plans, pos, n_elig)
    f_ref = _subset_frequency(X, subsets["model"])
    f_full = _subset_frequency(X, named["archive"])
    d_ref = f_ref - f_full
    rng = np.random.default_rng(SPATIAL_NULL_SEED)
    freq_agreement = {}
    for k in ("scenario", "decision:core", "triple"):
        mask = subsets[k]
        kk = int(mask.sum())
        # one permutation per draw scores both forms, so `r`'s null is unchanged
        null = np.empty(SPATIAL_NULL_DRAWS)
        null_d = np.empty(SPATIAL_NULL_DRAWS)
        for i in range(SPATIAL_NULL_DRAWS):
            f = _subset_frequency(X, rng.permutation(n)[:kk])
            null[i] = np.corrcoef(f_ref, f)[0, 1]
            null_d[i] = np.corrcoef(d_ref, f - f_full)[0, 1]
        f_k = _subset_frequency(X, mask)
        ov = int((mask & subsets["model"]).sum())
        freq_agreement[k] = {
            "r": float(np.corrcoef(f_ref, f_k)[0, 1]),
            "null_median": float(np.median(null)),
            "null_p95": float(np.quantile(null, 0.95)),
            "deviation_r": float(np.corrcoef(d_ref, f_k - f_full)[0, 1]),
            "deviation_null_median": float(np.median(null_d)),
            "deviation_null_p95": float(np.quantile(null_d, 0.95)),
            "overlap_with_model": ov,
            "overlap_share": ov / max(kk, 1)}

    calibration = _freq_calibration(plans, X, f_ref, f_full,
                                    int(subsets["model"].sum()), shape,
                                    np.random.default_rng(SPATIAL_NULL_SEED))

    # 3. what kind of land the cross-source-robust plans over-select, vs the archive:
    #    the delta-weighted mean of two per-cell layers over cells they select more often.
    layers = {
        "benefit_weight": np.asarray(L["d"][D_REF_TAG], np.float64).ravel()[elig],
        "scenario_survival": survive.mean(axis=0),
    }
    base = {name: float(np.average(v, weights=f_full)) for name, v in layers.items()}
    concentration = {"baseline": base}
    for k in ("decision:core", "triple"):
        w_over = np.clip(_subset_frequency(X, subsets[k]) - f_full, 0.0, None)
        row = {}
        for name, v in layers.items():
            mean_over = float(np.average(v, weights=w_over)) if w_over.sum() else float("nan")
            row[name] = mean_over
            row[name + "_vs_baseline"] = mean_over / base[name] if base[name] else float("nan")
        concentration[k] = row

    # 4. land-cover composition of the cross-source-robust core, against a size-matched
    #    frequency-ranked slice of the archive (the third of section 4.6's three tests).
    land_cover = _triple_land_cover(X, elig, shape, subsets["triple"], f_full)

    return {"geometry": geometry, "freq_agreement": freq_agreement,
            "freq_calibration": calibration,
            "concentration": concentration, "land_cover": land_cover,
            "params": {"spatial_null_draws": SPATIAL_NULL_DRAWS,
                       "geometry_null_draws": GEOMETRY_NULL_DRAWS,
                       "spatial_null_seed": SPATIAL_NULL_SEED, "d_ref_tag": D_REF_TAG}}


def main(argv=()):
    n = len(pd.read_csv(UNC_DIR / "plan_summary.csv", usecols=["plan_id"]))
    subsets = {"model": model_robust_mask(UNC_DIR / "plan_summary.csv", n)}

    sp = SCEN_DIR / "robust_plan_ids.npy"
    if not sp.exists():
        raise SystemExit(f"{sp} missing - run Core_optimisation.scenario_regret all")
    m = np.zeros(n, bool); m[np.load(sp)] = True
    subsets["scenario"] = m

    rules = []
    for f in sorted(ALLOC_DIR.glob("robust_plan_ids_*.npy")):
        r = f.stem.replace("robust_plan_ids_", "")
        v = np.zeros(n, bool); v[np.load(f)] = True
        subsets[f"decision:{r}"] = v
        rules.append(f"decision:{r}")
    if not rules:
        raise SystemExit(f"no robust_plan_ids_*.npy in {ALLOC_DIR} - run allocation_regret all")
    subsets["decision:core"] = np.logical_and.reduce([subsets[r] for r in rules])

    sizes = {k: int(v.sum()) for k, v in subsets.items()}
    pairwise = {f"{a}|{b}": overlap_stats(subsets[a], subsets[b], n)
                for a, b in itertools.combinations(subsets, 2)}

    triples = {}
    for r in rules + ["decision:core"]:
        triples[f"model|scenario|{r}"] = _triple(
            subsets["model"], subsets["scenario"], subsets[r], n)

    # transfer: of the model-robust plans, the share that also clears each other source
    mr = subsets["model"]
    transfer = {k: float((mr & v).sum() / max(int(mr.sum()), 1))
                for k, v in subsets.items() if k != "model"}

    # is the decision-robust core stable across rules, or does the answer move?
    rule_jac = {r: jaccard(np.flatnonzero(subsets["decision:core"]), np.flatnonzero(subsets[r]))
                for r in rules}

    # the plans robust to all three sources at once - named here for the spatial block,
    # deliberately after sizes/pairwise so those tables are unchanged.
    subsets["triple"] = subsets["model"] & subsets["scenario"] & subsets["decision:core"]
    spatial = _spatial_block(subsets, n)

    write_report(REPORT_JSON, "synthesis", {
        "n_plans": n, "subset_sizes": sizes,
        "pairwise": pairwise, "triples": triples,
        "model_transfer_share": transfer,
        "decision_core_vs_rule_jaccard": rule_jac,
        "decision_core_jaccard_min": float(min(rule_jac.values())),
        "spatial": spatial,
    }, {"rules": [r.split(":")[1] for r in rules]})

    print("  subsets: " + ", ".join(f"{k}={v}" for k, v in sizes.items()))
    for k, o in pairwise.items():
        print(f"    {k:<28} inter {o['intersection']:5d}  J {o['jaccard']:.3f}  "
              f"{o['ratio_vs_chance']:.2f}x chance")
    t = triples["model|scenario|decision:core"]
    print(f"  model & scenario & decision-core: {t['intersection']} plans "
          f"({t['ratio_vs_chance']:.2f}x chance, {t['share_of_a']:.1%} of model-robust)")

    g = spatial["geometry"]
    print("  spatial:")
    for k in ("archive", "model", "scenario", "decision:core", "triple"):
        pt, rgk = g[k]["n_patches"], g[k]["rg_km"]
        tail = "" if k == "archive" else (
            f"  (null {pt['null_median']:.1f} p {pt['p_value']:.3f} | "
            f"{rgk['null_median']:.1f} p {rgk['p_value']:.3f})")
        print(f"    {k:<14} patches {pt['median']:.0f}  rg {rgk['median']:.1f} km{tail}")
    for k in ("scenario", "decision:core", "triple"):
        fa = spatial["freq_agreement"][k]
        print(f"    {k:<14} freq-map r {fa['r']:.3f} vs null {fa['null_median']:.3f} "
              f"(p95 {fa['null_p95']:.3f});  deviation r {fa['deviation_r']:.3f} "
              f"vs null {fa['deviation_null_median']:.3f} "
              f"(p95 {fa['deviation_null_p95']:.3f}), "
              f"{fa['overlap_share']:.0%} inside model-robust")
    fc = spatial["freq_calibration"]
    print(f"    calibration: displaced-by-construction subsets score r "
          f"{fc['westmost_centroid']['r']:.3f}/{fc['eastmost_centroid']['r']:.3f} "
          f"(west/east) against a random null of {fc['random_null']['median']:.3f} "
          f"(min {fc['random_null']['min']:.3f})")
    cc = spatial["concentration"]["triple"]
    print(f"    triple over-selects: benefit x{cc['benefit_weight_vs_baseline']:.2f}, "
          f"survival x{cc['scenario_survival_vs_baseline']:.2f}")
    lc = spatial["land_cover"]
    if lc["chisq_vs_matched"] is not None:
        cs = lc["chisq_vs_matched"]
        print(f"    triple land-cover vs matched-full: Cramer V {cs['cramers_v']:.3f}, "
              f"TV dist {cs['tv_distance']:.3f}, p {cs['p_value']:.2g} "
              f"(core {lc['n_triple_core']} cells)")
    else:
        print(f"    triple land-cover: no core at freq >= {lc['core_freq']:.2f}")
    print(f"  -> {REPORT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
