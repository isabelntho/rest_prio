"""Section 3.6 - scenario (LULC + climate) uncertainty as archive-wide regret.

`scenario_eligibility.py` reports only class PERSISTENCE (how much of a plan stays on
restorable land). This module re-scores every pooled plan's restoration BENEFIT under
each Bern Landscape Change Explorer projection and applies the same regret logic as
`uncertainty_analysis.py` (section 3.5): a cost-matched benefit shortfall against the
best plan under that scenario, then a bottom-25%-worst-case robust subset.

Uncertainty set: the 225 scenario combinations (3 climate x 5 economic x 3 demographic
x 5 land-use) at 3 horizons (2040, 2050, 2060) = 675 scenario-horizons.

Benefit metric: DIRECT term only - b_s(plan) = -sum d_ref over the plan's cells that
still carry a restorable class under the scenario, with d_ref the `global_all` variant
layer. The engine's spillover term (decay 0.2, radius 3) is a near-constant fraction of
the direct term across plans, so it does not move plan ranks, the robust subset, or the
discriminate-vs-saturate verdict; a full-spillover recomputation (a dilation per
plan x scenario-horizon, ~10M) is a known more complete option, not done here. Cost is
the plan's native cost (a fixed commitment made on 2018 information).

  pixi run python -m Core_optimisation.scenario_regret <stage>

  masks   Read the 675 horizon rasters, build a (675, n_eligible) survive mask.
  score   Direct benefit of every plan under every scenario-horizon.
  regret  Per scenario-horizon front + regret; worst-case; saturation pre-check;
          robust subset; overlap with the model-uncertainty-robust subset (3.5).
  all     the above in order
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
import pandas as pd

from Core_optimisation.paths import OUTPUTS, ensure
from Core_optimisation.archive.regret_common import (
    model_robust_mask, overlap_stats, write_report,
)
from Core_optimisation.uncertainty_analysis import (
    load_archive, load_layers, front_reference, regret_against, ROBUST_QUANTILE,
)

# ===========================================================================
# configuration
# ===========================================================================
# Same share as scenario_eligibility.py; kept local so this module has no Debugs_tests
# import. Filenames are lulc-{year}-{climate}-{economic}-{demographic}-{landuse}-canton.tif.
RASTER_DIR = r"Y:\CH_Kanton_Bern\03_Workspaces\05_Web_platform\raster_data"
HORIZONS = [2040, 2050, 2060]
CLIMATES = ["rcp26", "rcp45", "rcp85"]
ECONOMIC = ["combined_urban", "ecolo_central", "ecolo_urban", "ref_central", "ref_peri_urban"]
DEMOGRAPHIC = ["low", "ref", "high"]
LANDUSE = ["bau", "ei_cul", "ei_nat", "ei_soc", "gr_ex"]
ELIGIBLE_CODES = [12, 13, 15, 16, 17]      # data_loader ECOSYSTEM_TYPES, ecosystem='all'
NODATA_U32 = 4294967295

D_REF_TAG = "global_all"                   # reference condition layer for the benefit
# Minimum IQR (p75 - p25) of worst-case scenario regret for the leg to "discriminate"
# plans rather than erode every plan by a scenario-wide constant.
SATURATION_EPS = 0.02
# Null for the selection-frequency correlation: how alike two random same-size subsets
# of this archive already look, since every plan is grown on the same landscape.
SPATIAL_NULL_DRAWS = 200
SPATIAL_NULL_SEED = 20260909

UNC_DIR = OUTPUTS / "uncertainty"
OUT_DIR = OUTPUTS / "scenario_regret"
REPORT_JSON = OUT_DIR / "report.json"
MASKS_NPZ = OUT_DIR / "scenario_masks.npz"
BENEFIT_NPZ = OUT_DIR / "scenario_benefit.npz"
INDEX_CSV = OUT_DIR / "scenario_index.csv"
ROBUST_IDS = OUT_DIR / "robust_plan_ids.npy"


def report_update(stage, payload):
    """Merge one stage's headline numbers into report.json under its own key."""
    write_report(REPORT_JSON, stage, payload, {
        "horizons": list(HORIZONS), "n_scenarios": len(_combos()),
        "n_scenario_horizons": len(_combos()) * len(HORIZONS),
        "d_ref_tag": D_REF_TAG, "benefit_metric": "direct_only",
        "benefit_metric_note": ("spillover term omitted; a full-model recomputation "
                                "is a known more complete option, not done here"),
        "robust_quantile": ROBUST_QUANTILE, "saturation_eps": SATURATION_EPS,
        "eligible_codes": list(ELIGIBLE_CODES),
    })


def _combos():
    """The 225 (landuse, climate, economic, demographic) tuples, in reading order."""
    return [(lu, cl, ec, dm) for lu in LANDUSE for cl in CLIMATES
            for ec in ECONOMIC for dm in DEMOGRAPHIC]


def _scenario_path(year, climate, economic, demographic, landuse):
    return os.path.join(
        RASTER_DIR, f"lulc-{year}-{climate}-{economic}-{demographic}-{landuse}-canton.tif")


def _gather_index(src, model_transform, elig, model_w):
    """Flat index into `src`'s flattened array, one per eligible model cell.

    Every raster is on the same 100 m EPSG:2056 lattice, so the map is a constant
    integer (row, col) offset, derived from the transforms and asserted.
    """
    import rasterio

    er, ec = np.divmod(elig, model_w)
    xs, ys = rasterio.transform.xy(model_transform, er, ec)
    sr, sc = rasterio.transform.rowcol(src.transform, xs, ys)
    sr, sc = np.asarray(sr), np.asarray(sc)
    assert src.crs is not None and src.crs.to_epsg() == 2056, f"not EPSG:2056: {src.crs}"
    assert src.res == (100.0, 100.0), f"not 100 m: {src.res}"
    assert np.ptp(sr - er) == 0 and np.ptp(sc - ec) == 0, "grids are not co-registered"
    return (sr.astype(np.int64) * src.width + sc).astype(np.int64)


# ===========================================================================
# stage: masks
# ===========================================================================
def cmd_masks(argv=()):
    """One (675, n_eligible) boolean survive mask over the model-eligible footprint."""
    import rasterio

    with rasterio.open(UNC_DIR / "freq_full.tif") as src:
        elig = np.flatnonzero(~np.isnan(src.read(1).ravel()))
        model_transform, model_w = src.transform, src.width
    print(f"model-eligible cells: {elig.size}")

    probe = _scenario_path(HORIZONS[0], CLIMATES[0], ECONOMIC[0], DEMOGRAPHIC[0], LANDUSE[0])
    with rasterio.open(probe) as src:
        gather = _gather_index(src, model_transform, elig, model_w)
        valid = src.read(1).ravel()[gather] != NODATA_U32
    print(f"  nodata at eligible cells: {int((~valid).sum())} (asserted constant below)")

    combos = _combos()
    n_var = len(combos) * len(HORIZONS)
    survive = np.zeros((n_var, elig.size), dtype=bool)
    rows = []
    t0 = time.perf_counter()
    k = 0
    for (landuse, climate, economic, demographic) in combos:
        for h in HORIZONS:
            with rasterio.open(_scenario_path(h, climate, economic, demographic, landuse)) as src:
                v = src.read(1).ravel()[gather]
            assert np.array_equal(v != NODATA_U32, valid), "nodata mask varies across the set"
            survive[k] = valid & np.isin(v, ELIGIBLE_CODES)
            rows.append(dict(k=k, horizon=h, landuse=landuse, climate=climate,
                             economic=economic, demographic=demographic,
                             n_survive=int(survive[k].sum()),
                             frac_lost=float(1.0 - survive[k].sum() / elig.size)))
            k += 1
        if k % 60 == 0 or k == n_var:
            el = time.perf_counter() - t0
            print(f"  [{k:3d}/{n_var}] {el:5.0f}s elapsed, "
                  f"{el / k * (n_var - k):5.0f}s left")

    ensure(OUT_DIR)
    np.savez(MASKS_NPZ, survive=np.packbits(survive, axis=1), elig=elig.astype(np.int64),
             n_elig=np.int64(elig.size))
    idx = pd.DataFrame(rows)
    idx.to_csv(INDEX_CSV, index=False)
    print(f"  -> {MASKS_NPZ}\n  -> {INDEX_CSV}")
    report_update("masks", {
        "n_eligible": int(elig.size), "n_scenario_horizons": n_var,
        "frac_lost": {"median": float(idx["frac_lost"].median()),
                      "min": float(idx["frac_lost"].min()),
                      "max": float(idx["frac_lost"].max())},
        "frac_lost_by_horizon": {int(h): float(idx.loc[idx["horizon"] == h, "frac_lost"].median())
                                 for h in HORIZONS},
    })


# ===========================================================================
# stage: score
# ===========================================================================
def _load_masks():
    z = np.load(MASKS_NPZ, allow_pickle=True)
    n_elig = int(z["n_elig"])
    survive = np.unpackbits(z["survive"], axis=1, count=n_elig).astype(bool)
    return z["elig"].astype(np.int64), n_elig, survive


def cmd_score(argv=()):
    """Direct benefit of every archive plan under every scenario-horizon."""
    A = load_archive()
    L = load_layers(tags=[D_REF_TAG])
    d_ref = np.asarray(L["d"][D_REF_TAG], np.float64).ravel()   # shards store 2-D rasters
    elig, n_elig, survive = _load_masks()
    dropped = ~survive                                     # (n_var, n_elig)
    d_ref_e = d_ref[elig]                                  # (n_elig,)

    pos = np.full(d_ref.size, -1, np.int64)
    pos[elig] = np.arange(n_elig)
    n_plans, n_var = len(A["plans"]), survive.shape[0]
    # (n_elig, n_var) C-contiguous: a plan then gathers CONTIGUOUS rows (its cells) and
    # one gemv gives its lost benefit under all 675 scenario-horizons at once.
    dropped_T = np.ascontiguousarray(dropped.T.astype(np.float32))

    b_base = np.zeros(n_plans)                             # direct benefit, engine sign (<= 0)
    lost = np.zeros((n_plans, n_var))
    t0 = time.perf_counter()
    for i, pl in enumerate(A["plans"]):
        c = pos[pl]
        assert c.min() >= 0, f"plan {i} selects a cell outside the eligible footprint"
        dref_c = d_ref_e[c]
        b_base[i] = -dref_c.sum()
        lost[i] = dref_c @ dropped_T[c]                    # sum d_ref over dropped plan cells
        if (i + 1) % 4000 == 0 or i + 1 == n_plans:
            el = time.perf_counter() - t0
            print(f"    {i + 1}/{n_plans} plans ({el:.0f}s, eta "
                  f"{el / (i + 1) * (n_plans - i - 1):.0f}s)")

    b_sh = b_base[:, None] + lost                          # scenario benefit, engine sign
    cost = np.asarray(A["native_cost"], float)

    # regression checks
    hand = -d_ref[A["plans"][0]].sum()
    assert abs(hand - b_base[0]) <= 1e-6 * abs(hand), (hand, b_base[0])
    assert (b_sh >= b_base[:, None] - 1e-9).all(), "scenario benefit better than baseline"

    ensure(OUT_DIR)
    np.savez(BENEFIT_NPZ, b_sh=b_sh.astype(np.float32), b_base=b_base.astype(np.float32),
             cost=cost)
    print(f"  -> {BENEFIT_NPZ}")
    loss_frac = 1.0 - b_sh / b_base[:, None]               # share of direct benefit lost
    report_update("score", {
        "n_plans": n_plans, "n_scenario_horizons": n_var,
        "loss_frac_overall_median": float(np.median(loss_frac)),
        "loss_frac_worst_scenario_median": float(np.median(loss_frac.max(axis=1))),
        "hand_check_rel_err": float(abs(hand - b_base[0]) / abs(hand)),
    })


# ===========================================================================
# stage: regret
# ===========================================================================
def _incidence(plans, pos, n_elig):
    """Sparse (n_plans, n_elig) 0/1 matrix: does plan i select eligible cell j.

    A subset's per-cell selection frequency is then one row-slice and mean, which makes
    the random-subset null cheap.
    """
    from scipy import sparse

    indptr = np.zeros(len(plans) + 1, np.int64)
    indptr[1:] = np.cumsum([p.size for p in plans])
    cols = np.concatenate([pos[p] for p in plans])
    assert cols.min() >= 0, "a plan selects a cell outside the eligible footprint"
    return sparse.csr_matrix((np.ones(cols.size, np.float32), cols, indptr),
                             shape=(len(plans), n_elig))


def _subset_frequency(X, mask):
    """Per-eligible-cell selection frequency over the plans flagged by `mask`."""
    return np.asarray(X[mask].mean(axis=0)).ravel()


def cmd_regret(argv=()):
    """Per scenario-horizon regret, the saturation pre-check, and the robust subset."""
    z = np.load(BENEFIT_NPZ, allow_pickle=True)
    b_sh = z["b_sh"].astype(np.float64)
    b_base = z["b_base"].astype(np.float64)
    cost = z["cost"].astype(np.float64)
    idx = pd.read_csv(INDEX_CSV)
    n_plans, n_var = b_sh.shape

    R = np.zeros((n_plans, n_var))
    for s in range(n_var):
        R[:, s], _, _ = regret_against(front_reference(b_sh[:, s], cost), b_sh[:, s], cost)
    smax = R.max(axis=1)
    smean = R.mean(axis=1)

    # --- saturation pre-check ------------------------------------------------
    loss_frac = 1.0 - b_sh / b_base[:, None]
    per_scen_iqr = np.percentile(loss_frac, 75, axis=0) - np.percentile(loss_frac, 25, axis=0)
    grand = float(loss_frac.mean())
    ss_scen = n_plans * float(((loss_frac.mean(axis=0) - grand) ** 2).sum())
    ss_plan = n_var * float(((loss_frac.mean(axis=1) - grand) ** 2).sum())
    ss_tot = float(((loss_frac - grand) ** 2).sum())
    smax_iqr = float(np.percentile(smax, 75) - np.percentile(smax, 25))
    verdict = "discriminating" if smax_iqr > SATURATION_EPS else "plan-invariant"

    horizon = idx["horizon"].to_numpy()
    per_h = {}
    for h in HORIZONS:
        cols = np.flatnonzero(horizon == h)
        hmax = R[:, cols].max(axis=1)
        per_h[int(h)] = {"worst_regret_median": float(np.median(hmax)),
                         "worst_regret_p90": float(np.quantile(hmax, 0.9)),
                         "loss_frac_median": float(np.median(loss_frac[:, cols]))}

    cutoff = float(np.quantile(smax, ROBUST_QUANTILE))
    robust = smax <= cutoff
    ensure(OUT_DIR)
    np.save(ROBUST_IDS, np.flatnonzero(robust).astype(np.int64))

    # --- overlap with the model-uncertainty-robust subset (3.5) ------------
    mrob = model_robust_mask(UNC_DIR / "plan_summary.csv", n_plans)
    ov = overlap_stats(robust, mrob, n_plans)

    # 3.5's own worst-case regret, on the same cost-matched-regret scale, so the qmd can
    # show the two side by side (fig-unc-scenregret).
    mreg = pd.read_csv(UNC_DIR / "plan_summary.csv", usecols=["max_regret"])["max_regret"].to_numpy()
    model_worst = {q: float(np.percentile(mreg, p))
                   for q, p in (("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))}

    # Do the two subsets also put restoration in the same PLACES, not just agree on
    # which plans are robust? Correlate their selection-frequency maps - but read that
    # against a null, because every plan is grown on the same landscape and shares a
    # core, so ANY two large subsets of the archive have similar maps. The null draws
    # SPATIAL_NULL_DRAWS random subsets the size of the scenario-robust set and
    # correlates each against the model-robust map.
    elig, n_elig, _ = _load_masks()
    pos = np.full(int(elig.max()) + 1, -1, np.int64)
    pos[elig] = np.arange(n_elig)
    X = _incidence(load_archive()["plans"], pos, n_elig)
    f_model = _subset_frequency(X, mrob)
    rng = np.random.default_rng(SPATIAL_NULL_SEED)
    k = int(robust.sum())
    null_r = np.array([
        np.corrcoef(f_model, _subset_frequency(X, rng.permutation(n_plans)[:k]))[0, 1]
        for _ in range(SPATIAL_NULL_DRAWS)])
    ov["selection_frequency_pearson_r"] = float(np.corrcoef(f_model, _subset_frequency(X, robust))[0, 1])
    ov["selection_frequency_null_median"] = float(np.median(null_r))
    ov["selection_frequency_null_p95"] = float(np.quantile(null_r, 0.95))

    print(f"  worst-case scenario regret: median {np.median(smax):.4f}, "
          f"IQR {smax_iqr:.4f}, p90 {np.quantile(smax, 0.9):.4f}, max {smax.max():.4f}")
    print(f"  verdict: {verdict} (IQR vs eps {SATURATION_EPS})")
    print(f"  robust subset: {int(robust.sum())}/{n_plans} (cutoff {cutoff:.4f})")
    print(f"  overlap with model-robust: {ov['intersection']} plans, "
          f"Jaccard {ov['jaccard']:.3f}, {ov['ratio_vs_chance']:.2f}x chance")
    print(f"  selection-frequency r {ov['selection_frequency_pearson_r']:.3f} "
          f"(random-subset null median {ov['selection_frequency_null_median']:.3f}, "
          f"p95 {ov['selection_frequency_null_p95']:.3f})")

    report_update("regret", {
        "n_plans": n_plans, "n_scenario_horizons": n_var,
        "verdict": verdict,
        "worst_regret": {"median": float(np.median(smax)), "p25": float(np.percentile(smax, 25)),
                         "p75": float(np.percentile(smax, 75)), "iqr": smax_iqr,
                         "p90": float(np.quantile(smax, 0.9)), "max": float(smax.max())},
        "model_worst_regret": model_worst,
        "mean_regret_median": float(np.median(smean)),
        "loss_frac": {"overall_median": grand,
                      "per_scenario_iqr_median": float(np.median(per_scen_iqr)),
                      "between_scenario_var_share": ss_scen / max(ss_tot, 1e-30),
                      "between_plan_var_share": ss_plan / max(ss_tot, 1e-30)},
        "by_horizon": per_h,
        "robust_quantile": ROBUST_QUANTILE, "cutoff": cutoff, "n_robust": int(robust.sum()),
        "model_robust_overlap": dict(ov, n_scenario_robust=ov["n_a"],
                                     n_model_robust=ov["n_b"]),
    })


# ===========================================================================
# main
# ===========================================================================
STAGES = {"masks": cmd_masks, "score": cmd_score, "regret": cmd_regret}


def cmd_all(argv=()):
    for name, fn in STAGES.items():
        print(f"\n{'=' * 74}\n{name.upper()}\n{'=' * 74}")
        fn(argv)


def main(argv):
    if not argv or argv[0] not in {**STAGES, "all": cmd_all}:
        raise SystemExit("usage: python -m Core_optimisation.scenario_regret "
                         f"<{'|'.join(list(STAGES) + ['all'])}>")
    ({**STAGES, "all": cmd_all}[argv[0]])(argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
