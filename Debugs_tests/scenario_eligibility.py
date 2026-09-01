"""Do the robust restoration plans stay on restorable land, out to 2060?

`Core_optimisation/uncertainty_analysis.py` picks robust plans on a decision space frozen
at 2018 land cover (restoration_eligible_mask = LULC codes {12,13,15,16,17} intersected
with non-NaN objectives, data_loader.py:981-999). Land cover is not frozen: the Kanton
Bern web-platform projections move it to 2060 under 225 scenario combinations. This asks
how much of each plan's selected pixel set still carries the land-cover class it was
selected as.

The method, in three steps:

  1. score each pixel of each scenario raster: does it still carry its 2018 class?
  2. reduce that to one number per scenario variant -- the mean, over plans, of the
     fraction of a plan's cells that persist
  3. per land-use scenario, take the mean and SD of that number over the 45 climate x
     economic x demographic variants it is crossed with

Step 2 needs no per-plan loop. The mean over plans of a per-plan fraction is a weighted
sum over cells:

    mean_p( |C_p & OK| / n_p )  ==  OK . w        w[cell] = mean_p( 1{cell in C_p} / n_p )

so each plan group collapses to one 437110-element weight vector, built once, and
scoring a raster is a dot product. `n_p` is the plan's count of non-nodata cells, which
makes the identity exact; that count is constant because the nodata mask is identical
across every raster in the set.

Definitions
-----------
baseline    LULC_2018_agg.tif -- the raster that actually defines eligibility in the
            optimisation (data_loader.py:73), not the projections' own 2020.
strict      a pixel persists while lulc_t == lulc_2018. Any class transition is loss.
            This is the headline metric.
class-set   a pixel persists while lulc_t is in {12,13,15,16,17}. Extra CSV columns; the
            gap to strict separates real loss from reshuffling inside the restorable set.

Grids: model, baseline and scenario rasters share a 100 m EPSG:2056 lattice but differ in
extent, so each is reached by a constant integer (row, col) offset, derived from the
transforms and asserted -- never resampled, never hardcoded.

Usage:  pixi run python Debugs_tests/scenario_eligibility.py         # scan, then plot
        pixi run python Debugs_tests/scenario_eligibility.py plot    # redraw from CSV
"""
import os
import sys
import csv
import time

from _common import use_agg

import numpy as np

from Core_optimisation.paths import OUTPUTS, FIGS_DIR, PROJECT_ROOT, ensure

# ===========================================================================
RASTER_DIR = r"Y:\CH_Kanton_Bern\03_Workspaces\05_Web_platform\raster_data"
BASELINE_PATH = r"W:\EU_BioES_SELINA\WP3\4. Spatially_Explicit_EC\Data\LULC\LULC_2018_agg.tif"

YEARS = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
CLIMATES = ["rcp26", "rcp45", "rcp85"]
ECONOMIC = ["combined_urban", "ecolo_central", "ecolo_urban",
            "ref_central", "ref_peri_urban"]
DEMOGRAPHIC = ["low", "ref", "high"]
LANDUSE = ["bau", "ei_cul", "ei_nat", "ei_soc", "gr_ex"]

BASELINE_YEAR = 2018
ELIGIBLE_CODES = [12, 13, 15, 16, 17]   # data_loader.ECOSYSTEM_TYPES, ecosystem='all'
NODATA_U32 = 4294967295                 # uint32 max; the projections' nodata
CORE_FREQ = 0.50                        # matches uncertainty_analysis.py:263
SAMPLE_SEED = 42

# LULC class names, from Readme-files/Readme_landuse_landcover.txt on the Y: share.
LULC_NAMES = {
    10: "settlement", 11: "static", 12: "open forest", 13: "closed forest",
    14: "shrubland", 15: "intensive agriculture", 16: "alpine pasture",
    17: "grassland", 18: "permanent crop", 19: "glacier", 20: "lake", 21: "river",
}

UNC_DIR = OUTPUTS / "uncertainty"
FIG_DIR = FIGS_DIR / "uncertainty"
OUT_CSV = UNC_DIR / "scenario_eligibility.csv"
BASE2020_CSV = UNC_DIR / "baseline_vs_2020.csv"
TRANSITIONS_CSV = UNC_DIR / "baseline_vs_2020_transitions.csv"
OUT_PNG = FIG_DIR / "scenario_eligibility.png"

# All four get a mean (from weights). Only these two get a per-plan loss distribution --
# `core` is a single cell set, not a set of plans, and the null needs no spread.
GROUPS = ["robust", "nonrobust", "random_null", "core"]
PLAN_GROUPS = ["robust", "nonrobust"]
CSV_FIELDS = ["landuse", "climate", "economic", "demographic", "year", "group",
              "strict_mean", "classset_mean", "lost_p10", "lost_p50", "lost_p90"]


def scenario_path(year, climate, economic, demographic, landuse):
    """Filenames are fully determined by the scenario fields, so they are built here."""
    return os.path.join(
        RASTER_DIR,
        f"lulc-{year}-{climate}-{economic}-{demographic}-{landuse}-canton.tif")


def combos():
    """The 225 (landuse, climate, economic, demographic) tuples, in reading order."""
    return [(lu, cl, ec, dm) for lu in LANDUSE for cl in CLIMATES
            for ec in ECONOMIC for dm in DEMOGRAPHIC]


# ===========================================================================
# 1. setup -- eligible cells, 2018 baseline codes, the gather maps


def gather_index(src, model_transform, elig, model_w):
    """Flat index into `src`'s flattened array, one per eligible model cell.

    Every raster here is on the same 100 m EPSG:2056 lattice, so the mapping is a
    constant integer (row, col) offset. Deriving it from the transforms and asserting it
    is constant is what catches a misaligned input -- otherwise a bad offset shows up as
    a plausible-looking percentage rather than an error.
    """
    import rasterio

    er, ec = np.divmod(elig, model_w)
    xs, ys = rasterio.transform.xy(model_transform, er, ec)
    sr, sc = rasterio.transform.rowcol(src.transform, xs, ys)
    sr, sc = np.asarray(sr), np.asarray(sc)

    assert src.crs is not None and src.crs.to_epsg() == 2056, f"not EPSG:2056: {src.crs}"
    assert src.res == (100.0, 100.0), f"not 100 m: {src.res}"
    assert np.ptp(sr - er) == 0 and np.ptp(sc - ec) == 0, "grids are not co-registered"
    assert 0 <= sr.min() and sr.max() < src.height, "row offset leaves the grid"
    assert 0 <= sc.min() and sc.max() < src.width, "col offset leaves the grid"
    print(f"  {os.path.basename(src.name):24s} {src.height}x{src.width}  "
          f"offset {int(sr[0] - er[0]):+d} row, {int(sc[0] - ec[0]):+d} col")
    return (sr.astype(np.int64) * src.width + sc).astype(np.int64)


def load_setup():
    """Eligible cells, their 2018 class, and the map into scenario-raster space."""
    import rasterio

    with rasterio.open(UNC_DIR / "freq_full.tif") as src:
        elig = np.flatnonzero(~np.isnan(src.read(1).ravel()))
        transform, (h, w) = src.transform, src.shape
    print(f"model grid {h}x{w}; eligible cells {elig.size}")

    with rasterio.open(BASELINE_PATH) as src:
        raw = src.read(1).ravel()[gather_index(src, transform, elig, w)]
    assert np.isfinite(raw).all(), "baseline is NaN at eligible cells -- gather off-grid"
    base = raw.astype(np.int32)
    assert np.array_equal(base, raw), "baseline values are not integral"
    assert np.isin(base, ELIGIBLE_CODES).all(), (
        "eligible cells carry non-restorable baseline codes -- gather misaligned")

    # The alignment proof: this must reproduce the repo's aligned copy exactly.
    with rasterio.open(PROJECT_ROOT / "data" / "ecosystem_lulc_masked.tif") as src:
        masked = src.read(1).ravel()[elig]
    assert np.array_equal(base, masked), "baseline disagrees with ecosystem_lulc_masked"
    codes, counts = np.unique(base, return_counts=True)
    print("  baseline classes: "
          + ", ".join(f"{int(c)}={int(n)}" for c, n in zip(codes, counts))
          + "  (matches ecosystem_lulc_masked.tif exactly)")

    probe = scenario_path(YEARS[0], CLIMATES[0], ECONOMIC[0], DEMOGRAPHIC[0], LANDUSE[0])
    with rasterio.open(probe) as src:
        scen = gather_index(src, transform, elig, w)
    return elig, base, scen


# ===========================================================================
# 2. weights -- one vector per plan group, so step 3 needs no per-plan loop


def load_plans(elig, valid):
    """Read the archive once and return both things step 3 needs.

    `weights[g]`  per-cell weights whose dot product with a persistence mask is the mean
                  over that group's plans: w[cell] = mean_p( 1{cell in C_p} / n_p ), with
                  n_p the plan's count of non-nodata cells. `valid` is constant across
                  the raster set, so n_p is too and the identity is exact.
    `cells`,      the plans of PLAN_GROUPS laid end to end, with segment starts, so a
    `starts`,     per-plan number can be reduced out directly. Needed only for the loss
    `of_group`    DISTRIBUTION across plans, which no weighted sum can give.

    Nothing is cached to disk: this costs ~15 s against a scan measured in tens of
    minutes, and a cache file would only be another thing to invalidate.
    """
    import pandas as pd
    import rasterio

    ps = pd.read_csv(UNC_DIR / "plan_summary.csv",
                     usecols=["plan_id", "is_robust"], dtype=str)
    assert np.array_equal(ps["plan_id"].astype(np.int64).to_numpy(),
                          np.arange(len(ps))), "plan_id is not the archive row order"
    is_robust = (ps["is_robust"] == "True").to_numpy()
    robust = np.flatnonzero(is_robust)
    other = np.flatnonzero(~is_robust)
    rng = np.random.default_rng(SAMPLE_SEED)
    # Size-matched, so the robust/non-robust contrast is not a difference in plan count.
    nonrobust = np.sort(rng.choice(other, size=robust.size, replace=False))
    print(f"plans {len(ps)}; robust {robust.size}; other {other.size}")

    arch = np.load(UNC_DIR / "archive.npz", allow_pickle=True)
    sel_flat, sel_off = arch["sel_flat"], arch["sel_offsets"]
    pos = np.full(int(elig.max()) + 1, -1, dtype=np.int32)
    pos[elig] = np.arange(elig.size, dtype=np.int32)

    def plan_cells(i):
        c = pos[sel_flat[sel_off[i]:sel_off[i + 1]]]
        assert c.min() >= 0, f"plan {i} selects a non-eligible cell"
        # Drop nodata cells up front. They can never persist, so they belong in neither
        # the numerator nor the denominator; leaving them in would also put weight where
        # no persistence can be scored, so the weights would stop summing to 1.
        return c[valid[c]]

    def weights_for(ids):
        w = np.zeros(elig.size)
        for i in ids:
            c = plan_cells(i)
            w[c] += 1.0 / c.size
        return w / len(ids)

    weights = {"robust": weights_for(robust), "nonrobust": weights_for(nonrobust)}
    sizes = np.diff(sel_off)[robust]

    # A null: uniform draws matched on size, so the robust curve can be read against
    # what any plan of that size would score.
    w = np.zeros(elig.size)
    for n in sizes:
        c = rng.choice(elig.size, size=n, replace=False)
        c = c[valid[c]]
        w[c] += 1.0 / c.size
    weights["random_null"] = w / sizes.size

    # The core consensus area, as a single cell set rather than a set of plans.
    with rasterio.open(UNC_DIR / "freq_robust.tif") as src:
        core = np.flatnonzero(src.read(1).ravel()[elig] >= CORE_FREQ)
    core = core[valid[core]]
    w = np.zeros(elig.size)
    w[core] = 1.0 / core.size
    weights["core"] = w
    print(f"core area (freq_robust >= {CORE_FREQ}): {core.size} cells")

    for name, v in weights.items():
        assert abs(v.sum() - 1.0) < 1e-9, f"{name} weights sum to {v.sum()}, not 1"

    # Per-plan index, for the loss distribution only.
    segs = [plan_cells(i) for ids in (robust, nonrobust) for i in ids]
    del sel_flat, arch
    cells = np.concatenate(segs).astype(np.int32)
    starts = np.zeros(len(segs) + 1, dtype=np.int64)
    np.cumsum([s.size for s in segs], out=starts[1:])
    of_group = np.repeat([0, 1], [robust.size, nonrobust.size])
    print(f"per-plan index: {len(segs)} plans, {cells.size} cells "
          f"({cells.nbytes / 1e6:.0f} MB)")
    return weights, cells, starts, of_group


# ===========================================================================
# 3. scan -- loop rasters, score each pixel, reduce to one number per variant


def read_scenario(path, scen_gather):
    """The scenario raster's values at the eligible cells."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).ravel()[scen_gather]


def scan(elig, base, scen_gather):
    """One CSV row per (scenario combination, year, group), plus the 2020 comparison."""
    valid = read_scenario(
        scenario_path(YEARS[0], CLIMATES[0], ECONOMIC[0], DEMOGRAPHIC[0], LANDUSE[0]),
        scen_gather) != NODATA_U32
    print(f"nodata at eligible cells: {int((~valid).sum())} (constant across the set)")

    weights, cells, starts, of_group = load_plans(elig, valid)
    plan_size = np.diff(starts)
    rows, base_rows, trans_rows = [], [], []
    t0 = time.perf_counter()
    todo = combos()

    for i, (landuse, climate, economic, demographic) in enumerate(todo):
        for year in YEARS:
            v = read_scenario(
                scenario_path(year, climate, economic, demographic, landuse),
                scen_gather)
            assert np.array_equal(v != NODATA_U32, valid), (
                f"nodata mask differs at {landuse}/{climate}/{year}; weights assume it "
                "is constant -- rebuild them per raster if this is now expected")

            # step 1: score each pixel against the 2018 raster
            persists = valid & (v == base)
            in_set = valid & np.isin(v, ELIGIBLE_CODES)

            # step 2a: the mean over plans, as a weighted sum over cells
            row = {g: dict(
                landuse=landuse, climate=climate, economic=economic,
                demographic=demographic, year=year, group=g,
                strict_mean=float(persists @ weights[g]),
                classset_mean=float(in_set @ weights[g])) for g in GROUPS}

            # step 2b: the DISTRIBUTION of loss across individual plans, which a
            # weighted sum cannot give -- this is the only reason the per-plan index
            # exists. np.add on bools is logical-or, so force the accumulator dtype.
            kept = np.add.reduceat(persists[cells].view(np.int8), starts[:-1],
                                   dtype=np.int64)
            lost = 1.0 - kept / plan_size
            for gi, g in enumerate(PLAN_GROUPS):
                p10, p50, p90 = np.percentile(lost[of_group == gi], [10, 50, 90])
                row[g].update(lost_p10=float(p10), lost_p50=float(p50),
                              lost_p90=float(p90))
            rows.extend(row[g] for g in GROUPS)

            if year == 2020:
                base_rows.append(dict(
                    landuse=landuse, climate=climate, economic=economic,
                    demographic=demographic,
                    n_eligible=int(v.size), n_nodata=int((~valid).sum()),
                    n_valid=int(valid.sum()),
                    frac_same_code=float(persists.sum() / valid.sum()),
                    frac_still_in_class_set=float(in_set.sum() / valid.sum())))
                changed = valid & (v != base)
                if changed.any():
                    pairs, n = np.unique(
                        np.stack([base[changed], v[changed].astype(np.int32)], 1),
                        axis=0, return_counts=True)
                    trans_rows += [dict(
                        landuse=landuse, climate=climate, economic=economic,
                        demographic=demographic,
                        code_2018=int(a), name_2018=LULC_NAMES.get(int(a), "?"),
                        code_2020=int(b), name_2020=LULC_NAMES.get(int(b), "?"),
                        n_cells=int(c)) for (a, b), c in zip(pairs, n)]

        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.perf_counter() - t0
            print(f"  [{i + 1:3d}/{len(todo)}] {landuse:7s} {climate} {economic:15s} "
                  f"{demographic:4s}  {el:5.0f}s elapsed, "
                  f"{el / (i + 1) * (len(todo) - i - 1):5.0f}s left")
    return rows, base_rows, trans_rows


# ===========================================================================
# 4. write


def write_csv(path, rows, fields=None):
    # `core` and `random_null` carry no loss distribution, so their rows are short;
    # restval leaves those cells empty rather than dropping the columns.
    with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()), restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV ({len(rows)} rows) -> {path}")


def report_baseline_vs_2020(base_rows, trans_rows):
    """The baseline-to-2020 step, separated so the later decline reads as scenario signal."""
    same = np.array([r["frac_same_code"] for r in base_rows])
    nod = np.array([r["n_nodata"] for r in base_rows])
    print(f"\nbaseline({BASELINE_YEAR}) vs 2020 over eligible cells: "
          f"same-code {same.min():.6f}..{same.max():.6f}; nodata {nod.min()}..{nod.max()}")
    if len(set(zip(same.tolist(), nod.tolist()))) == 1:
        print("  identical across all combinations -- 2020 predates scenario divergence")
    agg = {}
    for r in trans_rows:
        k = (r["code_2018"], r["code_2020"])
        agg[k] = agg.get(k, 0) + r["n_cells"]
    print("  transitions (summed over combinations):")
    for (a, b), n in sorted(agg.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {a:2d} {LULC_NAMES.get(a, '?'):22s} -> "
              f"{b:2d} {LULC_NAMES.get(b, '?'):22s} {n}")


# ===========================================================================
# 5. plot


def plot():
    """Panel a: land-use scenarios, mean +/- SD over variants. Panel b: the groups."""
    import pandas as pd

    if not os.path.exists(OUT_CSV):
        print(f"No {OUT_CSV}. Run without arguments first.")
        return
    use_agg()
    import matplotlib.pyplot as plt

    ensure(FIG_DIR)
    df = pd.read_csv(OUT_CSV)
    colors = dict(zip(LANDUSE, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # step 3: per land-use scenario, mean and SD over its 45 variants
    lu = (df[df["group"] == "robust"].groupby(["landuse", "year"])["strict_mean"]
            .agg(["mean", "std", "count"]).reset_index())
    n_var = int(lu["count"].max())

    ax = axes[0]
    for name in LANDUSE:
        s = lu[lu["landuse"] == name].sort_values("year")
        yr, m, sd = (s["year"].to_numpy(), s["mean"].to_numpy() * 100,
                     np.nan_to_num(s["std"].to_numpy()) * 100)
        ax.fill_between(yr, m - sd, m + sd, color=colors[name], alpha=0.18, lw=0)
        ax.plot(yr, m, color=colors[name], lw=1.8, label=name)
    ax.scatter([BASELINE_YEAR], [100.0], s=14, color="0.45", zorder=3)
    ax.set_xlabel("year")
    ax.set_ylabel("selected pixels still eligible (%)")
    ax.legend(fontsize=8, loc="lower left", title="land-use scenario", title_fontsize=8)

    # Panel b: how much of an INDIVIDUAL plan stops being eligible, and how much plans
    # differ from each other. The band is the p10-p90 across plans, averaged over the 45
    # variants -- averaging the percentiles is safe here because the between-variant SD
    # is ~0.12 pp against a several-pp spread across plans.
    ax = axes[1]
    d = df[df["group"].isin(PLAN_GROUPS)]
    agg = (d.groupby(["landuse", "group", "year"])[["lost_p10", "lost_p50", "lost_p90"]]
             .mean().reset_index())
    for name in LANDUSE:
        for g, ls in zip(PLAN_GROUPS, ["-", "--"]):
            s = agg[(agg["landuse"] == name) & (agg["group"] == g)].sort_values("year")
            if s.empty:
                continue
            yr = s["year"].to_numpy()
            ax.plot(yr, s["lost_p50"].to_numpy() * 100, color=colors[name], ls=ls,
                    lw=1.6, label=name if g == "robust" else "_nolegend_")
            if g == "robust":
                ax.fill_between(yr, s["lost_p10"].to_numpy() * 100,
                                s["lost_p90"].to_numpy() * 100,
                                color=colors[name], alpha=0.18, lw=0)
    ax.set_xlabel("year")
    ax.set_ylabel("of a plan's pixels, share no longer eligible (%)")
    ax.set_title("How much of an individual plan stops being valid\n"
                 "(band = p10-p90 across plans; solid robust, dashed non-robust)",
                 fontsize=10)
    handles = [plt.Line2D([], [], color="0.3", ls=ls, lw=1.4, label=g)
               for g, ls in zip(PLAN_GROUPS, ["-", "--"])]
    ax.add_artist(ax.legend(fontsize=8, loc="upper left", title="land-use scenario",
                            title_fontsize=8))
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"  figure -> {OUT_PNG}")


# ===========================================================================


def main(argv):
    if len(argv) > 1 and argv[1] == "plot":
        plot()
        return
    ensure(UNC_DIR)
    elig, base, scen_gather = load_setup()
    rows, base_rows, trans_rows = scan(elig, base, scen_gather)
    write_csv(OUT_CSV, rows, CSV_FIELDS)
    write_csv(BASE2020_CSV, base_rows)
    if trans_rows:
        write_csv(TRANSITIONS_CSV, trans_rows)
    report_baseline_vs_2020(base_rows, trans_rows)
    plot()


if __name__ == "__main__":
    main(sys.argv)
