"""Does indicator weighting actually move the restoration priority set?

The weighting campaign searches 13 levels - w_flat, w_cat and 11 focal vertices - all at
the `global` benchmark, 5 seeds each. That design is one-at-a-time on the weight simplex
and the perturbations are small: for agricultural (n = 8) a focal vertex moves one
indicator from 0.125 to 0.25 while the other seven only fall to 0.107. So an inert
weighting axis is ambiguous between two very different readings:

  (a) the priority map is genuinely robust to how indicators are weighted, or
  (b) the perturbations were too small and too local to move anything.

Only (b) is defensible from 13 vertices, so the axis cannot be written up as robustness
until the simplex is sampled properly. This module settles it WITHOUT new optimisation
runs, by exploiting the fact that weights reach a plan only through the ranking they induce
over pixels: the top-k set under a weight vector is a cheap, faithful screen for whether
any plan could differ. Building one composite is two matmuls, so thousands of weight
vectors cost what one optimisation run costs.

What it does, in order:

  1. VALIDATION GATE. Rebuilds all 13 existing weighting composites from the exported
     per-indicator layers and compares them against the on-disk abiotic_global_w_*.tif.
     Nothing downstream means anything until this passes, so a miss aborts the run.
  2. Indicator correlation matrix over eligible pixels, per ecosystem and pooled. This
     alone largely predicts weight sensitivity: a weighted mean of highly correlated
     indicators is weight-insensitive no matter what weights you choose.
  3. Dirichlet sampling of the weight simplex - uniform (alpha = 1) and concentrated
     (alpha = 5, 20, "roughly equal weights but not exactly"). Per draw: Jaccard overlap
     of the top-k benefit set against flat weights, the same under a cost-adjusted
     ranking, Spearman rank correlation of the benefit layer, and L1 distance from flat.
  4. Breakeven radius: how far from equal weighting you must go before the priority set
     changes materially, and where the existing 13 vertices sit on that curve. That
     placement is what says whether the 65 completed runs probed a meaningful range.

Reads data/anomaly_scenarios/indicators/, which is written by the
GENERATE_INDICATOR_LAYERS block of data/ec_anomalies.r (off by default - set it TRUE,
run once, set it back, and set GENERATE_WEIGHT_SCENARIOS FALSE first so the existing
weighting rasters are not rewritten underneath the completed runs).

    pixi run python Debugs_tests/weight_simplex_screen.py

With `--emit-vectors` it instead writes data/diagnostics/weight_vectors_simplex.csv: the
weight vectors for the EXTENDED weighting campaign, rejection-sampled to sit at chosen L1
distances from flat (the 13 vertices only reach 0.150). That CSV is the handoff to the
three config sites that run the campaign - data/ec_anomalies.r rasterises the vectors,
run_custom_nsga2.py names the tags, uncertainty_analysis.py declares the campaign - and it
carries this screen's PREDICTED overlap per tag, so the campaign tests the screen rather
than merely consuming it. The validation gate runs first either way.

Configuration is the module-level block below; `--emit-vectors` is the only argument.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

from _common import DIAG_DIR, BASE, load_ic, use_agg

from Core_optimisation.paths import DATA_DIR, ensure
# The saturating improvement weight and the set-overlap helper are already validated
# against the search engine in uncertainty_analysis.layers; a second copy here would be a
# second thing to keep in sync.
from Core_optimisation.uncertainty_analysis import anomaly_weight, jaccard
# The R-mirroring composite machinery, shared with Core_optimisation/prim.py.
from Core_optimisation.condition_composite import (
    ABIOTIC, BIOTIC, ECOSYSTEMS, IND_DIR, SCEN_DIR,
    build_indicator_weights, build_matrices, composite, composite_by_ecosystem,
    draw_to_eco_columns, ecosystem_masks, l1_from_flat, load_indicator_stack,
    read_raster, scheme_weight_columns, weight_scheme_names,
)


# ===========================================================================
# configuration
# ===========================================================================
BENCHMARK = "global"

# Dirichlet concentration levels. alpha = 1 is uniform over the whole simplex (every
# weighting anyone could defend, and many nobody would); larger alpha concentrates the
# draws near equal weights, which is the honest model of "we believe roughly equal
# weighting, but not exactly".
ALPHAS = (1.0, 5.0, 20.0)
N_DRAWS = 2000
RANDOM_SEED = 20260826

# Draws are processed in blocks: the full (n_elig x N_DRAWS) composite matrix would be
# ~3.5 GB, one block of this width is ~200 MB.
DRAW_BLOCK = 100

# Pixels sampled for the Spearman statistic, which needs a full argsort per draw.
SPEARMAN_SUBSAMPLE = 20000

# A plan is "materially unchanged" while it keeps this share of the flat-weight top-k.
OVERLAP_TARGET = 0.90
# Breakeven radius is read off the lower tail, not the median: the claim is "no draw at
# this distance changes the map", not "the average draw does not".
BREAKEVEN_PCTL = 5

# -- extended-campaign vector emission (`--emit-vectors`) -------------------
# Target L1 distances from flat for the extended weighting campaign. The 13 existing
# vertices span 0 to 0.150, so these sit beyond them, in the region where the screen's
# dose-response still has a clean gradient (median overlap ~0.62 at 0.20 down to ~0.38
# at 0.50) rather than out in the flat tail.
EMIT_BANDS = (0.20, 0.35, 0.50)
EMIT_DRAWS_PER_BAND = 4
EMIT_BAND_TOL = 0.01
# Accepted draws within a band must be at least this far from each other, so the four
# point in genuinely different DIRECTIONS rather than clustering at the same distance.
# Without it, rejection sampling on distance alone can return four near-identical
# vectors and the band would measure magnitude only.
EMIT_MIN_SEPARATION = 0.20
EMIT_SEED = 20260827
EMIT_CSV = DATA_DIR / "diagnostics" / "weight_vectors_simplex.csv"

# Validation gate: |rebuilt - on_disk| <= VALIDATE_ATOL + VALIDATE_RTOL * |on_disk|.
#
# The ABSOLUTE term is what does the work here, and it is not optional. The composite is
# a signed z-score spanning about [-2.2, 4.1] and crossing zero, so a pure relative
# criterion divides by ~0 on the pixels nearest the benchmark condition and reports a
# 1e-2 "error" on a rebuild that is exact. Measured max absolute error across all 13
# composites is 2.2e-07 - float32 round-off on that range - so ATOL carries ~50x
# headroom while still catching any genuine construction difference, which would show up
# at the scale of the layer itself.
VALIDATE_ATOL = 1e-5
VALIDATE_RTOL = 1e-5

OUT_DIR = ensure(DIAG_DIR)
OUT_CORR = os.path.join(OUT_DIR, "weight_indicator_correlations.csv")
OUT_DRAWS = os.path.join(OUT_DIR, "weight_simplex_overlap.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "weight_simplex_summary.csv")
OUT_JSON = os.path.join(OUT_DIR, "weight_simplex_screen.json")
OUT_PNG = os.path.join(OUT_DIR, "weight_simplex_screen.png")


# ===========================================================================
# inputs
# ===========================================================================
def emitted_vectors():
    """{tag: {indicator: weight}} from the emitted CSV, or {} if it has not been written.

    Lets the validation gate cover the extended campaign's tags on the same footing as
    the 13 rule-based ones: the vector this module sampled must be the vector R
    rasterised, which is the only check on the whole CSV -> raster handoff.
    """
    if not EMIT_CSV.exists():
        return {}
    df = pd.read_csv(EMIT_CSV)
    return {t: dict(zip(g["indicator"], g["weight"]))
            for t, g in df.groupby("tag")}


# ===========================================================================
# composite -> benefit -> priority set
# ===========================================================================
def benefit(C, eff_total):
    """Per-pixel direct benefit under the single_composite convention.

    A weighting scenario writes the SAME composite to both the abiotic and the biotic
    raster, so the engine's abiotic + biotic sum becomes
    (abiotic_effect + biotic_effect) * w(C) rather than eff*w(A) + eff*w(B).
    """
    d = eff_total * anomaly_weight(np.nan_to_num(C, nan=0.0))
    return np.where(np.isfinite(C), d, 0.0)


def top_k(score, k):
    """Sorted indices of the k largest entries (ties broken by index, deterministic)."""
    idx = np.argpartition(score, -k)[-k:]
    return np.sort(idx)


def overlap(a_mask, b_idx, k):
    """Jaccard of two equal-size top-k sets, one already expanded to a boolean mask."""
    inter = int(a_mask[b_idx].sum())
    return inter / float(2 * k - inter)


# ===========================================================================
# stages
# ===========================================================================
def validate(Z, P, eco_of_pixel, codes, ecos, layers, elig_idx):
    """THE GATE: rebuild the 13 on-disk weighting composites and compare.

    A miss means the python side is not reproducing the R construction, so every overlap
    number downstream would be measuring the discrepancy rather than the weighting.
    """
    # Rule-based vertices, then any extended-campaign tags the CSV defines and R has
    # already rasterised. Both are checked the same way.
    targets = [(s, sch, None) for s, sch in weight_scheme_names()]
    flat_col = scheme_weight_columns("flat", codes, ecos, layers)
    for tag, wmap in sorted(emitted_vectors().items()):
        suffix = tag.split(f"{BENCHMARK}_", 1)[-1]
        if any(suffix == s for s, _ in weight_scheme_names()):
            continue                      # flat/cat/focal are covered by the rule path
        w_union = np.array([wmap.get(c, 0.0) for c in codes], float)
        targets.append((suffix, "explicit", draw_to_eco_columns(w_union, flat_col)))

    print(f"\n[1] validation gate: rebuilding {len(targets)} weighting composites")
    rows, worst_abs, n_fail = [], 0.0, 0
    for suffix, scheme, Wcol in targets:
        path = SCEN_DIR / f"abiotic_{BENCHMARK}_{suffix}.tif"
        if not path.exists():
            print(f"    {suffix:<10} SKIPPED - {path.name} not on disk")
            continue
        ref = read_raster(path).ravel()[elig_idx]
        if Wcol is None:
            Wcol = scheme_weight_columns(scheme, codes, ecos, layers)
        got = composite_by_ecosystem(Z, P, eco_of_pixel, Wcol)
        both = np.isfinite(ref) & np.isfinite(got)
        n_ref, n_got = int(np.isfinite(ref).sum()), int(np.isfinite(got).sum())
        err = np.abs(got[both] - ref[both])
        tol = VALIDATE_ATOL + VALIDATE_RTOL * np.abs(ref[both])
        n_bad = int((err > tol).sum())
        max_abs = float(err.max()) if err.size else np.nan
        worst_abs = max(worst_abs, max_abs if np.isfinite(max_abs) else 0.0)
        # Coverage has to match too: a rebuild that is exact on the pixels it produces
        # but produces the wrong SET of pixels is still wrong.
        if n_ref != n_got or n_bad:
            n_fail += 1
        rows.append({"tag": f"{BENCHMARK}_{suffix}", "scheme": scheme,
                     "n_compared": int(both.sum()), "n_ref_valid": n_ref,
                     "n_rebuilt_valid": n_got, "n_outside_tol": n_bad,
                     "max_abs_err": max_abs})
        flag = "OK " if (n_bad == 0 and n_ref == n_got) else "FAIL"
        print(f"    {flag} {suffix:<10} max abs err {max_abs:.2e}  "
              f"{n_bad:,} px outside tol  (ref {n_ref:,} / rebuilt {n_got:,})")
    if not rows:
        raise SystemExit("no weighting rasters found to validate against")
    if n_fail:
        raise SystemExit(
            f"\nVALIDATION FAILED on {n_fail} of {len(rows)} composites "
            f"(worst absolute error {worst_abs:.2e}).\n"
            "The python rebuild does not reproduce the R composites, so no overlap\n"
            "number below would be meaningful. Fix this before reading anything else.")
    print(f"    PASSED - worst absolute error {worst_abs:.2e} over {len(rows)} "
          f"composites (float32 round-off is ~1e-07 on this range)")
    return rows, worst_abs


def correlations(Z, P, codes, eco_of_pixel, ecos):
    """Indicator correlation matrix, per ecosystem and pooled over eligible pixels."""
    print("\n[2] indicator correlations over eligible pixels")
    rows = []
    groups = [("pooled", np.ones(Z.shape[0], bool))]
    groups += [(eco, eco_of_pixel == ei) for ei, eco in enumerate(ecos)]
    for name, sel in groups:
        present = P[sel] > 0
        for i, a in enumerate(codes):
            for j, b in enumerate(codes):
                if j <= i:
                    continue
                both = present[:, i] & present[:, j]
                if both.sum() < 100:
                    continue
                x = Z[sel][both, i].astype(np.float64)
                y = Z[sel][both, j].astype(np.float64)
                if x.std() == 0 or y.std() == 0:
                    continue
                r = float(np.corrcoef(x, y)[0, 1])
                cat = ("within_abiotic" if a in ABIOTIC and b in ABIOTIC else
                       "within_biotic" if a in BIOTIC and b in BIOTIC else "between")
                rows.append({"group": name, "a": a, "b": b, "block": cat,
                             "n": int(both.sum()), "pearson_r": r})
    df = pd.DataFrame(rows)
    for name in df["group"].unique():
        sub = df[df["group"] == name]
        print(f"    {name:<14} mean |r| {sub['pearson_r'].abs().mean():.3f}   " +
              "  ".join(f"{b}: {g['pearson_r'].abs().mean():.3f}"
                        for b, g in sub.groupby("block")))
    return df


def sample_simplex(Z, P, codes, eco_of_pixel, ecos, layers, cost, k, eff_total,
                   eco_share):
    """Dirichlet draws -> overlap against flat, plus the 13 vertices on the same axis."""
    rng = np.random.default_rng(RANDOM_SEED)
    m = len(codes)

    flat_col = scheme_weight_columns("flat", codes, ecos, layers)
    C_flat = composite_by_ecosystem(Z, P, eco_of_pixel, flat_col)
    d_flat = benefit(C_flat, eff_total)
    ref_idx = top_k(d_flat, k)
    ref_mask = np.zeros(Z.shape[0], bool)
    ref_mask[ref_idx] = True
    ref_cost_idx = top_k(d_flat / cost, k)
    ref_cost_mask = np.zeros(Z.shape[0], bool)
    ref_cost_mask[ref_cost_idx] = True

    sub = rng.choice(Z.shape[0], size=min(SPEARMAN_SUBSAMPLE, Z.shape[0]), replace=False)
    flat_rank = np.argsort(np.argsort(d_flat[sub]))

    print(f"\n[3] sampling the simplex: {N_DRAWS} draws x {len(ALPHAS)} alpha level(s)")
    print(f"    top-k set size k = {k:,} of {Z.shape[0]:,} eligible pixels")
    records = []
    for alpha in ALPHAS:
        t0 = time.perf_counter()
        W = rng.dirichlet(np.full(m, alpha), size=N_DRAWS).T.astype(np.float32)
        for s in range(0, N_DRAWS, DRAW_BLOCK):
            block = W[:, s:s + DRAW_BLOCK]
            C = composite(Z, P, block)
            D = benefit(C, eff_total)
            for b in range(block.shape[1]):
                d = D[:, b]
                idx = top_k(d, k)
                idx_cost = top_k(d / cost, k)
                r_draw = np.argsort(np.argsort(d[sub]))
                records.append({
                    "alpha": alpha, "draw": s + b,
                    "l1_from_flat": l1_from_flat(
                        draw_to_eco_columns(block[:, b].astype(np.float64), flat_col),
                        flat_col, eco_share),
                    "overlap": overlap(ref_mask, idx, k),
                    "overlap_cost_adj": overlap(ref_cost_mask, idx_cost, k),
                    "spearman": float(np.corrcoef(flat_rank, r_draw)[0, 1]),
                })
        print(f"    alpha={alpha:<5g} done in {time.perf_counter() - t0:.0f}s")

    draws = pd.DataFrame(records)
    vertices = []
    for suffix, scheme in weight_scheme_names():
        Wcol = scheme_weight_columns(scheme, codes, ecos, layers)
        C = composite_by_ecosystem(Z, P, eco_of_pixel, Wcol)
        d = benefit(C, eff_total)
        idx = top_k(d, k)
        idx_cost = top_k(d / cost, k)
        r_v = np.argsort(np.argsort(d[sub]))
        vertices.append({
            "vertex": suffix, "scheme": scheme,
            "l1_from_flat": l1_from_flat(Wcol, flat_col, eco_share),
            "overlap": overlap(ref_mask, idx, k),
            "overlap_cost_adj": overlap(ref_cost_mask, idx_cost, k),
            "spearman": float(np.corrcoef(flat_rank, r_v)[0, 1]),
        })
    return draws, pd.DataFrame(vertices)


def breakeven(draws, n_bins=12):
    """Largest L1 distance at which the lower tail of overlap still clears the target."""
    d = draws.sort_values("l1_from_flat")
    if d.empty:
        return np.nan, pd.DataFrame()
    edges = np.linspace(d["l1_from_flat"].min(), d["l1_from_flat"].max(), n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = d[(d["l1_from_flat"] >= lo) & (d["l1_from_flat"] <= hi)]
        if len(sel) < 10:
            continue
        rows.append({"l1_lo": lo, "l1_hi": hi, "l1_mid": 0.5 * (lo + hi), "n": len(sel),
                     "overlap_p5": np.percentile(sel["overlap"], BREAKEVEN_PCTL),
                     "overlap_p50": np.median(sel["overlap"])})
    curve = pd.DataFrame(rows)
    if curve.empty:
        return np.nan, curve
    # Contiguous run from the origin only: overlap must hold at every distance up to the
    # radius, not merely recover at some isolated bin further out.
    ok = 0.0
    for _, r in curve.iterrows():
        if r["overlap_p5"] < OVERLAP_TARGET:
            break
        ok = float(r["l1_hi"])
    return ok, curve


# ===========================================================================
# stage: emit the extended campaign's weight vectors
# ===========================================================================
def band_tag(band, draw_i):
    """`global_w_d{band*100}{a-d}` - the grammar the four config sites agree on.

    Lowercase alphanumeric only, because uncertainty_analysis.py parses run labels with
    `global_w_[a-z0-9_]+` and a tag outside that grammar is silently skipped at discovery
    rather than failing loudly.
    """
    return f"{BENCHMARK}_w_d{int(round(band * 100)):02d}{chr(ord('a') + draw_i)}"


def sample_at_distance(band, n_wanted, flat_col, eco_share, rng, max_tries=200000):
    """Rejection-sample uniform Dirichlet draws landing at L1 ~= `band` from flat.

    Returns a list of union weight vectors. Distance is measured with the same
    pixel-share-weighted per-ecosystem metric the screen reports, so a vector's band
    label means the same thing as its position on the breakeven curve.
    """
    m = flat_col.shape[0]
    kept, tries = [], 0
    while len(kept) < n_wanted and tries < max_tries:
        tries += 1
        w = rng.dirichlet(np.ones(m))
        cols = draw_to_eco_columns(w, flat_col)
        d = l1_from_flat(cols, flat_col, eco_share)
        if abs(d - band) > EMIT_BAND_TOL:
            continue
        # Direction check: far enough from every vector already accepted in this band.
        if any(0.5 * np.abs(w - k).sum() < EMIT_MIN_SEPARATION for k in kept):
            continue
        kept.append(w)
    if len(kept) < n_wanted:
        raise SystemExit(
            f"band {band}: only {len(kept)} of {n_wanted} draws found in {tries:,} tries.\n"
            f"Loosen EMIT_BAND_TOL ({EMIT_BAND_TOL}) or EMIT_MIN_SEPARATION "
            f"({EMIT_MIN_SEPARATION}), or move the band nearer the bulk of the simplex.")
    return kept


def cmd_emit_vectors(Z, P, codes, eco_of_pixel, ecos, layers, cost, k, eff_total,
                     eco_share):
    """Write the extended campaign's weight vectors, with the screen's prediction.

    The `pred_overlap` column is the point of doing this here rather than in a throwaway
    script: it records what the cheap screen expects each tag to do BEFORE any run exists,
    so the campaign becomes a test of the screen rather than only a consumer of it.
    """
    rng = np.random.default_rng(EMIT_SEED)

    flat_col = scheme_weight_columns("flat", codes, ecos, layers)
    C_flat = composite_by_ecosystem(Z, P, eco_of_pixel, flat_col)
    d_flat = benefit(C_flat, eff_total)
    ref_mask = np.zeros(Z.shape[0], bool)
    ref_mask[top_k(d_flat, k)] = True
    ref_cost_mask = np.zeros(Z.shape[0], bool)
    ref_cost_mask[top_k(d_flat / cost, k)] = True

    print(f"\n[emit] sampling {EMIT_DRAWS_PER_BAND} draws at each of "
          f"{len(EMIT_BANDS)} band(s): {', '.join(f'{b:.2f}' for b in EMIT_BANDS)}")
    rows = []
    # The reference level. It is re-run in the new batch on purpose: cmd_noise derives
    # each campaign's seed floor from its own reference cell, so the extended campaign
    # needs `flat` present in its own r_inputs folder rather than borrowing the vertex
    # campaign's copy.
    specs = [("flat", 0.0, None)]
    for band in EMIT_BANDS:
        for i, w in enumerate(sample_at_distance(band, EMIT_DRAWS_PER_BAND,
                                                 flat_col, eco_share, rng)):
            specs.append((band_tag(band, i).split("_w_")[1], band, w))

    for suffix, band, w in specs:
        tag = f"{BENCHMARK}_w_{suffix}"
        if w is None:
            Wcol, w_union = flat_col, np.full(len(codes), 1.0 / len(codes))
        else:
            Wcol, w_union = draw_to_eco_columns(w, flat_col), w
        C = composite_by_ecosystem(Z, P, eco_of_pixel, Wcol)
        d = benefit(C, eff_total)
        l1 = l1_from_flat(Wcol, flat_col, eco_share)
        ov = overlap(ref_mask, top_k(d, k), k)
        ov_c = overlap(ref_cost_mask, top_k(d / cost, k), k)
        print(f"    {tag:<20} L1 {l1:.3f}   pred overlap {ov:.3f} "
              f"(cost-adj {ov_c:.3f})")
        for ci, code in enumerate(codes):
            rows.append({"tag": tag, "band": "flat" if w is None else f"{band:.2f}",
                         "draw": suffix[-1] if w is not None else "-",
                         "indicator": code, "weight": float(w_union[ci]),
                         "l1_from_flat": l1, "pred_overlap": ov,
                         "pred_overlap_cost_adj": ov_c})

    df = pd.DataFrame(rows)
    # Union vectors are what R consumes; it restricts and renormalises per ecosystem.
    # The composite is scale-invariant in w, so that restriction is exactly the
    # num/den renormalisation calculate_weighted_composite already performs.
    bad = df.groupby("tag")["weight"].sum().sub(1.0).abs().max()
    if bad > 1e-9:
        raise SystemExit(f"emitted union vectors do not sum to 1 (worst {bad:.2e})")
    ensure(EMIT_CSV.parent)
    df.to_csv(EMIT_CSV, index=False)
    n_tags = df["tag"].nunique()
    print(f"\n    {n_tags} tags x {len(codes)} indicators -> {EMIT_CSV}")
    print(f"    grid will be {n_tags} tags x 5 seeds = {n_tags * 5} runs")
    return df


# ===========================================================================
# figure
# ===========================================================================
def make_figure(corr, draws, vertices, curve, codes, radius):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    ax = axs[0, 0]
    pooled = corr[corr["group"] == "pooled"]
    M = np.full((len(codes), len(codes)), np.nan)
    for _, r in pooled.iterrows():
        i, j = codes.index(r["a"]), codes.index(r["b"])
        M[i, j] = M[j, i] = r["pearson_r"]
    np.fill_diagonal(M, 1.0)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=8)
    ax.set_title(f"Indicator correlation (pooled)\nmean |r| = "
                 f"{pooled['pearson_r'].abs().mean():.2f}")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axs[0, 1]
    for alpha, g in draws.groupby("alpha"):
        ax.scatter(g["l1_from_flat"], g["overlap"], s=4, alpha=0.25,
                   label=f"alpha = {alpha:g}")
    ax.scatter(vertices["l1_from_flat"], vertices["overlap"], s=70, marker="D",
               facecolor="none", edgecolor="k", linewidth=1.4, zorder=5,
               label="existing 13 vertices")
    ax.axhline(OVERLAP_TARGET, color="crimson", ls="--", lw=1)
    if np.isfinite(radius) and radius > 0:
        ax.axvline(radius, color="crimson", ls=":", lw=1.2)
    ax.set_xlabel("L1 distance from flat weights")
    ax.set_ylabel(f"Jaccard overlap with flat top-k")
    ax.set_title("Priority-set overlap vs weight perturbation")
    ax.legend(fontsize=8, loc="lower left")

    ax = axs[1, 0]
    data = [g["overlap"].values for _, g in draws.groupby("alpha")]
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels([f"{a:g}" for a in sorted(draws["alpha"].unique())])
    ax.axhline(OVERLAP_TARGET, color="crimson", ls="--", lw=1)
    ax.set_xlabel("Dirichlet alpha (1 = uniform over the simplex)")
    ax.set_ylabel("Jaccard overlap with flat top-k")
    ax.set_title("Overlap distribution by concentration")

    ax = axs[1, 1]
    if not curve.empty:
        ax.plot(curve["l1_mid"], curve["overlap_p50"], "o-", label="median")
        ax.plot(curve["l1_mid"], curve["overlap_p5"], "s--",
                label=f"p{BREAKEVEN_PCTL}")
        ax.axhline(OVERLAP_TARGET, color="crimson", ls="--", lw=1,
                   label=f"target {OVERLAP_TARGET:g}")
        if np.isfinite(radius) and radius > 0:
            ax.axvline(radius, color="crimson", ls=":", lw=1.2,
                       label=f"breakeven {radius:.3f}")
    vmax = vertices["l1_from_flat"].max()
    ax.axvspan(0, vmax, color="0.85", zorder=0)
    ax.text(vmax, ax.get_ylim()[0], " range probed by\n the 13 vertices",
            fontsize=8, va="bottom")
    ax.set_xlabel("L1 distance from flat weights")
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Breakeven curve")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"    figure  -> {OUT_PNG}")


# ===========================================================================
# main
# ===========================================================================
def main(argv=()):
    emit_only = "--emit-vectors" in argv
    use_agg()
    t_start = time.perf_counter()
    print("=" * 74)
    print("WEIGHT-SIMPLEX SCREEN: does indicator weighting move the priority set?"
          if not emit_only else
          "WEIGHT-SIMPLEX SCREEN: emitting the extended campaign's weight vectors")
    print("=" * 74)

    ic = load_ic(["restoration_benefit", "cost"],
                 condition_scenario=f"{BENCHMARK}_w_flat")
    shape = tuple(int(v) for v in ic["shape"])
    elig_idx = np.asarray(ic["restoration_eligible_indices"], np.int64)
    elig_mask = np.asarray(ic["restoration_eligible_mask"], bool)
    cost = np.asarray(ic["implementation_cost"]).ravel()[elig_idx].astype(np.float64)
    # Infinite (not NaN) for a non-positive cost: d/cost is then 0 and the pixel simply
    # never enters a cost-adjusted top-k. A NaN would sort to the END of argpartition and
    # be selected as if it were the best pixel on the map.
    cost = np.where(cost > 0, cost, np.inf)
    n_elig = elig_idx.size
    k = int(BASE["max_restoration_fraction"] * n_elig)
    eff_total = float(BASE["abiotic_effect"] + BASE["biotic_effect"])
    print(f"\n  eligible pixels {n_elig:,}   budget k = {k:,} "
          f"({BASE['max_restoration_fraction']:.0%})   grid {shape}")
    if n_elig != 437110:
        print(f"  NOTE: the weighting runs used 437,110 eligible pixels, this load gives "
              f"{n_elig:,}.\n        The screen is still internally consistent, but it is "
              "not the same pool as those runs.")

    layers, footprints = load_indicator_stack(BENCHMARK)
    eco_masks = ecosystem_masks(shape)
    Z, P, codes, eco_of_pixel, ecos = build_matrices(
        layers, footprints, eco_masks, elig_idx)
    print(f"  indicators {len(codes)}: {', '.join(codes)}")
    unassigned = int((eco_of_pixel < 0).sum())
    if unassigned:
        print(f"  NOTE: {unassigned:,} eligible pixels ({unassigned / n_elig:.1%}) fall "
              "outside every ecosystem footprint and carry no composite.")
    eco_share = np.array([(eco_of_pixel == ei).sum() for ei in range(len(ecos))], float)
    eco_share = eco_share / max(eco_share.sum(), 1.0)
    for ei, eco in enumerate(ecos):
        avail = [c for c in codes if (eco, c) in layers]
        print(f"    {eco:<14} {int((eco_of_pixel == ei).sum()):>8,} px  "
              f"n = {len(avail):<2} ({', '.join(avail)})")

    val_rows, worst = validate(Z, P, eco_of_pixel, codes, ecos, layers, elig_idx)

    # --emit-vectors still runs the gate above: the vectors are only worth emitting if
    # this rebuild reproduces the R construction, since R is what will rasterise them.
    if emit_only:
        cmd_emit_vectors(Z, P, codes, eco_of_pixel, ecos, layers, cost, k, eff_total,
                         eco_share)
        print(f"\n  total {time.perf_counter() - t_start:.0f}s")
        return

    corr = correlations(Z, P, codes, eco_of_pixel, ecos)
    draws, vertices = sample_simplex(Z, P, codes, eco_of_pixel, ecos, layers,
                                     cost, k, eff_total, eco_share)
    radius, curve = breakeven(draws)

    # -- self-checks ----------------------------------------------------------
    flat_row = vertices[vertices["vertex"] == "w_flat"]
    if len(flat_row) and abs(float(flat_row["overlap"].iloc[0]) - 1.0) > 1e-9:
        raise SystemExit("w_flat does not score overlap 1.0 against itself - the "
                         "reference top-k set is not being built consistently.")

    # -- report ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    pooled_r = corr[corr["group"] == "pooled"]["pearson_r"].abs().mean()
    print(f"  mean |r| between indicators (pooled)      {pooled_r:.3f}")
    for alpha, g in draws.groupby("alpha"):
        print(f"  alpha {alpha:<5g} overlap  p5 {np.percentile(g['overlap'], 5):.3f}"
              f"   median {g['overlap'].median():.3f}"
              f"   min {g['overlap'].min():.3f}"
              f"   |  cost-adj median {g['overlap_cost_adj'].median():.3f}")
    print(f"  breakeven radius (p{BREAKEVEN_PCTL} overlap >= {OVERLAP_TARGET:g})   "
          f"L1 = {radius:.3f}")
    v_max = float(vertices["l1_from_flat"].max())
    v_min_ov = float(vertices["overlap"].min())
    print(f"  the 13 vertices span L1 0 to {v_max:.3f}, worst overlap {v_min_ov:.3f}")
    uni = draws[draws["alpha"] == min(ALPHAS)]
    reach = float((uni["l1_from_flat"] > v_max).mean())
    print(f"  {reach:.0%} of uniform draws sit beyond the vertex range")
    if curve.empty:
        print("\n  READING: too few draws per distance bin to fit a breakeven curve - "
              "raise N_DRAWS.")
    elif radius >= uni["l1_from_flat"].max():
        print("\n  READING: overlap holds across the whole sampled simplex. The priority\n"
              "  set is robust to indicator weighting well beyond any defensible\n"
              "  disagreement - report this as a breakeven statement, and treat the 65\n"
              "  weighting runs as confirmation rather than the primary evidence.")
    elif radius > v_max:
        print("\n  READING: overlap holds past the range the 13 vertices probe but breaks\n"
              "  down inside the simplex. The existing design was too local to see it;\n"
              "  pick new weight vectors from beyond the breakeven radius rather than\n"
              "  adding more focal vertices near the centre.")
    else:
        print("\n  READING: overlap breaks down INSIDE the range the vertices already\n"
              "  probe, so weighting is a live axis and the screen cannot retire it.\n"
              "  Use the curve to choose informative weight vectors and run them.")

    # -- write ----------------------------------------------------------------
    corr.to_csv(OUT_CORR, index=False)
    draws.to_csv(OUT_DRAWS, index=False)
    summary = pd.concat([
        vertices.assign(kind="vertex"),
        draws.groupby("alpha").agg(
            overlap_p5=("overlap", lambda x: np.percentile(x, 5)),
            overlap_p50=("overlap", "median"),
            overlap_min=("overlap", "min"),
            overlap_cost_adj_p50=("overlap_cost_adj", "median"),
            spearman_p50=("spearman", "median"),
            l1_max=("l1_from_flat", "max"),
        ).reset_index().assign(kind="dirichlet"),
    ], ignore_index=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    with open(OUT_JSON, "w", encoding="ascii") as f:
        json.dump({
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"benchmark": BENCHMARK, "alphas": list(ALPHAS),
                       "n_draws": N_DRAWS, "seed": RANDOM_SEED,
                       "overlap_target": OVERLAP_TARGET,
                       "breakeven_pctl": BREAKEVEN_PCTL,
                       "budget_fraction": BASE["max_restoration_fraction"]},
            "n_eligible": int(n_elig), "k": int(k), "indicators": codes,
            "validation_max_abs_err": worst,
            "mean_abs_r_pooled": float(pooled_r),
            "breakeven_radius": float(radius),
            "vertex_l1_max": v_max, "vertex_overlap_min": v_min_ov,
            "uniform_overlap_p5": float(np.percentile(uni["overlap"], 5)),
            "uniform_overlap_median": float(uni["overlap"].median()),
        }, f, indent=1)
    print(f"\n  {OUT_CORR}\n  {OUT_DRAWS}\n  {OUT_SUMMARY}\n  {OUT_JSON}")
    make_figure(corr, draws, vertices, curve, codes, radius)
    print(f"\n  total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
