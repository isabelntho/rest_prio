"""Rebuild data/ec_anomalies.r's condition composites in python.

Reads the per-indicator z-score layers exported by that script's
GENERATE_INDICATOR_LAYERS block (data/anomaly_scenarios/indicators/) and reassembles
any weighted composite, at any benchmark quantile, without re-running R.

Two consumers: Debugs_tests/weight_simplex_screen.py (weighting axis) and
Core_optimisation/prim.py (weighting + continuous quantile). Both validate their
rebuild against the on-disk R rasters before using it.
"""
import numpy as np
import rasterio as rio

from Core_optimisation.paths import DATA_DIR

# ECT_CATEGORIES of data/ec_anomalies.r:94, and the LULC classes data_loader masks on.
ECOSYSTEMS = {"forest": (12, 13), "agricultural": (15,), "grassland": (16, 17)}
ABIOTIC = ("smd", "sbd", "soc")
BIOTIC = ("uzl", "tsd", "can", "cdi", "swf_h", "swf_t", "lai", "ndvi")

SCEN_DIR = DATA_DIR / "anomaly_scenarios"
IND_DIR = SCEN_DIR / "indicators"
LULC_PATH = DATA_DIR / "ecosystem_lulc_masked.tif"


# ===========================================================================
# inputs
# ===========================================================================
def read_raster(path):
    """One band as float64 with nodata as NaN."""
    with rio.open(path) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
    return a


def load_indicator_stack(benchmark="global"):
    """({(ecosystem, code): 2-D z-scores}, {ecosystem: footprint mask})."""
    if not IND_DIR.is_dir():
        raise SystemExit(
            f"{IND_DIR} does not exist.\n"
            "Set GENERATE_INDICATOR_LAYERS <- TRUE in data/ec_anomalies.r (and\n"
            "GENERATE_WEIGHT_SCENARIOS <- FALSE so the existing weighting rasters are\n"
            "not rewritten), run it once, then set the flag back.")
    layers, footprints = {}, {}
    for eco in ECOSYSTEMS:
        fp_path = IND_DIR / f"{eco}_footprint_{benchmark}.tif"
        if not fp_path.exists():
            raise SystemExit(f"missing footprint raster: {fp_path}")
        footprints[eco] = np.isfinite(read_raster(fp_path))
        for code in list(ABIOTIC) + list(BIOTIC):
            p = IND_DIR / f"{eco}_{code}_{benchmark}.tif"
            if p.exists():
                layers[(eco, code)] = read_raster(p)
    if not layers:
        raise SystemExit(f"no indicator layers found in {IND_DIR}")
    return layers, footprints


def ecosystem_masks(shape):
    """{ecosystem: bool mask} from the LULC raster, asserted disjoint."""
    lulc = read_raster(LULC_PATH)
    if lulc.shape != shape:
        raise SystemExit(f"LULC raster is {lulc.shape}, initial conditions are {shape}")
    masks = {eco: np.isin(lulc, codes) for eco, codes in ECOSYSTEMS.items()}
    stacked = np.sum([m.astype(np.int8) for m in masks.values()], axis=0)
    if stacked.max() > 1:
        raise SystemExit(f"ecosystem masks overlap on {int((stacked > 1).sum())} pixels")
    return masks


def build_matrices(layers, footprints, eco_masks, idx):
    """Design matrices over the flat raster indices `idx`.

    Returns (Z, P, codes, eco_of_pixel, ecos):
      Z  (n, n_codes) z-scores, zero where the indicator is absent
      P  (n, n_codes) 1 where present, so C = (Z @ w) / (P @ w) is R's per-pixel
         renormalised weighted mean - an absent indicator drops out of both sums.
    """
    codes = [c for c in list(ABIOTIC) + list(BIOTIC)
             if any((eco, c) in layers for eco in ECOSYSTEMS)]
    ecos = list(ECOSYSTEMS)
    idx = np.asarray(idx, np.int64)
    Z = np.zeros((idx.size, len(codes)), np.float32)
    P = np.zeros((idx.size, len(codes)), np.float32)
    eco_of_pixel = np.full(idx.size, -1, np.int8)

    for ei, eco in enumerate(ecos):
        sel = (eco_masks[eco] & footprints[eco]).ravel()[idx]
        if not sel.any():
            continue
        eco_of_pixel[sel] = ei
        rows = np.flatnonzero(sel)
        for ci, code in enumerate(codes):
            lyr = layers.get((eco, code))
            if lyr is None:
                continue
            v = lyr.ravel()[idx[rows]]
            ok = np.isfinite(v)
            Z[rows[ok], ci] = v[ok]
            P[rows[ok], ci] = 1.0
    return Z, P, codes, eco_of_pixel, ecos


# ===========================================================================
# weight schemes: faithful mirrors of data/ec_anomalies.r
# ===========================================================================
def build_indicator_weights(var_codes, scheme):
    """Port of build_indicator_weights (data/ec_anomalies.r:600).

    The fallbacks are reproduced exactly: they are what makes a focal vertex a no-op in
    the ecosystems that do not carry the focal indicator.
    """
    codes = list(var_codes)
    n = len(codes)
    if n == 0:
        raise ValueError("build_indicator_weights: no indicators supplied")
    base = {c: 1.0 / n for c in codes}

    if scheme == "flat":
        w = dict(base)

    elif scheme == "cat":
        abio = [c for c in codes if c in ABIOTIC]
        bio = [c for c in codes if c in BIOTIC]
        w = {c: 0.0 for c in codes}
        if abio and bio:
            for c in abio:
                w[c] = 0.5 / len(abio)
            for c in bio:
                w[c] = 0.5 / len(bio)
        elif abio:
            for c in abio:
                w[c] = 1.0 / len(abio)
        elif bio:
            for c in bio:
                w[c] = 1.0 / len(bio)
        else:
            raise ValueError(f"no abiotic/biotic indicators in ({', '.join(codes)})")

    elif scheme.startswith("focal:"):
        focal = scheme[len("focal:"):]
        if focal not in codes or n < 3:
            w = dict(base)
        else:
            rest = [c for c in codes if c != focal]
            rest_sum = sum(base[c] for c in rest)
            w = dict(base)
            w[focal] = 2.0 / n
            shrink = (1.0 - 2.0 / n) / rest_sum
            for c in rest:
                w[c] = base[c] * shrink
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")

    total = sum(w.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"scheme {scheme!r} weights sum to {total!r}, not 1")
    return w


def weight_scheme_names():
    """The 13 tag suffixes of WEIGHT_SCHEMES (data/ec_anomalies.r:359), in order."""
    schemes = [("w_flat", "flat"), ("w_cat", "cat")]
    schemes += [(f"w_{k}", f"focal:{k}") for k in list(ABIOTIC) + list(BIOTIC)]
    return schemes


def scheme_weight_columns(scheme, codes, ecos, layers):
    """(n_codes, n_ecos) weight matrix: column e is `scheme` resolved in ecosystem e."""
    Wcol = np.zeros((len(codes), len(ecos)), np.float64)
    for ei, eco in enumerate(ecos):
        avail = [c for c in codes if (eco, c) in layers]
        if not avail:
            continue
        w = build_indicator_weights(avail, scheme)
        for c, v in w.items():
            Wcol[codes.index(c), ei] = v
    return Wcol


def draw_to_eco_columns(w_union, flat_col):
    """One union weight vector -> (n_codes, n_ecos) per-ecosystem probability vectors.

    Availability is read off `flat_col`, non-zero exactly where an indicator exists in
    that ecosystem. Renormalising each column does not change the composite (num/den is
    scale-invariant in w), only the distance axis.
    """
    Wcol = np.asarray(w_union, float)[:, None] * (flat_col > 0)
    totals = Wcol.sum(axis=0)
    return np.divide(Wcol, np.where(totals > 0, totals, 1.0))


def l1_from_flat(Wcol, flat_col, eco_share):
    """Pixel-weighted mean total-variation distance from flat weights.

    Measured inside each ecosystem against its own flat vector, then averaged by
    eligible-pixel share, so a draw and a focal vertex - which reaches only the
    ecosystems carrying its indicator - land on one comparable axis.
    """
    per_eco = 0.5 * np.abs(Wcol - flat_col).sum(axis=0)
    return float((per_eco * eco_share).sum())


# ===========================================================================
# composite
# ===========================================================================
def composite(Z, P, W):
    """(n, B) composites for a block of union weight vectors W (n_codes, B)."""
    num = Z @ W
    den = P @ W
    with np.errstate(divide="ignore", invalid="ignore"):
        C = num / den
    return np.where(den > 0, C, np.nan)


def composite_by_ecosystem(Z, P, eco_of_pixel, Wcol, affine=None):
    """Composite where each pixel uses its own ecosystem's weight vector.

    `affine` is an optional (M, S) pair of (n_codes, n_ecos) arrays rescaling the
    z-scores to another benchmark, z' = (z - M) / S. Folded into the weights rather
    than applied to Z, since
        sum_c w_c (z_c - m_c) / s_c = Z @ (w / s) - P @ (w * m / s)
    which keeps the per-scenario cost at two matvecs and never copies Z.
    """
    C = np.full(Z.shape[0], np.nan)
    for ei in range(Wcol.shape[1]):
        rows = np.flatnonzero(eco_of_pixel == ei)
        if rows.size == 0:
            continue
        w = Wcol[:, ei]
        den = P[rows] @ w
        if affine is None:
            num = Z[rows] @ w
        else:
            M, S = affine
            ok = S[:, ei] > 0
            w_s = np.where(ok, w / np.where(ok, S[:, ei], 1.0), 0.0)
            w_m = np.where(ok, w * M[:, ei] / np.where(ok, S[:, ei], 1.0), 0.0)
            num = Z[rows] @ w_s - P[rows] @ w_m
            den = P[rows] @ np.where(ok, w, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            C[rows] = np.where(den > 0, num / den, np.nan)
    return C


# ===========================================================================
# the benchmark-quantile axis
# ===========================================================================
def quantile_prefix(layers):
    """Sorted values + suffix sums per layer, so any benchmark quantile is O(log n).

    R computes upper_qNN as mean/sd over {x >= quantile(x, p)}. A positive affine map
    preserves quantile membership, so that set is identical whether selected on the raw
    variable or on the exported z-score, and the reference statistics can be read off
    the z-scores alone (see prim.py's docstring).
    """
    out = {}
    for key, lyr in layers.items():
        a = np.sort(lyr[np.isfinite(lyr)].astype(np.float64))
        s1 = np.concatenate([np.cumsum(a[::-1])[::-1], [0.0]])
        s2 = np.concatenate([np.cumsum((a * a)[::-1])[::-1], [0.0]])
        out[key] = (a, s1, s2)
    return out


def _quantile_type7(a, q):
    """R's default quantile on an already-sorted ascending array."""
    if q <= 0:
        return a[0]
    h = q * (a.size - 1)
    lo = int(np.floor(h))
    if lo >= a.size - 1:
        return a[-1]
    return a[lo] + (h - lo) * (a[lo + 1] - a[lo])


def rescale_to_quantile(prefix, q, codes, ecos, min_n=5000):
    """Benchmark-quantile affine (M, S) as (n_codes, n_ecos) arrays, plus skipped keys.

    S is 0 for an indicator this benchmark cannot support - too few reference pixels or
    a degenerate sd - mirroring the skip in ec_anomalies.r:517-527, which drops the
    indicator from the composite rather than emitting NaNs.
    """
    M = np.zeros((len(codes), len(ecos)))
    S = np.zeros((len(codes), len(ecos)))
    skipped = []
    for ei, eco in enumerate(ecos):
        for ci, code in enumerate(codes):
            item = prefix.get((eco, code))
            if item is None:
                continue
            a, s1, s2 = item
            if q <= 0:
                M[ci, ei], S[ci, ei] = 0.0, 1.0
                continue
            start = int(np.searchsorted(a, _quantile_type7(a, q), side="left"))
            k = a.size - start
            if k < max(2, min_n):
                skipped.append((eco, code, k))
                continue
            mean = s1[start] / k
            # ddof=1: terra's global(r, "sd") is the sample sd.
            var = (s2[start] - k * mean * mean) / (k - 1)
            if not np.isfinite(var) or var <= 0:
                skipped.append((eco, code, k))
                continue
            M[ci, ei], S[ci, ei] = mean, np.sqrt(var)
    return M, S, skipped
