"""Screen candidate replacement objectives BEFORE spending a run on one.

    pixi run python Debugs_tests/candidate_objective_screen.py [--region CH]

Two bars, both of which have already killed an objective in this project:

  ORTHOGONALITY  pixel-level Spearman against cost and against combined abiotic+biotic
                 anomaly. Valid at pixel level because these are static layers, not
                 functions of a plan. Want |rho| below ~0.7 against both.

  ATTAINABLE RANGE  the bar restoration_benefit failed (span 1168x narrower than the
                 widest objective). Per-pixel sd/CV catches a dead layer; the PLAN-level
                 span across budget-feasible probe plans catches the subtler failure,
                 where a layer varies per pixel but averages out over the ~137k pixels a
                 plan selects. Reject anything more than an order of magnitude narrower
                 than the widest objective.

Also reports Moran's I of the anomaly surface first. For landscape_context - the 500 m
neighbour-mean of abiotic anomaly, and the strongest a-priori candidate - one quantity
decides both bars at once. High autocorrelation: the neighbourhood mean keeps its
variance AND degraded pixels really do sit in degraded neighbourhoods, so the tension
against restoration_benefit is real. Low autocorrelation: the mean collapses toward a
constant AND the tension evaporates, because degraded pixels in good surroundings are
freely available and both objectives can be satisfied at once.

No optimiser is run: a metric is an input to the search, so this screens candidates OUT
cheaply, it does not prove one works. That needs a run.
"""
import sys
import argparse

import numpy as np

from _common import REPO_ROOT

# loader name -> the initial_conditions key holding its per-pixel values.
# All indexed over RESTORATION-eligible pixels (data_loader.py:1295-1379), same
# population as cost and abiotic/biotic anomaly, so a direct per-pixel comparison is
# valid. 'connectivity' is deliberately EXCLUDED: connectivity_gain_1d is indexed over
# CONVERSION-eligible pixels (data_loader.py:1212) - a different decision (which land to
# convert, not which to restore) over a different pixel population - so it cannot
# replace spatial_clustering in a restoration-only objective set and a per-pixel
# comparison against cost/anomaly would compare unrelated pixel populations.
CANDIDATES = {
    "landscape_context": "landscape_context_1d",
    "es_future_val": "es_future_val_1d",
    "es_future_robustness": "es_future_robustness_1d",
    "population_proximity": "population_proximity",
}
# Objectives RestorationProblem can actually build (see its _candidates list).
USABLE_AS_OBJECTIVE = {"landscape_context", "es_future_val", "es_future_robustness"}


def spearman(a, b):
    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        r[o] = np.arange(len(v))
        return r
    a, b = rank(a), rank(b)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def morans_i(grid, mask):
    """Rook-contiguity Moran's I of `grid` over the True cells of `mask`.

    Each orthogonal pair is summed once here; the symmetric weight matrix counts it
    twice and S0 = 2 * n_pairs, so the 2s cancel to n / n_pairs * cross / denom.
    """
    x = np.asarray(grid, dtype=np.float64)
    xbar = float(np.nanmean(np.where(mask, x, np.nan)))
    d = np.where(mask, np.nan_to_num(x - xbar, nan=0.0), 0.0)
    cross = (float(np.sum(d[:, :-1] * d[:, 1:]))
             + float(np.sum(d[:-1, :] * d[1:, :])))
    n_pairs = (int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
               + int(np.count_nonzero(mask[:-1, :] & mask[1:, :])))
    denom = float(np.sum(d ** 2))
    n = int(np.count_nonzero(mask))
    if denom <= 0 or n_pairs == 0:
        return float("nan")
    return (n / n_pairs) * cross / denom


def neighbour_mean(grid, mask):
    """Mean of the 4 orthogonal neighbours, counting masked neighbours only."""
    v = np.where(mask, np.nan_to_num(np.asarray(grid, dtype=np.float64)), 0.0)
    m = mask.astype(np.float64)
    s = np.zeros_like(v)
    c = np.zeros_like(v)
    for dst, src in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                     ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                     ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                     ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        s[dst] += v[src]
        c[dst] += m[src]
    return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


def as_pixel_vector(ic, key, rest_mask):
    """Per-restoration-eligible-pixel values for `key`, or None if absent."""
    if key not in ic:
        return None
    v = np.asarray(ic[key])
    if v.ndim == 2:
        return v[rest_mask].astype(np.float64)
    return v.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="CH")
    ap.add_argument("--ecosystem", default="all")
    ap.add_argument("--budget-fraction", type=float, default=0.05)
    args = ap.parse_args()

    from Core_optimisation.data_loader import load_initial_conditions

    wanted = ["abiotic", "biotic", "cost"] + list(CANDIDATES)
    print("loading %s with objectives %s ..." % (args.region, wanted), flush=True)
    ic = None
    attempts = [wanted, ["abiotic", "biotic", "cost"] + sorted(USABLE_AS_OBJECTIVE)]
    for attempt in attempts:
        try:
            ic = load_initial_conditions(
                REPO_ROOT, objectives=attempt, region=args.region,
                ecosystem=args.ecosystem, sample_fraction=0.8, sample_seed=42,
                aggregation_factor=None, condition_scenario="global_all")
            dropped = sorted(set(wanted) - set(attempt))
            if dropped:
                print("  loaded WITHOUT %s (see the load failure above); those "
                      "candidates will show as MISSING below" % dropped)
            break
        except Exception as exc:
            print("  load failed for %s: %s" % (attempt, exc))
    if ic is None:
        raise SystemExit("could not load any candidate set")

    rest_mask = ic["restoration_eligible_mask"]
    cost = as_pixel_vector(ic, "implementation_cost", rest_mask)
    ab = as_pixel_vector(ic, "abiotic_anomaly", rest_mask)
    bi = as_pixel_vector(ic, "biotic_anomaly", rest_mask)
    anom = np.abs(ab) + np.abs(bi)
    n_rest = anom.size
    budget = max(int(args.budget_fraction * n_rest), 1)
    print("\n%d restoration-eligible pixels, budget %d px (%.0f%%)"
          % (n_rest, budget, 100 * args.budget_fraction))

    # --- the quantity that decides landscape_context -----------------------------
    print("\n== spatial autocorrelation of the anomaly surface ==")
    comb2d = np.abs(np.asarray(ic["abiotic_anomaly"], dtype=np.float64)) + \
        np.abs(np.asarray(ic["biotic_anomaly"], dtype=np.float64))
    mi = morans_i(comb2d, rest_mask)
    nb = neighbour_mean(comb2d, rest_mask)[rest_mask]
    ok = np.isfinite(nb)
    rho_nb = spearman(anom[ok], nb[ok])
    print("  Moran's I (rook) of abiotic+biotic anomaly : %+.4f" % mi)
    print("  rho(pixel anomaly, neighbour-mean anomaly) : %+.4f" % rho_nb)
    print("  sd(anomaly) %.6g  ->  sd(neighbour mean) %.6g  = %.1f%% retained"
          % (np.nanstd(anom), np.nanstd(nb), 100 * np.nanstd(nb) / max(np.nanstd(anom), 1e-300)))
    if mi < 0.3:
        print("  -> LOW autocorrelation: expect landscape_context to be both "
              "low-variance and weakly opposed to restoration_benefit.")
    else:
        print("  -> autocorrelation is substantial: the benefit-vs-context tension "
              "should be real, and the neighbourhood mean keeps variance.")

    # --- bar 1: orthogonality, and bar 2a: per-pixel range -----------------------
    layers = {"cost": cost, "anomaly(abiotic+biotic)": anom}
    for name, key in CANDIDATES.items():
        v = as_pixel_vector(ic, key, rest_mask)
        if v is None:
            print("\n  MISSING %s (key %r not in initial_conditions)" % (name, key))
            continue
        if v.shape[0] != n_rest:
            print("\n  SKIPPING %s: length %d != %d restoration-eligible pixels "
                  "(indexed over a different pixel population, e.g. conversion-eligible "
                  "- not comparable per-pixel to cost/anomaly)" % (name, v.shape[0], n_rest))
            continue
        layers[name] = v

    print("\n== bar 1, orthogonality + bar 2a, per-pixel range ==")
    print("  %-24s %10s %10s %12s %12s %10s"
          % ("layer", "rho|cost", "rho|anom", "sd", "CV", "q95/q05"))
    for name, v in layers.items():
        fin = np.isfinite(v)
        q05, q95 = np.nanpercentile(v[fin], [5, 95])
        mean = np.nanmean(v[fin])
        cv = np.nanstd(v[fin]) / abs(mean) if abs(mean) > 1e-300 else float("nan")
        ratio = (q95 / q05) if abs(q05) > 1e-300 else float("nan")
        rc = spearman(v[fin], cost[fin]) if name != "cost" else 1.0
        ra = spearman(v[fin], anom[fin]) if name != "anomaly(abiotic+biotic)" else 1.0
        flag = "" if (abs(rc) < 0.7 and abs(ra) < 0.7) else "   COLLINEAR"
        print("  %-24s %+10.4f %+10.4f %12.5g %12.5g %10.4g%s"
              % (name, rc, ra, np.nanstd(v[fin]), cv, ratio, flag))

    # --- bar 2b: PLAN-level attainable span --------------------------------------
    # A plan sums its layer over `budget` pixels, so per-pixel variance is averaged
    # away by ~sqrt(budget). What matters is how far the SUM can be pushed between the
    # best and worst budget-feasible selection of that layer.
    print("\n== bar 2b, plan-level attainable span (sum over %d selected px) ==" % budget)
    print("  %-24s %14s %14s %8s %12s"
          % ("layer", "greedy min", "greedy max", "max/min", "vs widest"))
    spans = {}
    for name, v in layers.items():
        fin = np.nan_to_num(v, nan=np.nanmedian(v))
        order = np.argsort(fin, kind="stable")
        lo = float(fin[order[:budget]].sum())
        hi = float(fin[order[-budget:]].sum())
        spans[name] = abs(hi - lo) / max(abs(hi), abs(lo), 1e-300)
        ratio = abs(hi) / abs(lo) if abs(lo) > 1e-300 else float("inf")
        print("  %-24s %14.6g %14.6g %8.3g" % (name, lo, hi, ratio))
    widest = max(spans.values())
    print("\n  relative span (1.00 = widest objective; reject below ~0.10):")
    for name, s in sorted(spans.items(), key=lambda kv: -kv[1]):
        verdict = "" if s / widest >= 0.10 else "   TOO NARROW"
        print("    %-24s %.4f   %.1fx narrower than widest%s"
              % (name, s / widest, widest / s if s > 0 else float("inf"), verdict))

    missing = sorted(set(CANDIDATES) - USABLE_AS_OBJECTIVE)
    if missing:
        print("\n  NOTE: %s can be loaded but is NOT in RestorationProblem._candidates, "
              "so it cannot be used as an objective without adding it there."
              % ", ".join(missing))


if __name__ == "__main__":
    sys.exit(main())
