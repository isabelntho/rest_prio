"""Exact benefit-cost front for the 2-objective restoration problem, by ILP.

The NSGA-II front for ["restoration_benefit", "cost"] has never been bounded. That
formulation is exactly linearisable: anomaly_improvement_weight depends only on the
BASELINE anomaly, so each pixel's direct gain is a constant, and spillover is a coverage
effect (a cell gains once however many restored neighbours it has, optimization_engine.py:398-420).
So the problem is an integer linear program.

This solves it with HiGHS (via scipy.optimize.milp) under an epsilon-constraint sweep on
cost, producing a reference front and - more usefully - a valid UPPER BOUND on achievable
benefit at each cost level, which holds even when the MIP does not close its gap.

Usage:
    pixi run python -m Core_optimisation.exact_front [--results-pkl <path> ...] [options]

Model (restoration only; 2x2 patches are disjoint tiles, so s_k = x_{p(k)} is a pure
substitution and the s variables are eliminated):

    maximise  sum_p D_p x_p + sum_k n_k z_k
    s.t.      z_k - sum_{p in P(k)} x_p <= 0     coverage: some patch in k's disc is chosen
              z_k + x_{p(k)}            <= 1     restored cells get no spillover
              lo <= sum_p m_p x_p <= hi          pixel budget, +/- pixel_tolerance
              sum_p C_p x_p <= C_j               epsilon-constraint on cost

z_k stays continuous: n_k >= 0 and we maximise, so z_k is driven to the coverage indicator.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from Core_optimisation.paths import OUTPUTS, RESULTS_DIR, PROJECT_ROOT, ensure
from Core_optimisation.data_loader import load_initial_conditions
from Core_optimisation.patch_approach import (
    create_patch_mappings,
    convert_patch_decisions_to_pixels,
)
from Core_optimisation.optimization_engine import RestorationProblem, anomaly_improvement_weight
from Core_optimisation.regret_common import write_report
from Core_optimisation.uncertainty_analysis import (
    effect_params_of,
    build_variant_layer,
    disc_kernel,
    evaluate,
    nondominated_2d,
    VALIDATE_RTOL_BENEFIT,
)

# NSGA-II results pkls to pool and overlay (one per seed). Bare file names are looked up in
# RESULTS_DIR; absolute or relative paths are used as given. --results-pkl overrides this.
NSGA2_PKLS = [
     "20260924_1753_ch_patch_long_seed101",
     "20260924_2159_ch_patch_long_seed102",
     "20260925_0205_ch_patch_long_seed103",
     "20260925_0610_ch_patch_long_seed104",
     "20260925_1119_ch_patch_long_seed105"
]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import pickle_load

# The ILP is built in float64 from the same coefficients, so it must reproduce its own
# arithmetic to rounding. A miss here is a model/matrix bug, not lost precision.
VALIDATE_RTOL_SELF = 1e-12
# The engine sums implementation cost as a float32 array (np.sum over the float32 cost
# raster), so its total carries ~1e-8 of accumulation error that an exact float64 sum
# cannot reproduce. Measured against the ILP on six real Bern plans: 1.5e-8 to 4.5e-8.
# uncertainty_analysis.VALIDATE_RTOL_COST (1e-9) assumes both sides sum in float32.
VALIDATE_RTOL_COST_ENGINE = 1e-6

OBJECTIVE_NAMES = ["restoration_benefit", "implementation_cost"]
# scipy.optimize.milp status codes.
STATUS_MSG = {0: "optimal", 1: "time/iteration limit", 2: "infeasible",
              3: "unbounded", 4: "other"}


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_coefficients(ic, pm, eff):
    """Per-pixel and per-patch ILP coefficients. Returns a dict."""
    shape = tuple(int(v) for v in ic["shape"])
    rest_idx = np.asarray(ic["restoration_eligible_indices"], np.int64)
    n_rest = int(ic["n_restoration_pixels"])
    elig = np.asarray(ic["restoration_eligible_mask"], bool)

    shp, scl = eff["anomaly_weight_shape"], eff["anomaly_weight_scale"]
    d_raster = (eff["abiotic_effect"]
                * anomaly_improvement_weight(np.asarray(ic["abiotic_anomaly"], float), shp, scl)
                + eff["biotic_effect"]
                * anomaly_improvement_weight(np.asarray(ic["biotic_anomaly"], float), shp, scl))
    d_raster = np.where(elig, d_raster, 0.0)

    # Cross-check against the vetted layer builder; they must be bit-identical.
    d_ref = build_variant_layer(ic["abiotic_anomaly"], ic["biotic_anomaly"], elig, eff)
    err = float(np.max(np.abs(d_raster - d_ref)))
    assert err == 0.0, f"d raster differs from build_variant_layer (max abs {err:.3e})"

    d = d_raster.ravel()[rest_idx]
    n = eff["neighbor_effect_decay"] * d
    cost_raster = np.asarray(ic["implementation_cost"])
    c = cost_raster.ravel()[rest_idx].astype(np.float64)

    rest = pm["restoration_patches"]
    n_patches = int(rest["n_patches"])
    patch_of_pixel = np.full(n_rest, -1, np.int32)
    total = 0
    for pid, pix in rest["patch_to_pixels"].items():
        pix = np.atleast_1d(np.asarray(pix, np.int64))
        patch_of_pixel[pix] = pid
        total += pix.size
    assert (patch_of_pixel >= 0).all(), "an eligible pixel belongs to no patch"
    assert total == n_rest, f"patches are not disjoint/complete ({total} vs {n_rest})"

    D_p = np.bincount(patch_of_pixel, weights=d, minlength=n_patches)
    C_p = np.bincount(patch_of_pixel, weights=c, minlength=n_patches)
    m_p = np.bincount(patch_of_pixel, minlength=n_patches).astype(np.float64)

    return dict(shape=shape, rest_idx=rest_idx, n_rest=n_rest, d_raster=d_raster,
                cost_raster=cost_raster, d=d, n=n, c=c, n_patches=n_patches,
                patch_of_pixel=patch_of_pixel, D_p=D_p, C_p=C_p, m_p=m_p)


def build_coverage(co, radius):
    """P(k) as a CSR patch-incidence matrix over the z pixels.

    z is kept only where n_k > 0; d_k is zero wherever both baseline anomalies are
    non-negative (anomaly_improvement_weight returns 0 there), so those z variables
    can never contribute. Exact reduction.
    """
    shape, n_rest = co["shape"], co["n_rest"]
    nrows, ncols = shape
    rows, cols = np.divmod(co["rest_idx"], ncols)

    patch_of_cell = np.full(shape, -1, np.int32)
    patch_of_cell[rows, cols] = co["patch_of_pixel"]

    zsel = np.flatnonzero(co["n"] > 0.0)
    n_z = zsel.size
    zr, zc = rows[zsel], cols[zsel]

    offs = np.argwhere(disc_kernel(radius)) - radius          # (29, 2) for radius 3
    nb = np.full((n_z, offs.shape[0]), -1, np.int32)
    for t in range(offs.shape[0]):
        dy, dx = int(offs[t, 0]), int(offs[t, 1])
        rr, cc = zr + dy, zc + dx
        ok = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
        v = np.full(n_z, -1, np.int32)
        v[ok] = patch_of_cell[rr[ok], cc[ok]]
        nb[:, t] = v

    # Distinct patches per row: sort, then drop -1 and repeats. Using distinct patches
    # rather than one term per neighbouring pixel is equivalent (the row is one-sided
    # and z_k <= 1 is already a bound) and ~3x fewer nonzeros.
    nb.sort(axis=1)
    keep = nb >= 0
    keep[:, 1:] &= nb[:, 1:] != nb[:, :-1]

    counts = keep.sum(axis=1).astype(np.int64)
    indptr = np.zeros(n_z + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = nb[keep].astype(np.int32)
    cov = sp.csr_matrix((np.ones(indices.size), indices, indptr),
                        shape=(n_z, co["n_patches"]))
    return zsel, cov


def assemble(co, zsel, cov, lo, hi):
    """Full constraint matrix and its bound vectors. Only ub[-1] (the cost cap) varies."""
    n_p, n_z = co["n_patches"], zsel.size
    eye = sp.identity(n_z, format="csr")

    a_cov = sp.hstack([-cov, eye], format="csr")                       # z - sum x <= 0
    excl = sp.csr_matrix((np.ones(n_z), (np.arange(n_z), co["patch_of_pixel"][zsel])),
                         shape=(n_z, n_p))
    a_excl = sp.hstack([excl, eye], format="csr")                      # z + x_p(k) <= 1

    wide = np.arange(n_p)
    a_bud = sp.csr_matrix((co["m_p"], (np.zeros(n_p, int), wide)), shape=(1, n_p + n_z))
    a_cost = sp.csr_matrix((co["C_p"], (np.zeros(n_p, int), wide)), shape=(1, n_p + n_z))

    A = sp.vstack([a_cov, a_excl, a_bud, a_cost], format="csr")
    A.sort_indices()
    lb = np.concatenate([np.full(2 * n_z, -np.inf), [lo], [-np.inf]])
    ub = np.concatenate([np.zeros(n_z), np.ones(n_z), [hi], [np.inf]])
    return A, lb, ub


def benefit_from_matrix(co, zsel, cov, x_p):
    """B(x) evaluated straight off the ILP coefficients and the coverage matrix.

    Independent of the solver: used to prove the assembled matrix encodes the intended
    objective before any solving happens.
    """
    covered = (cov @ x_p) > 0.5
    unrestored = x_p[co["patch_of_pixel"][zsel]] < 0.5
    return float(co["D_p"] @ x_p) + float(co["n"][zsel][covered & unrestored].sum())


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------
def _nan_if_none(v):
    """HiGHS reports None for gap/bound when the limit hits before any LP is solved."""
    return np.nan if v is None else float(v)


def solve(c_obj, A, lb, ub, integrality, time_limit, mip_gap, disp, label):
    """One HiGHS solve. Returns (x, record) with the record already sign-corrected."""
    t0 = time.perf_counter()
    res = milp(c=c_obj, constraints=LinearConstraint(A, lb, ub),
               integrality=integrality, bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_gap),
                        "presolve": True, "disp": bool(disp)})
    dt = time.perf_counter() - t0
    status = int(res.status)
    # The dual bound bounds the MINIMUM. Callers that minimise -B re-sign it into an
    # upper bound on achievable benefit; it is the number that brackets the true front
    # when the gap does not close.
    rec = {"status": STATUS_MSG.get(status, str(status)), "status_code": status,
           "status_msg": str(res.message), "mip_gap": _nan_if_none(getattr(res, "mip_gap", None)),
           "dual_bound": _nan_if_none(getattr(res, "mip_dual_bound", None)), "runtime_s": dt}
    print(f"  [{label}] {rec['status']} in {dt:.1f}s  gap={rec['mip_gap']:.3e}")
    if res.x is None:
        return None, rec
    return np.asarray(res.x, float), rec


def solve_min_cost(co, lo, hi):
    """C_min: cheapest plan meeting the budget, by greedy fill against the LP bound.

    One knapsack row over ~1e5 binaries is a poor MIP (HiGHS stalls in it), but the LP
    relaxation is trivial: fill patches in ascending cost-per-pixel order until lo pixels.
    The integer fill is the same list, run until the count first reaches lo; it overshoots
    the fractional optimum by at most one patch, which the window absorbs. The returned
    mip_gap is the gap to that LP lower bound, so it is checked like any solver gap.
    """
    t0 = time.perf_counter()
    m, C = co["m_p"], co["C_p"]
    order = np.argsort(C / np.maximum(m, 1.0), kind="mergesort")
    cum = np.cumsum(m[order])
    k = int(np.searchsorted(cum, lo))                  # first index with cum >= lo
    if k >= order.size:
        raise SystemExit("min-cost: the budget window cannot be reached")
    x = np.zeros(co["n_patches"])
    x[order[:k + 1]] = 1.0
    n_pix = float(m @ x)
    if not (lo <= n_pix <= hi):
        raise SystemExit(f"min-cost greedy fill gave {n_pix:.0f} pixels, outside [{lo}, {hi}]")
    prev = float(cum[k - 1]) if k > 0 else 0.0
    lp_bound = float(C[order[:k]].sum() + C[order[k]] / m[order[k]] * (lo - prev))
    cost = float(C @ x)
    gap = (cost - lp_bound) / max(abs(cost), 1e-30)
    dt = time.perf_counter() - t0
    print(f"  [min_cost] greedy fill in {dt:.2f}s  cost={cost:.6g}  LP bound={lp_bound:.6g}  "
          f"gap={gap:.3e}")
    rec = {"status": "optimal" if gap < 1e-4 else "greedy", "status_code": 0,
           "status_msg": "greedy fill vs LP relaxation", "mip_gap": gap,
           "dual_bound": lp_bound, "runtime_s": dt}
    return x, rec


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate(x_p, co, zsel, cov, pm, problem, eff, level, c_cap, lo, hi):
    """Check one solution three ways. Aborts on any disagreement."""
    # HiGHS returns integers within its integrality tolerance (~1e-6), so the incumbent is
    # rounded and every reported number is computed from the rounded plan. Guard that the
    # rounding is a tidy-up and not a silent change of solution.
    raw_x = np.asarray(x_p[:co["n_patches"]], float)
    dev = float(np.max(np.abs(raw_x - np.rint(raw_x)))) if raw_x.size else 0.0
    if dev > 1e-4:
        raise SystemExit(f"[{level}] solver returned a fractional patch variable "
                         f"(max deviation {dev:.3e}); refusing to round.")
    x_p = np.rint(raw_x).astype(np.int64)
    sel_pix = convert_patch_decisions_to_pixels(x_p, pm["restoration_patches"], co["n_rest"])
    n_pix = int(sel_pix.sum())

    b_ilp = benefit_from_matrix(co, zsel, cov, x_p.astype(float))
    c_ilp = float(co["C_p"] @ x_p)

    # 1. float64 self-check: the same maths, computed by raster dilation instead.
    sel_flat = co["rest_idx"][sel_pix == 1]
    b_eval, _ = evaluate(sel_flat, co["d_raster"], co["cost_raster"], co["shape"],
                         eff["neighbor_radius"], eff["neighbor_effect_decay"])
    b_f64 = -b_eval
    # evaluate() sums cost in float32; the ILP sums the same values in float64, so the
    # like-for-like reference is a float64 gather of the cost raster.
    c_eval = float(np.asarray(co["cost_raster"]).ravel()[sel_flat].astype(np.float64).sum())
    _assert_close(b_ilp, b_f64, VALIDATE_RTOL_SELF, level, "benefit (ILP vs float64)")
    _assert_close(c_ilp, c_eval, VALIDATE_RTOL_SELF, level, "cost (ILP vs float64)")

    # 2. the engine itself.
    x_full = np.zeros(problem.n_restoration_pixels + problem.n_conversion_pixels, dtype=int)
    x_full[:co["n_rest"]] = sel_pix
    raw = problem.evaluate_raw_objectives(x_full)
    bi = problem.objective_names.index("restoration_benefit")
    ci = problem.objective_names.index("implementation_cost")
    _assert_close(-float(raw[bi]), b_ilp, VALIDATE_RTOL_BENEFIT, level, "benefit (vs engine)")
    _assert_close(float(raw[ci]), c_ilp, VALIDATE_RTOL_COST_ENGINE, level, "cost (vs engine)")

    # 3. constraints actually hold.
    if not (lo <= n_pix <= hi):
        raise SystemExit(f"[{level}] pixel count {n_pix} outside window [{lo}, {hi}]")
    if np.isfinite(c_cap) and c_ilp > c_cap + 1e-9:
        raise SystemExit(f"[{level}] cost {c_ilp:.6f} exceeds cap {c_cap:.6f}")

    norm = problem._normalize_objective_vector(list(raw))
    return dict(sel_pix=sel_pix.astype(np.uint8), n_pixels=n_pix,
                n_patches=int(x_p.sum()), cost=c_ilp, benefit_raw=float(raw[bi]),
                benefit_pos=b_ilp, benefit_norm=float(norm[bi]), cost_norm=float(norm[ci]))


def _assert_close(got, want, rtol, level, what):
    err = abs(got - want) / max(abs(want), 1e-30)
    if not (err < rtol):
        raise SystemExit(
            f"\nMISMATCH at level '{level}': {what}\n"
            f"  ILP value    {got!r}\n  reference    {want!r}\n"
            f"  abs error    {abs(got - want):.6e}\n  rel error    {err:.6e}  (tol {rtol:.0e})\n"
            "Stopping. The model disagrees with the evaluator; do not adjust the tolerance.")


def preflight(co, zsel, cov, pm, problem, eff, lo, hi, seed=0):
    """Validate the assembled matrix on known selections before spending solver time."""
    print("\n-- pre-flight --")
    n_p = co["n_patches"]
    rng = np.random.default_rng(seed)

    empty = np.zeros(n_p)
    b0 = benefit_from_matrix(co, zsel, cov, empty)
    c0 = float(co["C_p"] @ empty)
    assert b0 == 0.0 and c0 == 0.0, f"empty selection gave ({b0}, {c0})"
    print("  empty selection            benefit 0, cost 0  OK")

    # One patch, and a budget-sized random draw: both checked against the engine.
    one = np.zeros(n_p)
    one[int(np.argmax(co["D_p"]))] = 1.0
    validate(one, co, zsel, cov, pm, problem, eff, "preflight_one_patch", np.inf, 0, hi)
    print("  single best patch          matches engine  OK")

    target_patches = max(int(round(0.5 * (lo + hi) / max(co["m_p"].mean(), 1.0))), 1)
    pick = rng.choice(n_p, size=min(target_patches, n_p), replace=False)
    rand = np.zeros(n_p)
    rand[pick] = 1.0
    n_pix = int(co["m_p"] @ rand)
    validate(rand, co, zsel, cov, pm, problem, eff, "preflight_random", np.inf, 0, n_pix)
    print(f"  random plan ({n_pix} pixels)  matches engine  OK")


# ---------------------------------------------------------------------------
# NSGA-II overlay
# ---------------------------------------------------------------------------
def _load_one(pkl_path, cfg):
    """Non-dominated raw objectives from one results pkl. Guards the config."""
    res = pickle_load(pkl_path)
    names = list(res["objective_names"])
    if names != OBJECTIVE_NAMES:
        raise SystemExit(f"{pkl_path}: objective_names {names} != {OBJECTIVE_NAMES}; "
                         "this pkl is not a benefit-vs-cost run.")

    ic = res["initial_conditions"]
    sparams = res.get("scenario_params") or {}
    rc = res.get("run_config") or {}
    bad = []
    got_region = ic.get("region")
    if got_region is not None and str(got_region) != cfg["region"]:
        bad.append(f"region {got_region!r} != {cfg['region']!r}")
    got_scen = rc.get("condition_scenario")
    if got_scen is not None and str(got_scen) != cfg["condition_scenario"]:
        bad.append(f"condition_scenario {got_scen!r} != {cfg['condition_scenario']!r}")
    got_frac = sparams.get("max_restoration_fraction")
    if got_frac is not None and float(got_frac) != cfg["max_restoration_fraction"]:
        bad.append(f"max_restoration_fraction {got_frac} != {cfg['max_restoration_fraction']}")
    if bad:
        raise SystemExit(f"{pkl_path} does not match this ILP run:\n  "
                         + "\n  ".join(bad)
                         + "\nPass a pkl from the same configuration, or adjust the flags.")

    nd = np.asarray(res["is_nondominated"], bool)
    raw = np.asarray(res["objectives_raw"], float)[nd]
    print(f"  {os.path.basename(str(pkl_path))}: {raw.shape[0]} flagged non-dominated")
    return raw


def load_nsga2_front(pkl_paths, cfg):
    """Pooled non-dominated (cost, benefit_pos) points from one or more results pkls.

    Seeds of one scenario are pooled, then re-filtered to a single non-dominated set.
    """
    raw = np.vstack([_load_one(p, cfg) for p in pkl_paths])
    keep = nondominated_2d(raw)                       # both minimised, benefit negated
    raw = raw[keep]
    order = np.argsort(raw[:, 1], kind="mergesort")
    print(f"  NSGA-II front: {len(pkl_paths)} run(s) pooled -> {raw.shape[0]} non-dominated")
    return raw[order, 1], -raw[order, 0]               # cost, benefit_pos


def make_plot(df, nsga_cost, nsga_ben, out_png, cfg, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = df[df["level"].str.startswith("eps")].sort_values("cost")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    if nsga_cost is not None:
        ax.scatter(nsga_cost, nsga_ben, s=22, c="#7a7a7a", alpha=0.65,
                   label=f"NSGA-II non-dominated ({cfg['n_nsga_runs']} run(s) pooled)", zorder=2)
    ax.step(eps["C_j"], eps["benefit_bound"], where="post", ls="--", lw=1.4,
            c="#c0392b", label="ILP upper bound (HiGHS dual)", zorder=3)
    ax.plot(eps["cost"], eps["benefit_pos"], "-o", ms=5, lw=1.6, c="#1f4e79",
            label="ILP incumbent", zorder=4)
    for _, r in df[~df["level"].str.startswith("eps")].iterrows():
        ax.scatter([r["cost"]], [r["benefit_pos"]], marker="*", s=150, zorder=5,
                   label=r["level"].replace("_", " "))

    ax.set_xlabel("implementation cost (raw)")
    ax.set_ylabel("restoration benefit (raw, higher is better)")
    ax.set_title(f"Exact vs NSGA-II front - {cfg['region']} / {cfg['condition_scenario']}\n"
                 f"budget {cfg['target_pixels']} pixels, window [{lo}, {hi}] "
                 f"(+/-{cfg['pixel_tolerance']:.0%})", fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_png}")


# ---------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m Core_optimisation.exact_front",
        description="Exact benefit-cost front by ILP, for comparison with NSGA-II.")
    p.add_argument("--results-pkl", nargs="+", default=None,
                   help="NSGA-II results .pkl file(s) to overlay; several are pooled into one "
                        "non-dominated set. Default: NSGA2_PKLS at the top of this script")
    p.add_argument("--region", default="Bern")
    p.add_argument("--condition-scenario", default="global_all")
    p.add_argument("--ecosystem", default="combined",
                   help="'combined' maps to the loader's 'all', as run_custom_nsga2.py does")
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--max-restoration-fraction", type=float, default=0.05)
    p.add_argument("--pixel-tolerance", type=float, default=0.01)
    p.add_argument("--n-levels", type=int, default=10)
    p.add_argument("--time-limit", type=float, default=600.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--sample-fraction", type=float, default=None,
                   help="pass-through to load_initial_conditions for a tractable pilot")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--quiet", action="store_true", help="suppress the HiGHS solver log")
    return p.parse_args(argv)


def main(argv):
    a = parse_args(argv)
    names = a.results_pkl or NSGA2_PKLS
    if not names:
        raise SystemExit("No NSGA-II pkls: fill NSGA2_PKLS in this script or pass --results-pkl.")
    a.results_pkl = [str(q if os.path.dirname(q) else RESULTS_DIR / q) for q in map(str, names)]
    out_dir = ensure(a.out_dir or (OUTPUTS / "exact_front"))
    eco = "all" if a.ecosystem == "combined" else a.ecosystem
    disp = not a.quiet

    scenario_params = {"max_restoration_fraction": a.max_restoration_fraction,
                       "abiotic_effect": 0.01, "biotic_effect": 0.01,
                       "normalize_objectives": True, "min_patch_size": 1,
                       "sampling_strategy": "scattered"}

    print("=" * 74)
    print(f"EXACT FRONT  region={a.region}  scenario={a.condition_scenario}  eco={eco}")
    print("=" * 74)

    t0 = time.perf_counter()
    ic = load_initial_conditions(
        ".", objectives=["restoration_benefit", "cost"], region=a.region,
        ecosystem=eco, sample_fraction=a.sample_fraction, sample_seed=a.sample_seed,
        aggregation_factor=None, condition_scenario=a.condition_scenario,
    )
    print(f"  data loaded in {time.perf_counter() - t0:.1f}s")

    pm = create_patch_mappings(ic, patch_size=a.patch_size)
    problem = RestorationProblem(ic, scenario_params)
    if problem.objective_names != OBJECTIVE_NAMES:
        raise SystemExit(f"unexpected objective order {problem.objective_names}")

    eff = effect_params_of(scenario_params)
    co = build_coefficients(ic, pm, eff)
    zsel, cov = build_coverage(co, eff["neighbor_radius"])

    target = int(problem.max_action_pixels)
    lo = int(target * (1 - a.pixel_tolerance))
    hi = int(target * (1 + a.pixel_tolerance))
    A, lb, ub = assemble(co, zsel, cov, lo, hi)
    n_p, n_z = co["n_patches"], zsel.size
    integrality = np.concatenate([np.ones(n_p), np.zeros(n_z)])
    c_obj = -np.concatenate([co["D_p"], co["n"][zsel]])

    # ---- diagnostics, before any solving -----------------------------------
    print("\n-- model --")
    print(f"  eligible pixels      {co['n_rest']:,}   patches {n_p:,}")
    print(f"  z variables          {n_z:,}  (pruned {co['n_rest'] - n_z:,} with n_k == 0)")
    print(f"  variables            {n_p + n_z:,}  ({n_p:,} binary, {n_z:,} continuous)")
    print(f"  constraints          {A.shape[0]:,}   nonzeros {A.nnz:,}  "
          f"({A.data.nbytes / 2**20 + A.indices.nbytes / 2**20:.0f} MB)")
    print(f"  budget               target {target:,}  window [{lo:,}, {hi:,}]  "
          f"(+/-{a.pixel_tolerance:.0%})")
    nz_d, nz_c = co["d"][co["d"] > 0], co["c"][co["c"] > 0]
    print(f"  d_k  (>0)            min {nz_d.min():.3e}  max {nz_d.max():.3e}  "
          f"ratio {nz_d.max() / nz_d.min():.1e}")
    print(f"  c_k  (>0)            min {nz_c.min():.3e}  max {nz_c.max():.3e}  "
          f"ratio {nz_c.max() / nz_c.min():.1e}")
    print(f"  patches per z row    mean {cov.nnz / max(n_z, 1):.1f}  max {np.diff(cov.indptr).max()}")

    preflight(co, zsel, cov, pm, problem, eff, lo, hi)

    # ---- cost anchors ------------------------------------------------------
    print("\n-- anchors --")
    x_min, rec_min = solve_min_cost(co, lo, hi)
    C_min = float(co["C_p"] @ np.rint(x_min))

    ub_open = ub.copy()
    ub_open[-1] = np.inf
    x_max, rec_max = solve(c_obj, A, lb, ub_open, integrality,
                           a.time_limit, a.mip_gap, disp, "max_benefit")
    if x_max is None:
        raise SystemExit("max-benefit solve returned no solution: " + rec_max["status_msg"])
    C_max = float(co["C_p"] @ np.rint(x_max[:n_p]))
    print(f"  C_min = {C_min:.6g}   C_max = {C_max:.6g}")
    if C_max < C_min:
        raise SystemExit(f"C_max ({C_max:.6g}) < C_min ({C_min:.6g}); check the model.")

    # ---- epsilon sweep -----------------------------------------------------
    # (name, cost cap, x, record, objective was benefit) - min_cost minimises cost, so its
    # dual bound is a cost bound and carries no information about benefit.
    levels = [("min_cost", np.inf, x_min, rec_min, False),
              ("max_benefit", np.inf, x_max, rec_max, True)]
    caps = np.linspace(C_min, C_max, a.n_levels)
    print(f"\n-- epsilon sweep ({a.n_levels} levels) --")
    for j, cap in enumerate(caps):
        ub_j = ub.copy()
        ub_j[-1] = float(cap)
        xj, recj = solve(c_obj, A, lb, ub_j, integrality, a.time_limit, a.mip_gap,
                         disp, f"eps_{j:02d} C<={cap:.6g}")
        if xj is None:
            print(f"    no incumbent at C_j={cap:.6g} ({recj['status']}); recorded as NaN")
        levels.append((f"eps_{j:02d}", float(cap), xj, recj, True))

    # ---- validate, write ---------------------------------------------------
    print("\n-- validation --")
    rows = []
    for name, cap, xv, rec, is_benefit in levels:
        bound = -float(rec["dual_bound"]) if is_benefit else np.nan
        row = {"level": name, "C_j": cap, "benefit_bound": bound}
        if xv is None:
            row.update({k: np.nan for k in
                        ("cost", "benefit_raw", "benefit_pos", "benefit_norm",
                         "cost_norm", "n_pixels", "n_patches")})
        else:
            v = validate(xv, co, zsel, cov, pm, problem, eff, name, cap, lo, hi)
            np.save(out_dir / f"sel_{name}.npy", v.pop("sel_pix"))
            row.update(v)
            # The rounded incumbent can sit a hair above the dual bound, because the bound
            # is reported for the solver's own near-integral vector. Anything beyond the
            # integrality tolerance means the objective vector and the matrix disagree.
            if np.isfinite(bound) and v["benefit_pos"] > bound * (1 + 1e-6) + 1e-9:
                print(f"    WARNING: {name} incumbent {v['benefit_pos']:.9g} exceeds its "
                      f"dual bound {bound:.9g} by more than solver tolerance")
        row.update({k: rec[k] for k in ("status", "status_msg", "mip_gap", "runtime_s")})
        rows.append(row)
        print(f"  {name:14s} OK")

    df = pd.DataFrame(rows)[[
        "level", "C_j", "cost", "benefit_raw", "benefit_pos", "benefit_norm", "cost_norm",
        "benefit_bound", "n_pixels", "n_patches", "status", "status_msg", "mip_gap",
        "runtime_s"]]
    csv_path = out_dir / "exact_front.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    cfg = {"region": a.region, "condition_scenario": a.condition_scenario,
           "ecosystem": eco, "patch_size": a.patch_size,
           "max_restoration_fraction": a.max_restoration_fraction,
           "pixel_tolerance": a.pixel_tolerance, "target_pixels": target,
           "budget_window": [lo, hi], "n_levels": a.n_levels,
           "time_limit": a.time_limit, "mip_gap": a.mip_gap,
           "sample_fraction": a.sample_fraction, "results_pkl": a.results_pkl,
           "n_nsga_runs": len(a.results_pkl),
           "effect_params": eff, "solver": "scipy.optimize.milp (HiGHS)"}

    nsga_cost = nsga_ben = None
    try:
        nsga_cost, nsga_ben = load_nsga2_front(a.results_pkl, cfg)
    except SystemExit as e:
        print(f"\n  overlay skipped: {e}")
    make_plot(df, nsga_cost, nsga_ben, out_dir / "exact_front.png", cfg, lo, hi)

    write_report(out_dir / "report.json", "exact_front", {
        "C_min": C_min, "C_max": C_max,
        "n_variables": int(n_p + n_z), "n_binary": int(n_p), "n_constraints": int(A.shape[0]),
        "n_nonzeros": int(A.nnz),
        "n_optimal": int((df["status"] == "optimal").sum()), "n_levels_solved": len(df),
        "total_runtime_s": float(df["runtime_s"].sum()),
        "worst_mip_gap": float(np.nanmax(df["mip_gap"].to_numpy())),
    }, cfg)
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
