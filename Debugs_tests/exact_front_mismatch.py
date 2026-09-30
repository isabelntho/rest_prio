"""Where does the exact_front ILP disagree with the engine? Direct vs spillover, per plan.

    pixi run python Debugs_tests/exact_front_mismatch.py [--region Bern] [--sample-fraction F]

Builds the same ILP coefficients as Core_optimisation/exact_front.py but never solves. For
four real plans (best patch, clustered, dispersed, budget-sized random) it splits benefit
into a DIRECT term (restored cells) and a SPILLOVER term (their un-restored neighbours) and
compares three evaluations of each:

    ilp      D_p @ x  and  n_k over covered & unrestored cells   (the ILP matrix)
    f64      raster dilation in float64                          (uncertainty_analysis.evaluate)
    engine   restoration_effect, run with radius 0 for the direct term

It also compares the spillover CELL SETS ILP vs engine and, for cells in one set only, the
offset to the nearest restored pixel, which separates radius / kernel / eligibility /
grid-edge causes.
"""
import argparse

import numpy as np
from scipy import ndimage

from _common import REPO_ROOT  # noqa: F401  (puts the repo root on sys.path)

from Core_optimisation.data_loader import load_initial_conditions
from Core_optimisation.patch_approach import create_patch_mappings, convert_patch_decisions_to_pixels
from Core_optimisation.optimization_engine import RestorationProblem, restoration_effect
from Core_optimisation.uncertainty_analysis import effect_params_of, plan_masks
from Core_optimisation.exact_front import (
    build_coefficients, build_coverage, benefit_from_matrix, OBJECTIVE_NAMES)


def engine_delta(sel_pix, ic, params):
    """Per-cell (abiotic + biotic) improvement raster from restoration_effect."""
    up = restoration_effect(sel_pix, ic, dict(params))
    return ((up["abiotic_anomaly"] - ic["abiotic_anomaly"])
            + (up["biotic_anomaly"] - ic["biotic_anomaly"]))


def compare_plan(name, x_p, co, zsel, cov, pm, ic, problem, eff):
    shape = co["shape"]
    sel_pix = convert_patch_decisions_to_pixels(x_p.astype(int), pm["restoration_patches"],
                                                co["n_rest"])
    sel_flat = co["rest_idx"][sel_pix == 1]
    elig = np.asarray(ic["restoration_eligible_mask"], bool)

    covered = (cov @ x_p) > 0.5
    unrestored = x_p[co["patch_of_pixel"][zsel]] < 0.5
    ilp_direct = float(co["D_p"] @ x_p)
    ilp_spill = float(co["n"][zsel][covered & unrestored].sum())
    ilp_cells = co["rest_idx"][zsel][covered & unrestored]

    sel_m, nb_m = plan_masks(sel_flat, shape, eff["neighbor_radius"])
    f64_direct = float(co["d_raster"].ravel()[sel_flat].sum())
    f64_spill = float(eff["neighbor_effect_decay"] * co["d_raster"][nb_m].sum())

    p0 = dict(problem.effect_params, neighbor_radius=0)
    delta0 = engine_delta(sel_pix, ic, p0)
    delta = engine_delta(sel_pix, ic, problem.effect_params)
    eng_direct = float(delta0[elig].sum())
    eng_spill = float(delta[elig].sum()) - eng_direct
    eng_cells = np.flatnonzero(((delta > 0) & ~sel_m & elig).ravel())

    print(f"\n[{name}]  {int(x_p.sum())} patches, {int(sel_pix.sum())} pixels, "
          f"{ilp_cells.size} ILP spill cells, {eng_cells.size} engine spill cells")
    print(f"  {'':10s}{'direct':>16s}{'spillover':>16s}")
    for lab, dd, ss in (("ilp", ilp_direct, ilp_spill), ("f64", f64_direct, f64_spill),
                        ("engine", eng_direct, eng_spill)):
        print(f"  {lab:10s}{dd:16.9e}{ss:16.9e}")
    b_ilp = benefit_from_matrix(co, zsel, cov, x_p)
    b_eng = -float(problem.evaluate_raw_objectives(
        np.r_[sel_pix, np.zeros(problem.n_conversion_pixels, int)])[0])
    c_ilp = float(co["C_p"] @ x_p)
    c_eng = float(problem.evaluate_raw_objectives(
        np.r_[sel_pix, np.zeros(problem.n_conversion_pixels, int)])[1])
    print(f"  total ilp {b_ilp:.9e}   engine {b_eng:.9e}   rel err {abs(b_ilp - b_eng) / abs(b_eng):.2e}"
          f"   (tol 1e-5)")
    print(f"  cost  ilp {c_ilp:.9e}   engine {c_eng:.9e}   rel err {abs(c_ilp - c_eng) / abs(c_eng):.2e}"
          f"   (tol 1e-9)")

    only_ilp = np.setdiff1d(ilp_cells, eng_cells)
    only_eng = np.setdiff1d(eng_cells, ilp_cells)
    print(f"  cell sets: ilp-only {only_ilp.size}, engine-only {only_eng.size}, "
          f"both {np.intersect1d(ilp_cells, eng_cells).size}")
    if only_ilp.size or only_eng.size:
        _, (near_r, near_c) = ndimage.distance_transform_edt(~sel_m, return_indices=True)
        for lab, cells in (("ilp-only", only_ilp), ("engine-only", only_eng)):
            for c in cells[:6]:
                r, k = divmod(int(c), shape[1])
                dr, dc = r - near_r[r, k], k - near_c[r, k]
                print(f"    {lab:11s} cell ({r},{k}) offset to nearest restored ({dr:+d},{dc:+d})"
                      f"  d2={dr * dr + dc * dc}  eligible={bool(elig[r, k])}")

    both = np.intersect1d(ilp_cells, eng_cells)
    if both.size:
        pos = {int(c): i for i, c in enumerate(co["rest_idx"])}
        n_both = np.array([co["n"][pos[int(c)]] for c in both])
        d_both = delta.ravel()[both]
        rel = np.abs(d_both - n_both) / np.maximum(np.abs(n_both), 1e-30)
        print(f"  per-cell credit on shared cells: max rel diff engine vs n_k = {rel.max():.3e}")


def clustered_plan(co, k, seed_patch):
    """The k patches whose centroids are nearest the seed patch (adjacent plan)."""
    rows, cols = np.divmod(co["rest_idx"], co["shape"][1])
    cnt = np.bincount(co["patch_of_pixel"], minlength=co["n_patches"]).clip(1)
    cr = np.bincount(co["patch_of_pixel"], weights=rows, minlength=co["n_patches"]) / cnt
    cc = np.bincount(co["patch_of_pixel"], weights=cols, minlength=co["n_patches"]) / cnt
    d2 = (cr - cr[seed_patch]) ** 2 + (cc - cc[seed_patch]) ** 2
    x = np.zeros(co["n_patches"])
    x[np.argsort(d2, kind="mergesort")[:k]] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Bern")
    ap.add_argument("--condition-scenario", default="global_all")
    ap.add_argument("--patch-size", type=int, default=2)
    ap.add_argument("--max-restoration-fraction", type=float, default=0.05)
    ap.add_argument("--sample-fraction", type=float, default=None)
    ap.add_argument("--n-patches", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    scenario_params = {"max_restoration_fraction": a.max_restoration_fraction,
                       "abiotic_effect": 0.01, "biotic_effect": 0.01,
                       "normalize_objectives": True, "min_patch_size": 1,
                       "sampling_strategy": "scattered"}
    ic = load_initial_conditions(
        ".", objectives=["restoration_benefit", "cost"], region=a.region, ecosystem="all",
        sample_fraction=a.sample_fraction, sample_seed=42, aggregation_factor=None,
        condition_scenario=a.condition_scenario)
    pm = create_patch_mappings(ic, patch_size=a.patch_size)
    problem = RestorationProblem(ic, scenario_params)
    assert problem.objective_names == OBJECTIVE_NAMES, problem.objective_names
    eff = effect_params_of(scenario_params)
    co = build_coefficients(ic, pm, eff)
    zsel, cov = build_coverage(co, eff["neighbor_radius"])

    print("\n-- config --")
    print(f"  radius     ilp {eff['neighbor_radius']}   engine {problem.effect_params['neighbor_radius']}")
    print(f"  decay      ilp {eff['neighbor_effect_decay']}   engine {problem.effect_params['neighbor_effect_decay']}")
    print(f"  effects    ilp {eff['abiotic_effect']}/{eff['biotic_effect']}   engine "
          f"{problem.effect_params['abiotic_effect']}/{problem.effect_params['biotic_effect']}")
    elig = np.asarray(ic["restoration_eligible_mask"], bool)
    same = np.array_equal(np.flatnonzero(elig.ravel()), np.sort(co["rest_idx"]))
    print(f"  eligible mask == restoration_eligible_indices set: {same}")
    print(f"  dtypes     abiotic {ic['abiotic_anomaly'].dtype}  biotic {ic['biotic_anomaly'].dtype}")

    rng = np.random.default_rng(a.seed)
    n_p, k = co["n_patches"], min(a.n_patches, co["n_patches"])
    best = int(np.argmax(co["D_p"]))
    one = np.zeros(n_p)
    one[best] = 1.0
    rand = np.zeros(n_p)
    rand[rng.choice(n_p, size=k, replace=False)] = 1.0
    plans = [("best patch", one),
             ("clustered", clustered_plan(co, k, best)),
             ("dispersed", rand),
             ("random, budget-sized",
              np.isin(np.arange(n_p), rng.choice(
                  n_p, size=max(int(round(problem.max_action_pixels / max(co["m_p"].mean(), 1.0))), 1),
                  replace=False)).astype(float))]
    # cheapest plan: the ILP's cost anchor is this kind of plan (low benefit per pixel), where
    # the engine's float32 (updated - base) cancellation is largest in RELATIVE terms.
    lo = int(problem.max_action_pixels * 0.99)
    order = np.argsort(co["C_p"] / np.maximum(co["m_p"], 1.0), kind="mergesort")
    cheap = np.zeros(n_p)
    cheap[order[np.cumsum(co["m_p"][order]) <= lo]] = 1.0
    plans.append(("cheapest per pixel (greedy)", cheap))
    for name, x in plans:
        compare_plan(name, x, co, zsel, cov, pm, ic, problem, eff)


if __name__ == "__main__":
    main()
