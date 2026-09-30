"""Simple-ranking baseline front for the 2-objective (restoration_benefit, cost) problem.

Rank decision units (pixels, or tiles) by  D_u - lambda * C_u  where D_u is the unit's DIRECT
gain and C_u its cost (both scaled by their maximum), then take units in rank order until the
pixel budget is reached. Sweeping lambda gives a front. Spatial interaction (spillover) is
deliberately NOT in the score - that is the point of the baseline: it is what a planner gets
by ranking sites one at a time.

The finished plans are scored with the SAME engine model the GA was scored with
(RestorationProblem.evaluate_raw_objectives), so radius, decay and spillover_to_restored are
all honoured and the numbers sit on the GA's own axes (benefit negated, cost positive).

    front = build_ranking_front(ic, scenario_params, use_patch_approach, patch_size,
                                pixel_tolerance, out_dir, problem=problem)
"""
import os

import numpy as np
import pandas as pd

from .patch_approach import create_patch_mappings
from .resto_anom import RestorationProblem
from .uncertainty_analysis import build_variant_layer, effect_params_of, nondominated_2d

N_LAMBDA = 15   # 0, 13 log-spaced in [1e-2, 1e2], and inf (pure cheapest-first)


def lambda_grid(n_lambda=N_LAMBDA):
    return np.concatenate([[0.0], np.logspace(-2, 2, n_lambda - 2), [np.inf]])


def _unit_table(ic, eff, use_patch_approach, patch_size):
    """Per-unit direct gain D, cost C, pixel count m, and the unit id of each eligible pixel
    (None in pixel mode, where a unit is a pixel)."""
    if use_patch_approach:
        # Imported here: exact_front sets thread-count env vars at import time.
        from .exact_front import build_coefficients
        # Reuse the run's own tile mapping when it is the same tiling: a second copy of
        # ~1M-entry patch dicts is hundreds of MB per worker for nothing.
        if ic.get("patch_approach_enabled") and ic.get("patch_size") == patch_size:
            pm = ic["patch_mappings"]
        else:
            pm = create_patch_mappings(ic, patch_size=patch_size)
        co = build_coefficients(ic, pm, eff)
        return co["D_p"], co["C_p"], co["m_p"], co["patch_of_pixel"]

    rest_idx = np.asarray(ic["restoration_eligible_indices"], np.int64)
    d_raster = build_variant_layer(ic["abiotic_anomaly"], ic["biotic_anomaly"],
                                   ic["restoration_eligible_mask"], eff)
    D = d_raster.ravel()[rest_idx]
    C = np.asarray(ic["implementation_cost"]).ravel()[rest_idx].astype(np.float64)
    return D, C, np.ones(D.size), None


def _fill_to_budget(order, m, target, lo, hi):
    """Prefix of `order` whose pixel total is closest to `target`, inside [lo, hi]."""
    cum = np.cumsum(m[order])
    j = int(np.searchsorted(cum, target))            # first prefix reaching the target
    cands = [k for k in (j - 1, j) if 0 <= k < cum.size]
    best = min(cands, key=lambda k: abs(cum[k] - target))
    if lo <= cum[best] <= hi:
        return order[:best + 1]
    # Coarse units can straddle the window: walk the ranking and skip any unit that overshoots.
    chosen, cur = [], 0.0
    for u in order:
        if cur + m[u] > hi:
            continue
        chosen.append(u)
        cur += m[u]
        if cur >= lo:
            break
    return np.asarray(chosen, np.int64)


def build_ranking_front(ic, scenario_params, use_patch_approach, patch_size, pixel_tolerance,
                        out_dir, problem=None, n_lambda=N_LAMBDA):
    """Build, score and save the ranking front. Returns the result DataFrame."""
    problem = problem or RestorationProblem(ic, scenario_params, pixel_tolerance=pixel_tolerance)
    # Only the direct (per-unit, additive) term is used here, so the flag that changes the
    # spillover term is allowed through.
    eff = effect_params_of(scenario_params, allow_spillover_to_restored=True)

    D, C, m, unit_of_pixel = _unit_table(ic, eff, use_patch_approach, patch_size)
    Dn, Cn = D / max(D.max(), 1e-30), C / max(C.max(), 1e-30)

    n_rest = int(problem.n_restoration_pixels)
    target = int(problem.max_action_pixels)
    lo, hi = int(target * (1 - pixel_tolerance)), int(target * (1 + pixel_tolerance))
    bi = problem.objective_names.index("restoration_benefit")
    ci = problem.objective_names.index("implementation_cost")

    rows, plans = [], {}
    for j, lam in enumerate(lambda_grid(n_lambda)):
        score = -Cn if np.isinf(lam) else Dn - lam * Cn
        order = np.argsort(-score, kind="stable")
        units = _fill_to_budget(order, m, target, lo, hi)

        if unit_of_pixel is None:
            sel = np.sort(units)
        else:
            chosen = np.zeros(D.size, dtype=bool)
            chosen[units] = True
            sel = np.flatnonzero(chosen[unit_of_pixel])
        x = np.zeros(n_rest + int(problem.n_conversion_pixels), dtype=int)
        x[sel] = 1
        raw = problem.evaluate_raw_objectives(x)
        rows.append(dict(lam=lam, n_units=int(units.size), n_pixels=int(sel.size),
                         benefit_raw=float(raw[bi]), cost=float(raw[ci])))
        plans[f"lam{j:02d}"] = sel.astype(np.int32)

    df = pd.DataFrame(rows)
    df["in_window"] = df["n_pixels"].between(lo, hi)
    df["nondominated"] = nondominated_2d(df[["benefit_raw", "cost"]].to_numpy())
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "ranking_front.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "plans.npz"), **plans)
    return df
