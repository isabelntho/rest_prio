"""
Dynamic Restoration Optimisation
=================================
Prototype multi-objective optimisation where the *timing* of restoration is
the decision variable.  Each restoration patch is assigned one of T+1
possible values:

    0          — never restore
    1 … T      — restore at TIME_STEPS[k-1]

Three objectives are minimised simultaneously (NSGA-III):

    F[0]  final_landscape_rp      — negated landscape-wide anomaly improvement
                                     at the final year (direct + neighbour
                                     spillover effects; mirrors static method)
    F[1]  cumulative_landscape_rp — negated sum of landscape improvement at
                                     each time step (rewards early action)
    F[2]  total_cost              — undiscounted implementation cost (positive)

Constraints:
    G[k]  = pixel_count_at_step_k − per_step_max_pixels  ≤ 0
            (at-most budget per step; ≥0 = infeasible)

Spatial interactions mirror the static optimisation: restoration_effect() is
called for each candidate solution to propagate neighbour spillover effects,
so that nearby selections show diminishing returns in the objective.  Recovery
fractions are applied per-pixel for directly-restored cells; spillover
neighbours use the mean recovery of selected pixels at the relevant time step.

Usage
-----
Call :func:`run_dynamic_optimization_instance` from ``run_dynamic.py`` or
interactively.  ``initial_conditions`` must already contain ``patch_mappings``
and ``n_restoration_patches`` (populated by
:func:`Core_optimisation.patch_approach.create_patch_mappings`).
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import uniform_filter

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from .patch_approach import aggregate_patch_scores_from_pixel_scores
from .resto_anom import restoration_effect
from .results_saving import save_results_with_reports

# ---------------------------------------------------------------------------
# Recovery functions
# Each fn(t: float) -> float in [0, 1] where t = years since restoration.
# ---------------------------------------------------------------------------

def _fast_recovery(t: float) -> float:
    """Exponential approach — 90% recovered by ~12 years."""
    return 1.0 - np.exp(-t / 5.0)


def _gradual_recovery(t: float) -> float:
    """Linear ramp — full recovery in 25 years."""
    return float(np.clip(t / 25.0, 0.0, 1.0))


def _delayed_sigmoid_recovery(t: float) -> float:
    """Sigmoid with lag — slow start, rapid mid-phase, plateau."""
    return 1.0 / (1.0 + np.exp(-0.5 * (t - 10.0)))


def _partial_recovery(t: float) -> float:
    """Exponential approach capped at 60% — persistent residual anomaly."""
    return 0.6 * (1.0 - np.exp(-t / 8.0))


RECOVERY_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    "fast":    _fast_recovery,
    "gradual": _gradual_recovery,
    "delayed": _delayed_sigmoid_recovery,
    "partial": _partial_recovery,
}


def compute_pv_constant(
    recovery_fn: Callable[[float], float],
    discount_rate: float,
    t_max: float = 200.0,
    n_points: int = 2000,
) -> float:
    """∫₀^t_max recovery_fn(τ) * exp(-discount_rate * τ) dτ (trapezoidal approximation)."""
    tau = np.linspace(0.0, t_max, n_points)
    integrand = np.array([recovery_fn(float(t)) for t in tau]) * np.exp(-discount_rate * tau)
    return float(np.trapezoid(integrand, tau))

# ---------------------------------------------------------------------------
# Landscape context helper
# ---------------------------------------------------------------------------

# KERNEL_SIZE is derived at runtime from the actual pixel size so that the
# 500 m neighbourhood radius is preserved regardless of aggregation factor.
# At 100 m resolution: radius_px = round(500/100) = 5 → KERNEL_SIZE = 11.
_CONTEXT_RADIUS_M = 500  # physical neighbourhood radius (metres)


def _compute_dynamic_context_2d(
    abiotic_2d: np.ndarray,
    biotic_2d: np.ndarray,
    elig_mask: np.ndarray,
    kernel_size: int = 11,  # default = 500 m @ 100 m resolution; override for other resolutions
) -> np.ndarray:
    """
    Focal mean of anomaly values over restoration-eligible neighbours.

    Computes the mean of ``(abiotic + biotic) / 2`` over the box kernel for
    each pixel, excluding the centre pixel itself.  Returns a 2-D array in
    the same shape as the inputs; values are negative (anomaly scale) and
    approach 0 as neighbours recover.

    Parameters
    ----------
    abiotic_2d, biotic_2d : np.ndarray  (2-D, full grid)
    elig_mask : np.ndarray              (2-D bool, restoration-eligible pixels)
    kernel_size : int                   Box-kernel side length (must be odd).

    Returns
    -------
    np.ndarray  2-D, same shape as inputs.  Zero where no eligible neighbours.
    """
    # Combined anomaly (mean of abiotic + biotic)
    combined = ((abiotic_2d + biotic_2d) / 2.0).astype(np.float64)

    elig_float = elig_mask.astype(np.float64)

    # Count eligible pixels in focal window (×kernel² corrects uniform_filter scaling)
    k2 = float(kernel_size ** 2)
    cnt_focal = uniform_filter(elig_float, size=kernel_size, mode="constant") * k2
    # Exclude centre pixel from neighbour count
    cnt_neigh = np.maximum(cnt_focal - elig_float, 0.0)

    # Sum of combined anomaly over eligible pixels in focal window
    anom_elig = np.where(elig_mask, combined, 0.0)
    sum_focal = uniform_filter(anom_elig, size=kernel_size, mode="constant") * k2
    # Exclude centre pixel contribution
    sum_neigh = sum_focal - np.where(elig_mask, combined, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ctx_2d = np.where(cnt_neigh > 0, sum_neigh / cnt_neigh, 0.0)

    return ctx_2d


def _compute_landscape_context_2d(
    abiotic_2d: np.ndarray,
    biotic_2d: np.ndarray,
    elig_mask: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """
    Proportion of restoration-eligible neighbouring pixels in good condition.

    Good condition: abiotic_anomaly > 0 AND biotic_anomaly > 0.
    Excludes the centre pixel itself from the focal window.

    Parameters
    ----------
    abiotic_2d, biotic_2d : np.ndarray  (2-D, full grid)
    elig_mask : np.ndarray              (2-D bool, restoration-eligible pixels)
    radius_px : int                     Focal radius in pixels (500 m physical).

    Returns
    -------
    np.ndarray  2-D, values in [0, 1].  0 where pixel has no eligible neighbours.
    """
    ksize = 2 * int(radius_px) + 1
    elig_f = elig_mask.astype(np.float64)
    cnt_focal = uniform_filter(elig_f, size=ksize, mode="constant") * (ksize ** 2)
    cnt_neigh = np.maximum(cnt_focal - elig_f, 0.0)  # exclude centre pixel
    good = ((abiotic_2d > 0) & (biotic_2d > 0) & elig_mask).astype(np.float64)
    good_focal = uniform_filter(good, size=ksize, mode="constant") * (ksize ** 2)
    good_neigh = np.maximum(good_focal - good, 0.0)  # exclude centre pixel
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt_neigh > 0, good_neigh / cnt_neigh, 0.0)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_pathway(
    x_patches: np.ndarray,
    time_steps: List[int],
    initial_conditions: Dict,
    pv_weights: np.ndarray,
    effect_params: Dict,
    lc_radius_px: int,
    lc_baseline_2d: np.ndarray,
) -> Dict:
    """
    Evaluate a candidate timing solution.

    Parameters
    ----------
    x_patches : np.ndarray  (n_restoration_patches,)  int, values 0 … n_steps.
        0 = never restore; k = restore at time_steps[k-1].
    time_steps : list[int]  Restoration years, e.g. [2025, 2030, 2040, 2050].
    initial_conditions : dict  From :func:`load_initial_conditions` with
        patch_mappings already injected.
    pv_weights : np.ndarray  length n_steps.  Per-step present-value weights:
        ``pv_weights[k] = C_fn * exp(-r * (t_k - t_0))``, where C_fn is the
        recovery-scenario integral and r is the annual discount rate.
    effect_params : dict  Direct + neighbour effect parameters (from
        :class:`DynamicRestorationProblem`).

    Returns
    -------
    dict with keys:
        final_landscape_rp      float  (≤ 0; negated landscape-wide improvement at t_end)
        cumulative_landscape_rp float  (≤ 0; negated sum of improvement at each step)
        final_landscape_context float  (≤ 0; negated improvement in LC proportion at t_end)
        total_cost              float  (≥ 0)
        per_step_pixel_counts   list[int]  length == n_steps
    """
    n_steps = len(time_steps)
    rest_patches = initial_conditions["patch_mappings"]["restoration_patches"]
    patch_to_pixels = rest_patches["patch_to_pixels"]
    n_rest_patches = rest_patches["n_patches"]

    n_rest_pixels = initial_conditions["n_restoration_pixels"]
    rest_indices = initial_conditions["restoration_eligible_indices"]  # global flat indices
    rest_mask = initial_conditions["restoration_eligible_mask"]        # 2-D bool
    shape = initial_conditions["shape"]
    cost_2d = initial_conditions["implementation_cost"]
    x_conv_zeros = np.zeros(initial_conditions["n_conversion_pixels"], dtype=int)

    # ------------------------------------------------------------------
    # Build per-pixel timing array from patch decisions
    # ------------------------------------------------------------------
    timing_step = np.zeros(n_rest_pixels, dtype=int)   # 0=never, k=step index (1-based)
    per_step_pixel_counts = []

    x_rest = x_patches[:n_rest_patches]
    for k in range(1, n_steps + 1):
        patches_at_k = np.where(x_rest == k)[0]
        step_count = 0
        for pid in patches_at_k:
            if pid in patch_to_pixels:
                pix = np.atleast_1d(patch_to_pixels[pid])
                pix = pix[(pix >= 0) & (pix < n_rest_pixels)]
                if pix.size > 0:
                    timing_step[pix] = k
                    step_count += len(pix)
        per_step_pixel_counts.append(step_count)

    restored_mask_elig = timing_step > 0

    # ------------------------------------------------------------------
    # Total cost (undiscounted, order-independent)
    # ------------------------------------------------------------------
    total_cost = 0.0
    if np.any(restored_mask_elig):
        global_idx = rest_indices[restored_mask_elig]
        rows, cols = np.divmod(global_idx, shape[1])
        total_cost = float(np.sum(cost_2d[rows, cols]))

    # ------------------------------------------------------------------
    # Landscape-wide improvement objectives
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Final-state objectives at t_end
    # One restoration_effect call shared by final_landscape_rp AND
    # final_landscape_context (avoids a duplicate call).
    # ------------------------------------------------------------------
    x_final = (timing_step > 0).astype(int)
    if np.any(x_final):
        updated_final = restoration_effect(
            x_final, x_conv_zeros, initial_conditions, effect_params
        )
        delta_ab = updated_final["abiotic_anomaly"] - initial_conditions["abiotic_anomaly"]
        delta_bi = updated_final["biotic_anomaly"]  - initial_conditions["biotic_anomaly"]
        delta_rp_final = (delta_ab + delta_bi) / 2.0

        # Raw spatial improvement over eligible pixels — no recovery weighting.
        # Recovery-based timing tradeoffs are captured by cumulative_landscape_rp.
        final_improvement = float(np.sum(delta_rp_final[rest_mask]))

        # Landscape context improvement at t_end (one extra uniform_filter call)
        lc_updated = _compute_landscape_context_2d(
            updated_final["abiotic_anomaly"], updated_final["biotic_anomaly"],
            rest_mask, lc_radius_px,
        )
        lc_improvement = float(np.sum((lc_updated - lc_baseline_2d)[rest_mask]))
    else:
        final_improvement = 0.0
        lc_improvement = 0.0

    final_landscape_rp      = -final_improvement  # negate: minimise = maximise
    final_landscape_context = -lc_improvement      # negate: minimise = maximise

    # Cumulative PV objective: Σ_k spatial_RP(x_at_k) * pv_weights[k]
    cumulative_improvement = 0.0
    for k in range(1, n_steps + 1):
        x_at_k = ((timing_step > 0) & (timing_step <= k)).astype(int)
        if not np.any(x_at_k):
            continue
        updated_k = restoration_effect(x_at_k, x_conv_zeros, initial_conditions, effect_params)
        delta_ab_k = updated_k["abiotic_anomaly"] - initial_conditions["abiotic_anomaly"]
        delta_bi_k = updated_k["biotic_anomaly"]  - initial_conditions["biotic_anomaly"]
        delta_rp_k = (delta_ab_k + delta_bi_k) / 2.0
        cumulative_improvement += float(np.sum(delta_rp_k[rest_mask])) * pv_weights[k - 1]
    cumulative_landscape_rp = -cumulative_improvement  # negate: minimise = maximise

    return {
        "final_landscape_rp":      final_landscape_rp,
        "cumulative_landscape_rp": cumulative_landscape_rp,
        "final_landscape_context": final_landscape_context,
        "total_cost":              total_cost,
        "per_step_pixel_counts":   per_step_pixel_counts,
    }


# ---------------------------------------------------------------------------
# Pymoo Problem
# ---------------------------------------------------------------------------

class DynamicRestorationProblem(ElementwiseProblem):
    """
    Multi-objective problem for timing-based restoration optimisation.

    Decision variables
    ------------------
    n_var = n_restoration_patches  (integer, values 0 … n_steps)

    Objectives (all minimised)
    --------------------------
    F[0]  normalised final_landscape_rp      (≤ 0; negated landscape-wide improvement at t_end)
    F[1]  normalised cumulative_landscape_rp  (≤ 0; negated cumulative landscape improvement)
    F[2]  normalised final_landscape_context  (≤ 0; negated improvement in LC proportion at t_end)
    F[3]  normalised total_cost               (≥ 0)

    Constraints
    -----------
    G[k] = pixel_count_at_step_k − per_step_max_pixels  ≤ 0   for k = 0…n_steps-1
    """

    def __init__(
        self,
        initial_conditions: Dict,
        time_steps: List[int],
        recovery_fn: Callable[[float], float],
        per_step_max_pixels: int,
        n_jobs: int = 1,
        discount_rate: float = 0.03,
    ) -> None:
        self.initial_conditions = initial_conditions
        self.time_steps = time_steps
        self.recovery_fn = recovery_fn
        self.per_step_max_pixels = per_step_max_pixels
        self.discount_rate = discount_rate
        self.n_steps = len(time_steps)
        self.n_restoration_patches = initial_conditions["n_restoration_patches"]

        # Build effect_params using the same resolution-aware formula as RestorationProblem
        _pixel_size = abs(initial_conditions["transform"].a)
        _neighbor_radius_px = max(1, round(300.0 / _pixel_size))
        self.effect_params = {
            "abiotic_effect": 0.01,
            "biotic_effect": 0.01,
            "neighbor_radius": _neighbor_radius_px,  # 300 m physical radius
            "neighbor_effect_decay": 0.2,
        }

        # Landscape context params: 500 m physical radius, binary good-condition threshold
        _lc_radius_px = max(1, round(500.0 / _pixel_size))
        self.lc_radius_px = _lc_radius_px
        _elig_mask = initial_conditions["restoration_eligible_mask"]
        self.lc_baseline_2d = _compute_landscape_context_2d(
            initial_conditions["abiotic_anomaly"],
            initial_conditions["biotic_anomaly"],
            _elig_mask,
            _lc_radius_px,
        )

        # Per-step planning-horizon weights:
        #   w_k = Σ_{j≥k} recovery_fn(t_j − t_k) × exp(−r × (t_j − t_0))
        # This is the discounted cumulative recovery a patch accrues from t_k
        # to the end of the planning window.  Fast-recovery scenarios have a
        # much larger w_1/w_4 ratio than slow/partial scenarios, so the
        # optimizer is incentivised to time restoration differently per scenario.
        _t0 = float(time_steps[0])
        self.pv_weights = np.array([
            sum(
                recovery_fn(float(time_steps[j]) - float(time_steps[k]))
                * np.exp(-discount_rate * (float(time_steps[j]) - _t0))
                for j in range(k, len(time_steps))
            )
            for k in range(len(time_steps))
        ])
        self.C_fn = float(self.pv_weights[0])  # retained for metadata/logging

        super().__init__(
            n_var=self.n_restoration_patches,
            n_obj=4,
            n_constr=self.n_steps,
            xl=np.zeros(self.n_restoration_patches, dtype=int),
            xu=np.full(self.n_restoration_patches, self.n_steps, dtype=int),
            type_var=int,
        )

        # Precompute normalisation denominators
        self._norm = self._compute_normalisations()

    def _compute_normalisations(self) -> Dict[str, float]:
        """Derive per-objective normalisation denominators.

        Uses a full-restoration baseline call to restoration_effect() so that
        the landscape-wide objectives are normalised on the same scale as the
        maximum possible improvement (direct + spillover, full recovery).
        """
        ic = self.initial_conditions
        x_restore_all = np.ones(ic["n_restoration_pixels"], dtype=int)
        x_conv_zeros  = np.zeros(ic["n_conversion_pixels"], dtype=int)

        updated_max = restoration_effect(
            x_restore_all, x_conv_zeros, ic, self.effect_params
        )
        delta_ab = updated_max["abiotic_anomaly"] - ic["abiotic_anomaly"]
        delta_bi = updated_max["biotic_anomaly"]  - ic["biotic_anomaly"]
        delta_max = (delta_ab + delta_bi) / 2.0

        elig_mask = ic["restoration_eligible_mask"]
        norm_final = max(1e-9, float(np.sum(delta_max[elig_mask])))
        norm_cumul  = max(1e-9, norm_final * float(self.pv_weights[0]))

        # Landscape context normalization: max improvement when all pixels restored
        lc_updated_max = _compute_landscape_context_2d(
            updated_max["abiotic_anomaly"], updated_max["biotic_anomaly"],
            elig_mask, self.lc_radius_px,
        )
        norm_lc = max(1e-9, float(np.sum((lc_updated_max - self.lc_baseline_2d)[elig_mask])))

        rest_indices = ic["restoration_eligible_indices"]
        cost_2d = ic["implementation_cost"]
        rows, cols = np.divmod(rest_indices, ic["shape"][1])
        norm_cost = max(1e-9, float(np.sum(cost_2d[rows, cols])))

        return {
            "final":  norm_final,
            "cumul":  norm_cumul,
            "lc":     norm_lc,
            "cost":   norm_cost,
        }

    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs) -> None:
        pw = simulate_pathway(
            x, self.time_steps, self.initial_conditions,
            self.pv_weights, self.effect_params,
            self.lc_radius_px, self.lc_baseline_2d,
        )
        n = self._norm
        F = [
            pw["final_landscape_rp"]      / n["final"],  # ≤ 0, minimise
            pw["cumulative_landscape_rp"] / n["cumul"],  # ≤ 0, minimise
            pw["final_landscape_context"] / n["lc"],     # ≤ 0, minimise
            pw["total_cost"]              / n["cost"],   # ≥ 0, minimise
        ]
        G = [
            float(max(0, cnt - self.per_step_max_pixels))
            for cnt in pw["per_step_pixel_counts"]
        ]
        out["F"] = F
        out["G"] = G

    def evaluate_raw(self, x: np.ndarray) -> Dict:
        """Return raw (un-normalised) objectives as a dict.  For result packaging."""
        return simulate_pathway(
            x, self.time_steps, self.initial_conditions,
            self.pv_weights, self.effect_params,
            self.lc_radius_px, self.lc_baseline_2d,
        )


# ---------------------------------------------------------------------------
# Pymoo Operators
# ---------------------------------------------------------------------------

class DynamicPatchSampling(Sampling):
    """
    Initial population for the dynamic integer problem.

    Each solution is built by randomly assigning each patch a timing value
    0 … n_steps, weighted by patch scores.  Per-step budgets are enforced
    afterwards by demoting the lowest-scoring excess patches to 0.
    """

    def __init__(
        self,
        n_restoration_patches: int,
        n_steps: int,
        per_step_max_pixels: int,
        patch_to_pixels: Dict,
        patch_scores: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.n_patches = n_restoration_patches
        self.n_steps = n_steps
        self.per_step_max_pixels = per_step_max_pixels
        self.patch_to_pixels = patch_to_pixels
        self.patch_scores = (
            np.asarray(patch_scores, dtype=np.float64)
            if patch_scores is not None
            else np.ones(n_restoration_patches, dtype=np.float64)
        )
        # Precompute pixels-per-patch
        self.pixels_per_patch = np.array([
            len(np.atleast_1d(patch_to_pixels.get(pid, [])))
            for pid in range(n_restoration_patches)
        ], dtype=np.int64)

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        X = np.zeros((n_samples, self.n_patches), dtype=int)
        # Scale scores so that mean selection probability is ~60%; higher-score
        # (more degraded) patches are more likely to be assigned a timing step.
        weights = np.clip(self.patch_scores, 1e-12, None)
        weights_norm = weights / weights.mean()  # mean = 1.0
        select_probs = np.clip(0.60 * weights_norm, 0.0, 0.99)

        for i in range(n_samples):
            x = np.zeros(self.n_patches, dtype=int)
            for pid in range(self.n_patches):
                if np.random.random() < select_probs[pid]:
                    x[pid] = np.random.randint(1, self.n_steps + 1)
                # else x[pid] = 0 (never restore)
            X[i] = self._enforce_budgets(x)
        return X

    def _enforce_budgets(self, x: np.ndarray) -> np.ndarray:
        """Demote lowest-score excess patches to 0 so per-step budget is met."""
        x = x.copy()
        for k in range(1, self.n_steps + 1):
            at_k = np.where(x == k)[0]
            pixel_count = int(np.sum(self.pixels_per_patch[at_k]))
            if pixel_count <= self.per_step_max_pixels:
                continue
            # Sort descending: greedily keep highest-score (most degraded) patches first
            scores_at_k = self.patch_scores[at_k]
            order = np.argsort(scores_at_k)[::-1]
            running = 0
            keep = np.zeros(len(at_k), dtype=bool)
            for idx in order:
                pid = at_k[idx]
                new_running = running + int(self.pixels_per_patch[pid])
                if new_running <= self.per_step_max_pixels:
                    running = new_running
                    keep[idx] = True
            x[at_k[~keep]] = 0
        return x


class DynamicPatchRepair(Repair):
    """
    Repair operator: enforce per-step pixel-count budgets.

    For each time step, if the pixel count exceeds ``per_step_max_pixels``,
    the lowest-scoring patches at that step are demoted to 0 until feasible.
    """

    def __init__(
        self,
        n_steps: int,
        per_step_max_pixels: int,
        patch_to_pixels: Dict,
        n_restoration_patches: int,
        patch_scores: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.per_step_max_pixels = per_step_max_pixels
        self.patch_to_pixels = patch_to_pixels
        self.n_patches = n_restoration_patches
        self.patch_scores = (
            np.asarray(patch_scores, dtype=np.float64)
            if patch_scores is not None
            else np.ones(n_restoration_patches, dtype=np.float64)
        )
        self.pixels_per_patch = np.array([
            len(np.atleast_1d(patch_to_pixels.get(pid, [])))
            for pid in range(n_restoration_patches)
        ], dtype=np.int64)

    def _do(self, problem, X, **kwargs):
        X = X.copy().astype(int)
        for i in range(len(X)):
            X[i] = self._repair_individual(X[i])
        return X

    def _repair_individual(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        # Clip to valid range
        x = np.clip(x, 0, self.n_steps)
        for k in range(1, self.n_steps + 1):
            at_k = np.where(x == k)[0]
            if at_k.size == 0:
                continue
            pixel_count = int(np.sum(self.pixels_per_patch[at_k]))
            if pixel_count <= self.per_step_max_pixels:
                continue
            # Sort descending: greedily keep highest-score (most degraded) patches first
            scores_at_k = self.patch_scores[at_k]
            order = np.argsort(scores_at_k)[::-1]
            running = 0
            for idx in order:
                pid = at_k[idx]
                new_running = running + int(self.pixels_per_patch[pid])
                if new_running > self.per_step_max_pixels:
                    x[pid] = 0  # patch doesn't fit; demote
                else:
                    running = new_running
        return x


class IntegerTimingMutation(Mutation):
    """
    Uniform integer mutation: each variable is independently reassigned to a
    random value in {0, 1, …, n_steps} with probability ``prob_var``.
    """

    def __init__(self, n_steps: int, prob_var: float = 0.01) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.prob_var = prob_var

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        X = X.copy().astype(int)
        for i in range(len(X)):
            mask = np.random.random(X.shape[1]) < self.prob_var
            if np.any(mask):
                X[i, mask] = np.random.randint(0, self.n_steps + 1, size=int(np.sum(mask)))
        return X


# ---------------------------------------------------------------------------
# Main optimisation runner
# ---------------------------------------------------------------------------

def run_dynamic_optimization_instance(
    initial_conditions: Dict,
    time_steps: List[int],
    recovery_fn: Callable[[float], float],
    recovery_fn_name: str,
    per_step_budget_fraction: float = 0.05,
    pop_size: Optional[int] = None,
    n_generations: int = 100,
    n_partitions: int = 6,
    n_jobs: int = 1,
    save_results: bool = True,
    output_dir: str = "results_files",
    verbose: bool = True,
    random_seed: Optional[int] = None,
    run_label: str = "",
    run_config: Optional[Dict] = None,
    discount_rate: float = 0.03,
) -> Optional[Dict]:
    """
    Run one dynamic restoration optimisation instance.

    Parameters
    ----------
    initial_conditions : dict
        Must include ``patch_mappings`` and ``n_restoration_patches``
        (call :func:`Core_optimisation.patch_approach.create_patch_mappings` first).
    time_steps : list[int]
        Restoration planning years, e.g. ``[2025, 2030, 2040, 2050]``.
    recovery_fn : callable  fn(years: float) -> float in [0, 1].
    recovery_fn_name : str  Label used in output filenames.
    per_step_budget_fraction : float
        Maximum fraction of restoration pixels that may be restored at each
        time step (default 0.05 = 5%).
    pop_size : int or None
        Population size.  None = number of NSGA-III reference directions
        (recommended; set by ``n_partitions`` and 4 objectives).
    n_generations : int   Maximum number of NSGA-III generations.
    n_partitions : int    Das-Dennis simplex partitions (default 6 → 84 dirs).
    n_jobs : int          Parallel evaluation threads (1 = serial).
    save_results : bool   Write results pickle to ``output_dir``.
    output_dir : str      Directory for output files.
    verbose : bool        Print progress.
    random_seed : int or None   RNG seed.
    run_label : str       Human-readable label for filenames.
    run_config : dict     Snapshot of run configuration for the registry.

    Returns
    -------
    dict or None  Optimisation results, or None on failure.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_rest_pixels = initial_conditions["n_restoration_pixels"]
    n_steps = len(time_steps)
    # per_step_budget_fraction is the TOTAL budget across all steps; divide by n_steps
    total_max_pixels = max(1, int(per_step_budget_fraction * n_rest_pixels))
    per_step_max_pixels = max(1, total_max_pixels // n_steps)

    if verbose:
        print(f"\n=== Dynamic Optimisation [{recovery_fn_name}] ===")
        print(f"  Time steps: {time_steps}")
        print(f"  Restoration pixels: {n_rest_pixels:,}")
        print(f"  Total budget: {total_max_pixels:,} ({per_step_budget_fraction*100:.1f}% of eligible)")
        print(f"  Per-step max pixels: {per_step_max_pixels:,} (budget / {n_steps} steps)")
        print(f"  n_partitions={n_partitions}, n_generations={n_generations}, seed={random_seed}")
        print(f"  discount_rate={discount_rate}, C_fn will be computed per scenario")

    # ------------------------------------------------------------------
    # Build patch scores (proxy: negated mean rp per patch = more degraded → higher score)
    # ------------------------------------------------------------------
    rest_indices = initial_conditions["restoration_eligible_indices"]
    flat_abiotic = initial_conditions["abiotic_anomaly"].flatten()
    flat_biotic = initial_conditions["biotic_anomaly"].flatten()
    rp_1d = (flat_abiotic[rest_indices] + flat_biotic[rest_indices]) / 2.0
    # Score = -rp (positive for degraded pixels; higher score = higher priority)
    pixel_scores = -rp_1d

    patch_scores = aggregate_patch_scores_from_pixel_scores(
        patch_mappings=initial_conditions["patch_mappings"],
        restoration_pixel_scores=pixel_scores,
        conversion_pixel_scores=None,
        mode="mean",
    )
    # Only take restoration portion (aggregate_patch_scores returns [rest + conv])
    n_rest_patches = initial_conditions["n_restoration_patches"]
    patch_scores = patch_scores[:n_rest_patches]

    patch_to_pixels = initial_conditions["patch_mappings"]["restoration_patches"]["patch_to_pixels"]

    # ------------------------------------------------------------------
    # Build problem and operators
    # ------------------------------------------------------------------
    problem = DynamicRestorationProblem(
        initial_conditions=initial_conditions,
        time_steps=time_steps,
        recovery_fn=recovery_fn,
        per_step_max_pixels=per_step_max_pixels,
        n_jobs=n_jobs,
        discount_rate=discount_rate,
    )

    sampling = DynamicPatchSampling(
        n_restoration_patches=n_rest_patches,
        n_steps=n_steps,
        per_step_max_pixels=per_step_max_pixels,
        patch_to_pixels=patch_to_pixels,
        patch_scores=patch_scores,
    )

    repair = DynamicPatchRepair(
        n_steps=n_steps,
        per_step_max_pixels=per_step_max_pixels,
        patch_to_pixels=patch_to_pixels,
        n_restoration_patches=n_rest_patches,
        patch_scores=patch_scores,
    )

    mutation = IntegerTimingMutation(
        n_steps=n_steps,
        prob_var=max(0.001, 3.0 / n_rest_patches),
    )

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
    _pop_size = pop_size if pop_size is not None else len(ref_dirs)

    if verbose:
        print(f"  n_var={n_rest_patches}, pop_size={_pop_size}, ref_dirs={len(ref_dirs)}")

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=_pop_size,
        sampling=sampling,
        crossover=UniformCrossover(),
        mutation=mutation,
        repair=repair,
    )
    termination = get_termination("n_gen", n_generations)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    try:
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=random_seed,
            verbose=False,
            save_history=False,
        )
    except Exception as exc:
        print(f"  ERROR during optimisation [{recovery_fn_name}]: {exc}")
        return None

    elapsed = time.perf_counter() - t_start
    if verbose:
        n_nd = len(result.F) if result.F is not None else 0
        n_pop = len(result.pop.get("F")) if result.pop is not None else 0
        print(f"  Done in {elapsed/60:.1f} min — {n_nd} non-dominated / {n_pop} total solutions")

    # ------------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------------
    X_full = result.pop.get("X").astype(int)
    F_full_norm = result.pop.get("F")
    X_nd = result.X.astype(int) if result.X is not None else X_full[:0]
    F_nd_norm = result.F if result.F is not None else F_full_norm[:0]

    # Raw objectives for all population members
    raw_list = []
    for xi in X_full:
        pw = problem.evaluate_raw(xi)
        raw_list.append([
            pw["final_landscape_rp"],
            pw["cumulative_landscape_rp"],
            pw["final_landscape_context"],
            pw["total_cost"],
        ])
    F_full_raw = np.array(raw_list, dtype=float)

    nd_set = {tuple(row) for row in X_nd}
    is_nd = np.array([tuple(row) in nd_set for row in X_full])

    optimization_results = {
        "run_label": run_label,
        "recovery_fn_name": recovery_fn_name,
        "time_steps": time_steps,
        "per_step_max_pixels": per_step_max_pixels,
        "per_step_budget_fraction": per_step_budget_fraction,
        "objective_names": ["final_landscape_rp", "cumulative_landscape_rp", "final_landscape_context", "total_cost"],
        "objectives": F_full_raw,
        "objectives_raw": F_full_raw,
        "objectives_normalized": F_full_norm,
        "decisions": X_full,
        "is_nondominated": is_nd,
        "n_solutions": len(X_full),
        "n_nondominated_solutions": int(np.sum(is_nd)),
        "problem_info": {
            "n_restoration_pixels": n_rest_pixels,
            "n_restoration_patches": n_rest_patches,
            "n_steps": n_steps,
        },
        "algorithm_info": {
            "pop_size": _pop_size,
            "n_generations": n_generations,
            "n_partitions": n_partitions,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "random_seed": random_seed,
            "discount_rate": discount_rate,
            "C_fn": float(problem.C_fn),
            "pv_weights": problem.pv_weights.tolist(),
        },
        "normalisations": problem._norm,
        "run_config": run_config or {},
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_slug = run_label.replace(" ", "_") if run_label else recovery_fn_name
        fname = f"dynamic_{label_slug}_{ts}.pkl"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "wb") as fh:
            pickle.dump(optimization_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"  Saved: {fpath}")
        optimization_results["output_path"] = fpath

    return optimization_results
