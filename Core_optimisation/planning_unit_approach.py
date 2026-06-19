"""
Planning Unit Optimization
==========================

Coarse-resolution multi-objective optimization where each decision variable
represents a single binary choice for a *planning unit* — either a regular
grid cell or an administrative boundary (municipality/district/canton) — rather
than a per-pixel or per-patch decision.

When a planning unit is selected, ALL eligible pixels within it (both
restoration-eligible and conversion-eligible) are activated and passed to
``restoration_effect()`` unchanged.  This dramatically reduces ``n_var``
compared with pixel or patch modes while keeping the full pixel-level accuracy
of objective evaluation.

MODES
-----
  grid  – non-overlapping N×N pixel blocks.  Analogous to patch_approach but at
          a much coarser scale (e.g. 20×20 px = 2 km at 100 m resolution).
  admin – one unit per administrative boundary polygon (municipality/district).

KEY DESIGN CHOICES
------------------
* Single binary per unit (no separate restoration/conversion sub-vectors).
* Budget enforced through repair only (no pymoo G constraint), consistent with
  the patch approach.  Tolerance ±20 % (wider than the ±10–15 % used for
  patches because unit sizes vary more).
* Scoring and repair follow the same patterns as PatchAwareSampling /
  PatchRepair, scaled to unit granularity.
* Inherits from RestorationProblem, overrides n_var and _evaluate only.

USAGE
-----
  See run_planning_unit_optimization.py for a complete standalone example.

Created: May 2026
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.operators.crossover.hux import HUX
from pymoo.util.ref_dirs import get_reference_directions

from .resto_anom import (
    RestorationProblem,
    ProgressCallback,
    build_fixed_ref_point,
    _build_algorithm,
    build_repair_scores,
    build_per_objective_repair_scores,
    InstrumentedBitflipMutation,
)
from .data_loader import load_planning_units
from .results_saving import save_results_with_reports

logger = logging.getLogger("resto_prio")


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _safe_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    t = max(float(temperature), 1e-9)
    z = (x - np.max(x)) / t
    exp_z = np.exp(np.clip(z, -60.0, 60.0))
    s = np.sum(exp_z)
    if s <= 0 or not np.isfinite(s):
        return np.full_like(exp_z, 1.0 / len(exp_z), dtype=np.float64)
    return exp_z / s


# =============================================================================
# PIXEL EXPANSION AND BUDGET COUNTING
# =============================================================================

def expand_unit_decisions_to_pixels(
    unit_decisions: np.ndarray,
    unit_mappings: Dict,
    n_restoration_pixels: int,
    n_conversion_pixels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand unit-level binary decisions into pixel-level binary arrays.

    Parameters
    ----------
    unit_decisions        : (n_units,) binary int array
    unit_mappings         : planning unit mappings dict
    n_restoration_pixels  : total number of restoration-eligible pixels
    n_conversion_pixels   : total number of conversion-eligible pixels

    Returns
    -------
    x_restore : (n_restoration_pixels,) binary int8 array
    x_convert : (n_conversion_pixels,) binary int8 array
    """
    x_restore = np.zeros(n_restoration_pixels, dtype=np.int8)
    x_convert = np.zeros(n_conversion_pixels, dtype=np.int8)

    selected = np.where(unit_decisions == 1)[0]
    if selected.size == 0:
        return x_restore, x_convert

    rest_map = unit_mappings['unit_to_restoration_pixels']
    conv_map = unit_mappings['unit_to_conversion_pixels']

    for uid in selected:
        uid = int(uid)
        rp = rest_map.get(uid)
        if rp is not None and len(rp) > 0:
            x_restore[rp] = 1
        cp = conv_map.get(uid)
        if cp is not None and len(cp) > 0:
            x_convert[cp] = 1

    return x_restore, x_convert


def count_selected_pixels(unit_decisions: np.ndarray, unit_mappings: Dict) -> int:
    """Total eligible pixels activated by the current unit decisions."""
    return int(np.sum(unit_mappings['unit_pixel_counts'][unit_decisions == 1]))


# =============================================================================
# UNIT SCORES
# =============================================================================

def compute_unit_scores(
    unit_mappings: Dict,
    pixel_scores: np.ndarray,
    mode: str = 'mean',
) -> np.ndarray:
    """
    Aggregate per-pixel repair scores into one score per planning unit.

    Uses only the restoration-eligible pixel scores, since unit selection
    activates all eligible pixels (restoration and conversion).

    Parameters
    ----------
    unit_mappings : planning unit mappings dict
    pixel_scores  : (n_restoration_pixels,) float array from build_repair_scores
    mode          : 'mean' or 'quantile90'

    Returns
    -------
    (n_units,) float64 array, normalized to [0, 1]
    """
    n_units  = unit_mappings['n_units']
    rest_map = unit_mappings['unit_to_restoration_pixels']
    scores   = np.zeros(n_units, dtype=np.float64)

    for uid in range(n_units):
        pix = rest_map.get(uid)
        if pix is None or len(pix) == 0:
            continue
        pix   = np.asarray(pix, dtype=int)
        valid = pix[(pix >= 0) & (pix < len(pixel_scores))]
        if valid.size == 0:
            continue
        vals = pixel_scores[valid]
        scores[uid] = (
            float(np.nanquantile(vals, 0.9)) if mode == 'quantile90'
            else float(np.nanmean(vals))
        )

    lo, hi = np.nanmin(scores), np.nanmax(scores)
    span = hi - lo
    if np.isfinite(span) and span > 1e-12:
        scores = (scores - lo) / span
    else:
        scores = np.full(n_units, 0.5, dtype=np.float64)

    return np.nan_to_num(scores, nan=0.0)


def compute_per_objective_unit_scores(
    unit_mappings: Dict,
    per_obj_pixel_scores: Dict,
) -> np.ndarray:
    """
    Per-objective unit scores for direction-aware sampling / repair.

    Parameters
    ----------
    unit_mappings         : planning unit mappings dict
    per_obj_pixel_scores  : dict{'abiotic', 'biotic', 'cost'} → (n_restoration_pixels,) arrays
                            (from build_per_objective_repair_scores)

    Returns
    -------
    np.ndarray shape (3, n_units)  — row order: abiotic, biotic, cost
    """
    return np.stack([
        compute_unit_scores(unit_mappings, per_obj_pixel_scores['abiotic']),
        compute_unit_scores(unit_mappings, per_obj_pixel_scores['biotic']),
        compute_unit_scores(unit_mappings, per_obj_pixel_scores['cost']),
    ], axis=0)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_planning_unit_approach(
    initial_conditions: Dict,
    mode: str = 'grid',
    unit_size_px: int = 20,
    workspace_dir: Optional[str] = None,
    region: str = 'Bern',
    admin_data: Optional[Dict] = None,
    max_unit_pixels: Optional[int] = None,
) -> Dict:
    """
    Attach planning unit mappings to ``initial_conditions`` in-place and return it.

    This mirrors ``initialize_patch_approach`` in resto_anom.py and must be
    called before constructing ``PlanningUnitProblem``.

    Parameters
    ----------
    initial_conditions : dict (must already be populated by load_initial_conditions)
    mode               : 'grid' or 'admin'
    unit_size_px       : grid mode only — cell side length in pixels
    workspace_dir      : admin mode only — used to load admin shapefiles if admin_data is None
    region             : admin mode only — 'Bern' or 'CH'
    admin_data         : admin mode only — pre-loaded admin data (skips file loading)
    max_unit_pixels    : admin mode only — split units larger than this threshold

    Returns
    -------
    dict : modified initial_conditions with 'unit_mappings', 'n_units',
           and 'planning_unit_approach_enabled' keys added.
    """
    if initial_conditions.get('planning_unit_approach_enabled'):
        return initial_conditions  # Already initialized

    unit_mappings = load_planning_units(
        initial_conditions,
        mode=mode,
        unit_size_px=unit_size_px,
        workspace_dir=workspace_dir,
        region=region,
        admin_data=admin_data,
        max_unit_pixels=max_unit_pixels,
    )

    initial_conditions['unit_mappings'] = unit_mappings
    initial_conditions['n_units']       = unit_mappings['n_units']
    initial_conditions['planning_unit_approach_enabled'] = True

    return initial_conditions


# =============================================================================
# PROBLEM CLASS
# =============================================================================

class PlanningUnitProblem(RestorationProblem):
    """
    Multi-objective restoration problem with planning-unit decision variables.

    Inherits all objective and normalization logic from RestorationProblem.
    Overrides ``n_var`` and ``_evaluate`` to work at unit granularity:
      - n_var = n_units  (one binary per planning unit)
      - _evaluate expands unit decisions to pixel arrays, then calls parent logic

    Budget constraint is enforced entirely by PlanningUnitRepair (no pymoo G
    hard constraint is used during NSGA-III selection, matching the patch
    approach behaviour).
    """

    def __init__(
        self,
        initial_conditions: Dict,
        scenario_params: Dict,
        n_jobs: Optional[int] = None,
        pixel_tolerance: float = 0.20,
    ):
        if not initial_conditions.get('planning_unit_approach_enabled', False):
            raise ValueError(
                "Planning unit approach not initialized. "
                "Call initialize_planning_unit_approach() first."
            )

        self.unit_mappings = initial_conditions['unit_mappings']
        self.n_units       = initial_conditions['n_units']

        # Parent __init__ sets n_var = n_restoration_pixels + n_conversion_pixels,
        # max_action_pixels, objective_names, objective_scales, etc.
        super().__init__(initial_conditions, scenario_params, n_jobs=n_jobs)

        # Override to unit-level decision space (one binary per unit)
        self.n_var = self.n_units

        self.pixel_tolerance          = pixel_tolerance
        self.target_constraint_value  = self.max_action_pixels

        print(
            f"Planning unit problem initialized:\n"
            f"  Decision variables : {self.n_units} units "
            f"({self.unit_mappings['unit_mode']} mode)\n"
            f"  Objectives         : {len(self.objective_names)} "
            f"({', '.join(self.objective_names)})\n"
            f"  Budget target      : {self.max_action_pixels} pixels "
            f"(±{pixel_tolerance*100:.0f}% tolerance)\n"
            f"  Pixel counts range : "
            f"[{self.unit_mappings['unit_pixel_counts'].min()}, "
            f"{self.unit_mappings['unit_pixel_counts'].max()}] px/unit"
        )

    def evaluate_raw_objectives(self, x_units: np.ndarray) -> List[float]:
        """
        Evaluate raw (non-normalized) objectives for a unit-level decision vector.

        Overrides RestorationProblem.evaluate_raw_objectives so that the result-
        packaging code in _package_results (which calls this method on unit-level
        X arrays) works correctly.
        """
        x_restore, x_convert = expand_unit_decisions_to_pixels(
            x_units, self.unit_mappings,
            self.n_restoration_pixels, self.n_conversion_pixels,
        )
        x_pixels = np.concatenate([x_restore, x_convert])
        return super().evaluate_raw_objectives(x_pixels)

    def _evaluate(self, x_units: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate a unit-level solution.

        Expands unit decisions to pixel-level arrays, delegates to parent
        RestorationProblem._evaluate, then overrides the constraint with the
        unit-granularity pixel count.
        """
        x_restore, x_convert = expand_unit_decisions_to_pixels(
            x_units, self.unit_mappings,
            self.n_restoration_pixels, self.n_conversion_pixels,
        )
        x_pixels = np.concatenate([x_restore, x_convert])

        # Parent computes all objectives
        super()._evaluate(x_pixels, out, *args, **kwargs)

        # Override constraint: unit-granularity pixel count vs. budget
        n_px    = count_selected_pixels(x_units, self.unit_mappings)
        target  = self.target_constraint_value
        tol     = self.pixel_tolerance
        min_px  = int(target * (1 - tol * 1.5))   # 50% wider tolerance for evaluation
        max_px  = int(target * (1 + tol * 1.5))
        if min_px <= n_px <= max_px:
            out["G"] = [0.0]
        else:
            out["G"] = [float(abs(n_px - target))]


# =============================================================================
# SAMPLING
# =============================================================================

class PlanningUnitSampling(Sampling):
    """
    Initial population generator for planning unit optimization.

    Builds solutions whose selected pixel count falls within
    [target × (1 − tolerance), target × (1 + tolerance)].

    Strategy
    --------
    * Weighted random selection: units are drawn with probability proportional to
      a blend of quality score and unit size (in pixels).
    * Warm-start seeds: one extreme solution per objective (greedy fill in
      descending score order) to anchor the Pareto front extremes.
    * Random share: ~30 % of population uses uniform weights to ensure diversity.
    """

    def __init__(
        self,
        unit_mappings: Dict,
        target_pixels: int,
        pixel_tolerance: float = 0.20,
        unit_scores: Optional[np.ndarray] = None,
        score_temperature: float = 0.5,
        random_share: float = 0.30,
        per_objective_unit_scores: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.unit_mappings              = unit_mappings
        self.target_pixels              = int(target_pixels)
        self.pixel_tolerance            = float(pixel_tolerance)
        self.score_temperature          = float(score_temperature)
        self.random_share               = float(np.clip(random_share, 0.0, 1.0))
        self.per_objective_unit_scores  = (
            np.asarray(per_objective_unit_scores, dtype=np.float64)
            if per_objective_unit_scores is not None else None
        )

        self.n_units       = unit_mappings['n_units']
        self.pixel_counts  = unit_mappings['unit_pixel_counts'].astype(np.float64)

        # Precompute blended base weights once
        if unit_scores is not None and len(unit_scores) == self.n_units:
            score_part = _safe_softmax(
                np.asarray(unit_scores, dtype=np.float64), temperature=score_temperature
            )
            size_part  = np.clip(self.pixel_counts, 0.0, None)
            s = size_part.sum()
            size_part  = size_part / s if s > 0 else np.full(self.n_units, 1.0 / self.n_units)
            self._base_weights = np.clip(
                (1.0 - self.random_share) * score_part + self.random_share * size_part,
                1e-12, None
            )
        else:
            self._base_weights = np.clip(self.pixel_counts, 1e-12, None)

    def _build_extreme_solution(self, score_vec: np.ndarray) -> np.ndarray:
        """Greedy fill in descending score order up to target_max."""
        target_max = int(self.target_pixels * (1 + self.pixel_tolerance))
        order  = np.argsort(-score_vec)
        active = np.zeros(self.n_units, dtype=int)
        total  = 0
        for idx in order:
            px = int(self.pixel_counts[idx])
            if px <= 0:
                continue
            if total + px > target_max:
                continue
            active[idx] = 1
            total += px
        return active

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        X = np.zeros((n_samples, self.n_units), dtype=int)

        target_min = int(self.target_pixels * (1 - self.pixel_tolerance))
        target_max = int(self.target_pixels * (1 + self.pixel_tolerance))

        # --- Warm seeding ---
        n_extreme = 0
        if (self.per_objective_unit_scores is not None
                and n_samples > self.per_objective_unit_scores.shape[0]):
            n_extreme = self.per_objective_unit_scores.shape[0]
            for k in range(n_extreme):
                X[n_samples - n_extreme + k] = self._build_extreme_solution(
                    self.per_objective_unit_scores[k]
                )

        # --- Stochastic construction ---
        for i in range(n_samples - n_extreme):
            w = self._base_weights.copy()
            total_w = w.sum()
            probs = w / total_w if total_w > 0 else np.full(self.n_units, 1.0 / self.n_units)

            perm    = np.random.choice(self.n_units, size=self.n_units, replace=False, p=probs)
            active  = np.zeros(self.n_units, dtype=bool)
            current = 0

            # Phase 1: fill to target_min
            phase2_start = len(perm)
            for j, idx in enumerate(perm):
                if current >= target_min:
                    phase2_start = j
                    break
                active[idx] = True
                current += int(self.pixel_counts[idx])

            # Phase 2: continue to target_pixels with early-stop probability
            for j in range(phase2_start, len(perm)):
                if current >= self.target_pixels:
                    break
                idx = int(perm[j])
                px  = int(self.pixel_counts[idx])
                if current + px > target_max:
                    continue
                if current >= target_min and np.random.random() < 0.35:
                    break
                active[idx] = True
                current += px

            X[i, active] = 1

        return X


# =============================================================================
# REPAIR
# =============================================================================

class PlanningUnitRepair(Repair):
    """
    Constraint repair for planning unit optimization.

    Enforces the pixel-count budget by adding or removing whole units, sorted
    by unit score (add high-score units; remove low-score units).  A small
    amount of randomisation prevents deterministic collapse across the population.

    Tolerance ±20 % (wider than the patch approach) to accommodate the coarser
    granularity of planning units.
    """

    def __init__(
        self,
        target_pixels: int,
        unit_mappings: Dict,
        pixel_tolerance: float = 0.20,
        unit_scores: Optional[np.ndarray] = None,
        per_objective_unit_scores: Optional[np.ndarray] = None,
        ref_dirs: Optional[np.ndarray] = None,
        top_k: int = 8,
    ):
        super().__init__()
        self.target_pixels             = int(target_pixels)
        self.unit_mappings             = unit_mappings
        self.pixel_tolerance           = float(pixel_tolerance)
        self.top_k                     = int(max(2, top_k))
        self.unit_scores               = (
            np.asarray(unit_scores, dtype=np.float64)
            if unit_scores is not None else None
        )
        self.per_objective_unit_scores = (
            np.asarray(per_objective_unit_scores, dtype=np.float64)
            if per_objective_unit_scores is not None else None
        )
        self.ref_dirs  = ref_dirs
        self.pixel_counts = unit_mappings['unit_pixel_counts'].astype(np.float64)

        # Logging
        self.repair_log = []

    def _get_scores_for_individual(self, f_values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Blend per-objective scores using current objective values (like PatchRepair)."""
        if (self.per_objective_unit_scores is None or self.ref_dirs is None
                or f_values is None):
            return self.unit_scores

        # Find the closest reference direction to the individual's objective vector
        try:
            f = np.asarray(f_values, dtype=np.float64)
            f_pos = np.clip(f, 0.0, None)
            f_norm = np.linalg.norm(f_pos)
            if f_norm < 1e-12:
                return self.unit_scores
            f_dir = f_pos / f_norm

            # Cosine similarity against all ref dirs
            ref_norms = np.linalg.norm(self.ref_dirs, axis=1, keepdims=True)
            ref_norms = np.where(ref_norms < 1e-12, 1.0, ref_norms)
            rd_normed = self.ref_dirs / ref_norms
            cos_sim   = rd_normed @ f_dir
            best_rd   = self.ref_dirs[np.argmax(cos_sim)]

            # Use reference direction as blending weights
            w = np.clip(best_rd, 0.0, None)
            w_sum = w.sum()
            if w_sum < 1e-12:
                return self.unit_scores
            w = w / w_sum

            n_obj_scores = self.per_objective_unit_scores.shape[0]
            w = w[:n_obj_scores]
            w = w / w.sum() if w.sum() > 0 else np.full(n_obj_scores, 1.0 / n_obj_scores)

            blended = np.zeros(self.unit_mappings['n_units'], dtype=np.float64)
            for k in range(n_obj_scores):
                blended += float(w[k]) * self.per_objective_unit_scores[k]
            return blended
        except Exception:
            return self.unit_scores

    def _repair_individual(
        self,
        x: np.ndarray,
        f_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Repair a single individual to satisfy the pixel-count budget."""
        x = x.copy()
        scores   = self._get_scores_for_individual(f_values)
        px_cnts  = self.pixel_counts
        target   = self.target_pixels
        tol      = self.pixel_tolerance
        min_px   = int(target * (1 - tol))
        max_px   = int(target * (1 + tol))
        current  = int(np.sum(px_cnts[x == 1]))

        max_iters = self.unit_mappings['n_units'] * 2

        # --- Add units ---
        iters = 0
        while current < min_px and iters < max_iters:
            inactive = np.where(x == 0)[0]
            if inactive.size == 0:
                break
            # Prefer units that bring us closest to target without overshooting max_px
            feasible = inactive[px_cnts[inactive] <= max_px - current]
            pool = feasible if feasible.size > 0 else inactive  # relax if nothing fits

            if scores is not None:
                pool_scores = scores[pool]
                # Stochastically pick from top-k
                top_k = min(self.top_k, len(pool))
                top_idx = np.argpartition(-pool_scores, top_k - 1)[:top_k]
                chosen = pool[np.random.choice(top_idx)]
            else:
                chosen = np.random.choice(pool)

            x[chosen] = 1
            current += int(px_cnts[chosen])
            iters += 1

        # --- Remove units ---
        iters = 0
        while current > max_px and iters < max_iters:
            active = np.where(x == 1)[0]
            if active.size == 0:
                break

            if scores is not None:
                active_scores = scores[active]
                top_k = min(self.top_k, len(active))
                # Remove from the *lowest* scoring active units
                bottom_idx = np.argpartition(active_scores, top_k - 1)[:top_k]
                chosen = active[np.random.choice(bottom_idx)]
            else:
                chosen = np.random.choice(active)

            x[chosen] = 0
            current -= int(px_cnts[chosen])
            iters += 1

        return x

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        """Repair the full population matrix."""
        pop = kwargs.get('pop')
        Xr  = X.copy()

        for i in range(len(Xr)):
            f_values = None
            if pop is not None and hasattr(pop[i], 'F') and pop[i].F is not None:
                try:
                    f_values = pop[i].get('F')
                except Exception:
                    pass

            n_before = int(np.sum(self.pixel_counts[Xr[i] == 1]))
            Xr[i]    = self._repair_individual(Xr[i], f_values)
            n_after  = int(np.sum(self.pixel_counts[Xr[i] == 1]))

            if abs(n_before - self.target_pixels) > abs(n_after - self.target_pixels):
                self.repair_log.append({'before': n_before, 'after': n_after})

        return Xr


# =============================================================================
# OPERATOR BUILDER
# =============================================================================

def build_planning_unit_operators(
    initial_conditions: Dict,
    scenario_params: Dict,
    problem: PlanningUnitProblem,
    pixel_tolerance: float = 0.20,
    warm_seeding: bool = True,
    n_partitions: int = 8,
    verbose: bool = True,
) -> Tuple:
    """
    Build sampling and repair operators for the planning unit approach.

    Returns
    -------
    (PlanningUnitSampling, PlanningUnitRepair)
    """
    unit_mappings = initial_conditions['unit_mappings']

    # Per-pixel composite score → per-unit aggregated score
    pixel_scores    = build_repair_scores(initial_conditions, scenario_params)
    unit_scores     = compute_unit_scores(unit_mappings, pixel_scores, mode='mean')

    # Per-objective scores for direction-aware repair/sampling
    per_obj_pixel   = build_per_objective_repair_scores(initial_conditions, scenario_params)
    per_obj_unit    = compute_per_objective_unit_scores(unit_mappings, per_obj_pixel)

    score_temp = float(scenario_params.get('patch_score_temperature', 0.5))
    top_k      = int(scenario_params.get('patch_repair_top_k', 8))

    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)

    sampling = PlanningUnitSampling(
        unit_mappings              = unit_mappings,
        target_pixels              = problem.target_constraint_value,
        pixel_tolerance            = pixel_tolerance,
        unit_scores                = unit_scores,
        score_temperature          = score_temp,
        random_share               = 0.30,
        per_objective_unit_scores  = per_obj_unit if warm_seeding else None,
    )

    repair = PlanningUnitRepair(
        target_pixels              = problem.target_constraint_value,
        unit_mappings              = unit_mappings,
        pixel_tolerance            = pixel_tolerance,
        unit_scores                = unit_scores,
        per_objective_unit_scores  = per_obj_unit if warm_seeding else None,
        ref_dirs                   = ref_dirs,
        top_k                      = top_k,
    )

    if verbose:
        print(
            f"Planning unit operators built: {unit_mappings['n_units']} units, "
            f"score_temp={score_temp}, top_k={top_k}, tolerance=±{pixel_tolerance*100:.0f}%"
        )

    return sampling, repair


# =============================================================================
# RUN FUNCTION
# =============================================================================

def run_planning_unit_instance(
    initial_conditions: Dict,
    scenario_params: Dict,
    unit_type: str = 'grid',
    unit_size_px: int = 20,
    workspace_dir: Optional[str] = None,
    region: str = 'Bern',
    admin_data: Optional[Dict] = None,
    max_unit_pixels: Optional[int] = None,
    pixel_tolerance: float = 0.20,
    pop_size: int = 50,
    n_generations: int = 100,
    n_partitions: int = 8,
    hv_patience: int = 15,
    hv_min_improvement: float = 1e-6,
    n_jobs: Optional[int] = None,
    random_seed: Optional[int] = None,
    warm_seeding: bool = True,
    save_results: bool = True,
    output_dir: str = '.',
    run_label: str = 'planning_unit',
    run_config: Optional[Dict] = None,
    verbose: bool = True,
    r_export_parent: Optional[str] = None,
) -> Optional[Dict]:
    """
    Run a full planning-unit multi-objective optimization.

    This function mirrors run_optimization_instance() from resto_anom.py and
    is designed as a standalone entry point for testing the planning unit
    approach without modifying the existing optimization pipeline.

    Parameters
    ----------
    initial_conditions : dict
        Populated by load_initial_conditions().  Must NOT have planning unit
        approach already initialized (this function does the initialization).
    scenario_params : dict
        Same format as used in run_custom.py (max_restoration_fraction, effects, etc.)
    unit_type : 'grid' or 'admin'
    unit_size_px : grid mode — cell side length in pixels
    workspace_dir : admin mode — directory for admin shapefile loading
    region : admin mode — 'Bern' or 'CH'
    admin_data : admin mode — pre-loaded admin data (optional)
    max_unit_pixels : admin mode — split units larger than this (None = no cap)
    pixel_tolerance : budget tolerance fraction (default 0.20 = ±20%)
    pop_size : ignored (NSGA-III pop size is set by n_partitions); kept for API parity
    n_generations : maximum generations
    n_partitions : NSGA-III Das-Dennis partitions (controls pop size)
    hv_patience : early stopping patience (generations without HV improvement)
    hv_min_improvement : minimum HV delta to reset patience
    n_jobs : parallel evaluation threads (None = auto)
    random_seed : RNG seed for reproducibility
    warm_seeding : whether to seed extreme solutions per objective
    save_results : save PKL + HTML report to output_dir
    output_dir : directory for results output
    run_label : string used in output filenames
    run_config : snapshot config dict (stored in results for traceability)
    verbose : print progress

    Returns
    -------
    dict : optimization_results compatible with run_vis.py, or None on failure
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # --- 1. Initialize planning unit approach ---
    if not initial_conditions.get('planning_unit_approach_enabled', False):
        if verbose:
            print(f"Initializing planning unit approach: mode={unit_type}, "
                  f"unit_size_px={unit_size_px if unit_type=='grid' else 'N/A'} ...")
        initial_conditions = initialize_planning_unit_approach(
            initial_conditions,
            mode=unit_type,
            unit_size_px=unit_size_px,
            workspace_dir=workspace_dir,
            region=region,
            admin_data=admin_data,
            max_unit_pixels=max_unit_pixels,
        )

    n_units = initial_conditions['n_units']

    if verbose:
        print(f"\n=== PLANNING UNIT OPTIMIZATION ===")
        print(f"  Unit type  : {unit_type}")
        print(f"  Units      : {n_units}")
        print(f"  Objectives : {scenario_params.get('objectives', 'from initial_conditions')}")
        print(f"  Generations: {n_generations} (HV patience={hv_patience})")

    # --- 2. Create problem ---
    problem = PlanningUnitProblem(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params,
        n_jobs=n_jobs,
        pixel_tolerance=pixel_tolerance,
    )

    # --- 3. Build operators ---
    sampling, repair = build_planning_unit_operators(
        initial_conditions=initial_conditions,
        scenario_params=scenario_params,
        problem=problem,
        pixel_tolerance=pixel_tolerance,
        warm_seeding=warm_seeding,
        n_partitions=n_partitions,
        verbose=verbose,
    )

    # --- 4. Build HV reference point ---
    hv_warmup = int(scenario_params.get('hv_warmup_samples', 40))
    if verbose:
        print(f"Building HV reference point ({hv_warmup} warm-up samples)...")
    fixed_ref = build_fixed_ref_point(
        problem=problem,
        sampling=sampling,
        n_samples=hv_warmup,
        margin=0.05,
        seed=42,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    # --- 5. Build algorithm (reuse _build_algorithm from resto_anom) ---
    # Override mutation prob for smaller n_var (flip ~5 units minimum)
    from pymoo.util.ref_dirs import get_reference_directions as _get_ref_dirs
    ref_dirs  = _get_ref_dirs("das-dennis", problem.n_obj, n_partitions=n_partitions)
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=HUX(),
        mutation=InstrumentedBitflipMutation(
            prob=1.0,
            prob_var=max(5.0 / n_units, 0.02),  # flip ~5 units per offspring
        ),
        repair=repair,
    )
    termination = get_termination("n_gen", n_generations)

    # --- 6. Run optimization ---
    if verbose:
        print(f"Starting optimization at {datetime.now():%H:%M}...")

    callback = ProgressCallback(
        verbose=verbose,
        n_generations=n_generations,
        ref_point=fixed_ref,
        hv_patience=hv_patience,
        hv_min_improvement=hv_min_improvement,
        save_snapshots=False,
        snapshot_dir=None,
    )

    try:
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=random_seed,
            callback=callback,
            verbose=False,
            save_history=False,
        )
    except Exception as e:
        logger.error(f"Planning unit optimization failed: {e}")
        if verbose:
            print(f"✗ Optimization failed: {e}")
        return None

    if result is None or result.F is None or len(result.F) == 0:
        if verbose:
            print("✗ Optimization produced no Pareto solutions")
        return None

    # --- 7. Package results ---
    if verbose:
        actual_gens = len(callback.hv_callback.hv_history)
        reason      = "HV plateau" if callback.hv_callback.converged else "generation limit"
        print(f"✓ Completed after {actual_gens} generations ({reason})")
        print(f"  Pareto solutions : {len(result.F)}")
        hv_hist = callback.hv_callback.hv_history
        if hv_hist:
            print(f"  Final HV         : {hv_hist[-1]:.6f}")

    X_full    = result.pop.get("X")
    F_full_norm = result.pop.get("F")
    F_full_raw  = np.asarray(
        [problem.evaluate_raw_objectives(xi) for xi in X_full], dtype=float
    )

    nd_set = {tuple(row) for row in result.X}
    is_nd  = np.array([tuple(row) in nd_set for row in X_full])

    # Filter initial_conditions for return (exclude large arrays)
    from .resto_anom import _filter_initial_conditions_for_return
    ic_filtered = _filter_initial_conditions_for_return(initial_conditions)

    optimization_results = {
        'scenario_params'       : scenario_params,
        'objective_names'       : problem.objective_names,
        'objectives'            : F_full_raw,
        'objectives_raw'        : F_full_raw,
        'objectives_normalized' : F_full_norm,
        'decisions'             : X_full,
        'is_nondominated'       : is_nd,
        'n_solutions'           : len(X_full),
        'n_nondominated_solutions': len(result.F),
        'problem_info': {
            'n_pixels'              : initial_conditions['n_pixels'],
            'n_restoration_pixels'  : initial_conditions.get('n_restoration_pixels', 0),
            'n_conversion_pixels'   : initial_conditions.get('n_conversion_pixels', 0),
            'max_action_pixels'     : problem.max_action_pixels,
            'is_patch_based'        : False,
            'is_planning_unit_based': True,
            'n_units'               : n_units,
            'unit_type'             : unit_type,
            'unit_size_px'          : unit_size_px if unit_type == 'grid' else None,
        },
        'algorithm_info': {
            'pop_size'              : pop_size,
            'n_generations'         : n_generations,
            'actual_generations'    : len(callback.hv_callback.hv_history),
            'converged_early'       : callback.hv_callback.converged,
            'termination_reason'    : (
                'hypervolume_convergence' if callback.hv_callback.converged
                else 'generation_limit'
            ),
            'convergence_reason'    : (
                'hypervolume_plateau' if callback.hv_callback.converged
                else 'generation_limit'
            ),
            'hypervolume_history'   : callback.hv_callback.hv_history,
            'final_hypervolume'     : (
                callback.hv_callback.hv_history[-1]
                if callback.hv_callback.hv_history else None
            ),
            'population_statistics' : {
                'f_mean_history': callback.hv_callback.f_mean_history,
                'f_std_history' : callback.hv_callback.f_std_history,
                'f_min_history' : callback.hv_callback.f_min_history,
                'f_max_history' : callback.hv_callback.f_max_history,
            },
            'hv_patience'           : hv_patience,
            'hv_min_improvement'    : hv_min_improvement,
            'sampling_method'       : 'planning_unit_score_guided',
            'repair_operators'      : ['planning_unit_score_based_repair'],
            'objective_normalization': {
                'enabled': bool(problem.normalize_objectives),
                'scales' : {k: float(v) for k, v in problem.objective_scales.items()},
            },
            'timestamp'             : datetime.now().isoformat(),
        },
        'initial_conditions'    : ic_filtered,
        'hv_callback'           : callback.hv_callback,
        'run_label'             : run_label,
        'run_config'            : run_config,
        'unit_mappings'         : initial_conditions['unit_mappings'],
    }

    if save_results:
        save_results_with_reports(
            optimization_results,
            output_dir=output_dir,
            verbose=verbose,
            run_label=run_label,
            r_export_parent=r_export_parent,
        )

    return optimization_results
