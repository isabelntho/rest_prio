"""Debug and diagnostic utilities for restoration optimization."""

import logging
import numpy as np

logger = logging.getLogger("resto_prio")


def diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10):
    """
    Diagnose why optimization might not find solutions.

    Args:
        initial_conditions: Dict with initial conditions
        scenario_params: Dict with scenario parameters
        n_samples: Number of sample solutions to test
    """
    from Core_optimisation.resto_anom import RestorationProblem

    logger.info("=== Quick problem diagnostics ===")

    problem = RestorationProblem(initial_conditions, scenario_params, n_jobs=1)

    x_none = np.zeros(problem.n_var, dtype=int)
    out_none = {}
    problem._evaluate(x_none, out_none)

    feasible_count = 0
    infeasible_count = 0

    for i in range(n_samples):
        x = np.zeros(problem.n_var, dtype=int)
        n_restore = problem.max_action_pixels
        if n_restore > 0 and n_restore <= problem.n_pixels:
            restore_indices = np.random.permutation(problem.n_pixels)[:n_restore]
            x[restore_indices] = 1

        out = {}
        problem._evaluate(x, out)

        if out['G'][0] <= 0:
            feasible_count += 1
        else:
            infeasible_count += 1

    logger.info(f"Summary: {feasible_count}/{n_samples} feasible, {infeasible_count}/{n_samples} infeasible")

    if feasible_count > 0:
        logger.info("Feasible solutions exist; objectives show improvement with restoration")
    else:
        logger.warning(f"No feasible solutions found in {n_samples} samples! Check if max_action_pixels constraint is too restrictive")

    issues = []
    if out_none['F'][0] == 0:
        issues.append("   ✗ Baseline objective is zero - may indicate data loading issue")
    if np.any(np.isnan(out_none['F'])):
        issues.append("   ✗ NaN detected in objectives - data contains unmasked NaN values")

    if not issues:
        logger.info("No obvious setup issues detected")
    else:
        for issue in issues:
            logger.warning(issue)

    logger.info("OBJECTIVE SENSITIVITY TO RESTORATION AMOUNT:")
    baseline_parts = ", ".join(f"{obj_name}={out_none['F'][j]:.6f}" for j, obj_name in enumerate(problem.objective_names))
    logger.info(f"  Baseline objectives (no restoration): {baseline_parts}")

    test_fractions = [0.05, 0.10, 0.15, 0.20]

    for frac in test_fractions:
        n_restore = int(frac * problem.n_pixels)
        if n_restore > 0 and n_restore <= problem.n_pixels:
            x = np.zeros(problem.n_var, dtype=int)
            restore_indices = np.random.permutation(problem.n_pixels)[:n_restore]
            x[restore_indices] = 1

            out = {}
            problem._evaluate(x, out)

            total_improvement = 0
            n_improved = 0
            for j in range(len(out['F'])):
                if out_none['F'][j] == 0:
                    if out['F'][j] < 0:
                        total_improvement += abs(out['F'][j])
                        n_improved += 1
                elif out_none['F'][j] != 0:
                    total_improvement += (out_none['F'][j] - out['F'][j]) / abs(out_none['F'][j])
                    n_improved += 1

            avg_improvement = (total_improvement / n_improved) if n_improved > 0 else 0.0
            logger.info(f"  {frac*100:>5.1f}% restored ({n_restore:>6} pixels): Avg improvement = {avg_improvement*100:>6.2f}%")

            if frac == test_fractions[0]:
                per_obj = ", ".join(
                    f"{obj_name}: {out_none['F'][j]:.6f} -> {out['F'][j]:.6f} (D={out['F'][j]-out_none['F'][j]:.6f})"
                    for j, obj_name in enumerate(problem.objective_names)
                )
                logger.info(f"    Per-objective: {per_obj}")
