"""Debug and diagnostic utilities for restoration optimization."""

import numpy as np


def diagnose_optimization_setup(initial_conditions, scenario_params, n_samples=10):
    """
    Diagnose why optimization might not find solutions.

    Args:
        initial_conditions: Dict with initial conditions
        scenario_params: Dict with scenario parameters
        n_samples: Number of sample solutions to test
    """
    from resto_anom import RestorationProblem

    print("\n=== Quick problem diagnostics ===")

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

    print(f"\n   Summary: {feasible_count}/{n_samples} feasible, {infeasible_count}/{n_samples} infeasible")

    if feasible_count > 0:
        print(f"   ✓ Feasible solutions exist, Objectives show improvement with restoration")
    else:
        print(f"   ✗ WARNING: No feasible solutions found in {n_samples} samples!")
        print(f"   → Check if max_action_pixels constraint is too restrictive")

    issues = []
    if out_none['F'][0] == 0:
        issues.append("   ✗ Baseline objective is zero - may indicate data loading issue")
    if np.any(np.isnan(out_none['F'])):
        issues.append("   ✗ NaN detected in objectives - data contains unmasked NaN values")

    if not issues:
        print("   ✓ No obvious setup issues detected")
    else:
        for issue in issues:
            print(issue)

    print(f"\n OBJECTIVE SENSITIVITY TO RESTORATION AMOUNT:")
    print(f"   Baseline objectives (no restoration):")
    for j, obj_name in enumerate(problem.objective_names):
        print(f"      {obj_name}: {out_none['F'][j]:.6f}")

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
            print(f"   {frac*100:>5.1f}% restored ({n_restore:>6} pixels): Avg improvement = {avg_improvement*100:>6.2f}%")

            if frac == test_fractions[0]:
                print(f"      Per-objective details:")
                for j, obj_name in enumerate(problem.objective_names):
                    baseline_val = out_none['F'][j]
                    restored_val = out['F'][j]
                    change = restored_val - baseline_val
                    print(f"         {obj_name}: {baseline_val:.6f} → {restored_val:.6f} (Δ={change:.6f})")
