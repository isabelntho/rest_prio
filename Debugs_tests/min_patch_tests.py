"""MinPatchSizeRepair / minimum-patch-size constraint tests.

Subcommands (pixi run python Debugs_tests/min_patch_tests.py <cmd>):

  unit    Unit checks for MinPatchSizeRepair on the real Bern layers: after repair every
          4-connected component has >= S pixels, the pixel count stays within the budget
          tolerance band, and S=1 imposes no size floor. Runs scattered / random /
          contiguous / empty input plans.

  smoke   Short end-to-end smoke: 2-obj NSGA-II + region_grow + min_patch_size constraint.
          Confirms the n_constr=2 wiring flows through NSGA-II and audits that every final
          solution satisfies the minimum-patch-size constraint.
"""
import sys
from types import SimpleNamespace
import numpy as np

from _common import load_ic
from Core_optimisation.resto_anom import run_optimization_instance
from Core_optimisation.spatial_operations import (
    MinPatchSizeRepair, _label_components, build_restoration_neighbor_table,
    grow_region_plan,
)

TOL = 0.05


# ===========================================================================
# unit  (was test_min_patch_repair.py)
# ===========================================================================
def _comp_sizes(sel, shape, rows, cols):
    return [c.size for c in _label_components(sel, shape, rows, cols)]


def cmd_unit():
    ic = load_ic(["restoration_potential", "cost"])
    n_rest = int(ic["n_restoration_pixels"])
    k = max(int(0.05 * n_rest), 1)
    shape = tuple(ic["shape"])
    nbr, rows, cols = build_restoration_neighbor_table(ic)
    prob = SimpleNamespace(n_restoration_pixels=n_rest, n_var=n_rest, max_action_pixels=k)
    rng = np.random.default_rng(0)
    lo, hi = int(k * (1 - TOL)), int(k * (1 + TOL))

    # Build a few input plans (restoration block only; conversion block absent here).
    def scattered():
        x = np.zeros(n_rest, dtype=int)
        x[rng.choice(n_rest, size=k, replace=False)] = 1  # k singletons
        return x

    def contiguous():
        return grow_region_plan(nbr, n_rest, k, 25, np.zeros(n_rest), 'neutral', rng).astype(int)

    def empty():
        return np.zeros(n_rest, dtype=int)

    inputs = {"scattered": scattered, "contiguous": contiguous, "empty": empty}

    print(f"n_rest={n_rest}  budget k={k}  tol band=[{lo},{hi}]\n")
    all_ok = True
    for S in [1, 25, 100, 400]:
        rep = MinPatchSizeRepair(ic, k, S, scores=None, pixel_tolerance=TOL,
                                 growth_bias='neutral')
        for name, gen in inputs.items():
            X = np.stack([gen() for _ in range(4)])
            Xr = rep._do(prob, X)
            ok = True
            worst_min = None
            for i in range(len(Xr)):
                sel = Xr[i, :n_rest].astype(bool)
                cnt = int(sel.sum())
                sizes = _comp_sizes(sel, shape, rows, cols)
                mn = min(sizes) if sizes else 0
                worst_min = mn if worst_min is None else min(worst_min, mn)
                budget_ok = lo <= cnt <= hi
                size_ok = (S <= 1) or (mn >= S) or (cnt == 0)
                ok = ok and budget_ok and size_ok
            flag = "OK" if ok else "FAIL"
            all_ok = all_ok and ok
            print(f"  S={S:<4} {name:<11} -> {flag}  (worst min-component={worst_min}, "
                  f"budget in band={lo<=int(Xr[0,:n_rest].sum())<=hi})")
    print("\nRESULT:", "ALL PASS" if all_ok else "SOME FAILED")


# ===========================================================================
# smoke  (was smoke_min_patch.py)
# ===========================================================================
def cmd_smoke():
    S = 100
    params = {
        "max_restoration_fraction": 0.05,
        "spatial_clustering": 0,
        "biotic_effect": 0.01,
        "abiotic_effect": 0.01,
        "normalize_objectives": True,
        "burden_sharing": "no",
        "rp_formulation": "sum",
        "rp_threshold": 0.0,
        "sampling_strategy": "region_grow",
        "min_patch_size": S,
        "region_seeds": 25,
        "region_seeds_min": 5,
        "region_growth_bias": "scored",
        "region_mutation_edits": 100,
    }

    ic = load_ic(["restoration_potential", "cost"])
    res = run_optimization_instance(
        initial_conditions=ic, scenario_params=params,
        pop_size=12, n_generations=2, save_results=False, verbose=True,
        n_jobs=1, random_seed=101, use_repair=True, use_patch_approach=False,
        pixel_tolerance=0.05, algorithm_type="nsga2", run_label="smoke_min_patch",
    )

    print("\n=== constraint audit on final solutions ===")
    dec = np.asarray(res["decisions"])
    n_rest = int(ic["n_restoration_pixels"])
    shape = tuple(ic["shape"])
    _, rows, cols = build_restoration_neighbor_table(ic)
    worst = None
    for i in range(dec.shape[0]):
        sel = dec[i, :n_rest].astype(bool)
        sizes = [c.size for c in _label_components(sel, shape, rows, cols)]
        mn = min(sizes) if sizes else 0
        worst = mn if worst is None else min(worst, mn)
    print(f"solutions={dec.shape[0]}  S={S}  smallest component over all solutions={worst}")
    print("AUDIT:", "PASS (all components >= S)" if (worst is None or worst >= S) else "FAIL")


# ===========================================================================
COMMANDS = {
    "unit": cmd_unit,
    "smoke": cmd_smoke,
}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    if cmd not in COMMANDS:
        print(__doc__)
        print(f"Subcommands: {', '.join(COMMANDS)}")
        return
    COMMANDS[cmd]()


if __name__ == "__main__":
    main(sys.argv)
