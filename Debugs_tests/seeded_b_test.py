"""Seed B's simple-ranking plans into B's own GA starting population.

Does the GA at least match the ranking baseline when given it as a starting point, rather than
discovering it (or something better) from scratch?

    pixi run python Debugs_tests/seeded_b_test.py --smoke   # 1 seed, 6 generations, sanity check
    pixi run python Debugs_tests/seeded_b_test.py           # 3 seeds, 150 generations (real run)

B's 15 ranking plans (outputs/spatial_sweep/runs/B_base-2x2_seed101/simple_ranking/plans.npz,
already computed, deterministic, and confirmed identical across B's 4 seeds - see the sweep
qmd's discussion) are converted from pixel-index selections to patch-level decision vectors and
injected via RestorationProblem's extra_seed_X hook (resto_anom.py::run_optimization_instance),
exploiting the confirmed property that every ranking-front pixel selection is a union of WHOLE
2x2 tiles - re-verified here rather than assumed, so a stale plans.npz fails loudly.

Same config as setting B in Core_optimisation/spatial_sweep.py (2x2 tiles, spillover radius 3
decay 0.2 to un-restored cells, 5% budget, pixel tolerance 0.01, NSGA-II pop 100, no HV early
stopping) except warm_seeding=False (the unrelated per-objective auto-seed stays off, so only
the ranking plans are being tested) and extra_seed_X set.

Output: outputs/b_seeded_test/B_seeded_seed<n>/ (results_files/, summary_files/, r_inputs/,
ga_front.csv, ga_summary.json - same per-run layout spatial_sweep.py uses).
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _common import REPO_ROOT  # noqa: F401 - import puts REPO_ROOT on sys.path as a side effect

from Core_optimisation.data_loader import load_initial_conditions
from Core_optimisation.paths import OUTPUTS
from Core_optimisation.resto_anom import RestorationProblem, initialize_patch_approach, run_optimization_instance
from Core_optimisation.spatial_sweep import SETTINGS

B = next(s for s in SETTINGS if s["id"] == "B")
SEEDS = (101, 102, 103)
N_GENERATIONS = 150
POP_SIZE = 100
OUT_ROOT = os.path.join(str(OUTPUTS), "b_seeded_test")
RANKING_PLANS_SRC = os.path.join(str(OUTPUTS), "spatial_sweep", "runs", "B_base-2x2_seed101",
                                 "simple_ranking", "plans.npz")


def scenario_params(smoke):
    sp = {"max_restoration_fraction": 0.05, "spatial_clustering": 0, "biotic_effect": 0.01,
          "abiotic_effect": 0.01, "normalize_objectives": True, "burden_sharing": "no",
          "rp_formulation": "sum", "rp_threshold": 0.0, "sampling_strategy": "scattered",
          "repair_scored": False, "min_patch_size": 1,
          "neighbor_radius": B["radius"], "neighbor_effect_decay": B["decay"],
          "spillover_to_restored": B["to_restored"]}
    if smoke:
        sp["hv_warmup_samples"] = 20
    return sp


def build_seed_patch_vectors(ic):
    """B's 15 ranking plans (pixel-index space) -> patch-level 0/1 decision vectors."""
    pm = ic["patch_mappings"]["restoration_patches"]["patch_to_pixels"]
    n_rest = int(ic["n_restoration_pixels"])
    patch_of_pixel = np.full(n_rest, -1, np.int32)
    for pid, pix in pm.items():
        patch_of_pixel[np.atleast_1d(np.asarray(pix, np.int64))] = pid
    assert (patch_of_pixel >= 0).all(), "an eligible pixel belongs to no patch"

    plans = dict(np.load(RANKING_PLANS_SRC))
    n_var = int(ic["n_restoration_patches"]) + int(ic["n_conversion_patches"])
    seeds = []
    for key in sorted(plans):
        sel = plans[key]
        selmask = np.zeros(n_rest, bool)
        selmask[sel] = True
        touched = np.unique(patch_of_pixel[sel])
        for pid in touched:
            members = np.atleast_1d(np.asarray(pm[pid], np.int64))
            # Re-verify tile alignment here rather than trust the earlier one-off check: a stale
            # or hand-edited plans.npz must fail loudly, not silently seed a half-tile plan.
            assert selmask[members].all(), f"{key}: patch {pid} is only partially selected"
        row = np.zeros(n_var, dtype=int)
        row[touched] = 1
        seeds.append(row)
    return np.array(seeds, dtype=int)


def run_one(seed, smoke):
    label = f"B_seeded_seed{seed}"
    out_dir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w", buffering=1, encoding="ascii", errors="replace")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = logf
    t0 = time.perf_counter()
    try:
        ic = load_initial_conditions(".", objectives=["restoration_benefit", "cost"], region="CH",
                                     ecosystem="all", sample_fraction=None, sample_seed=42,
                                     aggregation_factor=None, condition_scenario="global_all")
        ic = initialize_patch_approach(ic, patch_size=2)
        seed_X = build_seed_patch_vectors(ic)
        print(f"[seed] {seed_X.shape[0]} ranking plans -> patch vectors "
              f"(patches selected per plan: {seed_X.sum(axis=1).tolist()})", flush=True)

        n_gen = 6 if smoke else N_GENERATIONS
        pop = 20 if smoke else POP_SIZE
        res = run_optimization_instance(
            initial_conditions=ic, scenario_params=scenario_params(smoke),
            pop_size=pop, n_generations=n_gen, save_results=True, verbose=True,
            random_seed=seed, use_repair=True, use_patch_approach=True, patch_size=2,
            patch_constraint_type="pixel_count", pixel_tolerance=0.01, output_dir=out_dir,
            warm_seeding=False, extra_seed_X=seed_X, run_label=label,
            algorithm_type="nsga2", hv_patience=n_gen + 1,
        )
        minutes = (time.perf_counter() - t0) / 60
        if res is None:
            return (label, False, minutes, "run returned no front")

        F = np.asarray(res["objectives_raw"], float)
        nd = np.asarray(res["is_nondominated"], bool)
        names = res["objective_names"]
        bi, ci = names.index("restoration_benefit"), names.index("implementation_cost")
        with open(os.path.join(out_dir, "ga_front.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["benefit_raw", "cost", "nondominated"])
            for row, flag in zip(F, nd):
                w.writerow([repr(float(row[bi])), repr(float(row[ci])), int(flag)])
        ai = res["algorithm_info"]
        with open(os.path.join(out_dir, "ga_summary.json"), "w") as f:
            json.dump({"label": label, "n_solutions": int(res["n_solutions"]),
                       "n_nondominated": int(res["n_nondominated_solutions"]),
                       "actual_generations": ai["actual_generations"],
                       "converged_early": bool(ai["converged_early"]), "minutes": round(minutes, 2)},
                      f, indent=1)
        return (label, True, minutes, None)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return (label, False, (time.perf_counter() - t0) / 60, f"{type(e).__name__}: {e}")
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    print(f"Launching {len(seeds)} seed(s), {'smoke' if args.smoke else N_GENERATIONS} "
          f"generations, workers={args.workers}, output {OUT_ROOT}", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_one, s, args.smoke): s for s in seeds}
        for fut in as_completed(futs):
            label, ok, minutes, err = fut.result()
            status = "ok" if ok else "FAILED"
            print(f"{status:6s} {label} ({minutes:.1f} min){'' if ok else ': ' + str(err)}", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
