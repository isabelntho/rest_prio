"""Per-generation mutation / repair volume for sets of seed-replicate runs.

Reads the result pickles (outputs/results_files/res_*.pkl, 1-3 GB each at CH scale, so
too big for R) and writes one small CSV for Documentation/results_overview_CH.qmd:

  outputs/r_inputs/operator_summary/operator_log.csv
    mode, seed, generation, mean_raw_flips, mean_bits_changed, px_per_unit

  mean_raw_flips    mean bits flipped by mutation per individual (mutation_diagnostics)
  mean_bits_changed mean bits changed by repair per individual (repair_diagnostics); empty
                    when the run logged none (pixel-mode region runs)
  px_per_unit       pixels per decision bit: patch_size^2 for patch mode, else 1

USAGE (project root):
  pixi run python Debugs_tests/operator_log_summary.py
  pixi run python Debugs_tests/operator_log_summary.py region=ch_region_condgrid_seed patch=ch_patch_tol001_seed

Each argument is mode=<run_label prefix>; the newest pickle per seed is used.
"""
import glob
import os
import pickle
import re
import sys

sys.path.insert(0, ".")
import pandas as pd

RESULTS = "outputs/results_files"
OUT = "outputs/r_inputs/operator_summary/operator_log.csv"
DEFAULTS = {"region": "ch_region_condgrid_seed", "patch": "ch_patch_baseline_seed"}


def newest_per_seed(prefix):
    hits = {}
    for f in glob.glob(os.path.join(RESULTS, f"res_*_{prefix}*.pkl")):
        m = re.search(rf"{re.escape(prefix)}(\d+)\.pkl$", f)
        if m and (m.group(1) not in hits or f > hits[m.group(1)]):
            hits[m.group(1)] = f
    return hits


def main():
    sets = dict(a.split("=", 1) for a in sys.argv[1:]) or DEFAULTS
    rows = []
    for mode, prefix in sets.items():
        for seed, f in sorted(newest_per_seed(prefix).items()):
            print(f"{mode} seed {seed}: {os.path.basename(f)}", flush=True)
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            px = r["run_config"].get("patch_size", 1) ** 2 if r["run_config"].get("use_patch_approach") else 1
            mut = pd.DataFrame(r.get("mutation_diagnostics") or [], columns=["generation", "mean_raw_flips"])
            rep = pd.DataFrame(r.get("repair_diagnostics") or [], columns=["generation", "mean_bits_changed"])
            d = mut.merge(rep, on="generation", how="outer").sort_values("generation")
            d.insert(0, "seed", int(seed))
            d.insert(0, "mode", mode)
            d["px_per_unit"] = px
            rows.append(d)
            del r
    if not rows:
        sys.exit("No matching pickles found under " + RESULTS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.concat(rows).to_csv(OUT, index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
