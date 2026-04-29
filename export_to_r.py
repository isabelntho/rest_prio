"""
export_to_r.py
==============
Export key optimization outputs to R-readable formats.

Outputs (written to <output_dir>/):
  objectives.csv            — all solutions: objective values + is_nondominated flag
  objectives_normalized.csv — same, normalized objective space
  hypervolume_evolution.csv — hypervolume per generation
  population_stats.csv      — per-generation mean/std/min/max for each objective
  pixel_selection.csv       — long-format: solution_id, action_type, pixel_row/x, pixel_col/y (non-dominated solutions only)
  metadata.json             — run config, scenario params, problem info, algorithm scalars

Usage:
    python export_to_r.py results_files/res_fg_20260415_1840_cost_corrected.pkl
    python export_to_r.py results_files/res_20260427_1100_3obj_rf10_bs.pkl --output-dir r_inputs/3obj_bs
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_evolution_json(pkl_path: Path) -> Path | None:
    """Try to locate the matching evolution JSON for a given pkl file."""
    stem = pkl_path.stem  # e.g. res_fg_20260415_1840_cost_corrected

    # New-style filenames: res_fg_YYYYMMDD_HHMM_label.pkl  →  evo__fg_YYYYMMDD_HHMM_label.json
    evo_dir = pkl_path.parent.parent / "evolution_reports"
    candidate = evo_dir / f"evo__{stem[len('res_'):]}.json"
    if candidate.exists():
        return candidate

    # Older-style: results_fg_1903_1.pkl  →  evolution_report_fg_1903_1.json
    candidate2 = evo_dir / f"evolution_report_{stem[len('results_'):]}.json"
    if candidate2.exists():
        return candidate2

    return None


def _expand_patches_to_pixels(decisions_patches, patch_mappings, n_restoration_patches):
    """
    Expand patch-level binary decisions to a list of eligible-pixel indices per solution.

    Returns a list of length n_solutions; each element is a sorted 1-D int array of
    eligible-pixel indices (0-based into restoration_eligible_indices) that are
    selected by that solution.
    """
    restoration_pm = patch_mappings.get("restoration_patches", {})
    patch_to_pixels = restoration_pm.get("patch_to_pixels")  # dict: patch_idx → list of px indices
    if patch_to_pixels is None:
        return None

    n_solutions = decisions_patches.shape[0]
    result = []
    for sol_idx in range(n_solutions):
        selected = np.where(decisions_patches[sol_idx, :n_restoration_patches] == 1)[0]
        px_indices = []
        for p in selected:
            px_indices.extend(patch_to_pixels.get(p, []))
        result.append(np.array(px_indices, dtype=np.int64))
    return result


def _expand_conversion_patches_to_pixels(decisions_patches, patch_mappings, n_restoration_patches):
    """
    Expand conversion patch columns of the decision matrix to pixel indices per solution.

    Returns a list of length n_solutions; each element is a 1-D int array of
    eligible-pixel indices (0-based into conversion_eligible_indices).
    Returns None if no conversion patch mapping is available.
    """
    conversion_pm = patch_mappings.get("conversion_patches", {})
    patch_to_pixels = conversion_pm.get("patch_to_pixels")
    if patch_to_pixels is None:
        return None
    n_conversion_patches = conversion_pm.get("n_patches", 0)
    if n_conversion_patches == 0:
        return None

    n_solutions = decisions_patches.shape[0]
    result = []
    for sol_idx in range(n_solutions):
        conv_slice = decisions_patches[sol_idx, n_restoration_patches:n_restoration_patches + n_conversion_patches]
        selected = np.where(conv_slice == 1)[0]
        px_indices = []
        for p in selected:
            px_indices.extend(patch_to_pixels.get(p, []))
        result.append(np.array(px_indices, dtype=np.int64))
    return result


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_results(pkl_path: str, output_dir: str = None, nondom_pixels_only: bool = True):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")

    if output_dir is None:
        output_dir = pkl_path.parent.parent / "r_inputs" / pkl_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pkl_path.name} ...")
    with open(pkl_path, "rb") as f:
        r = pickle.load(f)

    obj_names = r.get("objective_names", [])
    objectives_raw = np.asarray(r.get("objectives", r.get("objectives_raw", [])), dtype=float)
    objectives_norm = np.asarray(r.get("objectives_normalized", []), dtype=float)
    decisions = np.asarray(r.get("decisions", []))
    is_nondominated = np.asarray(r.get("is_nondominated", []), dtype=bool)
    scenario_params = r.get("scenario_params", {})
    algorithm_info = r.get("algorithm_info", {})
    problem_info = r.get("problem_info", {})
    run_config = r.get("run_config", {})
    ic = r.get("initial_conditions", {})
    patch_mappings = r.get("patch_mappings") or ic.get("patch_mappings")

    n_solutions = len(objectives_raw)

    # -------------------------------------------------------------------
    # 1. objectives.csv
    # -------------------------------------------------------------------
    obj_df = pd.DataFrame(objectives_raw, columns=obj_names)
    obj_df.insert(0, "solution_id", np.arange(n_solutions))
    obj_df["is_nondominated"] = is_nondominated.astype(int)
    obj_path = output_dir / "objectives.csv"
    obj_df.to_csv(obj_path, index=False)
    print(f"  → {obj_path.name}  ({n_solutions} rows × {len(obj_names)} objectives)")

    # -------------------------------------------------------------------
    # 2. objectives_normalized.csv
    # -------------------------------------------------------------------
    if objectives_norm.shape == objectives_raw.shape:
        norm_df = pd.DataFrame(objectives_norm, columns=[f"{n}_norm" for n in obj_names])
        norm_df.insert(0, "solution_id", np.arange(n_solutions))
        norm_df["is_nondominated"] = is_nondominated.astype(int)
        norm_path = output_dir / "objectives_normalized.csv"
        norm_df.to_csv(norm_path, index=False)
        print(f"  → {norm_path.name}")

    # -------------------------------------------------------------------
    # 3. Hypervolume evolution  (from algorithm_info or evolution JSON)
    # -------------------------------------------------------------------
    hv_history = algorithm_info.get("hypervolume_history", [])

    evo_json_path = _find_evolution_json(pkl_path)
    evo_data = {}
    if evo_json_path:
        print(f"  Found evolution JSON: {evo_json_path.name}")
        with open(evo_json_path) as f:
            evo_data = json.load(f)
        if not hv_history:
            hv_history = evo_data.get("hypervolume_evolution", {}).get("history", [])

    if hv_history:
        hv_df = pd.DataFrame({"generation": np.arange(len(hv_history)), "hypervolume": hv_history})
        hv_path = output_dir / "hypervolume_evolution.csv"
        hv_df.to_csv(hv_path, index=False)
        print(f"  → {hv_path.name}  ({len(hv_history)} generations)")

    # -------------------------------------------------------------------
    # 4. Per-generation population statistics
    # -------------------------------------------------------------------
    pop_stats = algorithm_info.get("population_statistics", {})
    if not pop_stats and evo_data:
        pop_evo = evo_data.get("population_evolution", {})
        pop_stats = {
            "f_mean_history": pop_evo.get("f_mean_history", []),
            "f_std_history":  pop_evo.get("f_std_history", []),
            "f_min_history":  pop_evo.get("f_min_history", []),
            "f_max_history":  pop_evo.get("f_max_history", []),
        }

    f_mean = pop_stats.get("f_mean_history", [])
    if f_mean:
        n_gen = len(f_mean)
        rows = []
        stat_keys = ["mean", "std", "min", "max"]
        hist_arrays = [
            pop_stats.get("f_mean_history", []),
            pop_stats.get("f_std_history", []),
            pop_stats.get("f_min_history", []),
            pop_stats.get("f_max_history", []),
        ]
        for gen in range(n_gen):
            row = {"generation": gen}
            for stat, arr in zip(stat_keys, hist_arrays):
                if gen < len(arr):
                    vals = arr[gen]
                    if not hasattr(vals, "__len__"):
                        vals = [vals]
                    for i, obj in enumerate(obj_names):
                        if i < len(vals):
                            row[f"{obj}_{stat}"] = vals[i]
            rows.append(row)
        pop_df = pd.DataFrame(rows)
        pop_path = output_dir / "population_stats.csv"
        pop_df.to_csv(pop_path, index=False)
        print(f"  → {pop_path.name}  ({n_gen} generations × {len(obj_names)} objectives)")

    # -------------------------------------------------------------------
    # 5. Pixel selection (long format, non-dominated solutions by default)
    #    Includes action_type column: "restore" or "convert"
    # -------------------------------------------------------------------
    restoration_indices = ic.get("restoration_eligible_indices")
    conversion_indices  = ic.get("conversion_eligible_indices")
    shape = ic.get("shape")

    if restoration_indices is not None and shape is not None and decisions.size > 0:
        is_patch_based = problem_info.get("is_patch_based", False)
        n_restoration_pixels = problem_info.get("n_restoration_pixels", len(restoration_indices))
        n_conversion_pixels  = problem_info.get("n_conversion_pixels",
                                                len(conversion_indices) if conversion_indices is not None else 0)

        # Expand patch decisions → pixel decisions if needed
        if is_patch_based and patch_mappings is not None:
            print("  Expanding patch decisions to pixel level ...")
            n_restoration_patches = problem_info.get("n_restoration_patches", decisions.shape[1])
            per_sol_px_indices   = _expand_patches_to_pixels(decisions, patch_mappings, n_restoration_patches)
            per_sol_conv_indices = _expand_conversion_patches_to_pixels(decisions, patch_mappings, n_restoration_patches)
        elif not is_patch_based:
            # pixel-level: first n_restoration_pixels = restore, next n_conversion_pixels = convert
            per_sol_px_indices = [
                np.where(decisions[i, :n_restoration_pixels] == 1)[0]
                for i in range(n_solutions)
            ]
            if n_conversion_pixels > 0 and conversion_indices is not None:
                per_sol_conv_indices = [
                    np.where(decisions[i, n_restoration_pixels:n_restoration_pixels + n_conversion_pixels] == 1)[0]
                    for i in range(n_solutions)
                ]
            else:
                per_sol_conv_indices = None
        else:
            per_sol_px_indices   = None
            per_sol_conv_indices = None

        if per_sol_px_indices is not None:
            mask = is_nondominated if nondom_pixels_only else np.ones(n_solutions, dtype=bool)
            sol_indices = np.where(mask)[0]

            # Build affine transform for pixel → coordinate conversion.
            # ic["transform"] may be a rasterio Affine object or a flat list/tuple
            # of the 6 affine coefficients (a, b, c, d, e, f).
            transform_raw = ic.get("transform")
            affine_transform = None
            if transform_raw is not None:
                try:
                    import rasterio.transform as _rt
                    from affine import Affine
                    if isinstance(transform_raw, Affine):
                        affine_transform = transform_raw
                    else:
                        coeffs = list(transform_raw)
                        # rasterio stores as (a, b, c, d, e, f); Affine takes same order
                        affine_transform = Affine(*coeffs[:6])
                except Exception as _e:
                    print(f"  Warning: could not build affine transform ({_e}); "
                          "falling back to pixel row/col.")

            # ── Export all eligible pixel coordinates ────────────────────────
            elig_rows_px, elig_cols_px = np.divmod(restoration_indices, shape[1])
            if affine_transform is not None:
                import rasterio.transform as _rt
                elig_xs, elig_ys = _rt.xy(affine_transform, elig_rows_px, elig_cols_px)
                elig_df = pd.DataFrame({"x": elig_xs, "y": elig_ys})
            else:
                elig_df = pd.DataFrame({"pixel_row": elig_rows_px, "pixel_col": elig_cols_px})
            elig_path = output_dir / "eligible_pixels.csv"
            elig_df.to_csv(elig_path, index=False)
            print(f"  → {elig_path.name}  ({len(elig_df):,} eligible pixels)")

            def _pixels_to_rows(sol_i, flat_pixel_indices, action_type_label):
                """Convert a flat array of global-grid indices to a tidy DataFrame block."""
                rows_px, cols_px = np.divmod(flat_pixel_indices, shape[1])
                if affine_transform is not None:
                    import rasterio.transform as _rt
                    xs, ys = _rt.xy(affine_transform, rows_px, cols_px)
                    return pd.DataFrame({
                        "solution_id": sol_i,
                        "action_type": action_type_label,
                        "x": xs,
                        "y": ys,
                    })
                else:
                    return pd.DataFrame({
                        "solution_id": sol_i,
                        "action_type": action_type_label,
                        "pixel_row": rows_px,
                        "pixel_col": cols_px,
                    })

            rows_list = []
            for sol_i in sol_indices:
                # Restoration pixels
                px_idx = per_sol_px_indices[sol_i]
                if len(px_idx) > 0:
                    rows_list.append(_pixels_to_rows(sol_i, restoration_indices[px_idx], "restore"))

                # Conversion pixels (present when a conversion objective is in the run)
                if per_sol_conv_indices is not None:
                    conv_idx = per_sol_conv_indices[sol_i]
                    if len(conv_idx) > 0:
                        rows_list.append(_pixels_to_rows(sol_i, conversion_indices[conv_idx], "convert"))

            if rows_list:
                px_df = pd.concat(rows_list, ignore_index=True)
                px_path = output_dir / "pixel_selection.csv"
                px_df.to_csv(px_path, index=False)
                coord_mode = "EPSG:2056 coordinates (x, y)" if affine_transform is not None else "pixel row/col"
                label = "non-dominated" if nondom_pixels_only else "all"
                n_convert_rows = int((px_df["action_type"] == "convert").sum())
                n_restore_rows = len(px_df) - n_convert_rows
                print(f"  → {px_path.name}  ({len(sol_indices)} {label} solutions, "
                      f"{len(px_df):,} pixel-solution rows "
                      f"[{n_restore_rows:,} restore, {n_convert_rows:,} convert], {coord_mode})")

    # -------------------------------------------------------------------
    # 6. metadata.json
    # -------------------------------------------------------------------
    def _json_safe(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    def _clean(d):
        return {k: _json_safe(v) for k, v in d.items() if not isinstance(v, (np.ndarray, dict))} | \
               {k: _clean(v) for k, v in d.items() if isinstance(v, dict)}

    hv_history_list = [float(x) for x in hv_history]

    metadata = {
        "pkl_file": pkl_path.name,
        "run_label": r.get("run_label", ""),
        "ecosystem": ic.get("ecosystem", run_config.get("ecosystem", "unknown")),
        "objective_names": obj_names,
        "n_solutions": n_solutions,
        "n_nondominated_solutions": int(r.get("n_nondominated_solutions", int(is_nondominated.sum()))),
        "scenario_params": {k: _json_safe(v) for k, v in scenario_params.items()},
        "run_config": {k: _json_safe(v) for k, v in run_config.items() if not isinstance(v, dict)},
        "problem_info": {k: _json_safe(v) for k, v in problem_info.items()},
        "algorithm": {
            "pop_size": algorithm_info.get("pop_size"),
            "n_generations": algorithm_info.get("n_generations"),
            "actual_generations": algorithm_info.get("actual_generations"),
            "converged_early": algorithm_info.get("converged_early"),
            "termination_reason": algorithm_info.get("termination_reason"),
            "final_hypervolume": algorithm_info.get("final_hypervolume"),
            "hv_patience": algorithm_info.get("hv_patience"),
            "random_seed": run_config.get("random_seed"),
            "objective_normalization": algorithm_info.get("objective_normalization", {}),
        },
        "raster_info": {
            "shape": list(shape) if shape else None,
            "crs": str(ic.get("crs", "")),
            "transform": list(ic.get("transform", [])) if ic.get("transform") is not None else None,
        },
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  → {meta_path.name}")

    print(f"\nAll outputs written to: {output_dir}/")
    return str(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export optimization results to R-readable formats.")
    parser.add_argument("pkl", help="Path to results pickle file")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: r_inputs/<pkl_stem>/)")
    parser.add_argument("--all-solutions", action="store_true",
                        help="Include pixel selection for all solutions, not just non-dominated (slow/large)")
    args = parser.parse_args()

    export_results(args.pkl, output_dir=args.output_dir, nondom_pixels_only=not args.all_solutions)


# File	Contents
# objectives.csv	One row per solution — raw objective values + is_nondominated flag
# objectives_normalized.csv	Same in normalized objective space
# hypervolume_evolution.csv	generation, hypervolume — auto-sourced from evolution JSON if present
# population_stats.csv	Per-generation mean/std/min/max for each objective
# pixel_selection.csv	Long-format: solution_id, action_type ("restore"/"convert"), pixel_row/x, pixel_col/y (non-dominated solutions by default)
# metadata.json	Run config, scenario params, algorithm settings, raster grid info (CRS, transform, shape)



## saving eligible pixels raster.. code snippet

#import numpy as np
#import rasterio as rio
#from data_loader import load_initial_conditions

#ic = load_initial_conditions(
#    ".",
#    objectives=["abiotic", "biotic", "cost"],
#    region="Bern",
#    ecosystem="fg",
#)

#eligible = ic["eligible_mask"].astype(np.uint8)  # 1=eligible, 0=not eligible

#with rio.open("inputs/abiotic_condition_anomaly.tif") as src:
#    profile = src.profile.copy()

#profile.update(dtype=rio.uint8, count=1, nodata=None)

#out_path = "inputs/eligible_pixels_fg.tif"
#with rio.open(out_path, "w", **profile) as dst:
#    dst.write(eligible, 1)

#n = int(eligible.sum())
#print(f"Saved {out_path}  ({n} eligible pixels, {100*n/eligible.size:.1f}% of raster)")
