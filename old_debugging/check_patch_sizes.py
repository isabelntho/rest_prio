"""Short patch-size sweep runner.

Runs one scenario for multiple patch sizes and seeds, and writes all outputs to
the requested folder: patch_tests_1803.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from Core_optimisation.data_loader import load_initial_conditions
from Core_optimisation.resto_anom import run_single_scenario_optimization
from Core_optimisation.patch_approach import create_patch_mappings_for_restoration_and_conversion
from utils import pickle_load as _load_results_pickle
from visualisations import (
    create_selection_frequency_map,
    plot_pareto_front,
    plot_parallel_coordinates,
    create_patch_size_comparison_grids,
    _extract_decisions,
    _extract_objectives,
)


ECOSYSTEM = "forest"
REGION = "Bern"
OBJECTIVES = ["abiotic", "biotic", "cost"]
SAMPLE_FRACTION = None  # Set to None for full domain (slower, especially for patch_size=2)

# Sweep requested in previous discussion.
PATCH_SIZES = [2]#, 3, 5, 10]
SEEDS = [100]

POP_SIZE = 50
N_GENERATIONS = 100
N_JOBS = 12

PATCH_CONSTRAINT_TYPE = "pixel_count"
PIXEL_TOLERANCE = 0.05

OUTPUT_DIR = "patch_tests_1903_2"
VISUALIZE_ONLY = False

SCENARIO_PARAMS = {
    "max_restoration_fraction": 0.1,
    "spatial_clustering": 0,
    "burden_sharing": "no",
    "abiotic_effect": 0.01,
    "biotic_effect": 0.01,
    "hv_warmup_samples": 30,
}


def _extract_decisions(results):
    """Get decision matrix from results dict with fallback key names."""
    if results is None:
        return None
    if "decisions" in results:
        return np.asarray(results["decisions"])
    if "X" in results:
        return np.asarray(results["X"])
    return None


def _extract_objectives(results):
    """Get objective matrix from results dict with fallback key names."""
    if results is None:
        return None
    if "objectives" in results:
        return np.asarray(results["objectives"])
    if "F" in results:
        return np.asarray(results["F"])
    return None


def _build_selection_frequency_map(decisions, initial_conditions):
    """Build raster of restoration selection frequency (% across Pareto solutions)."""
    shape = initial_conditions["shape"]
    n_rest = int(initial_conditions["n_restoration_pixels"])
    n_conv = int(initial_conditions.get("n_conversion_pixels", 0))
    rest_indices = initial_conditions["restoration_eligible_indices"]

    freq_map = np.full(shape, np.nan, dtype=np.float64)
    if decisions is None or decisions.size == 0:
        return freq_map

    decisions = np.asarray(decisions)
    n_var = decisions.shape[1]

    # Case A: full pixel-level decisions were returned (restoration + conversion).
    if n_var == (n_rest + n_conv):
        rest_decisions = decisions[:, :n_rest]
        freq = np.mean(rest_decisions, axis=0) * 100.0
    # Case A2: restoration-only pixel decisions were returned.
    elif n_var == n_rest:
        freq = np.mean(decisions, axis=0) * 100.0
    else:
        # Case B: patch-level decisions were returned -> convert to pixel-level restoration decisions.
        patch_mappings = initial_conditions.get("patch_mappings")
        if patch_mappings is None:
            raise ValueError(
                "Decision vector length does not match pixel space and patch_mappings are missing. "
                f"Got n_var={n_var}, expected {n_rest + n_conv}."
            )

        n_rest_patches = int(patch_mappings["restoration_patches"]["n_patches"])
        n_conv_patches = int(patch_mappings["conversion_patches"]["n_patches"])
        expected_patch_var = n_rest_patches + n_conv_patches
        if n_var != expected_patch_var:
            raise ValueError(
                "Decision vector length matches neither pixel nor patch space. "
                f"Got n_var={n_var}, expected pixel={n_rest + n_conv}, patch={expected_patch_var}."
            )

        from Core_optimisation.patch_approach import convert_patch_decisions_to_pixels

        rest_pixel_solutions = np.zeros((decisions.shape[0], n_rest), dtype=np.int8)
        for i in range(decisions.shape[0]):
            x_restore_patches = decisions[i, :n_rest_patches]
            rest_pixel_solutions[i] = convert_patch_decisions_to_pixels(
                x_restore_patches,
                patch_mappings["restoration_patches"],
                n_rest,
            )

        freq = np.mean(rest_pixel_solutions, axis=0) * 100.0

    # Extra guard in case of unexpected metadata mismatch.
    if len(freq) != len(rest_indices):
        raise ValueError(
            "Selection frequency length does not match restoration index length: "
            f"freq={len(freq)}, indices={len(rest_indices)}"
        )

    flat = freq_map.ravel()
    flat[rest_indices] = freq
    return flat.reshape(shape)


def _plot_parallel_coordinates(ax, F):
    """Simple parallel coordinates with normalized objective values."""
    if F is None or len(F) == 0:
        ax.text(0.5, 0.5, "No objective data", ha="center", va="center")
        ax.set_axis_off()
        return

    # Convert minimization objectives for interpretability:
    #  -abiotic and -biotic are improvements; cost is shown as-is.
    vals = np.column_stack([-F[:, 0], -F[:, 1], F[:, 2]])
    labels = ["Abiotic improve", "Biotic improve", "Cost"]

    mins = np.nanmin(vals, axis=0)
    maxs = np.nanmax(vals, axis=0)
    spans = np.where((maxs - mins) <= 1e-12, 1.0, maxs - mins)
    vals_n = (vals - mins) / spans

    x = np.arange(vals_n.shape[1])
    cost_norm = vals_n[:, 2]
    cmap = plt.cm.viridis

    for i in range(vals_n.shape[0]):
        ax.plot(x, vals_n[i], color=cmap(cost_norm[i]), alpha=0.25, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates")
    ax.grid(alpha=0.3)


def _plot_parallel_coordinates_with_global_scale(ax, F, mins, maxs):
    """Parallel coordinates normalized with shared limits across patch sizes."""
    if F is None or len(F) == 0:
        ax.text(0.5, 0.5, "No objective data", ha="center", va="center")
        ax.set_axis_off()
        return

    vals = np.column_stack([-F[:, 0], -F[:, 1], F[:, 2]])
    labels = ["Abiotic improve", "Biotic improve", "Cost"]
    spans = np.where((maxs - mins) <= 1e-12, 1.0, maxs - mins)
    vals_n = (vals - mins) / spans

    x = np.arange(vals_n.shape[1])
    cost_norm = vals_n[:, 2]
    cmap = plt.cm.viridis

    for i in range(vals_n.shape[0]):
        ax.plot(x, vals_n[i], color=cmap(cost_norm[i]), alpha=0.25, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value")
    ax.grid(alpha=0.3)


def create_selection_frequency_grid(run_data, output_dir):
    """Create one 2x2 selection-frequency grid across patch sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    im = None

    for ax, item in zip(axes, run_data):
        freq_map = _build_selection_frequency_map(item["decisions"], item["initial_conditions"])
        im = ax.imshow(freq_map, cmap="viridis", vmin=0, vmax=100)
        ax.set_title(f"Patch size {item['patch_size']}x{item['patch_size']}")
        ax.set_axis_off()

    for ax in axes[len(run_data):]:
        ax.set_axis_off()

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("% selected")
    fig.suptitle("Selection frequency across patch sizes", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(output_dir, "selection_frequency_grid.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_pareto_front_grid(run_data, output_dir):
    """Create one 2x2 Pareto-front grid across patch sizes."""
    all_F = [item["F"] for item in run_data if item["F"] is not None and len(item["F"]) > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    if all_F:
        F_all = np.vstack(all_F)
        x_all = -F_all[:, 0]
        y_all = -F_all[:, 1]
        cmin, cmax = np.nanmin(F_all[:, 2]), np.nanmax(F_all[:, 2])
        xlim = (np.nanmin(x_all), np.nanmax(x_all))
        ylim = (np.nanmin(y_all), np.nanmax(y_all))
    else:
        cmin, cmax = 0.0, 1.0
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)

    sc = None
    for ax, item in zip(axes, run_data):
        F = item["F"]
        if F is not None and len(F) > 0:
            sc = ax.scatter(-F[:, 0], -F[:, 1], c=F[:, 2], cmap="plasma", s=25,
                            alpha=0.85, vmin=cmin, vmax=cmax)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel("Abiotic improvement")
            ax.set_ylabel("Biotic improvement")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Pareto data", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(f"Patch size {item['patch_size']}x{item['patch_size']}")

    for ax in axes[len(run_data):]:
        ax.set_axis_off()

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Cost")

    fig.suptitle("Pareto front across patch sizes", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(output_dir, "pareto_front_grid.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_parallel_coordinates_grid(run_data, output_dir):
    """Create one 2x2 parallel-coordinates grid across patch sizes."""
    all_vals = []
    for item in run_data:
        F = item["F"]
        if F is not None and len(F) > 0:
            all_vals.append(np.column_stack([-F[:, 0], -F[:, 1], F[:, 2]]))

    if all_vals:
        vals_all = np.vstack(all_vals)
        mins = np.nanmin(vals_all, axis=0)
        maxs = np.nanmax(vals_all, axis=0)
    else:
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([1.0, 1.0, 1.0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, item in zip(axes, run_data):
        _plot_parallel_coordinates_with_global_scale(ax, item["F"], mins, maxs)
        ax.set_title(f"Patch size {item['patch_size']}x{item['patch_size']}")

    for ax in axes[len(run_data):]:
        ax.set_axis_off()

    fig.suptitle("Parallel coordinates across patch sizes", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(output_dir, "parallel_coordinates_grid.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def create_plot_type_grids(run_data, output_dir):
    """Create the three requested 2x2 plot-type grids (one panel per patch size)."""
    p1 = create_selection_frequency_grid(run_data, output_dir)
    p2 = create_pareto_front_grid(run_data, output_dir)
    p3 = create_parallel_coordinates_grid(run_data, output_dir)
    return [p1, p2, p3]


def _compose_image_grid(image_paths_by_patch, patch_sizes, out_png, title):
    """Compose pre-rendered images into a 2x2 grid ordered by patch size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, patch_size in enumerate(patch_sizes):
        ax = axes[i]
        path = image_paths_by_patch.get(patch_size)
        if path and os.path.exists(path):
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(f"Patch size {patch_size}x{patch_size}")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "Image missing", ha="center", va="center")
            ax.set_title(f"Patch size {patch_size}x{patch_size}")
            ax.axis("off")

    for ax in axes[len(patch_sizes):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _call_visualisation_no_show(func, *args, **kwargs):
    """Call visualisation function while suppressing interactive popups."""
    old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        return func(*args, **kwargs)
    finally:
        plt.show = old_show


def create_grids_using_visualisations(output_dir):
    """Generate per-patch plots with visualisations.py, then compose 2x2 grids."""
    per_patch_selection = {}
    per_patch_pareto = {}
    per_patch_parallel = {}

    for patch_size in PATCH_SIZES:
        run_output_dir = os.path.join(output_dir, f"{ECOSYSTEM}_{patch_size}_{SEEDS[0]}")
        pattern = os.path.join(run_output_dir, "results_*.pkl")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            print(f"No results pickle found for patch size {patch_size} in {run_output_dir}")
            continue

        pkl_path = candidates[-1]

        sel_png = os.path.join(run_output_dir, f"selection_frequency_patch{patch_size}.png")
        pareto_png = os.path.join(run_output_dir, f"pareto_patch{patch_size}.png")
        parallel_png = os.path.join(run_output_dir, f"parallel_patch{patch_size}.png")

        try:
            _call_visualisation_no_show(
                create_selection_frequency_map,
                pkl_path,
                save_path=sel_png,
                show_eligible=True,
                action_type="combined",
            )
            _call_visualisation_no_show(
                plot_pareto_front,
                pkl_path,
                save_path=pareto_png,
            )
            _call_visualisation_no_show(
                plot_parallel_coordinates,
                pkl_path,
                save_path=parallel_png,
            )

            per_patch_selection[patch_size] = sel_png
            per_patch_pareto[patch_size] = pareto_png
            per_patch_parallel[patch_size] = parallel_png
            print(f"Generated visualisations.py plots for patch size {patch_size}")
        except Exception as exc:
            print(f"Failed to generate visualisations for patch size {patch_size}: {exc}")

    out1 = _compose_image_grid(
        per_patch_selection,
        PATCH_SIZES,
        os.path.join(output_dir, "selection_frequency_grid.png"),
        "Selection frequency across patch sizes",
    )
    out2 = _compose_image_grid(
        per_patch_pareto,
        PATCH_SIZES,
        os.path.join(output_dir, "pareto_front_grid.png"),
        "Pareto front across patch sizes",
    )
    out3 = _compose_image_grid(
        per_patch_parallel,
        PATCH_SIZES,
        os.path.join(output_dir, "parallel_coordinates_grid.png"),
        "Parallel coordinates across patch sizes",
    )

    return [out1, out2, out3]


def _load_results_pickle(pkl_path):
    """Load pickle with compatibility shim for numpy module path changes."""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise

    # Retry once with shim for older pickle module references.
    import sys
    sys.modules["numpy._core"] = np.core
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_saved_run_data(output_dir):
    """Load existing run outputs and build run_data for visualization-only mode."""
    run_data = []

    for patch_size in PATCH_SIZES:
        initial_conditions = load_initial_conditions(
            ".",
            objectives=OBJECTIVES,
            region=REGION,
            ecosystem=ECOSYSTEM,
            sample_fraction=SAMPLE_FRACTION,
            sample_seed=42,
        )

        # Ensure patch mappings exist in visualization-only mode for patch-space decisions.
        if "patch_mappings" not in initial_conditions:
            initial_conditions["patch_mappings"] = create_patch_mappings_for_restoration_and_conversion(
                initial_conditions,
                patch_size=patch_size,
            )

        run_output_dir = os.path.join(output_dir, f"{ECOSYSTEM}_{patch_size}_{SEEDS[0]}")
        pattern = os.path.join(run_output_dir, "results_*.pkl")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            print(f"No results pickle found for patch size {patch_size} in {run_output_dir}")
            continue

        pkl_path = candidates[-1]
        try:
            results = _load_results_pickle(pkl_path)
            run_data.append({
                "patch_size": patch_size,
                "seed": SEEDS[0],
                "initial_conditions": initial_conditions,
                "decisions": _extract_decisions(results),
                "F": _extract_objectives(results),
            })
            print(f"Loaded saved results for patch size {patch_size}: {os.path.basename(pkl_path)}")
        except Exception as exc:
            print(f"Failed to load {pkl_path}: {exc}")

    run_data = sorted(run_data, key=lambda x: x["patch_size"])
    return run_data


def run_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if VISUALIZE_ONLY:
        grid_paths = create_grids_using_visualisations(OUTPUT_DIR)
        print("\nSaved grid figures:")
        for p in grid_paths:
            print(f"  {p}")
        return

    total = len(PATCH_SIZES) * len(SEEDS)
    run_id = 0
    summary = []
    run_data = []

    for patch_size in PATCH_SIZES:
        print(f"\nLoading inputs for patch size {patch_size}...")
        initial_conditions = load_initial_conditions(
            ".",
            objectives=OBJECTIVES,
            region=REGION,
            ecosystem=ECOSYSTEM,
            sample_fraction=SAMPLE_FRACTION,
            sample_seed=42,
        )

        for seed in SEEDS:
            run_id += 1
            print("\n" + "=" * 80)
            print(f"RUN {run_id}/{total} | patch_size={patch_size} | seed={seed}")
            print("=" * 80)

            run_output_dir = os.path.join(OUTPUT_DIR, f"{ECOSYSTEM}_{patch_size}_{seed}")
            os.makedirs(run_output_dir, exist_ok=True)

            try:
                results = run_single_scenario_optimization(
                    initial_conditions=initial_conditions,
                    scenario_params=SCENARIO_PARAMS,
                    pop_size=POP_SIZE,
                    n_generations=N_GENERATIONS,
                    save_results=True,
                    verbose=True,
                    n_jobs=N_JOBS,
                    use_repair=True,
                    random_seed=seed,
                    use_patch_approach=True,
                    patch_size=patch_size,
                    patch_constraint_type=PATCH_CONSTRAINT_TYPE,
                    pixel_tolerance=PIXEL_TOLERANCE,
                    output_dir=run_output_dir,
                )

                if results is not None:
                    run_data.append({
                        "patch_size": patch_size,
                        "seed": seed,
                        "initial_conditions": initial_conditions,
                        "decisions": _extract_decisions(results),
                        "F": _extract_objectives(results),
                    })

                hv = results.get("hypervolume") if results else None
                n_sol = len(results["results_df"]) if results and "results_df" in results else 0
                status = "SUCCESS" if results is not None else "FAILED"
            except Exception as exc:
                hv = None
                n_sol = 0
                status = f"ERROR: {exc}"

            summary.append({
                "patch_size": patch_size,
                "seed": seed,
                "status": status,
                "hypervolume": hv,
                "n_solutions": n_sol,
            })
            print(f"Status: {status}")

    if run_data:
        grid_paths = create_plot_type_grids(run_data, OUTPUT_DIR)
        print("\nSaved grid figures:")
        for p in grid_paths:
            print(f"  {p}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for row in summary:
        hv_str = f"{row['hypervolume']:.4f}" if row["hypervolume"] is not None else "N/A"
        print(
            f"patch={row['patch_size']:<2} seed={row['seed']:<3} "
            f"status={row['status']:<16} hv={hv_str:<8} n={row['n_solutions']}"
        )
    print(f"\nOutputs saved under: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_sweep()
