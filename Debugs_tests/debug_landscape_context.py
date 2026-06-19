"""
Compute and save landscape_context and restoration_potential rasters.

Outputs (written to inputs/):
  landscape_context.tif     — focal-mean neighbour anomaly (500 m radius)
  restoration_potential.tif — per-pixel mean abiotic + biotic anomaly

Also produces:
  - Side-by-side maps + histograms of both layers
  - Radius sensitivity grid: Pearson r between restoration_potential and
    landscape_context computed at a range of radii
"""

import os
import sys
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------------------
# Paths – adjust if needed
# ---------------------------------------------------------------------------
WORKSPACE = os.path.join(os.path.dirname(__file__), "..")

ABIOTIC_PATH  = os.path.join(WORKSPACE, "inputs", "anomaly_scenarios", "abiotic_global_all.tif")
BIOTIC_PATH   = os.path.join(WORKSPACE, "inputs", "anomaly_scenarios", "biotic_global_all.tif")
# Ecosystem LULC: try the pre-masked local copy first, then fall back to network
ECOSYSTEM_LULC_CANDIDATES = [
    os.path.join(WORKSPACE, "ecosystem_lulc_masked.tif"),
    r"W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif",
    r"W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018.tif",
]

# Ecosystem LULC codes that are restoration-eligible (forest + grassland = 'fg')
ECOSYSTEM_CODES = [12, 13, 15, 16, 17]   # forest=[12,13], grassland=[16,17]

# Focal-mean neighbourhood parameters (matching data_loader.py)
RADIUS_PX   = 5    # 500 m at 100 m resolution
KERNEL_SIZE = 2 * RADIUS_PX + 1   # 11 × 11 box approximation

# ---------------------------------------------------------------------------
# 1. Load anomaly rasters
# ---------------------------------------------------------------------------
def load_raster(path):
    with rio.open(path) as src:
        data = src.read(1)
        meta = {
            "crs":       src.crs,
            "transform": src.transform,
            "shape":     data.shape,
            "bounds":    src.bounds,
            "nodata":    src.nodata,
        }
    return data, meta


print("Loading abiotic anomaly …")
abiotic_raw, ref_meta = load_raster(ABIOTIC_PATH)

print("Loading biotic anomaly …")
biotic_raw, biotic_meta = load_raster(BIOTIC_PATH)

# Basic consistency check
assert abiotic_raw.shape == biotic_raw.shape, (
    f"Shape mismatch: abiotic {abiotic_raw.shape} vs biotic {biotic_raw.shape}"
)

shape = abiotic_raw.shape
print(f"Raster shape: {shape}")

# Replace NaN with 0 (same as data_loader.py)
abiotic_nan = np.isnan(abiotic_raw)
biotic_nan  = np.isnan(biotic_raw)
abiotic = np.nan_to_num(abiotic_raw, nan=0.0)
biotic  = np.nan_to_num(biotic_raw,  nan=0.0)

# ---------------------------------------------------------------------------
# 2. Build restoration-eligible mask (NaN union + ecosystem LULC)
# ---------------------------------------------------------------------------
# Base: exclude pixels that are NaN in either objective layer
base_mask = ~(abiotic_nan | biotic_nan)

# Find and load ecosystem LULC
lulc_path = None
for candidate in ECOSYSTEM_LULC_CANDIDATES:
    if os.path.exists(candidate):
        lulc_path = candidate
        break

if lulc_path is None:
    print(
        "WARNING: No ecosystem LULC file found – using base mask only (no ecosystem filtering).\n"
        f"Tried: {ECOSYSTEM_LULC_CANDIDATES}"
    )
    restoration_eligible = base_mask
else:
    print(f"Loading ecosystem LULC from: {lulc_path}")
    lulc_raw, lulc_meta = load_raster(lulc_path)

    # Resample LULC to match anomaly rasters if shapes differ
    if lulc_raw.shape != shape:
        from rasterio.warp import reproject, Resampling
        print(f"  Resampling LULC from {lulc_raw.shape} → {shape}")
        lulc_resampled = np.empty(shape, dtype=lulc_raw.dtype)
        with rio.open(lulc_path) as src:
            reproject(
                source=rio.band(src, 1),
                destination=lulc_resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.nearest,
            )
        lulc = lulc_resampled
    else:
        lulc = lulc_raw

    ecosystem_mask = np.isin(lulc, ECOSYSTEM_CODES)
    restoration_eligible = base_mask & ecosystem_mask
    print(
        f"  Ecosystem mask: {np.sum(ecosystem_mask):,} px  |  "
        f"Restoration-eligible: {np.sum(restoration_eligible):,} px  "
        f"({100 * np.sum(restoration_eligible) / restoration_eligible.size:.1f}%)"
    )

restoration_eligible_indices = np.where(restoration_eligible.flatten())[0]

# ---------------------------------------------------------------------------
# 3. Compute landscape_context (matching data_loader.py logic exactly)
# ---------------------------------------------------------------------------
def focal_mean_neighbours(anom_2d, elig_2d, kernel_size, cnt_neigh):
    """
    Focal mean of anomaly values over restoration-eligible neighbours within
    the box kernel, *excluding the centre pixel itself*.
    """
    anom_elig  = np.where(elig_2d, anom_2d, 0.0).astype(np.float64)
    sum_focal  = uniform_filter(anom_elig,  size=kernel_size, mode="constant") * (kernel_size ** 2)
    sum_focal  = sum_focal - anom_2d          # subtract self
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt_neigh > 0, sum_focal / cnt_neigh, 0.0)

#def compute_landscape_context(abiotic, biotic, elig_2d, radius_px):
#    """Return (2d array, 1d eligible slice) for a given radius in pixels."""
#    ksize = 2 * radius_px + 1
#    count_elig = elig_2d.astype(np.float64)
#    cnt_focal  = uniform_filter(count_elig, size=ksize, mode="constant") * (ksize ** 2)
#    cnt_neigh  = np.maximum(cnt_focal - 1.0, 0.0)
#    ctx_2d = np.mean(np.stack([
#        focal_mean_neighbours(abiotic, elig_2d, ksize, cnt_neigh),
#        focal_mean_neighbours(biotic,  elig_2d, ksize, cnt_neigh),
#    ], axis=0), axis=0)
#    flat = ctx_2d.flatten()
#    ctx_1d = flat[restoration_eligible_indices].astype(np.float64)
#    return ctx_2d, ctx_1d

def compute_landscape_context(abiotic, biotic, elig_2d, radius_px):
    """
    Return landscape context as:
    proportion of eligible neighbouring pixels in good condition.

    Good condition is defined as:
        abiotic > 0 and biotic > 0

    Output:
        ctx_2d: full raster, values 0 to 1
        ctx_1d: eligible-pixel slice
    """
    ksize = 2 * radius_px + 1

    # Eligible neighbours only
    elig_float = elig_2d.astype(np.float64)

    # Count eligible pixels in the focal window
    cnt_focal = uniform_filter(
        elig_float,
        size=ksize,
        mode="constant"
    ) * (ksize ** 2)

    # Exclude the centre pixel from the denominator
    cnt_neigh = np.maximum(cnt_focal - elig_float, 0.0)

    # Define good condition
    good_condition = (
        (abiotic > 0) &
        (biotic > 0) &
        elig_2d
    )

    good_float = good_condition.astype(np.float64)

    # Count good-condition eligible pixels in the focal window
    good_focal = uniform_filter(
        good_float,
        size=ksize,
        mode="constant"
    ) * (ksize ** 2)

    # Exclude the centre pixel from the numerator if it is good
    good_neigh = np.maximum(good_focal - good_float, 0.0)

    # Proportion of eligible neighbours that are already in good condition
    with np.errstate(invalid="ignore", divide="ignore"):
        ctx_2d = np.where(
            cnt_neigh > 0,
            good_neigh / cnt_neigh,
            0.0
        )

    flat = ctx_2d.flatten()
    ctx_1d = flat[restoration_eligible_indices].astype(np.float64)

    return ctx_2d, ctx_1d

print(f"Computing landscape_context (radius={RADIUS_PX} px) …")
landscape_context_2d, landscape_context_1d = compute_landscape_context(
    abiotic, biotic, restoration_eligible, RADIUS_PX
)

print(
    f"  2D shape : {landscape_context_2d.shape}\n"
    f"  1D length: {len(landscape_context_1d):,}\n"
    f"  min={landscape_context_1d.min():.4f}, "
    f"  max={landscape_context_1d.max():.4f}, "
    f"  mean={landscape_context_1d.mean():.4f}"
)

# ---------------------------------------------------------------------------
# 3b. Compute restoration_potential (per-pixel mean of abiotic + biotic anomaly)
#     Use the original NaN-aware arrays so non-eligible pixels stay NaN.
# ---------------------------------------------------------------------------
restoration_potential_2d = np.where(
    restoration_eligible,
    (abiotic_raw.astype(np.float64) + biotic_raw.astype(np.float64)) / 2.0,
    np.nan,
)
flat_rp = restoration_potential_2d.flatten()
restoration_potential_1d = flat_rp[restoration_eligible_indices].astype(np.float64)

print(
    f"\nrestoration_potential computed:"
    f"  min={restoration_potential_1d.min():.4f}, "
    f"  max={restoration_potential_1d.max():.4f}, "
    f"  mean={restoration_potential_1d.mean():.4f}"
)

# Mask landscape_context to NaN outside eligible pixels so data_loader's
# NaN-exclusion logic correctly limits the eligible extent on load.
landscape_context_2d = np.where(restoration_eligible, landscape_context_2d, np.nan)

# ---------------------------------------------------------------------------
# 4. Save GeoTIFFs to inputs/
# ---------------------------------------------------------------------------
NODATA = float("nan")

def save_tif(path, array_2d, meta):
    with rio.open(
        path, "w",
        driver="GTiff",
        height=array_2d.shape[0],
        width=array_2d.shape[1],
        count=1,
        dtype="float32",
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=NODATA,
    ) as dst:
        dst.write(array_2d.astype("float32"), 1)


LC_OUT  = os.path.join(WORKSPACE, "inputs", "landscape_context.tif")
RP_OUT  = os.path.join(WORKSPACE, "inputs", "restoration_potential.tif")

save_tif(LC_OUT, landscape_context_2d, ref_meta)
print(f"Saved: {LC_OUT}")

save_tif(RP_OUT, restoration_potential_2d, ref_meta)
print(f"Saved: {RP_OUT}")

# ---------------------------------------------------------------------------
# 5. Maps + histograms for both layers
# ---------------------------------------------------------------------------
def _display(arr_2d, mask):
    return np.where(mask, arr_2d, np.nan)


lc_disp = _display(landscape_context_2d, restoration_eligible)
rp_disp = _display(restoration_potential_2d, restoration_eligible)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for col, (disp, vals_1d, label, cmap) in enumerate([
    (lc_disp, landscape_context_1d, "landscape_context", "RdYlGn_r"),
    (rp_disp, restoration_potential_1d, "restoration_potential", "RdYlGn_r"),
]):
    vmin, vmax = np.nanpercentile(disp, [2, 98])
    im = axes[0, col].imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    plt.colorbar(im, ax=axes[0, col], shrink=0.8)
    title_suffix = (f"radius={RADIUS_PX}px ({RADIUS_PX*100}m)  kernel={KERNEL_SIZE}×{KERNEL_SIZE}"
                    if label == "landscape_context" else "per-pixel mean anomaly")
    axes[0, col].set_title(f"{label}\n{title_suffix}", fontsize=9)
    axes[0, col].axis("on")

    axes[1, col].hist(vals_1d, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    axes[1, col].axvline(vals_1d.mean(),    color="red",    linewidth=1.2,
                         label=f"mean = {vals_1d.mean():.3f}")
    axes[1, col].axvline(np.median(vals_1d), color="orange", linewidth=1.2, linestyle="--",
                         label=f"median = {np.median(vals_1d):.3f}")
    axes[1, col].set_xlabel(label)
    axes[1, col].set_ylabel("Count (eligible pixels)")
    axes[1, col].set_title(f"Distribution  n={len(vals_1d):,}")
    axes[1, col].legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(WORKSPACE, "Debugs_tests", "debug_landscape_context.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {fig_path}")
plt.show()

# ---------------------------------------------------------------------------
# 6. Radius sensitivity: Pearson r between restoration_potential and
#    landscape_context computed at different radii
# ---------------------------------------------------------------------------
RADII_TO_TEST = [1, 2, 3, 5, 7, 10, 15, 20, 30]   # pixels (×100 m = metres)

print("\nRadius sensitivity (Pearson r vs restoration_potential):")
print(f"  {'radius_px':>9}  {'radius_m':>9}  {'kernel':>8}  {'pearson_r':>10}")

radii_px  = []
radii_m   = []
pearson_rs = []

for r_px in RADII_TO_TEST:
    _, ctx_1d_r = compute_landscape_context(abiotic, biotic, restoration_eligible, r_px)
    r_val = np.corrcoef(restoration_potential_1d, ctx_1d_r)[0, 1]
    k = 2 * r_px + 1
    print(f"  {r_px:>9}  {r_px*100:>8}m  {k:>4}×{k:<4}  {r_val:>10.4f}")
    radii_px.append(r_px)
    radii_m.append(r_px * 100)
    pearson_rs.append(r_val)

fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot(radii_m, pearson_rs, marker="o", linewidth=1.8, color="steelblue")
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
# mark the currently saved radius
ax.axvline(RADIUS_PX * 100, color="red", linewidth=1.2, linestyle=":",
           label=f"current radius ({RADIUS_PX*100} m)")
ax.set_xlabel("Focal radius (m)")
ax.set_ylabel("Pearson r  (restoration_potential vs landscape_context)")
ax.set_title("Correlation sensitivity to landscape_context radius")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2_path = os.path.join(WORKSPACE, "Debugs_tests", "debug_lscontext_radius_sensitivity.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"\nSensitivity figure saved: {fig2_path}")
plt.show()
