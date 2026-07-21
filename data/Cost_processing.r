# =====================================================================
# Restoration implementation cost layers - Canton of Bern
#
# Builds spatially explicit implementation-cost proxies for Forest,
# Grassland and Cropland following the methodology in
# "Assessment of restoration Implementation Costs in the Canton of Bern".
#
# Four normalized (0 = low effort, 1 = high effort) cost drivers are
# combined per land use:
#   slope_norm  - terrain slope (from DEM, in percent)
#   soil_norm   - soil suitability class (land-use specific)
#   dist_norm   - distance to nearest road (accessibility)
#   own_norm    - ownership structure (administrative complexity)
#
# Aggregation (section 6):
#   unweighted:  (slope + soil + dist + own) / 4
#   weighted:    0.35*slope + 0.35*soil + 0.2*dist + 0.1*own
#
# The analysis extent is selectable (Kanton Bern or all of Switzerland;
# see `extent` below). Outputs (masked to each land use):
#   forest_cost,   grassland_cost,   cropland_cost      (unweighted)
#   forest_cost_w, grassland_cost_w, cropland_cost_w    (weighted)
# plus a single combined layer per weighting scheme.
# =====================================================================

library(terra)

## ---- Analysis extent ------------------------------------------------
# "KB" = Kanton Bern (full model, including the ownership driver).
# "CH" = whole of Switzerland. Ownership (GREIKA) is only available for
#        Bern, so at CH extent the ownership driver is dropped and the
#        remaining weights are renormalised (methodology section 4.5).
extent        <- "KB"                    # "KB" or "CH"
use_ownership <- (extent == "KB")

## ---- Directories & inputs -------------------------------------------
cost_data_dir <- "Y:/CH_Kanton_Bern/03_Workspaces/03_Habitat_condition/Restoration_potential/Implementation_cost_eva/Data/"
result_dir    <- "Y:/CH_Kanton_Bern/03_Workspaces/03_Habitat_condition/Restoration_potential/Implementation_cost_eva/Result/"
out_dir       <- paste0(result_dir, if (extent == "CH") "cost_layers_CH/" else "cost_layers/")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

DEM       <- paste0(cost_data_dir, "DEM/dem25_lv95.tif")
rd_dist   <- paste0(cost_data_dir, "distance to road/distance_road.tif")
soil      <- paste0(cost_data_dir, "Soil suitability/bek200_070625_shp/bek200_070625.shp")
ownership <- paste0(cost_data_dir, "Ownership/data/GREIKA_GREIKA.shp")
landcover <- paste0(cost_data_dir, "arealstatistik/AS18_3.tif")   # 1=crop, 2=grass, 3=forest, 0=other
kb_bound  <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"
ch_bound <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_LANDESGEBIET.shp"

## ---- Analysis grid --------------------------------------------------
target_crs <- "EPSG:2056"   # CH1903+ / LV95
target_res <- 100           # metres (native resolution of AS18_3)

## ---- Tuning parameters (methodology section 4 & 6) ------------------
slope_cap  <- 70            # slope (%) at/above which slope_norm = 1
slope_gain <- 0.8           # scaling below the cap: (slope/cap)*gain
dist_cap   <- 2000          # distance (m) at/above which dist_norm = 1

# Weighted-index weights (must sum to 1); section 6.2.
w_slope <- 0.35
w_soil  <- 0.35
w_dist  <- 0.20
w_own   <- 0.10

# Soil suitability class -> normalized cost (best soil = lowest cost,
# "inconvenient" = highest cost). The methodology does not give the 0-1
# values, so these were recovered by regressing the reference Result/
# layers on the drivers + soil class: the reference assigns "inconvenient"
# the maximum cost (~1.0) and spaces the remaining classes roughly evenly.
# Category codes were read from the shapefile attribute table:
#   Forest  (SoilCat) : 1 very good, 2 good, 3 medium, 4 poor, 0 inconvenient
#   Grass   (Grass_cat): 3 very good, 2 good, 1 medium,        0 inconvenient
#   Crop    (Crop_cat) : 3 very good, 2 good, 1 medium,        0 inconvenient
soil_forest_map <- c("1" = 0.00, "2" = 0.30, "3" = 0.57, "4" = 0.80, "0" = 1.00)
soil_grass_map  <- c("3" = 0.00, "2" = 0.36, "1" = 0.65,             "0" = 1.00)
soil_crop_map   <- c("3" = 0.00, "2" = 0.37, "1" = 0.68,             "0" = 1.00)

# =====================================================================
# 1. Study area (Canton of Bern) and reference grid
# =====================================================================
# Study-area boundary: Bern for "KB", the whole-country outline for "CH".
if (extent == "KB") {
  aoi <- vect(kb_bound)
  aoi <- aoi[aoi$NAME == "Bern", ]
} else {
  aoi <- vect(ch_bound)
}
aoi <- aggregate(project(aoi, target_crs))   # dissolve to a single polygon

# Land cover defines the masking classes and the analysis grid.
lc <- rast(landcover)

crs(lc) <- target_crs        # AS18_3 coords are already LV95; force the label, don't trust the file
lc <- crop(lc, aoi)          # single analysis grid (100 m, snapped)
lc_t <- mask(lc, aoi)

# Land-use masks: 1 where the land use applies, NA everywhere else.
# (Use ifel -> 1/NA so mask() below unambiguously drops all other cells;
# a logical mask with maskvalues=c(FALSE,NA) does NOT drop the FALSE cells.)
mask_forest <- ifel(lc == 3, 1, NA)
mask_grass  <- ifel(lc == 2, 1, NA)
mask_crop   <- ifel(lc == 1, 1, NA)

# Helper: clip any layer to the study-area boundary.
clip_aoi <- function(r) mask(r, aoi)

# =====================================================================
# 2. Slope -> slope_norm (section 4.1)
# =====================================================================
dem <- rast(DEM)
dem <- crop(dem, aoi)                             # DEM is national; trim to study area
slope_pct  <- tan(terrain(dem, v = "slope", unit = "radians")) * 100
slope_pct  <- project(slope_pct, lc_t, method = "bilinear")
slope_norm <- ifel(slope_pct > slope_cap, 1, (slope_pct / slope_cap) * slope_gain)
slope_norm <- clip_aoi(slope_norm)

# =====================================================================
# 3. Accessibility -> dist_norm (section 4.2)
# =====================================================================
dist <- rast(rd_dist)
dist <- crop(dist, aoi)                           # distance raster is national; trim to study area
dist <- project(dist, lc_t, method = "bilinear")
dist_norm <- ifel(dist > dist_cap, 1, dist / dist_cap)
dist_norm <- clip_aoi(dist_norm)

# =====================================================================
# 4. Ownership -> own_norm (section 4.3)
# =====================================================================
# KAT_norm holds the pre-computed 0-1 administrative-effort score.
# Ownership data covers only Bern, so it is skipped at CH extent.
if (use_ownership) {
  own_v    <- project(vect(ownership), target_crs)
  own_norm <- rasterize(own_v, lc_t, field = "KAT_norm")
  own_norm <- clip_aoi(own_norm)
}

# =====================================================================
# 5. Soil suitability -> soil_norm per land use (section 4.4)
# =====================================================================
soil_v <- vect(soil)
# The bek200 .prj is a broken ESRI Hotine-Oblique-Mercator definition that
# terra cannot match to an EPSG code, but the coordinates are plain
# CH1903 / LV03. Force the source CRS before reprojecting to LV95.
if (is.na(crs(soil_v, describe = TRUE)$code)) crs(soil_v) <- "EPSG:21781"
soil_v <- project(soil_v, target_crs)

# Translate the integer suitability class into a normalized cost field.
remap <- function(codes, lut) unname(lut[as.character(codes)])
soil_v$for_norm   <- remap(soil_v$SoilCat,   soil_forest_map)
soil_v$grass_norm <- remap(soil_v$Grass_cat, soil_grass_map)
soil_v$crop_norm  <- remap(soil_v$Crop_cat,  soil_crop_map)

soil_for_norm   <- clip_aoi(rasterize(soil_v, lc_t, field = "for_norm"))
soil_grass_norm <- clip_aoi(rasterize(soil_v, lc_t, field = "grass_norm"))
soil_crop_norm  <- clip_aoi(rasterize(soil_v, lc_t, field = "crop_norm"))

# =====================================================================
# 6. Aggregate into cost indices (section 6) and mask per land use
# =====================================================================
# When ownership is unavailable (CH extent) it is dropped from both
# indices and the remaining weights renormalised so the result stays 0-1.
cost_unw <- function(soil_norm) {
  drivers <- list(slope_norm, soil_norm, dist_norm)
  if (use_ownership) drivers <- c(drivers, list(own_norm))
  Reduce(`+`, drivers) / length(drivers)
}
cost_w <- function(soil_norm) {
  num <- w_slope * slope_norm + w_soil * soil_norm + w_dist * dist_norm
  den <- w_slope + w_soil + w_dist
  if (use_ownership) {
    num <- num + w_own * own_norm
    den <- den + w_own
  }
  num / den
}

forest_cost    <- mask(cost_unw(soil_for_norm),   mask_forest)
grassland_cost <- mask(cost_unw(soil_grass_norm), mask_grass)
cropland_cost  <- mask(cost_unw(soil_crop_norm),  mask_crop)

forest_cost_w    <- mask(cost_w(soil_for_norm),   mask_forest)
grassland_cost_w <- mask(cost_w(soil_grass_norm), mask_grass)
cropland_cost_w  <- mask(cost_w(soil_crop_norm),  mask_crop)

# Single combined layer (land uses are disjoint, so merge the three).
cost_combined   <- merge(forest_cost,   grassland_cost,   cropland_cost)
cost_combined_w <- merge(forest_cost_w, grassland_cost_w, cropland_cost_w)
names(cost_combined)   <- "cost"
names(cost_combined_w) <- "cost_w"

# =====================================================================
# 7. Write results
# =====================================================================
outputs <- list(
  slope_norm       = slope_norm,
  dist_norm        = dist_norm,
  soil_for_norm    = soil_for_norm,
  soil_grass_norm  = soil_grass_norm,
  soil_crop_norm   = soil_crop_norm,
  forest_cost      = forest_cost,
  grassland_cost   = grassland_cost,
  cropland_cost    = cropland_cost,
  forest_cost_w    = forest_cost_w,
  grassland_cost_w = grassland_cost_w,
  cropland_cost_w  = cropland_cost_w,
  cost_combined    = cost_combined,
  cost_combined_w  = cost_combined_w
)
if (use_ownership) outputs$own_norm <- own_norm

for (nm in names(outputs)) {
  writeRaster(outputs[[nm]], paste0(out_dir, nm, ".tif"), overwrite = TRUE)
}

message("Cost layers written to: ", out_dir)

# =====================================================================
# 8. Validation against the reference layers in Result/ (step 1)
# =====================================================================
# The three weighted cost layers were produced earlier (ArcGIS) and live
# directly in Result/. For each, report:
#   - grid ALIGNMENT   : CRS, resolution, and origin offset vs my grid
#   - COVERAGE         : valid cells in each and how much they overlap
#   - value AGREEMENT  : mean diff / RMSE / correlation on overlap cells
# The reference rasters are full rectangles that use 0 as background (no
# NA), so a reference "data" cell is taken as value > 0.
# Only meaningful for the KB extent - the reference layers cover Bern only.
if (extent == "KB") {

ref_result_dir <- "Y:/CH_Kanton_Bern/03_Workspaces/03_Habitat_condition/Restoration_potential/Implementation_cost_eva/Result/"

check_against_reference <- function(mine, refpath, landuse_mask) {
  if (!file.exists(refpath)) return(NULL)
  ref <- rast(refpath)

  # -- alignment (report before any resampling) --
  crs_ok <- crs(mine, describe = TRUE)$code == crs(ref, describe = TRUE)$code
  res_ok <- isTRUE(all.equal(res(mine), res(ref)))
  off    <- round(origin(mine) - origin(ref), 2)   # sub-cell grid shift, metres

  # -- coverage + agreement: put reference on my grid (sub-cell shift, so
  #    nearest-neighbour keeps co-located cells) --
  ref_al   <- resample(ref, mine, method = "near")
  vr <- as.vector(values(ref_al))
  vm <- as.vector(values(mine))
  ref_data <- is.finite(vr) & vr > 0.02      # reference computed a value
  my_data  <- is.finite(vm)                  # I computed a value
  both     <- ref_data & my_data
  d        <- vm[both] - vr[both]

  data.frame(
    crs_ok          = crs_ok,
    res_ok          = res_ok,
    origin_off_m    = paste(off, collapse = "/"),
    n_mine          = sum(my_data),
    n_ref           = sum(ref_data),
    n_overlap       = sum(both),
    pct_of_mine     = round(100 * sum(both) / max(sum(my_data), 1), 1),
    pct_of_ref      = round(100 * sum(both) / max(sum(ref_data), 1), 1),
    mean_diff       = round(mean(d), 4),
    rmse            = round(sqrt(mean(d^2)), 4),
    cor             = round(stats::cor(vr[both], vm[both]), 3),
    row.names = NULL
  )
}

val_targets <- list(
  forest_cost_w    = forest_cost_w,
  grassland_cost_w = grassland_cost_w,
  cropland_cost_w  = cropland_cost_w
)

validation <- do.call(rbind, lapply(names(val_targets), function(nm) {
  res <- check_against_reference(val_targets[[nm]], paste0(ref_result_dir, nm, ".tif"))
  if (is.null(res)) NULL else cbind(layer = nm, res)
}))

cat("\n=== Validation vs Result/ reference layers ===\n")
cat("alignment: crs_ok/res_ok TRUE and origin_off_m 0/0 means identical grids\n")
cat("coverage : n_overlap vs n_mine/n_ref; pct_of_* = overlap share\n")
cat("agreement: on overlapping cells (reference value > 0)\n\n")
print(validation, row.names = FALSE)
write.csv(validation, paste0(out_dir, "validation_vs_reference.csv"), row.names = FALSE)

} else {
  message("Extent = CH: skipping comparison against the KB reference layers.")
}
