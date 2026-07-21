# =====================================================================
# Ecological Condition (EC) indicator processing
#
# Goal: build a single multi-layer raster stack of EC indicators for
# Switzerland. Every layer is projected/resampled onto one common
# reference grid ("template") and masked to the national outline, so
# the layers align and can be combined into one SpatRaster.
#
# Indicators are organised by Ecosystem Condition Typology (ECT) class:
#   A1  Physical state     : smd, sbd
#   A2  Chemical state     : soc
#   B1  Compositional      : uzl, tsd
#   B2  Structural         : can, swf_h, swf_t
#   B3  Functional         : lai, ndvi
#   C1  Landscape          : snh, frag
#
# Some layers need only crop/mask/resample; others need extra
# processing (SWF rasterization, LAI cleaning, SNH focal density,
# canopy-height tile merge). Those are handled per class below.
# =====================================================================

library(terra)

## ---- Directories ----------------------------------------------------
data_dir      <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/"
kb_dir        <- "Y:/CH_Kanton_Bern/03_Workspaces/03_Habitat_condition/"
sdm_dir       <- "Y:/CH_Kanton_Bern/03_Workspaces/05_Web_platform/raster_data/"
arealstat_dir <- paste0(data_dir, "LULC/")   # NOAS04 land cover; adjust if needed

lu_agg <- rast(paste0(arealstat_dir, "LULC_2018_agg.tif"))

## ---- Reference grid -------------------------------------------------
# All layers are aligned to this grid. Set the target CRS and
# resolution once here; everything else follows.
target_crs <- "EPSG:2056"   # CH1903+ / LV95
target_res <- 100           # metres

# National outline, dissolved to a single polygon, in the target CRS.
ch <- vect(paste0(data_dir, "CH_shps/swissBOUNDARIES3D_1_4_TLM_LANDESGEBIET.shp"))
ch <- aggregate(ch)
ch <- project(ch, target_crs)

kb_bound  <- vect(paste0(data_dir, "CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"))
kb_only <- kb_bound[kb_bound$NAME == "Bern", ]

# Empty template raster covering CH at the target resolution. Snap the extent
# to the target_res lattice (origin 0,0) so it aligns with the standard Swiss
# grid used by the reference layers. Without this, rast(ch, ...) inherits ch's
# arbitrary extent -> a fractional-cell origin, which forces a second, value-
# scrambling resample whenever a template-gridded layer is compared to a
# reference on the 0/100 lattice (dropped tsd/can correlation ~0.96 -> ~0.72).
e <- ext(ch)
e <- ext(floor(e[1] / target_res) * target_res, ceiling(e[2] / target_res) * target_res,
         floor(e[3] / target_res) * target_res, ceiling(e[4] / target_res) * target_res)
template <- rast(e, resolution = target_res, crs = target_crs)

## ---- Helper ---------------------------------------------------------
# Load (if a path), reproject + resample onto the template grid, then
# mask to the CH outline. project() to a template raster handles CRS,
# extent (crop) and resolution (resample) in one step.
#   method = "bilinear" for continuous data, "near" for categorical.
prep_layer <- function(x, method = "bilinear") {
  if (is.character(x)) x <- rast(x)
  x <- project(x, template, method = method)
  x <- mask(x, ch)
  x
}

# =====================================================================
# A1  Physical state
# =====================================================================
smd <- prep_layer(paste0(data_dir,
  "SoilMoistureDeficit/eea_r_3035_5_km_smoisture-anomalies-gs-2021_p_2000-2021_v01_r00/SMA_GS_2018.tif"))
sbd <- prep_layer(paste0(data_dir,
  "SBD/sol_bulkdens.fineearth_usda.4a1h_m_250m_b10..10cm_1950..2017_v0.2.tif"))

# =====================================================================
# A2  Chemical state
# =====================================================================
soc <- prep_layer(paste0(data_dir, "SOC/Gupta/OC_0cm_mean_30m.tif"))

# =====================================================================
# B1  Compositional
# =====================================================================
uzl <- prep_layer(paste0(sdm_dir, "sdm-2025-rcp45-uzl_npa-all-bau-canton.tif"))
# Tree species mixture degree (waldmischungsgrad): categorical -> near
tsd <- prep_layer(paste0(kb_dir,
  "Restoration_potential/Data/Forests/landesforstinventar-waldmischungsgrad_2056.tif"),
  method = "near")
# Rescale to a unimodal (triangular) score: pure stands (0 and 100) -> 0,
# most-mixed (50) -> 1, linear on each side.
#tsd <- 1 - abs(tsd - 50) / 50

# =====================================================================
# B2  Structural
# =====================================================================
# --- Canopy height: merge the two ETH GlobalCanopyHeight tiles --------
can_1 <- rast(paste0(data_dir, "CAN/ETH_GlobalCanopyHeight_10m_2020_N45E006_Map.tif"))
can_2 <- rast(paste0(data_dir, "CAN/ETH_GlobalCanopyHeight_10m_2020_N45E009_Map.tif"))
can   <- merge(can_1, can_2)
can   <- prep_layer(can)

# --- Small woody features (SWF) from swissTLM3D -----------------------
# Rasterized directly onto the template grid: trees as a per-cell count,
# hedges as fractional cover.
feat_db <- paste0(data_dir, "SWF/swisstlm3d_2025-03_2056_5728.shp/")

trees <- rbind(
  vect(paste0(feat_db, "TLM_BB/swissTLM3D_TLM_EINZELBAUM_GEBUESCH_WEST.shp")),
  vect(paste0(feat_db, "TLM_BB/swissTLM3D_TLM_EINZELBAUM_GEBUESCH_OST.shp"))
)
trees <- crop(project(trees, template), template)
swf_t <- rasterizeGeom(trees, template, fun = "count")
swf_t <- mask(swf_t, ch)

hedges <- rbind(
  vect(paste0(feat_db, "TLM_BB/swissTLM3D_TLM_BODENBEDECKUNG_WEST.shp")),
  vect(paste0(feat_db, "TLM_BB/swissTLM3D_TLM_BODENBEDECKUNG_OST.shp"))
)
hedges <- crop(project(hedges, template), template)
hedges <- hedges[hedges$OBJEKTART == "Gehoelzflaeche", ]
swf_h  <- rasterize(hedges, template, cover = TRUE, background = 0)
swf_h  <- mask(swf_h, ch)

# =====================================================================
# B3  Functional
# =====================================================================
# --- LAI: clean, restrict to forest, gap-fill -------------------------
# 1) remove unrealistic pixels (<0 or >10000)
# 2) fill isolated gaps by neighbourhood averaging (focal mean)
# 3) larger gaps: inverse-distance weighting (see terra::interpIDW;
#    left as a project-specific step below)
lai <- rast(paste0(data_dir, "LAI/researchdata/2021_summer_LAI_nanmedian.tif"))
lai[lai < 0 | lai > 10000] <- NA
lai <- focal(lai, w = 3, fun = mean, na.policy = "only", na.rm = TRUE)
lai <- prep_layer(lai)
# Restrict to forest once a forest mask is available, e.g.:
#   lai <- mask(lai, frag > 0, maskvalues = FALSE)

ndvi <- prep_layer(paste0(data_dir, "NDVI/researchdata/annual_mean/NDVI_2018.tif"))

# =====================================================================
# C1  Landscape
# =====================================================================
# --- SNH: semi-natural habitat density from NOAS04 land cover ---------
# Proportion of focal (semi-natural) land-cover classes within a 300 m
# circular moving window. Computed on the native NOAS04 grid, then
# aligned to the template.
LU_72 <- rast(paste0(arealstat_dir, "AS72_2018.tif"))

focal_classes <- c(42:60, 64:67)                 # semi-natural land-cover classes
snh_bin <- classify(LU_72, cbind(focal_classes, 1), others = 0)

# Binary circular window: 1 inside the 300 m radius, NA outside. fillNA = TRUE
# puts NA (not 0) outside the circle so those cells drop out of the neighbourhood
# instead of being counted as valid zeros in the fraction below.
w <- focalMat(snh_bin, d = 300, type = "circle", fillNA = TRUE)
w[!is.na(w)] <- 1
snh <- focal(snh_bin, w = w, na.policy = "omit",
             fun = function(x) sum(x, na.rm = TRUE) / sum(!is.na(x)))
snh <- prep_layer(snh)

# --- Forest fragmentation / density -----------------------------------
frag <- prep_layer(paste0(kb_dir,
  "Restoration_potential/Data/Forests/CH_forest_density_ext.tif"))

# =====================================================================
# Combine into one aligned, CH-masked raster stack
# =====================================================================
EC_stack <- c(smd, sbd, soc, uzl, tsd, can, swf_h, swf_t, lai, ndvi, snh, frag)
names(EC_stack) <- c("smd", "sbd", "soc", "uzl", "tsd", "can",
                     "swf_h", "swf_t", "lai", "ndvi", "snh", "frag")


# Write each processed layer to its own single-band tif (named after the layer)
layer_dir <- paste0(kb_dir, "Restoration_potential/Data/EC_layers/")
dir.create(layer_dir, showWarnings = FALSE, recursive = TRUE)
for (nm in names(EC_stack)) {
  writeRaster(EC_stack[[nm]], paste0(layer_dir, nm, ".tif"),
              overwrite = TRUE)
}

writeRaster(EC_stack, paste0(kb_dir, "Restoration_potential/Data/EC_stack.tif"),
            overwrite = TRUE)

# =====================================================================
# Comparison against existing cropped reference layers
# ---------------------------------------------------------------------
# The layers in .../Restoration_potential/Data were produced earlier for KB
# For each newly created layer we align it to its reference (same grid AND
# the reference's smaller extent), then  report how closely the values agree.
#
# Outputs:
#   1. a printed summary table (one row per layer)
#   2. comparison_summary.csv
#   3. one PNG per layer: new | reference | difference maps
# =====================================================================

ref_dir  <- "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/"
comp_dir <- paste0(ref_dir, "comparisons/")
dir.create(comp_dir, showWarnings = FALSE)

# Forest-only mask (NOAS04 aggregated land-use classes 12 & 13). Used to
# restrict the canopy-height comparison to forest cells. 1 on forest, NA
# elsewhere so it can be used directly with mask().
forest_mask <- lu_agg == 12 | lu_agg == 13
forest_mask[forest_mask == 0] <- NA

# newly created layer object -> reference file (relative to ref_dir).
# "method" is the resampling used to put the new layer on the reference
# grid: "near" for counts/categorical, "bilinear" for continuous.
# "fmask" (optional) is an extra raster mask; only cells valid in it are kept.
comparisons <- list(
  smd   = list(mine = smd,   ref = "smd.tif",                           method = "bilinear"),
  sbd   = list(mine = sbd,   ref = "sbd.tif",                           method = "bilinear"),
  soc   = list(mine = soc,   ref = "soc.tif",                           method = "bilinear"),
  ndvi  = list(mine = ndvi,  ref = "ndvi.tif",                          method = "bilinear"),
  swf_t = list(mine = swf_t, ref = "swf_t.tif",                     method = "near"),
  swf_h = list(mine = swf_h, ref = "swf_h.tif",                    method = "bilinear"),
  snh   = list(mine = snh,   ref = "snh.tif",                       method = "bilinear"),
  frag  = list(mine = frag,  ref = "frag.tif", method = "bilinear"),
  tsd   = list(mine = tsd,   ref = "tsd.tif",    method = "near",
               fmask = forest_mask),
  lai   = list(mine = lai,   ref = "lai.tif",               method = "bilinear",
               fmask = forest_mask),
  can   = list(mine = can,   ref = "can_lang.tif",           method = "bilinear",
               fmask = forest_mask)
)

# Align one new layer to a reference and compute agreement metrics.
# bound restricts the comparison to cells inside that polygon (Kanton Bern).
# fmask is an optional raster whose valid (non-NA) cells are kept (e.g. forest).
compare_layer <- function(mine, ref, method = "bilinear", bound = NULL, fmask = NULL) {
  m  <- project(mine, ref, method = method)   # match CRS, grid and (smaller) extent
  if (!is.null(bound)) {                       # keep only cells within the boundary
    b   <- project(bound, crs(ref))
    m   <- mask(m,   b)
    ref <- mask(ref, b)
  }
  if (!is.null(fmask)) {                        # keep only cells valid in fmask
    fm  <- project(fmask, ref, method = "near")
    m   <- mask(m,   fm)
    ref <- mask(ref, fm)
  }
  vm <- as.vector(values(m))
  vr <- as.vector(values(ref))
  ok <- is.finite(vm) & is.finite(vr)         # cells valid in BOTH layers
  vm <- vm[ok]; vr <- vr[ok]
  d  <- vm - vr
  list(
    aligned = m,
    stats = data.frame(
      n_common  = length(vm),
      mean_mine = mean(vm),
      mean_ref  = mean(vr),
      mean_diff = mean(d),
      med_diff  = median(d),
      rmse      = sqrt(mean(d^2)),
      cor       = if (length(vm) > 2 && sd(vm) > 0 && sd(vr) > 0) cor(vm, vr) else NA_real_
    )
  )
}

comparison_summary <- data.frame()
low_cor <- list()                               # layers with cor < 0.75, for a joint plot
for (nm in names(comparisons)) {
  cfg <- comparisons[[nm]]
  ref <- rast(paste0(ref_dir, cfg$ref))
  ref[ref == -128] <- NA                        # -128 is a NoData sentinel, not data
  res <- compare_layer(cfg$mine, ref, cfg$method, bound = kb_only, fmask = cfg$fmask)

  # Restrict the maps to Kanton Bern too, so they match the metrics.
  b   <- project(kb_only, crs(ref))
  ref <- mask(ref, b)
  if (!is.null(cfg$fmask)) {                    # and to forest for masked layers
    ref <- mask(ref, project(cfg$fmask, ref, method = "near"))
  }

  # new | reference | difference maps
  png(paste0(comp_dir, nm, "_comparison.png"), width = 1500, height = 500)
  par(mfrow = c(1, 3))
  plot(res$aligned,       main = paste0(nm, " - new"))
  plot(ref,               main = paste0(nm, " - reference"))
  plot(res$aligned - ref, main = paste0(nm, " - difference (new - ref)"))
  dev.off()

  # Keep poorly-agreeing layers for a combined mine-vs-ref figure.
  if (is.finite(res$stats$cor) && res$stats$cor < 0.75) {
    low_cor[[nm]] <- list(mine = res$aligned, ref = ref, cor = res$stats$cor)
  }

  comparison_summary <- rbind(comparison_summary, cbind(layer = nm, res$stats))
}

# Combined figure: mine | ref side by side, one row per layer with cor < 0.75.
if (length(low_cor) > 0) {
  png(paste0(comp_dir, "low_cor_mine_vs_ref.png"),
      width = 1000, height = 500 * length(low_cor))
  par(mfrow = c(length(low_cor), 2))
  for (nm in names(low_cor)) {
    lc <- low_cor[[nm]]
    plot(lc$mine, main = sprintf("%s - new (cor = %.2f)", nm, lc$cor))
    plot(lc$ref,  main = paste0(nm, " - reference"))
  }
  dev.off()
}

# Round for readability and report.
num_cols <- sapply(comparison_summary, is.numeric)
comparison_summary[num_cols] <- lapply(comparison_summary[num_cols], round, 4)

print(comparison_summary, row.names = FALSE)
write.csv(comparison_summary, paste0(comp_dir, "comparison_summary.csv"), row.names = FALSE)
