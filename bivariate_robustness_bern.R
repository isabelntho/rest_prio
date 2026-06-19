# Bivariate robustness map — Canton of Bern
# Adapted from Core_optimisation/ben_robustness_figures.r (Map 1 section).
#
# Axes:
#   x  = Mean sum of NCP change     (higher = better performance)
#   y  = Undesirable deviation       (higher = worse robustness / more downside risk)
#
# Inputs:
#   Ben_robustness/Mean_sum_of_change_all_NCPs.tif
#   Ben_robustness/Undesirable_deviation_sum_of_change_all_NCPs.tif
#   A shapefile of the Canton of Bern boundary (set BERN_SHP below)
#
# Output:
#   Ben_robustness/bivariate_robustness_bern.png

library(terra)
library(sf)
library(ggplot2)
library(biscale)
library(patchwork)
library(dplyr)


# ── User settings ─────────────────────────────────────────────────────────────

BERN_SHP    <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"   # <-- set this to your shapefile

MEAN_TIF    <- "Ben_robustness/Mean_sum_of_change_all_NCPs.tif"
UNDES_TIF   <- "Ben_robustness/Undesirable_deviation_sum_of_change_all_NCPs.tif"
OUTPUT_PNG  <- "Ben_robustness/bivariate_robustness_bern.png"

NUM_CLASSES <- 4        # bins per axis: 3 (3×3 = 9 classes) or 4 (4×4 = 16)
PALETTE     <- "BlueOr" # biscale palette; alternatives: "GrPink", "DkViolet", "Brown"
MIN_CLUSTER <- 2        # minimum square block size for cluster filter (2 = 2×2, 3 = 3×3)
SWISS_SHP   <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_LANDESGEBIET.shp"  # <-- set to Switzerland (or all-cantons) shapefile


# ── Load rasters ──────────────────────────────────────────────────────────────

mean_r  <- rast(MEAN_TIF)
undes_r <- rast(UNDES_TIF)


# ── Load and reproject Bern boundary ─────────────────────────────────────────

bern      <- st_read(BERN_SHP, quiet = TRUE)
bern      <- bern %>% filter(NAME == "Bern") # subset to Bern if shapefile contains multiple features
bern      <- st_transform(bern, crs = crs(mean_r))
bern_vect <- vect(bern)


# ── Crop and mask to Bern ─────────────────────────────────────────────────────

mean_bern  <- crop(mean_r,  bern_vect) |> mask(bern_vect)
undes_bern <- crop(undes_r, bern_vect) |> mask(bern_vect)

# Align grids if resolutions differ
if (!compareGeom(mean_bern, undes_bern, stopOnError = FALSE)) {
  message("Resampling undesirable deviation raster to match mean grid...")
  undes_bern <- resample(undes_bern, mean_bern, method = "bilinear")
}


# ── Build classified data frame ───────────────────────────────────────────────

stk        <- c(mean_bern, undes_bern)
names(stk) <- c("Mean", "Var")

df <- as.data.frame(stk, xy = TRUE) |>
  filter(!is.na(Mean), !is.na(Var))

# Normalise undesirable deviation to [0, 1] (required: values must be >= 0)
df$Var_norm <- (df$Var - min(df$Var)) / (max(df$Var) - min(df$Var))

# Shared quantile thresholds — applied to both maps for comparability
q_probs <- (1:(NUM_CLASSES - 1)) / NUM_CLASSES
x_cuts  <- quantile(df$Mean,     probs = q_probs, na.rm = TRUE, names = FALSE)
y_cuts  <- quantile(df$Var_norm, probs = q_probs, na.rm = TRUE, names = FALSE)

df_class <- df |>
  mutate(bi_class = paste0(
    as.integer(cut(Mean,     breaks = c(-Inf, x_cuts, Inf), labels = FALSE)), "-",
    as.integer(cut(Var_norm, breaks = c(-Inf, y_cuts, Inf), labels = FALSE))
  ))

breaks <- bi_class_breaks(df,
                          x       = Mean,
                          y       = Var_norm,
                          style   = "quantile",
                          dim     = NUM_CLASSES,
                          dig_lab = 2,
                          split   = FALSE)


# ── Map ───────────────────────────────────────────────────────────────────────

map_p <- ggplot() +
  geom_raster(data = df_class,
              aes(x = x, y = y, fill = bi_class),
              show.legend = FALSE) +
  bi_scale_fill(pal      = PALETTE,
                dim      = NUM_CLASSES,
                na.value = "white") +
  geom_sf(data        = bern,
          fill        = NA,
          colour      = "black",
          linewidth   = 0.5,
          inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  theme_void() +
  theme(panel.background = element_rect(fill = "white", colour = NA))


# ── Location inset ────────────────────────────────────────────────────────────

swiss_all <- st_read(SWISS_SHP, quiet = TRUE) |>
  st_transform(crs = st_crs(bern))

locator_p <- ggplot() +
  geom_sf(data = swiss_all, fill = "grey80", colour = "white", linewidth = 0.2) +
  geom_sf(data = bern,      fill = "#c8873a", colour = "white", linewidth = 0.3) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA)
  )


# ── Legend ────────────────────────────────────────────────────────────────────

legend_p <- bi_legend(
  pal        = PALETTE,
  dim        = NUM_CLASSES,
  breaks     = breaks,
  xlab       = "Mean sum of change\n in NCP provision",
  ylab       = "Decreasing undesirable\n deviation",
  flip_axes  = FALSE,
  rotate_pal = FALSE,
  arrow      = FALSE
) +
  theme(
    panel.background = element_rect(fill = "transparent"),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    axis.text.x      = element_blank(),
    axis.text.y      = element_blank(),
    axis.title       = element_text(size = 8)
  )


# ── Combine and save ──────────────────────────────────────────────────────────

final <- map_p +
  inset_element(legend_p,
                left     = 0.72,
                bottom   = 0.65,
                right    = 1.0,
                top      = 1.0,
                align_to = "full") +
  inset_element(locator_p,
                left     = 0.75,
                bottom   = 0.49,
                right    = 0.98,
                top      = 0.64,
                align_to = "full")

ggsave(OUTPUT_PNG, final,
       dpi    = 300,
       width  = 20,
       height = 16,
       units  = "cm")

message("Saved: ", OUTPUT_PNG)


# ── Sens-masked bivariate map (class 3 pixels only) ──────────────────────────

sens_dir <- "C:/Users/inicholson/Documents/rest_prio/notebook_outputs/grid_plots/rlc_px/figures/robust_approach_C_map.tif"
sens_raw  <- rast(sens_dir)

# Assign CRS (sens has none stored) then align to Bern grid
crs(sens_raw)  <- crs(mean_bern)
sens_aligned   <- resample(crop(sens_raw, mean_bern), mean_bern, method = "near")

# Keep only class-3 pixels
sens_mask  <- ifel(sens_aligned >1, 1, NA)
#sens_mask <- sens_aligned

mean_sens  <- mask(mean_bern,  sens_mask)
undes_sens <- mask(undes_bern, sens_mask)

# Build classified data frame
stk_sens        <- c(mean_sens, undes_sens)
names(stk_sens) <- c("Mean", "Var")

df_sens <- as.data.frame(stk_sens, xy = TRUE) |>
  filter(!is.na(Mean), !is.na(Var))

# Normalise using full Bern range so colours are comparable to first map
df_sens$Var_norm <- (df_sens$Var - min(df$Var)) / (max(df$Var) - min(df$Var))

df_sens_class <- df_sens |>
  mutate(bi_class = paste0(
    as.integer(cut(Mean,     breaks = c(-Inf, x_cuts, Inf), labels = FALSE)), "-",
    as.integer(cut(Var_norm, breaks = c(-Inf, y_cuts, Inf), labels = FALSE))
  ))

# Map
map_sens <- ggplot() +
  geom_raster(data = df_sens_class,
              aes(x = x, y = y, fill = bi_class),
              show.legend = FALSE) +
  bi_scale_fill(pal      = PALETTE,
                dim      = NUM_CLASSES,
                na.value = "white") +
  geom_sf(data        = bern,
          fill        = NA,
          colour      = "black",
          linewidth   = 0.5,
          inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  theme_void() +
  theme(panel.background = element_rect(fill = "white", colour = NA))

final_sens <- map_sens +
  inset_element(legend_p,
                left     = 0.72,
                bottom   = 0.65,
                right    = 1.0,
                top      = 1.0,
                align_to = "full") +
  inset_element(locator_p,
                left     = 0.75,
                bottom   = 0.49,
                right    = 0.98,
                top      = 0.64,
                align_to = "full")

OUTPUT_PNG_SENS <- "Ben_robustness/bivariate_robustness_bern_sens.png"
ggsave(OUTPUT_PNG_SENS, final_sens,
       dpi    = 300,
       width  = 20,
       height = 16,
       units  = "cm")

message("Saved: ", OUTPUT_PNG_SENS)


# ── Clustered version: remove isolated pixels (connected patches < 4 pixels) ──

# Morphological opening: erode then dilate with a MIN_CLUSTER x MIN_CLUSTER kernel.
# terra requires odd kernel sizes; round even values up by 1.
ksize    <- if (MIN_CLUSTER %% 2 == 0) MIN_CLUSTER + 1 else MIN_CLUSTER
w_kernel <- matrix(1, ksize, ksize)

sens_bin     <- ifel(is.na(sens_mask), 0, 1)
eroded       <- focal(sens_bin, w_kernel, fun = "min")   # 1 only where full block is class-3
cluster_mask <- ifel(focal(eroded, w_kernel, fun = "max") == 1 & sens_bin == 1, 1, NA)

n_sens  <- global(sens_mask,    "notNA")[[1]]
n_clust <- global(cluster_mask, "notNA")[[1]]
message(sprintf("Sens pixels: %d  |  After 2x2 filter: %d  (removed %.1f%%)",
                n_sens, n_clust, 100 * (1 - n_clust / n_sens)))

mean_clust  <- mask(mean_bern,  cluster_mask)
undes_clust <- mask(undes_bern, cluster_mask)

stk_clust        <- c(mean_clust, undes_clust)
names(stk_clust) <- c("Mean", "Var")

df_clust <- as.data.frame(stk_clust, xy = TRUE) |>
  filter(!is.na(Mean), !is.na(Var))

df_clust$Var_norm <- (df_clust$Var - min(df$Var)) / (max(df$Var) - min(df$Var))

df_clust_class <- df_clust |>
  mutate(bi_class = paste0(
    as.integer(cut(Mean,     breaks = c(-Inf, x_cuts, Inf), labels = FALSE)), "-",
    as.integer(cut(Var_norm, breaks = c(-Inf, y_cuts, Inf), labels = FALSE))
  ))

map_clust <- ggplot() +
  geom_raster(data = df_clust_class,
              aes(x = x, y = y, fill = bi_class),
              show.legend = FALSE) +
  bi_scale_fill(pal      = PALETTE,
                dim      = NUM_CLASSES,
                na.value = "white") +
  geom_sf(data        = bern,
          fill        = NA,
          colour      = "black",
          linewidth   = 0.5,
          inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  theme_void() +
  theme(panel.background = element_rect(fill = "white", colour = NA))

final_clust <- map_clust +
  inset_element(legend_p,
                left     = 0.72,
                bottom   = 0.65,
                right    = 1.0,
                top      = 1.0,
                align_to = "full") +
  inset_element(locator_p,
                left     = 0.75,
                bottom   = 0.49,
                right    = 0.98,
                top      = 0.64,
                align_to = "full")

OUTPUT_PNG_CLUST <- "Ben_robustness/bivariate_robustness_bern_clust.png"
ggsave(OUTPUT_PNG_CLUST, final_clust,
       dpi    = 300,
       width  = 20,
       height = 16,
       units  = "cm")

message("Saved: ", OUTPUT_PNG_CLUST)


# ── Bar chart: future value vs stability ─────────────────────────────────────
# Collapse 4 quantile bins  Low / Medium / High for each axis.
# Note: y_class is undesirable deviation, so high class = low stability.

bar_df <- df_sens_class |>
  mutate(
    x_class = as.integer(sub("-.*", "", bi_class)),
    y_class = as.integer(sub(".*-", "", bi_class)),
    Value = factor(
      case_when(x_class == 1 ~ "Low", x_class %in% c(2, 3) ~ "Medium", TRUE ~ "High"),
      levels = c("Low", "Medium", "High")
    ),
    Stability = factor(
      case_when(y_class == 4 ~ "Low", y_class %in% c(2, 3) ~ "Medium", TRUE ~ "High"),
      levels = c("Low", "Medium", "High")
    )
  ) |>
  count(Value, Stability)

# Extract hex colours directly from the map scale at mid-value, varying stability.
# Classes used: "2-1" (high stability), "2-<mid>" (medium), "2-<max>" (low).
ref_df   <- data.frame(
  bi_class = c(paste0("2-1"),
               paste0("2-", ceiling(NUM_CLASSES / 2)),
               paste0("2-", NUM_CLASSES)),
  x = 1:3, y = 1
)
ref_cols <- ggplot_build(
  ggplot(ref_df, aes(x = x, y = y, fill = bi_class)) +
    geom_tile() +
    bi_scale_fill(pal = PALETTE, dim = NUM_CLASSES)
)$data[[1]] |>
  arrange(x) |>
  pull(fill)

stab_cols <- setNames(ref_cols, c("High", "Medium", "Low"))

bar_p <- ggplot(bar_df, aes(x = Value, y = n, fill = Stability)) +
  geom_col(position = "dodge", width = 0.7, colour = "white") +
  scale_fill_manual(values = stab_cols, name = "Stability") +
  labs(
    x     = "Future value",
    y     = "Number of pixels"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position    = "right",
    panel.grid.major.y = element_line(colour = "grey90"),
    plot.title         = element_text(face = "bold")
  )

OUTPUT_BAR <- "Ben_robustness/bar_sens_value_stability.png"
ggsave(OUTPUT_BAR, bar_p, dpi = 300, width = 18, height = 12, units = "cm")
message("Saved: ", OUTPUT_BAR)


# ── Bar chart: clustered pixels only ─────────────────────────────────────────

bar_clust_df <- df_clust_class |>
  mutate(
    x_class = as.integer(sub("-.*", "", bi_class)),
    y_class = as.integer(sub(".*-", "", bi_class)),
    Value = factor(
      case_when(x_class == 1 ~ "Low", x_class %in% c(2, 3) ~ "Medium", TRUE ~ "High"),
      levels = c("Low", "Medium", "High")
    ),
    Stability = factor(
      case_when(y_class == 4 ~ "Low", y_class %in% c(2, 3) ~ "Medium", TRUE ~ "High"),
      levels = c("Low", "Medium", "High")
    )
  ) |>
  count(Value, Stability)

bar_clust_p <- ggplot(bar_clust_df, aes(x = Value, y = n, fill = Stability)) +
  geom_col(position = "dodge", width = 0.7, colour = "white") +
  scale_fill_manual(values = stab_cols, name = "Stability") +
  labs(
    x = "Future value",
    y = "Number of pixels"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position    = "right",
    panel.grid.major.y = element_line(colour = "grey90"),
    plot.title         = element_text(face = "bold")
  )

OUTPUT_BAR_CLUST <- "Ben_robustness/bar_clust_value_stability.png"
ggsave(OUTPUT_BAR_CLUST, bar_clust_p, dpi = 300, width = 18, height = 12, units = "cm")
message("Saved: ", OUTPUT_BAR_CLUST)


# ── Simple flat-colour map of clustered sens pixels ───────────────────────────

clust_df_xy <- as.data.frame(sens_mask, xy = TRUE) |> na.omit()

map_clust_simple <- ggplot() +
  geom_raster(data = clust_df_xy,
              aes(x = x, y = y),
              fill = "#2E7D32",
              show.legend = FALSE) +
  geom_sf(data        = bern,
          fill        = NA,
          colour      = "black",
          linewidth   = 0.5,
          inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  theme_void() +
  theme(panel.background = element_rect(fill = "white", colour = NA))

final_clust_simple <- map_clust_simple +
  inset_element(locator_p,
                left     = 0.70,
                bottom   = 0.75,
                right    = 0.98,
                top      = 0.95,
                align_to = "full")

OUTPUT_PNG_CLUST_SIMPLE <- "Ben_robustness/bivariate_robustness_bern_clust_simple.png"
ggsave(OUTPUT_PNG_CLUST_SIMPLE, final_clust_simple,
       dpi    = 300,
       width  = 20,
       height = 16,
       units  = "cm")

message("Saved: ", OUTPUT_PNG_CLUST_SIMPLE)


# ── Individual univariate maps ────────────────────────────────────────────────
# 4 discrete quantile bins using the same x_cuts / y_cuts as the bivariate map,
# so colours are directly comparable across the two map types.

make_univar_map <- function(df_in, var_col, cuts, fill_cols,
                            fill_name, bern_sf) {
  df_plot <- df_in
  df_plot$uni_class <- factor(
    as.integer(cut(df_plot[[var_col]],
                   breaks = c(-Inf, cuts, Inf),
                   labels = FALSE))
  )
  ggplot() +
    geom_raster(data = df_plot,
                aes(x = x, y = y, fill = uni_class),
                show.legend = TRUE) +
    scale_fill_manual(values   = fill_cols,
                      name     = fill_name,
                      na.value = "white",
                      labels   = paste0("Q", seq_len(NUM_CLASSES))) +
    geom_sf(data        = bern_sf,
            fill        = NA,
            colour      = "black",
            linewidth   = 0.5,
            inherit.aes = FALSE) +
    coord_sf(expand = FALSE) +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position  = "right",
      legend.title     = element_text(size = 8),
      legend.text      = element_text(size = 7)
    )
}

orange_cols <- setNames(
  colorRampPalette(RColorBrewer::brewer.pal(9, "Oranges"))(NUM_CLASSES),
  as.character(seq_len(NUM_CLASSES))
)
blue_cols <- setNames(
  colorRampPalette(RColorBrewer::brewer.pal(9, "Blues"))(NUM_CLASSES),
  as.character(seq_len(NUM_CLASSES))
)

univar_cfgs <- list(
  list(df = df,       var = "Mean", cuts = x_cuts, cols = orange_cols,
       name = "Mean sum of change\nin NCP provision",
       out  = "univar_bern_mean.png"),
  list(df = df,       var = "Var_norm", cuts = y_cuts, cols = blue_cols,
       name = "Undesirable\ndeviation",
       out  = "univar_bern_undes.png"),
  list(df = df_sens,  var = "Mean", cuts = x_cuts, cols = orange_cols,
       name = "Mean sum of change\nin NCP provision",
       out  = "univar_sens_mean.png"),
  list(df = df_sens,  var = "Var_norm", cuts = y_cuts, cols = blue_cols,
       name = "Undesirable\ndeviation",
       out  = "univar_sens_undes.png"),
  list(df = df_clust, var = "Mean", cuts = x_cuts, cols = orange_cols,
       name = "Mean sum of change\nin NCP provision",
       out  = "univar_clust_mean.png"),
  list(df = df_clust, var = "Var_norm", cuts = y_cuts, cols = blue_cols,
       name = "Undesirable\ndeviation",
       out  = "univar_clust_undes.png")
)

for (cfg in univar_cfgs) {
  p <- make_univar_map(cfg$df, cfg$var, cfg$cuts, cfg$cols,
                       cfg$name, bern) +
    inset_element(locator_p,
                  left     = 0.75,
                  bottom   = 0.75,
                  right    = 0.98,
                  top      = 0.98,
                  align_to = "full")
  out_path <- file.path("Ben_robustness", cfg$out)
  ggsave(out_path, p, dpi = 300, width = 20, height = 16, units = "cm")
  message("Saved: ", out_path)
}