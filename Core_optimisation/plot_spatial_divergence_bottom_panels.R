# Produces the two-panel spatial figure from compute_spatial_divergence_by_step():
#   Left  - pathway sensitivity (always selected vs sensitive to pathway)
#   Right - timing disagreement (std of mean timing step across scenarios)
#
# Inputs: GeoTIFFs written by export_spatial_divergence_tifs() in
#         visualise_recovery_uncertainty.py -> r_inputs/spatial_divergence/
#
# Output: figs/spatial_divergence_bottom_panels_r.png
#
# Style follows bivariate_robustness_bern.R:
# theme_void, coord_sf, inset_element.

library(terra)
library(ggplot2)
library(sf)
library(patchwork)
library(dplyr)


# ---- Paths ------------------------------------------------------------------

tif_dir   <- "r_inputs/spatial_divergence"
fig_dir   <- "figs"
bern_shp  <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"
swiss_shp <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_LANDESGEBIET.shp"


# ---- Load rasters -----------------------------------------------------------

n_sel_rast <- rast(file.path(tif_dir, "n_scenarios_selecting.tif"))
tstd_rast  <- rast(file.path(tif_dir, "timing_std.tif"))

n_sel_df <- as.data.frame(n_sel_rast, xy = TRUE) |>
  setNames(c("x", "y", "n_sel")) |>
  filter(!is.na(n_sel))

tstd_df <- as.data.frame(tstd_rast, xy = TRUE) |>
  setNames(c("x", "y", "tstd")) |>
  filter(!is.na(tstd))

n_sel_df$n_sel_f <- dplyr::case_when(
  round(n_sel_df$n_sel) == 0 ~ "Eligible, not selected",
  round(n_sel_df$n_sel) == 4 ~ "Always selected",
  TRUE                       ~ "Sensitive to pathway"
) |>
  factor(levels = c("Eligible, not selected", "Sensitive to pathway", "Always selected"))


# ---- Boundaries -------------------------------------------------------------

bern <- st_read(bern_shp, quiet = TRUE) |>
  filter(NAME == "Bern") |>
  st_transform(crs(n_sel_rast))

swiss_all <- tryCatch(
  st_read(swiss_shp, quiet = TRUE) |> st_transform(st_crs(bern)),
  error = function(e) NULL
)


# ---- Locator inset (matches bivariate_robustness_bern.R) --------------------

locator_p <- if (!is.null(swiss_all)) {
  ggplot() +
    geom_sf(data = swiss_all, fill = "grey80",  colour = "white", linewidth = 0.2) +
    geom_sf(data = bern,      fill = "#c8873a", colour = "white", linewidth = 0.3) +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background  = element_rect(fill = "transparent", colour = NA)
    )
} else {
  NULL
}


# ---- Colour scales ----------------------------------------------------------

agree_colors <- c(
  "Eligible, not selected" = "#f2f0f0",
  "Sensitive to pathway"   = "#ee9d5a",
  "Always selected"        = "#1a9641"
)

yor_colors <- c("#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026")


# ---- Panel 1: scenario robustness -------------------------------------------

map_robust <- ggplot() +
  geom_raster(data = n_sel_df,
              aes(x = x, y = y, fill = n_sel_f),
              show.legend = TRUE) +
  scale_fill_manual(
    values         = agree_colors,
    na.value       = "white",
    name           = "Scenarios selecting patch",
    guide          = guide_legend(title.position = "top")
  ) +
  geom_sf(data        = bern,
          fill        = NA,
          colour      = "black",
          linewidth   = 0.5,
          inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    legend.position  = "bottom",
    legend.text      = element_text(size = 12),
    legend.title     = element_text(size = 12, face = "bold")
  )


# ---- Panel 2: timing disagreement -------------------------------------------

map_tstd <- ggplot() +
  geom_raster(data = tstd_df,
              aes(x = x, y = y, fill = tstd)) +
  scale_fill_gradientn(
    colours  = yor_colors,
    limits   = c(0, 1.5),
    na.value = "white",
    name     = "Timing disagreement (step SD)",
    breaks   = c(0, 0.5, 1.0, 1.5),
    labels   = c("0", "0.5", "1.0", "1.5"),
    guide    = guide_colorbar(
      title.position = "top",
      barwidth       = unit(8, "cm"),
      barheight      = unit(0.4, "cm")
    )
  ) +
  geom_sf(data        = bern,
          fill        = NA,
          colour      = "black",
          linewidth   = 0.5,
          inherit.aes = FALSE) +
  coord_sf(expand = FALSE) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    legend.position  = "bottom",
    legend.text      = element_text(size = 12),
    legend.title     = element_text(size = 12, face = "bold")
  )


# ---- Add locator inset to each panel (matches bivariate_robustness_bern.R) --

if (!is.null(locator_p)) {
  map_robust <- map_robust +
    inset_element(locator_p,
                  left     = 0.75,
                  bottom   = 0.75,
                  right    = 0.98,
                  top      = 0.98,
                  align_to = "full")

  map_tstd <- map_tstd +
    inset_element(locator_p,
                  left     = 0.75,
                  bottom   = 0.75,
                  right    = 0.98,
                  top      = 0.98,
                  align_to = "full")
}


# ---- Combine and save -------------------------------------------------------

final <- map_robust + map_tstd + plot_layout(ncol = 2)

dir.create(fig_dir, showWarnings = FALSE)
ggsave(
  file.path(fig_dir, "spatial_divergence_bottom_panels_r.png"),
  final,
  dpi    = 300,
  width  = 30,
  height = 16,
  units  = "cm"
)

message("Saved: figs/spatial_divergence_bottom_panels_r.png")
