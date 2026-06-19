# plot_input_layers.R
# ─────────────────────────────────────────────────────────────────────────────
# Loads and plots the five primary input raster layers with distinct colour
# schemes for each layer.
#
# Run from the workspace root (rest_prio/).
# ─────────────────────────────────────────────────────────────────────────────

library(terra)
library(ggplot2)
library(dplyr)
library(tidyr)

# ── File paths ────────────────────────────────────────────────────────────────
layer_files <- c(
  Abiotic             = "inputs/abiotic_condition_anomaly.tif",
  Biotic              = "inputs/biotic_condition_anomaly.tif",
  Cost                = "inputs/implementation_cost_corrected.tif",
  Restoration_Potential = "inputs/restoration_potential.tif",
  Landscape_Context   = "inputs/landscape_context.tif"
)

missing <- layer_files[!file.exists(layer_files)]
if (length(missing) > 0) {
  warning("Missing files (skipped): ", paste(names(missing), collapse = ", "))
  layer_files <- layer_files[file.exists(layer_files)]
}

# ── Load rasters ──────────────────────────────────────────────────────────────
rasters <- lapply(layer_files, rast)

# Crop all to common extent of first layer
ref_ext <- ext(rasters[[1]])
rasters <- lapply(rasters, function(r) crop(r, ref_ext))

# ── Convert to long data frame ────────────────────────────────────────────────
df_list <- lapply(names(rasters), function(nm) {
  as.data.frame(rasters[[nm]], xy = TRUE) |>
    rename(value = 3) |>
    mutate(Layer = nm)
})
df <- bind_rows(df_list) |>
  filter(is.finite(value)) |>
  mutate(Layer = factor(Layer, levels = names(rasters)))

# ── Colour palettes per layer ─────────────────────────────────────────────────
palettes <- list(
  Abiotic               = scale_fill_distiller(palette = "BrBG",  direction =  1, name = "Abiotic"),
  Biotic                = scale_fill_distiller(palette = "BrBG",     direction =  1, name = "Biotic"),
  Cost                  = scale_fill_distiller(palette = "BrBG",   direction = -1, name = "Cost"),
  Restoration_Potential = scale_fill_distiller(palette = "BrBG",    direction =  1, name = "Rest. potential"),
  Landscape_Context     = scale_fill_distiller(palette = "BrBG",     direction =  1, name = "Landscape ctx")
)

# ── Build one plot per layer, then arrange ────────────────────────────────────
plots <- lapply(names(rasters), function(nm) {
  sub_df <- filter(df, Layer == nm)
  ggplot(sub_df, aes(x = x, y = y, fill = value)) +
    geom_raster() +
    palettes[[nm]] +
    coord_equal() +
    ggtitle(gsub("_", " ", nm)) +
    theme_minimal(base_size = 9) +
    theme(
      axis.title  = element_blank(),
      axis.text   = element_blank(),
      axis.ticks  = element_blank(),
      panel.grid  = element_blank(),
      plot.title  = element_text(hjust = 0.5, face = "bold"),
      legend.key.width = unit(0.4, "cm")
    )
})

# ── Combine and display ───────────────────────────────────────────────────────
if (!requireNamespace("patchwork", quietly = TRUE)) {
  message("Install 'patchwork' for a combined layout: install.packages('patchwork')")
  for (p in plots) print(p)
} else {
  library(patchwork)
  combined <- Reduce(`+`, plots) + plot_layout(ncol = 3)
  print(combined)
}
