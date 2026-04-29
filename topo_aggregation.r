library(terra)
library(gstat)
library(ggplot2)
library(sf)
library(patchwork)

raster_path <- "inputs/implementation_cost.tif"
plot_dir <- "topo_aggregation_plots"
n_variogram_samples <- 5000
n_seeds <- 100
idw_power <- 2
idw_nmax <- 12
random_seed <- 1

dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

r <- rast(raster_path)

if (nlyr(r) != 1) {
  stop("This workflow expects a single-layer raster.")
}

if (!hasValues(r)) {
  stop("The input raster has no values.")
}

sample_variogram_points <- function(x, n, seed = 1) {
  valid_cells <- which(!is.na(values(x)))

  if (!length(valid_cells)) {
    stop("No non-missing raster cells are available for variogram sampling.")
  }

  set.seed(seed)
  sampled_cells <- sample(valid_cells, size = min(n, length(valid_cells)))
  xy <- xyFromCell(x, sampled_cells)
  z <- values(x)[sampled_cells]

  data.frame(
    x = xy[, 1],
    y = xy[, 2],
    z = z
  )
}

estimate_range <- function(x, n = 5000, seed = 1) {
  pts <- sample_variogram_points(x, n = n, seed = seed)
  raster_extent <- ext(x)
  raster_res    <- max(res(x))
  dmax_full <- 0.5 * sqrt(
    diff(raster_extent[c(1, 2)])^2 + diff(raster_extent[c(3, 4)])^2
  )
  # cap cutoff: no point fitting beyond 50 cells — avoids inflated range from
  # large-scale trends or a variogram that never reaches a sill
  dmax <- min(dmax_full, raster_res * 50)

  v <- variogram(
    z ~ 1,
    locations = ~ x + y,
    data = pts,
    cutoff = dmax,
    width = dmax / 15
  )
  m <- fit.variogram(v, vgm("Sph"))

  p_v <- ggplot(v, aes(x = dist, y = gamma)) +
    geom_point(size = 2) +
    geom_line(alpha = 0.4) +
    labs(x = "Distance", y = "Semivariance", title = "Empirical variogram") +
    theme_minimal()

  pred_df <- variogramLine(m, maxdist = max(v$dist), n = 200)
  p_vm <- p_v +
    geom_line(data = pred_df, aes(x = dist, y = gamma),
              colour = "steelblue", linewidth = 1) +
    labs(title = "Variogram with fitted spherical model")

  ggsave(file.path(plot_dir, "variogram_empirical.png"), plot = p_v, width = 24, height = 14, units = "cm", dpi = 200)
  ggsave(file.path(plot_dir, "variogram_fitted_model.png"), plot = p_vm, width = 24, height = 14, units = "cm", dpi = 200)

  print(p_v)
  print(p_vm)

  range_m <- m$range[m$model != "Nug"][1]

  # --- diagnostics ---
  message("Fitted variogram model:")
  print(m)
  message(sprintf(
    "range_m = %.4g  |  raster res = %.4g  |  cutoff used = %.4g  |  full diagonal/2 = %.4g",
    range_m, raster_res, dmax, dmax_full
  ))

  # degenerate fit: range has railed against the cutoff
  if (!is.na(range_m) && range_m >= 0.95 * dmax) {
    warning(
      "Fitted range (", round(range_m, 4), ") is >=95% of the variogram cutoff (", round(dmax, 4), ").\n",
      "The model is likely degenerate — the variogram may not reach a sill within the sampled distances.\n",
      "Check the variogram plots. Consider increasing n_variogram_samples or inspecting for large-scale trends."
    )
    range_m <- NA
  }

  if (is.na(range_m) || range_m <= 0) {
    range_m <- raster_res * 5
    warning("Fitted variogram range was not usable. Falling back to 5 raster cells.")
  }

  range_m
}

smooth_focal <- function(x, range_m) {
  radius_m <- range_m/2
  w <- focalMat(x, d = radius_m, type = "circle")
  smoothed <- focal(x, w = w, fun = mean, na.rm = TRUE)
  mask(smoothed, x)
}

smooth_idw <- function(x, idp = 2, nmax = 12, maxdist = NULL) {
  pts_all <- as.data.frame(x, xy = TRUE, na.rm = TRUE)
  names(pts_all)[3] <- "z"

  r_template <- rast(x)
  values(r_template) <- NA

  idw_model <- gstat(
    formula = z ~ 1,
    locations = ~ x + y,
    data = pts_all,
    set = list(idp = idp)
  )

  if (is.null(maxdist)) {
    maxdist <- range_m/2
  }

  smoothed <- interpolate(
    r_template,
    idw_model,
    xyNames = c("x", "y"),
    nmax = nmax,
    maxdist = maxdist,
    debug.level = 0
  )

  smoothed <- smoothed[["var1.pred"]]
  mask(smoothed, x)
}

make_segmentation <- function(smoothed_raster, mask_raster, n_seeds = 100, seed = 1) {
  set.seed(seed)
  seed_points <- spatSample(
    smoothed_raster,
    size = n_seeds,
    method = "random",
    as.points = TRUE,
    na.rm = TRUE
  )

  seg_poly_seed <- voronoi(seed_points)
  seg_poly_seed <- crop(seg_poly_seed, ext(smoothed_raster))
  seg_poly_seed$id <- seq_len(nrow(seg_poly_seed))

  seg_r <- rasterize(seg_poly_seed, smoothed_raster, field = "id")
  seg_r <- mask(seg_r, mask_raster)

  patch_means <- zonal(smoothed_raster, seg_r, fun = "mean", na.rm = TRUE)
  names(patch_means)[2] <- "Mean"

  seg_poly <- as.polygons(seg_r, dissolve = TRUE, na.rm = TRUE)
  seg_poly <- merge(seg_poly, patch_means, by = "id", all.x = TRUE)

  list(
    seed_points = seed_points,
    seg_r = seg_r,
    seg_poly = seg_poly
  )
}

plot_segmentation_comparison <- function(original_raster, focal_segmentation, idw_segmentation) {
  original_values <- values(original_raster)
  original_values <- original_values[!is.na(original_values)]

  focal_sf <- st_as_sf(focal_segmentation$seg_poly)
  idw_sf <- st_as_sf(idw_segmentation$seg_poly)

  focal_vals <- focal_sf$Mean[!is.na(focal_sf$Mean)]
  idw_vals <- idw_sf$Mean[!is.na(idw_sf$Mean)]
  all_vals <- c(original_values, focal_vals, idw_vals)

  robust_limits <- as.numeric(stats::quantile(all_vals, probs = c(0.02, 0.98), na.rm = TRUE))
  if (diff(robust_limits) <= 0) {
    robust_limits <- range(all_vals, na.rm = TRUE)
  }

  fill_scale <- scale_fill_viridis_c(
    option = "viridis",
    limits = robust_limits,
    oob = scales::squish,
    name = "Value"
  )

  common_theme <- theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10),
      legend.position = "right"
    )

  rdf <- as.data.frame(original_raster, xy = TRUE, na.rm = TRUE)
  names(rdf)[3] <- "value"
  p_orig <- ggplot(rdf, aes(x = x, y = y, fill = value)) +
    geom_raster() +
    fill_scale +
    coord_equal() +
    labs(title = "Original raster") +
    common_theme

  p_foc <- ggplot(focal_sf, aes(fill = Mean)) +
    geom_sf(colour = NA) +
    fill_scale +
    coord_sf() +
    labs(title = "Segmentation — focal smoothing") +
    common_theme

  p_idw <- ggplot(idw_sf, aes(fill = Mean)) +
    geom_sf(colour = NA) +
    fill_scale +
    coord_sf() +
    labs(title = "Segmentation — IDW smoothing") +
    common_theme

  hist_theme <- theme_minimal(base_size = 8) +
    theme(
      plot.title = element_blank(),
      axis.title.x = element_text(size = 8),
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "none"
    )

  p_hist_orig <- ggplot(data.frame(value = original_values), aes(x = value, fill = value)) +
    geom_histogram(bins = 30, colour = NA) +
    fill_scale +
    labs(x = "Value") +
    hist_theme

  p_hist_foc <- ggplot(data.frame(value = focal_vals), aes(x = value, fill = value)) +
    geom_histogram(bins = 30, colour = NA) +
    fill_scale +
    labs(x = "Value") +
    hist_theme

  p_hist_idw <- ggplot(data.frame(value = idw_vals), aes(x = value, fill = value)) +
    geom_histogram(bins = 30, colour = NA) +
    fill_scale +
    labs(x = "Value") +
    hist_theme

  panel_orig <- gridExtra::arrangeGrob(p_orig, p_hist_orig, ncol = 1, heights = c(3.5, 1))
  panel_foc  <- gridExtra::arrangeGrob(p_foc, p_hist_foc, ncol = 1, heights = c(3.5, 1))
  panel_idw  <- gridExtra::arrangeGrob(p_idw, p_hist_idw, ncol = 1, heights = c(3.5, 1))

  p_combined <- gridExtra::grid.arrange(panel_orig, panel_foc, panel_idw, ncol = 3)

  ggsave(file.path(plot_dir, "segmentation_comparison_ab.png"), plot = p_combined, width = 36, height = 14, units = "cm", dpi = 200)

  message(sprintf(
    "Map colour limits set to 2nd–98th percentiles: [%.4f, %.4f]",
    robust_limits[1], robust_limits[2]
  ))

  print(p_combined)
}

range_m <- estimate_range(r, n = n_variogram_samples, seed = random_seed)

r_s_focal <- smooth_focal(r, range_m = range_m)
names(r_s_focal) <- "Mean"

r_s_idw <- smooth_idw(
  r,
  idp = idw_power,
  nmax = idw_nmax,
  maxdist = range_m / 2
)
names(r_s_idw) <- "Mean"

segmentation_focal <- make_segmentation(
  r_s_focal,
  mask_raster = r,
  n_seeds = n_seeds,
  seed = random_seed
)

segmentation_idw <- make_segmentation(
  r_s_idw,
  mask_raster = r,
  n_seeds = n_seeds,
  seed = random_seed
)

seg_poly_focal <- segmentation_focal$seg_poly
seg_poly_idw <- segmentation_idw$seg_poly

plot_segmentation_comparison(r, segmentation_focal, segmentation_idw)

message("Saved plots to: ", normalizePath(plot_dir, winslash = "/", mustWork = FALSE))

#reconvert seg_poly_idw to raster for export
seg_r_idw <- rasterize(seg_poly_idw, r, field = "Mean", fun = "mean", background = NA)
writeRaster(seg_r_idw, "cost_idw.tif", overwrite = TRUE)
