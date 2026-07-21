# _plot_functions.R
# All plotting functions for opt_results_comparison.qmd.
# Sourced once in the setup chunk so both per-run tabs and the comparison
# section can share a single definition of every function.
#
# Global objects expected to exist in the calling environment:
#   COL_ND, COL_DOM, ALPHA_ND, ALPHA_DOM  - colour constants from config chunk
#   BE                                     - sf object for canton boundary
#   lulc, abiotic_anomaly                  - terra SpatRaster objects
#   ECOSYSTEM                              - character scalar for plot titles


# -- Spatial filter to Bern canton ---------------------------------------------
# Keep only rows whose (x, y) falls within the BE polygon.
# Uses .BE_MASK (terra SpatRaster at 100 m, precomputed in produce_figures.R).
# terra::extract is a fast raster-cell lookup -- no polygon intersection needed.
# Returns df unchanged when the mask is unavailable or df is empty.
.filter_within_be <- function(df) {
  if (!exists(".BE_MASK") || nrow(df) == 0L) return(df)
  pts  <- terra::vect(data.frame(x = df$x, y = df$y),
                      geom = c("x", "y"), crs = "EPSG:2056")
  vals <- terra::extract(.BE_MASK, pts)[, 2L]
  df[!is.na(vals), , drop = FALSE]
}


# -- Objective label lookup ----------------------------------------------------
# Hard-coded display labels for known objective names.
# Unknown objectives fall back to title-cased name.
.OBJ_LABELS <- c(
  abiotic_anomaly      = "Abiotic condition",
  biotic_anomaly       = "Biotic condition",
  implementation_cost  = "Implementation cost",
  landscape_context    = "Landscape context",
  connectivity_gain    = "Connectivity gain",
  restoration_potential = "Restoration potential"
)

.obj_labels <- function(obj_names) {
  vapply(obj_names, function(n) {
    if (n %in% names(.OBJ_LABELS)) .OBJ_LABELS[[n]]
    else gsub("_", " ", n) |> tools::toTitleCase()
  }, character(1), USE.NAMES = FALSE)
}

# -- Algorithm evolution -------------------------------------------------------

prep_evo_data <- function(run, obj_names = run$obj_names, obj_labels = .obj_labels(obj_names)) {
  if (is.null(run$df_pop)) return(NULL)
  pop <- run$df_pop

  long_rows <- lapply(seq_along(obj_names), function(i) {
    obj      <- obj_names[i]
    lbl      <- obj_labels[i]
    mean_col <- paste0(obj, "_mean")
    std_col  <- paste0(obj, "_std")

    if (!mean_col %in% names(pop)) return(NULL)

    data.frame(
      generation = pop$generation,
      value      = pop[[mean_col]],
      ymin       = pop[[mean_col]] - if (std_col %in% names(pop)) pop[[std_col]] else NA_real_,
      ymax       = pop[[mean_col]] + if (std_col %in% names(pop)) pop[[std_col]] else NA_real_,
      metric     = lbl,
      stringsAsFactors = FALSE
    )
  })

  bind_rows(Filter(Negate(is.null), long_rows))
}

make_hv_evo_plot <- function(run, obj_names = run$obj_names, obj_labels = .obj_labels(obj_names)) {
  hv_data <- NULL

  if (!is.null(run$df_hv)) {
    hv_data <- run$df_hv %>%
      transmute(
        generation = generation,
        value      = hypervolume,
        ymin       = NA_real_,
        ymax       = NA_real_,
        metric     = "Hypervolume"
      )
  }

  evo_data  <- prep_evo_data(run, obj_names, obj_labels)
  plot_data <- bind_rows(hv_data, evo_data)

  metric_levels        <- c("Hypervolume", obj_labels)
  plot_data$metric     <- factor(plot_data$metric, levels = metric_levels)

  ggplot(plot_data, aes(x = generation, y = value)) +
    geom_ribbon(
      data = subset(plot_data, !is.na(ymin)),
      aes(ymin = ymin, ymax = ymax),
      fill = "steelblue", alpha = 0.15
    ) +
    geom_line(colour = "steelblue", linewidth = 0.8) +
    facet_wrap(~metric, scales = "free_y", ncol = 2) +
    scale_y_continuous(labels = scales::scientific) +
    labs(
      title = "Hypervolume and objective evolution (mean \u00b1 1 SD)",
      x = "Generation", y = NULL
    ) +
    theme(strip.text = element_text(size = 10, face = "bold"))
}


# -- Pareto front --------------------------------------------------------------

make_pareto_extremes_plot <- function(run, obj_names = run$obj_names, obj_labels = .obj_labels(obj_names), title = NULL,
                                       scale_by_n_pixels = FALSE, normalise = TRUE) {
  df <- run$df_obj

  # Only use objectives present in this run's data
  present    <- obj_names[obj_names %in% names(df)]
  obj_labels <- .obj_labels(present)
  obj_names  <- present

  if (length(obj_names) < 3) stop("make_pareto_extremes_plot requires at least 3 objectives in the data.")

  # Positional mapping: [1]=x, [2]=y, [3]=fill, [4]=size (optional)
  o1 <- obj_names[1]; o2 <- obj_names[2]; o3 <- obj_names[3]
  l1 <- obj_labels[1]; l2 <- obj_labels[2]; l3 <- obj_labels[3]
  has_size <- length(obj_names) >= 4
  o4 <- if (has_size) obj_names[4] else NULL
  l4 <- if (has_size) obj_labels[4] else NULL

  if (scale_by_n_pixels && !is.null(run$df_psel)) {
    n_pix <- run$df_psel |>
      dplyr::count(solution_id, name = "n_pixels")
    df <- df |>
      dplyr::left_join(n_pix, by = "solution_id") |>
      dplyr::mutate(dplyr::across(dplyr::all_of(obj_names), ~ . / n_pixels)) |>
      dplyr::select(-n_pixels)
    l1 <- paste0(l1, " / pixel")
    l2 <- paste0(l2, " / pixel")
    l3 <- paste0(l3, " / pixel")
  }

  if (normalise) {
    # Rescale each objective to [0, 1] so all are on a comparable scale.
    # Constant columns map to 0 to avoid division by zero.
    .rescale01 <- function(x) {
      rng <- range(x, na.rm = TRUE)
      if (!is.finite(rng[1]) || diff(rng) == 0) return(rep(0, length(x)))
      (x - rng[1]) / diff(rng)
    }
    df <- df |>
      dplyr::mutate(dplyr::across(dplyr::all_of(obj_names), .rescale01))
  }

  if (has_size) {
    p <- ggplot(df, aes(x = .data[[o1]], y = .data[[o2]])) +
      geom_point(aes(fill = .data[[o3]], size = .data[[o4]]), shape = 21, colour = "grey70",
                 stroke = 0.8, alpha = 0.8) +
      scale_fill_viridis_c(name = l3, option = "plasma") +
      scale_size_continuous(name = l4, range = c(1.5, 6)) +
      guides(size = guide_legend(override.aes = list(shape = 21, fill = "grey60"))) +
      labs(x = l1, y = l2)
  } else {
    p <- ggplot(df, aes(x = .data[[o1]], y = .data[[o2]])) +
      geom_point(aes(fill = .data[[o3]]), shape = 21, colour = "grey70",
                 size = 4, stroke = 0.8, alpha = 0.8) +
      scale_fill_viridis_c(name = l3, option = "plasma") +
      labs(x = l1, y = l2)
  }

  # White background, no grid lines, enlarged text.
  p +
    theme_minimal(base_size = 16) +
    theme(
      panel.grid       = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      axis.line        = element_line(colour = "black"),
      axis.ticks       = element_line(colour = "black")
    )
}

make_pareto_pairwise <- function(run, obj_names = run$obj_names, obj_labels = .obj_labels(obj_names)) {
  k <- length(obj_names)
  if (k < 2) { cat("Need >= 2 objectives.\n"); return(invisible(NULL)) }

  df <- run$df_obj %>%
    mutate(domination = factor(
      ifelse(is_nondominated == 1, "Non-dominated", "Dominated"),
      levels = c("Non-dominated", "Dominated")
    ))

  pairs <- combn(k, 2, simplify = FALSE)

  plots <- lapply(pairs, function(p) {
    xi <- obj_names[p[1]]; yi <- obj_names[p[2]]
    xl <- obj_labels[p[1]]; yl <- obj_labels[p[2]]
    ggplot(df, aes(x = .data[[xi]], y = .data[[yi]],
                   colour = domination, size = domination)) +
      geom_point() +
      scale_colour_manual(values = c("Non-dominated" = COL_ND, "Dominated" = COL_DOM),
                          name = NULL) +
      scale_size_manual(values = c("Non-dominated" = 2., "Dominated" = 1.5), name = NULL) +
      labs(x = xl, y = yl)
  })

  wrap_plots(plots, nrow = length(plots)/3, guides = "collect") +
    plot_annotation(theme = theme(plot.title = element_text(size = 13, face = "bold")))
}

make_corr_plot <- function(run, obj_names = run$obj_names, obj_labels = .obj_labels(obj_names), title = NULL, maximize = NULL) {
  # Determine direction: TRUE = higher is better (same heuristic as prep_parcoord_data).
  if (is.null(maximize)) {
    maximize <- grepl("anomaly|gain", obj_names, ignore.case = TRUE)
  }

  make_corr_tiles <- function(df, panel_label, n) {
    # Flip minimised objectives so that for all objectives higher = better.
    # This ensures that a positive correlation means genuine alignment (both
    # improve together) and a negative correlation means a true trade-off.
    df_dir <- df
    for (i in seq_along(obj_names)) {
      if (!maximize[i]) df_dir[[obj_names[i]]] <- -df_dir[[obj_names[i]]]
    }
    corr <- cor(df_dir[, obj_names], use = "complete.obs")
    expand.grid(x_idx = seq_along(obj_names), y_idx = seq_along(obj_names)) %>%
      filter(y_idx > x_idx) %>%
      mutate(
        x_label = factor(obj_labels[x_idx], levels = obj_labels),
        y_label = factor(obj_labels[y_idx], levels = obj_labels),
        r       = mapply(function(i, j) corr[j, i], x_idx, y_idx),
        r_text  = sprintf("%.2f", r),
        panel   = panel_label,
        n_label = n
      )
  }

  all_df <- run$df_obj
  nd_df  <- all_df[all_df$is_nondominated == 1, ]

  tiles <- bind_rows(
    make_corr_tiles(all_df, sprintf("All solutions (n = %d)",  nrow(all_df)), nrow(all_df)),
    make_corr_tiles(nd_df,  sprintf("Non-dominated (n = %d)", nrow(nd_df)),  nrow(nd_df))
  ) %>% mutate(panel = factor(panel, levels = unique(panel)))

  all_cells <- expand.grid(
    x_label = factor(obj_labels, levels = obj_labels),
    y_label = factor(obj_labels, levels = obj_labels),
    panel   = factor(unique(tiles$panel), levels = levels(tiles$panel))
  )

  ggplot() +
    geom_tile(data = all_cells, aes(x = x_label, y = y_label),
              fill = "grey90", colour = "white", linewidth = 0.5) +
    geom_tile(data = tiles, aes(x = x_label, y = y_label, fill = r),
              colour = "white", linewidth = 0.5) +
    geom_text(data = tiles, aes(x = x_label, y = y_label, label = r_text,
                                colour = abs(r) > 0.5),
              size = 8, fontface = "bold") +
    facet_wrap(~panel, ncol = 2) +
    scale_fill_distiller(palette = "RdBu", direction = -1, limits = c(-1, 1),
                         name = "Pearson r") +
    scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "black"), guide = "none") +
    scale_y_discrete(limits = rev(obj_labels)) +
    labs(x = NULL, y = NULL) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      panel.grid  = element_blank(),
      strip.text  = element_text(size = 18, face = "bold"), 
      axis.text = element_text(size = 18),
      legend.position = "none"
    )
}

prep_parcoord_data <- function(run, best_colours = c("#E05C2A", "#2A7BE0", "#2AB05C", "#7A2AB0"),
                               maximize = NULL,
                               obj_names = run$obj_names) {
  df <- run$df_obj
  obj_names  <- obj_names[obj_names %in% names(df)]
  obj_labels <- .obj_labels(obj_names)

  # If best_colours is a named vector, look up by objective name so colours
  # are consistent regardless of objective order across runs.
  # Fall back to positional rep_len for unnamed vectors.
  if (!is.null(names(best_colours))) {
    default_col <- "grey60"
    extended_colours <- vapply(obj_names, function(nm) {
      if (nm %in% names(best_colours)) best_colours[[nm]] else default_col
    }, character(1))
  } else {
    extended_colours <- rep_len(best_colours, length(obj_names))
  }

  # Determine direction: TRUE = higher is better.
  # Defaults to detecting "anomaly" or "gain" in the objective name.
  if (is.null(maximize)) {
    maximize <- grepl("anomaly|gain", obj_names, ignore.case = TRUE)
  }

  # Normalise to [0, 1] with 1 = better for all objectives.
  # Maximise: (x - min) / (max - min)
  # Minimise: (max - x) / (max - min)
  df_norm_local <- df %>% mutate(solution_id = paste(run$label, row_number(), sep = "__"))
  for (i in seq_along(obj_names)) {
    x   <- df_norm_local[[obj_names[i]]]
    rng <- max(x, na.rm = TRUE) - min(x, na.rm = TRUE)
    nc  <- paste0(obj_names[i], "_norm")
    df_norm_local[[nc]] <- if (maximize[i]) {
      (x - min(x, na.rm = TRUE)) / rng
    } else {
      (max(x, na.rm = TRUE) - x) / rng
    }
  }

  norm_cols <- paste0(obj_names, "_norm")
  nd_rows   <- df_norm_local[df_norm_local$is_nondominated == 1, ]
  # 1 = best on all axes, so use which.max throughout
  best_sols <- vapply(norm_cols, function(nc) {
    nd_rows$solution_id[which.max(nd_rows[[nc]])]
  }, character(1))

  sol_colour <- rep("grey85", nrow(df_norm_local))
  names(sol_colour) <- df_norm_local$solution_id
  sol_colour[nd_rows$solution_id] <- "black"
  for (i in seq_along(norm_cols)) sol_colour[best_sols[i]] <- extended_colours[i]

  df_norm_local %>%
    select(solution_id, is_nondominated, all_of(norm_cols)) %>%
    pivot_longer(cols = all_of(norm_cols),
                 names_to = "objective", values_to = "value_norm") %>%
    mutate(
      objective   = factor(gsub("_norm$", "", objective),
                           levels = obj_names, labels = obj_labels),
      line_colour = sol_colour[solution_id],
      run_label   = run$label
    )
}

make_parcoord_plot <- function(run,
                               best_colours = c("#E05C2A", "#2A7BE0", "#2AB05C", "#7A2AB0"),
                               maximize = NULL,
                               obj_names = run$obj_names) {
  df <- run$df_obj
  obj_names  <- obj_names[obj_names %in% names(df)]
  obj_labels <- .obj_labels(obj_names)

  extended_colours <- rep_len(best_colours, length(obj_names))
  legend_breaks    <- c(extended_colours[seq_along(obj_names)], "black", "grey85")
  legend_labels    <- c(sprintf("Best: %s", obj_labels[seq_along(obj_names)]),
                        "Non-dominated", "Dominated")

  df_pc       <- prep_parcoord_data(run, best_colours, maximize, obj_names)
  df_dom      <- df_pc[df_pc$line_colour == "grey85", ]
  df_nd_plain <- df_pc[df_pc$line_colour == "black",  ]
  df_nd_best  <- df_pc[!df_pc$line_colour %in% c("grey85", "black"), ]

  ggplot(mapping = aes(x = objective, y = value_norm,
                       group = solution_id, colour = line_colour)) +
    geom_line(data = df_dom,      aes(), alpha = 1, linewidth = 0.5) +
    geom_line(data = df_nd_plain, aes(), alpha = 1, linewidth = 0.5) +
    geom_line(data = df_nd_best,  aes(), alpha = 1, linewidth = 1.1) +
    scale_colour_identity(
      name   = NULL,
      guide  = guide_legend(override.aes = list(linewidth = 1.8, alpha = 1)),
      labels = setNames(legend_labels, legend_breaks),
      breaks = legend_breaks
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       breaks = c(0, 0.25, 0.5, 0.75, 1)) +
    labs(title = run$label, x = NULL, y = "Normalised objective value") +
    theme(axis.text.x = element_text(angle = 25, hjust = 1, size = 10),
          legend.position = "right")
}


# -- Spatial trade-offs (Jaccard) ----------------------------------------------

compute_jaccard_pairs <- function(run, obj_names = run$obj_names) {
  if (is.null(run$df_psel)) {
    message("pixel_selection.csv not available \u2013 skipping Jaccard.")
    return(NULL)
  }
  if (is.null(run$df_norm)) {
    message("objectives_normalized.csv not available \u2013 skipping Jaccard.")
    return(NULL)
  }

  psel    <- run$df_psel
  has_xy  <- all(c("x", "y") %in% names(psel))
  psel    <- psel %>%
    mutate(pkey = if (has_xy)
             paste(round(x, 1), round(y, 1), sep = "_")
           else
             paste(pixel_row, pixel_col, sep = "_"))

  sol_ids <- sort(unique(psel$solution_id))
  if (length(sol_ids) < 2) {
    message("Fewer than 2 non-dominated solutions \u2013 skipping Jaccard.")
    return(NULL)
  }

  sol_pixels  <- split(psel$pkey, psel$solution_id)
  pairs       <- combn(sol_ids, 2, simplify = FALSE)

  jaccard_df <- bind_rows(lapply(pairs, function(p) {
    s1    <- sol_pixels[[as.character(p[1])]]
    s2    <- sol_pixels[[as.character(p[2])]]
    inter <- length(intersect(s1, s2))
    uni   <- length(union(s1, s2))
    data.frame(solution_i = p[1], solution_j = p[2],
               jaccard_similarity = if (uni == 0) 0 else inter / uni)
  }))

  norm_cols <- paste0(obj_names, "_norm")
  if (!all(norm_cols %in% names(run$df_norm))) {
    df_for_dist <- run$df_obj[run$df_obj$is_nondominated == 1, obj_names, drop = FALSE]
    df_for_dist <- as.data.frame(scale(df_for_dist))
    row_ids     <- run$df_obj$solution_id[run$df_obj$is_nondominated == 1]
  } else {
    df_for_dist <- run$df_norm[run$df_norm$is_nondominated == 1, norm_cols, drop = FALSE]
    row_ids     <- run$df_norm$solution_id[run$df_norm$is_nondominated == 1]
  }

  dist_mat            <- as.matrix(dist(df_for_dist))
  rownames(dist_mat)  <- row_ids
  colnames(dist_mat)  <- row_ids

  jaccard_df$objective_distance <- mapply(
    function(i, j) {
      ri <- as.character(i); rj <- as.character(j)
      if (ri %in% rownames(dist_mat) && rj %in% colnames(dist_mat))
        dist_mat[ri, rj]
      else NA_real_
    },
    jaccard_df$solution_i, jaccard_df$solution_j
  )

  jaccard_df
}

make_jaccard_plot <- function(run, obj_names = run$obj_names, title = NULL) {
  jdf <- compute_jaccard_pairs(run, obj_names)
  if (is.null(jdf) || nrow(jdf) == 0) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "Jaccard data not available", size = 5) +
             theme_void())
  }

  jdf_clean  <- jdf %>% filter(!is.na(objective_distance))
  pearson_r  <- cor(jdf_clean$objective_distance, jdf_clean$jaccard_similarity, method = "pearson")
  spearman_r <- cor(jdf_clean$objective_distance, jdf_clean$jaccard_similarity, method = "spearman")

  ggplot(jdf_clean, aes(x = objective_distance, y = jaccard_similarity)) +
    geom_point(alpha = 0.35, size = 2, colour = "steelblue") +
    geom_smooth(method = "loess", formula = y ~ x,
                se = FALSE, colour = "#E05C2A", linewidth = 0.8) +
    scale_y_continuous(limits = c(0, 1),
                       labels = scales::percent_format(accuracy = 1)) +
    labs(
      subtitle = sprintf("%d pairs | Pearson r = %.3f | Spearman \u03c1 = %.3f",
                         nrow(jdf_clean), pearson_r, spearman_r),
      x = "Euclidean distance in normalised objective space",
      y = "Spatial similarity (Jaccard)"
    )
}


# -- Selection frequency maps --------------------------------------------------

# Internal helper: builds a small vertical histogram inset coloured by the same
# YlOrRd ramp as the map.  Returns a ggplot (consumed by patchwork::inset_element).
.rfop_inset_grob <- function(freq_df, binwidth = 5) {
  ylord_cols <- c("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C",
                  "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026")
  pal_fn <- colorRampPalette(ylord_cols)

  bin_df <- freq_df %>%
    mutate(rfop_bin = floor(rfop_pct / binwidth) * binwidth + binwidth / 2) %>%
    dplyr::count(rfop_bin, name = "n_pixels") %>%
    mutate(bar_fill = pal_fn(100)[pmax(1L, pmin(100L, as.integer(rfop_bin)))])

  ggplot(bin_df, aes(x = rfop_bin, y = n_pixels, fill = bar_fill)) +
    geom_col(width = binwidth * 0.85, colour = NA) +
    scale_fill_identity() +
    scale_x_continuous(
      limits = c(0, 105),
      breaks = c(0, 25, 50, 75, 100),
      labels = c("0", "25%", "50%", "75%", "100%"),
      expand = expansion(0)
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.12)),
      labels = scales::label_number(scale_cut = scales::cut_short_scale())
    ) +
    coord_flip() +
    labs(x = "RFOP", y = "") +
    theme_minimal(base_size = 6) +
    theme(
      plot.background  = element_rect(fill = "#f1f1f1", colour = "#ffffff",
                                      linewidth = 0.4),
      panel.background = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_blank(),
      axis.title       = element_text(size = 8),
      axis.text        = element_text(size = 8),
      axis.text.x      = element_blank(),
      plot.margin      = margin(3, 4, 3, 3)
    )
}

# Place RFOP inset in the top-right corner using patchwork::inset_element().
# Coordinates are normalised (0-1) relative to the panel, so they work
# regardless of the map's coordinate system (including coord_sf).
.add_rfop_inset <- function(p_map, freq_df,
                             left = 0.72, bottom = 0.58,
                             right = 0.99, top = 0.99) {
  p_map + patchwork::inset_element(
    .rfop_inset_grob(freq_df),
    left = left, bottom = bottom, right = right, top = top,
    align_to = "panel"
  )
}

make_sel_freq_plot <- function(run) {
  if (is.null(run$df_psel)) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "pixel_selection.csv not found", size = 5) +
             theme_void())
  }

  freq_df <- run$df_psel %>%
    count(x, y, name = "n_selected") %>%
    mutate(rfop_pct = n_selected / run$n_nondom * 100)

  elig_unsel <- if (!is.null(run$df_elig)) {
    anti_join(run$df_elig, freq_df, by = c("x", "y"))
  } else NULL

  p_map <- ggplot() +
    theme_void() +
    theme(panel.background = element_rect(fill = "white", colour = NA),
          legend.position  = "none")

  if (!is.null(elig_unsel) && nrow(elig_unsel) > 0)
    p_map <- p_map + geom_raster(data = elig_unsel, aes(x = x, y = y), fill = "grey80")

  p_map <- p_map +
    geom_raster(data = freq_df, aes(x = x, y = y, fill = rfop_pct)) +
    scale_fill_distiller(palette = "YlOrRd", direction = 1, limits = c(0, 100)) +
    geom_sf(data = BE, fill = NA, color = "black", inherit.aes = FALSE)

  .add_rfop_inset(p_map, freq_df)
}

# iEMSs conference version of the selection-frequency map.
# Differences from make_sel_freq_plot:
#   - aggregates to cell_m-metre cells (default 1 km, matching R3d plots)
#   - discrete 4-bin colour scheme (0-25 / 25-50 / 50-75 / 75-100 %)
#   - grey "Never selected" included in legend
#   - Switzerland inset in top-right (requires CH sf object in calling env)
#   - legend at bottom; no title
make_iemss_sel_freq_plot <- function(run, cell_m = 1000) {
  if (is.null(run$df_psel)) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "pixel_selection.csv not found", size = 5) +
             theme_void())
  }

  # Per-pixel RFOP at native resolution, then average into cell_m cells.
  freq_native <- run$df_psel |>
    dplyr::count(x, y, name = "n_selected") |>
    dplyr::mutate(rfop_pct = n_selected / run$n_nondom * 100)

  freq_agg <- freq_native |>
    dplyr::mutate(
      x = cell_m * (x %/% cell_m) + cell_m / 2,
      y = cell_m * (y %/% cell_m) + cell_m / 2
    ) |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(rfop_pct = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

  # Eligible pixels aggregated to same grid (for the "never selected" underlay).
  elig_agg <- if (!is.null(run$df_elig) && nrow(run$df_elig) > 0) {
    run$df_elig |>
      dplyr::mutate(
        x = cell_m * (x %/% cell_m) + cell_m / 2,
        y = cell_m * (y %/% cell_m) + cell_m / 2
      ) |>
      dplyr::distinct(x, y)
  } else NULL

  # Discrete bins and colours (YlOrRd-inspired, 4 levels + grey for unselected).
  bin_labels <- c("0-25%", "25-50%", "50-75%", "75-100%")
  all_levels <- c(bin_labels, "Never selected")
  bin_cols <- c(
    "0-25%"          = "#FED976",
    "25-50%"         = "#FD8D3C",
    "50-75%"         = "#E31A1C",
    "75-100%"        = "#800026",
    "Never selected" = "grey80"
  )

  freq_df <- freq_agg |>
    dplyr::mutate(
      rfop_bin = cut(rfop_pct,
                     breaks = c(0, 25, 50, 75, 100),
                     labels = bin_labels,
                     include.lowest = TRUE, right = TRUE),
      rfop_bin = factor(as.character(rfop_bin), levels = all_levels)
    )

  elig_unsel_df <- if (!is.null(elig_agg)) {
    dplyr::anti_join(elig_agg, freq_df, by = c("x", "y")) |>
      dplyr::mutate(rfop_bin = factor("Never selected", levels = all_levels))
  } else NULL

  p_map <- ggplot() +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position  = "right",
      legend.direction = "vertical",
      legend.text      = element_text(size = 10),
      legend.title     = element_text(size = 11, face = "bold")
    )

  if (!is.null(elig_unsel_df) && nrow(elig_unsel_df) > 0)
    p_map <- p_map +
      geom_raster(data = elig_unsel_df, aes(x = x, y = y, fill = rfop_bin))

  p_map <- p_map +
    geom_raster(data = freq_df, aes(x = x, y = y, fill = rfop_bin)) +
    scale_fill_manual(
      values = bin_cols,
      name   = "Selection frequency",
      drop   = FALSE,
      guide  = guide_legend(title.position = "top", ncol = 1)
    )

  if (exists("BE"))
    p_map <- p_map +
      geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4,
              inherit.aes = FALSE)
  p_map
}

make_sel_freq_plot_agg_hex <- function(run, bins = 30) {
  if (is.null(run$df_psel)) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "pixel_selection.csv not found", size = 5) +
             theme_void())
  }

  freq_df <- run$df_psel %>%
    count(x, y, name = "n_selected") %>%
    mutate(rfop_pct = n_selected / run$n_nondom * 100)

  # Convert BE to plain coordinates to avoid coord_sf clipping stat_summary_hex
  be_outline <- BE %>%
    st_transform(crs = st_crs(2056)) %>%
    st_coordinates() %>%
    as.data.frame()

  p_map <- ggplot() +
    theme_void() +
    theme(panel.background = element_rect(fill = "white", colour = NA),
          legend.position  = "none")

  if (!is.null(run$df_elig)) {
    p_map <- p_map + stat_summary_hex(
      data = run$df_elig, aes(x = x, y = y, z = 1),
      fun = function(x) 1, bins = bins, fill = "grey80",
      colour = "white", size = 0.1
    )
  }

  p_map <- p_map +
    stat_summary_hex(data = freq_df, aes(x = x, y = y, z = rfop_pct),
                     fun = mean, bins = bins, colour = "white", size = 0.1) +
    scale_fill_distiller(palette = "YlOrRd", direction = 1, limits = c(0, 100)) +
    geom_path(data = be_outline,
              aes(x = X, y = Y, group = interaction(L1, L2)),
              color = "black", linewidth = 0.5, inherit.aes = FALSE)

  # Inset histogram uses pixel-level RFOP (not hex-aggregated means)
  elig_unsel <- if (!is.null(run$df_elig)) {
    anti_join(run$df_elig, freq_df, by = c("x", "y"))
  } else NULL

  .add_rfop_inset(p_map, freq_df)
}


# -- Selection frequency histograms --------------------------------------------

# Stacked RFOP histogram coloured by an arbitrary grouping variable.
# `fill_values`:      factor of length == nrow(count(df_psel, x, y)).
# `elig_fill_values`: optional factor for ALL eligible pixels (from elig_fill_lulc /
#                     elig_fill_objective).  When supplied the y-axis shows the
#                     proportion of each class's eligible pixels that fall in each
#                     RFOP bin rather than the raw pixel count.
# `palette`:          RColorBrewer name OR named colour vector.
make_sel_freq_unified_plot <- function(run, fill_values, fill_label = "Class",
                                       elig_fill_values = NULL,
                                       palette = "Set3", binwidth = 10,
                                       position = "stack",
                                       title = NULL,
                                       subtitle = NULL,
                                       top_n = NULL) {
  if (is.null(run$df_psel)) return(NULL)

  freq_df <- run$df_psel %>%
    count(x, y, name = "n_selected") %>%
    mutate(rfop_pct = n_selected / run$n_nondom * 100)

  stopifnot(length(fill_values) == nrow(freq_df))
  freq_df$fill_group <- as.factor(fill_values)

  # Collapse all but the top_n most frequent classes into "Other"
  if (!is.null(top_n)) {
    top_classes <- freq_df %>%
      count(fill_group, wt = n_selected, name = "total") %>%
      slice_max(total, n = top_n) %>%
      pull(fill_group) %>%
      as.character()
    freq_df <- freq_df %>%
      mutate(fill_group = factor(
        ifelse(as.character(fill_group) %in% top_classes,
               as.character(fill_group), "Other"),
        levels = c(top_classes, "Other")
      ))
    if (!is.null(elig_fill_values)) {
      elig_fill_values <- ifelse(
        as.character(elig_fill_values) %in% top_classes,
        as.character(elig_fill_values), "Other"
      )
    }
  }

  # When eligible-pixel class counts are available, weight each selected pixel
  # by 1 / (total eligible pixels in that class).  The histogram then sums
  # weights per bin, giving the proportion of each class's eligible area
  # that appears at each RFOP level.
  if (!is.null(elig_fill_values)) {
    class_totals <- table(as.factor(elig_fill_values))
    freq_df <- freq_df %>%
      mutate(weight = 1 / as.numeric(class_totals[as.character(fill_group)]))
    y_label <- "Proportion of eligible class pixels"
    y_scale <- scale_y_continuous(labels = scales::percent_format(accuracy = 0.1))
  } else {
    freq_df$weight <- 1
    y_label <- if (position == "fill") "Proportion" else "Pixel Count"
    y_scale <- scale_y_continuous()
  }

  p <- ggplot(freq_df, aes(x = rfop_pct, fill = fill_group, weight = weight)) +
    geom_histogram(binwidth = binwidth, colour = "black", linewidth = 0.1,
                   position = position) +
    scale_x_continuous(limits = c(0, 100)) +
    y_scale +
    labs(title = title, subtitle = subtitle,
         x = "Selection Frequency (RFOP %)",
         y = y_label) +
    theme_minimal() +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())

  if (is.character(palette) && length(palette) == 1)
    p + scale_fill_brewer(palette = palette, name = fill_label)
  else
    p + scale_fill_manual(values = palette, name = fill_label)
}

# Reclassify a factor of raw LULC codes into named habitat classes.
# Codes not in the map are set to NA.  Useful before passing to
# make_sel_freq_unified_plot or elig_fill_lulc.
# Usage: rfop_fill_lulc(run, lulc) |> reclass_lulc()
reclass_lulc <- function(lulc_codes, reclass_map = NULL) {
  if (is.null(reclass_map)) {
    reclass_map <- c(
      "12" = "Forest",
      "13" = "Forest",
      "15" = "Agricultural land",
      "16" = "Pastures & grasslands",
      "17" = "Pastures & grasslands"
    )
  }
  codes_chr <- as.character(lulc_codes)
  labels    <- reclass_map[codes_chr]
  labels[is.na(labels)] <- NA_character_
  lvls <- unique(reclass_map)                 # stable order from the map
  factor(labels, levels = lvls)
}

rfop_fill_lulc <- function(run, lulc_raster, col_idx = 2, pts_crs = NULL) {
  freq_df <- run$df_psel %>% count(x, y)
  coords  <- freq_df[, c("x", "y")]
  if (!is.null(pts_crs)) {
    pts  <- terra::vect(coords, geom = c("x", "y"), crs = pts_crs)
    pts  <- terra::project(pts, terra::crs(lulc_raster))
    vals <- terra::extract(lulc_raster, pts)
  } else {
    vals <- terra::extract(lulc_raster, coords)
  }
  as.factor(vals[, col_idx])
}

# Quartile breaks are defined over the eligible pixel population (run$df_elig)
# when available, so levels are consistent with elig_fill_objective().
rfop_fill_objective <- function(run, obj_raster, n_breaks = 4,
                                labels = paste0("Q", seq_len(n_breaks)),
                                col_idx = 1) {
  ref_coords <- if (!is.null(run$df_elig)) {
    run$df_elig[, c("x", "y")]
  } else {
    run$df_psel %>% count(x, y) %>% select(x, y)
  }
  ref_vals <- terra::extract(obj_raster, ref_coords)[, col_idx + 1]
  brk      <- quantile(ref_vals, probs = seq(0, 1, length.out = n_breaks + 1), na.rm = TRUE)
  brk[1]           <- -Inf
  brk[n_breaks + 1] <- Inf
  brk <- unique(brk)
  labels <- paste0("Q", seq_len(length(brk) - 1))

  sel_coords <- run$df_psel %>% count(x, y) %>% select(x, y)
  vals       <- terra::extract(obj_raster, sel_coords)[, col_idx + 1]
  as.factor(cut(vals, breaks = brk, labels = labels, include.lowest = TRUE))
}

# -- Eligible-pixel fill helpers (denominators for make_sel_freq_unified_plot) -

elig_fill_lulc <- function(run, lulc_raster, col_idx = 2, pts_crs = NULL) {
  if (is.null(run$df_elig)) return(NULL)
  coords <- run$df_elig[, c("x", "y")]
  if (!is.null(pts_crs)) {
    pts  <- terra::vect(coords, geom = c("x", "y"), crs = pts_crs)
    pts  <- terra::project(pts, terra::crs(lulc_raster))
    vals <- terra::extract(lulc_raster, pts)
  } else {
    vals <- terra::extract(lulc_raster, coords)
  }
  as.factor(vals[, col_idx])
}

# Quartile breaks derived from the eligible population (same breaks as
# rfop_fill_objective so factor levels match across the two functions).
elig_fill_objective <- function(run, obj_raster, n_breaks = 4,
                                labels = paste0("Q", seq_len(n_breaks)),
                                col_idx = 1) {
  if (is.null(run$df_elig)) return(NULL)
  vals <- terra::extract(obj_raster, run$df_elig[, c("x", "y")])[, col_idx + 1]
  brk  <- quantile(vals, probs = seq(0, 1, length.out = n_breaks + 1), na.rm = TRUE)
  brk[1]           <- -Inf
  brk[n_breaks + 1] <- Inf
  brk <- unique(brk)
  labels <- paste0("Q", seq_len(length(brk) - 1))
  as.factor(cut(vals, breaks = brk, labels = labels, include.lowest = TRUE))
}


# -- Action-type helpers (restore vs convert) ----------------------------------

# Stacked-bar chart: one bar per non-dominated solution, stacked by action_type.
# Solutions are ordered by their value on `order_obj` (ascending = best first).
# Requires df_psel to have an action_type column ("restore" / "convert").
make_action_stacked_bar <- function(run, obj_names, obj_labels,
                                    order_obj = NULL,
                                    colours = c(restore = "#4DAF4A", convert = "#984EA3")) {
  if (is.null(run$df_psel)) {
    message("pixel_selection.csv not available - skipping action stacked bar.")
    return(invisible(NULL))
  }
  if (!"action_type" %in% names(run$df_psel)) {
    message("action_type column not found in pixel_selection.csv - re-run export_to_r.py.")
    return(invisible(NULL))
  }

  # Count actions per solution x action_type (non-dominated solutions only)
  nd_ids <- run$df_obj$solution_id[run$df_obj$is_nondominated == 1]
  counts <- run$df_psel %>%
    filter(solution_id %in% nd_ids) %>%
    count(solution_id, action_type, name = "n_pixels")

  # Order solutions by chosen objective (default: first in obj_names)
  if (is.null(order_obj)) order_obj <- obj_names[1]
  order_label <- obj_labels[match(order_obj, obj_names)]

  obj_order <- run$df_obj %>%
    filter(is_nondominated == 1) %>%
    arrange(.data[[order_obj]]) %>%
    mutate(sol_rank = row_number())

  counts <- counts %>%
    left_join(obj_order %>% select(solution_id, sol_rank), by = "solution_id") %>%
    mutate(action_type = factor(action_type, levels = c("restore", "convert")))

  ggplot(counts, aes(x = sol_rank, y = n_pixels, fill = action_type)) +
    geom_col(width = 1, colour = NA) +
    scale_fill_manual(values = colours, name = "Action type",
                      labels = c(restore = "Restore", convert = "Convert")) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(
      title    = sprintf("Action mix per non-dominated solution - %s", run$label),
      subtitle = sprintf("Solutions ordered by %s (ascending)", order_label),
      x = sprintf("Solution rank (by %s, best to worst)", order_label),
      y = "Number of pixels"
    ) +
    theme(legend.position = "top")
}

# Side-by-side spatial frequency maps split by action_type.
# Returns a patchwork of two RFOP% maps (restore | convert).
make_action_split_freq_plot <- function(run,
                                        colours_restore = "YlGn",
                                        colours_convert = "PuBuGn") {
  if (is.null(run$df_psel)) {
    message("pixel_selection.csv not available.")
    return(invisible(NULL))
  }
  if (!"action_type" %in% names(run$df_psel)) {
    message("action_type column not found in pixel_selection.csv - re-run export_to_r.py.")
    return(invisible(NULL))
  }
  if (!all(c("x", "y") %in% names(run$df_psel))) {
    message("x/y coordinates not found in pixel_selection.csv.")
    return(invisible(NULL))
  }

  nd_ids  <- run$df_obj$solution_id[run$df_obj$is_nondominated == 1]
  n_nd    <- length(nd_ids)
  psel_nd <- run$df_psel %>% filter(solution_id %in% nd_ids)

  make_one_map <- function(action, palette, title_suffix) {
    freq_df <- psel_nd %>%
      filter(action_type == action) %>%
      count(x, y, name = "n_selected") %>%
      mutate(rfop_pct = n_selected / n_nd * 100)

    if (nrow(freq_df) == 0) {
      return(ggplot() +
               annotate("text", x = 0.5, y = 0.5,
                        label = paste("No", action, "actions"), size = 5) +
               theme_void())
    }

    p <- ggplot() +
      theme_void() +
      theme(panel.background = element_rect(fill = "white", colour = NA),
            legend.position  = "none")

    # Eligible-but-unselected underlay: only meaningful for restore, because
    # df_elig contains restoration-eligible pixels; conversion-eligible pixels
    # occupy a different part of the landscape and would obscure the convert map.
    if (action == "restore" && !is.null(run$df_elig)) {
      elig_unsel <- anti_join(run$df_elig, freq_df, by = c("x", "y"))
      if (nrow(elig_unsel) > 0)
        p <- p + geom_raster(data = elig_unsel, aes(x = x, y = y), fill = "grey80")
    }

    p <- p +
      geom_raster(data = freq_df, aes(x = x, y = y, fill = rfop_pct)) +
      scale_fill_distiller(palette = palette, direction = 1,
                           limits = c(0, 100)) +
      geom_sf(data = BE, fill = NA, colour = "black", inherit.aes = FALSE)

    .add_rfop_inset(p, freq_df)
  }

  p_restore <- make_one_map("restore", colours_restore, "Restoration frequency")
  p_convert <- make_one_map("convert", colours_convert, "Conversion frequency")

  p_restore + p_convert +
    patchwork::plot_annotation(
      theme = theme(plot.title = element_text(size = 13, face = "bold"))
    )
}


# -- Comparison functions ------------------------------------------------------

compare_hv_plot <- function(run_a, run_b) {
  if (is.null(run_a$df_hv) && is.null(run_b$df_hv)) {
    cat("hypervolume_evolution.csv not available for either run.\n")
    return(invisible(NULL))
  }

  hv_df <- bind_rows(
    if (!is.null(run_a$df_hv)) mutate(run_a$df_hv, run = run_a$label) else NULL,
    if (!is.null(run_b$df_hv)) mutate(run_b$df_hv, run = run_b$label) else NULL
  ) %>% mutate(run = factor(run, levels = c(run_a$label, run_b$label)))

  run_cols <- setNames(c("#E05C2A", "steelblue"), c(run_a$label, run_b$label))
  hv_a     <- if (!is.null(run_a$df_hv)) tail(run_a$df_hv$hypervolume, 1) else NA_real_
  hv_b     <- if (!is.null(run_b$df_hv)) tail(run_b$df_hv$hypervolume, 1) else NA_real_
  ratio    <- if (!is.na(hv_a) && hv_a != 0) hv_b / hv_a else NA_real_

  ggplot(hv_df, aes(x = generation, y = hypervolume, colour = run)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.2, alpha = 0.6) +
    scale_colour_manual(values = run_cols, name = "Run") +
    scale_y_continuous(labels = scales::scientific) +
    labs(
      subtitle = sprintf("Final HV: %s = %.4g | %s = %.4g | ratio B/A = %.3f",
                         run_a$label, hv_a, run_b$label, hv_b,
                         if (!is.na(ratio)) ratio else NaN),
      x = "Generation", y = "Hypervolume"
    )
}

compare_pareto_overlay <- function(run_a, run_b, obj_names, obj_labels) {
  k <- length(obj_names)
  if (k < 2) { cat("Need >= 2 objectives.\n"); return(invisible(NULL)) }

  nd_a     <- run_a$df_obj[run_a$df_obj$is_nondominated == 1, ] %>% mutate(run = run_a$label)
  nd_b     <- run_b$df_obj[run_b$df_obj$is_nondominated == 1, ] %>% mutate(run = run_b$label)
  df       <- bind_rows(nd_a, nd_b) %>%
    mutate(run = factor(run, levels = c(run_a$label, run_b$label)))
  run_cols <- setNames(c("#E05C2A", "steelblue"), c(run_a$label, run_b$label))
  pairs    <- combn(k, 2, simplify = FALSE)

  plots <- lapply(pairs, function(p) {
    xi <- obj_names[p[1]]; yi <- obj_names[p[2]]
    ggplot(df, aes(x = .data[[xi]], y = .data[[yi]], colour = run)) +
      geom_point(size = 2.5, alpha = 0.7) +
      scale_colour_manual(values = run_cols, name = "Run") +
      labs(x = obj_labels[p[1]], y = obj_labels[p[2]])
  })

  wrap_plots(plots, ncol = length(plots), guides = "collect") +
    plot_annotation(
      title    = sprintf("Pareto Front Overlay \u2014 %s", toupper(ECOSYSTEM)),
      subtitle = sprintf("%d ND (%s)  +  %d ND (%s)",
                         run_a$n_nondom, run_a$label, run_b$n_nondom, run_b$label),
      theme = theme(plot.title    = element_text(size = 13, face = "bold"),
                    plot.subtitle = element_text(size = 11, colour = "grey40"))
    )
}

# Overlay non-dominated Pareto fronts from an arbitrary set of scenarios in
# objective space - the N-scenario generalisation of compare_pareto_overlay().
# Built for the R2 backup slide: it disaggregates the scenarios whose fronts feed
# the shared-reference hypervolume so the actual trade-off surfaces behind the HV
# summary are visible, one colour per scenario. Every pairwise 2D projection of
# the objectives is drawn (k objectives -> choose(k, 2) panels).
#
#   nd_df    : tidy data frame of NON-DOMINATED solutions, one row per solution,
#              carrying every objective column in `obj_names` plus a `scenario`
#              column (the grouping each front belongs to). Seeds of the same
#              scenario are simply bound together (the overlay then shows the
#              seed-pooled front for that scenario).
#   obj_names: objective columns to plot (>= 2). Only columns present in nd_df
#              are used.
#   normalize: when TRUE (default) each objective is min-max scaled to [0, 1]
#              before plotting, so scenarios on very different raw scales share a
#              common axis (front SHAPE, not absolute units; axis labels gain
#              "(norm.)").
#   norm_within: OPTIONAL column name. When supplied, the min-max is computed
#              WITHIN each level of that column rather than over the pooled nd_df.
#              This is required when an objective's raw magnitude differs across
#              groups for reasons of units rather than performance - e.g. the
#              `scaling` factor rescales restoration_potential ~100x (global ~1e3
#              vs upper_q75 ~1e5). Pooled normalisation does NOT fix that (min-max
#              is affine, so the small-magnitude group still collapses into a
#              stripe near 0); normalising within the scaling group spreads each
#              group across [0, 1] so front shape is visible for all. NB: a value
#              then means "fraction of that group's own achievable range", so
#              points from different groups at the same axis position are not
#              directly comparable in absolute terms.
make_pareto_overlay_scenarios <- function(nd_df, obj_names,
                                          obj_labels = .obj_labels(obj_names),
                                          title = NULL, subtitle = NULL,
                                          point_size = 1.8, point_alpha = 0.7,
                                          normalize = TRUE, norm_within = NULL) {
  if (is.null(nd_df) || nrow(nd_df) == 0L || !"scenario" %in% names(nd_df)) {
    message("make_pareto_overlay_scenarios(): empty nd_df or missing 'scenario' column.")
    return(NULL)
  }
  present    <- obj_names[obj_names %in% names(nd_df)]
  if (length(present) < 2) {
    message("make_pareto_overlay_scenarios(): need >= 2 objectives present in nd_df.")
    return(NULL)
  }
  obj_labels <- .obj_labels(present)
  obj_names  <- present

  # Min-max each objective to [0, 1]. When norm_within names a grouping column the
  # scaling is done within each group, so a group whose raw values are 100x smaller
  # still spans the full axis instead of collapsing to a stripe.
  if (normalize) {
    .minmax <- function(x) {
      rng <- max(x, na.rm = TRUE) - min(x, na.rm = TRUE)
      if (is.finite(rng) && rng > 0) (x - min(x, na.rm = TRUE)) / rng else rep(0, length(x))
    }
    use_within <- !is.null(norm_within) && norm_within %in% names(nd_df)
    for (o in obj_names) {
      nd_df[[o]] <- if (use_within) {
        stats::ave(nd_df[[o]], nd_df[[norm_within]], FUN = .minmax)
      } else {
        .minmax(nd_df[[o]])
      }
    }
    obj_labels <- paste0(obj_labels, " (norm.)")
  }

  nd_df$scenario <- as.factor(nd_df$scenario)
  n_scen <- nlevels(nd_df$scenario)
  pal <- if (n_scen <= 8 && requireNamespace("RColorBrewer", quietly = TRUE)) {
    RColorBrewer::brewer.pal(max(3L, n_scen), "Dark2")[seq_len(n_scen)]
  } else {
    scales::hue_pal()(n_scen)
  }

  pairs <- utils::combn(length(obj_names), 2, simplify = FALSE)
  plots <- lapply(pairs, function(p) {
    xi <- obj_names[p[1]]; yi <- obj_names[p[2]]
    ggplot(nd_df, aes(x = .data[[xi]], y = .data[[yi]], colour = scenario)) +
      geom_point(size = point_size, alpha = point_alpha) +
      scale_colour_manual(values = pal, name = "Scenario") +
      labs(x = obj_labels[p[1]], y = obj_labels[p[2]]) +
      theme(legend.position = "right")
  })

  if (requireNamespace("patchwork", quietly = TRUE)) {
    patchwork::wrap_plots(plots, ncol = length(plots), guides = "collect") +
      patchwork::plot_annotation(
        title = title, subtitle = subtitle,
        theme = theme(plot.title    = element_text(size = 13, face = "bold"),
                      plot.subtitle = element_text(size = 11, colour = "grey40")))
  } else {
    # No patchwork: fall back to the single most informative projection.
    plots[[1]] + labs(title = title, subtitle = subtitle)
  }
}

# C-metric and Generational Distance between two runs.
coverage_stats <- function(run_a, run_b, obj_names) {
  nd_a <- as.matrix(run_a$df_obj[run_a$df_obj$is_nondominated == 1, obj_names])
  nd_b <- as.matrix(run_b$df_obj[run_b$df_obj$is_nondominated == 1, obj_names])

  dom_by_b <- mean(apply(nd_a, 1, function(a)
    any(apply(nd_b, 1, function(b) all(b <= a) && any(b < a)))))
  dom_by_a <- mean(apply(nd_b, 1, function(b)
    any(apply(nd_a, 1, function(a) all(a <= b) && any(a < b)))))

  hv_a <- if (!is.null(run_a$df_hv)) tail(run_a$df_hv$hypervolume, 1) else NA_real_
  hv_b <- if (!is.null(run_b$df_hv)) tail(run_b$df_hv$hypervolume, 1) else NA_real_

  # GD: mean distance from each run's ND front to the combined best-known front
  combined <- rbind(nd_a, nd_b)
  is_nd_combined <- !apply(combined, 1, function(p)
    any(apply(combined, 1, function(q) all(q <= p) && any(q < p))))
  ref <- combined[is_nd_combined, , drop = FALSE]

  gd <- function(approx, reference) {
    mean(apply(approx, 1, function(p) sqrt(min(colSums((t(reference) - p)^2)))))
  }

  tibble::tibble(
    Metric = c(
      sprintf("C(B\u2192A): %% of %s solutions dominated by %s", run_a$label, run_b$label),
      sprintf("C(A\u2192B): %% of %s solutions dominated by %s", run_b$label, run_a$label),
      sprintf("GD (%s): mean dist to combined reference front", run_a$label),
      sprintf("GD (%s): mean dist to combined reference front", run_b$label)
    ),
    Value = c(
      sprintf("%.1f%%", dom_by_b * 100),
      sprintf("%.1f%%", dom_by_a * 100),
      sprintf("%.4g", gd(nd_a, ref)),
      sprintf("%.4g", gd(nd_b, ref))
    )
  )
}

compare_pairwise_scatter <- function(run_a, run_b, obj_names, obj_labels) {
  k <- length(obj_names)
  if (k < 2) { cat("Need at least 2 objectives.\n"); return(invisible(NULL)) }

  df <- bind_rows(
    run_a$df_obj %>% select(all_of(obj_names)) %>% mutate(run = run_a$label),
    run_b$df_obj %>% select(all_of(obj_names)) %>% mutate(run = run_b$label)
  ) %>% mutate(run = factor(run, levels = c(run_a$label, run_b$label)))

  run_cols <- setNames(c("#E05C2A", "steelblue"), c(run_a$label, run_b$label))
  pairs    <- combn(k, 2, simplify = FALSE)

  plots <- lapply(seq_along(pairs), function(pi) {
    p  <- pairs[[pi]]
    ggplot(df, aes(x = .data[[obj_names[p[1]]]], y = .data[[obj_names[p[2]]]],
                   colour = run)) +
      geom_point(size = 1.5, alpha = 0.35) +
      scale_colour_manual(values = run_cols, name = "Run") +
      labs(x = obj_labels[p[1]], y = obj_labels[p[2]]) +
      theme(legend.position = if (pi == 1) "top" else "none",
            axis.title = element_text(size = 9))
  })

  do.call(gridExtra::grid.arrange,
          c(plots, list(ncol = min(3L, length(plots)),
                        top  = grid::textGrob(
                          sprintf("Pairwise Objective Scatter \u2014 %s", toupper(ECOSYSTEM)),
                          gp = grid::gpar(fontsize = 13, fontface = "bold")))))
}

compute_cross_jaccard <- function(run_a, run_b) {
  if (is.null(run_a$df_psel) || is.null(run_b$df_psel)) {
    message("pixel_selection.csv not available for one or both runs.")
    return(NULL)
  }

  make_pixel_sets <- function(run) {
    psel   <- run$df_psel
    has_xy <- all(c("x", "y") %in% names(psel))
    psel   <- psel %>% mutate(
      pkey = if (has_xy) paste(round(x, 1), round(y, 1), sep = "_")
             else paste(pixel_row, pixel_col, sep = "_")
    )
    split(psel$pkey, psel$solution_id)
  }

  sets_a <- make_pixel_sets(run_a)
  sets_b <- make_pixel_sets(run_b)
  if (length(sets_a) == 0 || length(sets_b) == 0) return(NULL)

  bind_rows(unlist(lapply(names(sets_a), function(i)
    lapply(names(sets_b), function(j) {
      s1    <- sets_a[[i]]; s2 <- sets_b[[j]]
      inter <- length(intersect(s1, s2))
      uni   <- length(union(s1, s2))
      data.frame(jaccard = if (uni == 0) 0 else inter / uni,
                 type    = sprintf("Cross (%s \u00d7 %s)", run_a$label, run_b$label))
    })), recursive = FALSE))
}

plot_jaccard_comparison <- function(run_a, run_b, obj_names) {
  jac_a <- compute_jaccard_pairs(run_a, obj_names)
  jac_b <- compute_jaccard_pairs(run_b, obj_names)
  jac_x <- compute_cross_jaccard(run_a, run_b)

  parts <- Filter(Negate(is.null), list(
    if (!is.null(jac_a) && nrow(jac_a) > 0)
      data.frame(jaccard = jac_a$jaccard_similarity,
                 type    = sprintf("Within %s", run_a$label)),
    if (!is.null(jac_b) && nrow(jac_b) > 0)
      data.frame(jaccard = jac_b$jaccard_similarity,
                 type    = sprintf("Within %s", run_b$label)),
    if (!is.null(jac_x) && nrow(jac_x) > 0) jac_x
  ))

  if (length(parts) == 0) { cat("Jaccard data not available.\n"); return(invisible(NULL)) }

  df_all <- bind_rows(parts) %>% mutate(type = factor(type, levels = unique(type)))

  ggplot(df_all, aes(x = type, y = jaccard, fill = type)) +
    geom_violin(trim = FALSE, alpha = 0.45, colour = NA) +
    geom_boxplot(width = 0.15, outlier.size = 0.8, fill = "white") +
    scale_fill_manual(values = c("#E05C2A", "steelblue", "grey50"), guide = "none") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
      title    = sprintf("Spatial Similarity (Jaccard) \u2014 %s", toupper(ECOSYSTEM)),
      subtitle = "Pairwise Jaccard within each run's Pareto front and across runs",
      x = NULL, y = "Jaccard similarity"
    )
}


# -- Spatial priority classification ------------------------------------------
#
# These functions support the multi-run spatial classification analysis.
# The workflow is:
#   1. load_run_rfop()              - load one run's per-pixel RFOP + metadata
#   2. bind_rows() over many runs   - one row per pixel per run
#   3. compute_spatial_classification() - classify each eligible pixel
#   4. make_classification_map()    - categorical raster map
#   5. make_classification_summary_bar() - % area per class
#
# Classification dimensions:
#   dim_type = "indicator"  - condition/indicator assumption runs
#   dim_type = "policy"     - policy decision runs (e.g. burden sharing)
#   dim_type = "seed"       - stochastic replicate runs
#   dim_type = "param"      - continuous parameter sweep runs (optional)


# Load a single run's pixel-level RFOP and attach sensitivity-dimension metadata.
# Returns a data frame with columns: x, y, rfop_pct, run_label, dim_type, group_label.
# rfop_pct is computed relative to that run's own Pareto front size, so runs with
# different numbers of non-dominated solutions are directly comparable.
#
# -- Canonical run loaders (consolidated here so any driver can `source()` this
#    file and load exports without redefining loaders inline) ------------------

# Read a CSV only if it exists, else return NULL.
read_csv_if_exists <- function(path) {
  if (file.exists(path)) utils::read.csv(path) else NULL
}

# Fast reader for the (x, y) columns of large pixel_selection.csv files.
# Falls back gracefully: data.table::fread -> readr -> base read.csv.
.read_xy_fast <- function(path) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    as.data.frame(data.table::fread(path, select = c("x", "y"), showProgress = FALSE))
  } else if (requireNamespace("readr", quietly = TRUE)) {
    as.data.frame(readr::read_csv(path, col_select = c("x", "y"),
                                  progress = FALSE, show_col_types = FALSE))
  } else {
    utils::read.csv(path)[, c("x", "y")]
  }
}

# Non-dominated solution ids from an objectives data frame.
get_nd_ids <- function(objectives) {
  if ("is_nondominated" %in% names(objectives))
    objectives$solution_id[objectives$is_nondominated == 1L]
  else
    objectives$solution_id
}

# Full run loader: objectives, normalized objectives, HV evolution, population
# stats, pixel selection and eligible pixels, plus metadata. Maximised objectives
# (anomaly / connectivity_gain / restoration_potential / landscape_context) are
# sign-flipped so larger = better for plotting.
load_run_data <- function(dir_path, label) {
  dir_path <- normalizePath(dir_path, mustWork = TRUE)
  meta     <- jsonlite::read_json(file.path(dir_path, "metadata.json"))
  df_obj   <- utils::read.csv(file.path(dir_path, "objectives.csv"))
  df_norm  <- read_csv_if_exists(file.path(dir_path, "objectives_normalized.csv"))
  df_hv    <- read_csv_if_exists(file.path(dir_path, "hypervolume_evolution.csv"))
  df_pop   <- read_csv_if_exists(file.path(dir_path, "population_stats.csv"))
  df_psel  <- read_csv_if_exists(file.path(dir_path, "pixel_selection.csv"))
  df_elig  <- read_csv_if_exists(file.path(dir_path, "eligible_pixels.csv"))
  df_psel  <- if (!is.null(df_psel)) .filter_within_be(df_psel) else NULL
  df_elig  <- if (!is.null(df_elig)) .filter_within_be(df_elig) else NULL
  obj_names    <- unlist(meta$objective_names)
  raster_shape <- unlist(meta$raster_info$shape)
  norm_scales  <- tryCatch(unlist(meta$algorithm$objective_normalization$scales),
                           error = function(e) NULL)
  invert_cols <- grep(
    "anomaly|connectivity_gain|restoration_potential|landscape_context",
    obj_names, value = TRUE)
  if (length(invert_cols) > 0L) {
    df_obj[, invert_cols] <- -df_obj[, invert_cols]
    if (!is.null(df_norm))
      df_norm[, intersect(invert_cols, names(df_norm))] <-
        -df_norm[, intersect(invert_cols, names(df_norm))]
  }
  list(label = label, dir = dir_path, meta = meta, obj_names = obj_names,
       n_obj = length(obj_names), n_solutions = meta$n_solutions,
       n_nondom = meta$n_nondominated_solutions, raster_shape = raster_shape,
       norm_scales = norm_scales, df_obj = df_obj, df_norm = df_norm,
       df_hv = df_hv, df_pop = df_pop, df_psel = df_psel, df_elig = df_elig)
}

# Lightweight run loader (objectives + selection only).
load_run <- function(dir_path, label = basename(dir_path)) {
  dir_path <- normalizePath(dir_path, mustWork = TRUE)
  if (!file.exists(file.path(dir_path, "metadata.json")))
    stop("Missing metadata.json in ", dir_path)
  meta            <- jsonlite::read_json(file.path(dir_path, "metadata.json"))
  objectives      <- read_csv_if_exists(file.path(dir_path, "objectives.csv"))
  objectives_norm <- read_csv_if_exists(file.path(dir_path, "objectives_normalized.csv"))
  pixel_selection <- read_csv_if_exists(file.path(dir_path, "pixel_selection.csv"))
  eligible_pixels <- read_csv_if_exists(file.path(dir_path, "eligible_pixels.csv"))
  pixel_selection <- if (!is.null(pixel_selection)) .filter_within_be(pixel_selection) else NULL
  eligible_pixels <- if (!is.null(eligible_pixels)) .filter_within_be(eligible_pixels) else NULL
  obj_names       <- unlist(meta$objective_names)
  if (is.null(objectives))      stop("Missing objectives.csv in ",     dir_path)
  if (is.null(pixel_selection)) stop("Missing pixel_selection.csv in ", dir_path)
  list(label = label, dir = dir_path, meta = meta, obj_names = obj_names,
       objectives = objectives, objectives_norm = objectives_norm,
       pixel_selection = pixel_selection, eligible_pixels = eligible_pixels)
}

# dir_path    : path to an r_inputs/<label>/ export folder
# label       : human-readable label for this specific run
# dim_type    : sensitivity dimension this run belongs to ("indicator", "policy",
#               "seed", or "param")
# group_label : the specific group within that dimension (e.g. "global_all",
#               "burden_sharing_yes", "seed_101")
load_run_rfop <- function(dir_path, label, dim_type, group_label) {
  dir_path <- normalizePath(dir_path, mustWork = TRUE)

  psel_path <- file.path(dir_path, "pixel_selection.csv")
  if (!file.exists(psel_path)) {
    warning(sprintf("pixel_selection.csv not found in %s - skipping.", dir_path))
    return(NULL)
  }

  meta_path <- file.path(dir_path, "metadata.json")
  n_nondom  <- if (file.exists(meta_path)) {
    meta <- jsonlite::read_json(meta_path)
    as.integer(meta$n_nondominated_solutions %||% meta$n_solutions %||% 1L)
  } else {
    warning(sprintf("metadata.json not found in %s - rfop_pct will be approximate.", dir_path))
    NA_integer_
  }

  # pixel_selection.csv can be tens of MB; only x,y are needed (one row per
  # selected pixel per solution). Read just those columns with the fastest
  # available reader - base read.csv on full files is the bottleneck.
  df <- .filter_within_be(.read_xy_fast(psel_path))

  # Aggregate over solutions: count how many non-dominated solutions selected each pixel
  freq_df <- df |>
    dplyr::count(x, y, name = "n_selected") |>
    dplyr::mutate(
      rfop_pct    = n_selected / n_nondom * 100,
      run_label   = label,
      dim_type    = dim_type,
      group_label = group_label
    ) |>
    dplyr::select(x, y, rfop_pct, run_label, dim_type, group_label)

  freq_df
}

# Null-coalescing operator (available in R 4.4+; define here for older versions)
`%||%` <- function(a, b) if (!is.null(a)) a else b


# Classify each eligible pixel into one of four priority classes based on
# condition-scenario mean RFOP and sensitivity to indicator / policy assumptions.
# Seed variation (algorithm stochasticity) is excluded from the classification -
# it represents estimation noise, not true ecological uncertainty. It is retained
# as a diagnostic column (sens_seed) and used separately in compute_seed_snr().
#
# runs_df       : bind_rows() of load_run_rfop() outputs; columns
#                 x, y, rfop_pct, run_label, dim_type, group_label
# elig_df       : eligible_pixels.csv data frame (x, y) - used to ensure every
#                 eligible pixel receives a class (pixels absent from all runs
#                 are treated as rfop_pct = 0 and classified as "Low priority")
# thresh_low    : pixels with mean_rfop_cond < thresh_low  -> "Low priority"
# thresh_high   : pixels with mean_rfop_cond >= thresh_high AND low sensitivity -> "Robust priority"
# thresh_stable : normalised sensitivity threshold; nsens = range(group means) / (mean_rfop + 1)
#
# Returns a data frame with one row per eligible pixel:
#   x, y, mean_rfop_cond, sens_indicator, sens_policy, sens_seed,
#   nsens_ind, nsens_pol, dominant_cond_dim, classification (ordered factor)
# DEPRECATED - replaced by compute_rfop_sensitivity() below.
# Kept to avoid breaking any calls outside the main QMD section.
compute_spatial_classification <- function(
    runs_df,
    elig_df       = NULL,
    thresh_low    = 20,
    thresh_high   = 60,
    thresh_stable = 0.3
) {
  stopifnot(all(c("x", "y", "rfop_pct", "dim_type", "group_label") %in% names(runs_df)))

  cond_runs <- dplyr::filter(runs_df, dim_type %in% c("indicator", "benchmark", "policy"))

  per_dim_sens <- cond_runs %>%
    dplyr::group_by(x, y, dim_type, group_label) %>%
    dplyr::summarise(group_mean_rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop") %>%
    dplyr::group_by(x, y, dim_type) %>%
    dplyr::summarise(sens_raw = sd(group_mean_rfop), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = dim_type, values_from = sens_raw,
                       names_prefix = "sens_", values_fill = 0)

  for (dim in c("sens_indicator", "sens_benchmark", "sens_policy")) {
    if (!dim %in% names(per_dim_sens)) per_dim_sens[[dim]] <- 0
  }

  seed_runs <- dplyr::filter(runs_df, dim_type == "seed")
  if (nrow(seed_runs) > 0) {
    seed_sens <- seed_runs %>%
      dplyr::group_by(x, y, group_label) %>%
      dplyr::summarise(group_mean_rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop") %>%
      dplyr::group_by(x, y) %>%
      dplyr::summarise(sens_seed = max(group_mean_rfop) - min(group_mean_rfop), .groups = "drop")
  } else {
    seed_sens <- dplyr::distinct(runs_df, x, y) %>%
      dplyr::mutate(sens_seed = NA_real_)
  }

  mean_rfop <- cond_runs %>%
    dplyr::group_by(x, y) %>%
    dplyr::summarise(mean_rfop_cond = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

  class_df <- mean_rfop %>%
    dplyr::left_join(per_dim_sens, by = c("x", "y")) %>%
    dplyr::left_join(seed_sens,    by = c("x", "y"))

  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    never_selected <- dplyr::anti_join(elig_df[, c("x", "y")], class_df, by = c("x", "y")) %>%
      dplyr::mutate(mean_rfop_cond = 0, sens_indicator = 0,
                    sens_benchmark = 0, sens_policy = 0, sens_seed = NA_real_)
    class_df <- dplyr::bind_rows(class_df, never_selected)
  }

  class_df <- class_df %>%
    dplyr::mutate(dplyr::across(c(mean_rfop_cond, sens_indicator, sens_benchmark, sens_policy),
                                \(v) dplyr::coalesce(v, 0))) %>%
    dplyr::mutate(
      denom       = mean_rfop_cond + 1,
      nsens_ind   = sens_indicator / denom,
      nsens_bench = sens_benchmark / denom,
      nsens_pol   = sens_policy    / denom,
      dominant_cond_dim = dplyr::case_when(
        nsens_ind   >= pmax(nsens_bench, nsens_pol) & nsens_ind   >= thresh_stable ~ "indicator",
        nsens_bench >= nsens_pol                    & nsens_bench >= thresh_stable ~ "benchmark",
        nsens_pol   >= thresh_stable                                               ~ "policy",
        TRUE                                                                       ~ NA_character_
      ),
      classification = dplyr::case_when(
        mean_rfop_cond < thresh_low                                               ~ "Low priority",
        mean_rfop_cond >= thresh_high & nsens_ind < thresh_stable &
          nsens_bench < thresh_stable & nsens_pol < thresh_stable                 ~ "Robust priority",
        nsens_ind >= pmax(nsens_bench, nsens_pol) & nsens_ind >= thresh_stable    ~ "Indicator sensitive",
        nsens_bench >= nsens_pol & nsens_bench >= thresh_stable                   ~ "Benchmark sensitive",
        nsens_pol >= thresh_stable                                                ~ "Policy sensitive",
        TRUE                                                                      ~ "Robust priority"
      ),
      classification = factor(
        classification,
        levels = c("Robust priority", "Indicator sensitive", "Benchmark sensitive",
                   "Policy sensitive", "Low priority"),
        ordered = TRUE
      )
    )

  class_df
}


# -- Internal helper: per-pixel seed rank stability ----------------------------
# For each seed run (from indicator + benchmark dims), rank all pixels by RFOP
# percentile (0-100). Per-pixel mean_seed_rank_SD is the SD of those percentile
# ranks across seed runs. Scale-invariant: a pixel consistently in the 90th
# percentile has low rank SD even when its absolute RFOP is small.
.compute_pixel_rank_stability <- function(runs_df, seed_noise_dims = c("indicator", "benchmark")) {
  seed_runs <- dplyr::filter(runs_df, dim_type %in% seed_noise_dims)

  if (nrow(seed_runs) == 0) {
    return(dplyr::distinct(runs_df, x, y) |> dplyr::mutate(mean_seed_rank_SD = NA_real_))
  }

  # For each scenario (group_label), compute a per-pixel rank stability score:
  #   1. Rank pixels by RFOP within each seed run (among selected pixels only).
  #   2. Per-pixel SD of those ranks across seeds = within-scenario rank SD.
  #   3. Pixels with rfop = 0 in ALL seeds are consistently unselected  rank_sd = 0.
  #
  # Crucially, we rank ONLY among pixels that are non-zero in each seed column.
  # Including the mass of zero-RFOP pixels in the ranking inflates SD dramatically:
  # a pixel at rfop=5% in one seed and rfop=0 in another would jump from mid-rank
  # to bottom-rank relative to 130 000 other pixels, even though the absolute
  # change is tiny.  Ranking only among selected pixels makes the metric meaningful.
  #
  # NOTE: we process one group_label at a time so we never mix between-scenario
  # rank differences (ecological sensitivity signal) with within-scenario seed noise.
  scenarios <- unique(seed_runs[, c("dim_type", "group_label"), drop = FALSE])

  within_rank_sds <- lapply(seq_len(nrow(scenarios)), function(i) {
    dim_t       <- scenarios$dim_type[[i]]
    grp         <- scenarios$group_label[[i]]

    sub         <- dplyr::filter(seed_runs, dim_type == dim_t, group_label == grp)
    seed_labels <- unique(sub$run_label)
    if (length(seed_labels) < 2) return(NULL)

    wide <- sub |>
      dplyr::select(x, y, run_label, rfop_pct) |>
      tidyr::pivot_wider(
        id_cols     = c(x, y),
        names_from  = run_label,
        values_from = rfop_pct,
        values_fill = 0
      )

    seed_cols <- setdiff(names(wide), c("x", "y"))
    mat       <- as.matrix(wide[, seed_cols, drop = FALSE])

    # Start with all rank_sd = 0 (consistently unselected baseline)
    result <- dplyr::tibble(x = wide$x, y = wide$y, rank_sd = 0, group_label = grp)

    # Per-seed: rank only among pixels with rfop > 0 in that seed.
    # A pixel absent from a seed (rfop = 0) receives rank 0 in that seed.
    n_seeds  <- length(seed_cols)
    rank_mat <- matrix(0, nrow = nrow(mat), ncol = n_seeds)
    for (j in seq_len(n_seeds)) {
      v        <- mat[, j]
      sel      <- v > 0
      if (sum(sel) > 0) {
        n_sel <- sum(sel)
        rank_mat[sel, j] <- rank(v[sel], ties.method = "average") / n_sel * 100
      }
      # pixels with v == 0 keep rank 0 in this column
    }

    # Only report SD for pixels non-zero in at least one seed; others stay 0
    ever_sel <- rowSums(mat > 0) > 0
    if (any(ever_sel)) {
      result$rank_sd[ever_sel] <- apply(rank_mat[ever_sel, , drop = FALSE], 1,
                                        sd, na.rm = TRUE)
    }
    result
  })

  bound <- dplyr::bind_rows(Filter(Negate(is.null), within_rank_sds))
  if (nrow(bound) == 0) {
    return(dplyr::distinct(runs_df, x, y) |> dplyr::mutate(mean_seed_rank_SD = NA_real_))
  }

  # Average within-scenario rank SDs across all scenarios
  bound |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(mean_seed_rank_SD = mean(rank_sd, na.rm = TRUE), .groups = "drop")
}


# Load a single run's pixel-level RFOP using within-objective-cluster frequencies.
# Non-dominated solutions are clustered in normalised objective space (k-means,
# k = n_clusters). Per-pixel RFOP is computed within each cluster and rfop_pct
# is set to the maximum across clusters - i.e. the pixel's best consistency
# within any single trade-off region.
# Falls back to load_run_rfop() if objectives.csv is missing or clustering fails.
# Returns the same column structure as load_run_rfop() for drop-in compatibility.
#
# dir_path              : r_inputs export directory for this run
# label, dim_type, group_label : passed through unchanged
# n_clusters            : number of objective-space clusters (default 3)
compute_stratified_rfop <- function(dir_path, label, dim_type, group_label, n_clusters = 3L) {
  dir_path  <- normalizePath(dir_path, mustWork = TRUE)
  psel_path <- file.path(dir_path, "pixel_selection.csv")
  if (!file.exists(psel_path)) {
    warning(sprintf("pixel_selection.csv not found in %s - skipping.", dir_path))
    return(NULL)
  }

  meta_path <- file.path(dir_path, "metadata.json")
  obj_path  <- file.path(dir_path, "objectives.csv")
  if (!file.exists(meta_path) || !file.exists(obj_path)) {
    message(sprintf("metadata.json or objectives.csv missing in %s - falling back to unstratified RFOP.", dir_path))
    return(load_run_rfop(dir_path, label, dim_type, group_label))
  }

  meta      <- jsonlite::read_json(meta_path)
  obj_names <- unlist(meta$objective_names)

  obj_df <- read.csv(obj_path)
  nd_df  <- obj_df[obj_df$is_nondominated == 1, , drop = FALSE]

  if (nrow(nd_df) < n_clusters) {
    message(sprintf("Too few non-dominated solutions (%d) for %d clusters in %s - falling back.",
                    nrow(nd_df), n_clusters, dir_path))
    return(load_run_rfop(dir_path, label, dim_type, group_label))
  }

  # Normalise objective matrix column-wise to [0, 1]
  valid_obj <- intersect(obj_names, names(nd_df))
  obj_mat   <- as.matrix(nd_df[, valid_obj, drop = FALSE])
  col_min   <- apply(obj_mat, 2, min, na.rm = TRUE)
  col_rng   <- apply(obj_mat, 2, max, na.rm = TRUE) - col_min
  col_rng[col_rng == 0] <- 1
  obj_norm  <- sweep(sweep(obj_mat, 2, col_min, "-"), 2, col_rng, "/")

  # K-means with fixed seed for reproducibility across all runs
  set.seed(42L)
  km <- tryCatch(
    kmeans(obj_norm, centers = n_clusters, nstart = 10L, iter.max = 50L),
    error = function(e) NULL
  )
  if (is.null(km)) {
    message(sprintf("k-means failed in %s - falling back to unstratified RFOP.", dir_path))
    return(load_run_rfop(dir_path, label, dim_type, group_label))
  }
  nd_df$cluster  <- km$cluster
  cluster_sizes  <- as.integer(table(factor(nd_df$cluster, levels = seq_len(n_clusters))))

  # Join cluster assignment to pixel selections
  psel_df   <- read.csv(psel_path)
  psel_nd   <- psel_df[psel_df$solution_id %in% nd_df$solution_id, , drop = FALSE]
  psel_clus <- dplyr::left_join(psel_nd, nd_df[, c("solution_id", "cluster")],
                                by = "solution_id")

  # Per-pixel, per-cluster within-cluster RFOP
  pixel_clus <- psel_clus |>
    dplyr::count(x, y, cluster, name = "n_sel") |>
    dplyr::mutate(
      clus_size = cluster_sizes[cluster],
      rfop_c    = ifelse(clus_size > 0L, n_sel / clus_size * 100, 0)
    )

  wide_df <- pixel_clus |>
    tidyr::pivot_wider(
      id_cols      = c(x, y),
      names_from   = cluster,
      values_from  = rfop_c,
      names_prefix = "rfop_c",
      values_fill  = 0
    )

  rfop_cols <- paste0("rfop_c", seq_len(n_clusters))
  for (col in rfop_cols) if (!col %in% names(wide_df)) wide_df[[col]] <- 0

  wide_df$rfop_max <- apply(wide_df[, rfop_cols, drop = FALSE], 1, max, na.rm = TRUE)

  # Return in load_run_rfop()-compatible format; rfop_pct = rfop_max
  wide_df |>
    dplyr::mutate(
      rfop_pct    = rfop_max,
      run_label   = label,
      dim_type    = dim_type,
      group_label = group_label
    ) |>
    dplyr::select(x, y, rfop_pct, run_label, dim_type, group_label)
}


# -- Unified RFOP sensitivity and robustness classification -------------------
#
# Distinguishes five pixel types:
#   1. Robust high-priority areas
#   2. Areas sensitive to condition (indicator) assumptions
#   3. Areas sensitive to policy levers
#   4. Areas sensitive to benchmarking method
#   5. Areas uncertain because of NSGA-III stochastic noise
#
# Per-pixel statistics computed:
#   mean_RFOP        - equal-weighted average of the three per-dimension means
#                      (condition-mean, policy-mean, benchmark-mean)
#   SD_condition     - SD of scenario-mean RFOP across condition (indicator) scenarios
#   SD_policy        - SD of scenario-mean RFOP across policy scenarios
#   SD_benchmark     - SD of scenario-mean RFOP across benchmark scenarios
#   mean_seed_SD     - mean within-scenario seed SD, pooled from indicator + benchmark
#                      multi-seed runs (policy excluded when only 1 seed per policy scenario)
#   SNR_condition    - SD_condition / (mean_seed_SD + snr_denominator_constant)
#   SNR_policy       - SD_policy    / (mean_seed_SD + snr_denominator_constant);
#                      always computed; noise floor from indicator seed runs
#   SNR_benchmark    - SD_benchmark / (mean_seed_SD + snr_denominator_constant)
#
# Flags:
#   high_priority         - mean_RFOP >= rfop_cutoff  (percentile-based threshold)
#   seed_stable           - mean_seed_SD <= seed_sd_threshold
#   condition_detectable  - SNR_condition > snr_threshold
#   policy_detectable     - SNR_policy > snr_threshold  (FALSE when SNR_policy is NA)
#   benchmark_detectable  - SNR_benchmark > snr_threshold
#
# dominant_sensitivity: the epistemic dimension with the largest detectable SNR.
#   "mixed sensitive"            - >1 detectable SNR and top two within mixed_tolerance
#   "condition sensitive"        - SNR_condition is the uniquely largest detectable SNR
#   "policy sensitive"           - SNR_policy is the uniquely largest detectable SNR
#   "benchmark sensitive"        - SNR_benchmark is the uniquely largest detectable SNR
#   "low substantive sensitivity"- no SNR is detectable
#
# Outputs:
#   priority_status + dominant_sensitivity + seed_status   classification  (3-part string)
#   map_class                                              simplified class for figures
#
# Arguments:
#   runs_df                 : bind_rows() of load_run_rfop() outputs
#   elig_df                 : eligible_pixels data frame (x, y) for zero-filling absent pixels
#   high_priority_threshold : percentile (0-100) of mean_RFOP among selected pixels
#                             used as the "high priority" cutoff; default 80 = top 20%
#   snr_threshold           : SNR threshold above which a dimension is "detectable" (default 1)
#   seed_sd_threshold       : mean_seed_SD threshold; above this  "seed uncertain" (default 5)
#   mixed_tolerance         : fractional tolerance for calling "mixed sensitive" (default 0.2)
#   snr_denominator_constant: added to mean_seed_SD before dividing (default 1)
compute_rfop_sensitivity <- function(
    runs_df,
    elig_df                  = NULL,
    high_priority_threshold  = 80,
    snr_threshold            = 1,
    seed_sd_threshold        = 5,
    mixed_tolerance          = 0.2,
    snr_denominator_constant = 1,
    use_rank_stability       = TRUE,
    rank_stability_threshold = 15,
    noise_floor              = c("se", "sd")
) {
  stopifnot(all(c("x", "y", "rfop_pct", "dim_type", "group_label") %in% names(runs_df)))
  noise_floor <- match.arg(noise_floor)
  # noise_floor = "se": the seed noise floor is the standard ERROR of the
  #   scenario mean (within-scenario seed SD / sqrt(n_seeds)) - the correct floor
  #   when scenario means are averaged over many seeds, so it is comparable to the
  #   seed-averaged signal dimensions. "sd" uses the raw single-seed SD (the old,
  #   over-conservative behaviour).

  # -- Step 1: per-dimension SD of scenario-means ------------------------------
  # For each dimension, average across seeds within each scenario (group_label),
  # then take the SD of those scenario-means across scenarios.
  epistemic_dims <- c("indicator", "policy", "benchmark")

  scenario_means <- runs_df |>
    dplyr::filter(dim_type %in% epistemic_dims) |>
    dplyr::group_by(x, y, dim_type, group_label) |>
    dplyr::summarise(scenario_mean = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

  # Zero-fill: pixels absent from a particular scenario get scenario_mean = 0.
  # Without this, pixels never selected in some scenarios are silently dropped,
  # severely underestimating between-scenario SD (and therefore SNR).
  # A pixel selected in 5 of 13 indicator scenarios but not the other 8 should
  # have high SD; without zero-fill its SD is computed only over the 5 non-zero values.
  .ep_pixels <- dplyr::distinct(
    runs_df[runs_df$dim_type %in% epistemic_dims, c("x", "y")]
  )
  .ep_groups <- runs_df |>
    dplyr::filter(dim_type %in% epistemic_dims) |>
    dplyr::distinct(dim_type, group_label)
  scenario_means <- tidyr::crossing(.ep_pixels, .ep_groups) |>
    dplyr::left_join(scenario_means, by = c("x", "y", "dim_type", "group_label")) |>
    dplyr::mutate(scenario_mean = dplyr::coalesce(scenario_mean, 0))
  rm(.ep_pixels, .ep_groups)

  dim_sd <- scenario_means |>
    dplyr::group_by(x, y, dim_type) |>
    dplyr::summarise(
      dim_mean = mean(scenario_mean, na.rm = TRUE),
      dim_sd   = sd(scenario_mean,   na.rm = TRUE),
      n_groups = dplyr::n(),
      .groups  = "drop"
    ) |>
    tidyr::pivot_wider(
      id_cols     = c(x, y),
      names_from  = dim_type,
      values_from = c(dim_mean, dim_sd, n_groups)
    )

  # Ensure all expected columns exist even when a dimension has no runs
  for (dim in epistemic_dims) {
    for (pfx in c("dim_mean_", "dim_sd_", "n_groups_")) {
      col <- paste0(pfx, dim)
      if (!col %in% names(dim_sd)) dim_sd[[col]] <- rep(NA_real_, nrow(dim_sd))
    }
  }

  # -- Step 2: mean_RFOP - equal-weighted average of three dimension-means -----
  dim_sd <- dim_sd |>
    dplyr::mutate(
      mean_RFOP = rowMeans(
        cbind(dim_mean_indicator, dim_mean_policy, dim_mean_benchmark),
        na.rm = TRUE
      )
    )

  # Rename SD columns for readability
  dim_sd <- dim_sd |>
    dplyr::rename(
      SD_condition = dim_sd_indicator,
      SD_policy    = dim_sd_policy,
      SD_benchmark = dim_sd_benchmark
    )

  # -- Step 3: mean_seed_SD - pooled within-scenario seed noise -------------
  # Computed from indicator + benchmark multi-seed runs only.
  # Policy runs are excluded here: they typically have only 1 seed per scenario,
  # making within-scenario SD undefined.
  seed_noise_dims <- c("indicator", "benchmark")

  within_sd <- runs_df |>
    dplyr::filter(dim_type %in% seed_noise_dims) |>
    dplyr::group_by(x, y, dim_type, group_label) |>
    dplyr::summarise(
      within_sd = sd(rfop_pct, na.rm = TRUE),
      n_seeds   = dplyr::n(),
      .groups   = "drop"
    ) |>
    dplyr::filter(n_seeds > 1) |>   # exclude single-seed scenarios (SD would be NA)
    # Standard error of the scenario mean when noise_floor = "se".
    dplyr::mutate(noise = if (noise_floor == "se") within_sd / sqrt(n_seeds) else within_sd)

  if (nrow(within_sd) > 0) {
    mean_seed_sd_df <- within_sd |>
      dplyr::group_by(x, y) |>
      dplyr::summarise(mean_seed_SD = mean(noise, na.rm = TRUE), .groups = "drop")
  } else {
    mean_seed_sd_df <- dplyr::distinct(runs_df, x, y) |>
      dplyr::mutate(mean_seed_SD = NA_real_)
  }

  # -- Step 3b: per-pixel seed rank stability ---------------------------------
  # SD of each pixel's RFOP percentile rank across seed runs (0-100 scale).
  # Scale-invariant: a pixel consistently in the 90th percentile has low rank SD
  # even when its absolute RFOP is small.
  rank_stab_df <- .compute_pixel_rank_stability(runs_df, seed_noise_dims)

  # -- Step 4: detect whether policy has any seed replication ------------------
  policy_has_seeds <- runs_df |>
    dplyr::filter(dim_type == "policy") |>
    dplyr::group_by(group_label) |>
    dplyr::summarise(n_seeds = dplyr::n_distinct(run_label), .groups = "drop") |>
    dplyr::summarise(any_multi = any(n_seeds > 1)) |>
    dplyr::pull(any_multi)
  if (length(policy_has_seeds) == 0) policy_has_seeds <- FALSE

  # -- Step 5: join everything and compute SNRs -----------------------------
  sens_df <- dim_sd |>
    dplyr::left_join(mean_seed_sd_df, by = c("x", "y")) |>
    dplyr::left_join(rank_stab_df,    by = c("x", "y")) |>
    dplyr::mutate(
      dplyr::across(c(mean_RFOP, SD_condition, SD_policy, SD_benchmark),
                    \(v) dplyr::coalesce(v, 0)),
      mean_seed_SD      = dplyr::coalesce(mean_seed_SD, 0),
      mean_seed_rank_SD = dplyr::coalesce(mean_seed_rank_SD, 0),
      snr_denom    = mean_seed_SD + snr_denominator_constant,
      SNR_condition = SD_condition / snr_denom,
      SNR_benchmark = SD_benchmark / snr_denom,
      # SNR_policy is NA (suppressed) when policy runs lack seed replication.
      # Use base-R if/else (not dplyr::if_else) so a scalar condition can return
      # either a full-length vector or a recycled NA without a length mismatch.
      SNR_policy    = if (policy_has_seeds) SD_policy / snr_denom else NA_real_
    ) |>
    dplyr::select(-snr_denom)

  # -- Step 5b: percentile-based high-priority cutoff --------------------------
  rfop_cutoff <- as.numeric(quantile(
    sens_df$mean_RFOP[sens_df$mean_RFOP > 0],
    high_priority_threshold / 100,
    na.rm = TRUE
  ))

  # -- Step 6: boolean flags ---------------------------------------------------
  sens_df <- sens_df |>
    dplyr::mutate(
      high_priority        = mean_RFOP        >= rfop_cutoff,
      seed_stable          = mean_seed_SD     <= seed_sd_threshold,
      seed_stable_rank     = mean_seed_rank_SD <= rank_stability_threshold,
      # seed_ok: active stability flag - rank-based (scale-invariant) when use_rank_stability = TRUE
      seed_ok              = if (isTRUE(use_rank_stability)) seed_stable_rank else seed_stable,
      condition_detectable = !is.na(SNR_condition) & SNR_condition > snr_threshold,
      policy_detectable    = !is.na(SNR_policy)    & SNR_policy    > snr_threshold,
      benchmark_detectable = !is.na(SNR_benchmark) & SNR_benchmark > snr_threshold
    )

  # -- Step 7: dominant_sensitivity -----------------------------------------
  sens_df <- sens_df |>
    dplyr::mutate(
      dominant_sensitivity = purrr::pmap_chr(
        list(
          snr_c = SNR_condition,
          snr_p = SNR_policy,
          snr_b = SNR_benchmark,
          det_c = condition_detectable,
          det_p = policy_detectable,
          det_b = benchmark_detectable
        ),
        function(snr_c, snr_p, snr_b, det_c, det_p, det_b) {
          # Collect detectable SNR values (suppress NAs)
          vals <- c(
            if (det_c) c(condition = snr_c) else NULL,
            if (det_p) c(policy    = snr_p) else NULL,
            if (det_b) c(benchmark = snr_b) else NULL
          )
          if (length(vals) == 0) return("low substantive sensitivity")
          if (length(vals) == 1) return(paste(names(vals), "sensitive"))
          # More than one detectable: check if top two are within mixed_tolerance
          sorted  <- sort(vals, decreasing = TRUE)
          top_two_close <- (sorted[1] - sorted[2]) / sorted[1] <= mixed_tolerance
          if (top_two_close) return("mixed sensitive")
          return(paste(names(sorted)[1], "sensitive"))
        }
      )
    )

  # -- Step 8: priority_status, seed_status, detailed classification ----------
  sens_df <- sens_df |>
    dplyr::mutate(
      priority_status = dplyr::if_else(high_priority, "high priority", "low priority"),
      seed_status     = dplyr::if_else(seed_ok,   "seed stable",   "seed uncertain"),
      classification  = paste(priority_status, dominant_sensitivity, seed_status, sep = ", ")
    )

  # -- Step 9: map_class variants -----------------------------------------------
  # map_class          : combined (condition + policy + benchmark + seed)
  # map_class_cond_bench: condition/benchmark only - ignores policy dimension
  # map_class_policy   : policy only - no mixed category
  sens_df <- sens_df |>
    dplyr::mutate(
      # -- 9a: full combined map_class ------------------------------------------
      map_class = dplyr::case_when(
        !high_priority                                ~ "Low priority",
        !seed_ok                                      ~ "Seed uncertain",
        dominant_sensitivity == "condition sensitive" ~ "High priority, condition sensitive",
        dominant_sensitivity == "policy sensitive"    ~ "High priority, policy sensitive",
        dominant_sensitivity == "benchmark sensitive" ~ "High priority, benchmark sensitive",
        dominant_sensitivity == "mixed sensitive"     ~ "High priority, mixed sensitive",
        dominant_sensitivity == "low substantive sensitivity" ~ "High priority, low sensitivity",
        TRUE                                          ~ "Unclassified"
      ),
      map_class = factor(
        map_class,
        levels = c("High priority, low sensitivity", "High priority, condition sensitive", "High priority, policy sensitive",
                   "High priority, benchmark sensitive", "High priority, mixed sensitive", "Seed uncertain", "Low priority")
      ),

      # -- 9b: condition/benchmark map_class ------------------------------------
      # Re-derive dominant signal from only condition and benchmark SNRs.
      .dom_cb = purrr::pmap_chr(
        list(snr_c = SNR_condition, snr_b = SNR_benchmark,
             det_c = condition_detectable, det_b = benchmark_detectable),
        function(snr_c, snr_b, det_c, det_b) {
          vals <- c(
            if (isTRUE(det_c)) c(condition = snr_c) else NULL,
            if (isTRUE(det_b)) c(benchmark = snr_b) else NULL
          )
          if (length(vals) == 0) return("low substantive sensitivity")
          if (length(vals) == 1) return(paste(names(vals), "sensitive"))
          sorted <- sort(vals, decreasing = TRUE)
          if ((sorted[1] - sorted[2]) / sorted[1] <= mixed_tolerance)
            return("mixed sensitive")
          return(paste(names(sorted)[1], "sensitive"))
        }
      ),
      map_class_cond_bench = dplyr::case_when(
        !high_priority              ~ "low priority",
        !seed_ok                    ~ "seed uncertain",
        .dom_cb == "condition sensitive" ~ "condition sensitive",
        .dom_cb == "benchmark sensitive" ~ "benchmark sensitive",
        .dom_cb == "mixed sensitive"     ~ "mixed sensitive",
        TRUE                             ~ "core robust priority"
      ),
      map_class_cond_bench = factor(
        map_class_cond_bench,
        levels = c("Core robust priority", "Condition sensitive",
                   "Benchmark sensitive", "Mixed sensitive",
                   "Seed uncertain", "Low priority")
      ),

      # -- 9c: condition-only map_class -------------------------------------------
      map_class_condition = dplyr::case_when(
        !high_priority        ~ "Low priority",
        !seed_ok              ~ "Seed uncertain",
        condition_detectable  ~ "Condition sensitive",
        TRUE                  ~ "Robust priority"
      ),
      map_class_condition = factor(
        map_class_condition,
        levels = c("robust priority", "condition sensitive", "seed uncertain", "low priority")
      ),

      # -- 9d: benchmark-only map_class -----------------------------------------
      map_class_benchmark = dplyr::case_when(
        !high_priority        ~ "low priority",
        !seed_ok              ~ "seed uncertain",
        benchmark_detectable  ~ "benchmark sensitive",
        TRUE                  ~ "robust priority"
      ),
      map_class_benchmark = factor(
        map_class_benchmark,
        levels = c("robust priority", "benchmark sensitive", "seed uncertain", "low priority")
      ),

      # -- 9e: policy-only map_class --------------------------------------------
      map_class_policy = dplyr::case_when(
        !high_priority    ~ "low priority",
        !seed_ok          ~ "seed uncertain",
        policy_detectable ~ "policy sensitive",
        TRUE              ~ "robust priority"
      ),
      map_class_policy = factor(
        map_class_policy,
        levels = c("robust priority", "policy sensitive", "seed uncertain", "low priority")
      )
    ) |>
    dplyr::select(-.dom_cb)

  # -- Step 10: fill eligible pixels never selected in any run --------------
  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    never_selected <- dplyr::anti_join(elig_df[, c("x", "y")], sens_df, by = c("x", "y")) |>
      dplyr::mutate(
        mean_RFOP            = 0,
        SD_condition         = 0,  SD_policy    = 0,  SD_benchmark  = 0,
        mean_seed_SD         = 0,
        mean_seed_rank_SD    = 0,
        SNR_condition        = 0,  SNR_policy   = NA_real_, SNR_benchmark = 0,
        high_priority        = FALSE, seed_stable = TRUE,
        seed_stable_rank     = TRUE,  seed_ok = TRUE,
        condition_detectable = FALSE, policy_detectable = FALSE,
        benchmark_detectable = FALSE,
        dominant_sensitivity = "low substantive sensitivity",
        priority_status      = "low priority",
        seed_status          = "seed stable",
        classification       = "low priority, low substantive sensitivity, seed stable",
        map_class            = factor("Low priority",
                                      levels = levels(sens_df$map_class)),
        map_class_cond_bench = factor("low priority",
                                      levels = levels(sens_df$map_class_cond_bench)),
        map_class_condition  = factor("low priority",
                                      levels = levels(sens_df$map_class_condition)),
        map_class_benchmark  = factor("low priority",
                                      levels = levels(sens_df$map_class_benchmark)),
        map_class_policy     = factor("low priority",
                                      levels = levels(sens_df$map_class_policy))
      )
    # Drop auxiliary pivot columns from never_selected that may not match sens_df
    extra_cols <- setdiff(names(sens_df), names(never_selected))
    if (length(extra_cols) > 0 && nrow(never_selected) > 0)
      for (col in extra_cols) never_selected[[col]] <- rep(NA_real_, nrow(never_selected))
    sens_df <- dplyr::bind_rows(sens_df, never_selected)
  }

  attr(sens_df, "rfop_cutoff") <- rfop_cutoff
  sens_df
}


# Compute a per-pixel signal-to-noise ratio to assess where the condition
# sensitivity signal is detectable above algorithm stochasticity.
#
# -- Visualisation functions for compute_rfop_sensitivity() output -------------

# Colour palette for map_class (7 classes).
# Paul Tol "muted" qualitative palette (colorblind-safe under deuteranopia,
# protanopia, tritanopia). Hues chosen to avoid overlap with .factor_palette
# (Okabe-Ito blue/orange/green/pink-purple/vermillion). Greys retained as-is.
.sensitivity_map_palette <- c(
  "High priority, low sensitivity"     = "#44AA99",   # teal
  "High priority, condition sensitive" = "#CC6677",   # rose
  "High priority, policy sensitive"    = "#332288",   # indigo
  "High priority, benchmark sensitive" = "#DDCC77",   # sand
  "High priority, mixed sensitive"     = "#882255",   # wine
  "Seed uncertain"                     = "#7f7f7f",   # mid grey (keep)
  "Low priority"                       = "#bec2bf"    # light grey (keep)
)

# Colour palette for dominant_sensitivity (scatter / bar companions to above).
# Same hues as .sensitivity_map_palette so the two stay visually consistent.
.dominance_palette <- c(
  "condition sensitive"         = "#CC6677",
  "policy sensitive"            = "#332288",
  "benchmark sensitive"         = "#DDCC77",
  "mixed sensitive"             = "#882255",
  "low substantive sensitivity" = "#44AA99"
)


# Colour palette for map_class_cond_bench (6 classes; no policy)
.cond_bench_palette <- c(
  "core robust priority" = "#0c7b85",
  "condition sensitive"  = "#bd8c12",
  "benchmark sensitive"  = "#da3114",
  "mixed sensitive"      = "#e07e30",
  "seed uncertain"       = "#7f7f7f",
  "low priority"         = "#bec2bf"
)

# Shared 4-class palette for single-dimension maps (condition, benchmark, policy)
# Classes: "robust priority", "<dim> sensitive", "seed uncertain", "low priority"
.condition_palette <- c(
  "robust priority"      = "#0c7b85",
  "condition sensitive"  = "#bd8c12",
  "seed uncertain"       = "#7f7f7f",
  "low priority"         = "#bec2bf"
)

.benchmark_palette <- c(
  "robust priority"      = "#0c7b85",
  "benchmark sensitive"  = "#da3114",
  "seed uncertain"       = "#7f7f7f",
  "low priority"         = "#bec2bf"
)

.policy_palette <- c(
  "robust priority"  = "#0c7b85",
  "policy sensitive" = "#910b9b",
  "seed uncertain"   = "#7f7f7f",
  "low priority"     = "#bec2bf"
)


# 1a. Single spatial map from a named map_class column and a named palette.
.make_one_sensitivity_map <- function(sens_df, elig_df, col, palette, title) {
  p <- ggplot() +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position  = "right",
      legend.title     = element_text(size = 10),
      legend.text      = element_text(size = 10)
    )

  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    bg <- dplyr::anti_join(elig_df[, c("x", "y")], sens_df, by = c("x", "y"))
    if (nrow(bg) > 0)
      p <- p + geom_raster(data = bg, aes(x = x, y = y), fill = "#EEEEEE")
  }

  plot_df <- sens_df
  plot_df[["__fill__"]] <- plot_df[[col]]

  p <- p +
    geom_raster(data = plot_df, aes(x = x, y = y, fill = .data[[col]])) +
    scale_fill_manual(
      values = palette,
      name   = NULL,
      drop   = TRUE,
      guide  = guide_legend(
        keywidth  = unit(0.7, "cm"),
        keyheight = unit(0.6, "cm"),
        label.hjust = 0,
        byrow = TRUE
      )
    )

  if (exists("BE"))
    p <- p + geom_sf(data = BE, fill = NA, color = "black",
                     linewidth = 0.4, inherit.aes = FALSE)
  p
}


# 1b. Dual map: condition/benchmark (left) + policy (right) side by side.
# Uses patchwork to combine.
make_dual_sensitivity_map <- function(
    sens_df,
    elig_df    = NULL,
    ecosystem  = NULL,
    title_cb   = "Condition & benchmark sensitivity",
    title_pol  = "Policy sensitivity"
) {
  p_cb <- .make_one_sensitivity_map(
    sens_df, elig_df,
    col     = "map_class_cond_bench",
    palette = .cond_bench_palette,
    title   = title_cb
  )
  p_pol <- .make_one_sensitivity_map(
    sens_df, elig_df,
    col     = "map_class_policy",
    palette = .policy_palette,
    title   = title_pol
  )
  p_cb + p_pol +
    patchwork::plot_layout(ncol = 2)
}


# 1c. Triple map: condition (left) | benchmark (centre) | policy (right).
# Each panel shows only: robust priority / <dim> sensitive / seed uncertain / low priority.
make_triple_sensitivity_map <- function(
    sens_df,
    elig_df       = NULL,
    title_cond    = "Condition sensitivity",
    title_bench   = "Benchmark sensitivity",
    title_pol     = "Policy sensitivity"
) {
  p_cond  <- .make_one_sensitivity_map(
    sens_df, elig_df,
    col     = "map_class_condition",
    palette = .condition_palette,
    title   = title_cond
  )
  p_bench <- .make_one_sensitivity_map(
    sens_df, elig_df,
    col     = "map_class_benchmark",
    palette = .benchmark_palette,
    title   = title_bench
  )
  p_pol   <- .make_one_sensitivity_map(
    sens_df, elig_df,
    col     = "map_class_policy",
    palette = .policy_palette,
    title   = title_pol
  )
  p_cond + p_bench + p_pol +
    patchwork::plot_layout(ncol = 3)
}


# 1. Main classification map using map_class (all dimensions combined).
make_sensitivity_classification_map <- function(
    sens_df,
    elig_df = NULL
) {
  p <- ggplot() +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position  = "right",
      legend.title     = element_text(size = 10),
      legend.text      = element_text(size = 11)
    )

  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    unclassified_bg <- dplyr::anti_join(elig_df[, c("x", "y")], sens_df, by = c("x", "y"))
    if (nrow(unclassified_bg) > 0)
      p <- p + geom_raster(data = unclassified_bg, aes(x = x, y = y), fill = "#EEEEEE")
  }

  p <- p +
    geom_raster(data = sens_df, aes(x = x, y = y, fill = map_class)) +
    scale_fill_manual(
      values = .sensitivity_map_palette,
      labels = scales::label_wrap(25),
      name   = NULL,
      drop   = TRUE,
      guide  = guide_legend(
        override.aes = list(size = 4),
        keywidth     = unit(0.8, "cm"),
        keyheight    = unit(0.5, "cm"),
        label.hjust  = 0
      )
    )

  if (exists("BE"))
    p <- p + geom_sf(data = BE, fill = NA, color = "black",
                     linewidth = 0.4, inherit.aes = FALSE)
  p
}


# 2. Four-panel diagnostic map: SNR_condition, SNR_policy, SNR_benchmark, mean_seed_SD.
# All panels use the same spatial layout; SNR panels share a diverging scale
# centred on snr_threshold; mean_seed_SD uses a sequential scale.
#
# sens_df      : output of compute_rfop_sensitivity()
# elig_df      : eligible pixels for grey underlay
# snr_threshold: threshold value for the SNR panel midpoint (default 1)
# snr_cap      : upper cap for display of SNR values (default 4)
make_snr_diagnostic_panels <- function(
    sens_df,
    elig_df           = NULL,
    snr_threshold     = 1,
    seed_sd_threshold = 5
) {
  snr_cols <- c(
    "SNR_condition" = "Condition SNR",
    "SNR_policy"    = "Policy SNR",
    "SNR_benchmark" = "Benchmark SNR"
  )

  # Three-class SNR palette: below threshold = red, at/near threshold = yellow, above = green
  .snr_class <- function(snr_vec, threshold) {
    dplyr::case_when(
      is.na(snr_vec)          ~ NA_character_,
      snr_vec > threshold     ~ "above threshold",
      snr_vec < threshold     ~ "below threshold",
      TRUE                    ~ "at threshold"
    ) |> factor(levels = c("below threshold", "at threshold", "above threshold"))
  }

  snr_colours <- c(
    "below threshold" = "#d73027",
    "at threshold"    = "#ffffbf",
    "above threshold" = "#1a9850"
  )

  .snr_panel <- function(col, label) {
    plot_df <- sens_df |>
      dplyr::filter(!is.na(.data[[col]])) |>
      dplyr::mutate(snr_class = .snr_class(.data[[col]], snr_threshold))

    p <- ggplot()
    if (!is.null(elig_df) && nrow(elig_df) > 0)
      p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#e6e6e6")

    if (nrow(plot_df) > 0) {
      p <- p +
        geom_raster(data = plot_df, aes(x = x, y = y, fill = snr_class)) +
        scale_fill_manual(
          values   = snr_colours,
          name     = "SNR",
          drop     = FALSE,
          na.value = "#e6e6e6",
          labels   = c(
            "below threshold" = paste0("< ", snr_threshold, " (noise dominates)"),
            "at threshold"    = paste0("= ", snr_threshold),
            "above threshold" = paste0("> ", snr_threshold, " (signal detectable)")
          )
        )
    } else {
      p <- p + annotate("text", x = Inf, y = Inf, label = "no data",
                        hjust = 1, vjust = 1, colour = "grey50", size = 3)
    }

    if (exists("BE"))
      p <- p + geom_sf(data = BE, fill = NA, color = "black",
                       linewidth = 0.3, inherit.aes = FALSE)

    p + theme_void() +
      theme(
        panel.background  = element_rect(fill = "white", colour = NA),
        legend.position   = "right",
        legend.key.height = unit(0.5, "cm")
      )
  }

  .seed_sd_panel <- function() {
    plot_df <- sens_df |>
      dplyr::filter(!is.na(mean_seed_SD))

    p <- ggplot()
    if (!is.null(elig_df) && nrow(elig_df) > 0)
      p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#e6e6e6")

    if (nrow(plot_df) > 0) {
      p <- p +
        geom_raster(data = plot_df, aes(x = x, y = y, fill = mean_seed_SD)) +
        scale_fill_viridis_c(
          option    = "plasma",
          direction = -1,
          name      = "Seed SD\n(pp)",
          limits    = c(0, NA)
        )
    }

    if (exists("BE"))
      p <- p + geom_sf(data = BE, fill = NA, color = "black",
                       linewidth = 0.3, inherit.aes = FALSE)

    p <- p + theme_void() +
      theme(
        panel.background  = element_rect(fill = "white", colour = NA),
        legend.position   = "right",
        legend.key.height = unit(0.5, "cm")
      )
  }

  panels <- c(
    lapply(names(snr_cols), function(col) .snr_panel(col, snr_cols[[col]])),
    list(.seed_sd_panel())
  )

  patchwork::wrap_plots(panels, ncol = 2)
}


# 2b. Histogram of per-pixel mean RFOP, faceted by run type (condition / policy / benchmark).
# For each dim_type, the mean RFOP is computed by first averaging seeds within each
# scenario (group_label) and then averaging across scenarios for each pixel.
# An optional vertical dashed line shows rfop_cutoff (pass the value from
# attr(sens_df, "rfop_cutoff") or NULL to omit).
#
# runs_df    : bind_rows() of load_run_rfop() outputs
# rfop_cutoff: actual RFOP value (%) to show as dashed threshold line, or NULL
# binwidth   : histogram bin width in RFOP percentage points
make_rfop_histogram <- function(
    runs_df,
    rfop_cutoff = NULL,
    binwidth    = 5
) {
  dim_labels <- c(
    indicator = "Condition (indicator)",
    policy    = "Policy",
    benchmark = "Benchmark"
  )

  plot_df <- runs_df |>
    dplyr::filter(dim_type %in% names(dim_labels)) |>
    # Step 1: average seeds within each scenario
    dplyr::group_by(x, y, dim_type, group_label) |>
    dplyr::summarise(scenario_mean = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    # Step 2: average scenarios within each dim_type
    dplyr::group_by(x, y, dim_type) |>
    dplyr::summarise(mean_rfop_dim = mean(scenario_mean, na.rm = TRUE), .groups = "drop") |>
    dplyr::mutate(
      dim_label = factor(dim_labels[dim_type], levels = unname(dim_labels))
    )

  vline_layer <- if (!is.null(rfop_cutoff))
    geom_vline(xintercept = rfop_cutoff, linetype = "dashed", colour = "grey20", linewidth = 0.7)
  else
    NULL

  cutoff_label <- if (!is.null(rfop_cutoff))
    paste0(" Dashed line: high-priority cutoff (", round(rfop_cutoff, 1), "%).")
  else ""

  ggplot(plot_df, aes(x = mean_rfop_dim, fill = dim_label)) +
    geom_histogram(binwidth = binwidth, colour = "white", linewidth = 0.2) +
    vline_layer +
    scale_fill_brewer(palette = "Set2", guide = "none") +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
    facet_wrap(~dim_label, ncol = 1, scales = "free_y") +
    labs(
      title    = "Distribution of mean RFOP by run type",
      subtitle = paste0("Each pixel's mean RFOP averaged within each epistemic dimension.",
                        cutoff_label),
      x        = "Mean RFOP (%)",
      y        = "Pixel count"
    ) +
    theme_minimal() +
    theme(
      legend.position  = "none",
      panel.grid.minor = element_blank(),
      strip.text       = element_text(size = 9, face = "bold")
    )
}


# 3. Scatter plot: mean_RFOP (x) x mean_seed_SD (y), coloured by dominant_sensitivity.
# Vertical line at rfop_cutoff (extracted from attr(sens_df, "rfop_cutoff") if not supplied);
# horizontal line at seed_sd_threshold.
make_stability_scatter <- function(
    sens_df,
    rfop_cutoff       = NULL,
    seed_sd_threshold = 5,
    max_points        = 10000
) {
  if (is.null(rfop_cutoff)) rfop_cutoff <- attr(sens_df, "rfop_cutoff")

  plot_df <- sens_df |>
    dplyr::filter(!is.na(mean_RFOP), !is.na(mean_seed_SD))

  if (nrow(plot_df) > max_points)
    plot_df <- dplyr::slice_sample(plot_df, n = max_points)

  vline_layer <- if (!is.null(rfop_cutoff))
    geom_vline(xintercept = rfop_cutoff, linetype = "dashed", colour = "grey30", linewidth = 0.7)
  else NULL

  cutoff_text <- if (!is.null(rfop_cutoff))
    paste0("high-priority\ncutoff (", round(rfop_cutoff, 1), "%)")
  else NULL

  p <- ggplot(plot_df,
         aes(x = mean_RFOP, y = mean_seed_SD, colour = dominant_sensitivity)) +
    geom_point(alpha = 0.35, size = 0.7) +
    vline_layer +
    geom_hline(yintercept = seed_sd_threshold,
               linetype = "dashed", colour = "grey30", linewidth = 0.7) +
    scale_colour_manual(
      values = .dominance_palette,
      name   = "Dominant\nsensitivity",
      na.value = "grey70"
    )

  if (!is.null(rfop_cutoff) && !is.null(cutoff_text))
    p <- p + annotate("text", x = rfop_cutoff + 1, y = max(plot_df$mean_seed_SD, na.rm = TRUE),
             label = cutoff_text,
             hjust = 0, vjust = 1, size = 3, colour = "grey30")

  p +
    annotate("text", x = 0, y = seed_sd_threshold + 0.3,
             label = paste0("seed-noise reference (", seed_sd_threshold, " %)"),
             hjust = 0, vjust = 0, size = 3, colour = "grey30") +
    labs(
      x        = "Mean RFOP (%)",
      y        = "Mean seed noise - SE of scenario mean (%)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "right",
      panel.grid.minor = element_blank()
    )
}


# 4. Stacked bar: for each dominant_sensitivity class, proportion seed stable vs uncertain.
make_seed_stability_bar <- function(sens_df) {
  bar_df <- sens_df |>
    dplyr::count(dominant_sensitivity, seed_status) |>
    dplyr::group_by(dominant_sensitivity) |>
    dplyr::mutate(prop = n / sum(n), total = sum(n)) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      dominant_sensitivity = factor(dominant_sensitivity,
                                    levels = names(.dominance_palette)),
      seed_status = factor(seed_status, levels = c("seed stable", "seed uncertain"))
    )

  ggplot(bar_df,
         aes(x = dominant_sensitivity, y = prop, fill = seed_status)) +
    geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
    geom_text(
      data = dplyr::distinct(bar_df, dominant_sensitivity, total),
      aes(x = dominant_sensitivity, y = 1.04, label = scales::comma(total), fill = NULL),
      size = 3, colour = "grey30", vjust = 0
    ) +
    scale_fill_manual(
      values = c("seed stable" = "#3a9fbd", "seed uncertain" = "#c94040"),
      name   = NULL
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      limits = c(0, 1.12),
      expand = c(0, 0)
    ) +
    labs(
      title    = "Seed stability within each sensitivity class",
      subtitle = "Numbers above bars = pixel count",
      x        = NULL,
      y        = "Proportion of pixels"
    ) +
    theme_minimal() +
    theme(
      axis.text.x        = element_text(angle = 30, hjust = 1, size = 9),
      panel.grid.major.x = element_blank()
    )
}


# 5. Stacked bar: within high-priority pixels, proportion assigned to each map_class.
make_high_priority_breakdown_bar <- function(sens_df) {
  hp_df <- sens_df |>
    dplyr::filter(high_priority) |>
    dplyr::count(map_class) |>
    dplyr::mutate(
      prop      = n / sum(n),
      map_class = factor(map_class, levels = levels(sens_df$map_class))
    )

  total_hp <- sum(hp_df$n)

  rfop_cutoff <- attr(sens_df, "rfop_cutoff")
  cutoff_str  <- if (!is.null(rfop_cutoff))
    sprintf(" (mean RFOP \u2265 %.1f%%)", rfop_cutoff)
  else ""

  ggplot(hp_df, aes(x = "", y = prop, fill = map_class)) +
    geom_col(width = 0.6, colour = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(prop >= 0.02,
                                 paste0(round(prop * 100, 1), "%"), "")),
              position = position_stack(vjust = 0.5),
              size = 3.2, colour = "white", fontface = "bold") +
    scale_fill_manual(values = .sensitivity_map_palette, name = NULL, drop = FALSE) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), expand = c(0, 0)) +
    labs(
      title    = "Classification of high-priority pixels",
      subtitle = sprintf("n = %s high-priority pixels%s",
                         scales::comma(total_hp), cutoff_str),
      x        = NULL,
      y        = "Proportion"
    ) +
    theme_minimal() +
    theme(
      axis.text.x        = element_blank(),
      axis.ticks.x       = element_blank(),
      panel.grid.major.x = element_blank()
    )
}


# -- DEPRECATED - replaced by compute_rfop_sensitivity() and per-dim SNRs -----
# "Signal" = SD of scenario-mean RFOPs across indicator scenarios (how much
#   pixel selection shifts when indicator assumptions change, after averaging
#   out within-scenario seed noise).
# "Noise"  = pooled within-scenario SD of RFOP across seeds (how much
#   selection varies just from re-running the same scenario with a new seed).
# "SNR"    = signal / (noise + 1)  - +1 avoids inflation at very low noise.
#
# SNR > 1 : condition signal exceeds algorithm noise -> claims are reliable.
# SNR < 1 : seed noise dominates  -> more seeds needed before firm conclusions.
#
# runs_df : runs_all_df (must contain indicator and seed dim_type rows)
# Returns data frame: x, y, signal, noise, snr
compute_seed_snr <- function(runs_df) {
  stopifnot(all(c("x", "y", "rfop_pct", "dim_type", "group_label") %in% names(runs_df)))

  # Signal: SD of scenario-mean RFOP across indicator scenarios
  signal_df <- runs_df |>
    dplyr::filter(dim_type == "indicator") |>
    dplyr::group_by(x, y, group_label) |>
    dplyr::summarise(scenario_mean = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(signal = sd(scenario_mean, na.rm = TRUE), .groups = "drop")

  # Noise: pooled within-scenario seed SD across indicator scenarios
  noise_df <- runs_df |>
    dplyr::filter(dim_type == "indicator") |>
    dplyr::group_by(x, y, group_label) |>
    dplyr::summarise(within_sd = sd(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(noise = mean(within_sd, na.rm = TRUE), .groups = "drop")

  signal_df |>
    dplyr::left_join(noise_df, by = c("x", "y")) |>
    dplyr::mutate(
      signal = dplyr::coalesce(signal, 0),
      noise  = dplyr::coalesce(noise,  0),
      snr    = signal / (noise + 1)
    )
}


# DEPRECATED - replaced by make_snr_diagnostic_panels().
# Map of SNR values with a diverging palette centred on 1 (signal = noise).
# Grey underlay = eligible pixels not appearing in indicator runs.
#
# snr_df  : output of compute_seed_snr()
# elig_df : eligible pixels data frame for grey underlay (can be NULL)
make_snr_map <- function(snr_df, elig_df = NULL,
                         title = "Condition signal vs algorithm noise (SNR)") {
  plot_df <- snr_df |>
    dplyr::mutate(snr_capped = pmin(snr, 4))  # cap display at 4 for readability

  p <- ggplot()
  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#e6e6e6")
  }
  p +
    geom_raster(data = plot_df, aes(x = x, y = y, fill = snr_capped)) +
    scale_fill_gradient2(
      low      = "#d73027",
      mid      = "#ffffbf",
      high     = "#1a9850",
      midpoint = 1,
      limits   = c(0, 4),
      oob      = scales::squish,
      name     = "SNR\n(signal/noise)",
      labels   = c("0", "1\n(equal)", "2", "3", "4+")
    ) +
    coord_equal() +
    theme_sp +
    labs(
      title    = title,
      subtitle = "SNR > 1 (green): condition signal detectable. SNR < 1 (red): seed noise dominates."
    )
}

.class_palette <- c(
  "Robust priority"      = "#0c7b85",   # deep teal
  "Indicator sensitive"  = "#bd8c12",   # vivid amber
  "Benchmark sensitive"  = "#da3114",   # warm orange
  "Policy sensitive"     = "#910b9b",   # strong violet
  "Low priority"         = "#bec2bf"    # off-white / cream
)


# Plot a categorical raster map of the spatial classification.
# Styling mirrors make_sel_freq_plot() (theme_void, grey eligible underlay, BE boundary).
#
# class_df  : output of compute_spatial_classification()
# elig_df   : eligible_pixels.csv data frame for grey underlay (can be NULL)
# title     : plot title
make_classification_map <- function(class_df, elig_df = NULL,
                                    title = NULL) {
  # Grey underlay for all eligible pixels not classified as "Low priority"
  # (Low priority pixels are already rendered in grey, so we only need the
  # non-selected eligible pixels as background context)
  p <- ggplot() +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position  = "right",
      legend.title     = element_text(size = 10),
      legend.text      = element_text(size = 11)
    )

  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    elig_unclassified <- dplyr::anti_join(elig_df[, c("x", "y")], class_df, by = c("x", "y"))
    if (nrow(elig_unclassified) > 0)
      p <- p + geom_raster(data = elig_unclassified, aes(x = x, y = y), fill = "#EEEEEE")
  }

  p <- p +
    geom_raster(data = class_df, aes(x = x, y = y, fill = classification)) +
    scale_fill_manual(
      values = .class_palette,
      name   = NULL,
      drop   = FALSE,
      guide  = guide_legend(
        override.aes  = list(size = 4),
        keywidth      = unit(0.8, "cm"),
        keyheight     = unit(0.4, "cm"),
        label.hjust   = 0
      )
    )

  if (exists("BE")) {
    p <- p + geom_sf(data = BE, fill = NA, color = "black",
                     linewidth = 0.4, inherit.aes = FALSE)
  }

  p
}


# Structured 2x3 grid of per-dimension classification maps.
#
# Row 1 - sensitivity maps: one panel per epistemic dimension showing only the
#   pixels classified as sensitive to that dimension (in its palette colour).
# Row 2 - dimension-specific robustness maps: pixels that are high-RFOP AND
#   not sensitive to *that* dimension (regardless of the other two), shown in
#   the Robust priority teal. These differ from the global "Robust priority"
#   class (which requires low sensitivity on ALL dimensions simultaneously).
#
# class_df     : output of compute_spatial_classification() - must contain
#                mean_rfop_cond, nsens_ind, nsens_bench, nsens_pol, classification
# elig_df      : optional eligible-pixel data frame (x, y) for the grey underlay
# thresh_high  : RFOP threshold above which a pixel can be considered robust
# thresh_stable: normalised sensitivity threshold (same value used in classification)
make_classification_maps_per_class <- function(
    class_df,
    elig_df      = NULL,
    thresh_high  = 60,
    thresh_stable = 0.3
) {
  other_grey <- "#e4e4e4b9"

  # Helper: build one spatial panel
  .one_map <- function(focal_df, fill_col, title_txt) {
    p <- ggplot() +
      theme_void() +
      theme(
        panel.background = element_rect(fill = "white", colour = NA),
        legend.position  = "none"
      )

    # 1. All eligible pixels - lightest grey underlay
    if (!is.null(elig_df) && nrow(elig_df) > 0)
      p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#EEEEEE")

    # 2. Non-focal classified pixels - mid grey
    non_focal <- dplyr::anti_join(class_df[, c("x", "y")], focal_df[, c("x", "y")],
                                  by = c("x", "y"))
    if (nrow(non_focal) > 0)
      p <- p + geom_raster(data = non_focal, aes(x = x, y = y), fill = other_grey)

    # 3. Focal pixels - class colour
    if (nrow(focal_df) > 0)
      p <- p + geom_raster(data = focal_df, aes(x = x, y = y), fill = fill_col)

    # 4. Canton boundary
    if (exists("BE"))
      p <- p + geom_sf(data = BE, fill = NA, color = "black",
                       linewidth = 0.2, inherit.aes = FALSE)
    p
  }

  # -- Row 1: sensitive pixels per dimension ----------------------------------
  dims <- list(
    list(cls  = "Indicator sensitive",
         nsens = "nsens_ind",
         col  = .class_palette[["Indicator sensitive"]],
         title = "Indicator sensitive"),
    list(cls  = "Benchmark sensitive",
         nsens = "nsens_bench",
         col  = .class_palette[["Benchmark sensitive"]],
         title = "Benchmark sensitive"),
    list(cls  = "Policy sensitive",
         nsens = "nsens_pol",
         col  = .class_palette[["Policy sensitive"]],
         title = "Policy sensitive")
  )

  row1 <- lapply(dims, function(d) {
    if (!d$nsens %in% names(class_df)) {
      return(ggplot() + theme_void() +
               labs(title = paste0(d$title, "\n(no data)")))
    }
    focal <- dplyr::filter(class_df, .data[[d$nsens]] >= thresh_stable)
    .one_map(focal, d$col, d$title)
  })

  # -- Row 2: dimension-specific robust pixels --------------------------------
  # "Robust vs [dim]" = high RFOP AND not sensitive on that specific dimension.
  # This is looser than the global Robust class (which requires ALL dims stable).
  robust_col <- .class_palette[["Robust priority"]]

  row2 <- lapply(dims, function(d) {
    if (!d$nsens %in% names(class_df)) {
      # Dimension has no data - empty placeholder
      return(ggplot() + theme_void() +
               labs(title = paste0("Robust | ", d$title, "\n(no data)")))
    }
    focal <- dplyr::filter(
      class_df,
      mean_rfop_cond >= thresh_high,
      .data[[d$nsens]] < thresh_stable
    )
    .one_map(focal, robust_col, paste0("Robust | ", d$title))
  })

  patchwork::wrap_plots(c(row1, row2), ncol = 3)
}


# Faceted heatmaps of raw sensitivity scores per dimension.
# Useful for diagnosing which dimension drives variability where.
# class_df : output of compute_spatial_classification()
# dims     : which sensitivity dimensions to show (subset if some have no runs)
make_sensitivity_breakdown_maps <- function(
    class_df,
    dims = c("indicator", "policy", "seed", "param")
) {
  dim_labels <- c(
    indicator = "Condition / indicator sensitivity",
    policy    = "Policy sensitivity",
    seed      = "Stochastic / seed sensitivity",
    param     = "Parameter sensitivity"
  )

  available_dims <- intersect(dims, names(dim_labels))

  plot_data <- dplyr::bind_rows(lapply(available_dims, function(d) {
    col <- paste0("sens_", d)
    if (!col %in% names(class_df)) return(NULL)
    data.frame(
      x         = class_df$x,
      y         = class_df$y,
      sens      = class_df[[col]],
      dimension = dim_labels[[d]]
    )
  }))

  if (nrow(plot_data) == 0) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "No sensitivity data available", size = 5) +
             theme_void())
  }

  plot_data$dimension <- factor(plot_data$dimension,
                                levels = dim_labels[available_dims])

  plots <- plot_data %>%
  filter(sens>0) %>%
  split(.$dimension) %>%
  lapply(function(df) {
    ggplot(df, aes(x = x, y = y, fill = sens)) +
      geom_raster() +
      scale_fill_viridis_c(
        option    = "viridis",
        direction = 1,
        name      = "Sensitivity\n(RFOP range)",
        limits    = c(0, NA)
      ) +
      coord_equal() +
      theme_void() +
      ggtitle(unique(df$dimension)) +
      theme(
        plot.title        = element_text(size = 9, face = "bold", hjust = 0.5),
        panel.background  = element_rect(fill = "white", colour = NA),
        legend.position   = "right"
      )
  })

  combined_plot <- wrap_plots(plots, ncol = 3)
  combined_plot
}


# Bar chart showing the proportion of eligible pixels in each priority class.
# Optionally stacks by a grouping variable (e.g. LULC class).
#
# class_df : output of compute_spatial_classification()
make_classification_summary_bar <- function(class_df) {
  summary_df <- class_df |>
    dplyr::count(classification, name = "n_pixels") |>
    dplyr::mutate(pct = n_pixels / sum(n_pixels) * 100)

  ggplot(summary_df, aes(x = classification, y = pct, fill = classification)) +
    geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.1f%%", pct)),
              vjust = -0.4, size = 3.5) +
    scale_fill_manual(values = .class_palette, guide = "none") +
    scale_y_continuous(
      limits = c(0, max(summary_df$pct) * 1.15),
      labels = scales::percent_format(scale = 1, accuracy = 1)
    ) +
    labs(
      x = NULL,
      y = "% of eligible pixels",
      title = "Priority class distribution"
    ) +
    theme_minimal() +
    theme(
      axis.text.x  = element_text(angle = 25, hjust = 1, size = 9),
      panel.grid.major.x = element_blank()
    )
}


# -- Pareto front sensitivity analysis ----------------------------------------
#
# Two complementary summaries of how Pareto front *quality* varies across
# sensitivity dimensions (condition scenarios, policy variants, seeds):
#
#   1. collect_pareto_stats()     - extract HV + ideal/nadir per run
#   2. make_hv_sensitivity_plot() - HV dot plot grouped by dimension
#   3. make_ideal_nadir_plot()    - heatmap of ideal-point shift per objective


# Extract hypervolume, ideal point, and nadir from a named list of run objects.
#
# runs_meta : named list, each entry is a list with:
#               $run        - a run object from load_run_data()
#               $dim_type   - "indicator" | "policy" | "seed" | "param"
#               $group_label - e.g. "global_all", "drop_smd", "with_bs"
# obj_names : character vector of objective column names
# Returns: data frame with one row per run
collect_pareto_stats <- function(runs_meta, obj_names) {
  rows <- lapply(names(runs_meta), function(nm) {
    m   <- runs_meta[[nm]]
    run <- m$run

    hv <- tryCatch({
      if (!is.null(run$df_hv) && nrow(run$df_hv) > 0)
        tail(run$df_hv$hypervolume, 1)
      else NA_real_
    }, error = function(e) NA_real_)

    nd <- tryCatch({
      if (!is.null(run$df_obj))
        run$df_obj[run$df_obj$is_nondominated == 1, obj_names, drop = FALSE]
      else NULL
    }, error = function(e) NULL)

    ideal_row <- if (!is.null(nd) && nrow(nd) > 0)
      setNames(as.list(apply(nd, 2, min, na.rm = TRUE)), paste0("ideal_", obj_names))
    else
      setNames(as.list(rep(NA_real_, length(obj_names))), paste0("ideal_", obj_names))

    nadir_row <- if (!is.null(nd) && nrow(nd) > 0)
      setNames(as.list(apply(nd, 2, max, na.rm = TRUE)), paste0("nadir_", obj_names))
    else
      setNames(as.list(rep(NA_real_, length(obj_names))), paste0("nadir_", obj_names))

    c(list(
      run_label   = nm,
      dim_type    = m$dim_type,
      group_label = m$group_label,
      n_nondom    = if (!is.null(nd)) nrow(nd) else NA_integer_,
      hypervolume = hv
    ), ideal_row, nadir_row)
  })

  dplyr::bind_rows(lapply(rows, as.data.frame))
}


# -- Cross-run-comparable hypervolume (shared reference point) ----------------
# The per-run HV exported from the optimiser uses a per-run reference point and
# per-run objective normalisation, so HV is NOT comparable across scenarios.
# This recomputes HV on the RAW objectives of each run's non-dominated front
# against a SINGLE shared reference box (the pooled worst corner), making HV
# directly comparable. Dimension-agnostic Monte-Carlo estimate (no HV package
# required). Only pass runs whose objectives share the same definition/units
# (e.g. exclude the threshold-formulation runs).
#
#   runs_meta : named list; each element has $run = load_run_data() output
#   obj_names : objective columns to use (raw, as in df_obj)
# Returns a tibble: run_label, hv (shared-ref HV), n_nondom.
# Shared-reference Monte Carlo hypervolume so HV is comparable across runs.
#
#   runs_meta : named list of list(run = load_run_data(...)) keyed by run_label.
#   obj_names : objective columns; all are oriented to 'benefit' (larger = better).
#   groups    : OPTIONAL named vector mapping run_label -> comparability group.
#               A SEPARATE shared reference box is built per group, and HV is only
#               comparable WITHIN a group. Use this when an objective's units are
#               not shared across all runs - e.g. the `scaling` factor rescales
#               restoration_potential by ~100x (global ~1e3 vs upper_q75 ~1e5), so
#               pooling both into one box pins the global runs' rp near the box
#               floor and crushes their HV. NB: per-objective min-max normalising
#               the pooled matrix would NOT fix this - dominated-HV fraction is
#               invariant to independent positive affine rescaling of each axis,
#               so relative HV is unchanged. Splitting the reference by group is
#               the actual fix. NULL (default) = one shared box over all runs
#               (original behaviour). Runs with no entry in `groups` fall to "all".
# Returns tibble(run_label, hv, n_nondom, hv_group).
compute_shared_hv <- function(runs_meta, obj_names, n_mc = 2e4, margin = 0.05, seed = 42,
                              groups = NULL) {
  # Orient all objectives to a 'benefit' (larger = better) frame. load_run_data()
  # already flips rp/lc/anomaly/connectivity to benefit; negate raw minimise
  # objectives (cost, robustness) so every column is benefit-oriented.
  neg <- grepl("cost|robustness", obj_names)
  .benefit <- function(df) {
    M <- as.matrix(df[, obj_names, drop = FALSE])
    if (any(neg)) M[, neg] <- -M[, neg]
    M
  }
  mats <- lapply(runs_meta, function(m) {
    run <- m$run
    if (is.null(run) || is.null(run$df_obj)) return(NULL)
    df <- run$df_obj
    if ("is_nondominated" %in% names(df))
      df <- df[df$is_nondominated == 1, , drop = FALSE]
    if (nrow(df) == 0L || !all(obj_names %in% names(df))) return(NULL)
    .benefit(df)
  })
  mats <- mats[!vapply(mats, is.null, logical(1))]
  if (length(mats) == 0L) return(tibble::tibble())

  # Assign each surviving run to a comparability group (default: one box for all).
  if (is.null(groups)) {
    grp <- stats::setNames(rep("all", length(mats)), names(mats))
  } else {
    grp <- groups[names(mats)]
    grp[is.na(grp)] <- "all"
    names(grp) <- names(mats)
  }

  k <- length(obj_names)
  set.seed(seed)

  # One shared reference box + Monte Carlo sample per group; HV per run within it.
  hv_for_group <- function(gl) {
    gmats  <- mats[gl]
    pooled <- do.call(rbind, gmats)
    U   <- apply(pooled, 2, max)                 # ideal (best) corner
    L   <- apply(pooled, 2, min)
    rng <- pmax(U - L, 1e-12)
    R   <- L - margin * rng                       # group reference (worst corner)
    box_vol <- prod(U - R)

    S <- matrix(runif(n_mc * k), ncol = k)
    S <- sweep(sweep(S, 2, (U - R), `*`), 2, R, `+`)   # uniform in [R, U]

    hv_of <- function(M) {
      dom <- logical(nrow(S))
      for (i in seq_len(nrow(M))) {
        inside <- rep(TRUE, nrow(S))
        for (j in seq_len(k)) inside <- inside & (S[, j] <= M[i, j])
        dom <- dom | inside
      }
      mean(dom) * box_vol
    }
    tibble::tibble(
      run_label = gl,
      hv        = vapply(gmats, hv_of, numeric(1)),
      n_nondom  = vapply(gmats, nrow, integer(1)),
      hv_group  = grp[gl][[1]]
    )
  }

  dplyr::bind_rows(lapply(split(names(grp), grp), hv_for_group))
}


# -- Cross-scenario agreement of top-X% priority areas ------------------------
# Tests directly whether a robust priority core exists: for each scenario, take
# its top `top_pct`% RFOP pixels as its "priority set", then measure how much
# those sets agree across scenarios.
#
#   runs_df : load_run_rfop() rows (seed-averaged per scenario internally)
#   top_pct : percentage of pixels kept as each scenario's priority set
#   dims    : dim_types treated as scenarios to compare (default condition axes)
# Returns a list:
#   jaccard      : tibble(a, b, jaccard) of pairwise top-set overlap
#   consensus    : tibble(x, y, n_top, n_scen, frac) - how many scenarios rank a
#                  pixel in their top set (frac = n_top / n_scen)
#   mean_jaccard : mean pairwise Jaccard; n_scenarios; top_pct
compute_topx_overlap <- function(runs_df, top_pct = 15,
                                 dims = c("indicator", "benchmark")) {
  sm <- runs_df |>
    dplyr::filter(dim_type %in% dims) |>
    dplyr::group_by(group_label, x, y) |>
    dplyr::summarise(rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

  scen   <- sort(unique(sm$group_label))
  n_scen <- length(scen)

  top_df <- sm |>
    dplyr::group_by(group_label) |>
    dplyr::mutate(thr = stats::quantile(rfop, 1 - top_pct / 100, na.rm = TRUE)) |>
    dplyr::filter(rfop >= thr) |>
    dplyr::ungroup() |>
    dplyr::select(group_label, x, y)

  consensus <- top_df |>
    dplyr::count(x, y, name = "n_top") |>
    dplyr::mutate(n_scen = n_scen, frac = n_top / n_scen)

  sets    <- split(paste(top_df$x, top_df$y), top_df$group_label)
  jaccard <- if (n_scen >= 2) {
    pairs <- utils::combn(scen, 2, simplify = FALSE)
    do.call(rbind, lapply(pairs, function(p) {
      a <- sets[[p[1]]]; b <- sets[[p[2]]]
      uni <- length(union(a, b))
      data.frame(a = p[1], b = p[2],
                 jaccard = if (uni > 0) length(intersect(a, b)) / uni else NA_real_)
    }))
  } else data.frame(a = character(0), b = character(0), jaccard = numeric(0))

  list(jaccard      = tibble::as_tibble(jaccard),
       consensus    = tibble::as_tibble(consensus),
       mean_jaccard = mean(jaccard$jaccard, na.rm = TRUE),
       n_scenarios  = n_scen,
       top_pct      = top_pct)
}


# Dot plot of hypervolume across runs, faceted by sensitivity dimension.
# The reference run (e.g. global_all) is highlighted as a dashed line.
#
# stats_df    : output of collect_pareto_stats()
# ref_label   : run_label of the reference run to draw as baseline
make_hv_sensitivity_plot <- function(stats_df, ref_label = NULL) {
  df <- dplyr::filter(stats_df, !is.na(hypervolume))
  if (nrow(df) == 0) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "No hypervolume data available", size = 5) +
             theme_void())
  }

  # Reference HV for baseline line
  ref_hv <- if (!is.null(ref_label) && ref_label %in% df$run_label)
    df$hypervolume[df$run_label == ref_label][[1]]
  else NULL

  # Shared lollipop panel builder (x_col is a column name string)
  .hv_panel <- function(data, x_col, title) {
    y_min <- min(data$hypervolume, na.rm = TRUE)
    p <- ggplot(data, aes(x = reorder(.data[[x_col]], hypervolume),
                           y = hypervolume)) +
      geom_segment(aes(xend = reorder(.data[[x_col]], hypervolume),
                       yend = y_min * 0.999),
                   linewidth = 0.5, alpha = 0.5, colour = "steelblue") +
      geom_point(size = 3.5, colour = "steelblue") +
      scale_y_continuous(labels = scales::scientific) +
      coord_flip() +
      labs(x = NULL, y = "Final hypervolume", title = title) +
      theme_minimal() +
      theme(panel.grid.major.y = element_blank())
    if (!is.null(ref_hv))
      p <- p + geom_hline(yintercept = ref_hv, linetype = "dashed",
                           colour = "grey40", linewidth = 0.7)
    p
  }

  # -- Indicator panel -------------------------------------------------------
  # One point per drop-one scenario (group_label); multiple seed runs overlap
  # at the same x position, which is fine for a quick spread check.
  df_ind <- dplyr::filter(df, dim_type == "indicator")
  p_ind <- if (nrow(df_ind) > 0)
    .hv_panel(df_ind, "group_label", "Condition / indicator")
  else NULL

  # -- Policy panel ----------------------------------------------------------
  # Shows upper_q75_all (policy variant) alongside global_all (seed dimension)
  # as the baseline group so both scenarios appear in the same panel.
  df_pol <- dplyr::bind_rows(
    dplyr::filter(df, dim_type == "policy") |>
      dplyr::mutate(pol_label = group_label),
    dplyr::filter(df, dim_type == "seed") |>
      dplyr::mutate(pol_label = "global_all (baseline)")
  )
  p_pol <- if (nrow(df_pol) > 0)
    .hv_panel(df_pol, "pol_label", "Policy")
  else NULL

  # -- Seed panel ------------------------------------------------------------
  # Labels each run by its seed number extracted from run_label
  df_seed <- dplyr::filter(df, dim_type == "seed") |>
    dplyr::mutate(seed_label = sub(".*_seed", "seed", run_label))
  p_seed <- if (nrow(df_seed) > 0)
    .hv_panel(df_seed, "seed_label", "Seed (stochastic)")
  else NULL

  plots <- Filter(Negate(is.null), list(p_ind, p_pol, p_seed))
  if (length(plots) == 0) return(ggplot() + theme_void())

  patchwork::wrap_plots(plots, ncol = 1) +
    patchwork::plot_annotation(
      title    = "Pareto front quality across sensitivity dimensions",
      subtitle = "Each point is one run; dashed line = reference HV (global_all_seed100)"
    )
}


# Dumbbell plot combining ideal-point and nadir shift in one figure.
#
# For each run x objective, a segment connects the ideal-point % change (filled
# circle, best achievable) to the nadir % change (open circle, worst ND value).
# Segment length = Pareto front width change; position = overall shift direction.
#
# Three stacked panels (indicator / policy / seed), each faceted by objective.
# The zero line marks no change vs the reference run.
#
# stats_df    : output of collect_pareto_stats()
# obj_names   : character vector of objective column names
# obj_labels  : display labels (same order as obj_names)
# ref_label   : run_label of the reference run
make_ideal_nadir_dumbbell <- function(stats_df, obj_names, obj_labels,
                                      ref_label = NULL) {
  ideal_cols <- paste0("ideal_", obj_names)
  nadir_cols <- paste0("nadir_", obj_names)

  if (!all(c(ideal_cols, nadir_cols) %in% names(stats_df))) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "ideal_ or nadir_ columns missing from stats_df", size = 4) +
             theme_void())
  }

  # -- Helper: pivot one point type and compute % change vs reference ---------
  .pct_long <- function(cols, point_type) {
    stats_df |>
      dplyr::select(run_label, dim_type, group_label, dplyr::all_of(cols)) |>
      tidyr::pivot_longer(cols = dplyr::all_of(cols),
                          names_to = "objective", values_to = "value") |>
      dplyr::mutate(
        objective  = factor(gsub(paste0("^", point_type, "_"), "", objective),
                             levels = obj_names, labels = obj_labels),
        point_type = point_type
      )
  }

  long <- dplyr::bind_rows(.pct_long(ideal_cols, "ideal"),
                            .pct_long(nadir_cols, "nadir"))

  if (!is.null(ref_label) && ref_label %in% long$run_label) {
    ref_vals <- long |>
      dplyr::filter(run_label == ref_label) |>
      dplyr::select(objective, point_type, ref_value = value)
    long <- dplyr::left_join(long, ref_vals, by = c("objective", "point_type")) |>
      dplyr::mutate(pct_change = (value - ref_value) / (abs(ref_value) + 1e-10) * 100)
  } else {
    long <- dplyr::mutate(long, pct_change = value)
  }

  # -- Widen back to ideal / nadir columns for segment drawing ---------------
  wide <- long |>
    dplyr::select(run_label, dim_type, group_label, objective, point_type, pct_change) |>
    tidyr::pivot_wider(names_from = point_type, values_from = pct_change)

  # -- Row labels per dimension -----------------------------------------------
  ind_wide  <- dplyr::filter(wide, dim_type == "indicator") |>
    dplyr::group_by(group_label, objective) |>
    dplyr::summarise(ideal = mean(ideal, na.rm = TRUE),
                     nadir = mean(nadir, na.rm = TRUE), .groups = "drop") |>
    dplyr::mutate(row_label = group_label)

  pol_wide  <- dplyr::filter(wide, dim_type %in% c("policy", "seed")) |>
    dplyr::mutate(row_label = run_label)

  seed_wide <- dplyr::filter(wide, dim_type == "seed") |>
    dplyr::mutate(row_label = sub(".*_seed", "seed", run_label))

  # -- Panel builder ----------------------------------------------------------
  .db_panel <- function(data, title) {
    # Combine ideal and nadir into long for points, keep wide for segments
    pts <- dplyr::bind_rows(
      dplyr::mutate(data, pct = ideal, pt = "Ideal (best)"),
      dplyr::mutate(data, pct = nadir, pt = "Nadir (worst ND)")
    )

    ggplot(data, aes(y = reorder(row_label, (ideal + nadir) / 2))) +
      geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.5) +
      geom_segment(aes(x = ideal, xend = nadir,
                       yend = reorder(row_label, (ideal + nadir) / 2)),
                   linewidth = 0.8, colour = "grey50", alpha = 0.7) +
      geom_point(data = pts, aes(x = pct, shape = pt,
                                  colour = pct > 0),
                 size = 3) +
      scale_colour_manual(values = c("FALSE" = "#1976D2", "TRUE" = "#C62828"),
                          guide = "none") +
      scale_shape_manual(values = c("Ideal (best)" = 16, "Nadir (worst ND)" = 1),
                         name = NULL) +
      scale_x_continuous(labels = function(x) paste0(round(x), "%")) +
      facet_wrap(~objective, nrow = 1, scales = "free_x") +
      labs(x = sprintf("%% change from reference (%s)", ref_label),
           y = NULL, title = title) +
      theme_minimal() +
      theme(
        strip.text         = element_text(face = "bold", size = 9),
        panel.grid.major.y = element_blank(),
        panel.grid.minor   = element_blank(),
        axis.text.y        = element_text(size = 8),
        axis.text.x        = element_text(size = 8),
        plot.title         = element_text(face = "bold", size = 10),
        legend.position    = "bottom"
      )
  }

  plots <- list()
  if (nrow(ind_wide)  > 0) plots[["indicator"]] <- .db_panel(ind_wide,  "Condition / indicator sensitivity")
  if (nrow(pol_wide)  > 0) plots[["policy"]]    <- .db_panel(pol_wide,  "Policy sensitivity (global_all vs upper_q75_all)")
  if (nrow(seed_wide) > 0) plots[["seed"]]      <- .db_panel(seed_wide, "Seed (stochastic) sensitivity")

  if (length(plots) == 0)
    return(ggplot() + annotate("text", x = 0.5, y = 0.5, label = "No data", size = 5) + theme_void())

  patchwork::wrap_plots(plots, ncol = 1) +
    patchwork::plot_annotation(
      title    = "Pareto front range shift across sensitivity dimensions",
      subtitle = sprintf(
        "Filled circle = ideal point (best achievable); open circle = nadir (worst ND) | Blue = improved, Red = degraded | ref: %s",
        ref_label)
    )
}


# -- Indicator redundancy analysis ---------------------------------------------
#
# Computes the van den Wollenberg redundancy index per EC indicator across all
# condition scenarios.  With univariate Y (RFOP per eligible pixel), the index
# simplifies to R^2 = cor(rfop, indicator)^2, requiring only base-R cor().
#
# Workflow:
#   1. load_scenario_rfop_avg()     - average RFOP per pixel across seeds
#   2. extract_indicator_values()   - pull indicator raster values at elig. pixels
#   3. compute_redundancy_indices() - R^2 per indicator x scenario x ecosystem
#   4. make_redundancy_boxplot()    - distribution across scenarios per indicator


# Load RFOP for one condition scenario, averaged across multiple seed runs.
#
# dirs_list      : named list of r_inputs/ directories, one entry per seed
# scenario_label : label for this condition scenario (e.g. "global_all", "drop_smd")
# Returns: data frame (x, y, rfop_avg, n_seeds, scenario_label), or NULL
load_scenario_rfop_avg <- function(dirs_list, scenario_label) {
  seed_dfs <- lapply(names(dirs_list), function(nm) {
    dir <- dirs_list[[nm]]
    if (!dir.exists(dir)) {
      message(sprintf("  Skipping missing directory: %s", dir))
      return(NULL)
    }
    load_run_rfop(dir_path    = dir,
                  label       = nm,
                  dim_type    = "scenario",
                  group_label = scenario_label)
  })
  seed_dfs <- Filter(Negate(is.null), seed_dfs)
  if (length(seed_dfs) == 0) {
    message(sprintf("  No valid seed directories for scenario: %s", scenario_label))
    return(NULL)
  }

  dplyr::bind_rows(seed_dfs) |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(
      rfop_avg = mean(rfop_pct, na.rm = TRUE),
      n_seeds  = dplyr::n(),
      .groups  = "drop"
    ) |>
    dplyr::mutate(scenario_label = scenario_label)
}


# Extract raw indicator raster values at eligible pixel locations and assign
# each pixel to an ecosystem based on the LULC raster.
#
# elig_df         : data frame with columns x, y (EPSG:2056 coordinates)
# indicator_paths : named list mapping indicator code to path to .tif file
# lulc_raster     : SpatRaster; LULC class values used to assign ecosystem
# lulc_classes    : named list mapping ecosystem label to integer LULC class codes
# Returns: data frame (x, y, ecosystem, <indicator_code>, ...)
extract_indicator_values <- function(
    elig_df,
    indicator_paths,
    lulc_raster,
    lulc_classes = list(
      Forest       = c(12L, 13L),
      Agricultural = c(15L),
      Grassland    = c(16L, 17L)
    )
) {
  if (nrow(elig_df) == 0 || length(indicator_paths) == 0) return(NULL)

  pts <- terra::vect(
    cbind(elig_df$x, elig_df$y),
    type = "points",
    crs  = "EPSG:2056"
  )

  lulc_vals <- terra::extract(lulc_raster, pts)[, 2, drop = TRUE]
  ecosystem <- dplyr::case_when(
    lulc_vals %in% lulc_classes$Forest       ~ "Forest",
    lulc_vals %in% lulc_classes$Agricultural ~ "Agricultural",
    lulc_vals %in% lulc_classes$Grassland    ~ "Grassland",
    TRUE                                     ~ NA_character_
  )

  ind_df <- data.frame(x = elig_df$x, y = elig_df$y, ecosystem = ecosystem,
                       stringsAsFactors = FALSE)

  for (nm in names(indicator_paths)) {
    path <- indicator_paths[[nm]]
    if (!file.exists(path)) {
      message(sprintf("  Indicator raster not found, skipping: %s", path))
      ind_df[[nm]] <- NA_real_
      next
    }
    r   <- terra::rast(path)
    ext <- terra::extract(r, pts)[, 2, drop = TRUE]
    ind_df[[nm]] <- as.numeric(ext)
  }

  ind_df
}


# Compute van den Wollenberg redundancy index (R^2) for each indicator x
# condition scenario x ecosystem.
#
# all_scenarios_df  : bind_rows of load_scenario_rfop_avg() results
# indicator_vals_df : output of extract_indicator_values()
# indicator_meta    : data frame with cols: code, full_name, ect
# min_pixels        : skip combos with fewer complete cases than this
# Returns: data frame (scenario_label, ecosystem, indicator_code,
#          indicator_label, ect, redundancy_r2, n_pixels)
compute_redundancy_indices <- function(all_scenarios_df,
                                       indicator_vals_df,
                                       indicator_meta,
                                       min_pixels = 50) {
  indicator_codes <- intersect(indicator_meta$code, names(indicator_vals_df))
  if (length(indicator_codes) == 0) {
    warning("No indicator codes found in indicator_vals_df. Check names match indicator_meta$code.")
    return(NULL)
  }

  joined <- dplyr::inner_join(all_scenarios_df, indicator_vals_df, by = c("x", "y"))
  results <- list()

  for (scen in unique(joined$scenario_label)) {
    scen_df <- joined[joined$scenario_label == scen, ]

    for (eco in c("Forest", "Agricultural", "Grassland")) {
      eco_df <- scen_df[!is.na(scen_df$ecosystem) & scen_df$ecosystem == eco, ]
      if (nrow(eco_df) < min_pixels) next

      y <- eco_df$rfop_avg

      for (code in indicator_codes) {
        x_vals      <- eco_df[[code]]
        complete_i  <- !is.na(x_vals) & !is.na(y)
        if (sum(complete_i) < min_pixels) next

        r2 <- cor(y[complete_i], x_vals[complete_i])^2

        meta_row <- indicator_meta[indicator_meta$code == code, ]
        results[[length(results) + 1]] <- data.frame(
          scenario_label  = scen,
          ecosystem       = eco,
          indicator_code  = code,
          indicator_label = if (nrow(meta_row) > 0) meta_row$full_name[[1]] else code,
          ect             = if (nrow(meta_row) > 0) meta_row$ect[[1]]       else "unknown",
          redundancy_r2   = r2,
          n_pixels        = sum(complete_i),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(results) == 0) return(NULL)
  dplyr::bind_rows(results)
}


# Box plot: distribution of R^2 across condition scenarios per indicator.
# Faceted by ecosystem, coloured by ect category (abiotic / biotic).
make_redundancy_boxplot <- function(redundancy_df) {
  if (is.null(redundancy_df) || nrow(redundancy_df) == 0) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "No redundancy data available", size = 5) +
             theme_void())
  }

  # Order by overall median R^2 (descending)
  ind_order <- redundancy_df %>%
    dplyr::group_by(indicator_label) %>%
    dplyr::summarise(med = median(redundancy_r2, na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(med)) %>%
    dplyr::pull(indicator_label)

  redundancy_df$indicator_label <- factor(redundancy_df$indicator_label, levels = ind_order)
  redundancy_df$ecosystem       <- factor(redundancy_df$ecosystem,
                                          levels = c("Forest", "Agricultural", "Grassland"))

  ect_palette <- c(abiotic = "#F57C00", biotic = "#00838F")

  ggplot(redundancy_df, aes(x = indicator_label, y = redundancy_r2, fill = ect)) +
    geom_boxplot(outlier.size = 0.8, width = 0.6, colour = "grey30") +
    scale_fill_manual(values = ect_palette, name = "Category") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, NA)) +
    facet_wrap(~ecosystem, ncol = 2, scales = "free_x") +
    labs(
      x        = NULL,
      y        = expression("Redundancy index (R"^2*")"),
    ) +
    theme_minimal() +
    theme(
      axis.text.x        = element_text(angle = 30, hjust = 1, size = 9),
      strip.text         = element_text(face = "bold", size = 10),
      panel.grid.major.x = element_blank(),
      legend.position    = "bottom"
    )
}


# -- ANOVA variance decomposition ----------------------------------------------
#
# For each eligible pixel, RFOP varies across runs that differ in three
# sensitivity dimensions: indicator assumption, policy assumption, and random
# seed.  An ANOVA decomposes the total variance in RFOP (across all 39 runs)
# into the share attributable to each dimension (eta^2) plus residual.
#
# This gives a comparable effect-size measure across all sensitivity dimensions,
# using only the runs that already exist - no additional model evaluations needed.
#
# Workflow:
#   1. compute_rfop_anova()      - pixel-wise SD of group-mean RFOP per dimension
#   2. make_anova_map()          - spatial map of dominant dimension
#   3. make_anova_eta2_ridges()  - distribution of SD across pixels per dim
#   4. make_anova_summary_bar()  - mean SD per dimension (global summary)


# Pixel-wise variance attribution across sensitivity dimensions.
#
# -- Metric: SD of group-mean RFOP ---------------------------------------------
# For each pixel p and dimension d, we compute the standard deviation of
# group-averaged RFOP across the distinct groups within that dimension:
#
#   group_mean_g(p) = mean(rfop_pct) across all runs in group g
#   rfop_sd_d(p)    = SD of {group_mean_g(p)} across groups g in dimension d
#
# This is in RFOP percentage-point units and is directly comparable across
# dimensions regardless of how many runs each dimension has, and regardless
# of whether groups have 1 run (seeds) or many (indicator scenarios).
#
# For the seed dimension this measures "how much does the choice of random
# seed shift average pixel RFOP?"; for indicator it measures "how much does
# dropping one indicator shift average pixel RFOP?".  Both are on the same
# scale - a pixel with rfop_sd_seed = 20 is just as unstable to seed choice
# as a pixel with rfop_sd_indicator = 20 is to indicator choice.
#
# runs_df : bind_rows of load_run_rfop() - cols x, y, rfop_pct, dim_type, group_label
# dims    : character vector of dim_type values to decompose (default all present)
# Returns a data frame with one row per eligible pixel:
#   x, y, rfop_sd_<dim>, rfop_mean_<dim>, n_groups_<dim>, dominant_dim
compute_rfop_anova <- function(runs_df,
                               dims = c("indicator", "policy", "seed"),
                               se_dims = character(0)) {
  stopifnot(all(c("x", "y", "rfop_pct", "dim_type", "group_label") %in% names(runs_df)))
  # se_dims: dimensions to express as the standard ERROR of their group-mean RFOP
  #   (rfop_sd / sqrt(n_groups)) rather than the raw SD across groups. Use this for
  #   a pure-noise dimension (e.g. "seed") so it is on the same footing as the
  #   seed-averaged signal dimensions; otherwise the seed bar is inflated by ~sqrt(n).

  dims_present <- intersect(dims, unique(runs_df$dim_type))

  result_list <- lapply(dims_present, function(d) {
    runs_df |>
      dplyr::filter(dim_type == d) |>
      # Average within each group to remove within-scenario seed noise
      # (for indicator/policy dims each group already has multiple seeds;
      #  for seed dim each group_label is one seed so this is a no-op mean)
      dplyr::group_by(x, y, group_label) |>
      dplyr::summarise(g_mean = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
      dplyr::group_by(x, y) |>
      dplyr::summarise(
        !!paste0("rfop_sd_",   d) := sd(g_mean,   na.rm = TRUE),
        !!paste0("rfop_mean_", d) := mean(g_mean, na.rm = TRUE),
        !!paste0("n_groups_",  d) := dplyr::n(),
        .groups = "drop"
      )
  })

  result <- Reduce(function(a, b) dplyr::full_join(a, b, by = c("x", "y")),
                   result_list)

  # Express requested dimensions as the standard error of their group-mean RFOP
  # (SD / sqrt(n_groups)) before computing dominance, so a noise dimension is
  # compared fairly against seed-averaged signal dimensions.
  for (d in intersect(se_dims, dims_present)) {
    sd_col <- paste0("rfop_sd_", d)
    n_col  <- paste0("n_groups_", d)
    if (all(c(sd_col, n_col) %in% names(result)))
      result[[sd_col]] <- result[[sd_col]] / sqrt(pmax(result[[n_col]], 1))
  }

  # Dominant dimension: the one with the largest SD of group-mean RFOP
  sd_cols <- paste0("rfop_sd_", dims_present)
  result$dominant_dim <- apply(
    result[, sd_cols, drop = FALSE], 1,
    function(row) {
      if (all(is.na(row) | row == 0)) return(NA_character_)
      dims_present[which.max(row)]
    }
  )

  result
}


# Colour palette for ANOVA dimensions (matches classification palette tone).
.anova_palette <- c(
  indicator  = "#E9C46A",   # amber  - condition/indicator sensitivity
  policy     = "#6A0572",   # violet - policy sensitivity
  seed       = "#E63946",   # red    - stochastic noise
  residual   = "grey70"
)

.anova_labels <- c(
  indicator = "Indicator assumption",
  policy    = "Policy assumption",
  seed      = "Random seed",
  residual  = "Residual"
)


# Spatial map: for each pixel colour by dominant dimension (largest SD of group-mean RFOP).
# Grey underlay = eligible pixels where dominant_dim is NA (zero SD in all dimensions -
# always or never selected regardless of assumptions).
#
# anova_df : output of compute_rfop_anova()
# elig_df  : eligible_pixels.csv data frame for grey underlay (can be NULL)
make_anova_map <- function(anova_df, elig_df = NULL,
                           title = NULL) {
  plot_df <- anova_df |>
    dplyr::filter(!is.na(dominant_dim)) |>
    dplyr::mutate(
      dominant_dim = factor(dominant_dim, levels = names(.anova_palette)),
      label        = .anova_labels[as.character(dominant_dim)]
    )

  p <- ggplot()

  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#e6e6e6")
  }

  p +
    geom_raster(data = plot_df, aes(x = x, y = y, fill = label)) +
    scale_fill_manual(
      values = setNames(.anova_palette[names(.anova_labels)],
                        unname(.anova_labels)),
      name   = "Dominant dimension",
      na.value = "grey85"
    ) +
    coord_equal() +
    theme_sp +
    labs()
}


# Ridge plot: distribution of SD of group-mean RFOP across pixels, one ridge per dimension.
# x-axis is in RFOP percentage-point units - directly comparable across dimensions.
#
# anova_df : output of compute_rfop_anova()
# dims     : which dimensions to include
make_anova_eta2_ridges <- function(anova_df,
                                   dims = c("indicator", "policy", "seed")) {
  dims_present <- intersect(dims, sub("rfop_sd_", "",
                                      grep("^rfop_sd_", names(anova_df), value = TRUE)))
  if (length(dims_present) == 0) return(NULL)

  long_df <- dplyr::bind_rows(lapply(dims_present, function(d) {
    sd_col <- paste0("rfop_sd_", d)
    anova_df |>
      dplyr::select(x, y, rfop_sd = dplyr::all_of(sd_col)) |>
      dplyr::filter(!is.na(rfop_sd)) |>
      dplyr::mutate(
        dimension = d,
        dim_label = factor(.anova_labels[d], levels = unname(.anova_labels[dims_present]))
      )
  }))

  ggplot(long_df, aes(x = rfop_sd, y = dim_label, fill = dimension)) +
    ggridges::geom_density_ridges(
      alpha = 0.75, scale = 1.2,
      quantile_lines = TRUE, quantiles = 0.5,
      colour = "grey30"
    ) +
    scale_fill_manual(values = .anova_palette[dims_present], guide = "none") +
    scale_x_continuous(labels = scales::label_number(suffix = " %"),
                       expand = c(0.01, 0)) +
    labs(
      x        = "SD of group-mean RFOP (%)",
      y        = NULL
    ) +
    theme_minimal() +
    theme(
      panel.grid.major.y = element_blank(),
      strip.text         = element_text(face = "bold")
    )
}


# Summary bar chart: mean SD of group-mean RFOP (+/- SD across pixels) per dimension.
# All bars are on the same scale (RFOP percentage points) - directly comparable.
#
# anova_df : output of compute_rfop_anova()
# dims     : which dimensions to include
make_anova_summary_bar <- function(anova_df,
                                   dims = c("indicator", "policy", "seed")) {
  dims_present <- intersect(dims, sub("rfop_sd_", "",
                                      grep("^rfop_sd_", names(anova_df), value = TRUE)))

  summary_df <- dplyr::bind_rows(lapply(dims_present, function(d) {
    sd_col <- paste0("rfop_sd_", d)
    anova_df |>
      dplyr::select(rfop_sd = dplyr::all_of(sd_col)) |>
      dplyr::filter(!is.na(rfop_sd)) |>
      dplyr::summarise(
        mean_sd = mean(rfop_sd, na.rm = TRUE),
        se_sd   = sd(rfop_sd,   na.rm = TRUE) / sqrt(dplyr::n())
      ) |>
      dplyr::mutate(dimension = d)
  })) |>
    dplyr::mutate(
      dim_label = factor(.anova_labels[dimension],
                         levels = rev(unname(.anova_labels[dims_present])))
    )

  ggplot(summary_df, aes(x = mean_sd, y = dim_label, fill = dimension)) +
    geom_col(width = 0.6, colour = "grey30") +
    geom_errorbar(
      aes(xmin = pmax(0, mean_sd - se_sd),
          xmax = mean_sd + se_sd),
      width = 0.25, colour = "grey30"
    ) +
    scale_fill_manual(values = .anova_palette[dims_present], guide = "none") +
    scale_x_continuous(labels = scales::label_number(suffix = " %"),
                       expand = c(0.01, 0)) +
    labs(
      x        = "Mean SD of group-mean RFOP (%)",
      y        = NULL
    ) +
    theme_minimal() +
    theme(panel.grid.major.y = element_blank())
}


# =============================================================================
# FACTORIAL ATTRIBUTION - crossed design, variance partitioning (iEMSs Block 4)
# -----------------------------------------------------------------------------
# Replaces the one-at-a-time decomposition (compute_rfop_anova, above) with a
# proper crossed-factorial variance partition. For a fully-crossed run matrix
#   form x scaling x construction x policy  (seed-replicated)
# this estimates, per spatial cell, how much of the variation in per-cell
# selection frequency (RFOP) is attributable to each formulation factor's MAIN
# effect, to their INTERACTIONS, and to the RESIDUAL (seed) noise floor.
#
# Pipeline:
#   1. load_run_factorial()            - load one run + its factor levels from metadata
#   2. compute_rfop_variance_partition() - per-cell aov(); SS : eta^2 per term
#   3. make_variance_partition_bar()   - global variance-share bar (main/interaction/residual)
#   4. make_dominant_factor_map()      - per-cell dominant source map
# =============================================================================

# Colour palette / labels for factorial sources. Order here also sets bar/legend order.
# Okabe-Ito palette (colorblind-safe under deuteranopia, protanopia, tritanopia).
.factor_palette <- c(
  form         = "#0072B2",   # blue        - objective target form (sum vs threshold)
  scaling      = "#E69F00",   # orange      - condition scaling / reference (anomaly vs q75)
  construction = "#009E73",   # green       - condition-indicator construction
  policy       = "#CC79A7",   # pink-purple - policy / governance lever
  interaction  = "#D55E00",   # vermillion  - pooled interactions
  residual     = "#999999"    # grey        - residual (seed) noise floor
)
.factor_labels <- c(
  form         = "Objective form",
  scaling      = "Condition scaling",
  construction = "Indicator construction",
  policy       = "Policy",
  interaction  = "Interactions",
  residual     = "Residual (seed)"
)

# Load one factorial run's per-cell RFOP plus its factor levels (read from
# metadata.json, written by run_custom.py's factorial mode as flat factor_* keys
# in run_config). Returns a data frame:
#   x, y, rfop_pct, run_label, form, scaling, construction, policy, seed
# Factor levels are recovered from metadata, NOT parsed from the directory name,
# so the loader is robust to label-format changes.
load_run_factorial <- function(dir_path, label = basename(dir_path)) {
  dir_path  <- normalizePath(dir_path, mustWork = TRUE)
  psel_path <- file.path(dir_path, "pixel_selection.csv")
  meta_path <- file.path(dir_path, "metadata.json")
  if (!file.exists(psel_path)) {
    warning(sprintf("pixel_selection.csv not found in %s - skipping.", dir_path))
    return(NULL)
  }
  if (!file.exists(meta_path)) {
    warning(sprintf("metadata.json not found in %s - cannot recover factor levels; skipping.", dir_path))
    return(NULL)
  }
  meta <- jsonlite::read_json(meta_path)
  rc   <- meta$run_config %||% list()
  sp   <- meta$scenario_params %||% list()

  n_nondom <- as.integer(meta$n_nondominated_solutions %||% meta$n_solutions %||% 1L)

  # Factor levels: prefer the explicit flat factor_* keys; fall back to the
  # underlying scenario fields / the run label so older exports still load.
  .seed_from_label <- function(lbl) {
    s <- stringr::str_extract(lbl, "seed[0-9]+$")
    if (is.na(s)) NA_integer_ else as.integer(sub("seed", "", s))
  }
  form         <- rc$factor_form         %||% sp$rp_formulation %||% NA_character_
  scaling      <- rc$factor_scaling      %||% NA_character_
  construction <- rc$factor_construction %||% NA_character_
  policy       <- rc$factor_policy       %||% NA_character_
  seed         <- rc$random_seed         %||% .seed_from_label(label)

  # If scaling/construction were not written explicitly, split the condition tag
  # ("{scaling}_{construction}", e.g. "upper_q75_drop_smd") on its known prefix.
  cond_tag <- rc$condition_scenario %||% NA_character_
  if ((is.na(scaling) || is.na(construction)) && !is.na(cond_tag)) {
    for (pre in c("upper_q75", "global", "zones")) {
      if (startsWith(cond_tag, paste0(pre, "_"))) {
        if (is.na(scaling))      scaling      <- pre
        if (is.na(construction)) construction <- sub(paste0("^", pre, "_"), "", cond_tag)
        break
      }
    }
  }

  df <- .filter_within_be(.read_xy_fast(psel_path))
  df |>
    dplyr::count(x, y, name = "n_selected") |>
    dplyr::mutate(
      rfop_pct     = n_selected / n_nondom * 100,
      run_label    = label,
      form         = as.character(form),
      scaling      = as.character(scaling),
      construction = as.character(construction),
      policy       = as.character(policy),
      seed         = seed
    ) |>
    dplyr::select(x, y, rfop_pct, run_label, form, scaling, construction, policy, seed)
}


# Per-cell crossed-factorial variance partition of RFOP.
#
# For each spatial cell (x, y) it fits a fixed-effects factorial model
#   rfop_pct ~ f1 * f2 * ... (all factors that vary within the cell)
# and converts the ANOVA sums of squares to eta^2 (SS_term / SS_total) for every
# main effect and interaction. The model Residual captures seed replication (and
# any unmodelled variation) - i.e. the noise floor against which factor effects
# are read. eta^2 across all returned terms of a cell sums to 1.
#
# runs_df : bind_rows() of load_run_factorial() - cols x, y, rfop_pct + factors
# factors : factor columns to cross (those with a single level overall are dropped)
# min_obs : minimum observations in a cell to attempt a fit (guards rank-deficiency)
#
# Returns a long data frame: x, y, term, ss, eta2, component
#   component in {<factor name>, "interaction", "residual"}
compute_rfop_variance_partition <- function(runs_df,
                                            factors = c("form", "scaling", "construction", "policy"),
                                            min_obs = 8) {
  stopifnot(all(c("x", "y", "rfop_pct") %in% names(runs_df)))
  factors <- intersect(factors, names(runs_df))
  # Keep only factors that actually vary across the matrix.
  factors <- factors[vapply(factors, function(f) dplyr::n_distinct(runs_df[[f]]) > 1L, logical(1))]
  if (length(factors) == 0L)
    stop("compute_rfop_variance_partition(): no supplied factor varies across runs_df.")

  classify_term <- function(term) {
    if (term == "Residuals") return("residual")
    if (grepl(":", term, fixed = TRUE)) return("interaction")
    term
  }

  runs_df |>
    dplyr::mutate(dplyr::across(dplyr::all_of(factors), as.factor)) |>
    dplyr::group_by(x, y) |>
    dplyr::group_modify(function(.d, .key) {
      if (nrow(.d) < min_obs) return(tibble::tibble())
      # Factors that vary *within this cell's rows* (others are not estimable here).
      f_local <- factors[vapply(factors,
                                function(f) nlevels(droplevels(.d[[f]])) > 1L, logical(1))]
      if (length(f_local) == 0L) return(tibble::tibble())
      fml <- stats::as.formula(paste("rfop_pct ~", paste(f_local, collapse = " * ")))
      fit <- tryCatch(stats::aov(fml, data = .d), error = function(e) NULL)
      if (is.null(fit)) return(tibble::tibble())
      a <- tryCatch(stats::anova(fit), error = function(e) NULL)
      if (is.null(a) || !("Sum Sq" %in% names(a))) return(tibble::tibble())
      ss    <- a[["Sum Sq"]]
      terms <- rownames(a)
      tot   <- sum(ss, na.rm = TRUE)
      if (!is.finite(tot) || tot <= 0) return(tibble::tibble())
      tibble::tibble(
        term      = terms,
        ss        = ss,
        eta2      = ss / tot,
        component = vapply(terms, classify_term, character(1))
      )
    }) |>
    dplyr::ungroup()
}


# Pool a variance-partition long df to one eta^2 per (cell, component): every
# interaction term collapses into a single "interaction" share per cell.
.pool_vp_components <- function(vp_df) {
  vp_df |>
    dplyr::group_by(x, y, component) |>
    dplyr::summarise(eta2 = sum(eta2, na.rm = TRUE), .groups = "drop")
}

# Pretty-print an aov term ("form:scaling") using the .factor_labels, joined with
# " x " (e.g. "Objective form x Condition scaling"). Unknown tokens pass through.
.vp_term_label <- function(terms) {
  vapply(terms, function(t) {
    parts <- strsplit(t, ":", fixed = TRUE)[[1]]
    lab   <- ifelse(parts %in% names(.factor_labels), .factor_labels[parts], parts)
    paste(lab, collapse = " x ")
  }, character(1), USE.NAMES = FALSE)
}

# Break the pooled "interaction" share back into individual interaction terms.
# R4f reports a single aggregate interaction bar; this opens it up so a specific
# two-way interaction (e.g. scaling x form) can be named rather than lumped.
#
# vp : compute_rfop_variance_partition() output (term/eta2/component per cell)
# Returns one row per interaction term, ordered by mean_eta2 desc:
#   term, label, order (# factors), n_cells, mean_eta2, se_eta2
# mean_eta2 is averaged over the cells where that term is estimable; n_cells
# records that support (cells differ in which factors vary locally).
summarise_vp_interactions <- function(vp) {
  if (is.null(vp) || nrow(vp) == 0L) return(tibble::tibble())
  vp |>
    dplyr::filter(component == "interaction") |>
    dplyr::group_by(term) |>
    dplyr::summarise(
      n_cells   = dplyr::n(),
      mean_eta2 = mean(eta2, na.rm = TRUE),
      se_eta2   = stats::sd(eta2, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups   = "drop"
    ) |>
    dplyr::mutate(order = stringr::str_count(term, ":") + 1L,
                  label = .vp_term_label(term)) |>
    dplyr::arrange(dplyr::desc(mean_eta2)) |>
    dplyr::select(term, label, order, n_cells, mean_eta2, se_eta2)
}

# Horizontal bar of each interaction term's mean variance share (eta^2), with SE
# whiskers, coloured by interaction order (2-way / 3-way / ...).
#
# vp : compute_rfop_variance_partition() output
make_vp_interaction_bar <- function(vp) {
  s <- summarise_vp_interactions(vp)
  if (nrow(s) == 0L) return(NULL)
  s <- dplyr::mutate(s, label = factor(label, levels = rev(label)),
                     order_f = factor(order))
  ggplot(s, aes(x = label, y = mean_eta2, fill = order_f)) +
    geom_col(width = 0.7, colour = "grey30") +
    geom_errorbar(aes(ymin = pmax(0, mean_eta2 - se_eta2), ymax = mean_eta2 + se_eta2),
                  width = 0.25, colour = "grey30") +
    scale_fill_manual(values = c("2" = "#D55E00", "3" = "#D55E00", "4" = "#E8D4B0"),
                      name = "Interaction order") +
    scale_y_continuous(labels = scales::label_percent(accuracy = 0.1), expand = c(0.01, 0),
                       position = "left") +
    scale_x_discrete(labels = scales::label_wrap(20)) +
    labs(y = "Variance share", x = NULL) +
    theme_minimal() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = "none",
          axis.text.y = element_text(size = 14, hjust = 1),
          axis.text.x = element_text(size = 14),
          axis.title.y = element_text(size = 14, face = "bold"))
}

# Per-cell map of the dominant INTERACTION term: among interaction terms only,
# colour each cell by the term with the largest eta^2. Directly tests where a
# specific two-way interaction (e.g. scaling x form) is spatially concentrated.
#
# vp      : compute_rfop_variance_partition() output
# elig_df : eligible-pixel template for the grey underlay (optional)
# min_top : only colour cells whose top interaction term reaches this eta^2
#           (default 0 = colour every classified cell)
make_dominant_interaction_map <- function(vp, elig_df = NULL, min_top = 0,
                                          title = NULL) {
  if (is.null(vp) || nrow(vp) == 0L) return(NULL)
  dom <- vp |>
    dplyr::filter(component == "interaction") |>
    dplyr::group_by(x, y) |>
    dplyr::slice_max(eta2, n = 1, with_ties = FALSE) |>
    dplyr::ungroup() |>
    dplyr::filter(eta2 >= min_top) |>
    dplyr::mutate(label = .vp_term_label(term))
  if (nrow(dom) == 0L) return(NULL)
  # Legend order = global prevalence (most cells first) for a stable legend.
  lev <- dom |> dplyr::count(label, sort = TRUE) |> dplyr::pull(label)
  dom <- dplyr::mutate(dom, label = factor(label, levels = lev))

  p <- ggplot()
  if (!is.null(elig_df) && nrow(elig_df) > 0)
    p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#e6e6e6")
  p <- p +
    geom_raster(data = dom, aes(x = x, y = y, fill = label)) +
    scale_fill_brewer(palette = "Set2", name = "Dominant interaction", na.value = "grey85") +
    theme_void()
  if (exists("BE")) {
    p <- p + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
  } else {
    p <- p + coord_equal()
  }
  p
}

# Global variance-share bar: mean eta^2 per source across cells (+/- SE).
# Bars are ordered main effects to interactions to residual via .factor_palette.
#
# vp_df : output of compute_rfop_variance_partition()
make_variance_partition_bar <- function(vp_df,
                                        title = NULL) {
  if (is.null(vp_df) || nrow(vp_df) == 0L) return(NULL)
  summ <- .pool_vp_components(vp_df) |>
    dplyr::group_by(component) |>
    dplyr::summarise(
      mean_eta2 = mean(eta2, na.rm = TRUE),
      se_eta2   = stats::sd(eta2, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups   = "drop"
    ) |>
    dplyr::filter(component %in% names(.factor_palette)) |>
    dplyr::mutate(
      comp_f = factor(component, levels = rev(names(.factor_palette))),
      label  = .factor_labels[as.character(component)]
    )

  ggplot(summ, aes(x = mean_eta2, y = comp_f, fill = component)) +
    geom_col(width = 0.65, colour = "grey30") +
    geom_errorbar(aes(xmin = pmax(0, mean_eta2 - se_eta2), xmax = mean_eta2 + se_eta2),
                  width = 0.25, colour = "grey30") +
    scale_fill_manual(values = .factor_palette, guide = "none") +
    scale_y_discrete(labels = function(b) .factor_labels[b]) +
    scale_x_continuous(labels = scales::label_percent(accuracy = 1), expand = c(0.01, 0)) +
    labs(title = title,
         x = "Variance share", y = NULL) +
    theme_minimal() +
    theme(panel.grid.major.y = element_blank())
}

# Per-cell dominant-source map: colour each cell by the source with the largest
# eta^2 (which formulation choice - or interaction / seed - drives that cell).
#
# vp_df   : output of compute_rfop_variance_partition()
# elig_df : eligible_pixels.csv data frame for a grey underlay (optional)
# include : which components are eligible to "win" a cell
make_dominant_factor_map <- function(vp_df, elig_df = NULL,
                                     include = names(.factor_palette)) {
  if (is.null(vp_df) || nrow(vp_df) == 0L) return(NULL)
  dom <- .pool_vp_components(vp_df) |>
    dplyr::filter(component %in% include) |>
    dplyr::group_by(x, y) |>
    dplyr::slice_max(eta2, n = 1, with_ties = FALSE) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      component = factor(component, levels = names(.factor_palette)),
      label     = factor(.factor_labels[as.character(component)],
                         levels = unname(.factor_labels[names(.factor_palette)]))
    )

  p <- ggplot()
  if (!is.null(elig_df) && nrow(elig_df) > 0)
    p <- p + geom_raster(data = elig_df, aes(x = x, y = y), fill = "#e6e6e6")
  p <- p +
    geom_raster(data = dom, aes(x = x, y = y, fill = label)) +
    scale_fill_manual(
      values   = setNames(unname(.factor_palette), unname(.factor_labels[names(.factor_palette)])),
      name     = "Dominant source",
      na.value = "grey85", drop = TRUE
    ) +
    theme_void() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12))
  # geom_sf forces coord_sf, so only set coord_equal when no sf overlay is added
  # (coord_equal + geom_sf together error in ggplot).
  if (exists("BE")) {
    p <- p + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
  } else {
    p <- p + coord_equal()
  }
  p
}


# =============================================================================
# R4g - Is the seed (residual) noise floor uniform across factorial design cells?
# -----------------------------------------------------------------------------
# R4f treats seed variation as a single pooled "residual" share. R4g opens that
# residual up: it asks whether the WITHIN-cell seed noise (the SD of RFOP across
# seed replicates, per patch) has the same magnitude and the same spatial
# footprint in every formulation combination, or whether some formulations are
# intrinsically noisier / move the noise to different patches.
#
# A "design cell" is one fully-crossed factor combination (form x scaling x
# construction x policy); within a cell ONLY the random seed varies. A "patch" is
# one spatial unit (x, y) at whatever resolution the supplied arrays use - pass
# the same fac_agg that feeds compute_rfop_variance_partition() so this is read
# off the identical per-patch RFOP arrays.
#
# Pipeline:
#   1. compute_seed_noise_by_cell()             - per-patch seed SD + per-cell mean
#   2. make_seed_noise_bar()                    - mean seed SD per design cell (bar)
#   3. compute_seed_noise_overlap()             - Jaccard of high-noise patch sets
#   4. make_seed_noise_jaccard_heatmap()        - heatmap (R3c style)
#   5. compute_seed_noise_factor_association()  - which factor shifts the pattern
# =============================================================================

# Per-patch within-cell seed noise for a factorial run matrix.
#
# For each design cell (unique combination of `factors`, within which only the
# random seed varies) and each patch (x, y) it computes the SD of RFOP across the
# available seed replicates. Returns BOTH the per-patch SD (for the spatial
# overlap analysis) and the per-cell mean (the bar-chart summary).
#
# fac_df    : load_run_factorial() / fac_agg rows - x, y, rfop_pct, <factors>, seed
# factors   : factor columns whose unique combination defines a design cell
# min_seeds : (cell, patch) groups with fewer than this many seed values are
#             dropped (need >= 2 to compute an SD)
# zero_fill : TRUE (default) treats a patch absent from a seed as RFOP = 0, so a
#             patch's presence/absence flipping across seeds counts as seed noise
#             (the complete view of seed instability). FALSE matches R4f exactly -
#             a patch contributes only the seeds in which it was selected.
# Returns list(per_patch, per_cell, factors), where per_cell has one row per
# design cell with the factor columns + mean_seed_sd + n_patches.
compute_seed_noise_by_cell <- function(fac_df,
                                       factors   = c("form", "scaling", "construction", "policy"),
                                       min_seeds = 2L,
                                       zero_fill = TRUE) {
  factors <- intersect(factors, names(fac_df))
  stopifnot(all(c("x", "y", "rfop_pct", "seed") %in% names(fac_df)), length(factors) > 0L)
  # Keep only factors that actually vary, so the cell label stays informative.
  factors <- factors[vapply(factors, function(f) dplyr::n_distinct(fac_df[[f]]) > 1L, logical(1))]
  if (length(factors) == 0L)
    stop("compute_seed_noise_by_cell(): no supplied factor varies across fac_df.")

  per_patch <- fac_df |>
    dplyr::group_by(dplyr::across(dplyr::all_of(factors))) |>
    dplyr::group_modify(function(.d, .key) {
      if (zero_fill)
        .d <- tidyr::complete(.d, tidyr::nesting(x, y), seed,
                              fill = list(rfop_pct = 0))
      .d |>
        dplyr::group_by(x, y) |>
        dplyr::summarise(seed_sd = stats::sd(rfop_pct, na.rm = TRUE),
                         n_seeds = dplyr::n(), .groups = "drop")
    }) |>
    dplyr::ungroup() |>
    dplyr::filter(n_seeds >= min_seeds, !is.na(seed_sd))

  # Compact "a | b | c" cell label from the factor columns (order = `factors`).
  per_patch$cell <- do.call(paste, c(lapply(factors, function(f) per_patch[[f]]),
                                     list(sep = " | ")))

  per_cell <- per_patch |>
    dplyr::group_by(dplyr::across(dplyr::all_of(c(factors, "cell")))) |>
    dplyr::summarise(mean_seed_sd = mean(seed_sd, na.rm = TRUE),
                     n_patches    = dplyr::n(), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(mean_seed_sd))

  list(per_patch = per_patch, per_cell = per_cell, factors = factors)
}


# Horizontal bar chart of mean within-cell seed SD, one bar per design cell,
# ordered by magnitude. Uniform bar heights => seed noise is a constant floor;
# a long tail => some formulations are intrinsically noisier.
#
# noise : output of compute_seed_noise_by_cell()
make_seed_noise_bar <- function(noise,
                                title = "Within-cell seed noise across factorial design cells") {
  pc <- noise$per_cell
  if (is.null(pc) || nrow(pc) == 0L) return(NULL)
  pc <- pc |>
    dplyr::mutate(cell = factor(cell, levels = rev(cell)))   # already SD-sorted

  ggplot(pc, aes(x = mean_seed_sd, y = cell)) +
    geom_col(fill = "#E9C46A", colour = "grey30", width = 0.7) +
    scale_x_continuous(expand = c(0.01, 0)) +
    labs(title = title,
         subtitle = paste("Mean over patches of the per-patch SD of RFOP across",
                          "seed replicates (form | scaling | construction | policy)"),
         x = "Mean seed SD of RFOP (percentage points)", y = NULL) +
    theme_minimal() +
    theme(panel.grid.major.y = element_blank(),
          axis.text.y = element_text(size = 7))
}


# Pairwise Jaccard overlap of the high-noise patch sets across design cells.
# For each cell, the "high-noise" set is the top `top_pct`% of patches by seed SD.
#
# per_patch : the per_patch tibble from compute_seed_noise_by_cell()
# top_pct   : percentile cut for the high-noise set (default 20)
# Returns list(jaccard <a,b,jaccard>, mean_jaccard, n_cells, top_pct), mirroring
# compute_topx_overlap() so the same heatmap code applies.
compute_seed_noise_overlap <- function(per_patch, top_pct = 20) {
  stopifnot(all(c("cell", "x", "y", "seed_sd") %in% names(per_patch)))
  cells  <- sort(unique(per_patch$cell))
  n_cell <- length(cells)

  top_df <- per_patch |>
    dplyr::group_by(cell) |>
    dplyr::mutate(thr = stats::quantile(seed_sd, 1 - top_pct / 100, na.rm = TRUE)) |>
    dplyr::filter(seed_sd >= thr) |>
    dplyr::ungroup() |>
    dplyr::select(cell, x, y)

  sets    <- split(paste(top_df$x, top_df$y), top_df$cell)
  jaccard <- if (n_cell >= 2) {
    pairs <- utils::combn(cells, 2, simplify = FALSE)
    do.call(rbind, lapply(pairs, function(p) {
      a <- sets[[p[1]]]; b <- sets[[p[2]]]
      uni <- length(union(a, b))
      data.frame(a = p[1], b = p[2],
                 jaccard = if (uni > 0) length(intersect(a, b)) / uni else NA_real_)
    }))
  } else data.frame(a = character(0), b = character(0), jaccard = numeric(0))

  list(jaccard      = tibble::as_tibble(jaccard),
       mean_jaccard = mean(jaccard$jaccard, na.rm = TRUE),
       n_cells      = n_cell,
       top_pct      = top_pct)
}


# Symmetric Jaccard heatmap of high-noise patch overlap across design cells.
# Same visual style as the R3c top-X% priority overlap heatmap.
#
# overlap : output of compute_seed_noise_overlap()
make_seed_noise_jaccard_heatmap <- function(overlap,
                                            title = NULL) {
  jac <- overlap$jaccard
  if (is.null(jac) || nrow(jac) == 0L) return(NULL)
  if (is.null(title))
    title <- sprintf("High-noise patch overlap across design cells (top %d%% by seed SD)",
                     overlap$top_pct)
  jh <- dplyr::bind_rows(jac, dplyr::rename(jac, a = b, b = a))
  ggplot(jh, aes(a, b, fill = jaccard)) +
    geom_tile() +
    scale_fill_viridis_c(name = "Jaccard", limits = c(0, NA)) +
    labs(title = title,
         subtitle = "Jaccard of the top-noise patch sets; high & uniform = landscape-driven noise",
         x = NULL, y = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
          axis.text.y = element_text(size = 6))
}


# For each factor, compare mean high-noise Jaccard between design-cell PAIRS that
# share that factor's level vs pairs that differ on it. A large (same - diff) gap
# means the spatial footprint of seed noise shifts systematically with that
# factor; a near-zero gap for every factor means the footprint is formulation-
# invariant (landscape-driven).
#
# overlap  : output of compute_seed_noise_overlap()
# per_cell : the per_cell tibble from compute_seed_noise_by_cell()
# factors  : factor columns to test (default: those present in per_cell)
compute_seed_noise_factor_association <- function(overlap, per_cell, factors) {
  jac <- overlap$jaccard
  if (is.null(jac) || nrow(jac) == 0L) return(tibble::tibble())
  factors <- intersect(factors, names(per_cell))
  ia <- match(jac$a, per_cell$cell)
  ib <- match(jac$b, per_cell$cell)
  purrr::map_dfr(factors, function(f) {
    same <- per_cell[[f]][ia] == per_cell[[f]][ib]
    tibble::tibble(
      factor        = f,
      mean_jac_same = mean(jac$jaccard[same],  na.rm = TRUE),
      mean_jac_diff = mean(jac$jaccard[!same], na.rm = TRUE),
      n_same        = sum(same,  na.rm = TRUE),
      n_diff        = sum(!same, na.rm = TRUE)
    )
  }) |>
    dplyr::mutate(gap = mean_jac_same - mean_jac_diff) |>
    dplyr::arrange(dplyr::desc(gap))
}
