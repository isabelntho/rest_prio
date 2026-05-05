# _plot_functions.R
# All plotting functions for opt_results_comparison.qmd.
# Sourced once in the setup chunk so both per-run tabs and the comparison
# section can share a single definition of every function.
#
# Global objects expected to exist in the calling environment:
#   COL_ND, COL_DOM, ALPHA_ND, ALPHA_DOM  — colour constants from config chunk
#   BE                                     — sf object for canton boundary
#   lulc, abiotic_anomaly                  — terra SpatRaster objects
#   ECOSYSTEM                              — character scalar for plot titles


# ── Algorithm evolution ───────────────────────────────────────────────────────

prep_evo_data <- function(run, obj_names, obj_labels) {
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

make_hv_evo_plot <- function(run, obj_names, obj_labels) {
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


# ── Pareto front ──────────────────────────────────────────────────────────────

make_pareto_extremes_plot <- function(run, obj_names, obj_labels, title = NULL,
                                       scale_by_n_pixels = FALSE) {
  o1 <- obj_names[1]; o2 <- obj_names[2]; o3 <- obj_names[3]
  l1 <- obj_labels[1]; l2 <- obj_labels[2]; l3 <- obj_labels[3]

  df <- run$df_obj

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

  ggplot(df, aes(x = .data[[o1]], y = .data[[o2]])) +
    geom_point(aes(fill = .data[[o3]]), shape = 21, colour = "grey70",
               size = 4, stroke = 0.8, alpha = 0.8) +
    scale_fill_viridis_c(name = l3, option = "plasma") +
    labs(x = l1, y = l2)
}

make_pareto_pairwise <- function(run, obj_names, obj_labels) {
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

  wrap_plots(plots, ncol = length(plots), guides = "collect") +
    plot_annotation(theme = theme(plot.title = element_text(size = 13, face = "bold")))
}

make_corr_plot <- function(run, obj_names, obj_labels, title = NULL, maximize = NULL) {
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
    facet_wrap(~panel, ncol = 1) +
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

prep_parcoord_data <- function(run, obj_names, obj_labels,
                               best_colours = c("#E05C2A", "#2A7BE0", "#2AB05C"),
                               maximize = NULL) {
  df <- run$df_obj

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

make_parcoord_plot <- function(run, obj_names, obj_labels,
                               best_colours = c("#E05C2A", "#2A7BE0", "#2AB05C"),
                               maximize = NULL) {
  extended_colours <- rep_len(best_colours, length(obj_names))
  legend_breaks    <- c(extended_colours[seq_along(obj_names)], "black", "grey85")
  legend_labels    <- c(sprintf("Best: %s", obj_labels[seq_along(obj_names)]),
                        "Non-dominated", "Dominated")

  df_pc       <- prep_parcoord_data(run, obj_names, obj_labels, best_colours, maximize)
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


# ── Spatial trade-offs (Jaccard) ──────────────────────────────────────────────

compute_jaccard_pairs <- function(run, obj_names) {
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

make_jaccard_plot <- function(run, obj_names, title = NULL) {
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


# ── Selection frequency maps ──────────────────────────────────────────────────

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
# Coordinates are normalised (0–1) relative to the panel, so they work
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
          plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
          legend.position  = "none") +
    labs(title = run$label)

  if (!is.null(elig_unsel) && nrow(elig_unsel) > 0)
    p_map <- p_map + geom_raster(data = elig_unsel, aes(x = x, y = y), fill = "grey80")

  p_map <- p_map +
    geom_raster(data = freq_df, aes(x = x, y = y, fill = rfop_pct)) +
    scale_fill_distiller(palette = "YlOrRd", direction = 1, limits = c(0, 100)) +
    geom_sf(data = BE, fill = NA, color = "black", inherit.aes = FALSE)

  .add_rfop_inset(p_map, freq_df)
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
          plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
          legend.position  = "none") +
    labs(title = run$label)

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


# ── Selection frequency histograms ────────────────────────────────────────────

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

  sel_coords <- run$df_psel %>% count(x, y) %>% select(x, y)
  vals       <- terra::extract(obj_raster, sel_coords)[, col_idx + 1]
  as.factor(cut(vals, breaks = brk, labels = labels, include.lowest = TRUE))
}

# ── Eligible-pixel fill helpers (denominators for make_sel_freq_unified_plot) ─

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
  as.factor(cut(vals, breaks = brk, labels = labels, include.lowest = TRUE))
}


# ── Action-type helpers (restore vs convert) ──────────────────────────────────

# Stacked-bar chart: one bar per non-dominated solution, stacked by action_type.
# Solutions are ordered by their value on `order_obj` (ascending = best first).
# Requires df_psel to have an action_type column ("restore" / "convert").
make_action_stacked_bar <- function(run, obj_names, obj_labels,
                                    order_obj = NULL,
                                    colours = c(restore = "#4DAF4A", convert = "#984EA3")) {
  if (is.null(run$df_psel)) {
    message("pixel_selection.csv not available – skipping action stacked bar.")
    return(invisible(NULL))
  }
  if (!"action_type" %in% names(run$df_psel)) {
    message("action_type column not found in pixel_selection.csv – re-run export_to_r.py.")
    return(invisible(NULL))
  }

  # Count actions per solution × action_type (non-dominated solutions only)
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
      title    = sprintf("Action mix per non-dominated solution — %s", run$label),
      subtitle = sprintf("Solutions ordered by %s (ascending)", order_label),
      x = sprintf("Solution rank (by %s, best → worst)", order_label),
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
    message("action_type column not found in pixel_selection.csv – re-run export_to_r.py.")
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
               theme_void() +
               labs(title = sprintf("%s", title_suffix)))
    }

    p <- ggplot() +
      theme_void() +
      theme(panel.background = element_rect(fill = "white", colour = NA),
            plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
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
      geom_sf(data = BE, fill = NA, colour = "black", inherit.aes = FALSE) +
      labs(title = sprintf("%s — %s", run$label, title_suffix))

    .add_rfop_inset(p, freq_df)
  }

  p_restore <- make_one_map("restore", colours_restore, "Restoration frequency")
  p_convert <- make_one_map("convert", colours_convert, "Conversion frequency")

  p_restore + p_convert +
    patchwork::plot_annotation(
      theme = theme(plot.title = element_text(size = 13, face = "bold"))
    )
}


# ── Comparison functions ──────────────────────────────────────────────────────

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


# ── Spatial priority classification ──────────────────────────────────────────
#
# These functions support the multi-run spatial classification analysis.
# The workflow is:
#   1. load_run_rfop()              — load one run's per-pixel RFOP + metadata
#   2. bind_rows() over many runs   — one row per pixel per run
#   3. compute_spatial_classification() — classify each eligible pixel
#   4. make_classification_map()    — categorical raster map
#   5. make_classification_summary_bar() — % area per class
#
# Classification dimensions:
#   dim_type = "indicator"  — condition/indicator assumption runs
#   dim_type = "policy"     — policy decision runs (e.g. burden sharing)
#   dim_type = "seed"       — stochastic replicate runs
#   dim_type = "param"      — continuous parameter sweep runs (optional)


# Load a single run's pixel-level RFOP and attach sensitivity-dimension metadata.
# Returns a data frame with columns: x, y, rfop_pct, run_label, dim_type, group_label.
# rfop_pct is computed relative to that run's own Pareto front size, so runs with
# different numbers of non-dominated solutions are directly comparable.
#
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
    warning(sprintf("pixel_selection.csv not found in %s — skipping.", dir_path))
    return(NULL)
  }

  meta_path <- file.path(dir_path, "metadata.json")
  n_nondom  <- if (file.exists(meta_path)) {
    meta <- jsonlite::read_json(meta_path)
    as.integer(meta$n_nondominated_solutions %||% meta$n_solutions %||% 1L)
  } else {
    warning(sprintf("metadata.json not found in %s — rfop_pct will be approximate.", dir_path))
    NA_integer_
  }

  df <- read.csv(psel_path)

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


# Classify each eligible pixel into one of five priority classes based on
# its mean RFOP across all runs and the sensitivity profile across dimensions.
#
# runs_df       : bind_rows() of load_run_rfop() outputs; columns
#                 x, y, rfop_pct, run_label, dim_type, group_label
# elig_df       : eligible_pixels.csv data frame (x, y) — used to ensure every
#                 eligible pixel receives a class (pixels absent from all runs
#                 are treated as rfop_pct = 0 and classified as "Low priority")
# thresh_low    : pixels with mean_rfop < thresh_low across all runs → "Low priority"
# thresh_high   : pixels with mean_rfop >= thresh_high AND low sensitivity → "Robust priority"
# thresh_stable : normalised sensitivity threshold below which a dimension is
#                 considered negligible; nsens = range(group means) / (mean_rfop + 1)
#
# Returns a data frame with one row per eligible pixel:
#   x, y, mean_rfop_all, sens_indicator, sens_policy, sens_seed, sens_param,
#   dominant_dim, classification (ordered factor)
compute_spatial_classification <- function(
    runs_df,
    elig_df       = NULL,
    thresh_low    = 20,
    thresh_high   = 60,
    thresh_stable = 0.3
) {
  stopifnot(all(c("x", "y", "rfop_pct", "dim_type", "group_label") %in% names(runs_df)))

  # ── Step 1: mean RFOP per pixel per (dim_type × group_label) ─────────────
  # For each sensitivity dimension, compute the mean rfop_pct across all runs
  # that belong to the same group, then take the range across groups.
  # This separates within-group noise from between-group signal.
  per_dim_sens <- runs_df |>
    dplyr::group_by(x, y, dim_type, group_label) |>
    dplyr::summarise(group_mean_rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    dplyr::group_by(x, y, dim_type) |>
    dplyr::summarise(
      sens_raw = max(group_mean_rfop, na.rm = TRUE) -
                 min(group_mean_rfop, na.rm = TRUE),
      .groups = "drop"
    ) |>
    tidyr::pivot_wider(names_from = dim_type, values_from = sens_raw,
                       names_prefix = "sens_", values_fill = 0)

  # Ensure all four dimension columns exist (fill 0 when a dimension has no runs)
  for (dim in c("sens_indicator", "sens_policy", "sens_seed", "sens_param")) {
    if (!dim %in% names(per_dim_sens))
      per_dim_sens[[dim]] <- 0
  }

  # ── Step 2: mean RFOP across ALL runs ─────────────────────────────────────
  mean_rfop <- runs_df |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(mean_rfop_all = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

  # ── Step 3: join and add eligible pixels that never appeared (rfop = 0) ──
  class_df <- mean_rfop |>
    dplyr::left_join(per_dim_sens, by = c("x", "y"))

  if (!is.null(elig_df) && nrow(elig_df) > 0) {
    never_selected <- dplyr::anti_join(elig_df[, c("x", "y")], class_df, by = c("x", "y")) |>
      dplyr::mutate(
        mean_rfop_all  = 0,
        sens_indicator = 0,
        sens_policy    = 0,
        sens_seed      = 0,
        sens_param     = 0
      )
    class_df <- dplyr::bind_rows(class_df, never_selected)
  }

  class_df[is.na(class_df)] <- 0

  # ── Step 4: normalised sensitivity (avoids /0 for low-rfop pixels) ────────
  class_df <- class_df |>
    dplyr::mutate(
      denom        = mean_rfop_all + 1,
      nsens_ind    = sens_indicator / denom,
      nsens_pol    = sens_policy    / denom,
      nsens_seed   = sens_seed      / denom,
      nsens_param  = sens_param     / denom
    )

  # ── Step 5: classify ──────────────────────────────────────────────────────
  # Priority: Low → Robust → sensitivity dimensions (indicator > policy > seed > param)
  class_df <- class_df |>
    dplyr::mutate(
      classification = dplyr::case_when(
        mean_rfop_all < thresh_low
          ~ "Low priority",
        mean_rfop_all >= thresh_high &
          nsens_ind   < thresh_stable &
          nsens_pol   < thresh_stable &
          nsens_seed  < thresh_stable &
          nsens_param < thresh_stable
          ~ "Robust priority",
        nsens_ind >= pmax(nsens_pol, nsens_seed, nsens_param) &
          nsens_ind >= thresh_stable
          ~ "Condition sensitive",
        nsens_pol >= pmax(nsens_ind, nsens_seed, nsens_param) &
          nsens_pol >= thresh_stable
          ~ "Policy sensitive",
        nsens_seed >= pmax(nsens_ind, nsens_pol, nsens_param) &
          nsens_seed >= thresh_stable
          ~ "Unstable / noisy",
        nsens_param >= thresh_stable
          ~ "Condition sensitive",  # fold param sensitivity into condition
        TRUE
          ~ "Robust priority"       # medium freq, no dominant sensitivity
      ),
      classification = factor(
        classification,
        levels = c("Robust priority", "Condition sensitive", "Policy sensitive",
                   "Unstable / noisy", "Low priority"),
        ordered = TRUE
      )
    )

  class_df
}


# Palette for the five priority classes (consistent across map + bar).
# Chosen for maximum hue + lightness separation at small pixel sizes:
#   Robust      — deep teal       (high priority, clearly distinct from orange/red)
#   Condition   — vivid amber     (warm, distinct from teal and purple)
#   Policy      — strong violet   (cool, separates from amber and teal)
#   Unstable    — bright magenta  (very distinct from all others; flags caution)
#   Low         — off-white/cream (recedes clearly; avoids confusion with map background)
.class_palette <- c(
  "Robust priority"      = "#006D77",   # deep teal
  "Condition sensitive"  = "#E9C46A",   # vivid amber
  "Policy sensitive"     = "#6A0572",   # strong violet
  "Unstable / noisy"     = "#E63946",   # bright red-pink
  "Low priority"         = "#F0EDE4"    # off-white / cream
)


# Plot a categorical raster map of the spatial classification.
# Styling mirrors make_sel_freq_plot() (theme_void, grey eligible underlay, BE boundary).
#
# class_df  : output of compute_spatial_classification()
# elig_df   : eligible_pixels.csv data frame for grey underlay (can be NULL)
# title     : plot title
make_classification_map <- function(class_df, elig_df = NULL,
                                    title = "Spatial priority classification") {
  # Grey underlay for all eligible pixels not classified as "Low priority"
  # (Low priority pixels are already rendered in grey, so we only need the
  # non-selected eligible pixels as background context)
  p <- ggplot() +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.position  = "right",
      legend.title     = element_text(size = 9),
      legend.text      = element_text(size = 8)
    ) +
    labs(title = title)

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


# Faceted heatmaps of raw sensitivity scores per dimension.
# Useful for diagnosing which dimension drives variability where.
#
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

  ggplot(plot_data, aes(x = x, y = y, fill = sens)) +
    geom_raster() +
    scale_fill_distiller(
      palette   = "YlOrRd",
      direction = 1,
      name      = "Sensitivity\n(RFOP range)",
      limits    = c(0, NA)
    ) +
    facet_wrap(~dimension, ncol = 2) +
    theme_void() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      strip.text       = element_text(size = 9, face = "bold", hjust = 0.5),
      legend.position  = "right"
    )
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


# ── Pareto front sensitivity analysis ────────────────────────────────────────
#
# Two complementary summaries of how Pareto front *quality* varies across
# sensitivity dimensions (condition scenarios, policy variants, seeds):
#
#   1. collect_pareto_stats()     — extract HV + ideal/nadir per run
#   2. make_hv_sensitivity_plot() — HV dot plot grouped by dimension
#   3. make_ideal_nadir_plot()    — heatmap of ideal-point shift per objective


# Extract hypervolume, ideal point, and nadir from a named list of run objects.
#
# runs_meta : named list, each entry is a list with:
#               $run        — a run object from load_run_data()
#               $dim_type   — "indicator" | "policy" | "seed" | "param"
#               $group_label — e.g. "global_all", "drop_smd", "with_bs"
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

  # ── Indicator panel ───────────────────────────────────────────────────────
  # One point per drop-one scenario (group_label); multiple seed runs overlap
  # at the same x position, which is fine for a quick spread check.
  df_ind <- dplyr::filter(df, dim_type == "indicator")
  p_ind <- if (nrow(df_ind) > 0)
    .hv_panel(df_ind, "group_label", "Condition / indicator")
  else NULL

  # ── Policy panel ──────────────────────────────────────────────────────────
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

  # ── Seed panel ────────────────────────────────────────────────────────────
  # Labels each run by its seed number extracted from run_label
  # e.g. "global_all_seed100" → "seed100"
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


# Heatmap of ideal-point values per objective × run, normalised relative to
# the reference run so that cells show % change from baseline.
# Positive = worse (higher cost or lower gain), Negative = better.
#
# stats_df    : output of collect_pareto_stats()
# obj_names   : character vector of objective names
# obj_labels  : display labels (same order as obj_names)
# ref_label   : run_label of the reference run
# point       : "ideal" (best achieved per objective) or "nadir" (worst ND value)
make_ideal_nadir_plot <- function(stats_df, obj_names, obj_labels,
                                  ref_label = NULL,
                                  point = c("ideal", "nadir")) {
  point <- match.arg(point)
  prefix <- paste0(point, "_")
  cols   <- paste0(prefix, obj_names)

  if (!all(cols %in% names(stats_df))) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = sprintf("No %s-point data found in stats_df", point), size = 5) +
             theme_void())
  }

  # Pivot to long
  long <- stats_df |>
    dplyr::select(run_label, dim_type, group_label, dplyr::all_of(cols)) |>
    tidyr::pivot_longer(cols = dplyr::all_of(cols),
                        names_to = "objective", values_to = "value") |>
    dplyr::mutate(
      objective = factor(
        gsub(prefix, "", objective, fixed = TRUE),
        levels = obj_names,
        labels = obj_labels
      )
    )

  # Normalise relative to reference run
  if (!is.null(ref_label) && ref_label %in% long$run_label) {
    ref_vals <- long |>
      dplyr::filter(run_label == ref_label) |>
      dplyr::select(objective, ref_value = value)
    long <- dplyr::left_join(long, ref_vals, by = "objective") |>
      dplyr::mutate(pct_change = (value - ref_value) / (abs(ref_value) + 1e-10) * 100)
    fill_col  <- "pct_change"
    fill_name <- sprintf("%% change from\n%s", ref_label)
    lim_sym   <- max(abs(long$pct_change), na.rm = TRUE)
    fill_scale <- scale_fill_gradient2(
      low      = "#1976D2",   # blue = improvement
      mid      = "white",
      high     = "#C62828",   # red  = degradation
      midpoint = 0,
      limits   = c(-lim_sym, lim_sym),
      name     = fill_name,
      labels   = scales::percent_format(scale = 1, accuracy = 1)
    )
  } else {
    fill_col   <- "value"
    fill_name  <- sprintf("%s point value", tools::toTitleCase(point))
    fill_scale <- scale_fill_distiller(palette = "YlOrRd", direction = 1,
                                        name = fill_name)
  }

  dim_labels <- c(indicator = "Condition", policy = "Policy",
                  seed = "Seed", param = "Parameter")
  long$dim_label <- dplyr::recode(long$dim_type, !!!dim_labels)

  ggplot(long, aes(x = group_label, y = objective, fill = .data[[fill_col]])) +
    geom_tile(colour = "white", linewidth = 0.4) +
    fill_scale +
    facet_wrap(~dim_label, scales = "free_x", nrow = 1) +
    labs(
      x        = NULL,
      y        = NULL,
      title    = sprintf("%s-point sensitivity across runs",
                         tools::toTitleCase(point)),
      subtitle = if (!is.null(ref_label))
        sprintf("Red = degraded vs %s, Blue = improved", ref_label)
      else
        "Raw objective values"
    ) +
    theme_minimal() +
    theme(
      axis.text.x      = element_text(angle = 40, hjust = 1, size = 8),
      axis.text.y      = element_text(size = 9),
      strip.text       = element_text(face = "bold", size = 10),
      panel.grid       = element_blank(),
      legend.position  = "right"
    )
}


# Diverging dot-plot of ideal-point or nadir shift across sensitivity dimensions.
#
# Produces three stacked panels (indicator / policy / seed), each faceted by
# objective.  Within each panel, runs are shown as rows and % change from the
# reference run is shown on the x-axis.  A vertical zero line and red/blue
# colouring make direction of change immediately readable.
#
# Row labelling per dimension:
#   indicator — group_label (seeds averaged within each drop-one scenario)
#   policy    — run_label   (shows each seed of both groups as a distinct row)
#   seed      — seed suffix extracted from run_label (e.g. "seed100")
#
# stats_df    : output of collect_pareto_stats()
# obj_names   : character vector of objective column names
# obj_labels  : display labels (same order as obj_names)
# ref_label   : run_label of the reference run (its row always shows 0 % change)
# point       : "ideal" or "nadir"
make_ideal_nadir_dotplot <- function(stats_df, obj_names, obj_labels,
                                     ref_label = NULL,
                                     point = c("ideal", "nadir")) {
  point  <- match.arg(point)
  prefix <- paste0(point, "_")
  cols   <- paste0(prefix, obj_names)

  if (!all(cols %in% names(stats_df))) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = sprintf("No %s-point data found in stats_df", point), size = 5) +
             theme_void())
  }

  # ── Pivot to long and compute % change from reference ──────────────────────
  long <- stats_df |>
    dplyr::select(run_label, dim_type, group_label,
                  dplyr::all_of(cols)) |>
    tidyr::pivot_longer(cols = dplyr::all_of(cols),
                        names_to = "objective", values_to = "value") |>
    dplyr::mutate(
      objective = factor(
        gsub(prefix, "", objective, fixed = TRUE),
        levels = obj_names,
        labels = obj_labels
      )
    )

  if (!is.null(ref_label) && ref_label %in% long$run_label) {
    ref_vals <- long |>
      dplyr::filter(run_label == ref_label) |>
      dplyr::select(objective, ref_value = value)
    long <- dplyr::left_join(long, ref_vals, by = "objective") |>
      dplyr::mutate(pct_change = (value - ref_value) / (abs(ref_value) + 1e-10) * 100)
  } else {
    long <- dplyr::mutate(long, pct_change = value, ref_value = NA_real_)
  }

  # ── Shared panel builder ───────────────────────────────────────────────────
  .dot_panel <- function(data, title) {
    ggplot(data, aes(x = pct_change, y = row_label,
                     colour = pct_change > 0)) +
      geom_vline(xintercept = 0, linetype = "solid",
                 colour = "grey60", linewidth = 0.5) +
      geom_segment(aes(x = 0, xend = pct_change,
                       y = row_label, yend = row_label),
                   linewidth = 0.5, alpha = 0.6) +
      geom_point(size = 3) +
      scale_colour_manual(values = c("FALSE" = "#1976D2",  # blue  = improved
                                     "TRUE"  = "#C62828"), # red   = degraded
                          guide = "none") +
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
        plot.title         = element_text(face = "bold", size = 10)
      )
  }

  # ── Indicator panel: average across seeds within each group ───────────────
  ind_data <- long |>
    dplyr::filter(dim_type == "indicator") |>
    dplyr::group_by(group_label, objective) |>
    dplyr::summarise(pct_change = mean(pct_change, na.rm = TRUE), .groups = "drop") |>
    dplyr::mutate(row_label = group_label)

  # ── Policy panel: every run as its own row ─────────────────────────────────
  pol_data <- long |>
    dplyr::filter(dim_type %in% c("policy", "seed")) |>
    dplyr::mutate(row_label = run_label)

  # ── Seed panel: extract seed suffix as row label ───────────────────────────
  seed_data <- long |>
    dplyr::filter(dim_type == "seed") |>
    dplyr::mutate(row_label = sub(".*_seed", "seed", run_label))

  plots <- list()

  if (nrow(ind_data) > 0)
    plots[["indicator"]] <- .dot_panel(ind_data, "Condition / indicator sensitivity")

  if (nrow(pol_data) > 0)
    plots[["policy"]] <- .dot_panel(pol_data, "Policy sensitivity (global_all vs upper_q75_all)")

  if (nrow(seed_data) > 0)
    plots[["seed"]] <- .dot_panel(seed_data, "Seed (stochastic) sensitivity")

  if (length(plots) == 0)
    return(ggplot() + annotate("text", x = 0.5, y = 0.5,
                               label = "No data", size = 5) + theme_void())

  patchwork::wrap_plots(plots, ncol = 1) +
    patchwork::plot_annotation(
      title    = sprintf("%s-point shift across sensitivity dimensions",
                         tools::toTitleCase(point)),
      subtitle = sprintf("x = %% change from reference run (%s) | Blue = improved, Red = degraded",
                         ref_label)
    )
}


# Dumbbell plot combining ideal-point and nadir shift in one figure.
#
# For each run × objective, a segment connects the ideal-point % change (filled
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

  # ── Helper: pivot one point type and compute % change vs reference ─────────
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

  # ── Widen back to ideal / nadir columns for segment drawing ───────────────
  wide <- long |>
    dplyr::select(run_label, dim_type, group_label, objective, point_type, pct_change) |>
    tidyr::pivot_wider(names_from = point_type, values_from = pct_change)

  # ── Row labels per dimension ───────────────────────────────────────────────
  ind_wide  <- dplyr::filter(wide, dim_type == "indicator") |>
    dplyr::group_by(group_label, objective) |>
    dplyr::summarise(ideal = mean(ideal, na.rm = TRUE),
                     nadir = mean(nadir, na.rm = TRUE), .groups = "drop") |>
    dplyr::mutate(row_label = group_label)

  pol_wide  <- dplyr::filter(wide, dim_type %in% c("policy", "seed")) |>
    dplyr::mutate(row_label = run_label)

  seed_wide <- dplyr::filter(wide, dim_type == "seed") |>
    dplyr::mutate(row_label = sub(".*_seed", "seed", run_label))

  # ── Panel builder ──────────────────────────────────────────────────────────
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


# ── Indicator redundancy analysis ─────────────────────────────────────────────
#
# Computes the van den Wollenberg redundancy index per EC indicator across all
# condition scenarios.  With univariate Y (RFOP per eligible pixel), the index
# simplifies to R² = cor(rfop, indicator)², requiring only base-R cor().
#
# Workflow:
#   1. load_scenario_rfop_avg()     — average RFOP per pixel across seeds
#   2. extract_indicator_values()   — pull indicator raster values at elig. pixels
#   3. compute_redundancy_indices() — R² per indicator × scenario × ecosystem
#   4. make_redundancy_boxplot()    — distribution across scenarios per indicator


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
# indicator_paths : named list mapping indicator code → path to .tif file
# lulc_raster     : SpatRaster; LULC class values used to assign ecosystem
# lulc_classes    : named list mapping ecosystem label → integer LULC class codes
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


# Compute van den Wollenberg redundancy index (R²) for each indicator ×
# condition scenario × ecosystem.
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


# Box plot: distribution of R² across condition scenarios per indicator.
# Faceted by ecosystem, coloured by ect category (abiotic / biotic).
make_redundancy_boxplot <- function(redundancy_df) {
  if (is.null(redundancy_df) || nrow(redundancy_df) == 0) {
    return(ggplot() +
             annotate("text", x = 0.5, y = 0.5,
                      label = "No redundancy data available", size = 5) +
             theme_void())
  }

  # Order by overall median R² (descending)
  ind_order <- redundancy_df |>
    dplyr::group_by(indicator_label) |>
    dplyr::summarise(med = median(redundancy_r2, na.rm = TRUE), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(med)) |>
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
    facet_wrap(~ecosystem, ncol = 1, scales = "free_x") +
    labs(
      x        = NULL,
      y        = expression("Redundancy index (R"^2*")"),
      title    = "Indicator importance for pixel selection across condition scenarios",
      subtitle = "Each box spans all 13 condition scenarios; higher = more explanatory power"
    ) +
    theme_minimal() +
    theme(
      axis.text.x        = element_text(angle = 30, hjust = 1, size = 9),
      strip.text         = element_text(face = "bold", size = 10),
      panel.grid.major.x = element_blank(),
      legend.position    = "bottom"
    )
}
