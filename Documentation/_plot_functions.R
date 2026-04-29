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
