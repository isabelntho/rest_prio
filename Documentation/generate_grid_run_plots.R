# generate_grid_run_plots.R
# ─────────────────────────────────────────────────────────────────────────────
# Generates and saves all visualisation plots for ONE grid optimisation run.
# Run this script once per grid configuration; plots are saved under:
#   Documentation/notebook_outputs/grid_plots/<RUN_SLUG>/figures/
#
# Working directory should be Documentation/ (same folder as _plot_functions.R).
# Suggested usage (from Documentation/ in a terminal):
#   Rscript generate_grid_run_plots.R
#
# Run slugs and their suggested configurations:
#   original3_objs   a/b/c objectives, pixel approach
#   abc_grid  a/b/c objectives, grid / planning-unit approach
#   r_inputs\20260518_1005_restpo_LC_grid     r/l/c objectives, pixel approach
#   rlc_grid   r/l/c objectives, grid / planning-unit approach
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. User configuration ─────────────────────────────────────────────────────
# Change these four lines for each run; everything else is derived automatically.

RUN_SLUG  <- "abc_grid"          # one of: original3_objs  abc_grid  rlc_px  rlc_grid
RUN_LABEL <- "a/b/c grid"        # human-readable label used in plot titles

GRID_DIR  <- "20260518_1005_restpo_LC_grid"   # directory name under ../r_inputs/
BASELINE_SEED <- 100L                          # seed used for the primary single-run display

# Seeds present in this grid run.
SEEDS <- c(seq(100L, 106L), 108, 109)

# Indicator exclusion scenarios (variable suffixes used in directory names).
DROP_VARS <- c(
  "smd", "sbd", "soc", "uzl", "tsd", "can", "cdi",
  "swf_h", "swf_t", "lai", "ndvi"
)

# Optional policy grid.  Set to NULL when there is no policy grid.
#POLICY_GRID_DIR <- NULL
POLICY_GRID_DIR <- "../r_inputs/20260518_1752_restpo_LC_polgrid"

# Ecosystem label used in plot titles only.
ECOSYSTEM <- "fg"

# ── RFOP / sensitivity thresholds ─────────────────────────────────────────────
high_priority_percentile     <- 70L
seed_sd_threshold            <- 10
sensitivity_range_percentile <- 75L
n_tradeoff_clusters          <- 4L
conditional_high_percentile  <- 70L
min_cluster_solutions        <- 5L
set.seed(123L)

# Colour scheme (mirrored in QMD).
COL_ND    <- "#E05C2A"
COL_DOM   <- "grey60"
ALPHA_ND  <- 0.85
ALPHA_DOM <- 0.15

# Objective name remapping: metadata key  CSV column name.
OBJECTIVE_NAME_MAP <- c("cost" = "implementation_cost")

# Set TRUE to write all spatial layers (RFOP, diagnostics, etc.) as GeoTIFF
# files to <OUTPUT_DIR>/spatial_tifs/.  Character/factor columns are encoded as
# integers; a companion *_lookup.csv maps codes back to their labels.
SAVE_SPATIAL_TIFS <- TRUE

# ── 2. Output directories ─────────────────────────────────────────────────────
OUTPUT_DIR <- file.path("notebook_outputs", "grid_plots", RUN_SLUG)
FIGS_DIR   <- file.path(OUTPUT_DIR, "figures")
TABLE_DIR  <- file.path(OUTPUT_DIR, "tables")
TIFS_DIR   <- file.path(OUTPUT_DIR, "spatial_tifs")
dir.create(FIGS_DIR,   recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR,  recursive = TRUE, showWarnings = FALSE)
if (SAVE_SPATIAL_TIFS) dir.create(TIFS_DIR, recursive = TRUE, showWarnings = FALSE)

# ── 3. Packages ───────────────────────────────────────────────────────────────
pkgs <- c(
  "jsonlite", "dplyr", "tidyr", "purrr", "ggplot2", "scales",
  "viridis", "stringr", "tibble", "forcats", "patchwork",
  "gridExtra", "grid", "sf", "terra", "hexbin", "ggridges"
)
missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing_pkgs)) stop("Missing packages: ", paste(missing_pkgs, collapse = ", "))
invisible(lapply(pkgs, library, character.only = TRUE))

# Use data.table::fread when available for faster CSV reading.
# Install with: install.packages("data.table")
USE_FREAD <- requireNamespace("data.table", quietly = TRUE)
if (USE_FREAD) message("data.table available — using fread for fast CSV reading.")

theme_set(
  theme_minimal(base_size = 11L) +
    theme(
      plot.title    = element_text(face = "bold"),
      plot.subtitle = element_text(colour = "grey40"),
      legend.position = "right"
    )
)

theme_map <- theme_void() +
  theme(
    panel.background = element_rect(fill = "#e6e6e6", colour = NA),
    strip.text  = element_text(face = "bold"),
    plot.title  = element_text(face = "bold"),
    legend.position = "right"
  )

source("_plot_functions.R")

# ── 4. Derived paths ──────────────────────────────────────────────────────────
GRID_BASE_DIR    <- file.path("../r_inputs", GRID_DIR)
BASELINE_RUN_DIR <- file.path(GRID_BASE_DIR, paste0("global_all_seed", BASELINE_SEED))

scenario_names <- c(
  paste0("global_all_seed",          SEEDS),
  unlist(lapply(DROP_VARS, \(x) paste0("global_drop_", x, "_seed", SEEDS))),
  paste0("upper_q75_all_seed",       SEEDS)
)

SCENARIO_DIRS <- setNames(
  file.path(GRID_BASE_DIR, scenario_names),
  scenario_names
)

SEED_RUNS <- as.list(SCENARIO_DIRS[grepl("^global_all_seed",  names(SCENARIO_DIRS))])
names(SEED_RUNS) <- sub("global_all_", "", names(SEED_RUNS))

INDICATOR_RUNS <- as.list(SCENARIO_DIRS[!grepl("^upper_q75", names(SCENARIO_DIRS))])

BENCHMARK_RUNS <- as.list(c(
  SCENARIO_DIRS[grepl("^global_all_seed", names(SCENARIO_DIRS))],
  SCENARIO_DIRS[grepl("^upper_q75",       names(SCENARIO_DIRS))]
))

POLICY_RUNS <- if (!is.null(POLICY_GRID_DIR) && dir.exists(POLICY_GRID_DIR)) {
  pol_dirs <- list.dirs(POLICY_GRID_DIR, full.names = TRUE, recursive = FALSE)
  pol_dirs <- pol_dirs[grepl("policy_", basename(pol_dirs))]
  as.list(setNames(pol_dirs, basename(pol_dirs)))
} else {
  NULL
}

# ── 5. Helper: save a plot ─────────────────────────────────────────────────────
save_plot <- function(p, name, width = 12, height = 6, dpi = 250) {
  if (is.null(p)) {
    message("  [skip] ", name, " — plot is NULL")
    return(invisible(NULL))
  }
  path <- file.path(FIGS_DIR, paste0(name, ".png"))
  ggplot2::ggsave(path, plot = p, width = width, height = height, dpi = dpi)
  message("  saved: ", path)
  invisible(path)
}

# ── 5b. Helper: save a data frame with x/y columns as a GeoTIFF ─────────────
# value_cols: columns to write as raster bands (NULL = all non-x/y columns).
# crs_ref:    a terra::SpatRaster whose CRS is applied; NULL leaves CRS unset.
# Factor/character columns are integer-encoded; a *_lookup.csv is written for
# each such column so labels can be recovered.
save_spatial_tif <- function(df, name, value_cols = NULL, crs_ref = NULL) {
  if (!SAVE_SPATIAL_TIFS || is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  if (is.null(value_cols)) value_cols <- setdiff(names(df), c("x", "y"))
  # Only keep columns that actually exist in df.
  value_cols  <- intersect(value_cols, names(df))
  if (length(value_cols) == 0L) return(invisible(NULL))
  encoded     <- df[, c("x", "y"), drop = FALSE]
  lookup_list <- list()
  for (col in value_cols) {
    v <- df[[col]]
    if (is.character(v) || is.factor(v)) {
      lvls              <- sort(unique(v[!is.na(v)]))
      encoded[[col]]    <- match(v, lvls)
      lookup_list[[col]] <- data.frame(code = seq_along(lvls), label = lvls)
    } else {
      encoded[[col]] <- as.numeric(v)
    }
  }
  r <- terra::rast(encoded[, c("x", "y", value_cols), drop = FALSE], type = "xyz")
  if (!is.null(crs_ref)) terra::crs(r) <- terra::crs(crs_ref)
  path <- file.path(TIFS_DIR, paste0(name, ".tif"))
  terra::writeRaster(r, path, overwrite = TRUE)
  message("  saved tif: ", path)
  if (length(lookup_list) > 0L) {
    lookup_path <- file.path(TIFS_DIR, paste0(name, "_lookup.csv"))
    lookup_df   <- purrr::imap_dfr(lookup_list,
                     ~ dplyr::mutate(.x, layer = .y, .before = 1L))
    utils::write.csv(lookup_df, lookup_path, row.names = FALSE)
    message("  saved lookup: ", lookup_path)
  }
  invisible(path)
}

# ── 6. Helper: load_run_data (used by _plot_functions.R) ─────────────────────
# Mirrors the load_run_data() function defined inline in the QMD.
load_run_data <- function(dir_path, label) {
  dir_path <- normalizePath(dir_path, mustWork = TRUE)

  meta   <- jsonlite::read_json(file.path(dir_path, "metadata.json"))
  df_obj <- read.csv(file.path(dir_path, "objectives.csv"))

  norm_path <- file.path(dir_path, "objectives_normalized.csv")
  df_norm   <- if (file.exists(norm_path)) read.csv(norm_path) else NULL

  hv_path  <- file.path(dir_path, "hypervolume_evolution.csv")
  df_hv    <- if (file.exists(hv_path)) read.csv(hv_path) else NULL

  pop_path <- file.path(dir_path, "population_stats.csv")
  df_pop   <- if (file.exists(pop_path)) read.csv(pop_path) else NULL

  psel_path <- file.path(dir_path, "pixel_selection.csv")
  df_psel   <- if (file.exists(psel_path)) read.csv(psel_path) else NULL

  elig_path <- file.path(dir_path, "eligible_pixels.csv")
  df_elig   <- if (file.exists(elig_path)) read.csv(elig_path) else NULL

  obj_names    <- unlist(meta$objective_names)
  raster_shape <- unlist(meta$raster_info$shape)

  norm_scales <- tryCatch(
    unlist(meta$algorithm$objective_normalization$scales),
    error = function(e) NULL
  )

  invert_cols <- grep(
    "anomaly|connectivity_gain|restoration_potential|landscape_context",
    obj_names, value = TRUE
  )
  if (length(invert_cols) > 0L) {
    df_obj[, invert_cols] <- -df_obj[, invert_cols]
    if (!is.null(df_norm))
      df_norm[, intersect(invert_cols, names(df_norm))] <-
        -df_norm[, intersect(invert_cols, names(df_norm))]
  }

  list(
    label        = label,
    dir          = dir_path,
    meta         = meta,
    obj_names    = obj_names,
    n_obj        = length(obj_names),
    n_solutions  = meta$n_solutions,
    n_nondom     = meta$n_nondominated_solutions,
    raster_shape = raster_shape,
    norm_scales  = norm_scales,
    df_obj       = df_obj,
    df_norm      = df_norm,
    df_hv        = df_hv,
    df_pop       = df_pop,
    df_psel      = df_psel,
    df_elig      = df_elig
  )
}

# ── 7. Grid-analysis helper functions ─────────────────────────────────────────
# Mirrors the helper-functions chunk in the QMD.

read_csv_if_exists <- function(path) {
  if (!file.exists(path)) return(NULL)
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

# Fast column-selective reader: uses fread when available, falls back to read.csv.
# `cols` is a character vector of column names to keep (NULL = all columns).
read_csv_fast <- function(path, cols = NULL) {
  if (!file.exists(path)) return(NULL)
  if (USE_FREAD) {
    dt <- data.table::fread(path, select = cols, data.table = FALSE)
    return(dt)
  }
  df <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!is.null(cols)) df <- df[, intersect(cols, names(df)), drop = FALSE]
  df
}

write_csv_base <- function(x, path) {
  utils::write.csv(x, path, row.names = FALSE)
}

resolve_objective_names <- function(obj_names, available_names) {
  mapped  <- dplyr::recode(obj_names, !!!OBJECTIVE_NAME_MAP)
  missing <- setdiff(mapped, available_names)
  if (length(missing) > 0L)
    stop("Objective columns not found: ", paste(missing, collapse = ", "),
         "\nAvailable: ", paste(available_names, collapse = ", "))
  mapped
}

load_run <- function(dir_path, label = basename(dir_path)) {
  dir_path  <- normalizePath(dir_path, mustWork = TRUE)
  meta_path <- file.path(dir_path, "metadata.json")
  if (!file.exists(meta_path)) stop("Missing metadata.json in ", dir_path)

  meta              <- jsonlite::read_json(meta_path)
  objectives        <- read_csv_if_exists(file.path(dir_path, "objectives.csv"))
  objectives_norm   <- read_csv_if_exists(file.path(dir_path, "objectives_normalized.csv"))
  pixel_selection   <- read_csv_if_exists(file.path(dir_path, "pixel_selection.csv"))
  eligible_pixels   <- read_csv_if_exists(file.path(dir_path, "eligible_pixels.csv"))
  obj_names         <- unlist(meta$objective_names)

  if (is.null(objectives))      stop("Missing objectives.csv in ",     dir_path)
  if (is.null(pixel_selection)) stop("Missing pixel_selection.csv in ", dir_path)

  list(
    label           = label,
    dir             = dir_path,
    meta            = meta,
    obj_names       = obj_names,
    objectives      = objectives,
    objectives_norm = objectives_norm,
    pixel_selection = pixel_selection,
    eligible_pixels = eligible_pixels
  )
}

get_nd_ids <- function(objectives) {
  if ("is_nondominated" %in% names(objectives)) {
    objectives |>
      dplyr::filter(.data$is_nondominated == 1L) |>
      dplyr::pull(.data$solution_id)
  } else {
    objectives |> dplyr::pull(.data$solution_id)
  }
}

orient_objective_matrix <- function(df, obj_names) {
  out <- df[, obj_names, drop = FALSE]
  for (nm in obj_names) {
    if (grepl("anomaly|connectivity_gain|restoration_potential", nm, ignore.case = TRUE))
      out[[nm]] <- -out[[nm]]
    else if (grepl("cost", nm, ignore.case = TRUE))
      out[[nm]] <- -out[[nm]]
  }
  out
}

scale_objective_matrix <- function(mat) {
  scaled <- scale(mat)
  scaled[!is.finite(scaled)] <- 0
  scaled
}

complete_rfop_grid <- function(runs_df) {
  all_pixels <- runs_df |> dplyr::distinct(.data$x, .data$y)
  all_runs   <- runs_df |> dplyr::distinct(.data$dim_type, .data$group_label)
  tidyr::crossing(all_pixels, all_runs) |>
    dplyr::left_join(runs_df, by = c("x", "y", "dim_type", "group_label")) |>
    dplyr::mutate(
      rfop_pct    = tidyr::replace_na(.data$rfop_pct,    0),
      n_selected  = tidyr::replace_na(.data$n_selected,  0),
      n_solutions = tidyr::replace_na(.data$n_solutions, 0),
      run_label   = dplyr::if_else(is.na(.data$run_label), .data$group_label, .data$run_label)
    )
}

load_run_rfop <- function(dir_path, label, dim_type, group_label = label) {
  # Lightweight version: only reads the two columns needed for RFOP from each
  # CSV, avoiding the cost of loading objectives_norm / eligible_pixels etc.
  dir_path <- normalizePath(dir_path, mustWork = TRUE)

  obj <- read_csv_fast(
    file.path(dir_path, "objectives.csv"),
    cols = c("solution_id", "is_nondominated")
  )
  if (is.null(obj)) stop("Missing objectives.csv in ", dir_path)

  nd_ids <- get_nd_ids(obj)
  n_nd   <- length(unique(nd_ids))
  if (n_nd == 0L) stop("No non-dominated solution IDs found in ", dir_path)

  psel <- read_csv_fast(
    file.path(dir_path, "pixel_selection.csv"),
    cols = c("solution_id", "x", "y")
  )
  if (is.null(psel)) stop("Missing pixel_selection.csv in ", dir_path)

  psel[psel$solution_id %in% nd_ids, ] |>
    dplyr::distinct(.data$solution_id, .data$x, .data$y) |>
    dplyr::count(.data$x, .data$y, name = "n_selected") |>
    dplyr::mutate(
      rfop_pct    = 100 * .data$n_selected / n_nd,
      n_solutions = n_nd,
      run_label   = label,
      dim_type    = dim_type,
      group_label = group_label,
      .before     = 1L
    )
}

load_dim_runs <- function(run_dirs, dim_type) {
  if (is.null(run_dirs) || length(run_dirs) == 0L) return(tibble::tibble())
  purrr::map_dfr(names(run_dirs), function(nm) {
    d <- run_dirs[[nm]]
    if (!dir.exists(d)) {
      message("  [skip] ", nm, " — directory not found")
      return(tibble::tibble())
    }
    tryCatch(
      load_run_rfop(d, nm, dim_type, nm),
      error = function(e) {
        message("  [skip] ", nm, ": ", conditionMessage(e))
        tibble::tibble()
      }
    )
  })
}

compute_global_rfop_diagnostics <- function(runs_df, high_priority_percentile = 70L) {
  if (is.null(runs_df) || nrow(runs_df) == 0L) return(tibble::tibble())

  runs_complete <- complete_rfop_grid(runs_df)

  dim_summary <- runs_complete |>
    dplyr::group_by(.data$x, .data$y, .data$dim_type) |>
    dplyr::summarise(
      dim_mean_RFOP           = mean(.data$rfop_pct, na.rm = TRUE),
      dim_median_RFOP         = stats::median(.data$rfop_pct, na.rm = TRUE),
      dim_max_RFOP            = max(.data$rfop_pct, na.rm = TRUE),
      dim_min_RFOP            = min(.data$rfop_pct, na.rm = TRUE),
      dim_range_RFOP          = .data$dim_max_RFOP - .data$dim_min_RFOP,
      dim_sd_RFOP             = stats::sd(.data$rfop_pct, na.rm = TRUE),
      dim_n_runs_selected     = sum(.data$rfop_pct > 0, na.rm = TRUE),
      dim_n_runs_total        = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::mutate(dim_selection_consistency = .data$dim_n_runs_selected / .data$dim_n_runs_total) |>
    tidyr::pivot_wider(
      names_from  = .data$dim_type,
      values_from = c(
        .data$dim_mean_RFOP, .data$dim_median_RFOP, .data$dim_max_RFOP,
        .data$dim_min_RFOP,  .data$dim_range_RFOP,  .data$dim_sd_RFOP,
        .data$dim_n_runs_selected, .data$dim_n_runs_total, .data$dim_selection_consistency
      ),
      values_fill = NA_real_
    )

  global_summary <- runs_complete |>
    dplyr::group_by(.data$x, .data$y) |>
    dplyr::summarise(
      mean_global_RFOP        = mean(.data$rfop_pct, na.rm = TRUE),
      median_global_RFOP      = stats::median(.data$rfop_pct, na.rm = TRUE),
      max_global_RFOP         = max(.data$rfop_pct, na.rm = TRUE),
      min_global_RFOP         = min(.data$rfop_pct, na.rm = TRUE),
      sd_global_RFOP          = stats::sd(.data$rfop_pct, na.rm = TRUE),
      n_runs_selected_global  = sum(.data$rfop_pct > 0, na.rm = TRUE),
      n_runs_total_global     = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      global_selection_consistency = .data$n_runs_selected_global / .data$n_runs_total_global
    )

  positive_mean  <- global_summary$mean_global_RFOP[global_summary$mean_global_RFOP > 0]
  global_cutoff  <- stats::quantile(positive_mean, probs = high_priority_percentile / 100,
                                    na.rm = TRUE)

  out <- global_summary |>
    dplyr::mutate(high_global_priority = .data$mean_global_RFOP >= global_cutoff) |>
    dplyr::left_join(dim_summary, by = c("x", "y"))

  attr(out, "global_cutoff") <- global_cutoff
  attr(out, "runs_complete") <- runs_complete
  out
}

ensure_numeric_col <- function(df, nm) {
  if (!nm %in% names(df)) df[[nm]] <- NA_real_
  df
}

add_sensitivity_flags <- function(df, seed_sd_threshold = 10,
                                  sensitivity_range_percentile = 75L) {
  needed <- c("dim_sd_RFOP_seed", "dim_range_RFOP_indicator",
               "dim_range_RFOP_benchmark", "dim_range_RFOP_policy")
  for (nm in needed) df <- ensure_numeric_col(df, nm)

  indicator_cutoff <- stats::quantile(df$dim_range_RFOP_indicator,
    probs = sensitivity_range_percentile / 100, na.rm = TRUE)
  benchmark_cutoff <- stats::quantile(df$dim_range_RFOP_benchmark,
    probs = sensitivity_range_percentile / 100, na.rm = TRUE)
  policy_cutoff    <- stats::quantile(df$dim_range_RFOP_policy,
    probs = sensitivity_range_percentile / 100, na.rm = TRUE)

  out <- df |>
    dplyr::mutate(
      seed_stable = dplyr::if_else(
        is.na(.data$dim_sd_RFOP_seed), TRUE,
        .data$dim_sd_RFOP_seed <= seed_sd_threshold
      ),
      indicator_sensitive = dplyr::if_else(
        is.na(.data$dim_range_RFOP_indicator), FALSE,
        .data$dim_range_RFOP_indicator >= indicator_cutoff
      ),
      benchmark_sensitive = dplyr::if_else(
        is.na(.data$dim_range_RFOP_benchmark), FALSE,
        .data$dim_range_RFOP_benchmark >= benchmark_cutoff
      ),
      policy_sensitive = dplyr::if_else(
        is.na(.data$dim_range_RFOP_policy), FALSE,
        .data$dim_range_RFOP_policy >= policy_cutoff
      ),
      sensitivity_stable =
        .data$seed_stable &
        !.data$indicator_sensitive &
        !.data$benchmark_sensitive &
        !.data$policy_sensitive
    )

  attr(out, "seed_cutoff")      <- seed_sd_threshold
  attr(out, "indicator_cutoff") <- indicator_cutoff
  attr(out, "benchmark_cutoff") <- benchmark_cutoff
  attr(out, "policy_cutoff")    <- policy_cutoff
  out
}

load_solution_profiles <- function(run_dirs) {
  purrr::map_dfr(names(run_dirs), function(nm) {
    if (!dir.exists(run_dirs[[nm]])) return(tibble::tibble())
    run    <- tryCatch(load_run(run_dirs[[nm]], nm), error = function(e) NULL)
    if (is.null(run)) return(tibble::tibble())
    nd_ids <- get_nd_ids(run$objectives)

    obj_names <- resolve_objective_names(
      obj_names       = run$obj_names,
      available_names = names(run$objectives)
    )

    invert_cols <- grep(
      "anomaly|connectivity_gain|restoration_potential|landscape_context",
      obj_names, value = TRUE
    )

    df <- run$objectives |>
      dplyr::filter(.data$solution_id %in% nd_ids) |>
      dplyr::select(.data$solution_id, dplyr::all_of(obj_names))

    if (length(invert_cols) > 0L) df[, invert_cols] <- -df[, invert_cols]

    df |> dplyr::mutate(run_label = nm, .before = 1L)
  })
}

compute_conditional_rfop <- function(run_dirs, solution_clusters) {
  purrr::map_dfr(names(run_dirs), function(nm) {
    if (!dir.exists(run_dirs[[nm]])) return(tibble::tibble())
    run <- tryCatch(load_run(run_dirs[[nm]], nm), error = function(e) NULL)
    if (is.null(run)) return(tibble::tibble())

    clusters_this_run <- solution_clusters |>
      dplyr::filter(.data$run_label == nm) |>
      dplyr::select(.data$solution_id, .data$tradeoff_label)

    purrr::map_dfr(unique(clusters_this_run$tradeoff_label), function(cl) {
      sol_ids    <- clusters_this_run |>
        dplyr::filter(.data$tradeoff_label == cl) |>
        dplyr::pull(.data$solution_id)
      n_solutions <- length(unique(sol_ids))

      run$pixel_selection |>
        dplyr::filter(.data$solution_id %in% sol_ids) |>
        dplyr::distinct(.data$solution_id, .data$x, .data$y) |>
        dplyr::count(.data$x, .data$y, name = "n_selected") |>
        dplyr::mutate(
          run_label      = nm,
          tradeoff_label = cl,
          n_solutions    = n_solutions,
          cond_rfop_pct  = 100 * .data$n_selected / n_solutions,
          .before = 1L
        )
    })
  })
}

# ── 8. Load baseline run data ─────────────────────────────────────────────────
message("\n=== ", RUN_LABEL, " ===")
message("Loading baseline run: ", BASELINE_RUN_DIR)
run_main <- load_run_data(BASELINE_RUN_DIR, RUN_LABEL)

# Compute seed union (first 3 seeds only, mirrors QMD behaviour).
UNION_SEEDS    <- SEEDS[seq_len(min(3L, length(SEEDS)))]
UNION_SEED_DIRS <- setNames(
  lapply(UNION_SEEDS, \(s) file.path(GRID_BASE_DIR, paste0("global_all_seed", s))),
  paste0("seed", UNION_SEEDS)
)

message("Computing seed union from seeds: ", paste(UNION_SEEDS, collapse = ", "))
run_union <- local({
  sid_offset <- 0L
  first_meta <- NULL
  all_obj    <- vector("list", length(UNION_SEED_DIRS))
  all_psel   <- vector("list", length(UNION_SEED_DIRS))

  for (i in seq_along(UNION_SEED_DIRS)) {
    sdir <- UNION_SEED_DIRS[[i]]
    if (!dir.exists(sdir)) {
      message("  [skip seed union] ", sdir)
      next
    }
    r <- load_run_data(sdir, names(UNION_SEED_DIRS)[i])
    if (is.null(first_meta)) first_meta <- r

    nd        <- r$df_obj[r$df_obj$is_nondominated %in% c(1L, TRUE), ]
    nd$solution_id <- nd$solution_id + sid_offset
    nd$seed_label  <- names(UNION_SEED_DIRS)[i]
    all_obj[[i]]   <- nd

    if (!is.null(r$df_psel)) {
      nd_ids         <- r$df_obj$solution_id[r$df_obj$is_nondominated %in% c(1L, TRUE)]
      ps             <- r$df_psel[r$df_psel$solution_id %in% nd_ids, ]
      ps$solution_id <- ps$solution_id + sid_offset
      all_psel[[i]]  <- ps
    }
    sid_offset <- sid_offset + max(r$df_obj$solution_id, na.rm = TRUE) + 1L
  }

  combined_obj  <- do.call(rbind, Filter(Negate(is.null), all_obj))
  combined_obj$is_nondominated <- 1L
  combined_psel <- if (all(vapply(all_psel, is.null, logical(1L)))) NULL
                   else do.call(rbind, Filter(Negate(is.null), all_psel))

  list(
    label        = paste0(RUN_LABEL, " — seed union"),
    obj_names    = first_meta$obj_names,
    n_obj        = first_meta$n_obj,
    n_solutions  = nrow(combined_obj),
    n_nondom     = nrow(combined_obj),
    raster_shape = first_meta$raster_shape,
    norm_scales  = NULL,
    df_obj       = combined_obj,
    df_norm      = NULL,
    df_hv        = NULL,
    df_pop       = NULL,
    df_psel      = combined_psel,
    df_elig      = first_meta$df_elig
  )
})

# ── 8b. Save baseline and seed-union RFOP as TIF ─────────────────────────────
if (SAVE_SPATIAL_TIFS && !is.null(run_main$df_psel)) {
  nd_ids_main  <- run_main$df_obj$solution_id[run_main$df_obj$is_nondominated %in% c(1L, TRUE)]
  n_nd_main    <- length(unique(nd_ids_main))
  rfop_main_df <- run_main$df_psel[run_main$df_psel$solution_id %in% nd_ids_main, ] |>
    dplyr::distinct(solution_id, x, y) |>
    dplyr::count(x, y, name = "n_selected") |>
    dplyr::mutate(rfop_pct = 100 * n_selected / n_nd_main)
  save_spatial_tif(rfop_main_df, "rfop_baseline_seed", c("rfop_pct", "n_selected"))
}

if (SAVE_SPATIAL_TIFS && !is.null(run_union$df_psel)) {
  n_nd_union    <- run_union$n_nondom
  rfop_union_df <- run_union$df_psel |>
    dplyr::distinct(solution_id, x, y) |>
    dplyr::count(x, y, name = "n_selected") |>
    dplyr::mutate(rfop_pct = 100 * n_selected / n_nd_union)
  save_spatial_tif(rfop_union_df, "rfop_seed_union", c("rfop_pct", "n_selected"))
}

# ── 9. Load raster data ───────────────────────────────────────────────────────
message("Loading raster data...")

lulc <- tryCatch(
  terra::rast("Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif"),
  error = function(e) { message("  [skip] lulc: ", conditionMessage(e)); NULL }
)

BE <- tryCatch({
  shp <- sf::st_read(
    "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp",
    quiet = TRUE
  )
  shp |> dplyr::filter(NAME == "Bern")
}, error = function(e) { message("  [skip] BE boundary: ", conditionMessage(e)); NULL })

abiotic_anomaly    <- tryCatch(terra::rast("../inputs/abiotic_condition_anomaly.tif"),
                                error = function(e) { message("  [skip] abiotic_anomaly"); NULL })
biotic_anomaly     <- tryCatch(terra::rast("../inputs/biotic_condition_anomaly.tif"),
                                error = function(e) { message("  [skip] biotic_anomaly"); NULL })
implementation_cost <- tryCatch(terra::rast("../inputs/implementation_cost.tif"),
                                 error = function(e) { message("  [skip] implementation_cost"); NULL })
landscape_context_rast <- tryCatch(terra::rast("../inputs/abiotic_condition_anomaly.tif"),
                                    error = function(e) NULL)

# ── 10. Per-run plots — baseline seed ─────────────────────────────────────────
message("\nGenerating per-run plots (baseline seed)...")

save_plot(make_hv_evo_plot(run_main), "hv_evolution", height = 8L)

save_plot(
  make_pareto_extremes_plot(run_main, obj_names = run_main$obj_names),
  "pareto_extremes", height = 5L
)

save_plot(
  make_pareto_pairwise(run_main, obj_names = run_main$obj_names),
  "pareto_pairwise", height = 6L
)

save_plot(
  make_corr_plot(run_main, obj_names = run_main$obj_names),
  "corr_plot", height = 4L, width = 8L
)

save_plot(
  make_parcoord_plot(run_main, obj_names = run_main$obj_names),
  "parcoord", height = 5L
)

save_plot(
  make_jaccard_plot(run_main, obj_names = run_main$obj_names),
  "jaccard", height = 4L, width = 6L
)

save_plot(make_sel_freq_plot(run_main),             "sel_freq_raster", height = 8L)
save_plot(make_sel_freq_plot_agg_hex(run_main, bins = 30L), "sel_freq_hex",    height = 8L)

if (!is.null(lulc))
  save_plot(
    make_sel_freq_unified_plot(run_main,
      fill_values = rfop_fill_lulc(run_main, lulc),
      fill_label  = "LULC Class",
      subtitle    = "Coloured by Land Use/Land Cover type"),
    "sel_freq_lulc", height = 5L
  )

if (!is.null(abiotic_anomaly))
  save_plot(
    make_sel_freq_unified_plot(run_main,
      fill_values = rfop_fill_objective(run_main, abiotic_anomaly),
      fill_label  = "Abiotic anomaly (quartile)",
      subtitle    = "Baseline abiotic condition anomaly"),
    "sel_freq_abiotic", height = 5L
  )

if (!is.null(biotic_anomaly))
  save_plot(
    make_sel_freq_unified_plot(run_main,
      fill_values = rfop_fill_objective(run_main, biotic_anomaly),
      fill_label  = "Biotic anomaly (quartile)",
      subtitle    = "Baseline biotic condition anomaly"),
    "sel_freq_biotic", height = 5L
  )

if (!is.null(implementation_cost))
  save_plot(
    make_sel_freq_unified_plot(run_main,
      fill_values = rfop_fill_objective(run_main, implementation_cost),
      fill_label  = "Implementation cost (quartile)",
      subtitle    = "Implementation cost"),
    "sel_freq_cost", height = 5L
  )

if (!is.null(landscape_context_rast))
  save_plot(
    make_sel_freq_unified_plot(run_main,
      fill_values = rfop_fill_objective(run_main, landscape_context_rast),
      fill_label  = "Landscape context (quartile)",
      subtitle    = "Mean abiotic anomaly of eligible neighbours (500 m radius)"),
    "sel_freq_lscontext", height = 5L
  )

# ── 11. Per-run plots — seed union ────────────────────────────────────────────
message("\nGenerating per-run plots (seed union)...")

save_plot(
  make_pareto_extremes_plot(run_union, obj_names = run_union$obj_names),
  "union_pareto_extremes", height = 5L
)
save_plot(
  make_pareto_pairwise(run_union, obj_names = run_union$obj_names),
  "union_pareto_pairwise", height = 6L
)
save_plot(
  make_corr_plot(run_union, obj_names = run_union$obj_names),
  "union_corr_plot", height = 4L, width = 8L
)
save_plot(
  make_parcoord_plot(run_union, obj_names = run_union$obj_names),
  "union_parcoord", height = 5L
)
save_plot(
  make_jaccard_plot(run_union, obj_names = run_union$obj_names),
  "union_jaccard", height = 4L, width = 6L
)

# ── 12. Baseline conditional RFOP (trade-off clusters) ───────────────────────
message("\nComputing baseline solution profiles...")

baseline_solution_profiles <- load_solution_profiles(SEED_RUNS)

if (nrow(baseline_solution_profiles) > 0L) {
  obj_names_grid <- setdiff(names(baseline_solution_profiles), c("run_label", "solution_id"))

  oriented_obj <- orient_objective_matrix(baseline_solution_profiles, obj_names_grid)
  scaled_obj   <- scale_objective_matrix(oriented_obj)

  km <- stats::kmeans(scaled_obj, centers = n_tradeoff_clusters,
                      nstart = 50L, iter.max = 100L)

  baseline_solution_profiles <- baseline_solution_profiles |>
    dplyr::mutate(tradeoff_cluster = paste0("cluster_", km$cluster))

  cluster_sizes <- baseline_solution_profiles |>
    dplyr::count(.data$tradeoff_cluster, name = "n_solutions")

  if (any(cluster_sizes$n_solutions < min_cluster_solutions))
    warning("At least one trade-off cluster has fewer than min_cluster_solutions solutions.")

  cluster_means <- baseline_solution_profiles |>
    dplyr::bind_cols(
      as.data.frame(scaled_obj) |> dplyr::rename_with(~ paste0("z_", .x))
    ) |>
    dplyr::group_by(.data$tradeoff_cluster) |>
    dplyr::summarise(dplyr::across(dplyr::starts_with("z_"), mean, na.rm = TRUE),
                     .groups = "drop")

  cluster_best <- cluster_means |>
    tidyr::pivot_longer(cols = dplyr::starts_with("z_"),
                        names_to = "objective", values_to = "mean_z") |>
    dplyr::group_by(.data$tradeoff_cluster) |>
    dplyr::slice_max(.data$mean_z, n = 1L, with_ties = FALSE) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      objective = sub("^z_", "", .data$objective),
      strategy_label = dplyr::case_when(
        grepl("implementation_cost|cost", .data$objective, ignore.case = TRUE) ~ "low cost oriented",
        grepl("surround|context|landscape",  .data$objective, ignore.case = TRUE) ~ "surrounding condition oriented",
        grepl("condition|potential|restoration", .data$objective, ignore.case = TRUE) ~ "condition improvement oriented",
        TRUE ~ paste("oriented to", .data$objective)
      ),
      tradeoff_label = paste(.data$tradeoff_cluster, .data$strategy_label, sep = ": ")
    ) |>
    dplyr::select(.data$tradeoff_cluster, .data$strategy_label, .data$tradeoff_label)

  baseline_solution_profiles <- baseline_solution_profiles |>
    dplyr::left_join(cluster_best, by = "tradeoff_cluster")

  write_csv_base(baseline_solution_profiles,
                 file.path(TABLE_DIR, "baseline_solution_profiles.csv"))
  write_csv_base(
    cluster_sizes |> dplyr::left_join(cluster_best, by = "tradeoff_cluster"),
    file.path(TABLE_DIR, "baseline_tradeoff_cluster_sizes.csv")
  )

  # Trade-off cluster facets plot.
  objective_pairs <- utils::combn(obj_names_grid, 2L, simplify = FALSE)
  facet_df <- purrr::map_dfr(objective_pairs, function(pair) {
    tibble::tibble(
      x_value       = baseline_solution_profiles[[pair[1L]]],
      y_value       = baseline_solution_profiles[[pair[2L]]],
      tradeoff_label = baseline_solution_profiles$tradeoff_label,
      pair_label    = paste(
        stringr::str_replace_all(pair[1L], "_", " "), "vs",
        stringr::str_replace_all(pair[2L], "_", " ")
      )
    )
  })

  p_tradeoff_facets <- ggplot(facet_df,
      aes(x = x_value, y = y_value, colour = tradeoff_label)) +
    geom_point(size = 2L, alpha = 0.8) +
    facet_wrap(~ pair_label, scales = "free") +
    labs(title = "Baseline Pareto trade-off regions",
         x = "Objective value", y = "Objective value",
         colour = "Trade-off region") +
    theme(legend.position = "bottom")

  save_plot(p_tradeoff_facets, "baseline_tradeoff_clusters", width = 12L, height = 8L)

  # ── 13. Conditional RFOP ───────────────────────────────────────────────────
  message("Computing conditional RFOP...")
  baseline_cond_rfop_raw <- compute_conditional_rfop(SEED_RUNS, baseline_solution_profiles)

  if (nrow(baseline_cond_rfop_raw) > 0L) {
    all_baseline_pixels <- baseline_cond_rfop_raw |> dplyr::distinct(.data$x, .data$y)
    all_tradeoffs       <- baseline_solution_profiles |> dplyr::distinct(.data$tradeoff_label)

    baseline_cond_rfop <- baseline_cond_rfop_raw |>
      dplyr::group_by(.data$x, .data$y, .data$tradeoff_label) |>
      dplyr::summarise(
        cond_rfop_pct = mean(.data$cond_rfop_pct, na.rm = TRUE),
        n_seed_runs   = dplyr::n_distinct(.data$run_label),
        .groups = "drop"
      ) |>
      dplyr::right_join(
        tidyr::crossing(all_baseline_pixels, all_tradeoffs),
        by = c("x", "y", "tradeoff_label")
      ) |>
      dplyr::mutate(
        cond_rfop_pct = tidyr::replace_na(.data$cond_rfop_pct, 0),
        n_seed_runs   = tidyr::replace_na(.data$n_seed_runs,   0)
      )

    cond_cutoffs <- baseline_cond_rfop |>
      dplyr::filter(.data$cond_rfop_pct > 0) |>
      dplyr::group_by(.data$tradeoff_label) |>
      dplyr::summarise(
        conditional_cutoff = stats::quantile(.data$cond_rfop_pct,
          conditional_high_percentile / 100L, na.rm = TRUE),
        .groups = "drop"
      )

    baseline_cond_rfop <- baseline_cond_rfop |>
      dplyr::left_join(cond_cutoffs, by = "tradeoff_label") |>
      dplyr::mutate(
        high_conditional = .data$cond_rfop_pct >= .data$conditional_cutoff & .data$cond_rfop_pct > 0
      )

    baseline_strategy_summary <- baseline_cond_rfop |>
      dplyr::group_by(.data$x, .data$y) |>
      dplyr::summarise(
        max_cond_rfop           = max(.data$cond_rfop_pct, na.rm = TRUE),
        mean_cond_rfop          = mean(.data$cond_rfop_pct, na.rm = TRUE),
        n_high_tradeoff_regions = sum(.data$high_conditional, na.rm = TRUE),
        best_tradeoff_region    = .data$tradeoff_label[which.max(.data$cond_rfop_pct)],
        .groups = "drop"
      ) |>
      dplyr::mutate(
        baseline_strategy_class = dplyr::case_when(
          .data$n_high_tradeoff_regions >= 2L ~ "Robust",
          .data$n_high_tradeoff_regions == 1L ~ "Objective specific",
          TRUE ~ "Not robust"
        )
      )

    write_csv_base(baseline_cond_rfop,
                   file.path(TABLE_DIR, "baseline_conditional_rfop_by_tradeoff_region.csv"))
    write_csv_base(baseline_strategy_summary,
                   file.path(TABLE_DIR, "baseline_strategy_robustness_by_pixel.csv"))

    # Conditional RFOP: pivot to wide (one band per trade-off region) and save.
    if (SAVE_SPATIAL_TIFS && nrow(baseline_cond_rfop) > 0L) {
      cond_rfop_wide <- baseline_cond_rfop |>
        dplyr::select(x, y, tradeoff_label, cond_rfop_pct) |>
        tidyr::pivot_wider(names_from  = tradeoff_label,
                           values_from = cond_rfop_pct,
                           values_fill = 0)
      save_spatial_tif(cond_rfop_wide, "rfop_conditional_by_tradeoff")
      save_spatial_tif(baseline_strategy_summary, "rfop_strategy_summary",
        value_cols = c("max_cond_rfop", "mean_cond_rfop",
                       "n_high_tradeoff_regions", "best_tradeoff_region",
                       "baseline_strategy_class"))
    }

    p_conditional_rfop <- ggplot(baseline_cond_rfop,
        aes(x = x, y = y, fill = cond_rfop_pct)) +
      geom_raster() +
      facet_wrap(~ tradeoff_label) +
      coord_equal() +
      scale_fill_viridis_c(name = "Conditional RFOP (%)", option = "magma") +
      theme_map

    p_strategy_class <- ggplot(baseline_strategy_summary,
        aes(x = x, y = y, fill = baseline_strategy_class)) +
      geom_raster() +
      coord_equal() +
      scale_fill_viridis_d(name = "Class", option = "viridis", na.value = "grey90") +
      labs(title = "Baseline strategy robustness from conditional RFOP") +
      theme_map

    save_plot(p_conditional_rfop, "conditional_rfop_maps",    width = 14L, height = 9L)
    save_plot(p_strategy_class,   "strategy_robustness_map",  width = 12L, height = 6L)
  }
} else {
  message("No baseline solution profiles loaded — skipping cluster/conditional RFOP plots.")
  baseline_strategy_summary <- NULL
}

# ── 14. Global RFOP diagnostics ───────────────────────────────────────────────
message("\nLoading sensitivity runs for global RFOP diagnostics...")

runs_all_df <- dplyr::bind_rows(
  load_dim_runs(INDICATOR_RUNS, "indicator"),
  load_dim_runs(BENCHMARK_RUNS, "benchmark"),
  load_dim_runs(POLICY_RUNS,    "policy"),
  load_dim_runs(SEED_RUNS,      "seed")
)

if (nrow(runs_all_df) > 0L) {
  sens_df <- compute_global_rfop_diagnostics(runs_all_df, high_priority_percentile)
  global_rfop_cutoff <- attr(sens_df, "global_cutoff")

  sens_df <- add_sensitivity_flags(sens_df, seed_sd_threshold, sensitivity_range_percentile)
  write_csv_base(sens_df, file.path(TABLE_DIR, "global_rfop_diagnostics_by_pixel.csv"))

  save_spatial_tif(sens_df, "rfop_global_diagnostics",
    value_cols = c("mean_global_RFOP", "median_global_RFOP", "sd_global_RFOP",
                   "high_global_priority", "seed_stable",
                   "indicator_sensitive", "benchmark_sensitive",
                   "policy_sensitive", "sensitivity_stable"),
    crs_ref = abiotic_anomaly)

  p_global_rfop <- sens_df |>
    dplyr::mutate(mean_global_RFOP_plot = dplyr::na_if(mean_global_RFOP, 0)) |>
    ggplot(aes(x = x, y = y, fill = mean_global_RFOP_plot)) +
    geom_raster() + coord_equal() +
    scale_fill_viridis_c(name = "Mean RFOP (%)", option = "viridis", na.value = "grey85") +
    labs(title = "Global RFOP across sensitivity runs",
         subtitle = paste0("High priority cutoff = ", round(global_rfop_cutoff, 1L), "%")) +
    theme_map

  p_high_global <- ggplot(sens_df, aes(x = x, y = y, fill = high_global_priority)) +
    geom_raster() + coord_equal() +
    scale_fill_manual(values = c("TRUE" = "black", "FALSE" = "grey85"),
                      name = "High global priority") +
    labs(title = "High global priority areas") +
    theme_map

  save_plot(p_global_rfop,  "global_rfop_map",          width = 12L, height = 6L)
  save_plot(p_high_global,  "high_global_priority_map", width = 12L, height = 6L)

  # Sensitivity flags map.
  flag_cols <- c("seed_stable", "indicator_sensitive", "benchmark_sensitive", "policy_sensitive")
  sensitivity_flags_long <- sens_df |>
    dplyr::select(x, y, dplyr::all_of(flag_cols)) |>
    tidyr::pivot_longer(dplyr::all_of(flag_cols),
                        names_to = "diagnostic", values_to = "flag") |>
    dplyr::mutate(
      diagnostic = dplyr::recode(diagnostic,
        seed_stable           = "Seed stable",
        indicator_sensitive   = "Indicator sensitive",
        benchmark_sensitive   = "Benchmark sensitive",
        policy_sensitive      = "Policy sensitive"),
      flag = factor(flag, levels = c(FALSE, TRUE))
    )

  p_sens_flags <- ggplot(sensitivity_flags_long, aes(x, y, fill = flag)) +
    geom_raster() + coord_equal() +
    facet_wrap(~ diagnostic, ncol = 2L) +
    scale_fill_manual(values = c("FALSE" = "grey85", "TRUE" = "black"),
                      labels = c("No", "Yes"), name = NULL, drop = FALSE) +
    theme_map +
    theme(strip.text = element_text(face = "bold"), legend.position = "bottom")

  save_plot(p_sens_flags, "sensitivity_diagnostics_map", width = 12L, height = 6L)

  # ── 15. Persistence of baseline robust areas ─────────────────────────────────
  if (!is.null(baseline_strategy_summary)) {
    message("Computing baseline persistence...")

    sens_with_baseline <- sens_df |>
      dplyr::left_join(baseline_strategy_summary, by = c("x", "y")) |>
      dplyr::mutate(
        baseline_strategy_class = tidyr::replace_na(.data$baseline_strategy_class, "Not robust"),
        no_regret_priority =
          .data$baseline_strategy_class == "baseline broadly robust" &
          .data$high_global_priority &
          .data$sensitivity_stable,
        robust_but_indicator_sensitive =
          .data$baseline_strategy_class %in% c("baseline broadly robust", "baseline strategy specific") &
          .data$indicator_sensitive,
        robust_but_benchmark_sensitive =
          .data$baseline_strategy_class %in% c("baseline broadly robust", "baseline strategy specific") &
          .data$benchmark_sensitive,
        robust_but_policy_sensitive =
          .data$baseline_strategy_class %in% c("baseline broadly robust", "baseline strategy specific") &
          .data$policy_sensitive
      )

    persistence_summary <- sens_with_baseline |>
      dplyr::group_by(.data$baseline_strategy_class) |>
      dplyr::summarise(
        n_pixels                = dplyr::n(),
        mean_global_RFOP        = mean(.data$mean_global_RFOP, na.rm = TRUE),
        pct_high_global_priority = 100 * mean(.data$high_global_priority, na.rm = TRUE),
        pct_seed_stable         = 100 * mean(.data$seed_stable, na.rm = TRUE),
        pct_indicator_sensitive = 100 * mean(.data$indicator_sensitive, na.rm = TRUE),
        pct_benchmark_sensitive = 100 * mean(.data$benchmark_sensitive, na.rm = TRUE),
        pct_policy_sensitive    = 100 * mean(.data$policy_sensitive, na.rm = TRUE),
        pct_sensitivity_stable  = 100 * mean(.data$sensitivity_stable, na.rm = TRUE),
        pct_no_regret           = 100 * mean(.data$no_regret_priority, na.rm = TRUE),
        .groups = "drop"
      )

    write_csv_base(sens_with_baseline,  file.path(TABLE_DIR, "sensitivity_with_baseline_strategy_classes.csv"))
    write_csv_base(persistence_summary, file.path(TABLE_DIR, "baseline_strategy_persistence_summary.csv"))

    save_spatial_tif(sens_with_baseline, "rfop_sensitivity_with_baseline",
      value_cols = c("mean_global_RFOP", "high_global_priority",
                     "baseline_strategy_class", "sensitivity_stable",
                     "seed_stable", "indicator_sensitive",
                     "benchmark_sensitive", "policy_sensitive",
                     "no_regret_priority"),
      crs_ref = abiotic_anomaly)

    p_persistence_bar <- persistence_summary |>
      tidyr::pivot_longer(cols = dplyr::starts_with("pct_"),
                          names_to = "metric", values_to = "percent") |>
      dplyr::mutate(
        metric = dplyr::recode(stringr::str_remove(.data$metric, "pct_"),
          high_global_priority  = "High global priority",
          sensitivity_stable    = "Stable across all dimensions",
          seed_stable           = "Seed stable",
          no_regret             = "No-regret (intersection)",
          indicator_sensitive   = "Indicator-sensitive",
          benchmark_sensitive   = "Benchmark-sensitive",
          policy_sensitive      = "Policy-sensitive"
        ),
        panel = dplyr::if_else(
          .data$metric %in% c("Indicator-sensitive", "Benchmark-sensitive", "Policy-sensitive"),
          "Sensitivity flags\n(can co-occur)",
          "Robustness & priority traits"
        )
      ) |>
      ggplot(aes(x = baseline_strategy_class, y = percent, fill = metric)) +
      geom_col(position = "dodge") +
      facet_wrap(~ panel, scales = "free_x") +
      coord_flip() +
      scale_y_continuous(labels = scales::label_percent(scale = 1L), limits = c(0, 100)) +
      scale_fill_brewer(palette = "Set2") +
      labs(title = "Characteristics of baseline strategy classes",
           x = NULL, y = "% of pixels in class", fill = NULL)

    p_no_regret_map <- ggplot(sens_with_baseline,
        aes(x = x, y = y, fill = no_regret_priority)) +
      geom_raster() + coord_equal() +
      scale_fill_manual(values = c("TRUE" = "black", "FALSE" = "grey85"),
                        name = "No-regret priority") +
      labs(title = "No-regret priorities",
           subtitle = "Baseline broadly robust \u2229 high global RFOP \u2229 stable across sensitivity dimensions") +
      theme_map

    save_plot(p_persistence_bar, "persistence_bar",    width = 12L, height = 6L)
    save_plot(p_no_regret_map,   "no_regret_map",      width = 12L, height = 6L)
  }
} else {
  message("No sensitivity run data found — skipping global RFOP and persistence plots.")
}

message("\nDone. Plots saved to: ", FIGS_DIR)
