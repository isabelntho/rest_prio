# =============================================================================
# produce_figures.R  —  single driver for the iEMSs conference figure set
# =============================================================================
# One place to turn an exported run matrix (r_inputs/<...>) into the ordered
# conference figures (R1-R5). It sources the shared function library and only
# orchestrates; all plotting/analysis lives in Documentation/_plot_functions.R.
#
# RUN (from the project root, in radian/R):
#   source("Documentation/produce_figures.R")
#
# Figures are written to OUT_DIR as PNG. Edit the CONFIG block to point at your
# run-matrix exports (defaults point at existing test exports so it runs today).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(jsonlite); library(stringr); library(purrr); library(tibble)
})
# patchwork is optional (used only to combine panels); degrade gracefully.
.has_patchwork <- requireNamespace("patchwork", quietly = TRUE)

source("Documentation/_plot_functions.R")

# ----------------------------------------------------------------------------
# CONFIG  — edit these paths after running the full matrix (Track A step 4)
# ----------------------------------------------------------------------------
R_INPUTS <- "r_inputs"

# Which figure groups to run. Set to "all" (default) to run everything, or list
# the top-level groups you want, e.g. c("R2") or c("R1", "R3"). Subsections run
# with their parent group (R3 includes R3c/R3d; R4 includes R4f/R4g). The shared
# data setup above the figure sections always runs. Case-insensitive.
RUN_SECTIONS <- "R3"

# The run plan has three input blocks; the two export roots below map to them.
# (Variable names AXIS1_DIR / AXIS2_DIR / AXIS3_DIR are legacy and kept only so
# the downstream code need not change — read them as the block roles described.)
#
# Blocks 1+2 — baseline + LOO condition grid (condition_grid export root).
#   Block 1 = baseline default formulation (global_all), seed-replicated.
#   Block 2 = LOO — full indicator set + each indicator dropped in turn, at
#             baseline settings, seed-replicated. The q75 run is carried here as
#             a benchmark (see BENCHMARK_SCENARIOS).
#   Feeds: R1, R3d-LOO, R3c, R2 (indicator dim), D/D2.
AXIS1_DIR <- file.path(R_INPUTS, "20260625_1618_iEMSs_fact_clustobj")
# Policy / formulation are factors INSIDE the Block 3 factorial, not separate
# one-at-a-time grids, so there are no standalone policy/formulation exports.
AXIS3_DIR <- NULL   # (legacy "policy axis" — unused; policy lives in Block 3)
AXIS2_DIR <- NULL   # (legacy "formulation axis" — unused; form lives in Block 3)

# Block 3 — fully-crossed factorial export root (SCENARIO_MODE = "factorial" in
# run_custom_parallel.py): form {sum, threshold} × scaling {anomaly/global, q75}
# × construction × policy {status_quo, ...}, seed-replicated. Set this to the
# r_inputs/<timestamp>_<label> grid folder once the factorial has been run; else
# NULL and the Block 3 outputs (R3d-factorial, R3d-full, R3d comparison, R4f,
# and the sum-form factorial points in R2/D) are skipped.
FACTORIAL_DIR <- file.path(R_INPUTS, "20260625_1925_iEMSs_fullfact_clustobj")

# Factorial policy levels to EXCLUDE from ALL analysis that uses FACTORIAL_DIR
# (R2 sum-form points, R3d, R4f). Matched against the "pol-<value>" token in the
# factorial run-dir name (e.g. "ambitious" drops ...__pol-ambitious__seed101).
# Set to character(0) to keep every policy level.
EXCLUDE_FACTORIAL_POLICIES <- c("ambitious", "burden_shared")

# Baseline scenario name (Block 1; a sub-folder prefix within AXIS1_DIR, minus _seed<n>).
BASELINE_SCENARIO <- "global_all"
# Benchmark scenario names within Blocks 1+2 (dim_type = "benchmark", not "indicator").
BENCHMARK_SCENARIOS <- c("upper_q75_all")

OUT_DIR <- "figs/iEMSs/iEMSs_fact_nopol"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  
# R3 robustness map is produced at these spatial scales (metres; x,y are EPSG:2056
# at 100 m). 0 = native pixel. Coarser cells average out fine-grained spatial
# degeneracy so a coherent robust core can emerge at a decision-relevant scale.
R3_AGG_CELLS <- c(0, 1000, 2000)

# Optional raster layers for the R3d summary bar chart (six-panel).
# LULC: same source as map_bern_landuse.R (network path; gracefully skipped if
# unavailable). Anomaly rasters: global_all scenario (baseline condition).
# Set any path to NULL or a non-existent file to skip that panel gracefully.
LULC_RASTER    <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif"
HABITAT_RASTER <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/Delarze/habitatmap_v1_1_20241025.tif"
DELARZE_CSV    <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/Delarze/typoch_num_lookuptable_26112025.csv"
ABIOTIC_RASTER <- file.path("inputs", "anomaly_scenarios",
                             paste0("abiotic_", BASELINE_SCENARIO, ".tif"))
BIOTIC_RASTER  <- file.path("inputs", "anomaly_scenarios",
                             paste0("biotic_",  BASELINE_SCENARIO, ".tif"))

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
.scenario_of <- function(dirname) sub("_seed[0-9]+$", "", dirname)   # strip seed suffix

# Section gate: TRUE if figure `group` (e.g. "R2") should run, per RUN_SECTIONS.
.run <- function(group) {
  rs <- toupper(RUN_SECTIONS)
  "ALL" %in% rs || toupper(group) %in% rs
}
.seed_of     <- function(dirname) stringr::str_extract(dirname, "seed[0-9]+$")

# List scenario_seed sub-folders that contain an export.
.list_run_dirs <- function(root) {
  if (is.null(root) || !dir.exists(root)) return(character(0))
  ds <- list.dirs(root, full.names = TRUE, recursive = FALSE)
  ds[file.exists(file.path(ds, "metadata.json"))]
}

# As .list_run_dirs(), but drops factorial run dirs whose policy token
# (pol-<value>) is listed in EXCLUDE_FACTORIAL_POLICIES. Use this for every
# FACTORIAL_DIR ingestion so the exclusion applies consistently across figures.
.list_factorial_run_dirs <- function(root) {
  ds <- .list_run_dirs(root)
  if (length(ds) == 0L || length(EXCLUDE_FACTORIAL_POLICIES) == 0L) return(ds)
  pat <- paste0("(^|__)pol-(", paste(EXCLUDE_FACTORIAL_POLICIES, collapse = "|"), ")(__|$)")
  ds[!grepl(pat, basename(ds))]
}

# Build a tidy runs_df (x, y, rfop_pct, dim_type, group_label, run_label) by
# binding load_run_rfop() over every run dir under `root`. dim_type/group_label
# are assigned by `tagger(scenario, dirname)`.
.build_runs_df <- function(root, tagger) {
  dirs <- .list_run_dirs(root)
  if (length(dirs) == 0L) return(tibble::tibble())
  purrr::map_dfr(dirs, function(d) {
    nm  <- basename(d)
    tag <- tagger(.scenario_of(nm), nm)
    tryCatch(
      load_run_rfop(d, label = nm, dim_type = tag$dim_type, group_label = tag$group_label),
      error = function(e) { message("  [skip] ", nm, ": ", conditionMessage(e)); tibble::tibble() }
    )
  })
}

.save <- function(plot, name, w = 9, h = 7, panel_cm = NULL, leg_in = 1.8) {
  if (is.null(plot)) { message("  [skip fig] ", name, " (NULL)"); return(invisible()) }
  path <- file.path(OUT_DIR, name)
  tryCatch({
    if (!is.null(panel_cm)) {
      # Separate the legend from the map so the map (Bern canton) is always
      # panel_cm x panel_cm cm regardless of legend size.
      # cowplot::get_legend returns NULL when there is no legend.
      map_in <- panel_cm / 2.54   # cm -> inches
      leg    <- cowplot::get_legend(plot)
      p0     <- plot + ggplot2::theme(legend.position = "none")
      if (!is.null(leg)) {
        # Detect legend position from the plot theme; default to "right".
        lpos <- tryCatch(plot$theme$legend.position %||% "right",
                         error = function(e) "right")
        if (identical(lpos, "bottom")) {
          out <- cowplot::plot_grid(p0, leg, ncol = 1,
                                    rel_heights = c(map_in, leg_in),
                                    align = "v", axis = "lr")
          ggsave(path, out, width = map_in, height = map_in + leg_in, units = "in", dpi = 200)
        } else {
          out <- cowplot::plot_grid(p0, leg, nrow = 1,
                                    rel_widths = c(map_in, leg_in),
                                    align = "h", axis = "tb")
          ggsave(path, out, width = map_in + leg_in, height = map_in, units = "in", dpi = 200)
        }
      } else {
        ggsave(path, p0, width = map_in, height = map_in, units = "in", dpi = 200)
      }
    } else {
      ggsave(path, plot, width = w, height = h, dpi = 200)
    }
    message("  [ok] ", path)
  }, error = function(e) message("  [FAIL fig] ", name, ": ", conditionMessage(e)))
}

# Bin a runs_df to coarse `cell_m`-metre cells (cell centre as new x,y), averaging
# RFOP within each cell × run. cell_m <= 0 returns the input unchanged (native).
.aggregate_rfop <- function(df, cell_m) {
  if (is.null(cell_m) || cell_m <= 0 || nrow(df) == 0L) return(df)
  df |>
    dplyr::mutate(x = cell_m * (x %/% cell_m) + cell_m / 2,
                  y = cell_m * (y %/% cell_m) + cell_m / 2) |>
    dplyr::group_by(x, y, run_label, dim_type, group_label) |>
    dplyr::summarise(rfop_pct = mean(rfop_pct, na.rm = TRUE), .groups = "drop")
}
.aggregate_elig <- function(edf, cell_m) {
  if (is.null(edf) || cell_m <= 0) return(edf)
  edf |>
    dplyr::mutate(x = cell_m * (x %/% cell_m) + cell_m / 2,
                  y = cell_m * (y %/% cell_m) + cell_m / 2) |>
    dplyr::distinct(x, y)
}

# ----------------------------------------------------------------------------
# Load eligible-pixel template (for map backgrounds) from the baseline run.
# ----------------------------------------------------------------------------
.baseline_dir <- {
  ds <- .list_run_dirs(AXIS1_DIR)
  hit <- ds[.scenario_of(basename(ds)) == BASELINE_SCENARIO]
  if (length(hit) > 0L) hit[1] else if (length(ds) > 0L) ds[1] else NA_character_
}
elig_df <- if (!is.na(.baseline_dir))
  read_csv_if_exists(file.path(.baseline_dir, "eligible_pixels.csv")) else NULL

# ----------------------------------------------------------------------------
# Canton boundary `BE` — the map functions expect this sf object in the global
# env (some reference it unguarded). Load from the network shapefile and cache
# locally so repeat runs work offline. If unavailable, leave BE undefined so the
# guarded maps simply omit the outline and unguarded ones are skipped.
# ----------------------------------------------------------------------------
.BE_CACHE <- "Documentation/BE_bern.rds"
.BE_SHP   <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissboundaries3d_2026/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp"
if (file.exists(.BE_CACHE)) {
  BE <- readRDS(.BE_CACHE)
  message("Canton boundary BE loaded from cache.")
} else {
  BE <- tryCatch({
    shp <- sf::st_read(.BE_SHP, quiet = TRUE)
    shp <- dplyr::filter(shp, NAME == "Bern")
    shp <- sf::st_transform(shp, 2056)
    saveRDS(shp, .BE_CACHE)
    message("Canton boundary BE loaded from shapefile (cached -> ", .BE_CACHE, ").")
    shp
  }, error = function(e) {
    message("  [skip] BE boundary unavailable: ", conditionMessage(e)); NULL
  })
  if (is.null(BE)) rm(BE)   # leave undefined so exists('BE') guards behave
}

# Precompute a 100 m terra raster mask of BE so .filter_within_be() can do a
# fast raster-cell lookup instead of running sf::st_intersects per run directory.
if (exists("BE") && !is.null(BE)) {
  .BE_MASK <- terra::rasterize(
    terra::vect(BE),
    terra::rast(terra::ext(terra::vect(BE)), res = 100, crs = "EPSG:2056"),
    field = 1L
  )
  message("BE raster mask computed (100 m).")
}

# Mask elig_df to the canton boundary.
if (!is.null(elig_df)) elig_df <- .filter_within_be(elig_df)

# Switzerland boundary CH - needed for the make_iemss_sel_freq_plot inset.
.CH_CACHE <- "Documentation/CH_switzerland.rds"
.CH_SHP   <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissboundaries3d_2026/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp"
if (file.exists(.CH_CACHE)) {
  CH <- readRDS(.CH_CACHE)
  message("Switzerland boundary CH loaded from cache.")
} else {
  CH <- tryCatch({
    shp <- sf::st_read(.CH_SHP, quiet = TRUE)
    shp <- sf::st_transform(shp, 2056)
    saveRDS(shp, .CH_CACHE)
    message("Switzerland boundary CH loaded from shapefile (cached -> ", .CH_CACHE, ").")
    shp
  }, error = function(e) {
    message("  [skip] CH boundary unavailable: ", conditionMessage(e)); NULL
  })
  if (is.null(CH)) rm(CH)
}

# ----------------------------------------------------------------------------
# Build the combined runs_df spanning the run-plan blocks.
#   Blocks 1+2 condition scenarios -> dim_type "indicator" (q75 -> "benchmark")
#   Block 1 baseline across seeds  -> dim_type "seed" (group_label = seed id)
#   Block 3 factorial cells        -> dim_type "factorial" (added below)
#   (legacy Axis 3 policy grid     -> dim_type "policy"; unused — AXIS3_DIR NULL)
# ----------------------------------------------------------------------------
message("Building runs_df ...")
runs_indicator <- .build_runs_df(AXIS1_DIR, function(scn, nm) {
  list(dim_type   = if (scn %in% BENCHMARK_SCENARIOS) "benchmark" else "indicator",
       group_label = scn)
})
runs_policy <- .build_runs_df(AXIS3_DIR, function(scn, nm) {
  list(dim_type = "policy", group_label = scn)
})
runs_axis2 <- .build_runs_df(AXIS2_DIR, function(scn, nm) {
  list(dim_type = "formulation", group_label = scn)
})
# Seed dimension: baseline scenario only, grouped by seed id.
runs_seed <- runs_indicator |>
  dplyr::filter(group_label == BASELINE_SCENARIO) |>
  dplyr::mutate(dim_type = "seed",
                group_label = stringr::str_extract(run_label, "seed[0-9]+$"))

# Block 3 factorial: each crossed cell loaded via load_run_factorial(), tagged
# dim_type = "factorial" and grouped by the (form × scaling × construction ×
# policy) combination so seeds are averaged into one scenario per combination
# downstream (R3d consensus). NULL/missing FACTORIAL_DIR -> empty tibble.
runs_factorial <- {
  fdirs <- .list_factorial_run_dirs(FACTORIAL_DIR)
  if (length(fdirs) == 0L) tibble::tibble() else
    purrr::map_dfr(fdirs, function(d) {
      fr <- tryCatch(load_run_factorial(d, label = basename(d)),
                     error = function(e) { message("  [skip] ", basename(d), ": ", conditionMessage(e)); NULL })
      if (is.null(fr) || nrow(fr) == 0L) return(tibble::tibble())
      fr |>
        dplyr::mutate(dim_type    = "factorial",
                      group_label = paste(form, scaling, construction, policy, sep = "_")) |>
        dplyr::select(x, y, rfop_pct, run_label, dim_type, group_label)
    })
}

runs_df <- dplyr::bind_rows(runs_indicator, runs_policy, runs_axis2, runs_seed, runs_factorial)
message(sprintf("  runs_df: %d rows | dim_types: %s",
                nrow(runs_df), paste(unique(runs_df$dim_type), collapse = ", ")))

# ============================================================================
# R1 — Baseline trade-off + selection frequency (orientation)
# ============================================================================
if (.run("R1")) {
message("\nR1 - baseline Pareto + selection frequency")
if (!is.na(.baseline_dir)) {
  run_base <- tryCatch(load_run_data(.baseline_dir, BASELINE_SCENARIO), error = function(e) NULL)
  if (!is.null(run_base)) {
    .save(tryCatch(make_pareto_extremes_plot(run_base, title = "Baseline Pareto front"),
                   error = function(e) { message("  ", conditionMessage(e)); NULL }),
          "R1a_baseline_pareto.png", 10, 6)
    if (exists("BE")) {
      .save(tryCatch(make_sel_freq_plot(run_base), error = function(e) { message("  ", conditionMessage(e)); NULL }),
            "R1b_baseline_selection_frequency.png", 9, 8, panel_cm = 14)
      .save(tryCatch(make_iemss_sel_freq_plot(run_base), error = function(e) { message("  ", conditionMessage(e)); NULL }),
            "R1c_iemss_selection_frequency.png", 9, 8, panel_cm = 14)
    } else {
      message("  [skip] R1b/R1c selection-frequency maps (BE boundary unavailable)")
    }
  }
}
}  # end R1

# ============================================================================
# R2 — How each uncertainty axis reshapes the frontier (SHARED-reference HV)
# ----------------------------------------------------------------------------
# Uses compute_shared_hv() so HV is comparable across scenarios (the optimiser's
# per-run HV is not — different reference point + per-run normalisation). Only
# the SUM objective form is comparable in a shared objective space: the
# 'threshold' form changes restoration_potential's units, so threshold-form
# factorial runs are EXCLUDED here (kept = the form-sum half of Block 3).
# ============================================================================
if (.run("R2")) {
message("\nR2 - shared-reference hypervolume across scenarios (indicator + policy + sum-form factorial)")
runs_meta <- list()
for (root in c(AXIS1_DIR, AXIS3_DIR)) {
  for (d in .list_run_dirs(root)) {
    nm <- basename(d)
    runs_meta[[nm]] <- list(run = tryCatch(load_run_data(d, nm), error = function(e) NULL))
  }
}
# Block 3 factorial: include only sum-form runs (shared objective units).
fac_sum_dirs <- Filter(function(d) grepl("form-sum", basename(d)),
                       .list_factorial_run_dirs(FACTORIAL_DIR))
for (d in fac_sum_dirs) {
  nm <- basename(d)
  runs_meta[[nm]] <- list(run = tryCatch(load_run_data(d, nm), error = function(e) NULL))
}
runs_meta <- runs_meta[!vapply(runs_meta, function(m) is.null(m$run), logical(1))]
if (length(runs_meta) > 0L) {
  obj_names <- runs_meta[[1]]$run$obj_names
  # The `scaling` factor rescales restoration_potential ~100x (global ~1e3 vs
  # upper_q75 ~1e5), so rp is NOT comparable across scaling levels - the same
  # reason the threshold form is excluded above. Compute the shared-reference HV
  # WITHIN each scaling group so global runs are not crushed against a q75-set rp
  # ceiling. (Min-max normalising the pooled matrix would NOT help: dominated-HV
  # fraction is invariant to per-axis affine rescaling.) HV is only comparable
  # within a scaling group.
  .scaling_of <- function(nm) if (grepl("q75", nm)) "upper_q75" else "global"
  hv_groups   <- stats::setNames(vapply(names(runs_meta), .scaling_of, character(1)),
                                 names(runs_meta))
  shared_hv <- tryCatch(compute_shared_hv(runs_meta, obj_names, groups = hv_groups),
                        error = function(e) { message("  ", conditionMessage(e)); NULL })
  if (!is.null(shared_hv) && nrow(shared_hv) > 0L) {
    readr::write_csv(shared_hv, file.path(OUT_DIR, "R2_shared_hv_by_scaling.csv"))
    for (g in sort(unique(shared_hv$hv_group)))
      message(sprintf("  scaling=%-10s: %d runs, HV range %.4g - %.4g", g,
                      sum(shared_hv$hv_group == g),
                      min(shared_hv$hv[shared_hv$hv_group == g]),
                      max(shared_hv$hv[shared_hv$hv_group == g])))
    ax1 <- basename(.list_run_dirs(AXIS1_DIR))
    ax3 <- basename(.list_run_dirs(AXIS3_DIR))
    axf <- basename(fac_sum_dirs)
    # Tag dimensions for make_hv_sensitivity_plot; baseline also appears as the
    # 'seed' group so it anchors the policy panel.
    stats_df <- dplyr::bind_rows(
      tibble::tibble(run_label = ax1, dim_type = "indicator", group_label = .scenario_of(ax1)),
      tibble::tibble(run_label = ax3, dim_type = "policy",    group_label = .scenario_of(ax3)),
      tibble::tibble(run_label = axf, dim_type = "factorial", group_label = "factorial (sum-form)"),
      tibble::tibble(run_label = ax1[.scenario_of(ax1) == BASELINE_SCENARIO],
                     dim_type = "seed", group_label = BASELINE_SCENARIO)
    ) |>
      dplyr::left_join(dplyr::rename(shared_hv, hypervolume = hv), by = "run_label") |>
      dplyr::filter(!is.na(hypervolume))
    # Plot only the global-scaling group: HV is not comparable across scaling
    # levels. The shown panels (indicator/policy/seed) are global-scaling apart
    # from the upper_q75 benchmark, whose within-q75-group HV is captured in
    # R2_shared_hv_by_scaling.csv rather than mixed onto the global axis.
    stats_global <- dplyr::filter(stats_df, hv_group == "global")
    .save(tryCatch(make_hv_sensitivity_plot(stats_global, ref_label = NULL),
                   error = function(e) { message("  ", conditionMessage(e)); NULL }),
          "R2_hypervolume_sensitivity.png", 11, 7)
  }

  # -- R2 backup -- overlaid Pareto fronts per sum-form factorial scenario ------
  # Disaggregates the single "factorial (sum-form)" HV group above into the
  # individual formulation cells, drawing each cell's non-dominated front as
  # pairwise 2D panels, one colour per cell. Seeds of a cell are pooled into that
  # cell's front (label drops the seed token). Objectives are min-max normalised
  # WITHIN scaling level inside make_pareto_overlay_scenarios() because the scaling
  # factor rescales restoration_potential ~100x (global ~1e3 vs upper_q75 ~1e5).
  # Pooled normalisation does NOT fix this (min-max is affine, so the global cells
  # still collapse into a stripe near 0); normalising within scaling spreads each
  # group across [0,1] so front shape is visible for all (absolute units dropped).
  if (length(fac_sum_dirs) > 0L) {
    obj_names <- runs_meta[[1]]$run$obj_names
    .fac_label <- function(nm) {
      toks <- strsplit(nm, "__", fixed = TRUE)[[1]]
      kv   <- list()
      for (t in toks) {
        m <- regmatches(t, regexec("^([a-z]+)-(.+)$", t))[[1]]
        if (length(m) == 3L) kv[[m[2]]] <- m[3]
      }
      paste(c(kv$scal, kv$con, kv$pol), collapse = " / ")
    }
    nd_overlay <- purrr::map_dfr(fac_sum_dirs, function(d) {
      nm  <- basename(d)
      run <- runs_meta[[nm]]$run
      if (is.null(run) || is.null(run$df_obj)) return(tibble::tibble())
      df <- run$df_obj
      if ("is_nondominated" %in% names(df)) df <- df[df$is_nondominated == 1, , drop = FALSE]
      if (nrow(df) == 0L) return(tibble::tibble())
      df$scenario <- .fac_label(nm)
      # Scaling level governs restoration_potential's magnitude (~100x apart), so
      # it is the grouping the objectives must be normalised WITHIN.
      df$scaling  <- if (grepl("scal-upper_q75", nm)) "upper_q75" else "global"
      tibble::as_tibble(df)
    })
    if (nrow(nd_overlay) > 0L) {
      n_cells <- dplyr::n_distinct(nd_overlay$scenario)
      .save(tryCatch(
        make_pareto_overlay_scenarios(
          nd_overlay, obj_names, norm_within = "scaling",
          title = "R2 backup - Pareto fronts across sum-form factorial scenarios"
          #, subtitle = sprintf(
          #    "%d formulation cells (form = sum, scaling / construction / policy); non-dominated solutions, objectives min-max normalised WITHIN scaling level",
          #    n_cells)
        ),
        error = function(e) { message("  ", conditionMessage(e)); NULL }),
        "R2_pareto_overlay_factorial.png", 15, 6)
    }
  }
}
}  # end R2

# ============================================================================
# R3 — Hero robustness map: priorities robust to CONDITION-INDICATOR uncertainty
# ----------------------------------------------------------------------------
# Deliberately scoped to epistemic uncertainty in *problem specification*
# (indicator construction + benchmark), with the built-in seed-noise floor.
# Policy variants (a controlled decision lever, not uncertainty) and formulation
# are intentionally EXCLUDED here — they are compared in R4 (attribution) and
# shown via the frontier (R2). Mixing budget levers into a "robustness" map would
# mislabel intended, budget-driven differences as instability.
# ============================================================================
if (.run("R3")) {
message("\nR3 - robustness to condition-indicator uncertainty (policy excluded)")
runs_epistemic <- dplyr::filter(runs_df, dim_type %in% c("indicator", "benchmark"))
if (nrow(runs_epistemic) > 0L) {
  for (cell in R3_AGG_CELLS) {
    tag <- if (is.null(cell) || cell <= 0) "native" else paste0(cell / 1000, "km")
    message("  scale: ", tag)
    rd  <- .aggregate_rfop(runs_epistemic, cell)
    edf <- .aggregate_elig(elig_df, cell)
    sens_df <- tryCatch(compute_rfop_sensitivity(rd, elig_df = edf),
                        error = function(e) { message("    ", conditionMessage(e)); NULL })
    if (!is.null(sens_df)) {
      .save(tryCatch(make_sensitivity_classification_map(
                       sens_df, elig_df = edf),
                     error = function(e) { message("    ", conditionMessage(e)); NULL }),
            sprintf("R3a_robustness_classification_%s.png", tag), 9, 8, panel_cm = 14, leg_in = 2.8)
      .save(tryCatch(make_stability_scatter(sens_df),
                     error = function(e) { message("    ", conditionMessage(e)); NULL }),
            sprintf("R3b_stability_scatter_%s.png", tag), 8, 6)
    }
  }
} else {
  message("  [skip] no indicator/benchmark runs in runs_df")
}

# ============================================================================
# R3c — Cross-indicator priority overlap: does a robust core exist?
# ----------------------------------------------------------------------------
# For each condition scenario, take its top TOP_PCT% RFOP cells; measure pairwise
# overlap (Jaccard) and how many scenarios rank each cell in their top set
# (consensus). High overlap / consensus => a robust core exists and the strict
# categorical map was just too punitive; low => priorities are genuinely
# indicator-contingent (which is itself the headline finding).
# ============================================================================
message("\nR3c — cross-indicator top-X% priority overlap")
TOP_PCT  <- 15
OVL_CELL <- 1000   # compute at 1 km (the scale where the maps were cleanest)
if (nrow(runs_epistemic) > 0L) {
  rd_ovl <- .aggregate_rfop(runs_epistemic, OVL_CELL)
  ovl <- tryCatch(compute_topx_overlap(rd_ovl, top_pct = TOP_PCT),
                  error = function(e) { message("  ", conditionMessage(e)); NULL })
  if (!is.null(ovl)) {
    message(sprintf("  mean pairwise Jaccard of top-%d%% priority across %d scenarios = %.2f",
                    TOP_PCT, ovl$n_scenarios, ovl$mean_jaccard))
    message(sprintf("  cells in EVERY scenario's top set: %d | in >=80%%: %d",
                    sum(ovl$consensus$frac >= 0.999),
                    sum(ovl$consensus$frac >= 0.8)))

    # ── Indicator leverage vs baseline (the plan's R3c product) ────────────────
    # 1 - Jaccard(each LOO top set vs the full-set baseline). Higher leverage =
    # dropping that indicator moves the priority set furthest from baseline. This
    # ranking is what selects the 2 drops for the Block 3 factorial construction axis.
    if (nrow(ovl$jaccard) > 0) {
      base_lev <- ovl$jaccard |>
        dplyr::filter(a == BASELINE_SCENARIO | b == BASELINE_SCENARIO) |>
        dplyr::mutate(scenario = dplyr::if_else(a == BASELINE_SCENARIO, b, a),
                      leverage = 1 - jaccard) |>
        dplyr::filter(scenario != BASELINE_SCENARIO,
                      !scenario %in% BENCHMARK_SCENARIOS) |>
        dplyr::transmute(scenario,
                         construction = sub("^global_", "", scenario),
                         jaccard_vs_baseline = jaccard,
                         leverage) |>
        dplyr::arrange(dplyr::desc(leverage))

      if (nrow(base_lev) > 0) {
        message("  indicator leverage vs baseline (1 - Jaccard of top set):")
        for (i in seq_len(nrow(base_lev)))
          message(sprintf("    %-22s leverage=%.3f (Jaccard=%.3f)",
                          base_lev$scenario[i], base_lev$leverage[i],
                          base_lev$jaccard_vs_baseline[i]))
        if (nrow(base_lev) >= 2)
          message(sprintf("  -> top-2 leverage drops for FACTORIAL_CONSTRUCTIONS: \"%s\", \"%s\"",
                          base_lev$construction[1], base_lev$construction[2]))
        readr::write_csv(base_lev, file.path(OUT_DIR, "R3c_indicator_leverage_vs_baseline.csv"))

        p_lev <- base_lev |>
          dplyr::mutate(scenario = factor(scenario, levels = rev(scenario))) |>
          ggplot(aes(leverage, scenario)) +
          geom_col(fill = "#E9C46A", colour = "grey30", width = 0.65) +
          scale_x_continuous(expand = c(0.01, 0)) +
          labs(title = sprintf("Indicator leverage vs baseline (top %d%%, %g km)",
                               TOP_PCT, OVL_CELL / 1000),
               #subtitle = "1 - Jaccard of each LOO top-priority set vs the full-set baseline; higher = more leverage",
               x = "Leverage (1 - Jaccard vs baseline)", y = NULL) +
          theme_minimal() +
          theme(panel.grid.major.y = element_blank())
        .save(p_lev, "R3c_indicator_leverage_vs_baseline.png", 8, 6)
      }
    }

    # Consensus map: share of scenarios ranking each cell in its top set.
    edf <- .aggregate_elig(elig_df, OVL_CELL)
    pc <- ggplot()
    if (!is.null(edf)) {
      bg <- dplyr::anti_join(edf, ovl$consensus, by = c("x", "y"))
      if (nrow(bg) > 0) pc <- pc + geom_raster(data = bg, aes(x, y), fill = "#EEEEEE")
    }
    pc <- pc +
      geom_raster(data = ovl$consensus, aes(x, y, fill = frac)) +
      scale_fill_viridis_c(name = sprintf("share of\nscenarios\n(top %d%%)", TOP_PCT),
                           limits = c(0, 1), option = "D") +
      theme_void()
    if (exists("BE"))
      pc <- pc + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
    .save(pc, "R3c_priority_consensus_map.png", 9, 8, panel_cm = 14)

    # Pairwise Jaccard heatmap.
    if (nrow(ovl$jaccard) > 0) {
      jh <- dplyr::bind_rows(ovl$jaccard, dplyr::rename(ovl$jaccard, a = b, b = a))
      ph <- ggplot(jh, aes(a, b, fill = jaccard)) +
        geom_tile() +
        scale_fill_viridis_c(name = "Jaccard", limits = c(0, NA)) +
        labs(title = sprintf("Top-%d%% priority overlap between scenarios", TOP_PCT),
             x = NULL, y = NULL) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
              axis.text.y = element_text(size = 7))
      .save(ph, "R3c_jaccard_heatmap.png", 9, 8)
    }
    # NB: the core/fringe/excluded classification (formerly emitted here as
    # R3d_consensus_classification_map.png from the condition grid alone) now
    # lives in the dedicated R3d section below, computed over the LOO / factorial
    # / full scenario sets.
  }
}

# ============================================================================
# R3e -- Diagnostic: why do R3a (condition-sensitive) and R3c (low Jaccard
#        leverage) appear to contradict each other?
# ----------------------------------------------------------------------------
# R3a flags a pixel as condition-sensitive when SNR_condition > 1, i.e. the
# between-indicator SD of scenario-mean RFOP exceeds the within-scenario seed
# noise.  R3c measures Jaccard leverage = 1 - Jaccard(LOO top set vs baseline
# top set), which only moves when pixels cross the top-15% membership boundary.
# A pixel can be genuinely sensitive in absolute RFOP terms (R3a) while sitting
# so far above -- or below -- the threshold that no amount of indicator-driven
# shift actually swaps its top-set membership (R3c).  This section quantifies
# that gap by classifying each sensitive pixel by its distance to the cutoff.
# ============================================================================
message("\nR3e -- threshold-distance diagnostic (sensitive pixels vs top-15% flippers)")
R3E_TOP      <- 15L    # must match TOP_PCT from R3c
R3E_CELL     <- 1000L  # 1 km, matches OVL_CELL from R3c
R3E_BAND_SDS <- 1      # boundary band half-width in seed-SE units

if (nrow(runs_epistemic) > 0L) {

  rd_3e  <- .aggregate_rfop(runs_epistemic, R3E_CELL)
  edf_3e <- .aggregate_elig(elig_df, R3E_CELL)

  sens_3e <- tryCatch(
    compute_rfop_sensitivity(rd_3e, elig_df = edf_3e),
    error = function(e) { message("  ", conditionMessage(e)); NULL })

  if (!is.null(sens_3e)) {

    # -- Baseline seed-mean RFOP per pixel ------------------------------------
    baseline_px_3e <- rd_3e |>
      dplyr::filter(group_label == BASELINE_SCENARIO) |>
      dplyr::group_by(x, y) |>
      dplyr::summarise(baseline_rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

    # Baseline 85th-percentile cutoff -- same computation as compute_topx_overlap
    base_cutoff_3e <- stats::quantile(baseline_px_3e$baseline_rfop,
                                      1 - R3E_TOP / 100, na.rm = TRUE)

    # Per-scenario seed-mean RFOP and per-scenario top-sets (for flipper check)
    sm_3e <- rd_3e |>
      dplyr::filter(dim_type %in% c("indicator", "benchmark")) |>
      dplyr::group_by(group_label, x, y) |>
      dplyr::summarise(rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop")

    top_df_3e <- sm_3e |>
      dplyr::group_by(group_label) |>
      dplyr::mutate(thr = stats::quantile(rfop, 1 - R3E_TOP / 100, na.rm = TRUE)) |>
      dplyr::filter(rfop >= thr) |>
      dplyr::ungroup() |>
      dplyr::select(group_label, x, y)

    # -- Join and classify each pixel by distance to the baseline cutoff ------
    adf_3e <- sens_3e |>
      dplyr::left_join(baseline_px_3e, by = c("x", "y")) |>
      dplyr::mutate(
        baseline_rfop         = dplyr::coalesce(baseline_rfop, 0),
        distance_to_threshold = baseline_rfop - base_cutoff_3e,
        # Per-pixel noise floor: mean_seed_SD (= seed SE when noise_floor="se").
        # Floor at 2 pp so the band does not degenerate when seed replication
        # is unavailable (mean_seed_SD == 0 or NA).
        effective_noise = pmax(dplyr::coalesce(mean_seed_SD, 0), 2),
        in_baseline_top = baseline_rfop >= base_cutoff_3e,
        location_group  = dplyr::case_when(
          distance_to_threshold >  R3E_BAND_SDS * effective_noise ~ "deep-in-core",
          distance_to_threshold < -R3E_BAND_SDS * effective_noise ~ "deep-out",
          TRUE                                                     ~ "boundary band"
        )
      )

    # -- (A) Group shares among condition-sensitive pixels --------------------
    sensitive_3e <- dplyr::filter(adf_3e, condition_detectable)
    n_sens_3e    <- nrow(sensitive_3e)

    grp_shr <- sensitive_3e |>
      dplyr::count(location_group) |>
      dplyr::mutate(pct = round(100 * n / sum(n), 1)) |>
      dplyr::arrange(dplyr::desc(pct))

    message(sprintf("  Baseline top-%d%% cutoff = %.1f pp", R3E_TOP, base_cutoff_3e))
    message(sprintf("  Condition-sensitive pixels (1 km): %d", n_sens_3e))
    for (i in seq_len(nrow(grp_shr)))
      message(sprintf("    %-15s %4d  (%5.1f%%)",
                      grp_shr$location_group[i], grp_shr$n[i], grp_shr$pct[i]))

    # -- (B) Fraction of the top-set that is sensitive AND in boundary band ---
    top_set_3e <- dplyr::filter(adf_3e, in_baseline_top)
    n_top_3e   <- nrow(top_set_3e)
    n_sb_3e    <- dplyr::filter(top_set_3e, condition_detectable,
                                location_group == "boundary band") |> nrow()
    message(sprintf("  Top-%d%% set: %d pixels  |  sensitive + boundary band: %d (%.1f%%)",
                    R3E_TOP, n_top_3e, n_sb_3e, 100 * n_sb_3e / max(n_top_3e, 1L)))

    # -- (C) Flipper cross-check: pixels that change top-set membership -------
    # across any LOO run vs the baseline.
    loo_scens_3e <- setdiff(
      unique(dplyr::filter(sm_3e, !group_label %in% BENCHMARK_SCENARIOS)$group_label),
      BASELINE_SCENARIO
    )
    base_top_3e <- dplyr::filter(top_df_3e, group_label == BASELINE_SCENARIO) |>
      dplyr::select(x, y)

    flippers_3e <- purrr::map_dfr(loo_scens_3e, function(scn) {
      scn_top <- dplyr::filter(top_df_3e, group_label == scn) |> dplyr::select(x, y)
      dplyr::bind_rows(
        dplyr::anti_join(base_top_3e, scn_top, by = c("x", "y")) |>
          dplyr::mutate(direction = "dropped_from_top"),
        dplyr::anti_join(scn_top, base_top_3e, by = c("x", "y")) |>
          dplyr::mutate(direction = "entered_top")
      ) |> dplyr::mutate(scenario = scn)
    })

    uniq_flippers_3e <- dplyr::distinct(flippers_3e, x, y) |>
      dplyr::left_join(
        dplyr::select(adf_3e, x, y, condition_detectable, location_group,
                      baseline_rfop, distance_to_threshold, SNR_condition),
        by = c("x", "y"))

    n_flip      <- nrow(uniq_flippers_3e)
    n_flip_sens <- sum(uniq_flippers_3e$condition_detectable, na.rm = TRUE)
    n_flip_sb   <- dplyr::filter(uniq_flippers_3e, condition_detectable,
                                 location_group == "boundary band") |> nrow()
    message(sprintf("  Unique flipper pixels: %d", n_flip))
    message(sprintf("    flagged sensitive:          %d (%.1f%%)",
                    n_flip_sens, 100 * n_flip_sens / max(n_flip, 1L)))
    message(sprintf("    sensitive + boundary band:  %d (%.1f%%)",
                    n_flip_sb, 100 * n_flip_sb / max(n_flip, 1L)))

    readr::write_csv(grp_shr,          file.path(OUT_DIR, "R3e_sensitive_location_groups.csv"))
    readr::write_csv(uniq_flippers_3e, file.path(OUT_DIR, "R3e_flipper_pixels.csv"))

    # -- (D) Plot: RFOP distribution of sensitive pixels with cutoff line -----
    med_noise_3e <- median(sensitive_3e$effective_noise, na.rm = TRUE)
    band_lo_3e   <- base_cutoff_3e - R3E_BAND_SDS * med_noise_3e
    band_hi_3e   <- base_cutoff_3e + R3E_BAND_SDS * med_noise_3e

    p_3e_hist <- ggplot(sensitive_3e, aes(x = baseline_rfop)) +
      annotate("rect",
               xmin = band_lo_3e, xmax = band_hi_3e,
               ymin = -Inf, ymax = Inf,
               fill = "grey70", alpha = 0.4) +
      geom_histogram(bins = 40L, fill = "#bd8c12", colour = "white", linewidth = 0.2) +
      geom_vline(xintercept = base_cutoff_3e, colour = "black", linewidth = 0.9) +
      scale_x_continuous(limits = c(0, 100), expand = c(0.01, 0)) +
      labs(
        title    = sprintf(
          "RFOP distribution of condition-sensitive pixels (top-%d%% cutoff = %.1f pp, 1 km)",
          R3E_TOP, base_cutoff_3e),
        subtitle = sprintf(
          "Grey band = +/-%g seed-SE (median = %.1f pp) | n = %d sensitive | cutoff = %.1f pp",
          R3E_BAND_SDS, med_noise_3e, n_sens_3e, base_cutoff_3e),
        x = "Baseline RFOP (%)", y = "Pixel count (1 km)"
      ) +
      theme_minimal()
    .save(p_3e_hist, "R3e_sensitive_rfop_distribution.png", 8, 5)

    # -- (E) Scatter: SNR_condition vs distance_to_threshold ------------------
    p_3e_scat <- ggplot(sensitive_3e,
                        aes(x = distance_to_threshold, y = SNR_condition)) +
      annotate("rect",
               xmin = -R3E_BAND_SDS * med_noise_3e,
               xmax =  R3E_BAND_SDS * med_noise_3e,
               ymin = -Inf, ymax = Inf,
               fill = "grey70", alpha = 0.4) +
      geom_point(alpha = 0.25, size = 0.7, colour = "#bd8c12") +
      geom_vline(xintercept = 0, colour = "black", linewidth = 0.8) +
      labs(
        title    = "Sensitivity magnitude vs distance to top-15% cutoff",
        subtitle = "Grey band = boundary band (+/- 1 seed-SE); each point = 1 km pixel",
        x        = "Distance to cutoff (pp; positive = above threshold)",
        y        = "SNR_condition (indicator SD / seed-SE)"
      ) +
      theme_minimal()
    .save(p_3e_scat, "R3e_snr_vs_threshold_distance.png", 7, 5)

    # -- (F) Flipper map: are flippers concentrated in the boundary band? -----
    if (n_flip > 0L && !is.null(edf_3e)) {
      flip_cls <- dplyr::mutate(
        uniq_flippers_3e,
        flip_class = factor(
          dplyr::case_when(
            condition_detectable & location_group == "boundary band" ~ "sensitive + boundary",
            condition_detectable                                     ~ "sensitive (deep)",
            TRUE                                                     ~ "not sensitive"
          ),
          levels = c("sensitive + boundary", "sensitive (deep)", "not sensitive")
        )
      )
      p_3e_flip <- ggplot()
      bg_flip <- dplyr::anti_join(edf_3e, flip_cls, by = c("x", "y"))
      if (nrow(bg_flip) > 0)
        p_3e_flip <- p_3e_flip +
          geom_raster(data = bg_flip, aes(x, y), fill = "#EEEEEE")
      p_3e_flip <- p_3e_flip +
        geom_raster(data = flip_cls, aes(x, y, fill = flip_class)) +
        scale_fill_manual(
          values = c(
            "sensitive + boundary" = "#bd8c12",
            "sensitive (deep)"     = "#914906",
            "not sensitive"        = "#4393C3"
          ),
          name = NULL, drop = FALSE) +
        theme_void()
      if (exists("BE"))
        p_3e_flip <- p_3e_flip +
          geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4,
                  inherit.aes = FALSE)
      .save(p_3e_flip, "R3e_flipper_sensitivity_map.png", 9, 8, panel_cm = 14)
    }

  }   # end if (!is.null(sens_3e))
} else {
  message("  [skip] R3e: no epistemic runs")
}

# ============================================================================
# R3d — Core / fringe / excluded across scenario sets (slide 7 HEADLINE)
# ----------------------------------------------------------------------------
# Per-cell selection-frequency consensus, classified into core / fringe /
# excluded, computed over THREE scenario sets and compared:
#   loo        — Block 2 condition LOO set (each indicator dropped in turn)
#   factorial  — Block 3 crossed design  (form × scaling × construction × policy)
#   full       — Blocks 2 + 3 combined   (the headline figure)
# A cell is "core" if it is a top-X% priority in >= HI of the set's scenarios,
# "fringe" if in >= LO, else "excluded". Each distinct group_label counts as one
# scenario, with seeds averaged inside compute_topx_overlap() so every scenario
# carries equal weight regardless of how many seed replicates it has.
# ============================================================================
message("\nR3d — core/fringe/excluded across scenario sets (loo | factorial | full)")
R3D_TOP_PCT <- 15
R3D_CELL    <- 1000
R3D_HI      <- 0.8
R3D_LO      <- 0.2

# Classify one scenario set (a runs_df already carrying dim_type/group_label).
.r3d_classify <- function(runs_df_sub) {
  if (is.null(runs_df_sub) || nrow(runs_df_sub) == 0L) return(NULL)
  rd  <- .aggregate_rfop(runs_df_sub, R3D_CELL)
  ovl <- tryCatch(compute_topx_overlap(rd, top_pct = R3D_TOP_PCT,
                                       dims = unique(rd$dim_type)),
                  error = function(e) { message("    ", conditionMessage(e)); NULL })
  if (is.null(ovl)) return(NULL)
  cls <- ovl$consensus |>
    dplyr::mutate(r3d_class = factor(dplyr::case_when(
        frac >= R3D_HI ~ "Core",
        frac >= R3D_LO ~ "Fringe",
        TRUE           ~ "Marginal"),
      levels = c("Core", "Fringe", "Marginal")))
  list(cls = cls, n_scen = ovl$n_scenarios,
       n_core   = sum(cls$r3d_class == "Core"),
       n_fringe = sum(cls$r3d_class == "Fringe"))
}

# Core/fringe/excluded raster map (eligible-but-unclassified pixels = excluded).
.r3d_map <- function(cls, edf, title) {
  pcl <- ggplot()
  if (!is.null(edf)) {
    bg <- dplyr::anti_join(edf, cls, by = c("x", "y"))
    if (nrow(bg) > 0) pcl <- pcl + geom_raster(data = bg, aes(x, y), fill = "#EEEEEE")
  }
  pcl <- pcl +
    geom_raster(data = cls, aes(x, y, fill = r3d_class)) +
    scale_fill_manual(values = c(Core = "#2166AC", Fringe = "#91BFDB", Marginal = "#BBBBBB"),
                      name = NULL, drop = FALSE) +
    theme_void()
  if (exists("BE"))
    pcl <- pcl + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
  pcl
}

set_loo  <- dplyr::filter(runs_df, dim_type == "indicator")   # Block 2 LOO
set_fac  <- dplyr::filter(runs_df, dim_type == "factorial")   # Block 3 factorial

r3d_sets <- list(loo = list(df = set_loo,
                            title = NULL))
if (nrow(set_fac) > 0L) {
  r3d_sets$factorial <- list(df = set_fac,
                             title = NULL)
  r3d_sets$full      <- list(df = dplyr::bind_rows(set_loo, set_fac),
                             title = NULL)
} else {
  message("  [note] FACTORIAL_DIR not set/empty: only the LOO R3d map is produced.")
}

edf_r3d  <- .aggregate_elig(elig_df, R3D_CELL)
r3d_summ <- list(); r3d_cls <- list()
for (nm in names(r3d_sets)) {
  res <- .r3d_classify(r3d_sets[[nm]]$df)
  if (is.null(res)) { message("  [skip] R3d ", nm, " (no runs)"); next }
  r3d_cls[[nm]]  <- res$cls
  r3d_summ[[nm]] <- tibble::tibble(set = nm, n_scen = res$n_scen,
                                   core = res$n_core, fringe = res$n_fringe)
  message(sprintf("  %-9s: %d scenarios | core=%d fringe=%d cells (1 km, top %d%%, core>=%.0f%%)",
                  nm, res$n_scen, res$n_core, res$n_fringe, R3D_TOP_PCT, 100 * R3D_HI))
  fn <- if (nm == "full") "R3d_full_core_fringe_excluded.png"
        else sprintf("R3d_%s_core_fringe_excluded.png", nm)
  .save(.r3d_map(res$cls, edf_r3d, r3d_sets[[nm]]$title), fn, 9, 8, panel_cm = 14)
}

# ── R3d comparison — core size under each uncertainty axis ──────────────────
if (length(r3d_summ) > 0L) {
  comp <- dplyr::bind_rows(r3d_summ) |>
    tidyr::pivot_longer(c(core, fringe), names_to = "class", values_to = "n_cells") |>
    dplyr::mutate(
      set   = factor(set, levels = c("loo", "factorial", "full"),
                     labels = c("LOO\n(Block 2)", "Factorial\n(Block 3)", "Full\n(Blocks 2+3)")),
      class = factor(class, levels = c("Fringe", "Core")))
  p_cmp <- ggplot(comp, aes(set, n_cells, fill = class)) +
    geom_col(width = 0.65, colour = "grey30") +
    scale_fill_manual(values = c(Core = "#2166AC", Fringe = "#91BFDB"),
                      labels = c(Core = "Core", Fringe = "Fringe"),
                      breaks = c("Core", "Fringe"), name = NULL) +
    labs(title = NULL,
         #subtitle = sprintf("1 km cells classified core (>=%.0f%% of scenarios) / fringe, top %d%% priority",
         #                   100 * R3D_HI, R3D_TOP_PCT),
         x = NULL, y = "Number of 1 km cells") +
    theme_minimal() +
    theme(panel.grid.major.x = element_blank())
  .save(p_cmp, "R3d_core_size_comparison.png", 8, 6)
}

# ── R3d difference map — full (Blocks 2+3) vs LOO (Block 2) core membership ──
if (!is.null(r3d_cls$loo) && !is.null(r3d_cls$full)) {
  core_loo  <- r3d_cls$loo  |> dplyr::filter(r3d_class == "core") |>
    dplyr::transmute(x, y, in_loo = TRUE)
  core_full <- r3d_cls$full |> dplyr::filter(r3d_class == "core") |>
    dplyr::transmute(x, y, in_full = TRUE)
  diff <- dplyr::full_join(core_loo, core_full, by = c("x", "y")) |>
    dplyr::mutate(
      in_loo  = dplyr::coalesce(in_loo, FALSE),
      in_full = dplyr::coalesce(in_full, FALSE),
      change  = factor(dplyr::case_when(
          in_loo &  in_full ~ "core in both",
         !in_loo &  in_full ~ "core only with factorial",
          in_loo & !in_full ~ "core only in LOO"),
        levels = c("core in both", "core only with factorial", "core only in LOO")))
  pd <- ggplot()
  if (!is.null(edf_r3d)) {
    bg <- dplyr::anti_join(edf_r3d, diff, by = c("x", "y"))
    if (nrow(bg) > 0) pd <- pd + geom_raster(data = bg, aes(x, y), fill = "#EEEEEE")
  }
  pd <- pd +
    geom_raster(data = diff, aes(x, y, fill = change)) +
    scale_fill_manual(values = c("core in both"             = "#2166AC",
                                 "core only with factorial" = "#762A83",
                                 "core only in LOO"         = "#D73027"),
                      name = NULL, drop = FALSE) +
    theme_void()
  if (exists("BE"))
    pd <- pd + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
  .save(pd, "R3d_core_difference_full_vs_loo.png", 9, 8, panel_cm = 14)
}

# ---- R3d summary bar chart (full scenario set) --------------------------------
# Six panels per R3d priority class (core / fringe / excluded):
#   A. LULC class composition          (LULC_RASTER; map_bern_landuse.R classes)
#   B. Habitat class (Delarze)         (HABITAT_RASTER; top-3 codes per R3d class
#                                       are labelled; all others lumped as "Other")
#   C. Mean abiotic condition anomaly  (ABIOTIC_RASTER; lower = more degraded)
#   D. Mean biotic  condition anomaly  (BIOTIC_RASTER;  lower = more degraded)
#   E. Restoration potential = (abiotic + biotic) / 2  (map_bern_landuse.R formula)
#   F. Dominant factorial variance source (parsed from set_fac run labels)
# "Excluded" = pixels with frac < R3D_LO, plus eligible pixels never selected.
if (!is.null(r3d_cls[["full"]]) && nrow(r3d_cls[["full"]]) > 0L) {
  message("  R3d summary bar chart (full set)")

  r3d_cls_lvls <- c("Core", "Fringe", "Marginal")
  r3d_cls_cols <- c(Core = "#2166AC", Fringe = "#91BFDB", Marginal = "#BBBBBB")

  # Extend classified pixels to include eligible pixels never selected (frac=0).
  cls_bar <- if (!is.null(edf_r3d) && nrow(edf_r3d) > 0L) {
    never <- dplyr::anti_join(edf_r3d, r3d_cls[["full"]], by = c("x", "y")) |>
      dplyr::mutate(frac      = 0,
                    r3d_class = factor("Marginal", levels = r3d_cls_lvls))
    dplyr::bind_rows(r3d_cls[["full"]], never)
  } else r3d_cls[["full"]]

  # Extract a single raster layer at cls_bar pixel centres (EPSG:2056).
  # Returns a numeric vector the same length as nrow(cls_bar), or NULL on failure.
  .r3d_extract <- function(raster_path) {
    if (is.null(raster_path) || !file.exists(raster_path)) return(NULL)
    if (!requireNamespace("terra", quietly = TRUE)) return(NULL)
    r <- tryCatch(terra::rast(raster_path), error = function(e) NULL)
    if (is.null(r)) return(NULL)
    tryCatch({
      pts <- terra::vect(as.data.frame(cls_bar[, c("x", "y")]),
                         geom = c("x", "y"), crs = "EPSG:2056")
      pts <- terra::project(pts, terra::crs(r))
      terra::extract(r, pts)[, 2L]
    }, error = function(e) NULL)
  }

  # ---- Panel A: LULC class composition ----------------------------------------
  # Class map and colours match map_bern_landuse.R exactly.
  lu_map  <- c("12" = "Forest", "13" = "Forest",
               "15" = "Agricultural land",
               "16" = "Grassland & pasture", "17" = "Grassland & pasture")
  lu_lvls <- c("Forest", "Grassland & pasture", "Agricultural land")
  lu_cols <- c("Forest"               = "#1B7837",
               "Grassland & pasture"  = "#A6D96A",
               "Agricultural land"    = "#E9C46A")

  p_lulc <- tryCatch({
    lulc_vals <- .r3d_extract(LULC_RASTER)
    if (is.null(lulc_vals)) stop("LULC_RASTER not set, missing, or terra unavailable")
    lulc_df <- cls_bar |>
      dplyr::mutate(lulc      = lu_map[as.character(lulc_vals)],
                    lulc      = factor(lulc, levels = lu_lvls),
                    r3d_class = factor(r3d_class, levels = r3d_cls_lvls)) |>
      dplyr::filter(!is.na(lulc)) |>
      dplyr::count(r3d_class, lulc) |>
      dplyr::group_by(r3d_class) |>
      dplyr::mutate(prop = n / sum(n)) |>
      dplyr::ungroup()
    ggplot(lulc_df, aes(r3d_class, prop, fill = lulc)) +
      geom_col(width = 0.65, colour = "white", linewidth = 0.3) +
      scale_y_continuous(labels = scales::percent_format(1), expand = c(0.01, 0)) +
      scale_fill_manual(values = lu_cols, name = NULL, drop = FALSE) +
      labs(title = "LULC class", x = NULL, y = "Share of class pixels") +
      theme_minimal() +
      theme(panel.grid.major.x = element_blank(), legend.position = "bottom")
  }, error = function(e) { message("    [skip] LULC panel: ", conditionMessage(e)); NULL })

  # Panel B: Delarze habitat class - top-3 codes per R3d class labelled, rest "Other"
  p_hab <- tryCatch({
    hab_vals <- .r3d_extract(HABITAT_RASTER)
    if (is.null(hab_vals)) stop("HABITAT_RASTER not set, missing, or terra unavailable")
    # Load TypoCH_EN lookup: Type (integer raster value) -> English label
    hab_lut <- if (!is.null(DELARZE_CSV) && file.exists(DELARZE_CSV)) {
      lut <- readr::read_csv(DELARZE_CSV, show_col_types = FALSE, locale = locale(encoding = "ISO-8859-1"))
      # Strip the dot-notation prefix from TypoCH_EN, e.g.
      #   "4.5.3 Low and medium altitude pasture" -> "Low and medium altitude pasture"
      lbls <- sub("^[0-9.]+ *", "", lut$TypoCH_EN)
      # Fall back to the TypoCH dot-notation code when TypoCH_EN is absent.
      lbls <- ifelse(is.na(lut$TypoCH_EN) | nchar(trimws(lbls)) == 0L, lut$TypoCH, lbls)
      # Key by Type AND TypoCH_NUM so any raster value format is matched;
      # TypoCH_NUM-keyed entries are appended last and win on name collision.
      by_type <- setNames(lbls, trimws(as.character(lut$Type)))
      by_num  <- setNames(lbls, trimws(as.character(lut$TypoCH_NUM)))
      combined <- c(by_type, by_num)
      combined[!duplicated(names(combined), fromLast = TRUE)]
    } else NULL
    code_label <- function(codes) {
      if (!is.null(hab_lut)) {
        lbl <- hab_lut[codes]
        ifelse(is.na(lbl), paste0("Type ", codes), lbl)
      } else paste0("Type ", codes)
    }
    hab_raw <- cls_bar |>
      dplyr::mutate(code      = as.character(as.integer(hab_vals)),
                    r3d_class = factor(r3d_class, levels = r3d_cls_lvls)) |>
      dplyr::filter(!is.na(code) & code != "NA")
    # Union of top-3 codes by pixel count within each R3d class
    top_codes <- hab_raw |>
      dplyr::count(r3d_class, code) |>
      dplyr::group_by(r3d_class) |>
      dplyr::slice_max(n, n = 3L, with_ties = FALSE) |>
      dplyr::ungroup() |>
      dplyr::pull(code) |>
      unique() |>
      sort()
    top_labels <- code_label(top_codes)
    hab_df <- hab_raw |>
      dplyr::mutate(
        lbl = dplyr::if_else(code %in% top_codes, code_label(code), "Other"),
        lbl = factor(lbl, levels = c(unique(top_labels), "Other"))
      ) |>
      dplyr::count(r3d_class, lbl) |>
      dplyr::group_by(r3d_class) |>
      dplyr::mutate(prop = n / sum(n)) |>
      dplyr::ungroup()
    n_top   <- length(top_codes)
    hab_pal <- setNames(c(scales::hue_pal()(n_top), "grey72"),
                        c(top_labels, "Other"))
    ggplot(hab_df, aes(r3d_class, prop, fill = lbl)) +
      geom_col(width = 0.65, colour = "white", linewidth = 0.3) +
      scale_y_continuous(labels = scales::percent_format(1), expand = c(0.01, 0)) +
      scale_fill_manual(values = hab_pal, name = NULL, drop = FALSE) +
      labs(title = "Habitat class (Delarze)", x = NULL, y = "Share of class pixels") +
      theme_minimal() +
      theme(panel.grid.major.x = element_blank(), legend.position = "bottom")
  }, error = function(e) { message("    [skip] habitat panel: ", conditionMessage(e)); NULL })

  # ---- Helper: anomaly bar panel (mean +/- SE, coloured by R3d class) ----------
  .r3d_anom_panel <- function(vals, title) {
    if (is.null(vals)) return(NULL)
    df <- cls_bar |>
      dplyr::mutate(anom      = vals,
                    r3d_class = factor(r3d_class, levels = r3d_cls_lvls)) |>
      dplyr::filter(!is.na(anom)) |>
      dplyr::group_by(r3d_class) |>
      dplyr::summarise(mean_v = mean(anom, na.rm = TRUE),
                       se_v   = sd(anom,   na.rm = TRUE) / sqrt(sum(!is.na(anom))),
                       .groups = "drop")
    ggplot(df, aes(r3d_class, mean_v, fill = r3d_class)) +
      geom_col(width = 0.6, colour = "grey30", show.legend = FALSE) +
      geom_errorbar(aes(ymin = mean_v - se_v, ymax = mean_v + se_v),
                    width = 0.2, colour = "grey40") +
      scale_fill_manual(values = r3d_cls_cols) +
      labs(title = title, x = NULL, y = "Mean +/- 1 SE",
           caption = "Lower = more degraded") +
      theme_minimal() +
      theme(panel.grid.major.x = element_blank(),
            plot.caption = element_text(size = 7, colour = "grey50"))
  }

  # ---- Panels B & C: abiotic and biotic anomaly --------------------------------
  p_abio <- tryCatch({
    vals <- .r3d_extract(ABIOTIC_RASTER)
    if (is.null(vals)) stop("ABIOTIC_RASTER not set, missing, or terra unavailable")
    .r3d_anom_panel(vals, "Abiotic condition anomaly")
  }, error = function(e) { message("    [skip] abiotic panel: ", conditionMessage(e)); NULL })

  p_bio <- tryCatch({
    vals <- .r3d_extract(BIOTIC_RASTER)
    if (is.null(vals)) stop("BIOTIC_RASTER not set, missing, or terra unavailable")
    .r3d_anom_panel(vals, "Biotic condition anomaly")
  }, error = function(e) { message("    [skip] biotic panel: ", conditionMessage(e)); NULL })

  # ---- Panel D: restoration potential (abiotic + biotic) / 2 ------------------
  # Equal-weight mean of the two anomaly rasters, matching map_bern_landuse.R.
  # Lower = more degraded = higher restoration need.
  p_rp <- tryCatch({
    abio_vals <- .r3d_extract(ABIOTIC_RASTER)
    bio_vals  <- .r3d_extract(BIOTIC_RASTER)
    if (is.null(abio_vals) || is.null(bio_vals))
      stop("ABIOTIC_RASTER or BIOTIC_RASTER not set, missing, or terra unavailable")
    .r3d_anom_panel((abio_vals + bio_vals) / 2, "Restoration potential")
  }, error = function(e) { message("    [skip] restoration potential panel: ", conditionMessage(e)); NULL })

  # ---- Panel E: dominant factorial variance source ----------------------------
  # Per-pixel dominant factor = the factor whose between-level range of mean RFOP
  # is greatest (seeds + all other factors averaged out). Marginal sensitivity
  # measure -- faster than a full ANOVA eta^2, same directional signal.
  p_var <- tryCatch({
    if (nrow(set_fac) == 0L) stop("no factorial runs in set_fac")
    # Parse factor tokens from run_label (format: form-X__scal-Y__con-Z__pol-W__seedN)
    .r3d_tok <- function(labels, key) {
      pfx <- paste0(key, "-")
      vapply(labels, function(nm) {
        toks <- strsplit(nm, "__", fixed = TRUE)[[1]]
        hit  <- toks[startsWith(toks, pfx)]
        if (length(hit) == 0L) NA_character_
        else sub(pfx, "", hit[1L], fixed = TRUE)
      }, character(1L), USE.NAMES = FALSE)
    }
    fac_p <- set_fac |>
      dplyr::mutate(
        xc    = R3D_CELL * (x %/% R3D_CELL) + R3D_CELL / 2,
        yc    = R3D_CELL * (y %/% R3D_CELL) + R3D_CELL / 2,
        .form = .r3d_tok(run_label, "form"),
        .scal = .r3d_tok(run_label, "scal"),
        .con  = .r3d_tok(run_label, "con"),
        .pol  = .r3d_tok(run_label, "pol")
      ) |>
      dplyr::filter(!is.na(.form))
    if (nrow(fac_p) == 0L) stop("factor token parsing produced no rows")

    fac_cols <- c(form = ".form", scaling = ".scal",
                  construction = ".con", policy = ".pol")
    fac_cols <- fac_cols[vapply(fac_cols, function(col)
      dplyr::n_distinct(fac_p[[col]], na.rm = TRUE) > 1L, logical(1))]
    if (length(fac_cols) == 0L) stop("no factor has more than one level")

    ranges <- purrr::map_dfr(names(fac_cols), function(flab) {
      col <- fac_cols[[flab]]
      fac_p |>
        dplyr::group_by(xc, yc, lvl = .data[[col]]) |>
        dplyr::summarise(m = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
        dplyr::group_by(xc, yc) |>
        dplyr::summarise(rng = diff(range(m, na.rm = TRUE)), .groups = "drop") |>
        dplyr::mutate(factor = flab)
    })

    dom <- ranges |>
      dplyr::group_by(xc, yc) |>
      dplyr::slice_max(rng, n = 1L, with_ties = FALSE) |>
      dplyr::ungroup() |>
      dplyr::rename(x = xc, y = yc)

    var_df <- cls_bar |>
      dplyr::mutate(r3d_class = factor(r3d_class, levels = r3d_cls_lvls)) |>
      dplyr::left_join(dom, by = c("x", "y")) |>
      dplyr::filter(!is.na(factor)) |>
      dplyr::mutate(factor = factor(factor, levels = names(fac_cols))) |>
      dplyr::count(r3d_class, factor) |>
      dplyr::group_by(r3d_class) |>
      dplyr::mutate(prop = n / sum(n)) |>
      dplyr::ungroup()

    fac_pal <- c(form = "#E41A1C", scaling = "#377EB8",
                 construction = "#4DAF4A", policy = "#984EA3")
    ggplot(var_df, aes(r3d_class, prop, fill = factor)) +
      geom_col(width = 0.65, colour = "white", linewidth = 0.3) +
      scale_y_continuous(labels = scales::percent_format(1), expand = c(0.01, 0)) +
      scale_fill_manual(values = fac_pal[names(fac_cols)], name = NULL) +
      labs(title = "Dominant variance source", x = NULL, y = "Share of class pixels") +
      theme_minimal() +
      theme(panel.grid.major.x = element_blank(), legend.position = "bottom")
  }, error = function(e) { message("    [skip] variance panel: ", conditionMessage(e)); NULL })

  # ---- Panel G: restoration potential distribution within each LULC class ------
  # Boxplot of per-pixel RP = (abiotic + biotic) / 2, grouped by LULC class on
  # the x-axis and split by r3d_class (fill) so Core/Fringe/Marginal are
  # side-by-side within each land-cover type.
  p_lulc_rp <- tryCatch({
    lulc_vals2 <- .r3d_extract(LULC_RASTER)
    abio_vals2 <- .r3d_extract(ABIOTIC_RASTER)
    bio_vals2  <- .r3d_extract(BIOTIC_RASTER)
    if (is.null(lulc_vals2) || is.null(abio_vals2) || is.null(bio_vals2))
      stop("LULC_RASTER, ABIOTIC_RASTER or BIOTIC_RASTER not set, missing, or terra unavailable")
    lulc_rp_df <- cls_bar |>
      dplyr::mutate(
        lulc      = lu_map[as.character(lulc_vals2)],
        lulc      = factor(lulc, levels = lu_lvls),
        rp        = (abio_vals2 + bio_vals2) / 2,
        r3d_class = factor(r3d_class, levels = r3d_cls_lvls)
      ) |>
      dplyr::filter(!is.na(lulc), !is.na(rp))
    ggplot(lulc_rp_df, aes(lulc, rp, fill = r3d_class)) +
      geom_boxplot(outlier.size = 0.3, outlier.alpha = 0.3,
                   position = position_dodge(0.75), width = 0.65, colour = "grey30") +
      scale_fill_manual(values = r3d_cls_cols, name = NULL, drop = FALSE) +
      labs(title = "Restoration potential by LULC class",
           x = NULL, y = "Restoration potential",
           caption = "Lower = more degraded") +
      theme_minimal() +
      theme(panel.grid.major.x = element_blank(),
            legend.position = "bottom",
            axis.text.x = element_text(angle = 20, hjust = 1),
            plot.caption = element_text(size = 7, colour = "grey50"))
  }, error = function(e) { message("    [skip] LULC x RP panel: ", conditionMessage(e)); NULL })

  # ---- Assemble and save -------------------------------------------------------
  # Individual panels (one PNG each)
  .r3d_panel_specs <- list(
    list(p = p_lulc,    nm = "R3d_panel_A_lulc.png"),
    list(p = p_hab,     nm = "R3d_panel_B_habitat.png"),
    list(p = p_abio,    nm = "R3d_panel_C_abiotic.png"),
    list(p = p_bio,     nm = "R3d_panel_D_biotic.png"),
    list(p = p_rp,      nm = "R3d_panel_E_restoration_potential.png"),
    list(p = p_var,     nm = "R3d_panel_F_variance.png"),
    list(p = p_lulc_rp, nm = "R3d_panel_G_lulc_rp.png")
  )
  for (.ps in .r3d_panel_specs) {
    if (!is.null(.ps$p)) .save(.ps$p, .ps$nm, w = 5, h = 5.5)
  }
  # Combined multi-panel figure
  panels <- Filter(Negate(is.null), list(p_lulc, p_hab, p_abio, p_bio, p_rp, p_var, p_lulc_rp))
  if (length(panels) > 0L) {
    if (.has_patchwork && length(panels) > 1L) {
      p_sum <- patchwork::wrap_plots(panels, nrow = 1) +
        patchwork::plot_annotation(
          title = "R3d full classification summary"
          #, subtitle = sprintf(
          #    "Core >= %.0f%%, fringe >= %.0f%%, top %d%% priority, 1 km cells",
          #    100 * R3D_HI, 100 * R3D_LO, R3D_TOP_PCT)
        )
    } else {
      p_sum <- panels[[1]]
    }
    .save(p_sum, "R3d_full_class_summary_bars.png",
          w = 3 + 3.5 * length(panels), h = 5.5)
  }
}
}  # end R3

# ============================================================================
# R4 — Uncertainty attribution (which assumption moves priorities most)
# ----------------------------------------------------------------------------
# One-at-a-time decomposition: per-pixel SD of group-mean RFOP for each axis
# that exists as a standalone grid. With the current exports that is only the
# indicator axis (Blocks 1+2) and the seed noise floor (Block 1 replicates);
# policy and formulation are NOT separate grids - they are factors inside the
# Block 3 factorial, so they cannot appear here. For the full side-by-side
# comparison including policy + form/scaling/construction, see R4f (crossed
# variance partition), which is the proper estimator for a factorial design.
# (anova_dims still lists policy/formulation so the bar auto-extends if those
# ever return as standalone exports via AXIS3_DIR / AXIS2_DIR.)
# ============================================================================
if (.run("R4")) {
message("\nR4 - uncertainty attribution (one-at-a-time SD; standalone axes only)")
if (nrow(runs_df) > 0L) {
  anova_dims <- intersect(c("indicator", "policy", "formulation", "seed"), unique(runs_df$dim_type))
  anova_df <- tryCatch(compute_rfop_anova(runs_df, dims = anova_dims, se_dims = "seed"),
                       error = function(e) { message("  ", conditionMessage(e)); NULL })
  if (!is.null(anova_df)) {
    .save(tryCatch(make_anova_summary_bar(anova_df, dims = anova_dims),
                   error = function(e) { message("  ", conditionMessage(e)); NULL }),
          "R4a_attribution_summary_bar.png", 8, 6)
    .save(tryCatch(make_anova_map(anova_df, elig_df = elig_df),
                   error = function(e) { message("  ", conditionMessage(e)); NULL }),
          "R4b_attribution_map.png", 9, 8, panel_cm = 14)
  }
}

# ============================================================================
# R4f — Factorial attribution: crossed-design variance partitioning (Block 3)
# ----------------------------------------------------------------------------
# Upgrades R4 from one-at-a-time (per-dimension SD) to a proper crossed-factorial
# variance partition. For each spatial cell it decomposes RFOP variance into the
# MAIN effects of each formulation factor (form, scaling, construction, policy),
# their INTERACTIONS, and the RESIDUAL (seed) noise floor — estimating
# interactions rather than assuming the factors are independent.
#
# Requires FACTORIAL_DIR to point at a "factorial" SCENARIO_MODE export. Runs at
# the same coarse cell scale as R3 so cell-wise aov() is stable and fast.
# ============================================================================
message("\nR4f — factorial variance partitioning (crossed design)")
VP_CELL <- 1000   # aggregate to 1 km before per-cell aov (matches R3c overlap scale)
if (!is.null(FACTORIAL_DIR) && dir.exists(FACTORIAL_DIR)) {
  fac_dirs <- .list_factorial_run_dirs(FACTORIAL_DIR)
  fac_df <- purrr::map_dfr(fac_dirs, function(d)
    tryCatch(load_run_factorial(d),
             error = function(e) { message("  [skip] ", basename(d), ": ", conditionMessage(e)); tibble::tibble() }))

  if (nrow(fac_df) > 0L) {
    # Aggregate RFOP to VP_CELL-metre cells, preserving the factor columns.
    fac_agg <- if (VP_CELL > 0) {
      fac_df |>
        dplyr::mutate(x = VP_CELL * (x %/% VP_CELL) + VP_CELL / 2,
                      y = VP_CELL * (y %/% VP_CELL) + VP_CELL / 2) |>
        dplyr::group_by(x, y, run_label, form, scaling, construction, policy, seed) |>
        dplyr::summarise(rfop_pct = mean(rfop_pct, na.rm = TRUE), .groups = "drop")
    } else fac_df

    vp <- tryCatch(
      compute_rfop_variance_partition(
        fac_agg, factors = c("form", "scaling", "construction", "policy")),
      error = function(e) { message("  ", conditionMessage(e)); NULL })

    if (!is.null(vp) && nrow(vp) > 0L) {
      # Global variance shares (mean eta^2 per source) — quick console summary.
      share <- .pool_vp_components(vp) |>
        dplyr::group_by(component) |>
        dplyr::summarise(mean_eta2 = mean(eta2, na.rm = TRUE), .groups = "drop") |>
        dplyr::arrange(dplyr::desc(mean_eta2))
      message("  mean variance share (eta^2) per source:")
      for (i in seq_len(nrow(share)))
        message(sprintf("    %-14s %.3f", share$component[i], share$mean_eta2[i]))

      edf_vp <- .aggregate_elig(elig_df, VP_CELL)
      .save(make_variance_partition_bar(vp), "R4f_variance_partition_bar.png", 8, 6)
      .save(make_dominant_factor_map(vp, elig_df = edf_vp),
            "R4f_dominant_factor_map.png", 9, 8, panel_cm = 14)

      # Break the aggregate interaction share into individual terms so a specific
      # two-way interaction can be named (e.g. scaling x form), and map where each
      # interaction is spatially concentrated.
      vp_int <- summarise_vp_interactions(vp)
      if (nrow(vp_int) > 0L) {
        message("  interaction breakdown (mean eta^2 per term):")
        for (i in seq_len(nrow(vp_int)))
          message(sprintf("    %-34s %.3f (n=%d)",
                          vp_int$label[i], vp_int$mean_eta2[i], vp_int$n_cells[i]))
        readr::write_csv(vp_int, file.path(OUT_DIR, "R4f_interaction_breakdown.csv"))
        .save(make_vp_interaction_bar(vp), "R4f_interaction_breakdown_bar.png", 8, 6)
        .save(make_dominant_interaction_map(vp, elig_df = edf_vp),
              "R4f_dominant_interaction_map.png", 9, 8, panel_cm = 14)
      }
    }

    # ------------------------------------------------------------------------
    # R4g - Is the seed (residual) noise floor uniform across design cells?
    # ------------------------------------------------------------------------
    # Reads the SAME fac_agg arrays feeding the R4f variance partition; opens up
    # the pooled "residual" into per-cell, per-patch seed SD. A patch absent from
    # a seed is treated as RFOP = 0 (zero_fill default), so presence/absence flips
    # across seeds count as seed noise.
    #   Step 1: mean within-cell seed SD per design cell (bar + table).
    #   Step 2: Jaccard overlap of each cell's high-noise patch set (heatmap),
    #           plus a per-factor association check on the overlap.
    # ------------------------------------------------------------------------
    message("\nR4g - within-cell seed noise: magnitude and spatial consistency")
    SEED_NOISE_TOP_PCT <- 20    # high-noise patch set = top 20% by within-cell seed SD
    noise <- tryCatch(
      compute_seed_noise_by_cell(fac_agg,
                                 factors = c("form", "scaling", "construction", "policy")),
      error = function(e) { message("  ", conditionMessage(e)); NULL })

    if (!is.null(noise) && nrow(noise$per_cell) > 0L) {
      # Step 1 - magnitude.
      message(sprintf("  seed SD across %d design cells: range %.2f-%.2f, mean %.2f pp",
                      nrow(noise$per_cell),
                      min(noise$per_cell$mean_seed_sd, na.rm = TRUE),
                      max(noise$per_cell$mean_seed_sd, na.rm = TRUE),
                      mean(noise$per_cell$mean_seed_sd, na.rm = TRUE)))
      readr::write_csv(noise$per_cell, file.path(OUT_DIR, "R4g_seed_noise_by_cell.csv"))
      .save(make_seed_noise_bar(noise), "R4g_seed_noise_by_cell_bar.png", 9, 10)

      # Step 2 - spatial consistency of the high-noise patches.
      ovl_noise <- tryCatch(
        compute_seed_noise_overlap(noise$per_patch, top_pct = SEED_NOISE_TOP_PCT),
        error = function(e) { message("  ", conditionMessage(e)); NULL })

      if (!is.null(ovl_noise) && nrow(ovl_noise$jaccard) > 0L) {
        message(sprintf("  mean pairwise Jaccard of high-noise (top-%d%%) patch sets = %.2f",
                        SEED_NOISE_TOP_PCT, ovl_noise$mean_jaccard))
        .save(make_seed_noise_jaccard_heatmap(ovl_noise),
              "R4g_seed_noise_jaccard_heatmap.png", 10, 9)

        if (ovl_noise$mean_jaccard > 0.7) {
          message("  -> high & uniform overlap: seed noise is landscape-driven (formulation-invariant).")
        } else {
          assoc <- compute_seed_noise_factor_association(ovl_noise, noise$per_cell,
                                                         factors = noise$factors)
          readr::write_csv(assoc, file.path(OUT_DIR, "R4g_seed_noise_factor_association.csv"))
          message("  -> overlap varies; factor association (mean Jaccard same vs diff level):")
          for (i in seq_len(nrow(assoc)))
            message(sprintf("       %-13s same=%.2f diff=%.2f gap=%+.2f",
                            assoc$factor[i], assoc$mean_jac_same[i],
                            assoc$mean_jac_diff[i], assoc$gap[i]))
          if (nrow(assoc) > 0L && assoc$gap[1] > 0.05)
            message(sprintf("  -> noise footprint shifts most with: %s (gap %+.2f)",
                            assoc$factor[1], assoc$gap[1]))
        }
      }
    } else {
      message("  [skip] R4g: no usable per-cell seed noise (need >=2 seeds per design cell)")
    }
  } else {
    message("  [skip] no loadable factorial runs in FACTORIAL_DIR")
  }
} else {
  message("  [skip] FACTORIAL_DIR not set or missing (run SCENARIO_MODE='factorial' first)")
}
}  # end R4

# ============================================================================
# R5 -- Cross-formulation regret matrix
# ----------------------------------------------------------------------------
# For every ordered pair (plan p, evaluating formulation f):
#   regret(p, f) = (V*_f - V_p,f) / V*_f
# where V_p,f = RP score of plan p's pixel selection evaluated under
# formulation f's objective, and V*_f = formulation f's own optimum
# (the score its champion plan achieves, used as the denominator so regret
# is unit-free and comparable across sum and threshold forms).
#
# Formulations (columns): 4 = (form=sum/threshold) x (scaling=global/q75).
# Plans        (rows):    one champion plan per formulation (best RP solution
#                         pooled over all constructions and seeds) + core plan.
# Core plan = top R5_BUDGET pixels by consensus RFOP from runs_factorial.
#
# Both forms are normalised to their own V*; regret fractions are comparable
# across sum-gain and area-over-threshold despite different native units.
#
# Outputs:
#   R5_regret_matrix.csv    -- regret matrix (%, 2 d.p.)
#   R5_regret_heatmap.png   -- tile heatmap with annotated X / Y values
# Console: X (core premium), Y (wrong-guess cost), X<<Y check.
# ============================================================================
if (.run("R5")) {
message("\nR5 -- cross-formulation regret matrix")

R5_BUDGET   <- 21855L   # max_action_pixels; same for every factorial cell
R5_ANOM_DIR <- file.path("inputs", "anomaly_scenarios")
R5_THRESH   <- 0.0      # threshold form counts pixels with combined_anomaly > this

# Helper: parse one factor token from a run-dir name (matches .r3d_tok style).
.r5_tok <- function(nm, key) {
  pfx  <- paste0(key, "-")
  toks <- strsplit(nm, "__", fixed = TRUE)[[1]]
  hit  <- toks[startsWith(toks, pfx)]
  if (length(hit) == 0L) NA_character_ else sub(pfx, "", hit[1L], fixed = TRUE)
}

r5_skip <- is.null(FACTORIAL_DIR) || !dir.exists(FACTORIAL_DIR) ||
           !requireNamespace("terra", quietly = TRUE)
if (r5_skip) {
  message("  [skip] R5: FACTORIAL_DIR not set or terra unavailable")
} else {

# -- Combined anomaly raster for each scaling: (abiotic + biotic) / 2 -------
# Always use the "all"-construction raster so the cross-formulation objective
# is evaluated against the full indicator set regardless of how each plan was
# built (construction varies the plan, not the formulation being evaluated).
# Sign convention: raster stores anomaly as negative for degraded pixels
# (lower = more degraded; matching R3d "lower = more degraded" panel label).
# The optimizer maximises RP = -sum(anomaly) for sum form and
# count(anomaly < rp_threshold) for threshold form, so both return positive
# values that increase with degradation depth / extent.
.r5_load_rast <- function(scal) {
  ab <- normalizePath(file.path(R5_ANOM_DIR, paste0("abiotic_", scal, "_all.tif")),
                      mustWork = FALSE)
  bi <- normalizePath(file.path(R5_ANOM_DIR, paste0("biotic_",  scal, "_all.tif")),
                      mustWork = FALSE)
  if (!file.exists(ab)) { message("  [warn] R5: not found: ", ab); return(NULL) }
  if (!file.exists(bi)) { message("  [warn] R5: not found: ", bi); return(NULL) }
  tryCatch({
    ra <- terra::rast(ab); rb <- terra::rast(bi)
    if (!terra::compareGeom(ra, rb, stopOnError = FALSE))
      message("  [warn] R5: ", scal, " abiotic/biotic extents differ; adding anyway")
    (ra + rb) / 2
  }, error = function(e) { message("  [warn] R5: ", scal, " raster load error: ",
                                    conditionMessage(e)); NULL })
}
r5_rasts <- list(global    = .r5_load_rast("global"),
                 upper_q75 = .r5_load_rast("upper_q75"))
for (k in names(r5_rasts))
  if (!is.null(r5_rasts[[k]]))
    message(sprintf("  raster loaded: %-10s  CRS=%s", k, terra::crs(r5_rasts[[k]], describe=TRUE)$code))
if (all(vapply(r5_rasts, is.null, logical(1)))) {
  message("  [skip] R5: no condition rasters found in ", R5_ANOM_DIR)
} else {

# Helper: extract combined anomaly values at pixel coords (x, y in EPSG:2056).
# Explicitly reprojects points to the raster CRS before extraction (same pattern
# as .r3d_extract in _plot_functions.R) so CRS mismatches are handled robustly.
.r5_extract <- function(scal_key, xy) {
  r <- r5_rasts[[scal_key]]
  if (is.null(r) || nrow(xy) == 0L) return(rep(NA_real_, max(nrow(xy), 1L)))
  tryCatch({
    pts <- terra::vect(as.data.frame(xy), geom = c("x", "y"), crs = "EPSG:2056")
    pts <- terra::project(pts, terra::crs(r))
    terra::extract(r, pts)[, 2L]
  }, error = function(e) { message("  [warn] .r5_extract: ", conditionMessage(e));
                           rep(NA_real_, nrow(xy)) })
}

# Helper: compute RP score from extracted anomaly values.
#   sum form       -> -sum(vals)          (negative anomaly -> positive degradation sum)
#   threshold form -> count(vals < thresh) (count degraded pixels below the threshold)
# Both return positive numbers that increase with restoration potential, matching
# the sign convention in objectives.csv after load_run_data's sign inversion.
.r5_score <- function(vals, form) {
  v <- vals[!is.na(vals)]
  if (length(v) == 0L) return(NA_real_)
  if (form == "sum") -sum(v) else sum(v < R5_THRESH)
}

# Four formulations (the objective-function space).
r5_forms <- list(
  "sum/global"    = list(form = "sum",       scaling = "global",    scal_key = "global"),
  "sum/q75"       = list(form = "sum",       scaling = "upper_q75", scal_key = "upper_q75"),
  "thresh/global" = list(form = "threshold", scaling = "global",    scal_key = "global"),
  "thresh/q75"    = list(form = "threshold", scaling = "upper_q75", scal_key = "upper_q75")
)

# Status-quo dirs only (ambitious + burden_shared already excluded by
# EXCLUDE_FACTORIAL_POLICIES via .list_factorial_run_dirs).
r5_sq_dirs <- Filter(
  function(d) identical(.r5_tok(basename(d), "pol"), "status_quo"),
  .list_factorial_run_dirs(FACTORIAL_DIR)
)

if (length(r5_sq_dirs) == 0L) {
  message("  [skip] R5: no status_quo factorial runs found")
} else {

# -- Step 1: Champion plan and pixel set for each formulation ----------------
# Champion = pixel selection of the single best-RP solution pooled across all
# constructions and seeds for that (form x scaling) pair.
# RP stored negative in objectives.csv (minimised); best = most-negative value.
r5_champ <- lapply(r5_forms, function(fspec) {
  dirs <- Filter(function(d) {
    identical(.r5_tok(basename(d), "form"), fspec$form) &&
    identical(.r5_tok(basename(d), "scal"), fspec$scaling)
  }, r5_sq_dirs)
  if (length(dirs) == 0L) return(NULL)

  best_rp <- Inf; best_d <- NULL; best_sid <- NA_integer_
  for (d in dirs) {
    obj <- tryCatch(read.csv(file.path(d, "objectives.csv")), error = function(e) NULL)
    if (is.null(obj) || !"restoration_potential" %in% names(obj)) next
    # Restrict to non-dominated solutions: pixel_selection.csv only stores those.
    nd <- obj[obj$is_nondominated == 1L, ]
    if (nrow(nd) == 0L) next
    idx <- which.min(nd$restoration_potential)    # most negative = best RP on Pareto front
    if (nd$restoration_potential[idx] < best_rp) {
      best_rp  <- nd$restoration_potential[idx]
      best_d   <- d
      best_sid <- nd$solution_id[idx]
    }
  }
  if (is.null(best_d)) return(NULL)

  psel <- tryCatch(read.csv(file.path(best_d, "pixel_selection.csv")),
                   error = function(e) NULL)
  pixels <- if (!is.null(psel))
    unique(psel[psel$solution_id == best_sid, c("x", "y")])
  else
    data.frame(x = numeric(0), y = numeric(0))
  pixels <- .filter_within_be(pixels)

  message(sprintf("  %s: champion in %s  (n_pixels=%d, raw_rp=%.1f)",
                  paste(fspec$form, fspec$scaling, sep = "/"),
                  basename(best_d), nrow(pixels), best_rp))
  list(form = fspec$form, scaling = fspec$scaling,
       scal_key = fspec$scal_key, pixels = pixels)
})
r5_champ <- Filter(Negate(is.null), r5_champ)

if (length(r5_champ) == 0L) {
  message("  [skip] R5: no champion plans loaded")
} else {

# -- Step 2: Core plan = top R5_BUDGET pixels by consensus RFOP --------------
# runs_factorial was built from .list_factorial_run_dirs() so it excludes
# ambitious/burden_shared; filter to dim_type == "factorial" for safety.
r5_fac_df <- dplyr::filter(runs_factorial, dim_type == "factorial")

r5_core_pixels <- if (nrow(r5_fac_df) > 0L) {
  # Average within each (form x scaling x construction) scenario (over seeds),
  # then average across scenarios so each design cell counts equally.
  core_rfop <- r5_fac_df |>
    dplyr::group_by(x, y, group_label) |>
    dplyr::summarise(rfop_pct = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    dplyr::group_by(x, y) |>
    dplyr::summarise(mean_rfop = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(mean_rfop))
  n_take <- min(R5_BUDGET, nrow(core_rfop))
  message(sprintf("  Core plan: top %d pixels by consensus RFOP (%.0f%% of budget)",
                  n_take, 100 * n_take / R5_BUDGET))
  core_rfop[seq_len(n_take), c("x", "y")]
} else {
  message("  [warn] R5: no factorial rows for core plan -- using empty selection")
  data.frame(x = numeric(0), y = numeric(0))
}

# -- Step 3: Score matrix: raw RP score of every plan under every formulation -
# V_p,f computed from the anomaly rasters for consistency across all (p, f) pairs;
# raster-evaluated V*_f (champion-plan diagonal) cancels any offset from the
# optimizer's internal computation.
all_plans <- c(r5_champ, list(core = list(form = NA, scaling = NA,
                                           scal_key = NA, pixels = r5_core_pixels)))
score_mat <- matrix(NA_real_, nrow = length(all_plans), ncol = length(r5_forms),
                    dimnames = list(names(all_plans), names(r5_forms)))

for (pnm in names(all_plans)) {
  pxy <- all_plans[[pnm]]$pixels
  for (fnm in names(r5_forms)) {
    fspec <- r5_forms[[fnm]]
    vals  <- .r5_extract(fspec$scal_key, pxy)
    score_mat[pnm, fnm] <- .r5_score(vals, fspec$form)
  }
}

# V*_f = raster-evaluated score of each formulation's own champion plan.
# Requires the champion name to match the formulation name (same ordering).
champ_names <- intersect(names(r5_forms), names(r5_champ))
V_star <- vapply(champ_names, function(fnm) score_mat[fnm, fnm], numeric(1))
names(V_star) <- champ_names

message("\n  V*_f (formulation optima, raster-evaluated):")
for (fnm in names(V_star))
  message(sprintf("    %-14s %.1f", fnm, V_star[fnm]))

# -- Step 4: Regret matrix ---------------------------------------------------
# regret(p, f) = (V*_f - V_p,f) / V*_f
# Diagonal entries are 0 by construction (plan-f under formulation-f = V*_f).
# Clip negatives to 0: a cross-plan that accidentally exceeds V*_f due to the
# "all"-construction raster proxy gets 0 regret rather than a negative value.
regret_mat <- matrix(NA_real_, nrow = nrow(score_mat), ncol = length(V_star),
                     dimnames = list(rownames(score_mat), names(V_star)))
for (fnm in names(V_star)) {
  if (is.na(V_star[fnm]) || V_star[fnm] <= 0) next
  regret_mat[, fnm] <- pmax(0, (V_star[fnm] - score_mat[, fnm]) / V_star[fnm])
}

# -- Step 5: Summary numbers -------------------------------------------------
# X: core premium -- core plan's regret averaged and worst-cased over all forms.
core_regrets <- regret_mat["core", names(V_star)]
X_mean <- mean(core_regrets, na.rm = TRUE)
X_max  <- max(core_regrets,  na.rm = TRUE)

# Y: wrong-guess cost -- worst-case off-diagonal regret among the 4 champion plans.
# Off-diagonal: plan p evaluated under formulation f != p's own formulation.
champ_sq  <- regret_mat[champ_names, champ_names, drop = FALSE]
diag(champ_sq) <- NA_real_   # zero by construction; exclude from max
Y <- max(champ_sq, na.rm = TRUE)

message(sprintf("\n  Core premium  X: mean=%.1f%%  worst-case=%.1f%%",
                100 * X_mean, 100 * X_max))
message(sprintf("  Wrong-guess Y:  worst-case off-diagonal = %.1f%%", 100 * Y))
if (!is.na(X_max) && !is.na(Y) && Y > 0) {
  ratio <- X_max / Y
  verdict <- if (ratio < 0.5) "YES (X_max < Y/2)" else if (ratio < 1) "marginal" else "NO"
  message(sprintf("  X << Y: %s  (ratio X_max/Y = %.2f)", verdict, ratio))
}

# -- Save CSV ----------------------------------------------------------------
regret_pct <- round(regret_mat[, names(V_star), drop = FALSE] * 100, 2)
regret_df  <- tibble::as_tibble(regret_pct, rownames = "plan")
readr::write_csv(regret_df, file.path(OUT_DIR, "R5_regret_matrix.csv"))
message("  Regret matrix written to R5_regret_matrix.csv")

# -- Heatmap -----------------------------------------------------------------
plan_order <- c(rev(champ_names), "core")
plan_order <- plan_order[plan_order %in% rownames(regret_pct)]
regret_long <- regret_df |>
  tidyr::pivot_longer(-plan, names_to = "formulation", values_to = "regret_pct") |>
  dplyr::mutate(
    plan        = factor(plan, levels = plan_order),
    formulation = factor(formulation, levels = names(V_star))
  )
subtitle_txt <- sprintf(
  "X (core premium): mean=%.1f%%  worst=%.1f%%  |  Y (wrong-guess): %.1f%%  |  X<<Y: %s",
  100 * X_mean, 100 * X_max, 100 * Y,
  if (!is.na(X_max) && !is.na(Y) && X_max < Y / 2) "yes" else "marginal/no")
p_regret <- ggplot(regret_long, aes(formulation, plan, fill = regret_pct)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.1f", regret_pct)),
            size = 3.8, colour = "grey10") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1,
                       limits = c(0, NA), name = "Regret (%)") +
  labs(title = "Cross-formulation regret matrix",
       subtitle = subtitle_txt,
       x = "Evaluating formulation (f)", y = "Plan (p)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        panel.grid   = element_blank())
.save(p_regret, "R5_regret_matrix_heatmap.png", 8, 5)

# -- Step 6: Per-formulation selection frequency maps ------------------------
# For each (form x scaling) pair, average RFOP over all construction x seed
# runs (filtered to status_quo policy when that column exists) to show spatial
# consensus within that formulation.
form_key_df <- dplyr::bind_rows(lapply(names(r5_forms), function(fnm) {
  data.frame(form    = r5_forms[[fnm]]$form,
             scaling = r5_forms[[fnm]]$scaling,
             formulation = fnm, stringsAsFactors = FALSE)
}))

# Use fac_df (has full factor columns) if available; reload otherwise.
r5_fac_full <- if (exists("fac_df") && is.data.frame(fac_df) && nrow(fac_df) > 0L) {
  fac_df
} else {
  purrr::map_dfr(.list_factorial_run_dirs(FACTORIAL_DIR), function(d)
    tryCatch(load_run_factorial(d), error = function(e) tibble::tibble()))
}

r5_sq_fac <- {
  df <- r5_fac_full
  if ("policy" %in% names(df))
    df <- dplyr::filter(df, is.na(policy) | policy == "status_quo")
  dplyr::inner_join(df, form_key_df, by = c("form", "scaling"))
}

if (nrow(r5_sq_fac) > 0L) {
  r5_freq_df <- r5_sq_fac |>
    dplyr::group_by(x, y, formulation) |>
    dplyr::summarise(rfop_pct = mean(rfop_pct, na.rm = TRUE), .groups = "drop") |>
    dplyr::mutate(formulation = factor(formulation, levels = names(r5_forms)))

  panel_labels <- c("sum/global"    = "Sum / global",
                    "sum/q75"       = "Sum / upper-q75",
                    "thresh/global" = "Threshold / global",
                    "thresh/q75"    = "Threshold / upper-q75")

  p_form_freq <- ggplot(r5_freq_df, aes(x, y, fill = rfop_pct)) +
    geom_raster() +
    facet_wrap(~ formulation, nrow = 2,
               labeller = labeller(formulation = panel_labels)) +
    scale_fill_distiller(palette = "YlOrRd", direction = 1,
                         limits = c(0, 100),
                         name = "Selection\nfrequency (%)") +
    labs(title = "Per-formulation selection frequency (mean over construction x seed)") +
    theme_void() +
    theme(strip.text      = element_text(size = 9, face = "bold"),
          plot.title      = element_text(size = 11),
          panel.spacing   = unit(4, "mm"),
          legend.position = "right")
  if (exists("BE"))
    p_form_freq <- p_form_freq +
      geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.35,
              inherit.aes = FALSE)
  .save(p_form_freq, "R5_formulation_freq_maps.png", panel_cm = 28)
  message("  Per-formulation frequency maps written to R5_formulation_freq_maps.png")
} else {
  message("  [skip] R5 freq maps: no status_quo factorial rows matched formulations")
}

}  # end if length(r5_champ)
}  # end if length(r5_sq_dirs)
}  # end if rasters available
}  # end if !r5_skip
}  # end R5

message("\nDone. Figures in: ", normalizePath(OUT_DIR))
