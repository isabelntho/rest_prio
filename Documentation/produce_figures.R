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

# Axis 1 — condition-indicator construction (condition_grid export root).
# iEMSs agricultural run, Blocks 1+2: baseline (global_all) + agricultural LOO drops
# + q75 benchmark, 5 seeds (50 runs).
AXIS1_DIR <- file.path(R_INPUTS, "20260623_1222_iEMSs_block1&2")
# Axis 3 — policy constraints. In the agricultural design policy is a factor INSIDE
# the Block 3 factorial (not a separate OAT grid), so there is no standalone policy
# export — keep NULL. (Old non-agricultural policy grids must NOT be mixed in here.)
AXIS3_DIR <- NULL
# Axis 2 — objective formulation (sum vs threshold). Folded into the factorial; NULL.
AXIS2_DIR <- NULL

# Block 4 — fully-crossed factorial export root (SCENARIO_MODE = "factorial" in
# run_custom.py): form × scaling × construction × policy × seeds. Set this to the
# r_inputs/<timestamp>_<label> grid folder once the factorial has been run; else
# NULL and the R4f variance-partition block is skipped.
FACTORIAL_DIR <- NULL

# Baseline scenario name (a sub-folder prefix within AXIS1_DIR, minus _seed<n>).
BASELINE_SCENARIO <- "global_all"
# Benchmark scenario names (treated as dim_type = "benchmark", not "indicator").
BENCHMARK_SCENARIOS <- c("upper_q75_all")

OUT_DIR <- "figs/iEMSs/iEMSs_agri"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# R3 robustness map is produced at these spatial scales (metres; x,y are EPSG:2056
# at 100 m). 0 = native pixel. Coarser cells average out fine-grained spatial
# degeneracy so a coherent robust core can emerge at a decision-relevant scale.
R3_AGG_CELLS <- c(0, 1000, 2000)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
.scenario_of <- function(dirname) sub("_seed[0-9]+$", "", dirname)   # strip seed suffix
.seed_of     <- function(dirname) stringr::str_extract(dirname, "seed[0-9]+$")

# List scenario_seed sub-folders that contain an export.
.list_run_dirs <- function(root) {
  if (is.null(root) || !dir.exists(root)) return(character(0))
  ds <- list.dirs(root, full.names = TRUE, recursive = FALSE)
  ds[file.exists(file.path(ds, "metadata.json"))]
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

.save <- function(plot, name, w = 9, h = 7) {
  if (is.null(plot)) { message("  [skip fig] ", name, " (NULL)"); return(invisible()) }
  path <- file.path(OUT_DIR, name)
  tryCatch({ ggsave(path, plot, width = w, height = h, dpi = 200); message("  [ok] ", path) },
           error = function(e) message("  [FAIL fig] ", name, ": ", conditionMessage(e)))
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
.BE_SHP   <- "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"
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

# ----------------------------------------------------------------------------
# Build the combined runs_df spanning all uncertainty axes.
#   Axis 1 condition scenarios  -> dim_type "indicator" (q75 -> "benchmark")
#   Axis 3 policy variants      -> dim_type "policy"
#   Baseline across seeds       -> dim_type "seed" (group_label = seed id)
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

runs_df <- dplyr::bind_rows(runs_indicator, runs_policy, runs_axis2, runs_seed)
message(sprintf("  runs_df: %d rows | dim_types: %s",
                nrow(runs_df), paste(unique(runs_df$dim_type), collapse = ", ")))

# ============================================================================
# D — Convergence diagnostic (gate check before trusting robustness/attribution)
# ----------------------------------------------------------------------------
# If hypervolume is still climbing at the final generation, seeds settle on
# different regions of the front and seed noise inflates the sensitivity/
# attribution results. Flags any run whose HV gained > CONV_TOL over the last
# CONV_WINDOW generations (relative to its final HV).
# ============================================================================
message("\nD — hypervolume convergence diagnostic")
CONV_WINDOW <- 10      # generations from the end over which to measure tail gain
CONV_TOL    <- 0.01    # > 1% tail gain (relative to final HV) = "still rising"

hv_all <- purrr::map_dfr(
  c(.list_run_dirs(AXIS1_DIR), .list_run_dirs(AXIS3_DIR), .list_run_dirs(AXIS2_DIR)),
  function(d) {
    hv <- read_csv_if_exists(file.path(d, "hypervolume_evolution.csv"))
    if (is.null(hv) || nrow(hv) < 2) return(tibble::tibble())
    hv$run_label <- basename(d)
    hv
  })

if (nrow(hv_all) > 0L) {
  conv <- hv_all |>
    dplyr::group_by(run_label) |>
    dplyr::arrange(generation, .by_group = TRUE) |>
    dplyr::summarise(
      gen_max  = max(generation),
      final_hv = dplyr::last(hypervolume),
      tail_hv0 = hypervolume[which.min(abs(generation - (max(generation) - CONV_WINDOW)))],
      .groups  = "drop"
    ) |>
    dplyr::mutate(
      tail_gain    = (final_hv - tail_hv0) / pmax(final_hv, 1e-12),
      still_rising = tail_gain > CONV_TOL
    )

  n_rising <- sum(conv$still_rising, na.rm = TRUE)
  message(sprintf("  %d/%d runs STILL RISING (>%.0f%% HV gain over last %d gens)",
                  n_rising, nrow(conv), 100 * CONV_TOL, CONV_WINDOW))
  if (n_rising > 0L)
    message("  -> ", paste(conv$run_label[conv$still_rising], collapse = ", "))

  # Normalised HV curves (per-run max = 1), coloured by convergence flag.
  hv_plot_df <- hv_all |>
    dplyr::group_by(run_label) |>
    dplyr::mutate(hv_norm = hypervolume / max(hypervolume, na.rm = TRUE)) |>
    dplyr::ungroup() |>
    dplyr::left_join(dplyr::select(conv, run_label, still_rising), by = "run_label")

  p_conv <- ggplot(hv_plot_df,
                   aes(generation, hv_norm, group = run_label, colour = still_rising)) +
    geom_line(alpha = 0.5, linewidth = 0.4) +
    scale_colour_manual(values = c(`FALSE` = "grey50", `TRUE` = "#D55E00"),
                        labels = c(`FALSE` = "converged", `TRUE` = "still rising"),
                        name = NULL) +
    labs(title = "Hypervolume convergence per run",
         subtitle = sprintf("Normalised HV (per-run max = 1). %d/%d runs still rising at final generation.",
                            n_rising, nrow(conv)),
         x = "Generation", y = "HV / max(HV)") +
    theme_minimal()
  .save(p_conv, "D_hv_convergence.png", 10, 6)

  # Per-run tail-gain bar (sorted) so individual offenders are visible.
  p_tail <- conv |>
    dplyr::arrange(dplyr::desc(tail_gain)) |>
    dplyr::mutate(run_label = factor(run_label, levels = run_label)) |>
    ggplot(aes(run_label, 100 * tail_gain, fill = still_rising)) +
    geom_col() +
    geom_hline(yintercept = 100 * CONV_TOL, linetype = "dashed") +
    scale_fill_manual(values = c(`FALSE` = "grey50", `TRUE` = "#D55E00"), guide = "none") +
    labs(title = "Tail HV gain per run",
         subtitle = sprintf("HV gain over last %d generations (relative to final). Dashed = %.0f%% threshold.",
                            CONV_WINDOW, 100 * CONV_TOL),
         x = NULL, y = "Tail HV gain (%)") +
    coord_flip() + theme_minimal() + theme(axis.text.y = element_text(size = 5))
  .save(p_tail, "D_hv_tail_gain.png", 8, 11)
} else {
  message("  [skip] no hypervolume_evolution.csv found")
}

# ============================================================================
# D2 — Convergence vs degeneracy: per-dimension SD on ALL vs CONVERGED runs
# ----------------------------------------------------------------------------
# Decisive test for the high seed-noise question. If the per-dimension SDs
# (especially seed) drop sharply when restricted to HV-converged runs, the
# noise was under-convergence -> add generations. If they barely move, the
# noise is spatial multimodality/degeneracy -> add seeds / coarsen the unit,
# not generations.
# ============================================================================
message("\nD2 — convergence vs degeneracy (per-dimension SD, all vs converged)")

# Mean per-pixel SD for each dimension present in a compute_rfop_anova() result.
.dim_mean_sd <- function(anova_df) {
  sd_cols <- grep("^rfop_sd_", names(anova_df), value = TRUE)
  if (length(sd_cols) == 0L) return(tibble::tibble(dimension = character(0), mean_sd = numeric(0)))
  tibble::tibble(
    dimension = sub("^rfop_sd_", "", sd_cols),
    mean_sd   = vapply(sd_cols, function(cc) mean(anova_df[[cc]], na.rm = TRUE), numeric(1))
  )
}

if (exists("conv") && nrow(runs_df) > 0L) {
  conv_labels <- conv$run_label[!conv$still_rising]
  runs_conv   <- dplyr::filter(runs_df, run_label %in% conv_labels)
  dims_cmp    <- intersect(c("indicator", "policy", "formulation", "seed"),
                           unique(runs_df$dim_type))

  a_all  <- tryCatch(compute_rfop_anova(runs_df, dims = dims_cmp, se_dims = "seed"),
                     error = function(e) { message("  ", conditionMessage(e)); NULL })
  a_conv <- tryCatch(compute_rfop_anova(runs_conv,
                       dims = intersect(dims_cmp, unique(runs_conv$dim_type)),
                       se_dims = "seed"),
                     error = function(e) { message("  ", conditionMessage(e)); NULL })

  if (!is.null(a_all) && !is.null(a_conv)) {
    cmp_df <- dplyr::bind_rows(
      dplyr::mutate(.dim_mean_sd(a_all),  set = "all runs"),
      dplyr::mutate(.dim_mean_sd(a_conv), set = "converged only")
    )
    cmp_wide <- tidyr::pivot_wider(cmp_df, names_from = set, values_from = mean_sd)

    message(sprintf("  (converged: %d/%d runs)", length(conv_labels), nrow(conv)))
    message("  per-dimension mean SD of group-mean RFOP (pp):")
    for (i in seq_len(nrow(cmp_wide))) {
      message(sprintf("    %-12s  all=%5.2f   converged=%5.2f   (%+.0f%%)",
                      cmp_wide$dimension[i],
                      cmp_wide[["all runs"]][i],
                      cmp_wide[["converged only"]][i],
                      100 * (cmp_wide[["converged only"]][i] / cmp_wide[["all runs"]][i] - 1)))
    }

    p_cmp <- ggplot(cmp_df, aes(dimension, mean_sd, fill = set)) +
      geom_col(position = position_dodge(width = 0.7), width = 0.6) +
      scale_fill_manual(values = c(`all runs` = "grey60", `converged only` = "#0072B2"),
                        name = NULL) +
      labs(title = "Convergence vs degeneracy",
           subtitle = "Per-dimension mean SD of RFOP. If 'seed' barely drops on converged-only,\nthe noise is spatial degeneracy (add seeds), not under-convergence (add generations).",
           x = NULL, y = "Mean SD of group-mean RFOP (pp)") +
      theme_minimal()
    .save(p_cmp, "D2_convergence_vs_degeneracy.png", 8, 6)
  }
} else {
  message("  [skip] D2 (no convergence table or empty runs_df)")
}

# ============================================================================
# R1 — Baseline trade-off + selection frequency (orientation)
# ============================================================================
message("\nR1 — baseline Pareto + selection frequency")
if (!is.na(.baseline_dir)) {
  run_base <- tryCatch(load_run_data(.baseline_dir, BASELINE_SCENARIO), error = function(e) NULL)
  if (!is.null(run_base)) {
    .save(tryCatch(make_pareto_extremes_plot(run_base, title = "Baseline Pareto front"),
                   error = function(e) { message("  ", conditionMessage(e)); NULL }),
          "R1a_baseline_pareto.png", 10, 6)
    if (exists("BE")) {
      .save(tryCatch(make_sel_freq_plot(run_base), error = function(e) { message("  ", conditionMessage(e)); NULL }),
            "R1b_baseline_selection_frequency.png", 9, 8)
    } else {
      message("  [skip] R1b selection-frequency map (BE boundary unavailable)")
    }
  }
}

# ============================================================================
# R2 — How each uncertainty axis reshapes the frontier (SHARED-reference HV)
# ----------------------------------------------------------------------------
# Uses compute_shared_hv() so HV is comparable across scenarios (the optimiser's
# per-run HV is not — different reference point + per-run normalisation). Axis 2
# (formulation) is EXCLUDED: its restoration_potential objective has different
# units (count vs sum) and is not comparable in the same objective space.
# ============================================================================
message("\nR2 — shared-reference hypervolume across scenarios (indicator + policy)")
runs_meta <- list()
for (root in c(AXIS1_DIR, AXIS3_DIR)) {
  for (d in .list_run_dirs(root)) {
    nm <- basename(d)
    runs_meta[[nm]] <- list(run = tryCatch(load_run_data(d, nm), error = function(e) NULL))
  }
}
runs_meta <- runs_meta[!vapply(runs_meta, function(m) is.null(m$run), logical(1))]
if (length(runs_meta) > 0L) {
  obj_names <- runs_meta[[1]]$run$obj_names
  shared_hv <- tryCatch(compute_shared_hv(runs_meta, obj_names),
                        error = function(e) { message("  ", conditionMessage(e)); NULL })
  if (!is.null(shared_hv) && nrow(shared_hv) > 0L) {
    ax1 <- basename(.list_run_dirs(AXIS1_DIR))
    ax3 <- basename(.list_run_dirs(AXIS3_DIR))
    # Tag dimensions for make_hv_sensitivity_plot; baseline also appears as the
    # 'seed' group so it anchors the policy panel.
    stats_df <- dplyr::bind_rows(
      tibble::tibble(run_label = ax1, dim_type = "indicator", group_label = .scenario_of(ax1)),
      tibble::tibble(run_label = ax3, dim_type = "policy",    group_label = .scenario_of(ax3)),
      tibble::tibble(run_label = ax1[.scenario_of(ax1) == BASELINE_SCENARIO],
                     dim_type = "seed", group_label = BASELINE_SCENARIO)
    ) |>
      dplyr::left_join(dplyr::rename(shared_hv, hypervolume = hv), by = "run_label") |>
      dplyr::filter(!is.na(hypervolume))
    .save(tryCatch(make_hv_sensitivity_plot(stats_df, ref_label = NULL),
                   error = function(e) { message("  ", conditionMessage(e)); NULL }),
          "R2_hypervolume_sensitivity.png", 11, 7)
  }
}

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
message("\nR3 — robustness to condition-indicator uncertainty (policy excluded)")
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
                       sens_df, elig_df = edf,
                       title = sprintf("Restoration priority: robust core vs indicator-sensitive (%s)", tag)),
                     error = function(e) { message("    ", conditionMessage(e)); NULL }),
            sprintf("R3a_robustness_classification_%s.png", tag), 9, 8)
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
               subtitle = "1 - Jaccard of each LOO top-priority set vs the full-set baseline; higher = more leverage",
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
      labs(title = sprintf("Priority consensus across condition scenarios (top %d%%, 1 km)", TOP_PCT)) +
      theme_void() +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12))
    if (exists("BE"))
      pc <- pc + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
    .save(pc, "R3c_priority_consensus_map.png", 9, 8)

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

    # ── Consensus-based robustness classification ───────────────────────────
    # "Robust" = a top priority in a HIGH SHARE of scenarios (not necessarily
    # all). This is the softer, more defensible definition: robustness as broad
    # agreement, not unanimity. Thresholds are tunable.
    CONSENSUS_HI <- 0.8   # >= 80% of scenarios rank it top -> robust priority
    CONSENSUS_LO <- 0.2   # < 20% -> marginal
    cls <- ovl$consensus |>
      dplyr::mutate(robust_class = factor(dplyr::case_when(
          frac >= CONSENSUS_HI ~ "robust priority",
          frac >= CONSENSUS_LO ~ "contested priority",
          TRUE                 ~ "marginal priority"),
        levels = c("robust priority", "contested priority", "marginal priority")))
    message(sprintf("  consensus classes (top %d%%): robust>=%.0f%%=%d | contested=%d | marginal=%d cells",
                    TOP_PCT, 100 * CONSENSUS_HI,
                    sum(cls$robust_class == "robust priority"),
                    sum(cls$robust_class == "contested priority"),
                    sum(cls$robust_class == "marginal priority")))
    pcl <- ggplot()
    if (!is.null(edf)) {
      bg2 <- dplyr::anti_join(edf, cls, by = c("x", "y"))
      if (nrow(bg2) > 0) pcl <- pcl + geom_raster(data = bg2, aes(x, y), fill = "#EEEEEE")
    }
    pcl <- pcl +
      geom_raster(data = cls, aes(x, y, fill = robust_class)) +
      scale_fill_manual(values = c("robust priority"    = "#1B7837",
                                   "contested priority" = "#D9A441",
                                   "marginal priority"  = "#BBBBBB"),
                        name = NULL, drop = FALSE) +
      labs(title = sprintf("Robust priority = top %d%% in >=%.0f%% of condition scenarios (1 km)",
                           TOP_PCT, 100 * CONSENSUS_HI)) +
      theme_void() +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12))
    if (exists("BE"))
      pcl <- pcl + geom_sf(data = BE, fill = NA, colour = "black", linewidth = 0.4, inherit.aes = FALSE)
    .save(pcl, "R3d_consensus_classification_map.png", 9, 8)
  }
}

# ============================================================================
# R4 — Uncertainty attribution (which assumption moves priorities most)
# ----------------------------------------------------------------------------
# The one place all axes are compared side-by-side: per-dimension variance of
# RFOP, decomposed by source (indicator, policy, formulation, seed). Policy is
# included here for reference but flagged as a controlled lever, not uncertainty.
# ============================================================================
message("\nR4 — uncertainty attribution (ANOVA / variance decomposition, all axes)")
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
          "R4b_attribution_map.png", 9, 8)
  }
}

# ============================================================================
# R4f — Factorial attribution: crossed-design variance partitioning (Block 4)
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
  fac_dirs <- .list_run_dirs(FACTORIAL_DIR)
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
            "R4f_dominant_factor_map.png", 9, 8)
    }
  } else {
    message("  [skip] no loadable factorial runs in FACTORIAL_DIR")
  }
} else {
  message("  [skip] FACTORIAL_DIR not set or missing (run SCENARIO_MODE='factorial' first)")
}

# ============================================================================
# R5 — Future-work teaser (dynamic + temporal robustness)
# ============================================================================
# Pulled from the `dynamic` branch results + Ben_robustness/ layers; assembled
# separately as it is framed as future work, not a headline result.
message("\nR5 — future-work teaser: assemble manually from dynamic-branch results.")

message("\nDone. Figures in: ", normalizePath(OUT_DIR))
