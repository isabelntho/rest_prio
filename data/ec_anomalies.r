# =============================================================================
# ECOSYSTEM CONDITION ANOMALIES CALCULATION
# =============================================================================
# Calculates abiotic and biotic anomalies across 13 condition scenarios and
# writes all outputs to data/anomaly_scenarios/.
#
# Scenario dimensions:
#   Benchmark:          global | upper_q75 | zones
#   Indicator LOO:      all | drop one abiotic (3) | drop one biotic (8)
#   The global benchmark is crossed with every LOO construction; the upper_q75
#   benchmark is crossed with the agricultural-relevant LOO constructions
#   (smd/sbd/soc/uzl/cdi/swf_h/swf_t/ndvi) for the Block 3 factorial. zones uses
#   all indicators only.

source("setup.R")

# Load required packages
required_packages <- c("dplyr", "terra", "sf", "readxl", "classInt", "tidyr")
for (pkg in required_packages) {
    library(pkg, character.only = TRUE)
}

# =============================================================================
# CONFIGURATION
# =============================================================================

ECOSYSTEM_TYPES <- c("forest", "agricultural", "grassland")

ECT_CATEGORIES <- list(
    "abiotic" = c("smd", "sbd", "soc"),
    "biotic"  = c("uzl", "tsd", "can", "cdi", "swf_h", "swf_t", "lai", "ndvi"),
    "landscape" = c("snh", "frag", "tcd")
)

OUTPUT_DIR <- "data/anomaly_scenarios"

# Indicator directionality:
#   positive  — higher raw value = better condition (default; z-score as-is)
#   negative  — lower raw value = better condition (smd, sbd); raster is negated
#               before computing the z-score so that degraded pixels still get
#               negative anomalies
#   unimodal  — optimal value lies at `optimum`; both extremes are degraded.
#               Raster is transformed to -|x - optimum| before z-scoring, so
#               pixels near the optimum receive positive anomalies.
INDICATOR_DIRECTIONALITY <- list(
    smd   = list(type = "positive"),
    sbd   = list(type = "negative"),
    soc   = list(type = "positive"),
    uzl   = list(type = "positive"),
    tsd   = list(type = "unimodal", optimum = 50),
    can   = list(type = "positive"),
    cdi   = list(type = "positive"),
    swf_h = list(type = "positive"),
    swf_t = list(type = "positive"),
    lai   = list(type = "positive"),
    ndvi  = list(type = "positive")
)

# 13 condition scenarios (separate axes)
CONDITION_SCENARIOS <- list(
    list(tag = "global_all",        benchmark = "global",    exclude_vars = character(0)),
    list(tag = "global_drop_smd",   benchmark = "global",    exclude_vars = "smd"),
    list(tag = "global_drop_sbd",   benchmark = "global",    exclude_vars = "sbd"),
    list(tag = "global_drop_soc",   benchmark = "global",    exclude_vars = "soc"),
    list(tag = "global_drop_uzl",   benchmark = "global",    exclude_vars = "uzl"),
    list(tag = "global_drop_tsd",   benchmark = "global",    exclude_vars = "tsd"),
    list(tag = "global_drop_can",   benchmark = "global",    exclude_vars = "can"),
    list(tag = "global_drop_cdi",   benchmark = "global",    exclude_vars = "cdi"),
    list(tag = "global_drop_swf_h", benchmark = "global",    exclude_vars = "swf_h"),
    list(tag = "global_drop_swf_t", benchmark = "global",    exclude_vars = "swf_t"),
    list(tag = "global_drop_lai",   benchmark = "global",    exclude_vars = "lai"),
    list(tag = "global_drop_ndvi",  benchmark = "global",    exclude_vars = "ndvi"),
    list(tag = "upper_q75_all",     benchmark = "upper_q75", exclude_vars = character(0)),
    list(tag = "zones_all",         benchmark = "zones",     exclude_vars = character(0)),

    # ── q75 × indicator-LOO (agricultural-relevant) — for the Block 3 factorial ──
    # The scaling × construction factorial needs each LOO construction at BOTH the
    # global (anomaly) and upper_q75 references. The global LOO rasters already
    # exist above; these add the matching upper_q75 versions. LOO is restricted to
    # the agricultural EC indicators (setup.r): smd/sbd/soc + uzl/cdi/swf_h/swf_t/ndvi.
    list(tag = "upper_q75_drop_smd",   benchmark = "upper_q75", exclude_vars = "smd"),
    list(tag = "upper_q75_drop_sbd",   benchmark = "upper_q75", exclude_vars = "sbd"),
    list(tag = "upper_q75_drop_soc",   benchmark = "upper_q75", exclude_vars = "soc"),
    list(tag = "upper_q75_drop_uzl",   benchmark = "upper_q75", exclude_vars = "uzl"),
    list(tag = "upper_q75_drop_cdi",   benchmark = "upper_q75", exclude_vars = "cdi"),
    list(tag = "upper_q75_drop_swf_h", benchmark = "upper_q75", exclude_vars = "swf_h"),
    list(tag = "upper_q75_drop_swf_t", benchmark = "upper_q75", exclude_vars = "swf_t"),
    list(tag = "upper_q75_drop_ndvi",  benchmark = "upper_q75", exclude_vars = "ndvi")
)

# Load data
cat("Loading EC data...\n")
ec_data <- load_ec_data()

DEM_PATH <- "Z:/people/inicholson/WP2/NCP_models/Data/DEM_mean_LV95.tif"
PROD_REGIONS_PATH <- "Z:/people/inicholson/WP2/NCP_models/Data/PRODUCTION_REGIONS/PRODREG.shp"

# =============================================================================
# ANOMALY CALCULATION FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# Zone raster (production region x altitude class) — built once and cached.
# ---------------------------------------------------------------------------
.zone_raster_cache <- NULL

#' Build a zone raster combining production regions and altitude classes.
#'
#' Zones are the Cartesian product of production-region labels (from
#' PROD_REGIONS_PATH) and three altitude bands derived from DEM_PATH:
#'   class 1 :    0 – 600 m
#'   class 2 :  600 – 1200 m
#'   class 3 : 1200 m +
#'
#' The first attribute column of PROD_REGIONS_PATH is used as the region label.
#'
#' @param template_rast SpatRaster — defines target CRS, extent, and resolution.
#' @return SpatRaster with integer zone IDs.
build_zone_raster <- function(template_rast) {
    if (!is.null(.zone_raster_cache) &&
            compareGeom(.zone_raster_cache, template_rast, stopOnError = FALSE)) {
        return(.zone_raster_cache)
    }

    cat("    Building zone raster (production regions x altitude classes)...\n")

    # --- Altitude classes (3 bands) ---
    dem      <- rast(DEM_PATH)
    dem_proj <- project(dem, template_rast, method = "bilinear")
    alt_class <- classify(dem_proj, rbind(
        c(-Inf,  600, 1),
        c( 600, 1200, 2),
        c(1200,  Inf, 3)
    ), include.lowest = TRUE)
    names(alt_class) <- "alt_class"

    # --- Production region IDs ---
    prod_regions <- st_read(PROD_REGIONS_PATH, quiet = TRUE)
    prod_regions <- st_transform(prod_regions, crs(template_rast))
    # Use the third attribute column as the region label
    label_col <- setdiff(names(prod_regions), attr(prod_regions, "sf_column"))[3]
    cat(sprintf("    Production region label column: '%s'\n", label_col))
    prod_regions$zone_prod_id <- as.integer(as.factor(prod_regions[[label_col]]))
    prod_rast <- rasterize(vect(prod_regions), template_rast,
                           field = "zone_prod_id", background = NA)
    names(prod_rast) <- "prod_id"

    # --- Combine: sequential IDs from 1 to (n_prod * 3) ---
    zone_rast        <- (prod_rast - 1L) * 3L + alt_class
    names(zone_rast) <- "zone_id"

    n_zones <- length(na.omit(unique(values(zone_rast))))
    cat(sprintf("    Zone raster built: %d unique zones\n", n_zones))

    .zone_raster_cache <<- zone_rast
    return(zone_rast)
}

#' Compute benchmark statistics (mean, sd) for a single masked variable raster.
#' @param r_var SpatRaster — ecosystem-masked variable layer
#' @param method "global", "upper_q75", or "zones"
#' @return list(mean, sd) — scalars for global/upper_q75; SpatRasters for zones
get_benchmark_stats <- function(r_var, method) {
    if (method == "global") {
        list(
            mean = global(r_var, "mean", na.rm = TRUE)[[1]],
            sd   = global(r_var, "sd",   na.rm = TRUE)[[1]]
        )
    } else if (method == "upper_q75") {
        q75   <- global(r_var, function(x) quantile(x, 0.75, na.rm = TRUE))[[1]]
        r_ref <- r_var
        r_ref[r_ref < q75] <- NA
        list(
            mean = global(r_ref, "mean", na.rm = TRUE)[[1]],
            sd   = global(r_ref, "sd",   na.rm = TRUE)[[1]]
        )
    } else if (method == "zones") {
        zone_rast <- build_zone_raster(r_var)
        if (!compareGeom(r_var, zone_rast, stopOnError = FALSE)) {
            zone_rast <- resample(zone_rast, r_var, method = "near")
        }
        # Per-zone mean and sd (columns: [zone_id, stat_value])
        zone_mean_df <- zonal(r_var, zone_rast, fun = "mean", na.rm = TRUE)
        zone_sd_df   <- zonal(r_var, zone_rast, fun = "sd",   na.rm = TRUE)
        # Map per-zone stats back to pixels via exact-value reclassification
        mean_rast <- classify(zone_rast, as.matrix(zone_mean_df[, c(1, 2)]))
        sd_rast   <- classify(zone_rast, as.matrix(zone_sd_df[,   c(1, 2)]))
        list(mean = mean_rast, sd = sd_rast)
    } else {
        stop(sprintf("Unknown benchmark method: '%s'. Use 'global', 'upper_q75', or 'zones'.", method))
    }
}

#' Calculate standardized anomalies for a given ecosystem and variable set.
#' @param ecosystem_name Name of ecosystem type
#' @param variable_codes Vector of variable codes to process
#' @param benchmark "global" or "upper_q75"
#' @return SpatRaster with anomaly layers, or NULL
calculate_anomalies_by_ecosystem <- function(ecosystem_name, variable_codes,
                                             benchmark = "global") {
    cat(sprintf("  Processing %s ecosystem...\n", ecosystem_name))

    available_rasters <- ec_data[variable_codes]
    available_rasters <- available_rasters[!sapply(available_rasters, is.null)]

    if (length(available_rasters) == 0) {
        cat(sprintf("  Warning: No data available for %s ecosystem\n", ecosystem_name))
        return(NULL)
    }

    r_stack  <- terra::rast(available_rasters)
    r_masked <- mask_by_ecosystem(r_stack, ecosystem_name)

    anomaly_stack <- NULL

    for (var_code in names(r_masked)) {
        r_var <- r_masked[[var_code]]

        # Apply directionality transformation before computing benchmark stats
        dir_cfg <- INDICATOR_DIRECTIONALITY[[var_code]]
        dir_type <- if (!is.null(dir_cfg)) dir_cfg$type else "positive"
        if (dir_type == "negative") {
            r_var <- -r_var
        } else if (dir_type == "unimodal") {
            r_var <- -abs(r_var - dir_cfg$optimum)
        }

        stats <- get_benchmark_stats(r_var, benchmark)

        # For scalar benchmarks (global, upper_q75) guard against degenerate sd.
        # For the zones benchmark stats$sd is a SpatRaster; pixels with sd == 0
        # or NA will produce NA anomalies naturally — no scalar guard needed.
        if (!inherits(stats$sd, "SpatRaster")) {
            if (is.na(stats$sd) || stats$sd <= 0) next
            cat(sprintf("    %s [%s]: ref_mean=%.3f, ref_sd=%.3f\n",
                        var_code, dir_type, stats$mean, stats$sd))
        } else {
            cat(sprintf("    %s [%s]: zone-wise benchmark (SpatRaster)\n", var_code, dir_type))
        }

        anomaly_layer <- (r_var - stats$mean) / stats$sd

        if (!inherits(anomaly_layer, "SpatRaster")) {
            cat(sprintf("  Warning: Invalid anomaly layer for %s\n", var_code))
            next
        }

        names(anomaly_layer) <- paste0(var_code, "_anom")
        anomaly_stack <- if (is.null(anomaly_stack)) anomaly_layer else c(anomaly_stack, anomaly_layer)
    }

    return(anomaly_stack)
}

#' Calculate mean anomaly for one ECT category, with optional indicator exclusion.
#' @param anomaly_stack SpatRaster with individual variable anomalies
#' @param ect_category "abiotic", "biotic", or "landscape"
#' @param exclude_vars Character vector of variable codes to drop from the mean
#' @return SpatRaster with mean anomaly, or NULL
calculate_ect_mean_anomaly <- function(anomaly_stack, ect_category,
                                       exclude_vars = character(0)) {
    if (is.null(anomaly_stack) || nlyr(anomaly_stack) == 0) return(NULL)

    target_variables <- setdiff(ECT_CATEGORIES[[ect_category]], exclude_vars)
    if (length(target_variables) == 0) {
        cat(sprintf("    All variables excluded for %s -- skipping\n", ect_category))
        return(NULL)
    }

    layer_names  <- names(anomaly_stack)
    valid_layers <- unlist(lapply(target_variables, function(var) {
        layer_names[grepl(paste0("^", var, "_anom"), layer_names)]
    }))
    valid_layers <- valid_layers[valid_layers %in% layer_names]

    if (length(valid_layers) == 0) {
        cat(sprintf("    No valid layers for %s (looking for: %s)\n",
                    ect_category, paste(target_variables, collapse = ", ")))
        return(NULL)
    }

    cat(sprintf("    %s: averaging %d layer(s): %s\n",
                ect_category, length(valid_layers), paste(valid_layers, collapse = ", ")))

    mean_anomaly <- if (length(valid_layers) == 1) {
        anomaly_stack[[valid_layers]]
    } else {
        mean(anomaly_stack[[valid_layers]], na.rm = TRUE)
    }

    names(mean_anomaly) <- paste0(ect_category, "_anomaly")
    return(mean_anomaly)
}

# =============================================================================
# SCENARIO PROCESSING
# =============================================================================

cat("=== CALCULATING ECOSYSTEM CONDITION ANOMALY SCENARIOS ===\n\n")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

for (scenario in CONDITION_SCENARIOS) {
    tag          <- scenario$tag
    benchmark    <- scenario$benchmark
    exclude_vars <- scenario$exclude_vars

    cat(sprintf("\n=== Scenario: %s (benchmark=%s, exclude=%s) ===\n",
                tag, benchmark,
                if (length(exclude_vars) == 0) "none" else paste(exclude_vars, collapse = ", ")))

    ect_results <- list(abiotic = NULL, biotic = NULL)

    for (ecosystem in ECOSYSTEM_TYPES) {
        cat(sprintf("\n--- %s ---\n", toupper(ecosystem)))

        # Use per-ecosystem indicator set from setup.R, restricted to abiotic/biotic
        eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
        eco_var_codes <- intersect(eco_all_codes, unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))

        eco_anomalies <- calculate_anomalies_by_ecosystem(
            ecosystem, eco_var_codes,
            benchmark = benchmark
        )
        if (is.null(eco_anomalies)) next

        for (ect_category in c("abiotic", "biotic")) {
            ect_anomaly <- calculate_ect_mean_anomaly(
                eco_anomalies, ect_category,
                exclude_vars = exclude_vars
            )
            if (!is.null(ect_anomaly)) {
                ect_results[[ect_category]] <- if (is.null(ect_results[[ect_category]])) {
                    ect_anomaly
                } else {
                    terra::cover(ect_results[[ect_category]], ect_anomaly)
                }
            }
        }
    }

    # Save abiotic and biotic rasters for this scenario
    for (ect_category in c("abiotic", "biotic")) {
        if (!is.null(ect_results[[ect_category]])) {
            out_path <- file.path(OUTPUT_DIR, sprintf("%s_%s.tif", ect_category, tag))
            writeRaster(ect_results[[ect_category]], out_path, overwrite = TRUE)
            cat(sprintf("  Saved %s (%.2f MB)\n", out_path, file.size(out_path) / 1024^2))
        } else {
            cat(sprintf("  Warning: No %s result for scenario %s\n", ect_category, tag))
        }
    }
}

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=== SCENARIO PROCESSING COMPLETE ===\n")
written_files <- list.files(OUTPUT_DIR, pattern = "\\.tif$", full.names = FALSE)
cat(sprintf("Files written to %s: %d\n", OUTPUT_DIR, length(written_files)))
cat(sprintf("Expected: %d (%d scenarios x 2 rasters)\n",
            length(CONDITION_SCENARIOS) * 2, length(CONDITION_SCENARIOS)))
for (f in written_files) cat(sprintf("  %s\n", f))
cat("\n✓ Ecosystem condition anomaly scenarios completed!\n")

# =============================================================================
# BASELINE INDICATOR CONTRIBUTION STATISTICS
# =============================================================================
# For the global_all baseline scenario, quantifies each indicator's contribution
# to the abiotic and biotic composites across all eligible pixels (pooled across
# all ecosystem types).
#
# Requires: ec_data and mask_by_ecosystem() from setup.R (loaded at top of script).
#
# Statistics:
#   mean_abs_contribution   — mean |z-score| across pixels; how strongly the
#                             indicator pulls the composite on average
#   coefficient of variation — indicator's share of total within-category variance;
#                             how spatially heterogeneous this indicator is relative
#                             to its peers
#   correlation_with_composite — Pearson r vs. the ECT composite; high = redundant
#                             (dropping it won't shift the composite much), low =
#                             unique signal (dropping it will cause observable change)
#   r_squared               — cor^2; variance of composite explained by indicator
#   dominance_frequency_pct — % of pixels where this indicator has the largest |z|
#                             in its ECT category; shows who drives extreme values

#' Compute per-indicator contribution statistics for one ECT category.
#' @param indicator_vals named list of numeric vectors (pooled pixel z-scores)
#' @param composite_vals named list of numeric vectors (pooled pixel z-scores)
#' @return data.frame with one row per indicator
compute_indicator_contribution_stats <- function(indicator_vals, composite_vals) {
    stats_rows <- lapply(names(indicator_vals), function(ind) {
        x    <- indicator_vals[[ind]]
        c_all <- composite_vals[[ind]]
        keep <- !is.na(x) & !is.na(c_all)
        x_k  <- x[keep]
        c_k  <- c_all[keep]

        mac  <- mean(abs(x_k))
        cv   <- if (mac > 0) sd(abs(x_k), na.rm = TRUE) / mac else NA_real_
        r    <- if (length(x_k) > 1 && sd(x_k) > 0 && sd(c_k) > 0) cor(x_k, c_k) else NA_real_
        r2   <- if (!is.na(r)) r^2 else NA_real_

        data.frame(
            indicator              = ind,
            n_pixels               = length(x_k),
            mean_abs_contribution  = mac,
            cv    = cv,
            correlation_with_composite = r,
            r_squared              = r2,
            stringsAsFactors       = FALSE
        )
    })
    result <- do.call(rbind, stats_rows)
    result
}

cat("\n=== BASELINE INDICATOR CONTRIBUTION STATISTICS (global_all) ===\n\n")

DIAG_DIR <- "diagnostics"
dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)

# Accumulators: per-ECT-category, per-indicator — pixel value vectors
baseline_indicator_vals   <- list(abiotic = list(), biotic = list())
baseline_composite_vals   <- list(abiotic = list(), biotic = list())
baseline_dominance_counts <- list(abiotic = integer(0), biotic = integer(0))

for (ecosystem in ECOSYSTEM_TYPES) {
    cat(sprintf("  Processing %s ecosystem...\n", ecosystem))

    # Use per-ecosystem indicator set from setup.R, restricted to abiotic/biotic
    eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
    eco_var_codes <- intersect(eco_all_codes, unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))

    eco_anomalies <- calculate_anomalies_by_ecosystem(
        ecosystem, eco_var_codes, benchmark = "global"
    )
    if (is.null(eco_anomalies)) next

    for (ect_category in c("abiotic", "biotic")) {
        ind_codes <- ECT_CATEGORIES[[ect_category]]

        all_layer_names <- names(eco_anomalies)
        cat_layers <- unlist(lapply(ind_codes, function(v) {
            all_layer_names[grepl(paste0("^", v, "_anom"), all_layer_names)]
        }))
        cat_layers <- cat_layers[cat_layers %in% all_layer_names]
        if (length(cat_layers) == 0) next

        ind_names <- sub("_anom$", "", cat_layers)

        composite_path <- file.path(OUTPUT_DIR, sprintf("%s_global_all.tif", ect_category))
        if (!file.exists(composite_path)) {
            cat(sprintf("    Composite raster not found: %s\n", composite_path))
            next
        }
        r_composite <- rast(composite_path)
        ind_stack   <- eco_anomalies[[cat_layers]]

        r_comp_matched <- if (!compareGeom(ind_stack, r_composite, stopOnError = FALSE)) {
            resample(r_composite, ind_stack, method = "near")
        } else {
            r_composite
        }

        eco_mask_vals <- as.vector(values(ind_stack[[1]]))
        eco_pixels    <- which(!is.na(eco_mask_vals))
        comp_vals_eco <- as.vector(values(r_comp_matched))[eco_pixels]

        # Accumulate indicator pixel vectors (composite paired per-indicator)
        for (i in seq_along(ind_names)) {
            ind  <- ind_names[i]
            vals <- as.vector(values(ind_stack[[i]]))[eco_pixels]
            baseline_indicator_vals[[ect_category]][[ind]] <-
                c(baseline_indicator_vals[[ect_category]][[ind]], vals)
            baseline_composite_vals[[ect_category]][[ind]] <-
                c(baseline_composite_vals[[ect_category]][[ind]], comp_vals_eco)
        }

        # Dominance: pixel-wise which indicator has largest |z|
        if (length(cat_layers) > 1) {
            abs_stack   <- abs(ind_stack)
            dom_indices <- as.vector(values(app(abs_stack, which.max)))[eco_pixels]
            dom_indices <- dom_indices[!is.na(dom_indices)]
            dom_counts  <- tabulate(dom_indices, nbins = length(ind_names))
            names(dom_counts) <- ind_names
            for (ind in ind_names) {
                prev <- baseline_dominance_counts[[ect_category]][ind]
                baseline_dominance_counts[[ect_category]][ind] <-
                    if (is.null(prev) || is.na(prev)) dom_counts[ind]
                    else prev + dom_counts[ind]
            }
        } else {
            ind <- ind_names[1]
            prev <- baseline_dominance_counts[[ect_category]][ind]
            baseline_dominance_counts[[ect_category]][ind] <-
                if (is.null(prev) || is.na(prev)) length(eco_pixels)
                else prev + length(eco_pixels)
        }
    }
}

# Compute stats and assemble output table
all_stats <- lapply(c("abiotic", "biotic"), function(ect_category) {
    ind_vals   <- baseline_indicator_vals[[ect_category]]
    comp_vals  <- baseline_composite_vals[[ect_category]]
    dom_counts <- baseline_dominance_counts[[ect_category]]

    if (length(ind_vals) == 0) {
        cat(sprintf("  No data accumulated for %s\n", ect_category))
        return(NULL)
    }
    cat(sprintf("\nComputing stats for %s (%d indicators, %d pooled pixels)...\n",
                ect_category, length(ind_vals), length(comp_vals)))

    df        <- compute_indicator_contribution_stats(ind_vals, comp_vals)
    total_dom <- sum(dom_counts, na.rm = TRUE)
    df$dominance_frequency_pct <- if (total_dom > 0) {
        round(100 * dom_counts[df$indicator] / total_dom, 2)
    } else {
        rep(NA_real_, nrow(df))
    }
    df$ect_category <- ect_category
    df[, c("ect_category", "indicator", "n_pixels",
           "mean_abs_contribution", "cv",
           "correlation_with_composite", "r_squared",
           "dominance_frequency_pct")]
})

contribution_stats <- do.call(rbind, all_stats)

# Save
out_csv <- file.path(DIAG_DIR, "indicator_contribution_baseline.csv")
write.csv(contribution_stats, out_csv, row.names = FALSE)
cat(sprintf("\n  Saved: %s\n", out_csv))

# Print summary
cat("\n--- INDICATOR CONTRIBUTION SUMMARY (baseline: global_all) ---\n")
print(contribution_stats, digits = 3, row.names = FALSE)
cat("\n✓ Indicator contribution statistics complete!\n")
