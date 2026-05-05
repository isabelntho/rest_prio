# =============================================================================
# ECOSYSTEM CONDITION ANOMALIES CALCULATION
# =============================================================================
# Calculates abiotic and biotic anomalies across 13 condition scenarios and
# writes all outputs to inputs/anomaly_scenarios/.
#
# Scenario dimensions (separate axes, 13 scenarios total):
#   Benchmark (2):      global | upper_q75
#   Indicator LOO (12): all | drop one abiotic (3) | drop one biotic (8)
#   upper_q75 always uses all indicators.
#   LOO variants always use the global benchmark.

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

OUTPUT_DIR <- "inputs/anomaly_scenarios"

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
    list(tag = "upper_q75_all",     benchmark = "upper_q75", exclude_vars = character(0))
)

# Load data
cat("Loading EC data...\n")
ec_data <- load_ec_data()

# =============================================================================
# ANOMALY CALCULATION FUNCTIONS
# =============================================================================

#' Compute benchmark statistics (mean, sd) for a single masked variable raster.
#' @param r_var SpatRaster — ecosystem-masked variable layer
#' @param method "global" or "upper_q75"
#' @return list(mean, sd)
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
    } else {
        stop(sprintf("Unknown benchmark method: '%s'. Use 'global' or 'upper_q75'.", method))
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
        stats <- get_benchmark_stats(r_var, benchmark)

        if (is.na(stats$sd) || stats$sd <= 0) next

        anomaly_layer <- (r_var - stats$mean) / stats$sd
        cat(sprintf("    %s: ref_mean=%.3f, ref_sd=%.3f\n", var_code, stats$mean, stats$sd))

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

        eco_variables <- ec_categories[["EC variables"]][[ecosystem]]
        if (is.null(eco_variables)) {
            cat(sprintf("Warning: No variables defined for %s ecosystem\n", ecosystem))
            next
        }

        eco_anomalies <- calculate_anomalies_by_ecosystem(
            ecosystem, names(eco_variables),
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
cat(sprintf("Expected: %d (13 scenarios x 2 rasters)\n", 13 * 2))
for (f in written_files) cat(sprintf("  %s\n", f))
cat("\n✓ Ecosystem condition anomaly scenarios completed!\n")