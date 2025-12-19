# =============================================================================
# ECOSYSTEM CONDITION ANOMALIES CALCULATION
# =============================================================================
# This script calculates abiotic, biotic, and landscape anomalies per ecosystem type
# and produces three final raster layers combining all ecosystem types

source("setup.R")

# Load required packages
required_packages <- c("dplyr", "terra", "sf", "readxl", "classInt", "tidyr")
for (pkg in required_packages) {
    library(pkg, character.only = TRUE)
}

# =============================================================================
# CONFIGURATION AND DATA LOADING
# =============================================================================

# Define ecosystem types and ECT categories
ECOSYSTEM_TYPES <- c("forest", "agricultural", "grassland")

ECT_CATEGORIES <- list(
    "abiotic" = c("smd", "sbd", "soc"),
    "biotic" = c("uzl", "tsd", "can", "cdi", "swf_h", "swf_t", "lai", "ndvi"),
    "landscape" = c("snh", "frag", "tcd")
)

# Load data
cat("Loading EC data...\n")
ec_data <- load_ec_data()

# =============================================================================
# ANOMALY CALCULATION FUNCTIONS
# =============================================================================

#' Calculate standardized anomalies for a given ecosystem and variable set
#' @param ecosystem_name Name of ecosystem type
#' @param variable_codes Vector of variable codes to process
#' @return SpatRaster with anomaly layers
calculate_anomalies_by_ecosystem <- function(ecosystem_name, variable_codes) {
    cat(sprintf("Processing %s ecosystem...\n", ecosystem_name))
    
    # Get available rasters for this ecosystem
    available_rasters <- ec_data[variable_codes]
    available_rasters <- available_rasters[!sapply(available_rasters, is.null)]
    
    if (length(available_rasters) == 0) {
        cat(sprintf("Warning: No data available for %s ecosystem\n", ecosystem_name))
        return(NULL)
    }
    
    # Create raster stack and mask by ecosystem
    r_stack <- rast(available_rasters)
    r_masked <- mask_by_ecosystem(r_stack, ecosystem_name)
    
    # Calculate anomalies (standardized values)
    anomaly_stack <- rast()
    
    for (var_code in names(r_masked)) {
        if (var_code %in% names(r_masked)) {
            r_var <- r_masked[[var_code]]
            
            # Calculate global statistics
            var_mean <- global(r_var, "mean", na.rm = TRUE)[[1]]
            var_sd <- global(r_var, "sd", na.rm = TRUE)[[1]]
            
            # Calculate anomaly (z-score) only if valid statistics
            if (!is.na(var_sd) && var_sd > 0) {
                anomaly_layer <- (r_var - var_mean) / var_sd
                names(anomaly_layer) <- paste0(var_code, "_anom")
                anomaly_stack <- c(anomaly_stack, anomaly_layer)
                cat(sprintf("  - %s: mean=%.3f, sd=%.3f\n", var_code, var_mean, var_sd))
            }
        }
    }
    
    return(anomaly_stack)
}

#' Calculate mean anomalies by ECT category for each ecosystem
#' @param anomaly_stack SpatRaster with individual variable anomalies
#' @param ect_category ECT category name ("abiotic", "biotic", "landscape")
#' @return SpatRaster with mean anomaly for the category
calculate_ect_mean_anomaly <- function(anomaly_stack, ect_category) {
    if (is.null(anomaly_stack) || nlyr(anomaly_stack) == 0) {
        return(NULL)
    }
    
    # Get variable codes for this ECT category
    target_variables <- ECT_CATEGORIES[[ect_category]]
    
    # Find layers that match these variables
    layer_names <- names(anomaly_stack)
    matching_layers <- layer_names[sapply(target_variables, function(var) {
        any(grepl(paste0("^", var, "_anom"), layer_names))
    })]
    
    if (length(matching_layers) == 0) {
        return(NULL)
    }
    
    # Calculate mean anomaly across matching layers
    if (length(matching_layers) == 1) {
        mean_anomaly <- anomaly_stack[[matching_layers]]
    } else {
        mean_anomaly <- mean(anomaly_stack[[matching_layers]], na.rm = TRUE)
    }
    
    names(mean_anomaly) <- paste0(ect_category, "_anomaly")
    return(mean_anomaly)
}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

cat("=== CALCULATING ECOSYSTEM CONDITION ANOMALIES ===\n\n")

# Initialize storage for results
ecosystem_anomalies <- list()
ect_results <- list(
    abiotic = rast(),
    biotic = rast(),
    landscape = rast()
)

# Process each ecosystem type
for (ecosystem in ECOSYSTEM_TYPES) {
    cat(sprintf("\n--- Processing %s ecosystem ---\n", toupper(ecosystem)))
    
    # Get variables for this ecosystem
    eco_variables <- ec_categories[["EC variables"]][[ecosystem]]
    if (is.null(eco_variables)) {
        cat(sprintf("Warning: No variables defined for %s ecosystem\n", ecosystem))
        next
    }
    
    variable_codes <- names(eco_variables)
    
    # Calculate anomalies for all variables in this ecosystem
    eco_anomalies <- calculate_anomalies_by_ecosystem(ecosystem, variable_codes)
    
    if (!is.null(eco_anomalies)) {
        ecosystem_anomalies[[ecosystem]] <- eco_anomalies
        
        # Calculate mean anomalies by ECT category for this ecosystem
        for (ect_category in names(ECT_CATEGORIES)) {
            ect_anomaly <- calculate_ect_mean_anomaly(eco_anomalies, ect_category)
            
            if (!is.null(ect_anomaly)) {
                # Add to combined ECT results
                if (nlyr(ect_results[[ect_category]]) == 0) {
                    ect_results[[ect_category]] <- ect_anomaly
                } else {
                    # Mosaic with existing data (sum where they overlap, keep individual values elsewhere)
                    ect_results[[ect_category]] <- mosaic(ect_results[[ect_category]], ect_anomaly, 
                                                         fun = "mean", na.rm = TRUE)
                }
                
                cat(sprintf("  - %s anomaly calculated and added\n", ect_category))
            }
        }
    }
}

# =============================================================================
# FINALIZE AND SAVE RESULTS
# =============================================================================

cat("\n=== FINALIZING RESULTS ===\n")

# Create final output rasters
output_rasters <- list()
output_files <- c(
    "abiotic_condition_anomaly.tif",
    "biotic_condition_anomaly.tif", 
    "landscape_condition_anomaly.tif"
)

for (i in seq_along(names(ECT_CATEGORIES))) {
    ect_category <- names(ECT_CATEGORIES)[i]
    
    if (nlyr(ect_results[[ect_category]]) > 0) {
        output_rasters[[ect_category]] <- ect_results[[ect_category]]
        
        # Save to file
        writeRaster(ect_results[[ect_category]], 
                   output_files[i], 
                   overwrite = TRUE)
    }
}

# =============================================================================
# SUMMARY AND VALIDATION
# =============================================================================

cat("\n=== PROCESSING SUMMARY ===\n")
cat(sprintf("Processed ecosystems: %s\n", paste(names(ecosystem_anomalies), collapse = ", ")))
cat(sprintf("Output rasters created: %d/3\n", sum(sapply(ect_results, function(x) nlyr(x) > 0))))

# Display final file info
for (i in seq_along(output_files)) {
    if (file.exists(output_files[i])) {
        file_size <- file.size(output_files[i]) / 1024^2  # MB
        cat(sprintf("  %s: %.2f MB\n", output_files[i], file_size))
    }
}

cat("\n✓ Ecosystem condition anomaly calculation completed!\n")

# Clean up workspace
rm(ec_data, ecosystem_anomalies)
gc()


