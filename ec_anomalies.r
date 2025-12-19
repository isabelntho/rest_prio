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
    r_stack <- terra::rast(available_rasters)
    r_masked <- mask_by_ecosystem(r_stack, ecosystem_name)
    
    # Calculate anomalies (standardized values)
    anomaly_stack <- NULL
    
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
                
                # Check if anomaly_layer is valid
                if (!inherits(anomaly_layer, "SpatRaster")) {
                    cat(sprintf("Warning: Invalid anomaly layer for %s\n", var_code))
                    next
                }
                
                # Add to stack (initialize with first layer, then combine)
                if (is.null(anomaly_stack)) {
                    anomaly_stack <- anomaly_layer
                } else {
                    anomaly_stack <- c(anomaly_stack, anomaly_layer)
                }
                
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
    if (is.null(anomaly_stack)) {
        return(NULL)
    }
    
    # Handle both single rasters and multi-layer rasters
    n_layers <- ifelse(class(anomaly_stack)[1] == "SpatRaster", 
                      ifelse("nlyr" %in% names(attributes(anomaly_stack)), nlyr(anomaly_stack), 1),
                      0)
    
    if (n_layers == 0) {
        return(NULL)
    }
    
    # Get variable codes for this ECT category
    target_variables <- ECT_CATEGORIES[[ect_category]]
    
    # Find layers that match these variables
    layer_names <- names(anomaly_stack)
    matching_layers <- c()
    
    for (var in target_variables) {
        matching <- layer_names[grepl(paste0("^", var, "_anom"), layer_names)]
        matching_layers <- c(matching_layers, matching)
    }
    
    # Remove any empty matches
    matching_layers <- matching_layers[matching_layers != ""]
    
    # Validate that all matching layers actually exist in the raster stack
    valid_layers <- matching_layers[matching_layers %in% layer_names]
    
    if (length(valid_layers) == 0) {
        cat(sprintf("    No valid matching layers found for %s category\n", ect_category))
        cat(sprintf("    Available layers: %s\n", paste(layer_names, collapse = ", ")))
        cat(sprintf("    Looking for: %s\n", paste(target_variables, collapse = ", ")))
        return(NULL)
    }
    
    cat(sprintf("    Found %d valid layers for %s: %s\n", length(valid_layers), ect_category, paste(valid_layers, collapse = ", ")))
    
    # Calculate mean anomaly across valid matching layers
    if (length(valid_layers) == 1) {
        mean_anomaly <- anomaly_stack[[valid_layers]]
    } else {
        mean_anomaly <- mean(anomaly_stack[[valid_layers]], na.rm = TRUE)
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
    abiotic = NULL,
    biotic = NULL,
    landscape = NULL
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
        cat(sprintf("Successfully created %d anomaly layers for %s\n", 
                   ifelse(class(eco_anomalies)[1] == "SpatRaster", 
                         ifelse("nlyr" %in% names(attributes(eco_anomalies)), nlyr(eco_anomalies), 1), 0), 
                   ecosystem))
        
        # Calculate mean anomalies by ECT category for this ecosystem
        for (ect_category in names(ECT_CATEGORIES)) {
            ect_anomaly <- calculate_ect_mean_anomaly(eco_anomalies, ect_category)
            
            if (!is.null(ect_anomaly)) {
                # Add to combined ECT results
                if (is.null(ect_results[[ect_category]])) {
                    ect_results[[ect_category]] <- ect_anomaly
                    cat(sprintf("  - %s anomaly initialized for %s\n", ect_category, ecosystem))
                } else {
                    # Merge with existing data
                    ect_results[[ect_category]] <- merge(ect_results[[ect_category]], ect_anomaly)
                    cat(sprintf("  - %s anomaly merged for %s\n", ect_category, ecosystem))
                }
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
    
    if (!is.null(ect_results[[ect_category]])) {
        output_rasters[[ect_category]] <- ect_results[[ect_category]]
        
        # Save to file
        writeRaster(ect_results[[ect_category]], 
                   output_files[i], 
                   overwrite = TRUE)
        
        cat(sprintf("Saved %s raster with extent: %s\n", ect_category, 
                   paste(as.vector(ext(ect_results[[ect_category]])), collapse = ", ")))
    } else {
        cat(sprintf("Warning: No data available for %s category\n", ect_category))
    }
}

# =============================================================================
# SUMMARY AND VALIDATION
# =============================================================================

cat("\n=== PROCESSING SUMMARY ===\n")
cat(sprintf("Processed ecosystems: %s\n", paste(names(ecosystem_anomalies), collapse = ", ")))
cat(sprintf("Output rasters created: %d/3\n", sum(sapply(ect_results, function(x) !is.null(x)))))

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