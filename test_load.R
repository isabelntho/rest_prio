library(jsonlite)

ecosystem_types <- c("forest", "agricultural", "grassland")

find_ecosystem_reports <- function(ecosystem_types) {
  ecosystem_reports <- list()
  
  for (ecosystem in ecosystem_types) {
    # Look for optimization reports matching this ecosystem
    pattern <- paste0("optimization_report_", ecosystem, ".*\\.json$")
    files <- list.files(".", pattern = pattern, full.names = TRUE)
    
    cat("Ecosystem:", ecosystem, "- Pattern:", pattern, "\n")
    cat("Files found:", length(files), "\n")
    
    if (length(files) > 0) {
      # Get the most recent file for this ecosystem
      latest_file <- files[which.max(file.info(files)$mtime)]
      cat("Latest file:", latest_file, "\n")
      
      tryCatch({
        data <- fromJSON(latest_file, flatten = FALSE)
        ecosystem_reports[[ecosystem]] <- data
        ecosystem_reports[[ecosystem]]$source_file <- latest_file
        cat("Successfully loaded", ecosystem, "report\n")
      }, error = function(e) {
        cat("✗ Error loading", ecosystem, "report:", e$message, "\n")
      })
    } else {
      cat("✗ No", ecosystem, "optimization report found\n")
    }
    cat("\n")
  }
  
  return(ecosystem_reports)
}

# Test the function
ecosystem_reports <- find_ecosystem_reports(ecosystem_types)
cat("Final result: Found", length(ecosystem_reports), "ecosystem reports\n")
cat("Ecosystem names:", names(ecosystem_reports), "\n")