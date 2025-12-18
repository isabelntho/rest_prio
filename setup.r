# Set up environment
libs <- c("ggplot2", "dplyr", "terra", "sf", "viridis", "corrplot", "ggpubr", "cowplot", "gridExtra")
for (lib in libs) {
  library(lib, character.only = TRUE)
}

#LU data
#change this to just AS
LU <- rast("W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif")
common_extent <- ext(2556200, 2677700, 1130600, 1243700)
LU <- crop(LU, common_extent)

kb <- st_read("W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp", quiet=TRUE)
kb <- kb %>% filter(NAME == "Bern")

lu_classes <- list(
  forest = c(12, 13),
  agricultural = c(15),
  grassland = c(16, 17)
)

# Data configuration from app.R
ec_categories <- list(
  "EC variables" = list(
    forest = c(
      "forest_index" = "Forest EC Index",
      "smd" = "Soil Moisture Deficit",
      "sbd" = "Soil Bulk Density",
      "soc" = "Soil Organic Carbon", 
      "tsd" = "Tree Species Diversity",
      "can" = "Canopy Height",
      "lai" = "Leaf Area Index",
      "frag" = "Forest Fragmentation"
    ),
    agricultural = c(
      "agricultural_index" = "Agricultural EC Index",
      "smd" = "Soil Moisture Deficit",
      "sbd" = "Soil Bulk Density",
      "soc" = "Soil Organic Carbon",
      "uzl" = "UZL Species",
      "cdi" = "Crop Diversity Index", 
      "swf_h" = "Small Woody Features (Hedgerows)",
      "swf_t" = "Small Woody Features (Trees)",
      "ndvi" = "NDVI",
      "snh" = "Semi-natural Habitat Density"
    ),
    grassland = c(
      "grass_index" = "Grassland & Pastures EC Index",
      "smd" = "Soil Moisture Deficit",
      "sbd" = "Soil Bulk Density",
      "soc" = "Soil Organic Carbon",
      "uzl" = "UZL Species",
      "swf_h" = "Small Woody Features (Hedgerows)",
      "swf_t" = "Small Woody Features (Trees)", 
      "ndvi" = "NDVI",
      "snh" = "Semi-natural Habitat Density"
    )
  )
)

# Load EC data function
load_ec_data <- function() {
  ec_vars <- list(
    forest_index = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/EC_index_forest.tif",
    agricultural_index = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/EC_index_agricultural.tif",
    grass_index = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/EC_index_grassland.tif",
    smd = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/smd.tif",
    sbd = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/sbd.tif",
    soc = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/soc.tif", 
    uzl = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/uzl.tif",
    swf_h = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/swf_h.tif",
    swf_t = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/swf_t.tif",
    ndvi = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/ndvi.tif",
    cdi = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/cdi.tif",
    snh = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/snh.tif",
    md = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/md.tif",
    tsd = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/tsd.tif",
    can = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/can_lang.tif",
    lai = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/lai.tif", 
    frag = "C:/Users/inicholson/Documents/Restoration_potential/shinyliveapp/ec_data_aligned/frag.tif"
  )
  
  ec_rasters <- lapply(names(ec_vars), function(var_name) {
    file_path <- ec_vars[[var_name]]
    if(file.exists(file_path)) {
      tryCatch({
        raster_data <- rast(file_path)
        if (var_name == "tsd") {
          NAflag(raster_data) <- -128
          values(raster_data)[values(raster_data) == -128] <- NA
        }
        message(paste("Successfully loaded:", var_name))
        return(raster_data)
      }, error = function(e) {
        warning(paste("Error loading", var_name, ":", e$message))
        return(NULL)
      })
    } else {
      warning(paste("File not found:", file_path))
      return(NULL)
    }
  })
  names(ec_rasters) <- names(ec_vars)
  
  loaded_data <- ec_rasters[!sapply(ec_rasters, is.null)]
  message(paste("Loaded", length(loaded_data), "EC datasets out of", length(ec_vars), "total"))
  
  return(loaded_data)
}

mask_by_ecosystem <- function(r, ecosystem) {
  if (ecosystem == "no_mask") return(r)
  if (!(ecosystem %in% names(lu_classes))) return(r)
  
  allowed <- lu_classes[[ecosystem]]
  lu_mask <- LU

  vals <- values(lu_mask)
  vals[!(vals %in% allowed)] <- NA
  vals[vals %in% allowed] <- 1
  values(lu_mask) <- vals
  
  r <- mask(r, lu_mask)
  crop(r, kb)
  mask(r, kb)
}
normalise_raster_zscore <- function(r) {
  stats <- global(r, c("mean", "sd"), na.rm = TRUE)
  m <- stats[1, 1]
  s <- stats[1, 2]
  
  if (is.na(s) || s == 0) {
    return(app(r, fun = function(x) { 0 }))
  }
  
  app(r, fun = function(x) { (x - m) / s })
}
