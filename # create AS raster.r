# create AS raster
df <- read.csv("Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/arealstatistik-zeitreihe_2056.csv",
                 sep=";", header=TRUE)

head(df)
names(df)

df <- df %>% select(E_COORD,N_COORD,AS18_72)

as <- terra::rast(df, type="xyz", crs="EPSG:2056")



ref_rast <- terra::rast("Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif")

#extend as to match ref_rast
as_ext <- terra::extend(as, ref_rast)
#resample as to match ref_rast
as_res <- terra::resample(as_ext, ref_rast, method="near")


#write AS raster
terra::writeRaster(as_res, "Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/AS72_2018.tif",
                   overwrite=TRUE)