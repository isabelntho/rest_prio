
import rasterio as rio
#open the raster file and extract its information      

with rio.open("W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif") as src:
    info = {
        "crs": src.crs,                       # rasterio.crs.CRS or None
        "transform": src.transform,           # affine transform
        "res": src.res,                       # (xres, yres)
        "bounds": src.bounds,
        "shape": src.shape               # left, bottom, right, top
    }

# Print in a readable way
for k, v in info.items():
    print(f"{k}: {v}")