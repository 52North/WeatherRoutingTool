# The WeatherRoutingTool uses the Python package global-land-mask (https://github.com/toddkarin/global-land-mask) to
# check if a point is on land or in the ocean. The package stores the global mask data in a compressed form as npz
# file. This script provides functionality to convert the npz file to a GeoTIFF file for easier visualization.
import numpy as np
import rasterio


if __name__ == "__main__":
    input_filename = "globe_combined_mask_compressed.npz"
    output_filename = 'globe_combined_mask_compressed.tif'

    npz_file = np.load(input_filename)
    mask = npz_file['mask']
    lon = npz_file['lon']
    lat = npz_file['lat']

    rows = len(lat)
    cols = len(lon)
    crs = 'EPSG:4326'
    transform = rasterio.transform.from_origin(-180, 90, 360 / cols, 180 / rows)

    profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': np.uint8,
        'crs': crs,
        'transform': transform,
    }

    with rasterio.open(output_filename, 'w', nbits=1, **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
