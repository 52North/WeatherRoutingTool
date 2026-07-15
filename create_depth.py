import numpy as np
import xarray as xr

# grid matching weather file
latitude = np.linspace(5, 22, 30)
longitude = np.linspace(78, 90, 30)

# depth grid (lat, lon)
depth_data = np.full((len(latitude), len(longitude)), 100.0)

ds = xr.Dataset(
    {
        "depth": (("latitude", "longitude"), depth_data)
    },
    coords={
        "latitude": latitude,
        "longitude": longitude
    }
)

ds.depth.attrs["units"] = "m"

ds.to_netcdf("depth.nc")

print("Depth dataset created successfully")
print(ds)