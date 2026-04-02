import xarray as xr
import time
import os

# -------------------------------
# LOAD DATASET WITH CHUNKING
# -------------------------------
print("\n--- Loading Dataset ---")
ds = xr.open_dataset("weather.nc", chunks={"time": 10})

print(ds)

# -------------------------------
# SUBSETTING REGION
# -------------------------------
ds = ds.sel(
    latitude=slice(10, 20),
    longitude=slice(80, 90)
)

print("\n--- After Subsetting ---")
print(ds)

# -------------------------------
# INTERPOLATION
# -------------------------------
ds = ds.interp(latitude=15, longitude=85)

print("\n--- After Interpolation ---")
print(ds)

# -------------------------------
# SAVE AS NETCDF
# -------------------------------
print("\n--- Saving as NetCDF ---")
start = time.time()

ds.to_netcdf("output.nc")

end = time.time()
netcdf_time = end - start

print("NetCDF Save Time:", netcdf_time, "seconds")

# -------------------------------
# SAVE AS ZARR
# -------------------------------
print("\n--- Saving as Zarr ---")
start = time.time()

ds.to_zarr("output.zarr", mode="w")

end = time.time()
zarr_time = end - start

print("Zarr Save Time:", zarr_time, "seconds")

# -------------------------------
# FILE SIZE (NetCDF)
# -------------------------------
if os.path.exists("output.nc"):
    nc_size = os.path.getsize("output.nc") / (1024 * 1024)
    print("\nNetCDF File Size:", round(nc_size, 2), "MB")

# -------------------------------
# VARIABLES INFO
# -------------------------------
print("\n--- Dataset Variables ---")
print(list(ds.data_vars))


print("\n--- Performance Summary ---")
print("NetCDF Time:", netcdf_time, "seconds")
print("Zarr Time:", zarr_time, "seconds")

if zarr_time < netcdf_time:
    print("Zarr is faster for this dataset")
else:
    print("NetCDF is faster for this dataset")

print("\n✔ Data pipeline example completed successfully")