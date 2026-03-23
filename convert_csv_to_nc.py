# convert_csv_to_nc.py - Final complete version
import pandas as pd
import xarray as xr
import numpy as np

# Load CSV
df = pd.read_csv("data/weather_data.csv", parse_dates=["time"])

# Remove timezone info
df["time"] = df["time"].dt.tz_localize(None)

# Get unique coordinates
times = df['time'].unique()
latitudes = np.sort(df['lat'].unique())
longitudes = np.sort(df['lon'].unique())

# Add depth coordinate (for ocean variables)
depths = np.array([0.494])

# Add height above ground coordinate (for wind variables)
# Wind is typically at 10m, but include a few levels for interpolation
heights = np.array([10.0])  # 10 meters above ground

print(f"Time steps: {len(times)}")
print(f"Latitude points: {len(latitudes)}")
print(f"Longitude points: {len(longitudes)}")
print(f"Depth levels: {len(depths)}")
print(f"Height levels: {len(heights)}")

# Create wind grids (3D with height dimension)
wind_speed_grid = np.zeros((len(times), len(heights), len(latitudes), len(longitudes)))
wind_dir_grid = np.zeros((len(times), len(heights), len(latitudes), len(longitudes)))

# Fill wind grids (same wind at all heights for simplicity)
for i, time in enumerate(times):
    time_data = df[df['time'] == time]
    for _, row in time_data.iterrows():
        lat_idx = np.where(latitudes == row['lat'])[0][0]
        lon_idx = np.where(longitudes == row['lon'])[0][0]
        # Same wind at all height levels
        for h in range(len(heights)):
            wind_speed_grid[i, h, lat_idx, lon_idx] = row['wind_speed']
            wind_dir_grid[i, h, lat_idx, lon_idx] = row['wind_direction']

# Convert wind to U/V components
wind_dir_rad = np.radians(wind_dir_grid)
u_wind = -wind_speed_grid * np.sin(wind_dir_rad)
v_wind = -wind_speed_grid * np.cos(wind_dir_rad)

# Wave variables (2D - no height/depth)
wave_height = wind_speed_grid[:, 0, :, :] * 0.3  # Take first height level
wave_direction = wind_dir_grid[:, 0, :, :].copy()
wave_period = 0.5 * wind_speed_grid[:, 0, :, :] + 2.0

# Atmospheric pressure (2D)
pressure_grid = np.zeros((len(times), len(latitudes), len(longitudes)))
for i, time in enumerate(times):
    hours = (time - times[0]).total_seconds() / 3600
    for lat_idx, lat in enumerate(latitudes):
        for lon_idx, lon in enumerate(longitudes):
            pressure = 101300.0
            pressure += 500.0 * np.sin(hours * np.pi / 24)
            pressure += 200.0 * np.sin((lat - 10.0) * np.pi) * np.cos((lon - 20.0) * np.pi)
            pressure += np.random.normal(0, 50)
            pressure_grid[i, lat_idx, lon_idx] = pressure

# Surface temperature (2D)
surface_temp_grid = np.zeros((len(times), len(latitudes), len(longitudes)))
for i, time in enumerate(times):
    hours = (time - times[0]).total_seconds() / 3600
    for lat_idx, lat in enumerate(latitudes):
        for lon_idx, lon in enumerate(longitudes):
            temp = 288.0
            temp += 5.0 * np.sin(hours * np.pi / 12)
            temp -= (lat - 10.0) * 2.0
            temp += np.random.normal(0, 0.5)
            surface_temp_grid[i, lat_idx, lon_idx] = temp

# Ocean variables (3D with depth)
current_u_3d = np.zeros((len(times), len(depths), len(latitudes), len(longitudes)))
current_v_3d = np.zeros((len(times), len(depths), len(latitudes), len(longitudes)))
temperature_3d = np.zeros((len(times), len(depths), len(latitudes), len(longitudes)))
salinity_3d = np.zeros((len(times), len(depths), len(latitudes), len(longitudes)))

for i in range(len(times)):
    for d in range(len(depths)):
        current_u_3d[i, d, :, :] = np.random.uniform(-0.3, 0.3, (len(latitudes), len(longitudes)))
        current_v_3d[i, d, :, :] = np.random.uniform(-0.3, 0.3, (len(latitudes), len(longitudes)))
        temperature_3d[i, d, :, :] = 15.0 + np.random.normal(0, 0.5, (len(latitudes), len(longitudes)))
        salinity_3d[i, d, :, :] = 35.0 + np.random.normal(0, 0.2, (len(latitudes), len(longitudes)))

# Add spatial gradients
for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):
        temp_factor = 1.0 - (lat - 10.0) / 0.8 * 0.1
        temperature_3d[:, :, i, j] *= temp_factor
        center_factor = 1.0 - abs(lat - 10.35) / 0.35 * 0.5
        current_u_3d[:, :, i, j] *= center_factor
        current_v_3d[:, :, i, j] *= center_factor

# Create Dataset with all dimensions
ds = xr.Dataset(coords={
    'time': times,
    'depth': depths,
    'height_above_ground': heights,
    'latitude': latitudes,
    'longitude': longitudes
})

# Add wind variables (3D with height)
ds['u-component_of_wind_height_above_ground'] = xr.DataArray(
    u_wind, dims=['time', 'height_above_ground', 'latitude', 'longitude']
)
ds['v-component_of_wind_height_above_ground'] = xr.DataArray(
    v_wind, dims=['time', 'height_above_ground', 'latitude', 'longitude']
)

# Add wave variables (2D)
ds['VHM0'] = xr.DataArray(wave_height, dims=['time', 'latitude', 'longitude'])
ds['VMDR'] = xr.DataArray(wave_direction, dims=['time', 'latitude', 'longitude'])
ds['VTPK'] = xr.DataArray(wave_period, dims=['time', 'latitude', 'longitude'])

# Add atmospheric variables (2D)
ds['Pressure_reduced_to_MSL_msl'] = xr.DataArray(pressure_grid, dims=['time', 'latitude', 'longitude'])
ds['Temperature_surface'] = xr.DataArray(surface_temp_grid, dims=['time', 'latitude', 'longitude'])

# Add ocean variables (3D with depth)
ds['utotal'] = xr.DataArray(current_u_3d, dims=['time', 'depth', 'latitude', 'longitude'])
ds['vtotal'] = xr.DataArray(current_v_3d, dims=['time', 'depth', 'latitude', 'longitude'])
ds['thetao'] = xr.DataArray(temperature_3d, dims=['time', 'depth', 'latitude', 'longitude'])
ds['so'] = xr.DataArray(salinity_3d, dims=['time', 'depth', 'latitude', 'longitude'])

# Add units
ds['u-component_of_wind_height_above_ground'].attrs['units'] = 'm/s'
ds['v-component_of_wind_height_above_ground'].attrs['units'] = 'm/s'
ds['VHM0'].attrs['units'] = 'm'
ds['VMDR'].attrs['units'] = 'degree'
ds['VTPK'].attrs['units'] = 's'
ds['Pressure_reduced_to_MSL_msl'].attrs['units'] = 'Pa'
ds['Temperature_surface'].attrs['units'] = 'K'
ds['utotal'].attrs['units'] = 'm/s'
ds['vtotal'].attrs['units'] = 'm/s'
ds['thetao'].attrs['units'] = 'degrees_C'
ds['so'].attrs['units'] = '1e-3'

# Save
ds.to_netcdf("data/weather_data.nc")

print("\n✅ FINAL - Complete weather data with all dimensions:")
print(f"   Variables: {list(ds.data_vars.keys())}")
print(f"   Dimensions:")
print(f"     - time: {len(times)}")
print(f"     - height_above_ground: {len(heights)}")
print(f"     - depth: {len(depths)}")
print(f"     - latitude: {len(latitudes)}")
print(f"     - longitude: {len(longitudes)}")
print(f"   Time range: {times[0]} to {times[-1]}")