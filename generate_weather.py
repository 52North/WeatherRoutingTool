import numpy as np
import xarray as xr
import pandas as pd

# grid
lats = np.linspace(5, 22, 30)
lons = np.linspace(78, 90, 30)

depth = [0]
height = [10]   # wind measurement height

# time
times = pd.date_range("2024-01-01", periods=25, freq="3h")

shape_surface = (len(times), len(lats), len(lons))
shape_ocean = (len(times), len(depth), len(lats), len(lons))
shape_wind = (len(times), len(height), len(lats), len(lons))

zeros_surface = np.zeros(shape_surface)
zeros_ocean = np.zeros(shape_ocean)
zeros_wind = np.zeros(shape_wind)

pressure = np.ones(shape_surface) * 101325
temperature = np.ones(shape_surface) * 288
wave_period = np.ones(shape_surface) * 5

ds = xr.Dataset(

{

# wind (with height dimension)
"u-component_of_wind_height_above_ground":
(["time","height_above_ground","latitude","longitude"], zeros_wind),

"v-component_of_wind_height_above_ground":
(["time","height_above_ground","latitude","longitude"], zeros_wind),

# waves
"VHM0":
(["time","latitude","longitude"], zeros_surface),

"VMDR":
(["time","latitude","longitude"], zeros_surface),

"VTPK":
(["time","latitude","longitude"], wave_period),

# ocean currents
"utotal":
(["time","depth","latitude","longitude"], zeros_ocean),

"vtotal":
(["time","depth","latitude","longitude"], zeros_ocean),

# ocean temperature
"thetao":
(["time","depth","latitude","longitude"], zeros_ocean),

# salinity
"so":
(["time","depth","latitude","longitude"], zeros_ocean),

# pressure
"Pressure_reduced_to_MSL_msl":
(["time","latitude","longitude"], pressure),

# surface temperature
"Temperature_surface":
(["time","latitude","longitude"], temperature),

},

coords={
"time": times,
"latitude": lats,
"longitude": lons,
"depth": depth,
"height_above_ground": height
}

)

ds.to_netcdf("weather.nc")

print("Complete synthetic weather dataset created successfully")