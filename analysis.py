import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import time


start = time.time()


ds = xr.open_dataset("weather.nc", chunks={"time": 10})


ds = ds.sel(
    latitude=slice(10, 20),
    longitude=slice(80, 90)
)


ds = ds.interp(latitude=15, longitude=85)


wave = ds['VHM0'] + np.random.rand(*ds['VHM0'].shape)

print("Wave Min:", wave.min().compute().item())
print("Wave Max:", wave.max().compute().item())

# Statistics
wave_mean = wave.mean().compute().item()
wave_median = wave.compute().median().item()
wave_std = wave.std().compute().item()

print("\n--- Wave Statistics ---")
print("Mean:", wave_mean)
print("Median:", wave_median)
print("Std Dev:", wave_std)



# Daily average
wave_daily = wave.resample(time='1D').mean()

# Plot time series
wave_daily.plot()
plt.title("Daily Average Wave Height (Spatially Averaged)")
plt.savefig("wave_timeseries.png")
plt.clf()

# Histogram
wave.plot.hist(bins=50)
plt.title("Wave Height Distribution")
plt.savefig("wave_histogram.png")
plt.clf()


u = ds['u-component_of_wind_height_above_ground'] + np.random.rand(*ds['u-component_of_wind_height_above_ground'].shape)
v = ds['v-component_of_wind_height_above_ground'] + np.random.rand(*ds['v-component_of_wind_height_above_ground'].shape)

wind_speed = np.sqrt(u**2 + v**2)

# Statistics
wind_mean = wind_speed.mean().compute().item()
wind_median = wind_speed.compute().median().item()
wind_std = wind_speed.std().compute().item()

print("\n--- Wind Statistics ---")
print("Mean:", wind_mean)
print("Median:", wind_median)
print("Std Dev:", wind_std)



# Daily average
wind_daily = wind_speed.resample(time='1D').mean()

# Plot time series
wind_daily.plot()
plt.title("Daily Average Wind Speed")
plt.savefig("wind_timeseries.png")
plt.clf()

# Histogram
wind_speed.plot.hist(bins=50)
plt.title("Wind Speed Distribution")
plt.savefig("wind_histogram.png")
plt.clf()


wave_flat = wave.compute().values.flatten()
wind_flat = wind_speed.compute().values.flatten()

correlation = np.corrcoef(wave_flat, wind_flat)[0, 1]

print("\n--- Correlation ---")
print("Wave vs Wind Correlation:", correlation)


end = time.time()
print("\nExecution Time:", end - start, "seconds")

# Wind direction (in radians)
wind_direction = np.arctan2(v, u)

# Statistics
dir_mean = wind_direction.mean().compute().item()
dir_median = wind_direction.compute().median().item()
dir_std = wind_direction.std().compute().item()

print("\n--- Wind Direction Statistics ---")
print("Mean:", dir_mean)
print("Median:", dir_median)
print("Std Dev:", dir_std)



plt.title("Daily Average Wind Direction")
plt.savefig("wind_direction_timeseries.png")
plt.clf()



wind_direction = np.arctan2(v, u)

# Statistics
print("\n--- Wind Direction Statistics ---")
print("Mean:", wind_direction.mean().compute().item())
print("Median:", wind_direction.compute().median().item())
print("Std Dev:", wind_direction.std().compute().item())

# Daily average (NO spatial avg needed)
dir_daily = wind_direction.resample(time='1D').mean()

# Plot
dir_daily.plot()
plt.title("Daily Average Wind Direction")
plt.savefig("wind_direction_timeseries.png")
plt.clf()