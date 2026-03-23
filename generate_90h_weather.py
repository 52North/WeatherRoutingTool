# save as generate_90h_weather.py
import pandas as pd
import numpy as np

start_time = pd.Timestamp('2026-03-04 00:00:00')
end_time = pd.Timestamp('2026-03-07 18:00:00')  # 90 hours
time_range = pd.date_range(start=start_time, end=end_time, freq='3H')

latitudes = np.arange(10.0, 10.9, 0.1)
longitudes = np.arange(20.0, 20.9, 0.1)

data = []
for time in time_range:
    for lat in latitudes:
        for lon in longitudes:
            hours = (time - start_time).total_seconds() / 3600
            wind_speed = 5.0 + 2.0 * np.sin(hours * np.pi / 12)
            wind_direction = (180.0 + hours * 0.5) % 360
            
            data.append({
                'time': time.strftime('%Y-%m-%dT%H:%MZ'),
                'lat': round(lat, 1),
                'lon': round(lon, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_direction': round(wind_direction, 1)
            })

pd.DataFrame(data).to_csv('data/weather_data.csv', index=False)
print(f"Created {len(time_range)} time steps")