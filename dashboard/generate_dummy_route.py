import json
import datetime
import math
import random

def generate_dummy_route(filename="dummy_route.json"):
    # Route from Tokyo (35.6, 139.6) to San Francisco (37.7, -122.4)
    # Simplified Great Circle-ish interpolation
    start_lat, start_lon = 35.6, 139.6
    end_lat, end_lon = 37.7, -122.4 + 360 # Normalizing crossing dateline approx
    
    steps = 50
    duration_hours = 240 # 10 days
    
    features = []
    start_time = datetime.datetime.now()
    
    for i in range(steps + 1):
        t = i / steps
        
        # Interpolate Lat/Lon
        lat = start_lat + (end_lat - start_lat) * t
        lon = start_lon + (end_lon - start_lon) * t
        if lon > 180:
            lon -= 360
            
        current_time = start_time + datetime.timedelta(hours=t * duration_hours)
        
        # Simulate Metrics
        # Speed: varies between 18 and 22 knots (approx 9-11 m/s)
        base_speed = 10.0
        speed = base_speed + math.sin(t * 10) * 1.0 + random.uniform(-0.5, 0.5)
        
        # Fuel: correlates with speed^3 roughly + noise
        fuel = (speed / 10.0)**3 * 1.5 + random.uniform(-0.1, 0.1)
        
        # Environmental
        wave_height = 2.0 + math.sin(t * 5) * 1.5 # 0.5 to 3.5m
        wind_speed = 5.0 + math.cos(t * 7) * 4.0 # 1 to 9 m/s
        
        props = {
            "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "speed": {"value": speed, "unit": "m/s"},
            "engine_power": {"value": 15000 * (speed/10)**3, "unit": "kW"},
            "fuel_consumption": {"value": fuel, "unit": "t/h"},
            "wave_height": {"value": wave_height, "unit": "m"},
            "u_wind_speed": {"value": wind_speed, "unit": "m/s"},
            # Add other required fields with dummy values to prevent weird errors if accessed
            "v_wind_speed": {"value": 0, "unit": "m/s"},
            "wave_direction": {"value": 0, "unit": "deg"},
            "wave_period": {"value": 8, "unit": "s"}
        }
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": props
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "route type": "Simulation Test",
        "features": features
    }
    
    with open(filename, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_dummy_route()
