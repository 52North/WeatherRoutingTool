import json
import pandas as pd
import geopandas as gpd
from datetime import datetime

def load_route_data(file_content):
    """
    Parses the WRT GeoJSON output into a pandas DataFrame and a GeoDataFrame.
    """
    try:
        data = json.load(file_content)
        
        # Extract features
        features = data.get('features', [])
        if not features:
            raise ValueError("No features found in GeoJSON")

        parsed_data = []
        for feature in features:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            coords = geom.get('coordinates', [])
            
            # Basic props
            row = {
                'longitude': coords[0] if len(coords) > 1 else None,
                'latitude': coords[1] if len(coords) > 1 else None,
                'time': pd.to_datetime(props.get('time')),
            }
            
            # Extract nested value/unit pairs
            # key: {'value': 123, 'unit': '...'}
            complex_keys = [
                'speed', 'engine_power', 'fuel_consumption', 'propeller_revolution',
                'calm_resistance', 'wind_resistance', 'wave_resistance', 
                'shallow_water_resistance', 'hull_roughness_resistance',
                'wave_height', 'wave_direction', 'wave_period',
                'u_currents', 'v_currents', 'u_wind_speed', 'v_wind_speed',
                'pressure', 'air_temperature', 'salinity', 'water_temperature'
            ]
            
            for key in complex_keys:
                item = props.get(key)
                if isinstance(item, dict):
                    row[key] = item.get('value')
                    # We could store units separately if needed, but for plotting we just need values
                    # We might want to capture units for labels later, but usually they are constant
            
            parsed_data.append(row)

        df = pd.DataFrame(parsed_data)
        
        # Filter out "dummy" last points or initialization artifacts if any (-99 values)
        # The WRT puts -99 for the last step's power/fuel usually, let's keep them but be aware for plotting
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        
        return df, gdf, data

    except Exception as e:
        raise ValueError(f"Failed to parse route file: {e}")

def get_summary_metrics(df):
    """
    Calculates summary metrics from the route dataframe.
    """
    # Filter out invalid steps for summation (where fuel is -99 or NaN)
    valid_fuel = df[df['fuel_consumption'] > -99]
    valid_power = df[df['engine_power'] > -99]
    
    total_time = df['time'].max() - df['time'].min()
    
    # Approx total fuel: sum of (rate * time_step)
    # This is a bit complex because we have rate at start of step.
    # WRT calculates this in get_full_fuel, let's approximate or just use the mean * time
    # Better: re-calculate step duration
    df['dt_sec'] = df['time'].diff().dt.total_seconds().shift(-1) # duration of current step
    
    # Calculate per-step consumption
    # Fuel rate is in t/h usually (based on routeparams.py line 168: 'unit': 't/h') aka kg/s in internal logic but t/h in output?
    # Let's check routeparams.py again. 
    # line 167: self.ship_params_per_step.fuel_rate[i].to(u.tonne / u.hour).value
    # So it is t/h.
    
    df['fuel_step_tons'] = df['fuel_consumption'] * (df['dt_sec'] / 3600.0)
    total_fuel_tons = df[df['fuel_consumption'] > 0]['fuel_step_tons'].sum()
    
    avg_speed = df[df['speed'] > -99]['speed'].mean() # m/s usually
    avg_speed_knots = avg_speed * 1.94384 if avg_speed else 0
    
    dist_total_km = 0
    # Simple accumulation of distances if we had them, but we can compute from lat/lon or just sum provided dists if they were in properties
    # They aren't in simple properties.
    
    return {
        'total_time': str(total_time),
        'total_fuel_tons': total_fuel_tons,
        'avg_speed_knots': avg_speed_knots,
        'start_time': df['time'].min(),
        'end_time': df['time'].max()
    }
