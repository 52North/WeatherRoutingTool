import argparse
import math
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.weather_factory import WeatherFactory


def run_maripower_test_scenario(calmfactor, windfactor, wavefactor, waypoint_dict, geojsondir, maripower_scenario,
                                weather_scenario, draught_fp, draught_ap):
    boat = Tanker(config)
    boat.set_ship_property('Draught_FP', draught_fp.mean())
    boat.set_ship_property('Draught_AP', draught_ap.mean())
    boat.set_ship_property('WindForcesFactor', windfactor)
    boat.set_ship_property('WaveForcesFactor', wavefactor)
    boat.set_ship_property('CalmWaterFactor', calmfactor)

    print('Running scenario for ' + weather_scenario + ' with maripower setting ' + maripower_scenario)

    ship_params = boat.get_ship_parameters(waypoint_dict['courses'], waypoint_dict['start_lats'],
                                           waypoint_dict['start_lons'], time[:-1], sog)

    start = (lat[0], lon[0])
    finish = (lat[-1], lon[-1])

    rp = RouteParams(
        count=lat.shape[0] - 2,
        start=start,
        finish=finish,
        gcr=None,
        route_type='read_from_csv',
        time=waypoint_dict['travel_times'],
        lats_per_step=lat,
        lons_per_step=lon,
        course_per_step=waypoint_dict['courses'],
        dists_per_step=waypoint_dict['dist'],
        starttime_per_step=time,
        ship_params_per_step=ship_params)

    if geojsondir:
        filename = os.path.join(geojsondir, 'route_' + weather_scenario + '_' + maripower_scenario + '.json')
        print('Writing file: ', filename)
        rp.return_route_to_API(filename)


def cut_indices(lat, lon, time, sog, fore_draught, aft_draught, cut_route):
    start_indices_lat = int(np.where(abs(lat - cut_route[0]) < 0.001)[0][0])
    end_indices_lat = int(np.where(abs(lat - cut_route[2]) < 0.001)[0][0])
    start_indices_lon = int(np.where(abs(lon - cut_route[1]) < 0.001)[0][0])
    end_indices_lon = int(np.where(abs(lon - cut_route[3]) < 0.001)[0][0])

    print('start_lat: ', lat[start_indices_lat])
    print('end_lat: ', lat[end_indices_lat])
    print('start_lon: ', lon[start_indices_lon])
    print('end_lon: ', lon[end_indices_lon])

    if (start_indices_lat != start_indices_lon) or (end_indices_lat != end_indices_lon):
        raise ValueError('Latitude and longitude are not matching for cut')

    lat = lat[start_indices_lat:end_indices_lat + 1]
    lon = lon[start_indices_lat:end_indices_lat + 1]
    time = time[start_indices_lat:end_indices_lat + 1]
    fore_draught = fore_draught[start_indices_lat:end_indices_lat + 1]
    aft_draught = aft_draught[start_indices_lat:end_indices_lat + 1]
    sog = sog[start_indices_lat:end_indices_lat]

    print('mean speed cut: ', sog.mean())
    print('start time cut: ', time[0])
    print('mean aft draugh cut: ', aft_draught.mean())
    print('mean fore draugh cut: ', fore_draught.mean())

    return lat, lon, time, sog, fore_draught, aft_draught


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)
    parser.add_argument('-r', '--route', help="Route file name (absolute path)", required=True, type=str)
    parser.add_argument('--write-geojson', help="<True|False>. Defaults to 'False'", required=False,
                        type=str, default='False')

    set_up_logging()

    args = parser.parse_args()

    config = Config(file_name=args.file)
    config.print()

    routename = 'original_resistances_calm_weather'
    # cut_route = [37.5057, 12.3046, 33.827, 34.311]    # MedSea
    # cut_route = [48.634715, -6.081798333333333, 43.53411833333333, -10.028818333333334]     # Ärmelkanal
    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    if str(args.write_geojson).lower() == 'true':
        routepath = config.ROUTE_PATH
    else:
        routepath = None
    coursesfile = config.COURSES_FILE
    figurefile = get_figure_path()
    time_resolution = config.DELTA_TIME_FORECAST
    time_forecast = config.TIME_FORECAST
    departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    default_map = Map(lat1, lon1, lat2, lon2)

    weather_type = 'real_weather'  # rough_weather, calm_weather
    wind_speed = -99
    VHMO = -99

    if weather_type == 'rough_weather':
        wind_speed = 12.5
        VHMO = 2
    if weather_type == 'calm_weather':
        wind_speed = 2.5
        VHMO = 1

    if (wind_speed == -99 or VHMO == -99) and (weather_type != 'real_weather'):
        raise ValueError('windspeed or VHM0 not set!')

    u_comp = - math.sin(math.radians(45)) * wind_speed
    v_comp = - math.cos(math.radians(45)) * wind_speed

    # currents = 0.1
    # utotal = math.sin(math.radians(45)) * currents
    # vtotal = - math.cos(math.radians(45)) * currents

    var_dict = {
        'thetao': 20,  # °C
        'so': 33.5,
        'Temperature_surface': 283,  # K
        'Pressure_reduced_to_MSL_msl': 101325,
        'u-component_of_wind_height_above_ground': u_comp,
        'v-component_of_wind_height_above_ground': v_comp,
        # 'utotal': utotal,
        # 'vtotal': vtotal,
        'VHM0': VHMO,
        'VMDR': 45,
        'VTPK': 10
    }

    maripower_test_scenarios_calm = {'original': 1., '95perc_calm': 0.95, '105perc_calm': 1.05, '80perc_wind': 1.,
                                     '120perc_wind': 1., '80perc_wave': 1., '120perc_wave': 1.}
    maripower_test_scenarios_wind = {'original': 1., '95perc_calm': 1., '105perc_calm': 1., '80perc_wind': 0.8,
                                     '120perc_wind': 1.2, '80perc_wave': 1., '120perc_wave': 1.}
    maripower_test_scenarios_wave = {'original': 1., '95perc_calm': 1., '105perc_calm': 1., '80perc_wind': 1.,
                                     '120perc_wind': 1., '80perc_wave': 0.8, '120perc_wave': 1.2}

    wt = WeatherFactory.get_weather(data_mode=config.DATA_MODE,
                                    file_path=windfile,
                                    departure_time=departure_time,
                                    time_forecast=time_forecast,
                                    time_resolution=time_resolution,
                                    default_map=default_map,
                                    var_dict=var_dict)
    fig, ax = plt.subplots(figsize=(12, 7))
    # wt.plot_weather_map(fig, ax, "2023-08-16T12:00:00", "wind")

    lat, lon, time, sog, fore_draught, aft_draught = RouteParams.from_gzip_file(args.route)
    # lat, lon, time, sog, fore_draught, aft_draught = cut_indices(lat, lon, time, sog, fore_draught,
    #                                                               aft_draught, cut_route)

    waypoint_dict = RouteParams.get_per_waypoint_coords(lon, lat, time[0], sog)

    for key in maripower_test_scenarios_wind:
        run_maripower_test_scenario(
            maripower_test_scenarios_calm[key],
            maripower_test_scenarios_wind[key],
            maripower_test_scenarios_wave[key],
            waypoint_dict,
            routepath,
            key,
            weather_type,
            fore_draught,
            aft_draught
        )
