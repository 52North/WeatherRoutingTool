import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.weather_factory import WeatherFactory


def run_maripower_test_scenario(calmfactor, windfactor, wavefactor, waypoint_dict, geojsondir, maripower_scenario,
                                weather_scenario):
    boat = Tanker(config)
    # boat.set_ship_property('Draught', [draught.mean()])
    boat.set_ship_property('WindForcesFactor', windfactor)
    boat.set_ship_property('WaveForcesFactor', wavefactor)
    boat.set_ship_property('CalmWaterFactor', calmfactor)

    print('Running scenario for ' + weather_scenario + ' with maripower setting ' + maripower_scenario)

    ship_params = boat.get_ship_parameters(waypoint_dict['courses'], waypoint_dict['start_lats'],
                                           waypoint_dict['start_lons'], time[:-1], sog)

    start = (lat[0], lon[0])
    finish = (lat[-1], lon[-1])

    rp = RouteParams(
        count=lat.shape[0] - 1,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)
    parser.add_argument('-r', '--route', help="Route file name (absolute path)", required=True, type=str)
    parser.add_argument('--write-geojson', help="<True|False>. Defaults to 'False'", required=False,
                        type=str, default='False')

    args = parser.parse_args()

    config = Config(file_name=args.file)
    config.print()

    routename = 'original_resistances_calm_weather'
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

    weather_type = 'rough_weather'  # rought_weather, calm_weather
    wind_speed = -99
    VHMO = -99

    if weather_type == 'rough_weather':
        wind_speed = 12.5
        VHMO = 2
    if weather_type == 'calm_weather':
        wind_speed = 2.5
        VHMO = 1

    if wind_speed == -99 or VHMO == -99:
        raise ValueError('windspeed or VHM0 not set!')

    u_comp = - math.sin(45) * wind_speed
    v_comp = - math.cos(45) * wind_speed

    # currents = 0.1
    # utotal = math.sin(45) * currents
    # vtotal = -math.cos(45) * currents

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

    lat, lon, time, sog, draught = RouteParams.from_gzip_file(args.route)
    waypoint_dict = RouteParams.get_per_waypoint_coords(lon, lat, time[0], sog)

    for key in maripower_test_scenarios_wind:
        run_maripower_test_scenario(
            maripower_test_scenarios_calm[key],
            maripower_test_scenarios_wind[key],
            maripower_test_scenarios_wave[key],
            waypoint_dict,
            routepath,
            key,
            weather_type)
