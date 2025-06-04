import argparse
import math
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.ship.maripower_tanker import MariPowerTanker
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.weather_factory import WeatherFactory


def run_maripower_test_scenario(calmfactor, windfactor, wavefactor, waypoint_dict, geojsondir, maripower_scenario,
                                draught_fp, draught_ap, sog):
    boat = MariPowerTanker(file_name = config.CONFIG_PATH)
    boat.set_ship_property('Draught_FP', draught_fp.mean())
    boat.set_ship_property('Draught_AP', draught_ap.mean())
    boat.set_ship_property('WindForcesFactor', windfactor)
    boat.set_ship_property('WaveForcesFactor', wavefactor)
    boat.set_ship_property('CalmWaterFactor', calmfactor)
    boat.speed = sog
    boat.load_data()

    print('Running maripower setting ' + maripower_scenario)

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
        filename = os.path.join(geojsondir, 'route_' + maripower_scenario + '.json')
        print('Writing file: ', filename)
        rp.return_route_to_API(filename)

def run_dpm_test_scenario(waypoint_dict, geojsondir, maripower_scenario, sog):
    boat = DirectPowerBoat(file_name = config.CONFIG_PATH)
    boat.speed = sog
    boat.load_data()

    print('Running direct power boat setting ' + maripower_scenario)

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
        filename = os.path.join(geojsondir, 'route_' + maripower_scenario + '.json')
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
    parser.add_argument('-n', '--name',help="Route file name (absolute path)", required=True, type=str)

    parser.add_argument('--write-geojson', help="<True|False>. Defaults to 'False'", required=False,
                        type=str, default='False')
    parser.add_argument('-wave', '--wave_scenario', help="Route file name (absolute path)", required=False, type=float, default=1.)
    parser.add_argument('-wind', '--wind_scenario', help="Route file name (absolute path)", required=False, type=float, default=1.)
    parser.add_argument('-calm', '--calm_water_scenario', help="Route file name (absolute path)", required=False, type=float, default=1.)
    parser.add_argument('-bt', '--boat_type', help="<maripower|direct_power_method>. Defaults to 'direct_power_method' ", required=False,
                        type=str, default='direct_power_method')

    set_up_logging()

    args = parser.parse_args()

    config = Config(file_name=args.file)
    config.print()

    scenario_name = args.name

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

    maripower_test_scenarios_calm = args.calm_water_scenario
    maripower_test_scenarios_wind = args.wind_scenario
    maripower_test_scenarios_wave = args.wave_scenario

    lat, lon, time, sog, fore_draught, aft_draught, power, fuel_rate = RouteParams.from_gzip_file(args.route)
    # lat, lon, time, sog, fore_draught, aft_draught = cut_indices(lat, lon, time, sog, fore_draught,
    #                                                               aft_draught, cut_route)

    # calculate power
    boat = DirectPowerBoat(file_name=config.CONFIG_PATH)
    power = power * boat.power_at_sp * 0.01 * 4/3 # power_at_sp is 75% of SMCR power

    waypoint_dict = RouteParams.get_per_waypoint_coords(lon, lat, time[0], sog)

    if str(args.boat_type) == 'direct_power_method':
        sog = np.average(sog)
        #sog = 7.7 * u.meter/ u.second
        run_dpm_test_scenario(
            waypoint_dict,
            routepath,
            scenario_name,
            sog
        )

    if str(args.boat_type) == 'maripower':
        sog = np.full(len(lon) - 1, np.average(sog)) * u.meter / u.second
        # sog =  np.full(len(lon) - 1, 7.7) * u.meter / u.second

        run_maripower_test_scenario(
            maripower_test_scenarios_calm,
            maripower_test_scenarios_wind,
            maripower_test_scenarios_wave,
            waypoint_dict,
            routepath,
            scenario_name,
            fore_draught,
            aft_draught,
            sog
        )

    if str(args.boat_type) == 'data':
        ship_params = ShipParams.set_default_array_1D(len(lon))
        ship_params.fuel_rate = fuel_rate
        ship_params.power = power

        start = (lat[0], lon[0])
        finish = (lat[-1], lon[-1])

        rp = RouteParams(
            count=lat.shape[0] - 2,
            start=start,
            finish=finish,
            gcr=None,
            route_type='read_from_gzip',
            time=waypoint_dict['travel_times'],
            lats_per_step=lat,
            lons_per_step=lon,
            course_per_step=waypoint_dict['courses'],
            dists_per_step=waypoint_dict['dist'],
            starttime_per_step=time,
            ship_params_per_step=ship_params,
        )

        if routepath:
            filename = os.path.join(routepath, 'route_' + scenario_name + '.json')
            print('Writing file: ', filename)
            rp.return_route_to_API(filename)
