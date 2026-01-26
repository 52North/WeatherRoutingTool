##
# Functionality to estimate the fuel consumption for a specific route via different fuel consumption models ('maripower'
# and 'direct_power_method') or from data.
##

import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
from astropy import units as u

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.maripower_tanker import MariPowerTanker
from WeatherRoutingTool.ship.shipparams import ShipParams


def run_dpm_test_scenario(waypoint_dict, geojsondir, sog, output_route):
    boat = DirectPowerBoat(file_name=config.CONFIG_PATH)
    boat.speed = sog
    boat.load_data()

    ship_params = boat.get_ship_parameters(waypoint_dict['courses'], waypoint_dict['start_lats'],
                                           waypoint_dict['start_lons'], waypoint_dict['start_times'], sog)

    start = (lat[0], lon[0])
    finish = (lat[-1], lon[-1])

    # add arrival at destination to 'start_times'
    travel_time = timedelta(seconds=waypoint_dict['travel_times'][-1].to("second").value)
    waypoint_dict['start_times'] = np.append(
        waypoint_dict['start_times'],
        waypoint_dict['start_times'][-1] + travel_time
    )

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
        starttime_per_step=waypoint_dict['start_times'],
        ship_params_per_step=ship_params)

    if geojsondir:
        filename = output_route
        print('Writing file: ', filename)
        rp.write_to_geojson(filename)


def lat_lon_from_file(filename):
    with open(filename) as file:
        print('reading file: ', filename)
        rp_dict = json.load(file)

    point_list = rp_dict['features']
    count = len(point_list)

    lats_per_step = np.full(count, -99.)
    lons_per_step = np.full(count, -99.)

    for ipoint in range(0, count):
        coord_pair = point_list[ipoint]['geometry']['coordinates']
        lats_per_step[ipoint] = coord_pair[1]
        lons_per_step[ipoint] = coord_pair[0]

    return lats_per_step, lons_per_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    required_args.add_argument('-f', '--file', help="Config file name (absolute path)",
                               required=True, type=str)
    required_args.add_argument('-r-in', '--input_route', help="Route file name (absolute path)",
                               required=True, type=str)
    required_args.add_argument('-r-out', '--output_route', help="Route file name (absolute path)",
                               required=True, type=str)

    set_up_logging()

    # read arguments
    args = parser.parse_args()
    config = Config.assign_config(Path(args.file))

    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    input_route = args.input_route
    output_route = args.output_route
    coursesfile = config.COURSES_FILE
    figurefile = get_figure_path()
    time_resolution = config.DELTA_TIME_FORECAST
    time_forecast = config.TIME_FORECAST
    departure_time = config.DEPARTURE_TIME
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    default_map = Map(lat1, lon1, lat2, lon2)
    sog = 7.717 * u.meter/u.second

    # obtain position, time and courses for every waypoint
    lat, lon = lat_lon_from_file(input_route)
    waypoint_dict = RouteParams.get_per_waypoint_coords(lon, lat, departure_time, sog)

    # obtain RouteParams object for different models or for gzip data
    run_dpm_test_scenario(
            waypoint_dict,
            input_route,
            sog,
            output_route
    )
