import argparse
import warnings
from datetime import datetime

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.algorithms.routingalg_factory import *
from WeatherRoutingTool.utils.maps import Map


def merge_figures_to_gif(path, nof_figures):
    graphics.merge_figs(path, nof_figures)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)
    parser.add_argument('--warnings-log-file',
                        help="Logging file name (absolute path) for warnings and above.", required=False, type=str)
    parser.add_argument('--info-log-file',
                        help="Logging file name (absolute path) for info and above.", required=False, type=str)
    parser.add_argument('--debug', help="Enable debug mode. <True|False>. Defaults to 'False'.",
                        required=False, type=str, default='False')
    args = parser.parse_args()
    if not args.file:
        raise RuntimeError("No config file name provided!")
    debug_str = str(args.debug).lower()
    if debug_str == 'true':
        args.debug = True
    elif debug_str == 'false':
        args.debug = False
    else:
        raise ValueError("--debug does not have a valid value")

    ##
    # initialise logging
    set_up_logging(args.info_log_file, args.warnings_log_file, args.debug)

    ##
    # create config object
    config = Config(file_name=args.file)
    config.print()

    ##
    # suppress warnings from mariPower
    # warnings.filterwarnings("ignore")

    # *******************************************
    # basic settings
    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    coursesfile = config.COURSES_FILE
    routepath = config.ROUTE_PATH
    time_resolution = config.DELTA_TIME_FORECAST
    time_forecast = config.TIME_FORECAST
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    # *******************************************
    # initialise weather
    #
    # wt = WeatherCondEnvAutomatic(departure_time, time_forecast, 3)
    # wt.set_map_size(default_map)
    # wt.read_dataset()
    # weather_path = wt.write_data('/home/kdemmich/MariData/Code/Data/WheatherFiles')

    # wt_read = WeatherCondFromFile(departure_time, time_forecast, 3)
    # wt_read.read_dataset(weather_path)
    # wt.write_data('/home/kdemmich/MariData/Code/Data/WheatherFiles')
    wf = WeatherFactory()
    wt = wf.get_weather(config.DATA_MODE, windfile, departure_time, time_forecast, time_resolution, default_map)

    # *******************************************
    # initialise boat
    bf = ShipFactory()
    boat = bf.get_ship(config)

    # *******************************************
    # initialise constraints
    '''pars = ConstraintPars()
    land_crossing = LandCrossing()
    # seamarks_crossing = SeamarkCrossing()
    # water_depth.plot_depth_map_from_file(depthfile, lat1, lon1, lat2, lon2)
    on_map = StayOnMap()
    on_map.set_map(lat1, lon1, lat2, lon2)
    continuous_checks_seamarks = SeamarkCrossing()
    continuous_checks_land = LandPolygonsCrossing()

    constraint_list = ConstraintsList(pars)
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(on_map)
    constraint_list.add_neg_constraint(water_depth)
    # constraint_list.add_neg_constraint(continuous_checks_seamarks,
    # 'continuous')
    # constraint_list.add_neg_constraint(continuous_checks_land, 'continuous')
    constraint_list.print_settings()'''

    water_depth = WaterDepth(config.DATA_MODE, config.BOAT_DRAUGHT, default_map, depthfile)
    constraint_list = ConstraintsListFactory.get_constraints_list(
        constraints_string_list=config.CONSTRAINTS_LIST, data_mode=config.DATA_MODE, boat_draught=config.BOAT_DRAUGHT,
        map_size=default_map, depthfile=depthfile, waypoints=config.INTERMEDIATE_WAYPOINTS)

    # *******************************************
    # initialise route
    min_fuel_route = RoutingAlgFactory.get_routing_alg(config)
    min_fuel_route.init_fig(water_depth=water_depth, map_size=default_map)

    # *******************************************
    # routing
    min_fuel_route = min_fuel_route.execute_routing(boat, wt, constraint_list)
    # min_fuel_route.print_route()
    # min_fuel_route.write_to_file(str(min_fuel_route.route_type) +
    # "route.json")
    min_fuel_route.return_route_to_API(routepath + '/' + str(min_fuel_route.route_type) + ".json")
