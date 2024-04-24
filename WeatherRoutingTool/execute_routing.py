# import cProfile
from datetime import datetime

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory, WaterDepth
from WeatherRoutingTool.algorithms.routingalg_factory import RoutingAlgFactory
from WeatherRoutingTool.utils.maps import Map


def merge_figures_to_gif(path, nof_figures):
    graphics.merge_figs(path, nof_figures)


def execute_routing(config):
    # prof = cProfile.Profile()
    # prof.enable()

    # *******************************************
    # basic settings
    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    routepath = config.ROUTE_PATH
    time_resolution = config.DELTA_TIME_FORECAST
    time_forecast = config.TIME_FORECAST
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    # *******************************************
    # initialise weather
    wt = WeatherFactory.get_weather(config.DATA_MODE, windfile, departure_time, time_forecast, time_resolution,
                                    default_map)

    # *******************************************
    # initialise boat
    boat = ShipFactory.get_ship(config)

    # *******************************************
    # initialise constraints
    max_draught = max(config.BOAT_DRAUGHT_AFT, config.BOAT_DRAUGHT_FORE)
    water_depth = WaterDepth(config.DATA_MODE, max_draught + config.UNDER_KEEL_CLEARANCE,
                             default_map, depthfile)
    constraint_list = ConstraintsListFactory.get_constraints_list(
        constraints_string_list=config.CONSTRAINTS_LIST, data_mode=config.DATA_MODE,
        min_depth=max_draught + config.UNDER_KEEL_CLEARANCE,
        map_size=default_map, depthfile=depthfile, waypoints=config.INTERMEDIATE_WAYPOINTS,
        courses_path=config.COURSES_FILE)

    # *******************************************
    # initialise route
    min_fuel_route = RoutingAlgFactory.get_routing_alg(config)
    min_fuel_route.init_fig(water_depth=water_depth, map_size=default_map)

    # *******************************************
    # routing
    min_fuel_route = min_fuel_route.execute_routing(boat, wt, constraint_list)
    # min_fuel_route.print_route()
    min_fuel_route.return_route_to_API(routepath + '/' + str(min_fuel_route.route_type) + ".json")

    # prof.disable()
    # prof.dump_stats('wrt_run.prof')
