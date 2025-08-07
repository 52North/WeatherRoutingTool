# import cProfile
from datetime import datetime

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory, WaterDepth
from WeatherRoutingTool.constraints.route_postprocessing import RoutePostprocessing
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
    departure_time = config.DEPARTURE_TIME
    default_map = Map(lat1, lon1, lat2, lon2)

    # *******************************************
    # initialise weather
    wt = WeatherFactory.get_weather(config._DATA_MODE_WEATHER, windfile, departure_time, time_forecast, time_resolution,
                                    default_map)

    # *******************************************
    # initialise boat
    boat = ShipFactory.get_ship(config)

    # *******************************************
    # initialise constraints
    water_depth = WaterDepth(config._DATA_MODE_DEPTH, boat.get_required_water_depth(),
                             default_map, depthfile)
    constraint_list = ConstraintsListFactory.get_constraints_list(
        constraints_string_list=config.CONSTRAINTS_LIST, data_mode=config._DATA_MODE_DEPTH,
        min_depth=boat.get_required_water_depth(),
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

    if config.ROUTE_POSTPROCESSING:
        postprocessed_route = RoutePostprocessing(min_fuel_route, boat)
        min_fuel_route_postprocessed = postprocessed_route.post_process_route()
        min_fuel_route_postprocessed.return_route_to_API(routepath + '/' + str(min_fuel_route_postprocessed.route_type)
                                                         + '_postprocessed' + ".json")
    # prof.disable()
    # prof.dump_stats('wrt_run.prof')
