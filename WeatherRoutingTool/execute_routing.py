# import cProfile
import logging

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory, WaterDepth
from WeatherRoutingTool.constraints.route_postprocessing import RoutePostprocessing
from WeatherRoutingTool.algorithms.routingalg_factory import RoutingAlgFactory
from WeatherRoutingTool.utils.errors import IsobasedRoutingError
from WeatherRoutingTool.utils.maps import Map

logger = logging.getLogger('WRT.execute_routing')


def merge_figures_to_gif(path, nof_figures):
    graphics.merge_figs(path, nof_figures)


def execute_routing(config, ship_config):
    """
    Execute route optimization based on the user-defined configuration.

    Initialises the weather data, the ship, the constraints and the routing algorithm from the configuration and
    runs the routing. The resulting route is written to ``<ROUTE_PATH>/<route_type>.json`` in GeoJSON format. If
    ``ROUTE_POSTPROCESSING`` is enabled, the postprocessed route is additionally written to
    ``<ROUTE_PATH>/<route_type>_postprocessed.json``. If no weather data is used, the constraint ``'in_time'``
    is ignored.

    :param config: validated configuration
    :type config: WeatherRoutingTool.config.Config
    :param ship_config: validated ship configuration
    :type ship_config: WeatherRoutingTool.ship.ship_config.ShipConfig
    :return: resulting route
    :rtype: WeatherRoutingTool.routeparams.RouteParams
    :raises WRTErrors: if the routing fails. If an isochrone-based algorithm terminates with an error
        (``IsobasedRoutingError``), the best route found up to this point is written before the error is raised.
    """
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
    if config._DATA_MODE_WEATHER == "skip":
        time_frame = None
        if "in_time" in config.CONSTRAINTS_LIST:
            logger.warning("No weather data used. Ignoring constraint 'in_time'.")
    else:
        time_frame = (wt.time_start, wt.time_end)

    # *******************************************
    # initialise boat
    boat = ShipFactory.get_ship(config.BOAT_TYPE, ship_config)

    # *******************************************
    # initialise constraints
    water_depth = WaterDepth(config._DATA_MODE_DEPTH, boat.get_required_water_depth(),
                             default_map, depthfile)
    constraint_list = ConstraintsListFactory.get_constraints_list(
        constraints_string_list=config.CONSTRAINTS_LIST, data_mode=config._DATA_MODE_DEPTH,
        min_depth=boat.get_required_water_depth(),
        map_size=default_map, depthfile=depthfile, waypoints=config.INTERMEDIATE_WAYPOINTS,
        courses_path=config.COURSES_FILE, time_frame=time_frame)

    # *******************************************
    # initialise route
    alg = RoutingAlgFactory.get_routing_alg(config)
    alg.init_fig(water_depth=water_depth, map_size=default_map)

    # *******************************************
    # routing
    try:
        min_fuel_route = alg.execute_routing(boat, wt, constraint_list)
    except IsobasedRoutingError as err:
        # The routing terminated regularly but with an error. Write the best route that was found before the error
        # is passed on.
        write_route(err.route, routepath, boat, config.ROUTE_POSTPROCESSING)
        raise
    write_route(min_fuel_route, routepath, boat, config.ROUTE_POSTPROCESSING)

    # prof.disable()
    # prof.dump_stats('wrt_run.prof')
    return min_fuel_route


def write_route(route, routepath, boat, postprocessing):
    """
    Write a route to ``<routepath>/<route_type>.json`` in GeoJSON format. If ``postprocessing`` is True, the
    postprocessed route is additionally written to ``<routepath>/<route_type>_postprocessed.json``. Nothing is
    written if ``route`` is None.

    :param route: route to be written
    :type route: WeatherRoutingTool.routeparams.RouteParams or None
    :param routepath: path to the output folder
    :type routepath: pathlib.Path or str
    :param boat: ship object, required for the postprocessing
    :type boat: WeatherRoutingTool.ship.ship.Boat
    :param postprocessing: whether to postprocess the route
    :type postprocessing: bool
    """
    if not route:
        return
    route.write_to_geojson(str(routepath) + '/' + str(route.route_type) + ".json")

    if postprocessing:
        postprocessed_route = RoutePostprocessing(route, boat).post_process_route()
        postprocessed_route.write_to_geojson(
            str(routepath) + '/' + str(postprocessed_route.route_type) + "_postprocessed.json")
