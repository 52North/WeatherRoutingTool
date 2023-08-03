import datetime as dt
import logging
import warnings
import logging.handlers

import WeatherRoutingTool.config as config
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.weather import WeatherCondFromFile, WeatherCondODC
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.algorithms.routingalg_factory import *
from WeatherRoutingTool.utils.maps import Map


def merge_figures_to_gif(path, nof_figures):
    graphics.merge_figs(path, nof_figures)


if __name__ == "__main__":
    ##
    # initialise logging
    logger = logging.getLogger('WRT')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(config.PERFORMANCE_LOG_FILE, mode='w')
    fh.setLevel(logging.WARNING)
    fhinfo = logging.FileHandler(config.INFO_LOG_FILE, mode='w')
    fhinfo.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    fhinfo.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(fhinfo)

    ##
    # suppress warnings from mariPower
    warnings.filterwarnings("ignore")

    # *******************************************
    # basic settings
    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    coursesfile = config.COURSES_FILE
    figurepath = config.FIGURE_PATH
    routepath = config.ROUTE_PATH
    time_forecast = config.TIME_FORECAST
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    departure_time = dt.datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    # *******************************************
    # initialise boat
    boat = Tanker(-99)
    boat.init_hydro_model_Route(windfile, coursesfile, depthfile)
    boat.set_boat_speed(config.BOAT_SPEED)

    # *******************************************
    # initialise weather
    wt = WeatherCondFromFile(departure_time, time_forecast, 3)
    # wt = WeatherCondODC(start_time, start_time,time_forecast,3)
    wt.set_map_size(default_map)
    wt.read_dataset(windfile)
    # wt.write_data('/home/kdemmich/MariData/Code/Data/WheatherFiles')

    # *******************************************
    # initialise constraints
    pars = ConstraintPars()
    land_crossing = LandCrossing()
    water_depth = WaterDepth(config.DEPTH_DATA, config.BOAT_DROUGHT, default_map,
                             False)
    # seamarks_crossing = SeamarkCrossing()
    # water_depth.write_reduced_depth_data(
    # '/home/kdemmich/MariData/Code/Data/DepthFiles/ETOPO_renamed.nc')
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
    constraint_list.print_settings()

    # *******************************************
    # initialise rout
    route_factory = RoutingAlgFactory()
    min_fuel_route = route_factory.get_routing_alg('ISOFUEL')
    min_fuel_route.init_fig(water_depth, default_map)

    # *******************************************
    # routing
    min_fuel_route = min_fuel_route.execute_routing(boat, wt, constraint_list)
    # min_fuel_route.print_route()
    # min_fuel_route.write_to_file(str(min_fuel_route.route_type) +
    # "route.json")
    min_fuel_route.return_route_to_API(
        routepath + '/' + str(min_fuel_route.route_type) + ".json")

    # *******************************************
    # plot route in constraints
    fig, ax = plt.subplots(figsize=(12, 7))
    water_depth.plot_route_in_constraint(min_fuel_route,
                                         graphics.get_colour(1), fig, ax)
    plt.savefig(figurepath + '/route_waterdepth.png')
