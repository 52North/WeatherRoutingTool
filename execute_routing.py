import io
import logging
import os
import warnings
import sys
import logging.handlers
from logging import FileHandler, Formatter

# Added because of package import error
# Define current working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current working directory as a search location for Python modules and Packages
sys.path.append(os.path.join(os.getcwd(), ""))


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import config
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.ship.ship import *
from WeatherRoutingTool.weather import *
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
    model = config.START_TIME
    boatfile = config.DEFAULT_BOAT
    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    coursesfile = config.COURSES_FILE
    figurepath = config.FIGURE_PATH
    routepath = config.ROUTE_PATH
    delta_time = config.DELTA_TIME_FORECAST
    delta_fuel = config.DELTA_FUEL
    hours = config.TIME_FORECAST
    routing_steps = config.ROUTING_STEPS
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    r_la1, r_lo1, r_la2, r_lo2 = config.DEFAULT_ROUTE
    start = (r_la1, r_lo1)
    finish = (r_la2, r_lo2)
    start_time = dt.datetime.strptime(config.START_TIME, '%Y%m%d%H')
    map = Map(lat1,lon1,lat2,lon2)

    # *******************************************
    # initialise boat
    boat = Tanker(-99)
    boat.init_hydro_model_Route(windfile, coursesfile, depthfile)
    boat.set_boat_speed(config.BOAT_SPEED)
    # boat.calibrate_simple_fuel()
    # boat.write_simple_fuel()
    # boat.test_power_consumption_per_course()
    # boat.test_power_consumption_per_speed()

    # *******************************************
    # initialise weather
    wt = WeatherCondFromFile(model, start_time, hours, 3)
    #wt = WeatherCondODC(model, start_time,hours,3)
    wt.set_map_size(map)
    wt.read_dataset(windfile)
    #wt.write_data('/home/kdemmich/MariData/Code/Data/WheatherFiles')

    # *******************************************
    # initialise constraints
    pars = ConstraintPars()
    land_crossing = LandCrossing()
    water_depth = WaterDepth(config.DEPTH_DATA, config.BOAT_DROUGHT, map)
    #seamarks_crossing = SeamarkCrossing()
    #water_depth.write_reduced_depth_data('/home/kdemmich/MariData/Code/Data/DepthFiles/ETOPO_renamed.nc')
    # water_depth.plot_depth_map_from_file(depthfile, lat1, lon1, lat2, lon2)
    on_map = StayOnMap()
    on_map.set_map(lat1, lon1, lat2, lon2)
    continuous_checks_seamarks = SeamarkCrossing()
    continuous_checks_land = LandPolygonsCrossing()

    #Simulationsstudie 2, Thames <-> Gothenburg
    #over_waypoint1 = PositiveConstraintPoint(51.128497, 1.700607)
    #over_waypoint2 = PositiveConstraintPoint(51.753670, 2.600120)
    #over_waypoint3 = PositiveConstraintPoint(53.121505, 2.722398)

    #over_waypoint4 = PositiveConstraintPoint(55.796111, 3.100278)  # good weather
    #over_waypoint4 = PositiveConstraintPoint(54.608889, 6.179722)  # ok weather
    #over_waypoint4 = PositiveConstraintPoint(55.048333, 5.130000)  # bad weather

    #Simulationsstudie 2, Thames <-> Bordeaux
    #over_waypoint1 = PositiveConstraintPoint(51.098903, 1.549883)
    #over_waypoint2 = PositiveConstraintPoint(50.600152, 0.609062)
    #over_waypoint3 = PositiveConstraintPoint(49.988757, -2.915933)

    over_waypoint3 = PositiveConstraintPoint(49.988757, -2.915933)
    #over_waypoint4 = PositiveConstraintPoint(48.850777, -5.870688)
    
    #over_waypoint4 = PositiveConstraintPoint(45.715, -5.502222)      # good weather
    #over_waypoint4 = PositiveConstraintPoint(54.608889, 6.179722)   # ok weather
    #over_waypoint4 = PositiveConstraintPoint(55.048333, 5.130000)   # bad weather


    constraint_list = ConstraintsList(pars)
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(on_map)
    constraint_list.add_neg_constraint(water_depth)
    constraint_list.add_neg_constraint(continuous_checks_seamarks, 'continuous')
    constraint_list.add_neg_constraint(continuous_checks_land, 'continuous')

    #constraint_list.add_pos_constraint(over_waypoint1)
    #constraint_list.add_pos_constraint(over_waypoint2)
    #constraint_list.add_pos_constraint(over_waypoint3)
    constraint_list.add_pos_constraint(over_waypoint3)

    #constraint_list.add_pos_constraint(over_waypoint4)
    constraint_list.print_settings()

    # *******************************************
    # initialise rout
    route_factory = RoutingAlgFactory()
    min_fuel_route = route_factory.get_routing_alg('ISOFUEL')
    min_fuel_route.init_fig(water_depth, map)

    # *******************************************
    # routing
    min_fuel_route = min_fuel_route.execute_routing(boat, wt, constraint_list)
    min_fuel_route.print_route()
    #min_fuel_route.write_to_file(str(min_fuel_route.route_type) + "route.json")
    min_fuel_route.return_route_to_API(routepath + '/' + str(min_fuel_route.route_type) + ".json")

    # *******************************************
    # plot route in constraints
    fig, ax = plt.subplots(figsize=(12, 7))
    water_depth.plot_route_in_constraint(min_fuel_route, graphics.get_colour(1), fig, ax)
    plt.savefig(figurepath + '/route_waterdepth.png')
