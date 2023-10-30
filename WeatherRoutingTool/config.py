import os

import WeatherRoutingTool.utils.formatting as format

##
# Output variables
COURSES_FILE = os.getenv(
    'WRT_BASE_PATH') + '/CoursesRoute.nc'  # path to file that acts as intermediate storage for courses per routing step
FIGURE_PATH = os.getenv('WRT_FIGURE_PATH')  # path to figure repository
PERFORMANCE_LOG_FILE = os.getenv('WRT_PERFORMANCE_LOG_FILE')  # path to log file which logs performance
INFO_LOG_FILE = os.getenv('WRT_INFO_LOG_FILE')  # path to log file which logs information
ROUTE_PATH = os.getenv('WRT_ROUTE_PATH')  # path to json file to which the route will be written
BASE_PATH = os.getenv('WRT_BASE_PATH')  # path towards the WeatherRoutingTool base directory

##
# Input variables
DEFAULT_ROUTE = format.get_bbox_from_string(os.getenv('WRT_DEFAULT_ROUTE', '-99'))
DEPARTURE_TIME = os.getenv('WRT_DEPARTURE_TIME')  # start time of travelling
DEFAULT_MAP = format.get_bbox_from_string(os.getenv('WRT_DEFAULT_MAP', '-99'))
BOAT_SPEED = float(os.getenv('WRT_BOAT_SPEED', '-99'))  # (m/s)
BOAT_DRAUGHT = 10  # os.getenv('WRT_BOAT_DRAUGHT')  # (m)

##
# Constant settings for isobased algorithm
TIME_FORECAST = 90  # forecast hours weather
ROUTING_STEPS = 60  # number of routing steps
DELTA_TIME_FORECAST = 3  # time resolution of weather forecast (hours)
DELTA_FUEL = 1 * 5000  # amount of fuel per routing step (kg)

ROUTER_HDGS_SEGMENTS = 30  # total number of courses : put even number!!
ROUTER_HDGS_INCREMENTS_DEG = 6  # increment of headings
# ROUTER_HDGS_SEGMENTS = 40  # total number of courses : put even number!!
# ROUTER_HDGS_INCREMENTS_DEG = 3  # increment of headings
# ROUTER_HDGS_SEGMENTS = 4  # total number of courses : put even number!!
# ROUTER_HDGS_INCREMENTS_DEG = 10  # increment of headings
ROUTER_RPM_SEGMENTS = 1  # not used yet
ROUTER_RPM_INCREMENTS_DEG = 1  # not used yet
ISOCHRONE_EXPECTED_SPEED_KTS = 8  # not used yet
ISOCHRONE_PRUNE_SECTOR_DEG_HALF = 91  # angular range of azimuth angle that is considered for pruning (only one half!)
ISOCHRONE_PRUNE_SEGMENTS = 20  # total number of azimuth bins that are used for pruning in prune sector which is 2x
# ISOCHRONE_PRUNE_SEGMENTS=20
ISOCHRONE_PRUNE_GCR_CENTERED = False
ISOCHRONE_PRUNE_BEARING = True
ISOCHRONE_MINIMISATION_CRITERION = 'squareddist_over_disttodest'  # minimisation criterion. Can be full travel distance ('dist') and travel
# distance squared divided by the distance towards the destination ('squareddist_over_disttodest')

##
# configurations for local execution
WEATHER_DATA = os.getenv('WRT_WEATHER_DATA')  # path to weather data
DEPTH_DATA = os.getenv('WRT_DEPTH_DATA')  # path to depth data
CMEMS_USER = os.getenv('CMEMS_USERNAME')
CMEMS_PASSWORD = os.getenv('CMEMS_PASSWORD')

##
# Database connection paramteters
HOST = os.getenv('WRT_HOST')
DATABASE = os.getenv('WRT_DATABASE')
MYUSERNAME = os.getenv('WRT_MYUSERNAME')
PASSWORD = os.getenv('WRT_PASSWORD')
PORT = os.getenv('WRT_PORT')

DATA_MODE = 'from_file'
