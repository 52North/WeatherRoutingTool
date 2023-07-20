import os
from dotenv import load_dotenv

load_dotenv()

##
# Defaults
DEFAULT_MAP = [42, -17.5, 52, 2.5] # med sea
#DEFAULT_MAP = [51, 1.2,60,12]
#DEFAULT_ROUTE = [51.121667, 1.355833, 57.56141, 11.65856]
DEFAULT_ROUTE = [51.121667, 1.355833, 45.4907, -1.491394]
TIME_FORECAST = 80              # forecast hours weather
ROUTING_STEPS = 20             # number of routing steps
DELTA_TIME_FORECAST = 3600      # time resolution of weather forecast (seconds)
#DELTA_FUEL = 30000000*1000*1 # [Ws]
DELTA_FUEL = 1*1000             # amount of fuel per routing step (kg)
START_TIME = '2023062609'       # start time of travelling
BOAT_SPEED = 20                 # (m/s)
BOAT_DROUGHT = 10                 # (m)

##
# File paths
WEATHER_DATA = os.getenv('WEATHER_DATA')   # path to weather data
DEPTH_DATA = os.getenv('DEPTH_DATA')       # path to depth data
PERFORMANCE_LOG_FILE = os.getenv('PERFORMANCE_LOG_FILE')   # path to log file which logs performance
INFO_LOG_FILE = os.getenv('INFO_LOG_FILE') # path to log file which logs information
FIGURE_PATH = os.getenv('FIGURE_PATH')     # path to figure repository
COURSES_FILE = os.getenv('BASE_PATH') + '/CoursesRoute.nc'     # path to file that acts as intermediate storage for courses per routing step
ROUTE_PATH = os.getenv('ROUTE_PATH')
BASE_PATH = os.getenv('BASE_PATH')

##
# Database connection paramteters
HOST = os.getenv('HOST')
DATABASE = os.getenv('DATABASE')
MYUSERNAME = os.getenv('MYUSERNAME')
PASSWORD = os.getenv('PASSWORD')
PORT = os.getenv('PORT')


##
# Isochrone routing parameters
ROUTER_HDGS_SEGMENTS =  30               # total number of courses : put even number!!
ROUTER_HDGS_INCREMENTS_DEG = 6           # increment of headings
ROUTER_RPM_SEGMENTS = 1                  # not used yet
ROUTER_RPM_INCREMENTS_DEG = 1            # not used yet
ISOCHRONE_EXPECTED_SPEED_KTS = 8         # not used yet
ISOCHRONE_PRUNE_SECTOR_DEG_HALF = 91     # angular range of azimuth angle that is considered for pruning (only one half!)
ISOCHRONE_PRUNE_SEGMENTS = 20            # total number of azimuth bins that are used for pruning in prune sector which is 2x ISOCHRONE_PRUNE_SECTOR_DEG_HALF : put even number !

##
# boat settings
DEFAULT_BOAT = os.getenv('BOAT_FILE')   # path to data for sailing boat (not maintained)

##
CMEMS_USER = os.getenv('CMEMS_USER')
CMEMS_PASSWORD = os.getenv('CMEMS_PASSWORD')
