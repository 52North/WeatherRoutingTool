import os
from dotenv import load_dotenv

load_dotenv()

##
# Defaults
DEFAULT_MAP = [48.327039,-2.768555, 52.032218, 3.801270] # med sea
#DEFAULT_MAP = [51, 1.2,60,12]
#DEFAULT_ROUTE = [51.121667, 1.355833, 57.56141, 11.65856]
DEFAULT_ROUTE = [50.5545, 0.2678, 50.7868, 0.9778]
TIME_FORECAST = 40              # forecast hours weather
ROUTING_STEPS = 20             # number of routing steps
DELTA_TIME_FORECAST = 3600      # time resolution of weather forecast (seconds)
#DELTA_FUEL = 30000000*1000*1 # [Ws]
DELTA_FUEL = 0.1*1000             # amount of fuel per routing step (kg)
START_TIME = '2023041709'       # start time of travelling
BOAT_SPEED = 20                 # (m/s)
BOAT_DROUGHT = 10                 # (m)

##
# File paths
WEATHER_DATA = os.environ['WEATHER_DATA']   # path to weather data
DEPTH_DATA = os.environ['DEPTH_DATA']       # path to depth data
PERFORMANCE_LOG_FILE = os.environ['PERFORMANCE_LOG_FILE']   # path to log file which logs performance
INFO_LOG_FILE = os.environ['INFO_LOG_FILE'] # path to log file which logs information
FIGURE_PATH = os.environ['FIGURE_PATH']     # path to figure repository
COURSES_FILE = os.environ['BASE_PATH'] + '/CoursesRoute.nc'     # path to file that acts as intermediate storage for courses per routing step
ROUTE_PATH = os.environ['ROUTE_PATH']

##
# Database paramteters
HOST = os.environ['HOST']
DATABASE =  os.environ['DATABASE']
MYUSERNAME =  os.environ['MYUSERNAME']
PASSWORD =  os.environ['PASSWORD']
PORT =  os.environ['PORT']


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
DEFAULT_BOAT = os.environ['BOAT_FILE']   # path to data for sailing boat (not maintained)

