import json
import logging
import os
import sys

logger = logging.getLogger('WRT.Config')

MANDATORY_CONFIG_VARIABLES = ['COURSES_FILE', 'DEFAULT_MAP', 'DEFAULT_ROUTE', 'DEPARTURE_TIME', 'DEPTH_DATA',
                              'ROUTE_PATH', 'WEATHER_DATA']

RECOMMENDED_CONFIG_VARIABLES = {
    'BOAT_DRAUGHT_AFT': 10,
    'BOAT_DRAUGHT_FORE': 10,
    'BOAT_ROUGHNESS_DISTRIBUTION_LEVEL': 1,
    'BOAT_ROUGHNESS_LEVEL': 1,
    'BOAT_SPEED': 6,
    'DATA_MODE': 'automatic'
}

# optional variables with default values
OPTIONAL_CONFIG_VARIABLES = {
    'ALGORITHM_TYPE': 'isofuel',
    'CONSTANT_FUEL_RATE': None,
    'CONSTRAINTS_LIST': ['land_crossing_global_land_mask', 'water_depth', 'on_map'],
    'DELTA_FUEL': 3000,
    'DELTA_TIME_FORECAST': 3,
    'FACTOR_CALM_WATER': 1.0,
    'FACTOR_WAVE_FORCES': 1.0,
    'FACTOR_WIND_FORCES': 1.0,
    'GENETIC_MUTATION_TYPE': 'grid_based',
    'GENETIC_NUMBER_GENERATIONS': 20,
    'GENETIC_NUMBER_OFFSPRINGS': 2,
    'GENETIC_POPULATION_SIZE': 20,
    'GENETIC_POPULATION_TYPE': 'grid_based',
    'INTERMEDIATE_WAYPOINTS': [],
    'ISOCHRONE_MAX_ROUTING_STEPS': 100,
    'ISOCHRONE_MINIMISATION_CRITERION': 'squareddist_over_disttodest',
    'ISOCHRONE_NUMBER_OF_ROUTES': 1,
    'ISOCHRONE_PRUNE_GROUPS': 'larger_direction',
    'ISOCHRONE_PRUNE_SYMMETRY_AXIS': 'gcr',
    'ISOCHRONE_PRUNE_SECTOR_DEG_HALF': 91,
    'ISOCHRONE_PRUNE_SEGMENTS': 20,
    'ROUTER_HDGS_INCREMENTS_DEG': 6,
    'ROUTER_HDGS_SEGMENTS': 30,
    'ROUTE_POSTPROCESSING': False,
    'SHIP_TYPE': 'CBT',
    'TIME_FORECAST': 90,
    'UNDER_KEEL_CLEARANCE': 20
}


class RequiredConfigError(RuntimeError):
    pass


class Config:

    def __init__(self, init_mode='from_json', file_name=None, config_dict=None):
        # Details in README
        self.ALGORITHM_TYPE = None  # options: 'isofuel', 'genetic', 'speedy_isobased'
        self.BOAT_DRAUGHT_AFT = None  # aft draught (draught at rudder) in m
        self.BOAT_DRAUGHT_FORE = None  # fore draught (draught at forward perpendicular) in m
        self.BOAT_ROUGHNESS_DISTRIBUTION_LEVEL = None  # numeric value
        self.BOAT_ROUGHNESS_LEVEL = None  # level of hull roughness, numeric value
        self.BOAT_SPEED = None  # in m/s
        self.CONSTRAINTS_LIST = None  # options: 'land_crossing_global_land_mask', 'land_crossing_polygons', 'seamarks',
        # 'water_depth', 'on_map', 'via_waypoints', 'status_error'
        self.CONSTANT_FUEL_RATE = None  # constant fuel rate passed from ConstantFuelBoat for 'speedy_isobased'
        self.COURSES_FILE = None  # path to file that acts as intermediate storage for courses per routing step
        self.DATA_MODE = None  # options: 'automatic', 'from_file', 'odc'
        self.DEFAULT_MAP = None  # bbox in which route optimization is performed (lat_min, lon_min, lat_max, lon_max)
        self.DEFAULT_ROUTE = None  # start and end point of the route (lat_start, lon_start, lat_end, lon_end)
        self.DELTA_FUEL = None  # amount of fuel per routing step (kg)
        self.DELTA_TIME_FORECAST = None  # time resolution of weather forecast (hours)
        self.DEPARTURE_TIME = None  # start time of travelling, format: 'yyyy-mm-ddThh:mmZ'
        self.DEPTH_DATA = None  # path to depth data
        self.FACTOR_CALM_WATER = None  # multiplication factor for the calm water resistance model
        self.FACTOR_WAVE_FORCES = None  # multiplication factor for the added resistance in waves model
        self.FACTOR_WIND_FORCES = None  # multiplication factor for the added resistance in wind model
        self.GENETIC_MUTATION_TYPE = None  # type for mutation (options: 'grid_based')
        self.GENETIC_NUMBER_GENERATIONS = None  # number of generations for genetic algorithm
        self.GENETIC_NUMBER_OFFSPRINGS = None  # number of offsprings for genetic algorithm
        self.GENETIC_POPULATION_SIZE = None  # population size for genetic algorithm
        self.GENETIC_POPULATION_TYPE = None  # type for initial population (options: 'grid_based', 'from_geojson')
        self.INTERMEDIATE_WAYPOINTS = None  # [[lat_one,lon_one], [lat_two,lon_two] ... ]
        self.ISOCHRONE_MAX_ROUTING_STEPS = None  # maximum number of routing steps
        self.ISOCHRONE_MINIMISATION_CRITERION = None  # options: 'dist', 'squareddist_over_disttodest'
        self.ISOCHRONE_NUMBER_OF_ROUTES = None  # integer specifying how many routes should be searched
        self.ISOCHRONE_PRUNE_GROUPS = None  # can be 'courses', 'larger_direction', 'branch'
        self.ISOCHRONE_PRUNE_SECTOR_DEG_HALF = None  # half of the angular range of azimuth angle considered for pruning; not used for branch-based pruning
        self.ISOCHRONE_PRUNE_SEGMENTS = None  # total number of azimuth bins used for pruning in prune sector; not used for branch-based pruning
        self.ISOCHRONE_PRUNE_SYMMETRY_AXIS = None  # symmetry axis for pruning. Can be 'gcr' or 'headings_based'; not used for branch-based pruning
        self.ROUTER_HDGS_INCREMENTS_DEG = None  # increment of headings
        self.ROUTER_HDGS_SEGMENTS = None  # total number of headings (put even number!!)
        self.ROUTE_PATH = None  # path to json file to which the route will be written
        self.ROUTE_POSTPROCESSING = None  # Route is postprocessed with Traffic Separation Scheme
        self.SHIP_TYPE = None  # options: 'CBT', 'SAL'
        self.UNDER_KEEL_CLEARANCE = None  # vertical distance between keel and ground
        self.TIME_FORECAST = None  # forecast hours weather
        self.WEATHER_DATA = None  # path to weather data

        if init_mode == 'from_json':
            assert file_name
            self.read_from_json(file_name)
        elif init_mode == 'from_dict':
            assert config_dict
            self.read_from_dict(config_dict)
        else:
            msg = f"Init mode '{init_mode}' for config is invalid. Supported options are 'from_json' and 'from_dict'."
            logger.error(msg)
            raise ValueError(msg)

        if os.getenv('CMEMS_USERNAME') is None or os.getenv('CMEMS_PASSWORD') is None:
            logger.warning("No CMEMS username and/or password provided! Note that these are necessary if "
                           "DATA_MODE='automatic' used.")

        if (os.getenv('WRT_DB_HOST') is None or os.getenv('WRT_DB_PORT') is None or os.getenv('WRT_DB_DATABASE')
                is None or os.getenv('WRT_DB_USERNAME') is None or os.getenv('WRT_DB_PASSWORD') is None):
            logger.warning("No complete database configuration provided! Note that it is needed for some "
                           "constraint modules.")

        if self.ALGORITHM_TYPE == 'speedy_isobased':
            if self.CONSTANT_FUEL_RATE is None:
                raise ValueError('Need to initialise CONSTANT_FUEL_RATE when running with algorithm type '
                                 'speedy_isobased')
            figurepath = os.getenv('WRT_FIGURE_PATH')
            if figurepath and os.path.isdir(figurepath) and os.access(figurepath, os.W_OK):
                logger.warning("For speedy execution of isobased algorithms, figures should be deactivated")
            if self.ISOCHRONE_PRUNE_SEGMENTS < 100:
                logger.warning("For speedy execution of isobased algorithms, it is more sensible to set a high"
                               " number of prune segments (e.g. 200). ")

    def print(self):
        # ToDo: prettify output
        logger.info(f"Config variables: \n{json.dumps(self.__dict__, indent=4)}")

    def read_from_dict(self, config_dict):
        self._set_mandatory_config(config_dict)
        self._set_recommended_config(config_dict)
        self._set_optional_config(config_dict)

    def read_from_json(self, json_file):
        with open(json_file) as f:
            config_dict = json.load(f)
            self.read_from_dict(config_dict)

    def _set_mandatory_config(self, config_dict):
        for mandatory_var in MANDATORY_CONFIG_VARIABLES:
            if mandatory_var not in config_dict.keys():
                raise RequiredConfigError(f"'{mandatory_var}' is mandatory!")
            else:
                setattr(self, mandatory_var, config_dict[mandatory_var])

    def _set_recommended_config(self, config_dict):
        for recommended_var, default_value in RECOMMENDED_CONFIG_VARIABLES.items():
            if recommended_var not in config_dict.keys():
                logger.warning(f"'{recommended_var}' was not provided in the config and is set to the default value")
                setattr(self, recommended_var, default_value)
            else:
                setattr(self, recommended_var, config_dict[recommended_var])

    def _set_optional_config(self, config_dict):
        for optional_var, default_value in OPTIONAL_CONFIG_VARIABLES.items():
            if optional_var not in config_dict.keys():
                setattr(self, optional_var, default_value)
            else:
                setattr(self, optional_var, config_dict[optional_var])


def set_up_logging(info_log_file=None, warnings_log_file=None, debug=False, stream=sys.stdout,
                   log_format='%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s'):
    formatter = logging.Formatter(log_format)
    logging.basicConfig(stream=stream, format=log_format)
    logger = logging.getLogger('WRT')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if info_log_file:
        if os.path.isdir(os.path.dirname(info_log_file)):
            fh_info = logging.FileHandler(info_log_file, mode='w')
            fh_info.setLevel(logging.INFO)
            fh_info.setFormatter(formatter)
            logger.addHandler(fh_info)
        else:
            logger.warning(f"Logging file '{info_log_file}' doesn't exist and cannot be created.")
    if warnings_log_file:
        if os.path.isdir(os.path.dirname(warnings_log_file)):
            fh_warnings = logging.FileHandler(warnings_log_file, mode='w')
            fh_warnings.setLevel(logging.WARNING)
            fh_warnings.setFormatter(formatter)
            logger.addHandler(fh_warnings)
        else:
            logger.warning(f"Logging file '{warnings_log_file}' doesn't exist and cannot be created.")
    return logger
