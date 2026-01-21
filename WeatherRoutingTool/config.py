import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path, PosixPath
from typing import Annotated, List, Literal, Optional, Self, Union

import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr, ValidationError, ValidationInfo

logger = logging.getLogger('WRT.Config')


def set_up_logging(info_log_file=None, warnings_log_file=None, debug=False, stream=sys.stdout,
                   log_format='%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s'):
    formatter = logging.Formatter(log_format)
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    logging.basicConfig(stream=stream, format=log_format, level=log_level)
    logger = logging.getLogger('WRT')

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


class Config(BaseModel):
    """
    Central configuration class of the Weather Routing Tool.
    Parameters are validated using pydantic.
    """

    # !!! IMPORTANT NOTE !!!
    # The order of the attributes matters. Attributes are validated in the order of their appearance.
    # This means for any attribute it is only possible to access those other attributes in a field validator
    # (via the ValidationInfo object) which have been declared earlier.

    # Other configuration
    ALGORITHM_TYPE: Literal[
        'dijkstra', 'gcr_slider', 'genetic', 'genetic_shortest_route', 'isofuel', 'speedy_isobased'
    ] = 'isofuel'
    ARRIVAL_TIME: datetime = '9999-99-99T99:99Z'  # arrival time at destination, format: 'yyyy-mm-ddThh:mmZ'

    BOAT_TYPE: Literal['CBT', 'SAL', 'speedy_isobased', 'direct_power_method'] = 'direct_power_method'
    BOAT_SPEED: float = -99.  # boat speed [m/s]
    CONSTRAINTS_LIST: List[Literal[
        'land_crossing_global_land_mask', 'land_crossing_polygons', 'seamarks',
        'water_depth', 'on_map', 'via_waypoints', 'status_error'
    ]]
    # options: 'land_crossing_global_land_mask', 'land_crossing_polygons',
    # 'seamarks','water_depth', 'on_map', 'via_waypoints', 'status_error'

    _DATA_MODE_DEPTH: str = PrivateAttr('from_file')  # options: 'automatic', 'from_file', 'odc'
    _DATA_MODE_WEATHER: str = PrivateAttr('from_file')  # options: 'automatic', 'from_file', 'odc'
    DEFAULT_ROUTE: Annotated[list[Union[int, float]], Field(min_length=4, max_length=4)]
    # start and end point of the route (lat_start, lon_start, lat_end, lon_end)
    DEFAULT_MAP: Annotated[list[Union[int, float]], Field(min_length=4, max_length=4)]
    # bbox in which route optimization is performed (lat_min, lon_min, lat_max, lon_max)
    DELTA_FUEL: float = 3000  # amount of fuel per routing step (kg)
    DELTA_TIME_FORECAST: float = 3  # time resolution of weather forecast (hours)
    DEPARTURE_TIME: datetime  # start time of travelling, format: 'yyyy-mm-ddThh:mmZ'

    # options for Dijkstra algorithm
    DIJKSTRA_MASK_FILE: str = None  # can be found with "find ~ -type f -name globe_combined_mask_compressed.npz"
    # or downloaded via https://github.com/toddkarin/global-land-mask/blob/master/global_land_mask/globe_combined_mask_compressed.npz  # noqa: E501
    DIJKSTRA_NOF_NEIGHBORS: int = 1  # number of neighbors to use when creating a graph from the grid
    DIJKSTRA_STEP: int = 1  # step used to save final route to prevent very dense waypoints

    # options for GCR Slider algorithm
    GCR_SLIDER_ANGLE_STEP: float = 30  # in degrees
    GCR_SLIDER_DISTANCE_MOVE: float = 10000  # in m
    GCR_SLIDER_DYNAMIC_PARAMETERS: bool = True
    GCR_SLIDER_LAND_BUFFER: float = 1000  # in m
    GCR_SLIDER_INTERPOLATE: bool = True
    GCR_SLIDER_INTERP_DIST: float = 0.1
    GCR_SLIDER_INTERP_NORMALIZED: bool = True
    GCR_SLIDER_MAX_POINTS: int = 300
    GCR_SLIDER_THRESHOLD: float = 10000  # in m

    # options for Genetic Algorithm
    GENETIC_NUMBER_GENERATIONS: int = 20  # number of generations
    GENETIC_NUMBER_OFFSPRINGS: int = 2  # total number of offsprings for every generation
    GENETIC_POPULATION_SIZE: int = 20  # population size for genetic algorithm
    GENETIC_POPULATION_TYPE: Literal[
        'grid_based', 'from_geojson', 'isofuel', 'gcrslider'] = 'grid_based'  # type for initial population  # noqa: E501
    GENETIC_POPULATION_PATH: Optional[str] = None  # path to initial population
    GENETIC_REPAIR_TYPE: List[Literal[
        'waypoints_infill', 'constraint_violation', 'no_repair'
    ]] = ["waypoints_infill", "constraint_violation"]
    GENETIC_MUTATION_TYPE: Literal[
        'random', 'rndm_walk', 'rndm_plateau', 'route_blend', 'no_mutation'
    ] = 'random'
    GENETIC_CROSSOVER_PATCHER: Literal['gcr', 'isofuel'] = 'isofuel'
    GENETIC_FIX_RANDOM_SEED: bool = False
    GENETIC_OBJECTIVES: List[Literal[
        'arrival_time', 'fuel_consumption'
    ]] = ["fuel_consumption"]

    INTERMEDIATE_WAYPOINTS: Annotated[
        list[Annotated[list[Union[int, float]], Field(min_length=2, max_length=2)]],
        Field(default_factory=list)]  # [[lat_one,lon_one], [lat_two,lon_two] ... ]

    # options for isobased algorithms
    ISOCHRONE_MAX_ROUTING_STEPS: int = 100  # maximum number of routing steps
    ISOCHRONE_MINIMISATION_CRITERION: Literal['dist', 'squareddist_over_disttodest'] = 'squareddist_over_disttodest'
    # options: 'dist', 'squareddist_over_disttodest'
    ISOCHRONE_NUMBER_OF_ROUTES: int = 1  # integer specifying how many routes should be searched
    ISOCHRONE_PRUNE_GROUPS: Literal[
        'courses', 'larger_direction', 'branch', 'multiple_routes'] = 'larger_direction'
    ISOCHRONE_PRUNE_SECTOR_DEG_HALF: int = 91  # half of the angular range of azimuth angle considered for pruning;
    # not used for branch-based pruning  # noqa: E501
    ISOCHRONE_PRUNE_SEGMENTS: int = 20  # total number of azimuth bins used for pruning in prune sector;
    # not used for branch-based pruning  # noqa: E501
    ISOCHRONE_PRUNE_SYMMETRY_AXIS: Literal['gcr', 'headings_based'] = 'gcr'  # symmetry axis for pruning.
    # Can be 'gcr' or 'headings_based'; not used for branch-based pruning  # noqa: E501

    ROUTER_HDGS_SEGMENTS: int = 30  # total number of headings (put even number!!)
    ROUTER_HDGS_INCREMENTS_DEG: int = 6  # increment of headings
    ROUTE_POSTPROCESSING: bool = False  # route is postprocessed with Traffic Separation Scheme
    ROUTING_STEPS: int = 60

    TIME_FORECAST: float = 90  # forecast hours weather

    # Filepaths
    COURSES_FILE: str = None  # needs to be declared after BOAT_TYPE
    # path to file that acts as intermediate storage for courses per routing step
    DEPTH_DATA: str = None  # path to depth data
    WEATHER_DATA: str = None  # path to weather data
    ROUTE_PATH: str  # path to the folder where the json file with the route will be written
    CONFIG_PATH: Union[str, PosixPath] = None  # path to config file

    @classmethod
    def validate_config(cls, config_data):
        """
        Validate the config by creating a Config class object and throw errors if necessary

        :param config_data: Config data provided by the user in form of a json file or a dict
        :type config_data: dict
        :return: Validated config
        :rtype: WeatherRoutingTool.config.Config
        """
        try:
            config = cls(**config_data)
            logger.debug("Config is valid!")
            return config
        except ValidationError as e:
            for err in e.errors():
                logger.info("Config-Validation failed:")
                loc = err['loc']
                loc_str = f"'{loc[0]}'" if loc else "<model-level>"
                logger.info(f" Field: {loc_str},  Error: {err['msg']}")
                raise
        except Exception as e:
            logger.info(f"Could not read config file: {e}")
            raise

    @classmethod
    def assign_config(cls, path=None, init_mode='from_json', config_dict=None):
        """
        Check input type of config data and run validate_config

        :param path: path to json with config data, defaults to None
        :type path: str, optional
        :param init_mode: _description_, defaults to 'from_json'
        :type init_mode: str, optional
        :param config_dict: dict with config data, defaults to None
        :type config_dict: dict, optional
        :raises ValueError: Path to json file doesn't exist although chosen as input type for config
        :raises ValueError: Dict doesn't exist although chosen as input type for config
        :raises ValueError: Mode chosen as input type for config doesn't exist
        :return: Validated config
        :rtype: WeatherRoutingTool.config.Config
        """

        if init_mode == 'from_json':
            if Path(path).exists:
                with path.open("r") as f:
                    config_data = json.load(f)
                    config = cls.validate_config(config_data)
                    config.CONFIG_PATH = path
                return config
            else:
                logger.info(f"Given path to config json file: {path}")
                raise ValueError("Path to config doesn't exist")
        elif init_mode == 'from_dict':
            if config_dict is not None:
                return cls.validate_config(config_dict)
            else:
                raise ValueError("You chose init_mode = 'from_dict' but config_dict has no value")
        else:
            msg = f"Init mode '{init_mode}' for config is invalid. Supported options are 'from_json' and 'from_dict'."
            raise ValueError(msg)

    @field_validator('DEPARTURE_TIME', 'ARRIVAL_TIME', mode='before')
    @classmethod
    def parse_and_validate_datetime(cls, v):
        if isinstance(v, datetime):
            return v

        try:
            dt = datetime.strptime(v, '%Y-%m-%dT%H:%MZ')
            return dt
        except ValueError:
            raise ValueError("'DEPARTURE_TIME' must be in format YYYY-MM-DDTHH:MMZ")

    @field_validator('COURSES_FILE', 'ROUTE_PATH', 'DIJKSTRA_MASK_FILE', mode='after')
    @classmethod
    def validate_path_exists(cls, v, info: ValidationInfo):
        if info.field_name == 'COURSES_FILE':
            if info.data.get('BOAT_TYPE') != 'CBT':
                return v
            else:
                path = Path(os.path.dirname(v))
        elif info.field_name == 'DIJKSTRA_MASK_FILE':
            if info.data.get('ALGORITHM_TYPE') != 'dijkstra':
                return v
            else:
                path = Path(v)
        else:
            path = Path(v)
        if not path.exists():
            raise ValueError(f"Path doesn't exist: {path}")
        return str(path)

    @field_validator('DEFAULT_ROUTE', mode='after')
    @classmethod
    def validate_route_coordinates(cls, v):
        """
        Check if the coordinates of DEFAULT_ROUTE have values that the program can work with

        :raises ValueError: DEFAULT_ROUTE doesn't contain exactly 4 values
        :raises ValueError: One of the coordinates isn't in the required range
        :return: Validated DEFAULT_ROUTE
        :rtype: tuple
        """
        if len(v) != 4:
            raise ValueError("Coordinate list must contain exactly 4 values: [lat_start, lon_start, lat_end, lon_end]")

        lat_start, lon_start, lat_end, lon_end = v

        if not -90 <= lat_start <= 90:
            raise ValueError(f"lat_start must be between -90 and 90, got {lat_start}")
        if not -90 <= lat_end <= 90:
            raise ValueError(f"lat_end must be between -90 and 90, got {lat_end}")
        if not -180 <= lon_start <= 180:
            raise ValueError(f"lon_start must be between -180 and 180, got {lon_start}")
        if not -180 <= lon_end <= 180:
            raise ValueError(f"lon_end must be between -180 and 180, got {lon_end}")

        return v

    @field_validator('DEFAULT_MAP', mode='after')
    @classmethod
    def validate_map_coordinates(cls, v, info):
        """
        Check if the coordinates of DEFAULT_MAP have values that the program can work with

        :raises ValueError: DEFAULT_MAP doesn't contain exactly 4 values
        :raises ValueError: One of the coordinates isn't in the required range
        :return: validated DEFAULT_MAP
        :rtype: tuple
        """
        if len(v) != 4:
            raise ValueError("Coordinate list must contain exactly 4 values: [lat_min, lon_max, lat_end, lon_end]")

        lat_start, lon_start, lat_end, lon_end = v

        if not -90 <= lat_start <= 90:
            raise ValueError(f"lat_start must be between -90 and 90, got {lat_start}")
        if not -90 <= lat_end <= 90:
            raise ValueError(f"lat_end must be between -90 and 90, got {lat_end}")
        if not -180 <= lon_start <= 180:
            raise ValueError(f"lon_start must be between -180 and 180, got {lon_start}")
        if not -180 <= lon_end <= 180:
            raise ValueError(f"lon_end must be between -180 and 180, got {lon_end}")

        return v

    @field_validator('CONSTRAINTS_LIST', mode='after')
    @classmethod
    def check_constraint_list(cls, v):
        """
        Check that the CONSTRAINTS_LIST contains 'land_crossing_global_land_mask'
        else the boat would be allowed to cross land

        :raises ValueError: CONSTRAINTS_LIST doesn't contain 'land_crossing_global_land_mask'
        :return: Validated CONSTRAINTS_LIST
        :rtype: list
        """
        if 'land_crossing_global_land_mask' not in v:
            raise ValueError("'land_crossing_global_land_mask' must be included in 'CONSTRAINTS_LIST'.")
        return v

    @field_validator('DELTA_FUEL', 'TIME_FORECAST', 'ROUTER_HDGS_INCREMENTS_DEG',
                     'ISOCHRONE_MAX_ROUTING_STEPS', mode='after')
    @classmethod
    def check_numeric_values_positivity(cls, v, info: ValidationInfo):
        if v <= 0:
            raise ValueError(f"'{info.field_name}' must be greater than zero, got {v}")
        return v

    @field_validator('GENETIC_REPAIR_TYPE', mode='after')
    @classmethod
    def check_genetic_repair_type(cls, v):
        if "no_repair" in v and len(v) > 1:
            raise ValueError(f"'repair types of genetic algorithm can not be paired with 'no_repair', got {v}")
        return v

    @field_validator('ROUTER_HDGS_SEGMENTS', mode='after')
    @classmethod
    def check_router_hdgs_segments_positive_and_even(cls, v):
        if not (v > 0 and v % 2 == 0):
            raise ValueError("'ROUTER_HDGS_SEGMENTS' must be a positive even integer.")
        return v

    @model_validator(mode='after')
    def check_route_on_map(self) -> Self:
        """
        Check that the route runs inside the bounds of the defined map

        :raises ValueError: Route coordinates are outside the DEFAULT_MAP bounds
        :return: Config object with validated DEFAULT_ROUTE-DEFAULT_MAP-compatibility
        :rtype: WeatherRoutingTool.config.Config
        """
        lat_min, lon_min, lat_max, lon_max = self.DEFAULT_MAP
        lat_start, lon_start, lat_end, lon_end = self.DEFAULT_ROUTE
        for lat, lon, name in [(lat_start, lon_start, "start"), (lat_end, lon_end, "end")]:
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                raise ValueError(f"{name} point ({lat}, {lon}) is outside the defined map bounds")
        return self

    @model_validator(mode='after')
    def check_boat_algorithm_compatibility(self) -> Self:
        """
        The boat type 'speedy_isobased' is configured to run only with the corresponding
        algorithm type 'speedy_isobased'

        :raises ValueError: BOAT_TYPE is not compatible with ALGORITHM_TYPE
        :return: Config object with validated BOAT_TYPE-ALGORITHM_TYPE-compatibility
        :rtype: WeatherRoutingTool.config.Config
        """

        if self.ALGORITHM_TYPE == 'speedy_isobased' and self.BOAT_TYPE != 'speedy_isobased':
            raise ValueError("If 'ALGORITHM_TYPE' is 'speedy_isobased', 'BOAT_TYPE' has to be 'speedy_isobased'.")

        if self.ALGORITHM_TYPE == 'genetic_shortest_route' and self.BOAT_TYPE != 'speedy_isobased':
            raise ValueError(
                "If 'ALGORITHM_TYPE' is 'genetic_shortest_route', 'BOAT_TYPE' has to be 'speedy_isobased'.")

        if self.BOAT_TYPE == 'speedy_isobased' and self.ALGORITHM_TYPE != 'genetic_shortest_route' and \
                self.ALGORITHM_TYPE != 'speedy_isobased':
            raise ValueError("'BOAT_TYPE'='speedy_isobased' can only be used together with "
                             "'ALGORITHM_TYPE'='genetic_shortest_route' and 'ALGORITHM_TYPE'='speedy_isobased'.")

        return self

    @model_validator(mode='after')
    def check_route_weather_data_compatibility(self) -> Self:
        """
        Check that the route runs inside the map that has weather data available
        considering place and time
        :raises ValueError: Weather data doesn't cover map
        :raises ValueError: Weather data doesn't cover full routing time range
        :raises ValueError: Weather data has no time dimension
        :return: Config object with validated WEATHER_DATA regarding place and time
        :rtype: WeatherRoutingTool.config.Config
        """
        # The Dijkstra algorithm does not consider weather data at the moment
        if self.ALGORITHM_TYPE in ['dijkstra', 'gcr_slider']:
            return self
        path = Path(self.WEATHER_DATA)
        if path.exists():
            try:
                ds = xr.open_dataset(self.WEATHER_DATA)

                # Check lat/lon bounds
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                map_coords = self.DEFAULT_MAP
                if map_coords:
                    lat_min, lon_min, lat_max, lon_max = map_coords
                    if not (lat.min() <= lat_min <= lat.max() and
                            lat.min() <= lat_max <= lat.max() and
                            lon.min() <= lon_min <= lon.max() and
                            lon.min() <= lon_max <= lon.max()):
                        logger.info(f"Map coverage of WEATHER_DATA:[{lat.min()},{lon.min()}, {lat.max()}, {lon.max()}]")
                        logger.info(f"DEFAULT_MAP:{self.DEFAULT_MAP}")
                        raise ValueError("Weather data does not cover the map region.")

                # Check time coverage
                if 'time' in ds:
                    start = self.DEPARTURE_TIME
                    end = start + timedelta(hours=self.TIME_FORECAST)
                    times = pd.to_datetime(ds['time'].values)
                    if not (times.min() <= start <= times.max() and times.min() <= end <= times.max()):
                        logger.info("Time coverage of WEATHER_DATA: "
                                    f"[{times.min()}, {times.max()}]")
                        logger.info(f"DEPARTURE_TIME: {self.DEPARTURE_TIME}")
                        logger.info(f"Time until which a weather forecast should exist: {end}")
                        raise ValueError("Weather data does not cover the full routing time range.")
                else:
                    raise ValueError("Weather data missing time dimension.")

            except Exception as e:
                raise ValueError(f"Failed to validate weather data: {e}")
        else:
            self._DATA_MODE_WEATHER = 'automatic'
        return self

    @model_validator(mode='after')
    def check_route_depth_data_compatibility(self) -> Self:
        """
        Check that the route runs inside the map that has depth data available
        considering only place as the depth data is time independent

        :raises ValueError: Depth data doesn't cover map
        :return: Config object with validated DEPTH_DATA regarding place
        :rtype: WeatherRoutingTool.config.Config
        """
        # The Dijkstra algorithm does not consider depth data at the moment
        if self.ALGORITHM_TYPE in ['dijkstra', 'gcr_slider']:
            return self
        path = Path(self.DEPTH_DATA)
        if path.exists():
            try:
                ds = xr.open_dataset(self.DEPTH_DATA)

                # Check lat/lon bounds
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                map_coords = self.DEFAULT_MAP
                if map_coords:
                    lat_min, lon_min, lat_max, lon_max = map_coords
                    if not (lat.min() <= lat_min <= lat.max() and
                            lat.min() <= lat_max <= lat.max() and
                            lon.min() <= lon_min <= lon.max() and
                            lon.min() <= lon_max <= lon.max()):
                        raise ValueError("Depth data does not cover the map region.")

            except Exception as e:
                raise ValueError(f"Failed to validate depth data: {e}")
        else:
            self._DATA_MODE_DEPTH = 'automatic'
        return self

    @field_validator('BOAT_SPEED', mode='after')
    @classmethod
    def check_boat_speed(cls, v):
        if v > 10:
            logger.warning(
                "Your 'BOAT_SPEED' is higher than 10 m/s."
                " Have you considered that this program works with m/s?")
        return v

    #@model_validator(mode='after')
    #def check_speed_determination(self) -> Self:
    #    print('arrival time: ', self.ARRIVAL_TIME)
    #    print('speed: ', self.BOAT_SPEED)
    #    if self.ARRIVAL_TIME == '9999-99-99T99:99Z' and self.BOAT_SPEED == -99.:
    #        raise ValueError('Please specify either the boat speed or the arrival time')
    #    if not self.ARRIVAL_TIME == '9999-99-99T99:99Z' and not self.BOAT_SPEED == -99.:
    #        raise ValueError('Please specify either the boat speed or the arrival time and not both.')
    #    if not self.ARRIVAL_TIME == '9999-99-99T99:99Z' and self.ALGORITHM_TYPE != 'genetic':
    #        raise ValueError('The determination of the speed from the arrival time is only possible for the'
    #                         ' genetic algorithm')
    #    return self
