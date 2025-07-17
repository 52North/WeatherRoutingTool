from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, PrivateAttr
from typing import Optional, List, Annotated, Union, Literal
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import os
import sys
import xarray as xr
import pandas as pd

logger = logging.getLogger('WRT.Config')


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


class Config(BaseModel):

    # Filepaths
    COURSES_FILE: str
    # path to the folder where the file that acts as intermediate storage for courses per routing step can be written
    DEPTH_DATA: str = None  # path to depth data
    WEATHER_DATA: str = None  # path to weather data
    ROUTE_PATH: str  # path to the folder where the json file with the route will be written
    CONFIG_PATH: str = None  # path to config file

    # Other configuration
    ALGORITHM_TYPE: Literal['isofuel', 'genetic', 'speedy_isobased'] = 'isofuel'
    # options: 'isofuel', 'genetic', 'speedy_isobased'

    BOAT_BREADTH: float  # ship breadth [m]
    BOAT_DRAUGHT_AFT: float = 10  # aft draught (draught at rudder) in m
    BOAT_DRAUGHT_FORE: float = 10  # fore draught (draught at forward perpendicular) in m
    BOAT_FUEL_RATE: float  # fuel rate at service propulsion point [g/kWh]
    BOAT_HBR: float  # height of top of superstructure (bridge etc.) [m]
    BOAT_LENGTH: float  # overall length [m]
    BOAT_SMCR_POWER: float  # Specific Maximum Continuous Rating power [kWh]
    BOAT_SPEED: float  # boat speed [m/s]
    BOAT_TYPE: Literal['CBT', 'SAL', 'speedy_isobased', 'direct_power_method'] = 'direct_power_method'
    # options: 'CBT', 'SAL','speedy_isobased', 'direct_power_method

    CONSTRAINTS_LIST: List[Literal[
        'land_crossing_global_land_mask', 'land_crossing_polygons', 'seamarks',
        'water_depth', 'on_map', 'via_waypoints', 'status_error'
        ]]
    # options: 'land_crossing_global_land_mask', 'land_crossing_polygons',
    # 'seamarks','water_depth', 'on_map', 'via_waypoints', 'status_error'
    CONSTANT_FUEL_RATE: float = 0.1  # wo wird das benutzt?

    _DATA_MODE_DEPTH: str = PrivateAttr('from_file')  # options: 'automatic', 'from_file', 'odc'
    _DATA_MODE_WEATHER: str = PrivateAttr('from_file')  # options: 'automatic', 'from_file', 'odc'
    DEFAULT_ROUTE: Annotated[list[Union[int, float]], Field(min_length=4, max_length=4, default_factory=list)]
    # start and end point of the route (lat_start, lon_start, lat_end, lon_end)
    DEFAULT_MAP: Annotated[list[Union[int, float]], Field(min_length=4, max_length=4, default_factory=list)]
    # bbox in which route optimization is performed (lat_min, lon_min, lat_max, lon_max)
    DELTA_FUEL: float = 3000  # amount of fuel per routing step (kg)
    DELTA_TIME_FORECAST: float = 3  # time resolution of weather forecast (hours)
    DEPARTURE_TIME: datetime  # start time of travelling, format: 'yyyy-mm-ddThh:mmZ'

    GENETIC_MUTATION_TYPE: Literal['grid_based'] = 'grid_based'  # type for mutation (options: 'grid_based')
    GENETIC_NUMBER_GENERATIONS: int = 20  # number of generations for genetic algorithm
    GENETIC_NUMBER_OFFSPRINGS: int = 2  # number of offsprings for genetic algorithm
    GENETIC_POPULATION_SIZE: int = 20  # population size for genetic algorithm
    GENETIC_POPULATION_TYPE: Literal['grid_based', 'from_geojson'] = 'grid_based'  # type for initial population
    # (options: 'grid_based', 'from_geojson')

    INTERMEDIATE_WAYPOINTS: Annotated[
      list[Annotated[list[Union[int, float]], Field(min_length=2, max_length=2)]],
      Field(default_factory=list)]  # [[lat_one,lon_one], [lat_two,lon_two] ... ]
    ISOCHRONE_MAX_ROUTING_STEPS: int = 100  # maximum number of routing steps
    ISOCHRONE_MINIMISATION_CRITERION: Literal['dist', 'squareddist_over_disttodest'] = 'squareddist_over_disttodest'
    # options: 'dist', 'squareddist_over_disttodest'
    ISOCHRONE_NUMBER_OF_ROUTES: int = 1  # integer specifying how many routes should be searched
    ISOCHRONE_PRUNE_BEARING: bool = False
    ISOCHRONE_PRUNE_GCR_CENTERED: bool = True
    ISOCHRONE_PRUNE_GROUPS: Literal['courses', 'larger_direction', 'branch'] = 'larger_direction'  # can be 'courses',
    # 'larger_direction', 'branch'
    ISOCHRONE_PRUNE_SECTOR_DEG_HALF: int = 91  # half of the angular range of azimuth angle considered for pruning;
    # not used for branch-based pruning  # noqa: E501
    ISOCHRONE_PRUNE_SEGMENTS: int = 20  # total number of azimuth bins used for pruning in prune sector;
    # not used for branch-based pruning  # noqa: E501
    ISOCHRONE_PRUNE_SYMMETRY_AXIS: Literal['gcr', 'headings_based'] = 'gcr'  # symmetry axis for pruning.
    # Can be 'gcr' or 'headings_based'; not used for branch-based pruning  # noqa: E501

    ROUTER_HDGS_SEGMENTS: int = 30  # total number of headings (put even number!!)
    ROUTER_HDGS_INCREMENTS_DEG: int = 6  # increment of headings
    ROUTE_POSTPROCESSING: bool = False  # Route is postprocessed with Traffic Separation Scheme
    ROUTING_STEPS: int = 60

    TIME_FORECAST: float = 90  # forecast hours weather

    @classmethod
    def validate_config(cls, config_data):
        try:
            config = cls(**config_data)
            logger.info("Config is valid!")
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
        if init_mode == 'from_json': 
            if Path(path).exists:
                with path.open("r") as f:
                    config_data = json.load(f)
                    config = cls.validate_config(config_data)
                    config.CONFIG_PATH = path
                return config
            else: 
                raise ValueError("Path doesn't exist: CONFIG_PATH")
        elif init_mode == 'from_dict':
            if config_dict != None:
                return cls.validate_config(config_dict)
            else:
                raise ValueError("You chose init_mode = 'from_dict' but config_dict = None")
        else:
            msg = f"Init mode '{init_mode}' for config is invalid. Supported options are 'from_json' and 'from_dict'."
            raise ValueError(msg)

    @field_validator('DEPARTURE_TIME', mode='before')
    def parse_and_validate_datetime(cls, v):
        try:
            dt = datetime.strptime(v, '%Y-%m-%dT%H:%MZ')
            return dt
        except ValueError:
            raise ValueError("'DEPARTURE_TIME' must be in format YYYY-MM-DDTHH:MMZ")

    @field_validator('COURSES_FILE', 'ROUTE_PATH')
    def validate_path_exists(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path doesn't exist: {v}")
        return str(path)

    @field_validator('DEFAULT_ROUTE')
    def validate_route_coordinates(cls, v):
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

    @field_validator('DEFAULT_MAP')
    def validate_map_coordinates(cls, v):
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

    @field_validator('CONSTRAINTS_LIST')
    def check_constraint_list(cls, v):
        if 'land_crossing_global_land_mask' not in v:
            raise ValueError("'land_crossing_global_land_mask' must be included in 'CONSTRAINTS_LIST'.")
        return v

    @field_validator('BOAT_SPEED')
    def check_boat_speed(cls, v):
        if v > 10:
            logger.warning("Your 'BOAT_SPEED' is higher than 10 m/s."
                        " Have you considered that this program works with m/s?")
        return v

    @field_validator('DELTA_FUEL', 'TIME_FORECAST', 'ROUTER_HDGS_INCREMENTS_DEG',
                     'ISOCHRONE_MAX_ROUTING_STEPS', mode='after')
    @classmethod
    def check_numeric_values_positivity(cls, v, info):
        if v <= 0:
            raise ValueError(f"'{info.field_name}' must be greater than zero, got {v}")
        return v

    @field_validator('ROUTER_HDGS_SEGMENTS')
    def check_router_hdgs_segments_positive_and_even(cls, v):
        if not (v > 0 and v % 2 == 0):
            raise ValueError("'ROUTER_HDGS_SEGMENTS' must be a positive even integer.")
        return v

    @model_validator(mode='after')
    def check_route_on_map(self) -> 'Config':
        lat_min, lon_min, lat_max, lon_max = self.DEFAULT_MAP
        lat_start, lon_start, lat_end, lon_end = self.DEFAULT_ROUTE

        for lat, lon, name in [(lat_start, lon_start, "start"), (lat_end, lon_end, "end")]:
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                raise ValueError(f"{name} point ({lat}, {lon}) is outside the defined map bounds")

        return self

    @model_validator(mode='after')
    def check_boat_algorithm_compatibility(self) -> 'Config':
        if (
            (self.BOAT_TYPE == 'speedy_isobased' or self.ALGORITHM_TYPE == 'speedy_isobased')
            and self.BOAT_TYPE != self.ALGORITHM_TYPE
        ):
            raise ValueError("If 'BOAT_TYPE' or 'ALGORITHM_TYPE' is 'speedy_isobased', so must be the other one.")
        return self

    @model_validator(mode='after')
    def check_route_weather_data_compatibility(self) -> 'Config':
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
    def check_route_depth_data_compatibility(self) -> 'Config':
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
