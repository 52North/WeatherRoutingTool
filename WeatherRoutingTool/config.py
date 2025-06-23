from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Optional, List, Annotated, Union, Literal
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import os
import sys

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

class ConfigModel(BaseModel):

    @classmethod
    def validate_config(cls, config_data):
        try:
            config = cls(**config_data)
            print("Config is valid!")
            return config
        except ValidationError as e:
            for err in e.errors():
                print("Config-Validation failed:")
                loc = err['loc']
                loc_str = f"'{loc[0]}'" if loc else "<model-level>"
                print(f" Field: {loc_str},  Error: {err['msg']}")
                raise
        except Exception as e:
            print(f"Could not read config file: {e}")
            raise
        
    @classmethod
    def assign_config(cls, path=None, init_mode='from_json', config_dict=None):
        if init_mode == 'from_json':
            with path.open("r") as f:
                config_data = json.load(f)
                config = cls.validate_config(config_data)
                config.CONFIG_PATH = path
            return config
        elif init_mode == 'from_dict':
            return cls.validate_config(config_dict)
        else:
            msg = f"Init mode '{init_mode}' for config is invalid. Supported options are 'from_json' and 'from_dict'."
            raise ValueError(msg)

    # Filepaths
    COURSES_FILE: str = None
    DEPTH_DATA: str
    WEATHER_DATA: str
    ROUTE_PATH: str
    CONFIG_PATH: str = None

    @field_validator('COURSES_FILE', 'DEPTH_DATA', 'WEATHER_DATA', 'ROUTE_PATH', 'CONFIG_PATH')
    def validate_path_exists(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path doesn't exist: {v}")
        return str(path)

    # Other configuration
    ALGORITHM_TYPE: Literal['isofuel', 'genetic', 'speedy_isobased'] = 'isofuel'

    BOAT_BREADTH: float
    BOAT_DRAUGHT_AFT: float = 10
    BOAT_DRAUGHT_FORE: float = 10
    BOAT_FUEL_RATE: float
    BOAT_HBR: float
    BOAT_LENGTH: float
    BOAT_SMCR_POWER: float
    BOAT_SPEED: float
    BOAT_TYPE: Literal['CBT', 'SAL', 'speedy_isobased', 'direct_power_method'] = 'direct_power_method'

    @model_validator(mode='after')
    def check_boat_algorithm_compatibility(self) -> 'ConfigModel':
        if (
            (self.BOAT_TYPE == 'speedy_isobased' or self.ALGORITHM_TYPE == 'speedy_isobased') 
            and self.BOAT_TYPE != self.ALGORITHM_TYPE
        ):
            raise ValueError("If BOAT_TYPE or ALGORITHM_TYPE is 'speedy_isobased', so must be the other one.")
        return self

    CONSTRAINTS_LIST: List[Literal['land_crossing_global_land_mask', 'land_crossing_polygons', 'seamarks',
              'water_depth', 'on_map', 'via_waypoints', 'status_error']]
    # CONSTANT_FUEL_RATE:0.1 wo wird das benutzt?

    DATA_MODE: Literal['automatic', 'from_file', 'odc'] = 'automatic'
    DEFAULT_ROUTE: Annotated[list[Union[int, float]], Field(min_length=4, max_length=4, default_factory=list)]
    DEFAULT_MAP: Annotated[list[Union[int, float]], Field(min_length=4, max_length=4, default_factory=list)]
    DELTA_FUEL: float = 3000
    DELTA_TIME_FORECAST: float = 3
    DEPARTURE_TIME: datetime

    @field_validator('DEPARTURE_TIME', mode='before')
    def parse_and_validate_datetime(cls, v):
        try:
            dt = datetime.strptime(v, '%Y-%m-%dT%H:%MZ')
            return dt
        except ValueError:
            raise ValueError("DEPARTURE_TIME must be in format YYYY-MM-DDTHH:MMZ")

    GENETIC_MUTATION_TYPE: Literal['grid_based'] = 'grid_based'
    GENETIC_NUMBER_GENERATIONS: int = 20
    GENETIC_NUMBER_OFFSPRINGS: int = 2
    GENETIC_POPULATION_SIZE: int = 20
    GENETIC_POPULATION_TYPE: Literal['grid_based', 'from_geojson'] = 'grid_based'

    INTERMEDIATE_WAYPOINTS: Annotated[
      list[Annotated[list[Union[int, float]], Field(min_length=2, max_length=2)]],
      Field(default_factory=list)]  
    ISOCHRONE_MAX_ROUTING_STEPS: int = 100   
    ISOCHRONE_MINIMISATION_CRITERION: Literal['dist', 'squareddist_over_disttodest'] = 'squareddist_over_disttodest'
    ISOCHRONE_NUMBER_OF_ROUTES: int = 1
    ISOCHRONE_PRUNE_BEARING: bool = False
    ISOCHRONE_PRUNE_GCR_CENTERED: bool = True
    ISOCHRONE_PRUNE_GROUPS: Literal['courses', 'larger_direction', 'branch'] = 'larger_direction'
    ISOCHRONE_PRUNE_SECTOR_DEG_HALF: int = 91
    ISOCHRONE_PRUNE_SEGMENTS: int = 20
    ISOCHRONE_PRUNE_SYMMETRY_AXIS: Literal['gcr', 'headings_based'] = 'gcr'

    ROUTER_HDGS_SEGMENTS: int = 30
    ROUTER_HDGS_INCREMENTS_DEG: int = 6
    ROUTE_POSTPROCESSING: bool = False
    ROUTING_STEPS: int = 60

    TIME_FORECAST: float = 90