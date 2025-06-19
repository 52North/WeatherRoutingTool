from pydantic import BaseModel, Field, conlist, ValidationError
from typing import Optional, List, Annotated, Union, Literal
from datetime import datetime
from pathlib import Path
import json

class ShipConfigModel(BaseModel):

    @classmethod
    def validate_config(cls, path: Path):
      try:
          with path.open( "r") as f:
              data = json.load(f)
          config = cls(**data)
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
      
    # Filepaths
    WEATHER_DATA: Path
    DEPTH_DATA: Path= None
    COURSES_FILE: Path = None

    # Mandatory configuration
    BOAT_BREADTH: float
    BOAT_FUEL_RATE: float
    BOAT_HBR: float
    BOAT_LENGTH: float
    BOAT_SMCR_POWER: float
    BOAT_SPEED: float

    # Recommended configuration
    BOAT_ROUGHNESS_DISTRIBUTION_LEVEL: float = 1
    BOAT_ROUGHNESS_LEVEL: float = 1.

    # Optional configuration
    AIR_MASS_DENSITY: float = 1.2225
    BOAT_AOD: float = -99
    BOAT_AXV: float = -99
    BOAT_AYV: float = -99
    BOAT_BS1: float = -99
    BOAT_CMC: float = -99
    BOAT_DRAUGHT_AFT: float = 10
    BOAT_DRAUGHT_FORE: float = 10
    BOAT_HC: float = -99
    BOAT_HS1: float = -99
    BOAT_HS2: float = -99
    BOAT_LS1: float = -99
    BOAT_LS2: float = -99
    BOAT_OVERLOAD_FACTOR: float = 0
    BOAT_PROPULSION_EFFICIENCY: float = 0.63  # assuming n_H = 1.05 n_0 = 0.1 n_R = 1
    BOAT_FACTOR_CALM_WATER: float = 1.0
    BOAT_FACTOR_WAVE_FORCES: float = 1.0
    BOAT_FACTOR_WIND_FORCES: float = 1.0
    BOAT_UNDER_KEEL_CLEARANCE: float = 20
