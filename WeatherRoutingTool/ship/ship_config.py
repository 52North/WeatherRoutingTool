from pydantic import BaseModel, ValidationError, field_validator
from pathlib import Path
import json
import logging

logger = logging.getLogger('WRT.ShipConfig')


class ShipConfig(BaseModel):

    # Filepaths
    WEATHER_DATA: Path  # path to weather data
    DEPTH_DATA: Path = None  # path to depth data
    COURSES_FILE: Path = None  # path to file that acts as intermediate storage for courses per routing step

    # Mandatory configuration
    BOAT_BREADTH: float  # ship breadth [m]
    BOAT_FUEL_RATE: float  # fuel rate at service propulsion point [g/kWh]
    BOAT_HBR: float  # height of top of superstructure (bridge etc.) [m]
    BOAT_LENGTH: float  # overall length [m]
    BOAT_SMCR_POWER: float  # Specific Maximum Continuous Rating power [kWh]
    BOAT_SMCR_SPEED: float
    BOAT_SPEED: float  # boat speed [m/s]

    # Recommended configuration
    BOAT_ROUGHNESS_DISTRIBUTION_LEVEL: float = 1  # numeric value
    BOAT_ROUGHNESS_LEVEL: float = 1.  # level of hull roughness, numeric value

    # Optional configuration
    AIR_MASS_DENSITY: float = 1.2225  # mass density of air [kg/m^3]
    BOAT_AOD: float = -99  # lateral projected area of superstructures etc. on deck [m]
    BOAT_AXV: float = -99  # area of maximum transverse section exposed to the winds [m]
    BOAT_AYV: float = -99  # projected lateral area above the waterline [m]
    BOAT_BS1: float = -99  # breadth of substructure 1 [m]
    BOAT_CMC: float = -99  # horizontal distance from midship section to centre of lateral projected area AYV [m]
    BOAT_DRAUGHT_AFT: float = 10  # aft draught (draught at rudder) in m
    BOAT_DRAUGHT_FORE: float = 10  # fore draught (draught at forward perpendicular) in m
    BOAT_HC: float = -99  # height of waterline to centre of lateral projected area Ayv [m]
    BOAT_HS1: float = -99  # height of substructure 1 [m]
    BOAT_HS2: float = -99  # height of substructure 2 [m]
    BOAT_LS1: float = -99  # length of substructure 1 [m]
    BOAT_LS2: float = -99  # length of substructure 2 [m]
    BOAT_OVERLOAD_FACTOR: float = 0
    BOAT_PROPULSION_EFFICIENCY: float = 0.63  # propulsion efficiency coefficient in ideal conditions;
    # assuming n_H = 1.05 n_0 = 0.1 n_R = 1
    BOAT_FACTOR_CALM_WATER: float = 1.0  # multiplication factor for the calm water resistance model of maripower
    BOAT_FACTOR_WAVE_FORCES: float = 1.0  # multiplication factor for added resistance in waves model of maripower
    BOAT_FACTOR_WIND_FORCES: float = 1.0  # multiplication factor for the added resistance in wind model of maripower
    BOAT_UNDER_KEEL_CLEARANCE: float = 20  # vertical distance between keel and ground

    @classmethod
    def validate_config(cls, config_data):
        try:
            config = cls(**config_data)
            logger.info("ShipConfig is valid!")
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
        if init_mode == 'from_json' and Path(path).exists:
            with path.open("r") as f:
                config_data = json.load(f)
            return cls.validate_config(config_data)
        elif init_mode == 'from_dict':
            return cls.validate_config(config_dict)
        else:
            msg = f"Init mode '{init_mode}' for config is invalid. Supported options are 'from_json' and 'from_dict'."
            logger.error(msg)
            raise ValueError(msg)

    @field_validator('BOAT_BREADTH', 'BOAT_LENGTH', 'BOAT_SMCR_POWER', 'BOAT_SPEED', mode='after')
    @classmethod
    def check_numeric_values_positivity(cls, v, info):
        if v <= 0:
            raise ValueError(f"'{info.field_name}' must be greater than zero, got {v}")
        return v

    @field_validator('BOAT_PROPULSION_EFFICIENCY')
    def check_boat_propulsion_efficiency_range(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"'BOAT_PROPULSION_EFFICIENCY' must be between 0 and 1, but got {v}.")

    @field_validator('BOAT_SPEED', mode='after')
    @classmethod
    def check_boat_speed(cls, v):
        if v > 10:
            logger.warning(
                "Your 'BOAT_SPEED' is higher than 10 m/s."
                " Have you considered that this program works with m/s?")
        return v
