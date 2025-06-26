import json
import logging
import os
import sys

logger = logging.getLogger('WRT.ShipConfig')

MANDATORY_CONFIG_VARIABLES = [
    'BOAT_BREADTH',
    'BOAT_FUEL_RATE',
    'BOAT_HBR',
    'BOAT_LENGTH',
    'BOAT_SMCR_POWER',
    'BOAT_SPEED',
    'BOAT_SMCR_SPEED',
    'WEATHER_DATA'
]

RECOMMENDED_CONFIG_VARIABLES = {
    'BOAT_ROUGHNESS_DISTRIBUTION_LEVEL': 1,
    'BOAT_ROUGHNESS_LEVEL': 1.,
    'DEPTH_DATA': " "
}

# optional variables with default values
OPTIONAL_CONFIG_VARIABLES = {
    'AIR_MASS_DENSITY': 1.2225,
    'BOAT_AOD': -99,
    'BOAT_AXV': -99,
    'BOAT_AYV': -99,
    'BOAT_BS1': -99,
    'BOAT_CMC': -99,
    'BOAT_DRAUGHT_AFT': 10,
    'BOAT_DRAUGHT_FORE': 10,
    'BOAT_HC': -99,
    'BOAT_HS1': -99,
    'BOAT_HS2': -99,
    'BOAT_LS1': -99,
    'BOAT_LS2': -99,
    'BOAT_OVERLOAD_FACTOR': 0,
    'BOAT_PROPULSION_EFFICIENCY': 0.63,  # assuming n_H = 1.05, n_0 = 0.1, n_R = 1
    'BOAT_FACTOR_CALM_WATER': 1.0,
    'BOAT_FACTOR_WAVE_FORCES': 1.0,
    'BOAT_FACTOR_WIND_FORCES': 1.0,
    'BOAT_UNDER_KEEL_CLEARANCE': 20,
    'COURSES_FILE': None
}


class RequiredConfigError(RuntimeError):
    pass


class ShipConfig:

    def __init__(self, init_mode='from_json', file_name=None, config_dict=None):
        # Details in README
        self.AIR_MASS_DENSITY = None  # mass density of air [kg/m^3]
        self.BOAT_AOD = None  # lateral projected area of superstructures etc. on deck [m]
        self.BOAT_AXV = None  # area of maximum transverse section exposed to the winds [m]
        self.BOAT_AYV = None  # projected lateral area above the waterline [m]
        self.BOAT_BREADTH = None  # ship breadth [m]
        self.BOAT_BS1 = None  # breadth of substructure 1 [m]
        self.BOAT_CMC = None  # horizontal distance from midship section to centre of lateral projected area AYV [m]
        self.BOAT_DRAUGHT_AFT = None  # aft draught (draught at rudder) in m
        self.BOAT_DRAUGHT_FORE = None  # fore draught (draught at forward perpendicular) in m
        self.BOAT_FUEL_RATE = None  # fuel rate at service propulsion point [g/kWh]
        self.BOAT_HBR = None  # height of top of superstructure (bridge etc.) [m]
        self.BOAT_HC = None  # height of waterline to centre of lateral projected area Ayv [m]
        self.BOAT_HS1 = None  # height of substructure 1 [m]
        self.BOAT_HS2 = None  # height of substructure 2 [m]
        self.BOAT_LENGTH = None  # overall length [m]
        self.BOAT_LS1 = None  # length of substructure 1 [m]
        self.BOAT_LS2 = None  # length of substructure 2 [m]
        self.BOAT_OVERLOAD_FACTOR = None
        self.BOAT_PROPULSION_EFFICIENCY = None  # propulsion efficiency coefficient in ideal conditions
        self.BOAT_ROUGHNESS_DISTRIBUTION_LEVEL = None  # numeric value
        self.BOAT_ROUGHNESS_LEVEL = None  # level of hull roughness, numeric value
        self.BOAT_SMCR_SPEED = None  # average speed for maximum continuous rating [m/s]
        self.BOAT_SMCR_POWER = None  # Specific Maximum Continuous Rating power [kWh]
        self.BOAT_SPEED = None  # boat speed [m/s]
        self.COURSES_FILE = None  # path to file that acts as intermediate storage for courses per routing step
        self.DEPTH_DATA = None  # path to depth data
        self.BOAT_FACTOR_CALM_WATER = None  # multiplication factor for the calm water resistance model of maripower
        self.BOAT_FACTOR_WAVE_FORCES = None  # multiplication factor for added resistance in waves model of maripower
        self.BOAT_FACTOR_WIND_FORCES = None  # multiplication factor for the added resistance in wind model of maripower
        self.BOAT_UNDER_KEEL_CLEARANCE = None  # vertical distance between keel and ground
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

    def print(self):
        # ToDo: prettify output
        logger.info(f"Config variables for ship: \n{json.dumps(self.__dict__, indent=4)}")

    def read_from_dict(self, config_dict):
        self._set_mandatory_config(config_dict)
        self._set_recommended_config(config_dict)
        self._set_optional_config(config_dict)
        self._validate_config()

    def read_from_json(self, json_file):
        with open(json_file) as f:
            config_dict = json.load(f)
            self.read_from_dict(config_dict)

    def _validate_config(self):
        """
        Validates the ship configuration attributes after they have been set.
        Raises ValueError if any attribute has an invalid value.
        """
        positive_attrs = [
            'BOAT_BREADTH',
            'BOAT_FUEL_RATE',
            'BOAT_LENGTH',
            'BOAT_SMCR_POWER',
            'BOAT_SPEED'
        ]
        for attr in positive_attrs:
            value = getattr(self, attr)
            if not (isinstance(value, (int, float)) and ((value > 0) or (value == -99))):
                raise ValueError(f"{attr} must be a positive number, but got {value}.")

        if not (0 <= self.BOAT_PROPULSION_EFFICIENCY <= 1):
            raise ValueError(
                f"BOAT_PROPULSION_EFFICIENCY must be between 0 and 1, but got {self.BOAT_PROPULSION_EFFICIENCY}.")

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
