import json
import logging
import os
import sys

logger = logging.getLogger('WRT.ShipConfig')

MANDATORY_CONFIG_VARIABLES = ['SMCR_POWER','LENGTH', 'BREADTH', 'HBR', 'FUEL_RATE']

RECOMMENDED_CONFIG_VARIABLES = {
}

# optional variables with default values
OPTIONAL_CONFIG_VARIABLES = {
    'PROPULSION_EFFICIENCY': 0.63, # assuming n_H = 1.05, n_0 = 0.1, n_R = 1
    'OVERLOAD_FACTOR': 0,
    'AIR_MASS_DENSITY': 1.2225,
    'AXV': -99,
    'AYV': -99,
    'AOD': -99,
    'LS1': -99,
    'LS2': -99,
    'HS1': -99,
    'HS2': -99,
    'BS1': -99,
    'CMC': -99,
    'HC': -99,
}

class RequiredConfigError(RuntimeError):
    pass

class ShipConfig:

    def __init__(self, init_mode='from_json', file_name=None, config_dict=None):
        # Details in README
        self.SMCR_POWER = None  # Specific Maximum Continuous Rating power [kWh]
        self.PROPULSION_EFFICIENCY = None # propulsion efficiency coefficient in ideal conditions
        self.OVERLOAD_FACTOR = None
        self.LENGTH = None # overall length [m]
        self.BREADTH = None # ship breadth [m]
        self.AIR_MASS_DENSITY = None # mass density of air [kg/m^3]
        self.CMC = None # horizontal distance from midship section to centre of lateral projected area AYV [m]
        self.HBR = None # height of top of superstructure (bridge etc.) [m]
        self.HC = None # height of waterline to centre of lateral projected area Ayv [m]
        self.AXV = None # area of maximum transverse section exposed to the winds [m]
        self.AYV = None # projected lateral area above the waterline [m]
        self.AOD = None # lateral projected area of superstructures etc. on deck [m]
        self.LS1 = None # length of substructure 1 [m]
        self.LS2 = None # length of substructure 2 [m]
        self.HS1 = None # height of substructure 1 [m]
        self.HS2 = None # height of substructure 2 [m]
        self.BS1 = None # breadth of substructure 1 [m]
        self.FUEL_RATE = None # fuel rate at service propulsion point [g/kWh]

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

