import logging

import numpy as np
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship_config import ShipConfig

logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.

class Boat:
    speed: float  # boat speed in m/s
    weather_path: str  # path to netCDF containing weather data

    def __init__(self, init_mode='from_file', file_name=None, config_dict=None):
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig(file_name=file_name)
        else:
            config_obj = ShipConfig(init_mode='from_dict', config_dict=config_dict)

        self.speed = config_obj.BOAT_SPEED * u.meter / u.second
        self.under_keel_clearance = config_obj.BOAT_UNDER_KEEL_CLEARANCE * u.meter
        self.draught_aft = config_obj.BOAT_DRAUGHT_AFT * u.meter
        self.draught_fore = config_obj.BOAT_DRAUGHT_FORE * u.meter

    def get_required_water_depth(self):
        needs_water_depth = max(self.draught_aft, self.draught_fore) + self.under_keel_clearance
        return needs_water_depth.value

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        pass

    def get_boat_speed(self):
        return self.speed

    def print_init(self):
        pass

    def set_boat_speed(self, speed):
        self.speed = speed

    def evaluate_weather(self, ship_params, lats, lons, time):
        weather_dict = mediator.get_weather()

        ship_params.wave_direction = weather_dict['wave_direction']
        ship_params.wave_period = weather_dict['wave_period']
        ship_params.wave_height = weather_dict['wave_height']
        ship_params.u_wind_speed = weather_dict['u_wind_speed']
        ship_params.v_wind_speed = weather_dict['v_wind_speed']
        ship_params.v_currents = weather_dict['v_currents']
        ship_params.u_currents = weather_dict['u_currents']
        ship_params.pressure = weather_dict['pressure']
        ship_params.air_temperature = weather_dict['air_temperature']
        ship_params.salinity = weather_dict['salinity']
        ship_params.water_temperature = weather_dict['water_temperature']

        return ship_params

    def load_data(self):
        pass


class ConstantFuelBoat(Boat):
    fuel_rate: float  # dummy value for fuel_rate that is returned
    speed: float  # boat speed

    def __init__(self, init_mode='from_file', file_name=None, config_dict=None):
        super().__init__(init_mode, file_name, config_dict)
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig(file_name=file_name)
        else:
            config_obj = ShipConfig(init_mode='from_dict', config_dict=config_dict)

        # mandatory variables
        self.fuel_rate = config_obj.BOAT_FUEL_RATE * u.kg / u.second

    def print_init(self):
        logger.info(form.get_log_step('boat speed' + str(self.speed), 1))
        logger.info(form.get_log_step('boat fuel rate' + str(self.fuel_rate), 1))

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False
        n_requests = len(courses)

        dummy_array = np.full(n_requests, -99)
        fuel_array = np.full(n_requests, self.fuel_rate)
        speed_array = np.full(n_requests, self.speed)

        ship_params = ShipParams(
            fuel_rate=fuel_array * u.kg / u.s,
            power=dummy_array * u.Watt,
            rpm=dummy_array * u.Hz,
            speed=speed_array * u.meter / u.second,
            r_wind=dummy_array * u.N,
            r_calm=dummy_array * u.N,
            r_waves=dummy_array * u.N,
            r_shallow=dummy_array * u.N,
            r_roughness=dummy_array * u.N,
            wave_height=dummy_array * u.meter,
            wave_direction=dummy_array * u.radian,
            wave_period=dummy_array * u.second,
            u_currents=dummy_array * u.meter / u.second,
            v_currents=dummy_array * u.meter / u.second,
            u_wind_speed=dummy_array * u.meter / u.second,
            v_wind_speed=dummy_array * u.meter / u.second,
            pressure=dummy_array * u.kg / u.meter / u.second ** 2,
            air_temperature=dummy_array * u.deg_C,
            salinity=dummy_array * u.dimensionless_unscaled,
            water_temperature=dummy_array * u.deg_C,
            status=dummy_array,
            message=np.full(n_requests, "")
        )

        if (debug):
            ship_params.print()
            form.print_step('fuel result' + str(ship_params.get_fuel_rate()))

        return ship_params
