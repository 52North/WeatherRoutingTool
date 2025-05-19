import logging

import numpy as np
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams

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
        weather_data = xr.open_dataset(self.weather_path)
        n_coords = len(lats)

        wave_height = []
        wave_direction = []
        wave_period = []
        u_wind_speed = []
        v_wind_speed = []
        u_currents = []
        v_currents = []
        pressure = []
        air_temperature = []
        salinity = []
        water_temperature = []

        for i_coord in range(0, n_coords):
            wave_direction.append(
                self.approx_weather(weather_data['VMDR'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_period.append(self.approx_weather(weather_data['VTPK'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_height.append(self.approx_weather(weather_data['VHM0'], lats[i_coord], lons[i_coord], time[i_coord]))
            v_currents.append(
                self.approx_weather(weather_data['vtotal'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            u_currents.append(
                self.approx_weather(weather_data['utotal'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            pressure.append(
                self.approx_weather(weather_data['Pressure_reduced_to_MSL_msl'], lats[i_coord], lons[i_coord],
                                    time[i_coord]))
            water_temperature.append(
                self.approx_weather(weather_data['thetao'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            salinity.append(
                self.approx_weather(weather_data['so'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            air_temperature.append(
                self.approx_weather(weather_data['Temperature_surface'], lats[i_coord], lons[i_coord], time[i_coord]))
            u_wind_speed.append(
                self.approx_weather(weather_data['u-component_of_wind_height_above_ground'], lats[i_coord],
                                    lons[i_coord], time[i_coord], 10))
            v_wind_speed.append(
                self.approx_weather(weather_data['v-component_of_wind_height_above_ground'], lats[i_coord],
                                    lons[i_coord], time[i_coord], 10))

        ship_params.wave_direction = np.array(wave_direction, dtype='float32') * u.radian
        ship_params.wave_period = np.array(wave_period, dtype='float32') * u.second
        ship_params.wave_height = np.array(wave_height, dtype='float32') * u.meter
        ship_params.u_wind_speed = np.array(u_wind_speed, dtype='float32') * u.meter / u.second
        ship_params.v_wind_speed = np.array(v_wind_speed, dtype='float32') * u.meter / u.second
        ship_params.v_currents = np.array(v_currents, dtype='float32') * u.meter / u.second
        ship_params.u_currents = np.array(u_currents, dtype='float32') * u.meter / u.second
        ship_params.pressure = np.array(pressure, dtype='float32') * u.kg / (u.meter * u.second ** 2)
        ship_params.air_temperature = np.array(air_temperature, dtype='float32') * u.Kelvin
        ship_params.air_temperature = ship_params.air_temperature.to(u.deg_C, equivalencies=u.temperature())
        ship_params.salinity = np.array(salinity, dtype='float32') * 0.001 * u.dimensionless_unscaled
        ship_params.water_temperature = np.array(water_temperature, dtype='float32') * u.deg_C

        return ship_params

    def approx_weather(self, var, lats, lons, time, height=None, depth=None):
        ship_var = var.sel(latitude=lats, longitude=lons, time=time, method='nearest', drop=False)
        if height:
            ship_var = ship_var.sel(height_above_ground=height, method='nearest', drop=False)
        if depth:
            ship_var = ship_var.sel(depth=depth, method='nearest', drop=False)
        ship_var = ship_var.fillna(0).to_numpy()

        return ship_var

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


