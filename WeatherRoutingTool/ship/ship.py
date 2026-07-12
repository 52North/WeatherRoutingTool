import logging
from pathlib import Path

import numpy as np
import xarray as xr
from astropy import units as u
from typing import Tuple, Optional, Sequence

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship_config import ShipConfig


logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.


class Boat:
    weather_path: str  # path to netCDF containing weather data

    def __init__(self, ship_config: ShipConfig):
        self.counter = 0
        self.under_keel_clearance = ship_config.BOAT_UNDER_KEEL_CLEARANCE * u.meter
        self.draught_aft = ship_config.BOAT_DRAUGHT_AFT * u.meter
        self.draught_fore = ship_config.BOAT_DRAUGHT_FORE * u.meter
        self._weather_data = None

    def _get_weather_data(self):
        if self._weather_data is not None:
            return self._weather_data
        self._weather_data = xr.open_dataset(self.weather_path)
        return self._weather_data

    def close(self):
        if self._weather_data is not None:
            self._weather_data.close()
            self._weather_data = None

    def get_required_water_depth(self):
        needs_water_depth = max(self.draught_aft, self.draught_fore) + self.under_keel_clearance
        return needs_water_depth.value

    def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False):
        pass

    def print_init(self):
        pass

    def evaluate_weather(self, ship_params, lats, lons, time):
        weather_data = self._get_weather_data()
        lat_da = xr.DataArray(np.asarray(lats, dtype='float64'), dims='points')
        lon_da = xr.DataArray(np.asarray(lons, dtype='float64'), dims='points')
        time_da = xr.DataArray(np.asarray(time, dtype='datetime64[ns]'), dims='points')

        ship_params.wave_direction = self.approx_weather(weather_data['VMDR'], lat_da, lon_da, time_da) * u.radian
        ship_params.wave_period = self.approx_weather(weather_data['VTPK'], lat_da, lon_da, time_da) * u.second
        ship_params.wave_height = self.approx_weather(weather_data['VHM0'], lat_da, lon_da, time_da) * u.meter
        ship_params.u_currents = self.approx_weather(weather_data['utotal'], lat_da, lon_da, time_da, None, 0.5) * u.meter / u.second
        ship_params.v_currents = self.approx_weather(weather_data['vtotal'], lat_da, lon_da, time_da, None, 0.5) * u.meter / u.second
        ship_params.pressure = self.approx_weather(
            weather_data['Pressure_reduced_to_MSL_msl'], lat_da, lon_da, time_da) * u.kg / (u.meter * u.second ** 2)
        ship_params.water_temperature = self.approx_weather(weather_data['thetao'], lat_da, lon_da, time_da, None, 0.5) * u.deg_C
        ship_params.salinity = self.approx_weather(weather_data['so'], lat_da, lon_da, time_da, None, 0.5) * 0.001 * u.dimensionless_unscaled
        ship_params.air_temperature = self.approx_weather(
            weather_data['Temperature_surface'], lat_da, lon_da, time_da) * u.Kelvin
        ship_params.u_wind_speed = self.approx_weather(
            weather_data['u-component_of_wind_height_above_ground'], lat_da, lon_da, time_da, 10) * u.meter / u.second
        ship_params.v_wind_speed = self.approx_weather(
            weather_data['v-component_of_wind_height_above_ground'], lat_da, lon_da, time_da, 10) * u.meter / u.second
        ship_params.air_temperature = ship_params.air_temperature.to(u.deg_C, equivalencies=u.temperature())

        return ship_params

    def approx_weather(self, var, lats, lons, time, height=None, depth=None):
        points = 'points'
        lat_da = xr.DataArray(np.asarray(lats, dtype='float64'), dims=points)
        lon_da = xr.DataArray(np.asarray(lons, dtype='float64'), dims=points)
        time_da = xr.DataArray(np.asarray(time, dtype='datetime64[ns]'), dims=points)
        ship_var = var.sel(latitude=lat_da, longitude=lon_da, time=time_da, method='nearest', drop=False)
        if height is not None:
            ship_var = ship_var.sel(height_above_ground=height, method='nearest', drop=False)
        if depth is not None:
            ship_var = ship_var.sel(depth=depth, method='nearest', drop=False)
        ship_var = ship_var.fillna(0).to_numpy()
        self.counter += 1

        return ship_var

    def load_data(self):
        pass

    def check_data_meaningful(self):
        """
        This is an optional method to check if default boat variables have been changed into meaningful values.
        It can be implemented in Child classes.
        """
        pass


class ConstantFuelBoat(Boat):
    fuel_rate: float  # dummy value for fuel_rate that is returned

    def __init__(self, ship_config: ShipConfig):
        super().__init__(ship_config)

        # mandatory variables
        self.fuel_rate = ship_config.BOAT_FUEL_RATE * u.kg / u.second

    def print_init(self):
        logger.info(form.get_log_step('boat fuel rate' + str(self.fuel_rate), 1))
        form.print_line()

    def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False):
        debug = False
        n_requests = len(courses)

        dummy_array = np.full(n_requests, -99)
        fuel_array = np.full(n_requests, self.fuel_rate)

        ship_params = ShipParams(
            fuel_rate=fuel_array * u.kg / u.s,
            power=dummy_array * u.Watt,
            rpm=dummy_array * u.Hz,
            speed=dummy_array * u.meter / u.second,
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
