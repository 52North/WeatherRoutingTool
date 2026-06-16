import logging
from pathlib import Path

import numpy as np
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship_config import ShipConfig
from collections import OrderedDict
from typing import Tuple, Optional, Sequence, Any


logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.

class Cache:
    """
    keys are arbitrary hashable objects
    stores arbitrary values
    evicts oldest entries when max_entries exceeded
    """

    def __init__(self, max_entries: int = 10_000):
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self.max_entries = int(max_entries)
        # OrderedDict preserves insertion order and supports popitem(last=False)
        # to remove the oldest entry in O(1).
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return len(self._cache)

    def contains(self, key: Any) -> bool:
        return key in self._cache

    def get(self, key: Any, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: Any, value: Any) -> None:
        if key in self._cache:
            # update value but keep current insertion order
            self._cache[key] = value
            return
        self._cache[key] = value
        # evict oldest if necessary
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)


class Boat:
    weather_path: str  # path to netCDF containing weather data

    def __init__(self, ship_config: ShipConfig):
        self.counter = 0
        self.under_keel_clearance = ship_config.BOAT_UNDER_KEEL_CLEARANCE * u.meter
        self.draught_aft = ship_config.BOAT_DRAUGHT_AFT * u.meter
        self.draught_fore = ship_config.BOAT_DRAUGHT_FORE * u.meter
        self._weather_data = None
        self._weather_access_dims = ("time", "latitude", "longitude")
        self._weather_cache = Cache(max_entries=10000)

    def _get_weather_data(self):
        if self._weather_data is not None:
            return self._weather_data

        # Keep the dataset lazy so only requested slices are read into memory.
        self._weather_data = xr.open_dataset(self.weather_path)
        return self._weather_data

    @staticmethod
    def _nearest_index(values, value):
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.datetime64):
            value = np.datetime64(value)
        return int(np.argmin(np.abs(values - value)))

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
        # weather_data= xr.open_dataset(self.weather_path)
        n_coords = len(lats)
        # only need to be read once, for nearest indexing and cache keys, so that it works more robust
        lat_values = weather_data.coords['latitude'].values
        lon_values = weather_data.coords['longitude'].values
        time_values = weather_data.coords['time'].values

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

        def cached_lookup(var_key, da, lat, lon, t, height=None, depth=None):
            ilat = self._nearest_index(lat_values, lat)
            ilon = self._nearest_index(lon_values, lon)
            itime = self._nearest_index(time_values, t)
            key = (var_key, ilat, ilon, itime, height, depth)
            val = self._weather_cache.get(key)
            if val is not None:
                return val
            v = self.approx_weather(da, lat, lon, t, height, depth)
            self._weather_cache.set(key, v)
            return v

        for i_coord in range(0, n_coords):

            wave_direction.append(cached_lookup(
                'VMDR', weather_data['VMDR'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_period.append(cached_lookup(
                'VTPK', weather_data['VTPK'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_height.append(cached_lookup(
                'VHM0', weather_data['VHM0'], lats[i_coord], lons[i_coord], time[i_coord]))
            v_currents.append(cached_lookup(
                'vtotal', weather_data['vtotal'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            u_currents.append(cached_lookup(
                'utotal', weather_data['utotal'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            pressure.append(cached_lookup(
                'Pressure_reduced_to_MSL_msl', weather_data['Pressure_reduced_to_MSL_msl'],
                lats[i_coord], lons[i_coord], time[i_coord]))
            water_temperature.append(cached_lookup(
                'thetao', weather_data['thetao'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            salinity.append(cached_lookup(
                'so', weather_data['so'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            air_temperature.append(cached_lookup(
                'Temperature_surface', weather_data['Temperature_surface'],
                lats[i_coord], lons[i_coord], time[i_coord]))
            u_wind_speed.append(cached_lookup(
                'u-component_of_wind_height_above_ground', weather_data['u-component_of_wind_height_above_ground'],
                lats[i_coord], lons[i_coord], time[i_coord], 10))
            v_wind_speed.append(cached_lookup(
                'v-component_of_wind_height_above_ground', weather_data['v-component_of_wind_height_above_ground'],
                lats[i_coord], lons[i_coord], time[i_coord], 10))

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
