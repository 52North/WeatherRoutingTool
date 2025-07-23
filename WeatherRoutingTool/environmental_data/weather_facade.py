"""Weather functions."""
import logging
from datetime import datetime, timedelta

import xarray as xr

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.utils.unit_conversion import round_time

logger = logging.getLogger('WRT.weather')

UNITS_DICT = {
    'Pressure_reduced_to_MSL_msl': ['Pa'],
    'so': ['1e-3'],
    'Temperature_surface': ['K'],
    'thetao': ['degrees_C', '°C'],
    'u-component_of_wind_height_above_ground': ['m s-1', 'm/s'],
    'v-component_of_wind_height_above_ground': ['m s-1', 'm/s'],
    'utotal': ['m s-1', 'm/s'],
    'vtotal': ['m s-1', 'm/s'],
    'VHM0': ['m'],
    'VMDR': ['degree'],
    'VTPK': ['s']
}

class WeatherCond:
    time_steps: int
    time_res: timedelta
    time_start: datetime
    time_end: timedelta
    map_size: Map
    ds: xr.Dataset

    def __init__(self, time, hours, time_res):
        form.print_line()
        logger.info('Initialising weather')

        self.time_res = time_res
        self.time_start = time
        self.time_end = time + timedelta(hours=hours)

        time_passed = self.time_end - self.time_start
        self.time_steps = int(time_passed.total_seconds() / self.time_res.total_seconds())

        logger.info(form.get_log_step('forecast from ' + str(self.time_start) + ' to ' + str(self.time_end), 1))
        logger.info(form.get_log_step('nof time steps ' + str(self.time_steps), 1))
        form.print_line()

    @property
    def time_res(self):
        return self._time_res

    @time_res.setter
    def time_res(self, value):
        if (value < 3):
            raise ValueError('Resolution below 3h not possible')
        self._time_res = timedelta(hours=value)
        logger.info(form.get_log_step('time resolution: ' + str(self._time_res) + ' hours', 1))

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, value):
        rounded_time = value - self.time_res / 2
        rounded_time = round_time(rounded_time, int(self.time_res.total_seconds()))
        self._time_start = rounded_time

    @property
    def time_end(self):
        return self._time_end

    @time_end.setter
    def time_end(self, value):
        rounded_time = value + self.time_res / 2
        rounded_time = round_time(rounded_time, int(self.time_res.total_seconds()))
        self._time_end = rounded_time

    def check_units(self):
        for var_name, data_array in self.ds.data_vars.items():
            if var_name in UNITS_DICT:
                if 'units' not in data_array.attrs:
                    logger.warning(f"Weather data variable '{var_name}' has no 'units' attribute.")
                else:
                    unit = data_array.attrs['units']
                    if unit not in UNITS_DICT[var_name]:
                        logger.warning(f"Weather data variable '{var_name}' has the wrong unit '{unit}', "
                                       f"should be one of '{UNITS_DICT[var_name]}'.")
            else:
                logger.warning(f"Weather data variable '{var_name}' found, but not expected. Will be ignored.")

    def set_map_size(self, map):
        self.map_size = map

    def get_map_size(self):
        return self.map_size

    def read_dataset(self, filepath=None):
        pass

    def get_weather(self):
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
            wave_period.append(
                self.approx_weather(weather_data['VTPK'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_height.append(
                self.approx_weather(weather_data['VHM0'], lats[i_coord], lons[i_coord], time[i_coord]))
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
                self.approx_weather(weather_data['Temperature_surface'], lats[i_coord], lons[i_coord],
                                    time[i_coord]))
            u_wind_speed.append(
                self.approx_weather(weather_data['u-component_of_wind_height_above_ground'], lats[i_coord],
                                    lons[i_coord], time[i_coord], 10))
            v_wind_speed.append(
                self.approx_weather(weather_data['v-component_of_wind_height_above_ground'], lats[i_coord],
                                    lons[i_coord], time[i_coord], 10))

        weather_dict = {
            "wave_direction": np.array(wave_direction, dtype='float32') * u.radian,
            "wave_period": np.array(wave_period, dtype='float32') * u.second,
            "wave_height": np.array(wave_height, dtype='float32') * u.meter,
            "u_wind_speed": np.array(u_wind_speed, dtype='float32') * u.meter / u.second,
            "v_wind_speed": np.array(v_wind_speed, dtype='float32') * u.meter / u.second,
            "v_currents": np.array(v_currents, dtype='float32') * u.meter / u.second,
            "u_currents" : np.array(u_currents, dtype='float32') * u.meter / u.second,
            "pressure" : np.array(pressure, dtype='float32') * u.kg / (u.meter * u.second ** 2),
            "air_temperature" : np.array(air_temperature, dtype='float32') * u.Kelvin,
          #  "air_temperature" : ship_params.air_temperature.to(u.deg_C, equivalencies=u.temperature()),
            "salinity" : np.array(salinity, dtype='float32') * 0.001 * u.dimensionless_unscaled,
            "water_temperature" : np.array(water_temperature, dtype='float32') * u.deg_C
        }
        weather_dict['air_temperature'] = weather_dict['air_temperature'].to(u.deg_C, equivalencies=u.temperature())
        return weather_dict



    def approx_weather(self, var, lats, lons, time, height=None, depth=None):
        var = var.sel(latitude=lats, longitude=lons, time=time, method='nearest', drop=False)
        if height:
            var = var.sel(height_above_ground=height, method='nearest', drop=False)
        if depth:
            var = var.sel(depth=depth, method='nearest', drop=False)
        var = var.fillna(0).to_numpy()

        return var