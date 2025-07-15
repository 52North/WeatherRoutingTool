import logging

import numpy as np
from astropy import units as u

logger = logging.getLogger('WRT.ship')


class ShipParams():
    fuel_rate: np.ndarray  # (kg/s)
    power: np.ndarray  # (W)
    rpm: np.ndarray  # (rpm)
    speed: np.ndarray  # (m/s)
    r_calm: np.ndarray  # (N)
    r_wind: np.ndarray  # (N)
    r_waves: np.ndarray  # (N)
    r_shallow: np.ndarray  # (N)
    r_roughness: np.ndarray  # (N)
    wave_height: np.ndarray  # (m)
    wave_direction: np.ndarray  # (radian)
    wave_period: np.ndarray  # (s)
    u_currents: np.ndarray  # (m/s)
    v_currents: np.ndarray  # (m/s)
    u_wind_speed: np.ndarray  # (m/s)
    v_wind_speed: np.ndarray  # (m/s)
    pressure: np.ndarray  # Pa
    air_temperature: np.ndarray  # °C
    salinity: np.ndarray  # dimensionless (kg/kg)
    water_temperature: np.ndarray  # °C
    status: np.array
    message: np.ndarray

    fuel_type: str

    _numeric_array_attributes = [
        'fuel_rate', 'power', 'rpm', 'speed', 'r_calm', 'r_wind', 'r_waves',
        'r_shallow', 'r_roughness', 'wave_height', 'wave_direction',
        'wave_period', 'u_currents', 'v_currents', 'u_wind_speed',
        'v_wind_speed', 'pressure', 'air_temperature', 'salinity', 'water_temperature'
    ]
    _non_numeric_array_attributes = [
        'status', 'message'
    ]
    _all_array_attributes = _numeric_array_attributes + _non_numeric_array_attributes


    def __init__(self, fuel_rate, power, rpm, speed, r_calm, r_wind, r_waves, r_shallow, r_roughness, wave_height,
                 wave_direction, wave_period, u_currents, v_currents, u_wind_speed, v_wind_speed, pressure,
                 air_temperature, salinity, water_temperature, status, message):
        self.fuel_rate = fuel_rate
        self.power = power
        self.rpm = rpm
        self.speed = speed
        self.r_calm = r_calm
        self.r_wind = r_wind
        self.r_waves = r_waves
        self.r_shallow = r_shallow
        self.r_roughness = r_roughness
        self.wave_height = wave_height
        self.wave_direction = wave_direction
        self.wave_period = wave_period
        self.u_currents = u_currents
        self.v_currents = v_currents
        self.u_wind_speed = u_wind_speed
        self.v_wind_speed = v_wind_speed
        self.pressure = pressure
        self.air_temperature = air_temperature
        self.salinity = salinity
        self.water_temperature = water_temperature
        self.status = status
        self.message = message

        self.fuel_type = 'HFO'

    @classmethod
    def set_default_array(cls):
        return cls(
            speed=np.array([[0]]) * u.meter/u.second,
            fuel_rate=np.array([[0]]) * u.kg/u.second,
            power=np.array([[0]]) * u.Watt,
            rpm=np.array([[0]]) * 1/u.minute,
            r_calm=np.array([[0]]) * u.newton,
            r_wind=np.array([[0]]) * u.newton,
            r_waves=np.array([[0]]) * u.newton,
            r_shallow=np.array([[0]]) * u.newton,
            r_roughness=np.array([[0]]) * u.newton,
            wave_height=np.array([[0]]) * u.meter,
            wave_direction=np.array([[0]]) * u.radian,
            wave_period=np.array([[0]]) * u.second,
            u_currents=np.array([[0]]) * u.meter/u.second,
            v_currents=np.array([[0]]) * u.meter/u.second,
            u_wind_speed=np.array([[0]]) * u.meter/u.second,
            v_wind_speed=np.array([[0]]) * u.meter/u.second,
            pressure=np.array([[0]]) * u.kg/u.meter/u.second**2,
            air_temperature=np.array([[0]]) * u.deg_C,
            salinity=np.array([[0]]) * u.dimensionless_unscaled,
            water_temperature=np.array([[0]]) * u.deg_C,
            status=np.array([[0]]),
            message=np.array([[""]])
        )

    @classmethod
    def set_default_array_1D(cls, ncoorinate_points):
        return cls(speed=np.full(shape=ncoorinate_points, fill_value=0) * u.meter/u.second,
                   fuel_rate=np.full(shape=ncoorinate_points, fill_value=0) * u.kg/u.second,
                   power=np.full(shape=ncoorinate_points, fill_value=0) * u.Watt,
                   rpm=np.full(shape=ncoorinate_points, fill_value=0) * 1/u.minute,
                   r_calm=np.full(shape=ncoorinate_points, fill_value=0) * u.newton,
                   r_wind=np.full(shape=ncoorinate_points, fill_value=0) * u.newton,
                   r_waves=np.full(shape=ncoorinate_points, fill_value=0) * u.newton,
                   r_shallow=np.full(shape=ncoorinate_points, fill_value=0) * u.newton,
                   r_roughness=np.full(shape=ncoorinate_points, fill_value=0) * u.newton,
                   wave_height=np.full(shape=ncoorinate_points, fill_value=0) * u.meter,
                   wave_direction=np.full(shape=ncoorinate_points, fill_value=0) * u.radian,
                   wave_period=np.full(shape=ncoorinate_points, fill_value=0) * u.second,
                   u_currents=np.full(shape=ncoorinate_points, fill_value=0) * u.meter/u.second,
                   v_currents=np.full(shape=ncoorinate_points, fill_value=0) * u.meter/u.second,
                   u_wind_speed=np.full(shape=ncoorinate_points, fill_value=0) * u.meter/u.second,
                   v_wind_speed=np.full(shape=ncoorinate_points, fill_value=0) * u.meter/u.second,
                   pressure=np.full(shape=ncoorinate_points, fill_value=0) * u.kg/u.meter/u.second**2,
                   air_temperature=np.full(shape=ncoorinate_points, fill_value=0) * u.deg_C,
                   salinity=np.full(shape=ncoorinate_points, fill_value=0) * u.dimensionless_unscaled,
                   water_temperature=np.full(shape=ncoorinate_points, fill_value=0) * u.deg_C,
                   status=np.full(shape=ncoorinate_points, fill_value=0),
                   message=np.full(shape=ncoorinate_points, fill_value=""))

    def print(self):
        for attr in self._numeric_array_attributes:
            value = getattr(self, attr)
            logger.info(f'{attr}: {value.value} {value.unit.to_string()}')
        for attr in self._non_numeric_array_attributes:
            value = getattr(self, attr)
            logger.info(f'{attr}: {value}')
        logger.info('fuel_type: ' + str(self.fuel_type))

    def print_shape(self):
        for attr in self._all_array_attributes:
            value = getattr(self, attr)
            logger.info(f'{attr}: {value.shape}')

    def define_courses(self, courses_segments):
        for attr in self._all_array_attributes:
            setattr(self, attr, np.repeat(getattr(self, attr), courses_segments + 1, axis=1))

    def get_power(self):
        return self.power

    def get_fuel_rate(self):
        return self.fuel_rate

    def get_rwind(self):
        return self.r_wind

    def get_rcalm(self):
        return self.r_calm

    def get_rwaves(self):
        return self.r_waves

    def get_rshallow(self):
        return self.r_shallow

    def get_rroughness(self):
        return self.r_roughness

    def get_fuel_type(self):
        return self.fuel_type

    def get_rpm(self):
        return self.rpm

    def get_speed(self):
        return self.speed

    def get_wave_height(self):
        return self.wave_height

    def get_wave_direction(self):
        return self.wave_direction

    def get_wave_period(self):
        return self.wave_period

    def get_u_currents(self):
        return self.u_currents

    def get_v_currents(self):
        return self.v_currents

    def get_u_wind_speed(self):
        return self.u_wind_speed

    def get_v_wind_speed(self):
        return self.v_wind_speed

    def get_pressure(self):
        return self.pressure

    def get_air_temperature(self):
        return self.air_temperature

    def get_salinity(self):
        return self.salinity

    def get_water_temperature(self):
        return self.water_temperature

    def get_status(self):
        return self.status

    def get_message(self):
        return self.message

    def set_speed(self, new_speed):
        self.speed = new_speed

    def set_fuel_rate(self, new_fuel):
        self.fuel_rate = new_fuel

    def set_rpm(self, new_rpm):
        self.rpm = new_rpm

    def set_power(self, new_power):
        self.power = new_power

    def set_rwind(self, new_rwind):
        self.r_wind = new_rwind

    def set_rcalm(self, new_rcalm):
        self.r_calm = new_rcalm

    def set_rwaves(self, new_rwaves):
        self.r_waves = new_rwaves

    def set_rshallow(self, new_rshallow):
        self.r_shallow = new_rshallow

    def set_rroughness(self, new_rroughnes):
        self.r_roughness = new_rroughnes

    def set_wave_height(self, new_wave_height):
        self.wave_height = new_wave_height

    def set_wave_direction(self, new_wave_direction):
        self.wave_direction = new_wave_direction

    def set_wave_period(self, new_wave_period):
        self.wave_period = new_wave_period

    def set_u_currents(self, new_u_currents):
        self.u_currents = new_u_currents

    def set_v_currents(self, new_v_currents):
        self.v_currents = new_v_currents

    def set_u_wind_speed(self, new_u_wind_speed):
        self.u_wind_speed = new_u_wind_speed

    def set_v_wind_speed(self, new_v_wind_speed):
        self.v_wind_speed = new_v_wind_speed

    def set_pressure(self, new_pressure):
        self.pressure = new_pressure

    def set_air_temperature(self, new_air_temperature):
        self.air_temperature = new_air_temperature

    def set_salinity(self, new_salinity):
        self.salinity = new_salinity

    def set_water_temperature(self, new_water_temperature):
        self.water_temperature = new_water_temperature

    def set_status(self, new_status):
        self.status = new_status

    def set_message(self, new_message):
        self.message = new_message

    def select(self, idxs):
        for attr in self._all_array_attributes:
            setattr(self, attr, getattr(self, attr)[:, idxs])

    def flip(self):
        for attr in self._all_array_attributes:
            val = getattr(self, attr)[:-1]
            val = np.flip(val, 0)
            if attr in self._numeric_array_attributes:
                val = np.append(val, -99 * val.unit)
            elif attr == 'status':
                val = np.append(val, -99)
            elif attr == 'message':
                val = np.append(val, "")
            setattr(self, attr, val)

    def expand_axis_for_intermediate(self):
        for attr in self._all_array_attributes:
            setattr(self, attr, np.expand_dims(getattr(self, attr), axis=1))

    def get_element(self, idx):
        try:
            return_values = {}
            for attr in self._all_array_attributes:
                return_values[attr] = getattr(self, attr)[idx]
        except ValueError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.speed.shape[0]))
        return tuple(return_values[attr] for attr in self._all_array_attributes)


    def get_single_object(self, idx):
        # ToDo: reuse get_element here
        try:
            attributes_at_idx = {attr: getattr(self, attr)[idx] for attr in self._all_array_attributes}
        except ValueError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.speed.shape[0]))

        sp = ShipParams(
            **attributes_at_idx
        )
        return sp

    def get_reduced_2D_object(self, row_start=None, row_end=None, col_start=None, col_end=None, idxs=None):

        try:
            if idxs is None:
                new_params = {}
                for attr in self._all_array_attributes:
                    new_params[attr] = getattr(self, attr)[row_start:row_end, col_start:col_end]
            else:
                new_params = {}
                for attr in self._all_array_attributes:
                    new_params[attr] = getattr(self, attr)[:, idxs]
        except ValueError:
            raise ValueError(
                'Index ' + str(col_start) + ' is not available for array with length ' + str(self.speed.shape[0]))

        sp = ShipParams(
            **new_params
        )
        return sp
    
