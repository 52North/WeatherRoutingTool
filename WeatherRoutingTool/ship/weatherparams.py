import logging

import numpy as np
from astropy import units as u

logger = logging.getLogger('WRT.ship')


class WeatherParams():
    """
    Container for weather-related parameters.
    Separated from ShipParams to better organize ship performance vs environmental data.
    """
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

    def __init__(self, wave_height, wave_direction, wave_period, u_currents, v_currents,
                 u_wind_speed, v_wind_speed, pressure, air_temperature, salinity,
                 water_temperature, status, message):
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

    @classmethod
    def set_default_array(cls):
        return cls(
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
    def set_default_array_1D(cls, ncoordinate_points):
        return cls(
            wave_height=np.full(shape=ncoordinate_points, fill_value=0) * u.meter,
            wave_direction=np.full(shape=ncoordinate_points, fill_value=0) * u.radian,
            wave_period=np.full(shape=ncoordinate_points, fill_value=0) * u.second,
            u_currents=np.full(shape=ncoordinate_points, fill_value=0) * u.meter/u.second,
            v_currents=np.full(shape=ncoordinate_points, fill_value=0) * u.meter/u.second,
            u_wind_speed=np.full(shape=ncoordinate_points, fill_value=0) * u.meter/u.second,
            v_wind_speed=np.full(shape=ncoordinate_points, fill_value=0) * u.meter/u.second,
            pressure=np.full(shape=ncoordinate_points, fill_value=0) * u.kg/u.meter/u.second**2,
            air_temperature=np.full(shape=ncoordinate_points, fill_value=0) * u.deg_C,
            salinity=np.full(shape=ncoordinate_points, fill_value=0) * u.dimensionless_unscaled,
            water_temperature=np.full(shape=ncoordinate_points, fill_value=0) * u.deg_C,
            status=np.full(shape=ncoordinate_points, fill_value=0),
            message=np.full(shape=ncoordinate_points, fill_value="")
        )

    def print(self):
        logger.info('wave_height: ' + str(self.wave_height.value) + ' ' + self.wave_height.unit.to_string())
        logger.info('wave_direction: ' + str(self.wave_direction.value) + ' ' + self.wave_direction.unit.to_string())
        logger.info('wave_period: ' + str(self.wave_period.value) + ' ' + self.wave_period.unit.to_string())
        logger.info('u_currents: ' + str(self.u_currents.value) + ' ' + self.u_currents.unit.to_string())
        logger.info('v_currents: ' + str(self.v_currents.value) + ' ' + self.v_currents.unit.to_string())
        logger.info('u_wind_speed: ' + str(self.u_wind_speed.value) + ' ' + self.u_wind_speed.unit.to_string())
        logger.info('v_wind_speed: ' + str(self.v_wind_speed.value) + ' ' + self.v_wind_speed.unit.to_string())
        logger.info('pressure: ' + str(self.pressure.value) + ' ' + self.pressure.unit.to_string())
        logger.info('air_temperature: ' + str(self.air_temperature.value) + ' ' + self.air_temperature.unit.to_string())
        logger.info('salinity: ' + str(self.salinity.value) + ' ' + self.salinity.unit.to_string())
        logger.info('water_temperature: ' + str(self.water_temperature.value) + ' ' +
                    self.water_temperature.unit.to_string())
        logger.info('status' + str(self.status))
        logger.info('message' + str(self.message))

    def print_shape(self):
        logger.info('wave_height: ' + str(self.wave_height.shape))
        logger.info('wave_direction: ' + str(self.wave_direction.shape))
        logger.info('wave_period: ' + str(self.wave_period.shape))
        logger.info('u_currents: ' + str(self.u_currents.shape))
        logger.info('v_currents: ' + str(self.v_currents.shape))
        logger.info('u_wind_speed: ' + str(self.u_wind_speed.shape))
        logger.info('v_wind_speed: ' + str(self.v_wind_speed.shape))
        logger.info('pressure: ' + str(self.pressure.shape))
        logger.info('air_temperature: ' + str(self.air_temperature.shape))
        logger.info('salinity: ' + str(self.salinity.shape))
        logger.info('water_temperature: ' + str(self.water_temperature.shape))
        logger.info('status' + str(self.status))
        logger.info('message' + str(self.message))

    def define_courses(self, courses_segments):
        """Expand arrays for course segments."""
        self.wave_height = np.repeat(self.wave_height, courses_segments + 1, axis=1)
        self.wave_direction = np.repeat(self.wave_direction, courses_segments + 1, axis=1)
        self.wave_period = np.repeat(self.wave_period, courses_segments + 1, axis=1)
        self.u_currents = np.repeat(self.u_currents, courses_segments + 1, axis=1)
        self.v_currents = np.repeat(self.v_currents, courses_segments + 1, axis=1)
        self.u_wind_speed = np.repeat(self.u_wind_speed, courses_segments + 1, axis=1)
        self.v_wind_speed = np.repeat(self.v_wind_speed, courses_segments + 1, axis=1)
        self.pressure = np.repeat(self.pressure, courses_segments + 1, axis=1)
        self.air_temperature = np.repeat(self.air_temperature, courses_segments + 1, axis=1)
        self.salinity = np.repeat(self.salinity, courses_segments + 1, axis=1)
        self.water_temperature = np.repeat(self.water_temperature, courses_segments + 1, axis=1)
        self.status = np.repeat(self.status, courses_segments + 1, axis=1)
        self.message = np.repeat(self.message, courses_segments + 1, axis=1)

    # Getter methods
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

    # Setter methods
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
        """Select specific indices from the arrays."""
        self.wave_height = self.wave_height[:, idxs]
        self.wave_direction = self.wave_direction[:, idxs]
        self.wave_period = self.wave_period[:, idxs]
        self.u_currents = self.u_currents[:, idxs]
        self.v_currents = self.v_currents[:, idxs]
        self.u_wind_speed = self.u_wind_speed[:, idxs]
        self.v_wind_speed = self.v_wind_speed[:, idxs]
        self.pressure = self.pressure[:, idxs]
        self.air_temperature = self.air_temperature[:, idxs]
        self.salinity = self.salinity[:, idxs]
        self.water_temperature = self.water_temperature[:, idxs]
        self.status = self.status[:, idxs]
        self.message = self.message[:, idxs]

    def flip(self):
        """Remove last element from all arrays."""
        self.wave_height = self.wave_height[:-1]
        self.wave_direction = self.wave_direction[:-1]
        self.wave_period = self.wave_period[:-1]
        self.u_currents = self.u_currents[:-1]
        self.v_currents = self.v_currents[:-1]
        self.u_wind_speed = self.u_wind_speed[:-1]
        self.v_wind_speed = self.v_wind_speed[:-1]
        self.pressure = self.pressure[:-1]
        self.air_temperature = self.air_temperature[:-1]
        self.salinity = self.salinity[:-1]
        self.water_temperature = self.water_temperature[:-1]
        self.status = self.status[:-1]
        self.message = self.message[:-1]

    def append_dummy(self):
        """Append dummy values to all arrays."""
        self.wave_height = np.append(self.wave_height, -99 * self.wave_height.unit)
        self.wave_direction = np.append(self.wave_direction, -99 * self.wave_direction.unit)
        self.wave_period = np.append(self.wave_period, -99 * self.wave_period.unit)
        self.u_currents = np.append(self.u_currents, -99 * self.u_currents.unit)
        self.v_currents = np.append(self.v_currents, -99 * self.v_currents.unit)
        self.u_wind_speed = np.append(self.u_wind_speed, -99 * self.u_wind_speed.unit)
        self.v_wind_speed = np.append(self.v_wind_speed, -99 * self.v_wind_speed.unit)
        self.pressure = np.append(self.pressure, -99 * self.pressure.unit)
        self.air_temperature = np.append(self.air_temperature, -99 * self.air_temperature.unit)
        self.salinity = np.append(self.salinity, -99 * self.salinity.unit)
        self.water_temperature = np.append(self.water_temperature, -99 * self.water_temperature.unit)
        self.status = np.append(self.status, -99)
        self.message = np.append(self.message, "")

    def get_element(self, idx):
        """Get weather parameters at a specific index."""
        try:
            wave_height = self.wave_height[idx]
            wave_direction = self.wave_direction[idx]
            wave_period = self.wave_period[idx]
            u_currents = self.u_currents[idx]
            v_currents = self.v_currents[idx]
            u_wind_speed = self.u_wind_speed[idx]
            v_wind_speed = self.v_wind_speed[idx]
            pressure = self.pressure[idx]
            air_temperature = self.air_temperature[idx]
            salinity = self.salinity[idx]
            water_temperature = self.water_temperature[idx]
            status = self.status[idx]
            message = self.message[idx]
        except IndexError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.wave_height.shape[0]))
        return (wave_height, wave_direction, wave_period, u_currents, v_currents, u_wind_speed, v_wind_speed,
                pressure, air_temperature, salinity, water_temperature, status, message)

    def get_single_object(self, idx):
        """Get a new WeatherParams object with a single element at the specified index."""
        wave_height, wave_direction, wave_period, u_currents, v_currents, u_wind_speed, v_wind_speed, \
            pressure, air_temperature, salinity, water_temperature, status, message = self.get_element(idx)

        return WeatherParams(
            wave_height=wave_height,
            wave_direction=wave_direction,
            wave_period=wave_period,
            u_currents=u_currents,
            v_currents=v_currents,
            u_wind_speed=u_wind_speed,
            v_wind_speed=v_wind_speed,
            pressure=pressure,
            air_temperature=air_temperature,
            salinity=salinity,
            water_temperature=water_temperature,
            status=status,
            message=message
        )

    def get_reduced_2D_object(self, row_start=None, row_end=None, col_start=None, col_end=None, idxs=None):
        """Get a reduced WeatherParams object with a subset of the data."""
        if idxs is not None:
            wave_height = self.wave_height[:, idxs]
            wave_direction = self.wave_direction[:, idxs]
            wave_period = self.wave_period[:, idxs]
            u_currents = self.u_currents[:, idxs]
            v_currents = self.v_currents[:, idxs]
            u_wind_speed = self.u_wind_speed[:, idxs]
            v_wind_speed = self.v_wind_speed[:, idxs]
            pressure = self.pressure[:, idxs]
            air_temperature = self.air_temperature[:, idxs]
            salinity = self.salinity[:, idxs]
            water_temperature = self.water_temperature[:, idxs]
            status = self.status[:, idxs]
            message = self.message[:, idxs]
        else:
            wave_height = self.wave_height[row_start:row_end, col_start:col_end]
            wave_direction = self.wave_direction[row_start:row_end, col_start:col_end]
            wave_period = self.wave_period[row_start:row_end, col_start:col_end]
            u_currents = self.u_currents[row_start:row_end, col_start:col_end]
            v_currents = self.v_currents[row_start:row_end, col_start:col_end]
            u_wind_speed = self.u_wind_speed[row_start:row_end, col_start:col_end]
            v_wind_speed = self.v_wind_speed[row_start:row_end, col_start:col_end]
            pressure = self.pressure[row_start:row_end, col_start:col_end]
            air_temperature = self.air_temperature[row_start:row_end, col_start:col_end]
            salinity = self.salinity[row_start:row_end, col_start:col_end]
            water_temperature = self.water_temperature[row_start:row_end, col_start:col_end]
            status = self.status[row_start:row_end, col_start:col_end]
            message = self.message[row_start:row_end, col_start:col_end]

        wp = WeatherParams(
            wave_height=wave_height,
            wave_direction=wave_direction,
            wave_period=wave_period,
            u_currents=u_currents,
            v_currents=v_currents,
            u_wind_speed=u_wind_speed,
            v_wind_speed=v_wind_speed,
            pressure=pressure,
            air_temperature=air_temperature,
            salinity=salinity,
            water_temperature=water_temperature,
            status=status,
            message=message
        )
        return wp
