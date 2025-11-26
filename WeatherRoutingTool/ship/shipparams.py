import logging

import numpy as np
from astropy import units as u
from WeatherRoutingTool.ship.weatherparams import WeatherParams

logger = logging.getLogger('WRT.ship')


class ShipParams():
    """
    Container for ship performance parameters.
    
    Weather parameters are now separated into a WeatherParams object, but backward
    compatibility is maintained through property accessors.
    """
    fuel_rate: np.ndarray  # (kg/s)
    power: np.ndarray  # (W)
    rpm: np.ndarray  # (rpm)
    speed: np.ndarray  # (m/s)
    r_calm: np.ndarray  # (N)
    r_wind: np.ndarray  # (N)
    r_waves: np.ndarray  # (N)
    r_shallow: np.ndarray  # (N)
    r_roughness: np.ndarray  # (N)
    
    # Weather parameters are now stored in separate WeatherParams object
    weather: WeatherParams

    fuel_type: str

    def __init__(self, fuel_rate, power, rpm, speed, r_calm, r_wind, r_waves, r_shallow, r_roughness, wave_height,
                 wave_direction, wave_period, u_currents, v_currents, u_wind_speed, v_wind_speed, pressure,
                 air_temperature, salinity, water_temperature, status, message, weather=None):
        """
        Initialize ShipParams.
        
        Args:
            weather: Optional WeatherParams object. If provided, individual weather parameters are ignored.
                    If not provided, weather parameters are created from individual arguments (backward compatible).
        """
        # Ship-specific parameters
        self.fuel_rate = fuel_rate
        self.power = power
        self.rpm = rpm
        self.speed = speed
        self.r_calm = r_calm
        self.r_wind = r_wind
        self.r_waves = r_waves
        self.r_shallow = r_shallow
        self.r_roughness = r_roughness

        # Weather parameters - use provided WeatherParams or create from individual params
        if weather is not None:
            self.weather = weather
        else:
            # Backward compatibility: create WeatherParams from individual parameters
            self.weather = WeatherParams(
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

        self.fuel_type = 'HFO'

    # Property accessors for backward compatibility - delegate to weather object
    @property
    def wave_height(self):
        return self.weather.wave_height
    
    @wave_height.setter
    def wave_height(self, value):
        self.weather.wave_height = value
    
    @property
    def wave_direction(self):
        return self.weather.wave_direction
    
    @wave_direction.setter
    def wave_direction(self, value):
        self.weather.wave_direction = value
    
    @property
    def wave_period(self):
        return self.weather.wave_period
    
    @wave_period.setter
    def wave_period(self, value):
        self.weather.wave_period = value
    
    @property
    def u_currents(self):
        return self.weather.u_currents
    
    @u_currents.setter
    def u_currents(self, value):
        self.weather.u_currents = value
    
    @property
    def v_currents(self):
        return self.weather.v_currents
    
    @v_currents.setter
    def v_currents(self, value):
        self.weather.v_currents = value
    
    @property
    def u_wind_speed(self):
        return self.weather.u_wind_speed
    
    @u_wind_speed.setter
    def u_wind_speed(self, value):
        self.weather.u_wind_speed = value
    
    @property
    def v_wind_speed(self):
        return self.weather.v_wind_speed
    
    @v_wind_speed.setter
    def v_wind_speed(self, value):
        self.weather.v_wind_speed = value
    
    @property
    def pressure(self):
        return self.weather.pressure
    
    @pressure.setter
    def pressure(self, value):
        self.weather.pressure = value
    
    @property
    def air_temperature(self):
        return self.weather.air_temperature
    
    @air_temperature.setter
    def air_temperature(self, value):
        self.weather.air_temperature = value
    
    @property
    def salinity(self):
        return self.weather.salinity
    
    @salinity.setter
    def salinity(self, value):
        self.weather.salinity = value
    
    @property
    def water_temperature(self):
        return self.weather.water_temperature
    
    @water_temperature.setter
    def water_temperature(self, value):
        self.weather.water_temperature = value
    
    @property
    def status(self):
        return self.weather.status
    
    @status.setter
    def status(self, value):
        self.weather.status = value
    
    @property
    def message(self):
        return self.weather.message
    
    @message.setter
    def message(self, value):
        self.weather.message = value

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
        logger.info('fuel_rate: ' + str(self.fuel_rate.value) + ' ' + self.fuel_rate.unit.to_string())
        logger.info('rpm: ' + str(self.rpm.value) + ' ' + self.rpm.unit.to_string())
        logger.info('power: ' + str(self.power.value) + ' ' + self.power.unit.to_string())
        logger.info('speed: ' + str(self.speed.value) + ' ' + self.speed.unit.to_string())
        logger.info('r_calm: ' + str(self.r_calm.value) + ' ' + self.r_calm.unit.to_string())
        logger.info('r_wind: ' + str(self.r_wind.value) + ' ' + self.r_wind.unit.to_string())
        logger.info('r_waves: ' + str(self.r_waves.value) + ' ' + self.r_waves.unit.to_string())
        logger.info('r_shallow: ' + str(self.r_shallow.value) + ' ' + self.r_shallow.unit.to_string())
        logger.info('r_roughness: ' + str(self.r_roughness.value) + ' ' + self.r_roughness.unit.to_string())
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
        logger.info('fuel_type: ' + str(self.fuel_type))

    def print_shape(self):
        logger.info('fuel_rate: ' + str(self.fuel_rate.shape))
        logger.info('rpm: ' + str(self.rpm.shape))
        logger.info('power: ' + str(self.power.shape))
        logger.info('speed: ' + str(self.speed.shape))
        logger.info('r_calm: ' + str(self.r_calm.shape))
        logger.info('r_wind: ' + str(self.r_wind.shape))
        logger.info('r_waves: ' + str(self.r_waves.shape))
        logger.info('r_shallow: ' + str(self.r_shallow.shape))
        logger.info('r_roughness: ' + str(self.r_roughness.shape))
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
        # Ship parameters
        self.speed = np.repeat(self.speed, courses_segments + 1, axis=1)
        self.fuel_rate = np.repeat(self.fuel_rate, courses_segments + 1, axis=1)
        self.power = np.repeat(self.power, courses_segments + 1, axis=1)
        self.rpm = np.repeat(self.rpm, courses_segments + 1, axis=1)
        self.r_calm = np.repeat(self.r_calm, courses_segments + 1, axis=1)
        self.r_wind = np.repeat(self.r_wind, courses_segments + 1, axis=1)
        self.r_waves = np.repeat(self.r_waves, courses_segments + 1, axis=1)
        self.r_shallow = np.repeat(self.r_shallow, courses_segments + 1, axis=1)
        self.r_roughness = np.repeat(self.r_roughness, courses_segments + 1, axis=1)
        
        # Weather parameters - delegate to weather object
        self.weather.define_courses(courses_segments)

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
        """Select specific indices from the arrays."""
        # Ship parameters
        self.speed = self.speed[:, idxs]
        self.fuel_rate = self.fuel_rate[:, idxs]
        self.power = self.power[:, idxs]
        self.rpm = self.rpm[:, idxs]
        self.r_wind = self.r_wind[:, idxs]
        self.r_calm = self.r_calm[:, idxs]
        self.r_waves = self.r_waves[:, idxs]
        self.r_shallow = self.r_shallow[:, idxs]
        self.r_roughness = self.r_roughness[:, idxs]
        
        # Weather parameters - delegate to weather object
        self.weather.select(idxs)

    def flip(self):
        """Flip and append dummy values to arrays."""
        # Ship parameters - remove last, flip, append dummy
        self.speed = self.speed[:-1]
        self.fuel_rate = self.fuel_rate[:-1]
        self.power = self.power[:-1]
        self.rpm = self.rpm[:-1]
        self.r_wind = self.r_wind[:-1]
        self.r_calm = self.r_calm[:-1]
        self.r_waves = self.r_waves[:-1]
        self.r_shallow = self.r_shallow[:-1]
        self.r_roughness = self.r_roughness[:-1]

        self.speed = np.flip(self.speed, 0)
        self.fuel_rate = np.flip(self.fuel_rate, 0)
        self.power = np.flip(self.power, 0)
        self.rpm = np.flip(self.rpm, 0)
        self.r_wind = np.flip(self.r_wind, 0)
        self.r_calm = np.flip(self.r_calm, 0)
        self.r_waves = np.flip(self.r_waves, 0)
        self.r_shallow = np.flip(self.r_shallow, 0)
        self.r_roughness = np.flip(self.r_roughness, 0)

        self.speed = np.append(self.speed, -99 * self.speed.unit)
        self.fuel_rate = np.append(self.fuel_rate, -99 * self.fuel_rate.unit)
        self.power = np.append(self.power, -99 * self.power.unit)
        self.rpm = np.append(self.rpm, -99 * self.rpm.unit)
        self.r_wind = np.append(self.r_wind, -99 * self.r_wind.unit)
        self.r_calm = np.append(self.r_calm, -99 * self.r_calm.unit)
        self.r_waves = np.append(self.r_waves, -99 * self.r_waves.unit)
        self.r_shallow = np.append(self.r_shallow, -99 * self.r_shallow.unit)
        self.r_roughness = np.append(self.r_roughness, -99 * self.r_roughness.unit)
        
        # Weather parameters - delegate to weather object
        self.weather.flip()
        self.weather.append_dummy()

    def expand_axis_for_intermediate(self):
        """Expand axis for intermediate waypoints."""
        # Ship parameters
        self.speed = np.expand_dims(self.speed, axis=1)
        self.fuel_rate = np.expand_dims(self.fuel_rate, axis=1)
        self.power = np.expand_dims(self.power, axis=1)
        self.rpm = np.expand_dims(self.rpm, axis=1)
        self.r_wind = np.expand_dims(self.r_wind, axis=1)
        self.r_calm = np.expand_dims(self.r_calm, axis=1)
        self.r_waves = np.expand_dims(self.r_waves, axis=1)
        self.r_shallow = np.expand_dims(self.r_shallow, axis=1)
        self.r_roughness = np.expand_dims(self.r_roughness, axis=1)
        
        # Weather parameters
        self.weather.wave_height = np.expand_dims(self.weather.wave_height, axis=1)
        self.weather.wave_direction = np.expand_dims(self.weather.wave_direction, axis=1)
        self.weather.wave_period = np.expand_dims(self.weather.wave_period, axis=1)
        self.weather.u_currents = np.expand_dims(self.weather.u_currents, axis=1)
        self.weather.v_currents = np.expand_dims(self.weather.v_currents, axis=1)
        self.weather.u_wind_speed = np.expand_dims(self.weather.u_wind_speed, axis=1)
        self.weather.v_wind_speed = np.expand_dims(self.weather.v_wind_speed, axis=1)
        self.weather.pressure = np.expand_dims(self.weather.pressure, axis=1)
        self.weather.air_temperature = np.expand_dims(self.weather.air_temperature, axis=1)
        self.weather.salinity = np.expand_dims(self.weather.salinity, axis=1)
        self.weather.water_temperature = np.expand_dims(self.weather.water_temperature, axis=1)
        self.weather.status = np.expand_dims(self.weather.status, axis=1)
        self.weather.message = np.expand_dims(self.weather.message, axis=1)

    def get_element(self, idx):
        try:
            speed = self.speed[idx]
            fuel_rate = self.fuel_rate[idx]
            power = self.power[idx]
            rpm = self.rpm[idx]
            r_wind = self.r_wind[idx]
            r_calm = self.r_calm[idx]
            r_waves = self.r_waves[idx]
            r_shallow = self.r_shallow[idx]
            r_roughness = self.r_roughness[idx]
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
        except ValueError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.speed.shape[0]))
        return (fuel_rate, power, rpm, speed, r_wind, r_calm, r_waves, r_shallow, r_roughness, wave_height,
                wave_direction, wave_period, u_currents, v_currents, u_wind_speed, v_wind_speed, pressure,
                air_temperature, salinity, water_temperature, status, message)

    def get_single_object(self, idx):
        # ToDo: reuse get_element here
        try:
            speed = self.speed[idx]
            fuel_rate = self.fuel_rate[idx]
            power = self.power[idx]
            rpm = self.rpm[idx]
            r_wind = self.r_wind[idx]
            r_calm = self.r_calm[idx]
            r_waves = self.r_waves[idx]
            r_shallow = self.r_shallow[idx]
            r_roughness = self.r_roughness[idx]
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

        except ValueError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.speed.shape[0]))

        sp = ShipParams(
            fuel_rate=fuel_rate,
            power=power,
            rpm=rpm,
            speed=speed,
            r_wind=r_wind,
            r_calm=r_calm,
            r_waves=r_waves,
            r_shallow=r_shallow,
            r_roughness=r_roughness,
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
        return sp

    def get_reduced_2D_object(self, row_start=None, row_end=None, col_start=None, col_end=None, idxs=None):

        try:
            if idxs is None:
                speed = self.speed[row_start:row_end, col_start:col_end]
                fuel_rate = self.fuel_rate[row_start:row_end, col_start:col_end]
                power = self.power[row_start:row_end, col_start:col_end]
                rpm = self.rpm[row_start:row_end, col_start:col_end]
                r_wind = self.r_wind[row_start:row_end, col_start:col_end]
                r_calm = self.r_calm[row_start:row_end, col_start:col_end]
                r_waves = self.r_waves[row_start:row_end, col_start:col_end]
                r_shallow = self.r_shallow[row_start:row_end, col_start:col_end]
                r_roughness = self.r_roughness[row_start:row_end, col_start:col_end]
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
            else:
                speed = self.speed[:, idxs]
                fuel_rate = self.fuel_rate[:, idxs]
                power = self.power[:, idxs]
                rpm = self.rpm[:, idxs]
                r_wind = self.r_wind[:, idxs]
                r_calm = self.r_calm[:, idxs]
                r_waves = self.r_waves[:, idxs]
                r_shallow = self.r_shallow[:, idxs]
                r_roughness = self.r_roughness[:, idxs]
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
        except ValueError:
            raise ValueError(
                'Index ' + str(col_start) + ' is not available for array with length ' + str(self.speed.shape[0]))

        sp = ShipParams(
            fuel_rate=fuel_rate,
            power=power,
            rpm=rpm,
            speed=speed,
            r_wind=r_wind,
            r_calm=r_calm,
            r_waves=r_waves,
            r_shallow=r_shallow,
            r_roughness=r_roughness,
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
        return sp
