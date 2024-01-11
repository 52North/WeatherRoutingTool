import logging

import numpy as np

logger = logging.getLogger('WRT.ship')


class ShipParams():
    fuel_rate: np.ndarray  # (kg/s)
    power: np.ndarray  # (W)
    rpm: np.ndarray  # (Hz)
    speed: np.ndarray  # (m/s)
    r_calm: np.ndarray  # (N)
    r_wind: np.ndarray  # (N)
    r_waves: np.ndarray  # (N)
    r_shallow: np.ndarray  # (N)
    r_roughness: np.ndarray  # (N)
    fuel_type: str

    def __init__(self, fuel_rate, power, rpm, speed, r_calm, r_wind, r_waves, r_shallow, r_roughness):
        self.fuel_rate = fuel_rate
        self.power = power
        self.rpm = rpm
        self.speed = speed
        self.r_calm = r_calm
        self.r_wind = r_wind
        self.r_waves = r_waves
        self.r_shallow = r_shallow
        self.r_roughness = r_roughness

        self.fuel_type = 'HFO'

    @classmethod
    def set_default_array(cls):
        return cls(speed=np.array([[0]]), fuel_rate=np.array([[0]]), power=np.array([[0]]), rpm=np.array([[0]]),
                   r_calm=np.array([[0]]), r_wind=np.array([[0]]), r_waves=np.array([[0]]), r_shallow=np.array([[0]]),
                   r_roughness=np.array([[0]]))

    @classmethod
    def set_default_array_1D(cls, ncoorinate_points):
        return cls(speed=np.full(shape=ncoorinate_points, fill_value=0),
                   fuel_rate=np.full(shape=ncoorinate_points, fill_value=0),
                   power=np.full(shape=ncoorinate_points, fill_value=0),
                   rpm=np.full(shape=ncoorinate_points, fill_value=0),
                   r_calm=np.full(shape=ncoorinate_points, fill_value=0),
                   r_wind=np.full(shape=ncoorinate_points, fill_value=0),
                   r_waves=np.full(shape=ncoorinate_points, fill_value=0),
                   r_shallow=np.full(shape=ncoorinate_points, fill_value=0),
                   r_roughness=np.full(shape=ncoorinate_points, fill_value=0), )

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
        logger.info('fuel_type: ' + str(self.fuel_type))

    def print_shape(self):
        logger.info('fuel_rate: ', self.fuel_rate.shape)
        logger.info('rpm: ', self.rpm.shape)
        logger.info('power: ', self.power.shape)
        logger.info('speed: ', self.speed.shape)
        logger.info('r_calm: ', self.r_calm.shape)
        logger.info('r_wind: ', self.r_wind.shape)
        logger.info('r_waves: ', self.r_waves.shape)
        logger.info('r_shallow: ', self.r_shallow.shape)
        logger.info('r_roughness: ', self.r_roughness.shape)

    def define_variants(self, variant_segments):
        self.speed = np.repeat(self.speed, variant_segments + 1, axis=1)
        self.fuel_rate = np.repeat(self.fuel_rate, variant_segments + 1, axis=1)
        self.power = np.repeat(self.power, variant_segments + 1, axis=1)
        self.rpm = np.repeat(self.rpm, variant_segments + 1, axis=1)
        self.r_calm = np.repeat(self.r_calm, variant_segments + 1, axis=1)
        self.r_wind = np.repeat(self.r_wind, variant_segments + 1, axis=1)
        self.r_waves = np.repeat(self.r_waves, variant_segments + 1, axis=1)
        self.r_shallow = np.repeat(self.r_shallow, variant_segments + 1, axis=1)
        self.r_roughness = np.repeat(self.r_roughness, variant_segments + 1, axis=1)

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

    def select(self, idxs):
        self.speed = self.speed[:, idxs]
        self.fuel_rate = self.fuel_rate[:, idxs]
        self.power = self.power[:, idxs]
        self.rpm = self.rpm[:, idxs]
        self.r_wind = self.r_wind[:, idxs]
        self.r_calm = self.r_calm[:, idxs]
        self.r_waves = self.r_waves[:, idxs]
        self.r_shallow = self.r_shallow[:, idxs]
        self.r_roughness = self.r_roughness[:, idxs]

    def flip(self):
        # should be replaced by more careful implementation
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

    def expand_axis_for_intermediate(self):
        self.speed = np.expand_dims(self.speed, axis=1)
        self.fuel_rate = np.expand_dims(self.fuel_rate, axis=1)
        self.power = np.expand_dims(self.power, axis=1)
        self.rpm = np.expand_dims(self.rpm, axis=1)
        self.r_wind = np.expand_dims(self.r_wind, axis=1)
        self.r_calm = np.expand_dims(self.r_calm, axis=1)
        self.r_waves = np.expand_dims(self.r_waves, axis=1)
        self.r_shallow = np.expand_dims(self.r_shallow, axis=1)
        self.r_roughness = np.expand_dims(self.r_roughness, axis=1)

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
        except ValueError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.speed.shape[0]))
        return fuel_rate, power, rpm, speed, r_wind, r_calm, r_waves, r_shallow, r_roughness

    def get_single_object(self, idx):
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
        except ValueError:
            raise ValueError(
                'Index ' + str(idx) + ' is not available for array with length ' + str(self.speed.shape[0]))

        sp = ShipParams(fuel_rate=fuel_rate, power=power, rpm=rpm, speed=speed, r_wind=r_wind, r_calm=r_calm, r_waves=r_waves,
                        r_shallow=r_shallow, r_roughness=r_roughness)
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
        except ValueError:
            raise ValueError(
                'Index ' + str(col_start) + ' is not available for array with length ' + str(self.speed.shape[0]))

        sp = ShipParams(fuel=fuel_rate, power=power, rpm=rpm, speed=speed, r_wind=r_wind, r_calm=r_calm, r_waves=r_waves,
                        r_shallow=r_shallow, r_roughness=r_roughness)
        return sp
