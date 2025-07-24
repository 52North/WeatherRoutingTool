import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.ship_config import ShipConfig

logger = logging.getLogger('WRT.ship')


class DirectPowerBoat(Boat):
    """
           estimates power & fuel consumption based on the so-called Direct Power Method

           The following approximations are used:
               - a fixed working point of 75% SMCR power and an average ship speed is assumed
                 (Currently it is only possible to travel at this fixed working point. No deviating speeds can be set.)
               - additional power and fuel consumption is derived from added resistances of the environmental conditions
               - currently only the wind resistance is considered; the wind resistance coefficient is calculated using
                 the Fujiwara approximation

               Returns:
                   ship_params  - ShipParams object containing ship parameters like power consumption and fuel rate
    """

    power_at_sp: float  # power at the service propulsion point
    eta_prop: float  # propulsion efficiency
    overload_factor: float  # overload factor
    head_wind_coeff: float  # wind coefficient for head wind (psi = 0°)
    fuel_rate: float  # fuel rate

    # ship geometry
    Axv: float
    Ayv: float
    Aod: float
    length: float
    breadth: float
    hs1: float
    hs2: float
    ls1: float
    ls2: float
    bs1: float
    cmc: float
    hbr: float
    hc: float

    air_mass_density: float  # air mass density

    def __init__(self, init_mode='from_file', file_name=None, config_dict=None):
        super().__init__(init_mode, file_name, config_dict)
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig.assign_config(Path(file_name))
        else:
            config_obj = ShipConfig.assign_config(init_mode='from_dict', config_dict=config_dict)
        print(config_obj.model_dump(exclude_unset=True))

        # mandatory parameters for direct power method
        # determine power at the service propulsion point i.e. 'subtract' 15% sea and 10% engine margin
        self.power_at_sp = config_obj.BOAT_SMCR_POWER * u.kiloWatt
        self.power_at_sp = self.power_at_sp.to(u.Watt) * 0.75
        self.speed_at_sp = config_obj.BOAT_SMCR_SPEED * u.meter/u.second

        self.eta_prop = config_obj.BOAT_PROPULSION_EFFICIENCY
        self.overload_factor = config_obj.BOAT_OVERLOAD_FACTOR
        self.head_wind_coeff = 1

        self.Axv = config_obj.BOAT_AXV * u.meter * u.meter
        self.Ayv = config_obj.BOAT_AYV * u.meter * u.meter
        self.Aod = config_obj.BOAT_AOD * u.meter * u.meter
        self.length = config_obj.BOAT_LENGTH * u.meter
        self.breadth = config_obj.BOAT_BREADTH * u.meter
        self.hs1 = config_obj.BOAT_HS1 * u.meter
        self.hs2 = config_obj.BOAT_HS2 * u.meter
        self.ls1 = config_obj.BOAT_LS1 * u.meter
        self.ls2 = config_obj.BOAT_LS2 * u.meter
        self.bs1 = config_obj.BOAT_BS1 * u.meter
        self.cmc = config_obj.BOAT_CMC * u.meter
        self.hbr = config_obj.BOAT_HBR * u.meter
        self.hc = config_obj.BOAT_HC * u.meter
        self.fuel_rate = config_obj.BOAT_FUEL_RATE * u.gram / (u.kiloWatt * u.hour)
        self.fuel_rate = self.fuel_rate.to(u.kg / (u.Watt * u.second))

        self.weather_path = config_obj.WEATHER_DATA
        self.air_mass_density = config_obj.AIR_MASS_DENSITY * u.kg / (u.meter * u.meter * u.meter)

    def interpolate_to_true_speed(self, power):
        const = power/(self.speed_at_sp**(3))
        power_interpolated = const * self.speed**(3)
        return power_interpolated

    def load_data(self):
        self.calculate_ship_geometry()
        self.calculate_head_wind_coeff()

        logger.info(form.get_log_step('The boat speed provided is assumed to be the speed that corresponds '
                                      'to 75% SMCR power.'))

    def check_data_meaningful(self):
        data = ['Axv', 'Ayv', 'Aod', 'length', 'breadth', 'hs1', 'hs2', 'ls1', 'ls2', 'bs1', 'cmc', 'hbr', 'hc']
        for d in data:
            value = getattr(self, d, None)
            if value is None or value == -99:
                logger.info(f"The ship attribute {value} has no meaningful value")

    def set_optional_parameter(self, par_string, par):
        approx_pars = {
            'hs1': 0.2 * self.hbr,
            'ls1': 0.2 * self.length,
            'hs2': 0.1 * self.hbr,
            'ls2': 0.3 * self.length,
            'cmc': -0.035 * self.length,
            'bs1': 0.9 * self.breadth,
            'hc': 10 * u.meter
        }
        if par < 0:
            par = approx_pars[par_string]
        return par

    def calculate_ship_geometry(self):
        # check for provided parameters
        self.hs1 = self.set_optional_parameter('hs1', self.hs1)
        self.ls1 = self.set_optional_parameter('ls1', self.ls1)
        self.hs2 = self.set_optional_parameter('hs2', self.hs2)
        self.ls2 = self.set_optional_parameter('ls2', self.ls2)
        self.cmc = self.set_optional_parameter('cmc', self.cmc)
        self.bs1 = self.set_optional_parameter('bs1', self.bs1)
        self.hc = self.set_optional_parameter('hc', self.hc)

        if self.Axv < 0:
            self.Axv = self.hbr * self.breadth + self.hs1 * self.bs1 - self.hs1 * self.breadth
        if self.Ayv < 0:
            self.Ayv = (self.hs1 * self.ls1 + self.hs2 * self.ls2 + self.hbr * self.length - 1 / 2 * self.hbr * self.hbr
                        - self.hs1 * self.length)
        if self.Aod < 0:
            self.Aod = self.hs1 * self.ls1 + self.hs2 * self.ls2

    def calculate_head_wind_coeff(self):
        """
            calculate wind coefficient for head wind (psi = 0°)
        """
        wind_fac = self.get_wind_factors_small_angle(0)
        self.head_wind_coeff = self.get_wind_coeff(0, wind_fac['CLF'], wind_fac['CXLI'], wind_fac['CALF'])

    def evaluate_resistance(self, ship_params, courses):
        r_wind = self.get_wind_resistance(ship_params.u_wind_speed, ship_params.v_wind_speed, courses)
        r_waves = self.get_wave_resistance(ship_params, ship_params.wave_height, ship_params.wave_direction,
                                           ship_params.wave_period)
        ship_params.r_wind = r_wind["r_wind"]
        ship_params.r_waves = r_waves

        return ship_params

    def get_wind_dir(self, u_wind_speed, v_wind_speed):
        """
            calculate true wind direction in degree from u and v
        """
        wind_dir = (180 * u.degree + 180 * u.degree / math.pi * np.arctan2(u_wind_speed.value, v_wind_speed.value)) % (
                360 * u.degree)
        return wind_dir

    def get_relative_wind_dir(self, ang_boat, ang_wind):
        """
            calculate relative wind direction [0°,180°] between ship course and true wind direction

            - head wind: 0°
            - tail wind: 180°
        """

        delta_ang = ang_wind - ang_boat

        delta_ang[delta_ang < 0 * u.degree] = abs(delta_ang[delta_ang < 0 * u.degree])
        delta_ang[delta_ang > 180 * u.degree] = abs(360 * u.degree - delta_ang[delta_ang > 180 * u.degree])

        return delta_ang

    def get_apparent_wind(self, true_wind_speed, true_wind_angle):
        """
            calculate apparent wind speed from true wind and ship course
        """
        apparent_wind_speed = (self.speed_at_sp * self.speed_at_sp + true_wind_speed * true_wind_speed
                               + 2.0 * self.speed_at_sp * true_wind_speed * np.cos(np.radians(true_wind_angle)))
        apparent_wind_speed = np.sqrt(apparent_wind_speed)

        angle_rad = np.radians(true_wind_angle.value)
        apparent_wind_angle = np.full(true_wind_angle.shape, - 99) * u.radian

        for iang in range(0, true_wind_speed.shape[0]):
            arg_arcsin = true_wind_speed[iang] * np.sin(np.radians(true_wind_angle[iang])) / apparent_wind_speed[
                iang] * u.radian

            # catch it if argument of arcsin is > 1 due to rounding issues but make sure to apply this only for
            # rounding issues
            diff_to_one = arg_arcsin - 1 * u.radian
            if diff_to_one > 0:
                assert diff_to_one < 0.000001 * u.radian
                arg_arcsin = 1 * u.radian

            if apparent_wind_speed[iang] > 0:
                apparent_wind_angle[iang] = np.arcsin(arg_arcsin.value) * u.radian
            else:
                apparent_wind_angle[iang] = 0 * u.radian

            # catch it if psi > 90° as arcsin is only defined for 0 < psi < 90°
            # - calculate true wind angle 'true_ang_perp' for which apparent wind angle is 90°
            # - if true wind angle is larger than 'true_ang_perp', subtract pi from apparent wind angle
            # - apparent wind angle is always < 90° if boat speed > true wind speed; skip correction here
            arg_arccos = self.speed_at_sp / true_wind_speed[iang]
            if arg_arccos > 1:
                continue
            true_ang_perp = np.pi * u.radian - np.arccos(self.speed_at_sp / true_wind_speed[iang])
            if angle_rad[iang] * u.radian > true_ang_perp:
                apparent_wind_angle[iang] = np.pi * u.radian - apparent_wind_angle[iang]

            if np.isnan(apparent_wind_angle[iang]):
                print('true_wind_speed: ', true_wind_speed[iang])
                print('apparent_wind_speed: ', apparent_wind_speed[iang])
                print('true_wind_angle: ', true_wind_angle[iang])
                print('true_ang_perp: ', true_ang_perp)
                print('angle_rad: ', angle_rad[iang])
                print('arg_arcsin: ', arg_arcsin)
                print('arg_arccos: ', arg_arccos)
                raise ValueError('Apparent wind angle is nan!')

        apparent_wind_angle = apparent_wind_angle.to(u.degree)

        return {'app_wind_speed': apparent_wind_speed, 'app_wind_angle': apparent_wind_angle}

    def get_wind_factors_small_angle(self, psi):
        """
            calculate factors CLF, CXLI and CALF for psi < 90°
        """
        beta10 = 0.922
        beta11 = -0.507
        beta12 = -1.162
        delta10 = -0.458
        delta11 = -3.245
        delta12 = 2.313
        eta10 = 0.585
        eta11 = 0.906
        eta12 = -3.239

        CLF = -99
        CXLI = -99
        CALF = -99

        CLF = beta10 + beta11 * self.Ayv / (self.length * self.breadth) + beta12 * self.cmc / self.length
        CXLI = delta10 + delta11 * self.Ayv / (self.length * self.hbr) + delta12 * self.Axv / (self.breadth * self.hbr)
        CALF = eta10 + eta11 * self.Aod / self.Ayv + eta12 * self.breadth / self.length

        return {'CLF': CLF, 'CXLI': CXLI, 'CALF': CALF}

    def get_wind_factors_large_angle(self, psi):
        """
            calculate factors CLF, CXLI and CALF for psi > 90°
        """

        beta20 = -0.018
        beta21 = 5.091
        beta22 = -10.367
        beta23 = 3.011
        beta24 = 0.341
        delta20 = 1.901
        delta21 = -12.727
        delta22 = -24.407
        delta23 = 40.310
        delta24 = 5.481
        eta20 = 0.314
        eta21 = 1.117

        CLF = -99
        CXLI = -99
        CALF = -99

        CLF = beta20 + beta21 * self.breadth / self.length + beta22 * self.hc / self.length + beta23 * self.Aod / (
                self.length * self.length) + beta24 * self.Axv / (self.breadth * self.breadth)
        CXLI = (delta20 + delta21 * self.Ayv / (
                self.length * self.hbr) + delta22 * self.Axv / self.Ayv + delta23 * self.breadth / self.length + delta24
                * self.Axv / (self.breadth * self.hbr))
        CALF = eta20 + eta21 * self.Aod / self.Ayv

        return {'CLF': CLF, 'CXLI': CXLI, 'CALF': CALF}

    def get_wind_coeff(self, psi_deg, CLF, CXLI, CALF):
        """
            calculate wind coefficient C_AA
        """

        psi = math.radians(psi_deg)

        sinpsi = math.sin(psi)
        cospsi = math.cos(psi)

        CAA = (CLF * cospsi +
               CXLI * (sinpsi - 1 / 2 * sinpsi * cospsi * cospsi) *
               sinpsi * cospsi + CALF * sinpsi * cospsi * cospsi * cospsi)

        return CAA

    def get_wind_resistance(self, u_wind_speed, v_winds_speed, courses):
        """
            calculate wind resistance r_wind
        """

        wind_fac = None
        wind_coeff_arr = []

        true_wind_speed = np.sqrt(
            u_wind_speed.value * u_wind_speed.value + v_winds_speed.value * v_winds_speed.value) * u.meter / u.second

        true_wind_dir = self.get_wind_dir(u_wind_speed, v_winds_speed)
        true_wind_dir = self.get_relative_wind_dir(courses, true_wind_dir)
        apparent_wind = self.get_apparent_wind(true_wind_speed, true_wind_dir)

        for psi in apparent_wind['app_wind_angle']:
            psi = psi.value
            if psi >= 0 and psi < 90:
                wind_fac = self.get_wind_factors_small_angle(psi)
            if psi > 90 and psi <= 180:
                wind_fac = self.get_wind_factors_large_angle(psi)
            if psi == 90:
                wind_fac_small = self.get_wind_factors_small_angle(psi - 10)
                wind_fac_large = self.get_wind_factors_large_angle(psi + 10)
                wind_coeff = 1 / 2 * (
                        self.get_wind_coeff(psi, wind_fac_small['CLF'], wind_fac_small['CXLI'],
                                            wind_fac_small['CALF']) +
                        self.get_wind_coeff(psi, wind_fac_large['CLF'], wind_fac_large['CXLI'], wind_fac_large['CALF'])
                )
            else:
                wind_coeff = self.get_wind_coeff(psi, wind_fac['CLF'], wind_fac['CXLI'], wind_fac['CALF'])
            wind_coeff_arr.append(wind_coeff)

        wind_coeff_arr = np.array(wind_coeff_arr)
        r_wind = (1 / 2 * self.air_mass_density * wind_coeff_arr * self.Axv * apparent_wind['app_wind_speed']
                  * apparent_wind['app_wind_speed'])

        return {"r_wind": r_wind.to(u.Newton), "wind_coeff": wind_coeff_arr}

    def get_wave_resistance(self, ship_params, wave_height, wave_direction, wave_period):
        """
            calculate wave resistance r_wave (not yet implemented)
        """

        n_coords = len(wave_height)
        dummy_array = np.full(n_coords, 0)
        return dummy_array * u.Newton

    def get_power(self, deltaR):
        Plin = deltaR * self.speed_at_sp / self.eta_prop
        P = self.power_at_sp * (Plin + self.power_at_sp) / (Plin * self.overload_factor + self.power_at_sp)
        return P

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False
        n_requests = len(courses)

        # initialise clean ship params object
        dummy_array = np.full(n_requests, -99)
        speed_array = np.full(n_requests, self.speed_at_sp)

        ship_params = ShipParams(
            fuel_rate=dummy_array * u.kg / u.s,
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
        # calculate added resistances & update ShipParams object respectively; update also for environmental conditions
        ship_params = self.evaluate_weather(ship_params, lats, lons, time)
        ship_params = self.evaluate_resistance(ship_params, courses)
        added_resistance = ship_params.r_wind + ship_params.r_waves

        P = self.get_power(added_resistance)
        P = self.interpolate_to_true_speed(P)

        ship_params.power = P
        ship_params.fuel_rate = self.fuel_rate * P

        if debug:
            ship_params.print()

        return ship_params
