import copy
import logging
import math
import os
import sys

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from astropy import units as u

import mariPower
import WeatherRoutingTool.utils.formatting as form
import WeatherRoutingTool.utils.unit_conversion as units
from mariPower import __main__
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.shipparams import ShipParams

logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.

class Boat:
    speed: float  # boat speed in m/s
    weather_path: str  # path to netCDF containing weather data

    def __init__(self, init_mode = 'from_file', file_name=None, config_dict = None):
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig(file_name = file_name)
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

    def __init__(self, init_mode = 'from_file', file_name=None, config_dict = None):
        super().__init__(init_mode, file_name, config_dict)
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig(file_name=file_name)
        else:
            config_obj = ShipConfig(init_mode='from_dict', config_dict=config_dict)
        config_obj.print()

        # mandatory parameters for direct power method
        # determine power at the service propulsion point i.e. 'subtract' 15% sea and 10% engine margin
        self.power_at_sp = config_obj.BOAT_SMCR_POWER * u.kiloWatt
        self.power_at_sp = self.power_at_sp.to(u.Watt) * 0.75

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

    def load_data(self):
        self.calculate_ship_geometry()
        self.calculate_head_wind_coeff()

        logger.info(form.get_log_step('The boat speed provided is assumed to be the speed that corresponds '
                                      'to 75% SMCR power.'))

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
        apparent_wind_speed = (self.speed * self.speed + true_wind_speed * true_wind_speed
                               + 2.0 * self.speed * true_wind_speed * np.cos(np.radians(true_wind_angle)))
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
            arg_arccos = self.speed / true_wind_speed[iang]
            if arg_arccos > 1:
                continue
            true_ang_perp = np.pi * u.radian - np.arccos(self.speed / true_wind_speed[iang])
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
        Plin = deltaR * self.speed / self.eta_prop
        P = self.power_at_sp * (Plin + self.power_at_sp) / (Plin * self.overload_factor + self.power_at_sp)
        return P

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False
        n_requests = len(courses)

        # initialise clean ship params object
        dummy_array = np.full(n_requests, -99)
        speed_array = np.full(n_requests, self.speed)

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
        ship_params.power = P
        ship_params.fuel_rate = self.fuel_rate * P

        if debug:
            ship_params.print()

        return ship_params

# FIXME: Decide whether this consumption model is still needed.
class ConstantFuelBoat(Boat):
    fuel_rate: float  # dummy value for fuel_rate that is returned
    speed: float  # boat speed



    def __init__(self, init_mode = 'from_file', file_name=None, config_dict = None):
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


##
# Class implementing connection to mariPower package.
#
# 'Flow' of information:
# 1) Before starting the routing procedure, the routing tool writes the environmental data to a netCDF file (in the
# following: 'EnvData netCDF').
# 2) The routing tool (WRT) writes the courses per space-time point to a netCDF file (in the following: 'courses
# netCDF') for which the power consumption will be requested.
#       -> Tanker.write_netCDF_courses
# 3) The WRT sends the paths to the 'EnvData netCDF' and the 'courses netCDF' to mariPower and requests the power
# calculation.
#       -> Tanker.get_fuel_netCDF_loop
# 4) The mariPower package writes the results for the power estimation to the 'courses netCDF'.
# 5) The WRT extracts the power from the 'courses netCDF'.
#       -> Tanker.extract_fuel_from_netCDF
#
# Steps 1), 3), and 5) are combined in the function
#       -> Tanker.get_fuel_per_time_netCDF
#
#
# Functions that are named something like *simple_fuel* are meant to be used as placeholders for the mariPower
# package. They should only be used for
# testing purposes.

class Tanker(Boat):
    # Connection to hydrodynamic modeling
    # hydro_model: mariPower.ship
    draught: float

    # additional information
    courses_path: str  # path to netCDF which contains the power estimation per course
    depth_path: str  # path to netCDF for depth data
    # FIXME: make separate weather path obsolete
    weather_path_maripower: str  # path to weather data which is converted to maripower requirements

    use_depth_data: bool

    def __init__(self, init_mode = 'from_file', file_name=None, config_dict = None):
        super().__init__(init_mode, file_name, config_dict)
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig(file_name = file_name)
        else:
            config_obj = ShipConfig(init_mode='from_dict', config_dict=config_dict)

        # mandatory variables for maripower
        if not config_obj.COURSES_FILE: raise Exception(
            'COURSES_FILE is a mandatory parameter for the maripower tanker!')

        self.courses_path = config_obj.COURSES_FILE
        self.weather_path = config_obj.WEATHER_DATA

        # optional variables for maripower
        if not config_obj.DEPTH_DATA == " ":
            self.use_depth_data = True
            self.depth_path = config_obj.DEPTH_DATA

        self.hydro_model = mariPower.ship.CBT()
        self.hydro_model.Draught_AP = np.array([config_obj.BOAT_DRAUGHT_AFT])
        self.hydro_model.Draught_FP = np.array([config_obj.BOAT_DRAUGHT_FORE])
        self.hydro_model.Roughness_Level = np.array([config_obj.BOAT_ROUGHNESS_LEVEL])
        self.hydro_model.Roughness_Distribution_Level = np.array([config_obj.BOAT_ROUGHNESS_DISTRIBUTION_LEVEL])
        self.hydro_model.WindForcesFactor = config_obj.BOAT_FACTOR_WIND_FORCES
        self.hydro_model.WaveForcesFactor = config_obj.BOAT_FACTOR_WAVE_FORCES
        self.hydro_model.CalmWaterFactor = config_obj.BOAT_FACTOR_CALM_WATER

        # Fine-tuning the following parameters might lead to a significant speedup of mariPower. However, they should
        # be set carefully because the accuracy of the predictions might also be affected
        # self.hydro_model.MaxIterations = 25  # mariPower default: 10
        # self.hydro_model.Tolerance = 0.000001  # mariPower default: 0.0
        # self.hydro_model.Relaxation = 0.7  # mariPower default: 0.3


    def load_data(self):
        self.use_depth_data = False
        if self.use_depth_data:
            self.depth_data = mariPower.environment.EnvironmentalData_Depth(self.depth_path)

        self.weather_adapter()


    # FIXME: make weather adapter obsolete
    def weather_adapter(self):
        debug = False

        weather_str = self.weather_path.split('.nc')
        self.weather_path_maripower = weather_str[0] + '_maripower.nc'

        if os.path.isfile(self.weather_path_maripower):
            return

        if debug:
            print('maripower weather adapter: reading weather data from ', self.weather_path)
            print('writing converted data to ', self.weather_path_maripower)

        ds = xr.open_dataset(self.weather_path)

        thetao = ds['thetao']
        so = ds['so']
        ucurrent = ds['utotal']
        vcurrent = ds['vtotal']

        ds_cut = ds.drop_vars(['thetao', 'so', 'utotal', 'vtotal'])
        ds_cut = ds_cut.drop_dims('depth')

        thetao = thetao.isel(depth=0)
        so = so.isel(depth=0)
        ucurrent = ucurrent.isel(depth=0)
        vcurrent = vcurrent.isel(depth=0)

        ds_cut = xr.merge([ds_cut, thetao])
        ds_cut = xr.merge([ds_cut, so])
        ds_cut = xr.merge([ds_cut, ucurrent])
        ds_cut = xr.merge([ds_cut, vcurrent])

        if debug:
            print('new data: ', ds_cut)

        ds_cut.to_netcdf(self.weather_path_maripower)
        ds_cut.close()
        ds.close()

    def set_ship_property(self, variable, value):
        print('Setting ship property ' + variable + ' to ' + str(value))
        setattr(self.hydro_model, variable, value)

    def print_init(self):
        logger.info(form.get_log_step('boat speed' + str(self.speed), 1))
        logger.info(form.get_log_step('path to weather data' + self.weather_path, 1))
        logger.info(form.get_log_step('path to CoursesRoute.nc' + self.courses_path, 1))
        logger.info(form.get_log_step('path to depth data' + self.depth_path, 1))

    def init_hydro_model_single_pars(self):
        self.hydro_model = mariPower.ship.CBT()
        # shipSpeed = 13 * 1852 / 3600
        self.hydro_model.WindDirection = math.radians(90)
        self.hydro_model.WindSpeed = 0
        self.hydro_model.TemperatureWater = 10

        self.hydro_model.WaveSignificantHeight = 2
        self.hydro_model.WavePeakPeriod = 10.0
        self.hydro_model.WaveDirection = math.radians(45)

        self.hydro_model.CurrentDirection = math.radians(0)
        self.hydro_model.CurrentSpeed = 0.5

        self.MaxIterations = 5

        logger.info('Setting environmental parameters of tanker:')
        logger.info('     water temp', self.hydro_model.TemperatureWater)
        logger.info('     wave significant height', self.hydro_model.WaveSignificantHeight)
        logger.info('     wave peak period', self.hydro_model.WavePeakPeriod)
        logger.info('     wave dir', self.hydro_model.WaveDirection)
        logger.info('     current dir', self.hydro_model.CurrentDirection)
        logger.info('     current speed', self.hydro_model.CurrentSpeed)

    #  initialise mariPower.ship for communication of courses via arrays and passing of environmental data as netCDF
    # def init_hydro_model_NetCDF(self, netCDF_filepath):
    #    self.hydro_model = mariPower.ship.CBT()
    #    self.environment_path = netCDF_filepath
    #    Fx, driftAngle, ptemp, n, delta = mariPower.__main__.PredictPowerForNetCDF(self.hydro_model, netCDF_filepath)

    def set_env_data_path(self, path):
        self.weather_path_maripower = path

    def set_courses_path(self, path):
        self.courses_path = path

    ##
    # function that implements a dummy model for the estimation of the fuel consumption. Only to be used for code
    # testing, for which it minimises excecution time. Does not provide fully accurate estimations. Take care to
    # initialise the simple model using
    # calibrate_simple_fuel()
    def get_fuel_per_course_simple(self, course, wind_speed, wind_dir):
        debug = False
        angle = np.abs(course - wind_dir)
        if angle > 180:
            angle = np.abs(360 - angle)
        if debug:
            form.print_line()
            form.print_step('course = ' + str(course), 1)
            form.print_step('wind_speed = ' + str(wind_speed), 1)
            form.print_step('wind_dir = ' + str(wind_dir), 1)
            form.print_step('delta angle = ' + str(angle), 1)
        wind_speed = wind_speed
        power = self.simple_fuel_model.interp(delta_angle=angle, wind_speed=wind_speed)['power'].to_numpy()

        if debug:
            form.print_step('power = ' + str(power), 1)
        return power

    # def get_fuel_per_time_simple(self, delta_time):
    #    f = 0.0007 * self.rpm ** 3 + 0.0297 * self.rpm ** 2 + 2.8414 * self.rpm - 19.359  # fuel [kg/h]
    #    f *= delta_time / 3600 * 1 / 1000  # amount of fuel for this time interval
    #    return f

    ##
    # initiate estimation of power consumption in mariPower for one particular course and
    # wind direction and speed as well as boat speed
    def get_fuel_per_course(self, course, wind_dir, wind_speed, boat_speed):
        # boat_speed = np.array([boat_speed])
        self.hydro_model.WindDirection = math.radians(wind_dir)
        self.hydro_model.WindSpeed = wind_speed
        form.print_step('course [degrees]= ' + str(course), 1)
        course = units.degree_to_pmpi(course)
        form.print_step('course [rad]= ' + str(course), 1)
        form.print_step('wind dir = ' + str(self.hydro_model.WindDirection), 1)
        form.print_step('wind speed = ' + str(self.hydro_model.WindSpeed), 1)
        form.print_step('boat_speed = ' + str(boat_speed), 1)
        # Fx, driftAngle, ptemp, n, delta = self.hydro_model.IterateMotionSerial(course, boat_speed, aUseHeading=True,
        #                                                                 aUpdateCalmwaterResistanceEveryIteration=False)
        Fx, driftAngle, ptemp, n, delta = self.hydro_model.IterateMotion(course, boat_speed, aUseHeading=True,
                                                                         aUpdateCalmwaterResistanceEveryIteration=False)

        return ptemp

    ##
    # initialisation of simple fuel model that is used as dummy for accurate power estimation via mariPower
    def calibrate_simple_fuel(self):
        self.simple_fuel_model = xr.open_dataset(
            "/home/kdemmich/MariData/Code/MariGeoRoute/Isochrone/Data/SimpleFuelModel/simple_fuel_model.nc")
        form.print_line()
        logger.info('Initialising simple fuel model')
        logger.info(self.simple_fuel_model)

    ##
    # function to write a simple fuel model to file which can be used as dummy for the power estimation with
    # mariPower. The
    # model only considers wind speed and angle as well as the boat speed. 'n_angle' times 'n_wind_speed' pairs of
    # wind speed and wind angle
    # are generated and send to mariPower. The calculated power consumption and wind data are written to file and can
    # in the following be used as input for Tanker.calibrate_simple_fuel.
    def write_simple_fuel(self):
        n_angle = 10
        n_wind_speed = 20
        power = np.zeros((n_angle, n_wind_speed))
        delta_angle = units.get_bin_centers(0, 180, n_angle)
        wind_speed = units.get_bin_centers(0, 60, n_wind_speed)

        coords = dict(delta_angle=(["delta_angle"], delta_angle), wind_speed=(["wind_speed"], wind_speed), )
        attrs = dict(description="Necessary descriptions added here.")

        for iang in range(0, n_angle):
            for iwind_speed in range(0, n_wind_speed):
                course = 0
                wind_dir = 0 + delta_angle[iang]
                power[iang, iwind_speed] = self.get_fuel_per_course(course, wind_dir, wind_speed[iwind_speed],
                                                                    self.speed)

        data_vars = dict(power=(["delta_angle", "wind_speed"], power), )

        ds = xr.Dataset(data_vars, coords, attrs)
        ds.to_netcdf('/home/kdemmich/MariData/Code/simple_fuel_model.nc')

        logger.info('Writing simple fuel model:')
        logger.info(ds)

    ##
    # Initialise power estimation for a tuple of courses in dependence on wind speed and direction. The information
    # is send to mariPower per course.
    def get_fuel_per_time(self, courses, wind):
        debug = False

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('Requesting power calculation')
            course_str = 'Courses:' + str(courses)
            form.print_step(course_str, 1)

        P = np.zeros(courses.shape)
        for icours in range(0, courses.shape[0]):
            # P[icours] = self.get_fuel_per_course(courses[icours], wind['twa'][icours], wind['tws'][icours],
            # self.speed)
            P[icours] = self.get_fuel_per_course_simple(courses[icours], wind['tws'][icours], wind['twa'][icours])
            if math.isnan(P[icours]):
                P[icours] = 1000000000000000

        if (debug):
            form.print_step('power consumption' + str(P))
        return P

    ##
    # Writes netCDF which stores courses in dependence on latitude, longitude and time for further processing by
    # mariPower.
    # Several courses can be provided per space point. In this case, the arrays lats and lons need to be filled
    # e.g. power estimation is requested for 3 courses (c1, c2, c3) for 1 space-time point (lat1, lon1) then:
    #   courses = {c1, c2, c3}
    #   lats = {lat1, lat1, lat1}
    #   lons = {lon1, lon1, lon1}

    def write_netCDF_courses(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False

        if speed is None:
            speed = np.repeat(self.speed, courses.shape, axis=0)

        courses = units.degree_to_pmpi(courses)

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('Requesting power calculation')
            time_str = 'Time:' + str(time.shape)
            lats_str = 'Latitude:' + str(lats.shape)
            lons_str = 'Longitude:' + str(lons.shape)
            course_str = 'Courses:' + str(courses.shape)
            speed_str = 'Boat speed:' + str(speed.shape)
            form.print_step(time_str, 1)
            form.print_step(lats_str, 1)
            form.print_step(lons_str, 1)
            form.print_step(course_str, 1)
            form.print_step(speed_str, 1)
        n_coords = None
        if unique_coords:
            it = sorted(np.unique(lons, return_index=True)[1])
            lons = lons[it]
            lats = lats[it]
            n_coords = lons.shape[0]
        else:
            n_coords = lons.shape[0]
        # number or coordinate pairs
        n_courses = int(courses.shape[0] / n_coords)  # number of courses per coordinate pair

        # generate iterator for courses and coordinate pairs
        it_course = np.arange(n_courses) + 1  # get iterator with a length of the number of unique longitude values
        it_course = np.hstack(
            (it_course,) * n_coords)  # prepare iterator for Data Frame: repeat each element as often as
        it_pos = np.arange(n_coords) + 1
        it_pos = np.repeat(it_pos, n_courses)  # np.hstack((it_pos,) * n_courses)

        if (debug):
            form.print_step('it_course=' + str(it_course))
            form.print_step('it_pos=' + str(it_pos))

        assert courses.shape == it_pos.shape
        assert courses.shape == it_course.shape
        assert courses.shape == speed.shape

        # generate pandas DataFrame
        df = pd.DataFrame({'it_pos': it_pos, 'it_course': it_course, 'courses': courses, 'speed': speed, })

        df = df.set_index(['it_pos', 'it_course'])
        # ToDo: use logger.debug and args.debug
        if (debug):
            print('pandas DataFrame:', df)

        ds = df.to_xarray()

        time_reshape = time.reshape(ds['it_pos'].shape[0], ds['it_course'].shape[0])[:, 0]

        logger.info('Request power calculation for ' + str(n_courses) + ' courses and ' + str(n_coords) +
                    ' coordinates')

        ds["lon"] = (['it_pos'], lons)
        ds["lat"] = (['it_pos'], lats)
        ds["time"] = (['it_pos'], time_reshape)
        assert ds['lon'].shape == ds['lat'].shape
        assert ds['time'].shape == ds['lat'].shape

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('xarray DataSet', ds)

        ds.to_netcdf(self.courses_path + str())
        if (debug):
            ds_read = xr.open_dataset(self.courses_path)
            print('read data set', ds_read)
        ds.close()

    ##
    # extracts power from 'courses netCDF' which has been written by mariPower and returns it as 1D array.
    def extract_params_from_netCDF(self, ds):
        debug = False
        if (debug):
            form.print_step('Dataset with ship parameters:' + str(ds), 1)

        power = ds['Power_brake'].to_numpy().flatten() * u.Watt
        rpm = ds['RotationRate'].to_numpy().flatten() * 1 / u.minute
        fuel = ds['Fuel_consumption_rate'].to_numpy().flatten() * u.tonne / u.hour
        fuel = fuel.to(u.kg / u.second)
        r_wind = ds['Wind_resistance'].to_numpy().flatten() * u.newton
        r_calm = ds['Calm_resistance'].to_numpy().flatten() * u.newton
        r_waves = ds['Wave_resistance'].to_numpy().flatten() * u.newton
        r_shallow = ds['Shallow_water_resistance'].to_numpy().flatten() * u.newton
        r_roughness = ds['Hull_roughness_resistance'].to_numpy().flatten() * u.newton
        status = ds['Status'].to_numpy().flatten()
        message = ds['Message'].to_numpy().flatten()
        speed = np.repeat(self.speed, power.shape)

        ship_params = ShipParams(
            fuel_rate=fuel,
            power=power,
            rpm=rpm,
            speed=speed,
            r_wind=r_wind,
            r_calm=r_calm,
            r_waves=r_waves,
            r_shallow=r_shallow,
            r_roughness=r_roughness,
            wave_height=-99,
            wave_direction=-99,
            wave_period=-99,
            u_currents=-99,
            v_currents=-99,
            u_wind_speed=-99,
            v_wind_speed=-99,
            pressure=-99,
            air_temperature=-99,
            salinity=-99,
            water_temperature=-99,
            status=status,
            message=message
        )

        if (debug):
            form.print_step('Dataset with fuel' + str(ds), 1)
            form.print_step('original shape power' + str(power.shape), 1)
            form.print_step('flattened shape power' + str(ship_params.get_power().shape), 1)
            form.print_step('power result' + str(ship_params.get_power()))

        return ship_params

    ##
    # dummy function uses to mimic writing of power estimation to 'courses netCDF'. Only used for testing purposes.
    #
    def get_fuel_netCDF_dummy(self, ds, courses, wind):
        debug = False

        power = self.get_fuel_per_time(courses, wind)
        if (debug):
            form.print_step('power shape' + str(power.shape), 1)
        power = power.reshape(ds['lat'].shape[0], ds['it'].shape[0])
        ds["power"] = (['lat', 'it'], power)
        if (debug):
            form.print_step('power new shape' + str(power.shape), 1)
            form.print_step('ds' + str(ds), 1)

        ds.to_netcdf(self.courses_path)
        ds_read = xr.open_dataset(self.courses_path)
        # ToDo: use logger.debug and args.debug
        if (debug):
            print('read data set', ds_read)

    ##
    # Passes paths for 'courses netCDF' and 'environmental data netCDF' to mariPower and request estimation of power
    # consumption.
    # Is not yet working as explained for Tanker.get_fuel_netCDF_loop
    #
    def get_fuel_netCDF(self):
        mariPower_ship = copy.deepcopy(self.hydro_model)
        if self.use_depth_data:
            status, message, envDataRoute = mariPower.__main__.PredictPowerOrSpeedRoute(
                mariPower_ship, self.courses_path, self.weather_path_maripower, self.depth_data)
        else:
            status, message, envDataRoute = mariPower.__main__.PredictPowerOrSpeedRoute(
                mariPower_ship, self.courses_path, self.weather_path_maripower)
        # form.print_current_time('time for mariPower request:', start_time)
        # ToDo: read messages from netCDF and store them in ship_params (changes in mariPower necessary)
        # for idx in range(0, len(status.flatten())):
        #     if status.flatten()[idx] != 1:
        #         logger.warning(f"{idx}: status.shape={status.shape}, status={status.flatten()[idx]}, "
        #                        f"message={message.flatten()[idx]}")

        ds_read = xr.open_dataset(self.courses_path)
        return ds_read

    ##
    # @brief splits data in 'courses netCDF' one bunches per course per space point, sends them to mariPower
    # separately and merges them again afterwards.
    #
    # mariPower can currently handle only requests with 1 course per space point. Thus, the data in the 'course
    # netCDF' is split in
    # several bunchs each one containing an xarray with only one course per space point. The bunches are send to
    # mariPower separately
    # and the returned data sets are merged into one. Will (hopefully) be redundant as soon as mariPower accepts
    # requests with several
    # courses per space-time point and will then be replaced by Tanker.get_fuel_netCDF()
    def get_fuel_netCDF_loop(self):
        debug = False
        filename_single = '/home/kdemmich/MariData/Code/MariGeoRoute/Isochrone/CoursesRouteSingle.nc'
        # filename_single = 'C:/Users/Maneesha/Documents/GitHub/MariGeoRoute/WeatherRoutingTool/CoursesRouteSingle.nc'
        ds = xr.load_dataset(self.courses_path)
        n_vars = ds['it'].shape[0]
        ds_merged = xr.Dataset()

        if (debug):
            form.print_line()
            form.print_step('get_fuel_netCDF_loop: loop over all variants per space point', 0)
            form.print_step('original dataset: ' + str(ds), 0)

        for ivar in range(1, n_vars + 1):
            ds_read_temp = ds.isel(it=[ivar - 1])
            ds_read_temp.coords['it'] = [1]
            ds_read_temp.to_netcdf(filename_single, mode='w')
            ds_read_temp.close()
            ship = mariPower.ship.CBT()
            if (debug):
                ds_read_test = xr.load_dataset(filename_single)
                courses_test = ds_read_test['courses']
                form.print_step('courses_test' + str(courses_test.to_numpy()), 1)
                form.print_step('speed' + str(ds_read_test['speed'].to_numpy()), 1)
            # start_time = time.time()
            mariPower.__main__.PredictPowerOrSpeedRoute(ship, filename_single, self.weather_path_maripower, None, False,
                                                        False)
            # form.print_current_time('time for mariPower request:', start_time)

            ds_temp = xr.load_dataset(filename_single)
            ds_temp.coords['it'] = [ivar]
            if ivar == 1:
                ds_merged = ds_temp.copy()
            else:
                ds_merged = xr.concat([ds_merged, ds_temp], dim="it")
            if (debug):
                form.print_step('step ' + str(ivar) + ': merged dataset:' + str(ds_merged), 1)
        ds_merged['lon'] = ds_merged['lon'].sel(it=1).drop('it')
        ds_merged['time'] = ds_merged['time'].sel(it=1).drop('it')

        if (debug):
            form.print_step('final merged dataset:' + str(ds_merged))
        ds.close()
        return ds_merged

    ##
    # main function for communication with mariPower package (see documentation above)
    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        self.write_netCDF_courses(courses, lats, lons, time, speed, unique_coords)

        # ds = self.get_fuel_netCDF_loop()
        # ds = self.get_fuel_netCDF_dummy(ds, courses, wind)
        ds = self.get_fuel_netCDF()
        ship_params = self.extract_params_from_netCDF(ds)
        ship_params = self.evaluate_weather(ship_params, lats, lons, time)
        ds.close()

        return ship_params

    ##
    # Function to test/plot power consumption in dependence of wind speed and direction. Works only with old versions
    # of mariPower package.
    # Has partly been replaced by test_polars: test_power_consumption_returned()
    def test_power_consumption_per_course(self):
        courses = np.linspace(0, 360, num=21, endpoint=True)
        wind_dir = 45
        wind_speed = 2
        power = np.zeros(courses.shape)

        # get_fuel_per_course gets angles in degrees from 0 to 360
        for i in range(0, courses.shape[0]):
            power[i] = self.get_fuel_per_course(courses[i], wind_dir, wind_speed,
                                                self.speed)  # power[i] = self.get_fuel_per_time_simple(i*3600)

        # plotting with matplotlib needs angles in radiants
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
        for i in range(0, courses.shape[0]):
            courses[i] = math.radians(courses[i])
        wind_dir = math.radians(wind_dir)

        axes[0].plot(courses, power)
        axes[0].legend()
        for ax in axes.flatten():
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.set_theta_zero_location("S")
            ax.grid(True)
        axes[1].plot([wind_dir, wind_dir], [0, wind_speed], color='blue', label='Wind')
        # axes[1].plot([self.hydro_model.WaveDirection, self.hydro_model.WaveDirection], [0,
        # 1 * (self.hydro_model.WaveSignificantHeight > 0.0)],
        #                color='green', label='Seaway')
        # axes[1].plot([self.hydro_model.CurrentDirection, self.hydro_model.CurrentDirection], [0,
        # 1 * (self.hydro_model.CurrentSpeed > 0.0)],
        #                color='purple', label='Current')
        axes[1].legend()

        axes[0].set_title("Power", va='bottom')
        axes[1].set_title("Environmental conditions", va='top')
        plt.show()

    ##
    # Function to test/plot power consumption in dependence of wind speed and direction. Works only with old versions
    # of mariPower package.
    # Has partly been replaced by test_polars: test_power_consumption_returned()
    def test_power_consumption_per_speed(self):
        course = 10
        boat_speed = np.linspace(1, 20, num=17)
        wind_dir = 45
        wind_speed = 2
        power = np.zeros(boat_speed.shape)

        for i in range(0, boat_speed.shape[0]):
            power[i] = self.get_fuel_per_course(course, wind_dir, wind_speed,
                                                boat_speed[i])  # power[i] = self.get_fuel_per_time_simple(i*3600)

        plt.plot(boat_speed, power, 'r--')
        plt.xlabel('speed (m/s)')
        plt.ylabel('power (W)')
        plt.show()
