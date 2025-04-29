from datetime import datetime, timedelta
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.unit_conversion as utils
import WeatherRoutingTool.utils.graphics as graphics

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.shipparams import ShipParams

def compare_times(time64, time):
    time64 = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    time = (time - datetime(1970, 1, 1, 0, 0))
    for iTime in range(0, time.shape[0]):
        time[iTime] = time[iTime].total_seconds()
    assert np.array_equal(time64, time)


class TestShip:

    def create_dummy_Direct_Power_Ship(self, ship_config_path):
        dirname = os.path.dirname(__file__)
        configpath = os.path.join(dirname, 'config.tests.json')
        config = Config(file_name=configpath)

        dirname = os.path.dirname(__file__)
        config.WEATHER_DATA = os.path.join(dirname, 'data/reduced_testdata_weather.nc')
        config.COURSES_FILE = os.path.join(dirname, 'data/CoursesRoute.nc')
        config.DEPTH_DATA = os.path.join(dirname, 'data/reduced_testdata_depth.nc')
        config.BOAT_CONFIG = os.path.join(dirname, ship_config_path + '.json')

        pol = DirectPowerBoat(config)
        return pol



    '''
        test whether single elements of fuel, power, rpm and speed are correctly returned by ShipParams.get_element(idx)
    '''
    def test_shipparams_get_element(self):
        fuel = np.array([1, 2, 3, 4])
        speed = np.array([0.1, 0.2, 0.3, 0.4])
        power = np.array([11, 21, 31, 41])
        rpm = np.array([21, 22, 23, 24])
        rwind = np.array([1.1, 1.2, 1.3, 1.4])
        rcalm = np.array([2.1, 2.2, 2.3, 2.4])
        rwaves = np.array([3.1, 3.2, 3.3, 3.4])
        rshallow = np.array([4.1, 4.2, 4.3, 4.4])
        rroughness = np.array([5.1, 5.2, 5.3, 5.4])
        wave_height = np.array([4.1, 4.2, 4.11, 4.12])
        wave_direction = np.array([4.4, 4.5, 4.41, 4.42])
        wave_period = np.array([4.7, 4.8, 4.71, 4.72])
        u_currents = np.array([.1, 5.2, 5.11, 5.12])
        v_currents = np.array([5.4, 5.5, 5.41, 5.42])
        u_wind_speed = np.array([7.1, 7.2, 7.11, 7.12])
        v_wind_speed = np.array([7.4, 7.5, 7.41, 7.42])
        pressure = np.array([5.7, 5.8, 5.71, 5.72])
        air_temperature = np.array([6.1, 6.2, 6.11, 6.12])
        salinity = np.array([6.4, 6.5, 6.41, 6.42])
        water_temperature = np.array([6.7, 6.8, 6.71, 6.72])
        status = np.array([1, 2, 2, 3])
        message = np.array(['OK', 'OK', 'Error' 'OK'])

        sp = ShipParams(fuel_rate=fuel, power=power, rpm=rpm, speed=speed, r_wind=rwind, r_calm=rcalm, r_waves=rwaves,
                    r_shallow=rshallow, r_roughness=rroughness, wave_height=wave_height,
                    wave_direction=wave_direction, wave_period=wave_period, u_currents=u_currents,
                    v_currents=v_currents, u_wind_speed=u_wind_speed, v_wind_speed=v_wind_speed, pressure=pressure,
                    air_temperature=air_temperature, salinity=salinity, water_temperature=water_temperature,
                    status=status, message=message)
        idx = 2

        fuel_test, power_test, rpm_test, speed_test, rwind_test, rcalm_test, rwaves_test, \
        rshallow_test, rroughness_test, wave_height_test, wave_direction_test, wave_period_test, \
        u_currents_test, v_currents_test, u_wind_speed_test, v_wind_speed_test, pressure_test, air_temperature_test, \
        salinity_test, water_temperature_test, status_test, message_test = sp.get_element(idx)

        assert fuel[idx] == fuel_test
        assert speed[idx] == speed_test
        assert power[idx] == power_test
        assert rpm[idx] == rpm_test
        assert rwind[idx] == rwind_test
        assert rcalm[idx] == rcalm_test
        assert rwaves[idx] == rwaves_test
        assert rshallow[idx] == rshallow_test
        assert rroughness[idx] == rroughness_test
        assert wave_height[idx] == wave_height_test
        assert wave_direction[idx] == wave_direction_test
        assert wave_period[idx] == wave_period_test
        assert u_currents[idx] == u_currents_test
        assert v_currents[idx] == v_currents_test
        assert u_wind_speed[idx] == u_wind_speed_test
        assert v_wind_speed[idx] == v_wind_speed_test
        assert pressure[idx] == pressure_test
        assert air_temperature[idx] == air_temperature_test
        assert salinity[idx] == salinity_test
        assert water_temperature[idx] == water_temperature_test
        assert status[idx] == status_test
        assert message[idx] == message_test


    '''
        test whether ShipParams object for single waypoint is correctly returned by ShipParams.get_single_object(idx)
    '''


    def test_shipparams_get_single(self):
        fuel = np.array([1, 2, 3, 4])
        speed = np.array([0.1, 0.2, 0.3, 0.4])
        power = np.array([11, 21, 31, 41])
        rpm = np.array([21, 22, 23, 24])
        rwind = np.array([1.1, 1.2, 1.3, 1.4])
        rcalm = np.array([2.1, 2.2, 2.3, 2.4])
        rwaves = np.array([3.1, 3.2, 3.3, 3.4])
        rshallow = np.array([4.1, 4.2, 4.3, 4.4])
        rroughness = np.array([5.1, 5.2, 5.3, 5.4])
        rroughness = np.array([5.1, 5.2, 5.3, 5.4])
        wave_height = np.array([4.1, 4.2, 4.11, 4.12])
        wave_direction = np.array([4.4, 4.5, 4.41, 4.42])
        wave_period = np.array([4.7, 4.8, 4.71, 4.72])
        u_currents = np.array([.1, 5.2, 5.11, 5.12])
        v_currents = np.array([5.4, 5.5, 5.41, 5.42])
        u_wind_speed = np.array([7.1, 7.2, 7.11, 7.12])
        v_wind_speed = np.array([7.4, 7.5, 7.41, 7.42])
        pressure = np.array([5.7, 5.8, 5.71, 5.72])
        air_temperature = np.array([6.1, 6.2, 6.11, 6.12])
        salinity = np.array([6.4, 6.5, 6.41, 6.42])
        water_temperature = np.array([6.7, 6.8, 6.71, 6.72])
        status = np.array([1, 2, 2, 3])
        message = np.array(['OK', 'OK', 'Error' 'OK'])

        sp = ShipParams(fuel_rate=fuel, power=power, rpm=rpm, speed=speed,
                        r_wind=rwind, r_calm=rcalm, r_waves=rwaves,
                        r_shallow=rshallow, r_roughness=rroughness,
                        wave_height=wave_height,
                        wave_direction=wave_direction, wave_period=wave_period,
                        u_currents=u_currents,
                        v_currents=v_currents, u_wind_speed=u_wind_speed,
                        v_wind_speed=v_wind_speed, pressure=pressure,
                        air_temperature=air_temperature, salinity=salinity,
                        water_temperature=water_temperature,
                        status=status, message=message)

        idx = 2

        sp_test = sp.get_single_object(idx)

        assert sp_test.fuel_rate == fuel[idx]
        assert sp_test.power == power[idx]
        assert sp_test.rpm == rpm[idx]
        assert sp_test.speed == speed[idx]
        assert sp_test.r_calm == rcalm[idx]
        assert sp_test.r_wind == rwind[idx]
        assert sp_test.r_waves == rwaves[idx]
        assert sp_test.r_shallow == rshallow[idx]
        assert sp_test.r_roughness == rroughness[idx]
        assert sp_test.wave_height == wave_height[idx]
        assert sp_test.wave_direction == wave_direction[idx]
        assert sp_test.wave_period == wave_period[idx]
        assert sp_test.u_currents == u_currents[idx]
        assert sp_test.v_currents == v_currents[idx]
        assert sp_test.u_wind_speed == u_wind_speed[idx]
        assert sp_test.v_wind_speed == v_wind_speed[idx]
        assert sp_test.pressure == pressure[idx]
        assert sp_test.air_temperature == air_temperature[idx]
        assert sp_test.salinity == salinity[idx]
        assert sp_test.water_temperature == water_temperature[idx]
        assert sp_test.status == status[idx]
        assert sp_test.message == message[idx]

   '''
        DIRECT POWER METHOD: check whether values of weather data are correctly read from file
    '''

    def test_evaluate_weather_for_direct_power_method(self):
         # dummy weather file
        dirname = os.path.dirname(__file__)
        weather_data = xr.open_dataset(os.path.join(dirname, 'data/reduced_testdata_weather.nc'))

        time_single = datetime.strptime('2023-07-20', '%Y-%m-%d')

        # courses test file
        courses_test = np.array([0, 180, 0, 180, 180, 0]) * u.degree
        lat_test = np.array([54.3, 54.3, 54.6, 55.6, 55.9, 55.9])
        lon_test = np.array([13.3, 13.3, 13.6, 16., 16.9, 13.9])
        time_test = np.array([time_single, time_single, time_single, time_single, time_single, time_single])

        # dummy course netCDF
        pol = self.create_dummy_Direct_Power_Ship('shipconfig_simpleship')

        ship_params = pol.get_ship_parameters(courses_test, lat_test, lon_test, time_test)
        ship_params.print()

        for i in range(0,6):
            ship_params.wave_direction[i] = np.nan_to_num(ship_params.wave_direction[i])
            wavedir_data = weather_data['VMDR'].sel(latitude = lat_test[i], longitude = lon_test[i], time = time_test[i],method='nearest', drop=False).fillna(0).to_numpy()

            ship_params.wave_height[i] = np.nan_to_num(ship_params.wave_height[i])
            waveheight_data = weather_data['VHM0'].sel(latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
                                                 method='nearest', drop=False).fillna(0).to_numpy()
            ship_params.wave_period[i] = np.nan_to_num(ship_params.wave_period[i])
            waveperiod_data = weather_data['VTPK'].sel(latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
                                                       method='nearest', drop=False).fillna(0).to_numpy()

            assert abs(ship_params.wave_direction[i].value-wavedir_data) < 0.0001
            assert abs(ship_params.wave_height[i].value-waveheight_data) < 0.0001
            assert abs(ship_params.wave_period[i].value-waveperiod_data) < 0.0001

        speed_test = float(weather_data['v-component_of_wind_height_above_ground'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i], height_above_ground=10,
            method='nearest', drop=False).to_numpy())
        uwind_test = float(weather_data['u-component_of_wind_height_above_ground'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i], height_above_ground=10,
            method='nearest', drop=False).to_numpy())
        utotal_test = float(weather_data['utotal'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
            method='nearest', drop=False).fillna(0).to_numpy()[0])
        vtotal_test = float(weather_data['vtotal'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
            method='nearest', drop=False).fillna(0).to_numpy()[0])
        pressure_test = float(weather_data['Pressure_reduced_to_MSL_msl'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
            method='nearest', drop=False).to_numpy())
        thetao_test = float(weather_data['thetao'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
            method='nearest', drop=False).fillna(0).to_numpy()[0])
        salinity_test = float(weather_data['so'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
            method='nearest', drop=False).fillna(0).to_numpy()[0])
        salinity_test = salinity_test * 0.001
        air_temp_test = float(weather_data['Temperature_surface'].sel(
            latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
            method='nearest', drop=False).to_numpy())
        air_temp_test = air_temp_test - 273.15

        assert abs(ship_params.v_wind_speed[i].value - speed_test) < 0.00001
        assert abs(ship_params.u_wind_speed[i].value - uwind_test) < 0.00001
        assert abs(ship_params.u_currents[i].value - utotal_test) < 0.00001
        assert abs(ship_params.v_currents[i].value - vtotal_test) < 0.00001
        assert abs(ship_params.pressure[i].value - pressure_test) < 0.01
        assert abs(ship_params.water_temperature[i].value - thetao_test) < 0.00001
        assert abs(ship_params.salinity[i].value - salinity_test) < 0.00001
        assert abs(ship_params.air_temperature[i].value - air_temp_test) < 0.0001


    '''
        DIRECT POWER METHOD: check whether class variables (speed, eta_prop, power_at_sp, overload_factor) are set as 
        expected and correct power and corresponding unit are returned
    '''
    @pytest.mark.parametrize("DeltaR,speed,design_power", [(5000,6,6502000 * 0.75)])
    def test_get_power_for_direct_power_method(self, DeltaR, speed, design_power):
        pol = self.create_dummy_Direct_Power_Ship('shipconfig_simpleship')
        P = DeltaR * speed/0.63 + design_power

        Ptest = pol.get_power(5000 * u.N)
        assert P == Ptest.value
        assert 'W' == Ptest.unit

    '''
        DIRECT POWER METHOD: check whether relative angle between wind direction and course of the ship is correctly 
        calculated from u_wind and v_wind
    '''
    def test_get_wind_dir(self):
        wind_dir = np.array([30,120, 210, 300]) * u.degree
        absv = 20
        courses = np.array([10,10,20,20]) * u.degree
        rel_wind_dir = np.array([20, 110, 170, 70]) * u.degree

        pol = self.create_dummy_Direct_Power_Ship('shipconfig_simpleship')

        v_wind = -absv * np.cos(np.radians(wind_dir)) * u.meter/u.second
        u_wind = -absv * np.sin(np.radians(wind_dir)) * u.meter/u.second

        true_wind_dir = pol.get_wind_dir(u_wind, v_wind)
        true_wind_dir = pol.get_relative_wind_dir(courses, true_wind_dir)

        assert np.all((rel_wind_dir -true_wind_dir)<0.0001 * u.degree)

    '''
        DIRECT POWER METHOD: check whether apparent wind speed and direction are correctly calculated for single values of
        wind speed and wind dir
    '''
    def test_get_apparent_wind(self):
        wind_dir = np.array([0, 45, 90, 135, 180]) * u.degree
        wind_speed = np.array([10, 10, 10, 10, 10]) * u.meter/u.second
        wind_speed_test = np.array([16, 14.86112, 11.66190, 7.15173, 4]) * u.meter/u.second
        wind_dir_test = np.array([0, 28.41, 59.04, 98.606, 180]) * u.degree

        pol = self.create_dummy_Direct_Power_Ship('shipconfig_simpleship')
        wind_result = pol.get_apparent_wind(wind_speed, wind_dir)

        for i in range(0, 4):
            assert abs(wind_result['app_wind_speed'][i] - wind_speed_test[i]) < 0.01 * u.meter/u.second
            assert abs(wind_result['app_wind_angle'][i] - wind_dir_test[i]) < 0.01 * u.degree


    '''
        DIRECT POWER METHOD: check whether apparent wind speed and direction look fine on polar plot
    '''
    @pytest.mark.manual
    def test_get_apparent_wind_polar_plot(self):
        wind_dir = np.linspace(0, 180, 19) * u.degree
        wind_speed = np.full(19,10) * u.meter/u.second

        pol = self.create_dummy_Direct_Power_Ship('shipconfig_simpleship')
        wind_result = pol.get_apparent_wind(wind_speed, wind_dir)

        fig, axes = plt.subplots(subplot_kw={'projection': 'polar'})
        axes.plot(np.radians(wind_dir), wind_speed, label="true wind")
        axes.plot(np.radians(wind_result['app_wind_angle']), wind_result['app_wind_speed'], label="apparent wind")
        axes.plot(np.radians(wind_result['app_wind_angle']), wind_speed, label="apparent wind dir, fixed speed")
        axes.legend(loc="upper right")
        axes.set_title("Wind direction", va='bottom')
        plt.show()


    '''
        DIRECT POWER METHOD: check whether ship geometry is approximated correctly if only mandatory parameters
        are supplied
    '''
    def test_calculate_geometry_simple_method(self):
        pol = self.create_dummy_Direct_Power_Ship('shipconfig_simpleship')
        hbr = 30 * u.meter
        breadth = 32 * u.meter
        length = 180 * u.meter
        ls1 = 0.2 * length
        ls2 = 0.3 * length
        hs1 = 0.2 * hbr
        hs2 = 0.1 * hbr
        bs1 = 0.9 * breadth
        cmc = -0.035 * length
        hc = 10 * u.meter
        Axv = 940.8 * u.meter * u.meter
        Ayv = 4248 * u.meter * u.meter
        Aod = 378 * u.meter * u.meter

        assert pol.hbr == hbr
        assert pol.breadth == breadth
        assert pol.length == length
        assert pol.ls1 == ls1
        assert pol.ls2 == ls2
        assert pol.bs1 == bs1
        assert pol.hs1 == hs1
        assert pol.hs2 == hs2
        assert pol.cmc == cmc
        assert pol.hc == hc
        assert pol.Axv == Axv
        assert pol.Ayv == Ayv
        assert pol.Aod == Aod

    '''
        DIRECT POWER METHOD: check whether ship geometry parameters are set correctly if manual values are supplied
    '''
    def test_calculate_geometry_manual_method(self):
        pol = self.create_dummy_Direct_Power_Ship('shipconfig_manualship')
        hbr = 30 * u.meter
        breadth = 32 * u.meter
        length = 180 * u.meter
        ls1 = 0.2 * length
        ls2 = 0.3 * length
        hs1 = 0.2 * hbr
        hs2 = 0.1 * hbr
        bs1 = 0.9 * breadth
        cmc = 8.1 * u.meter
        hc = 7.06 * u.meter
        Axv = 716 * u.meter * u.meter
        Ayv = 1910 * u.meter * u.meter
        Aod = 529 * u.meter * u.meter

        assert pol.hbr == hbr
        assert pol.breadth == breadth
        assert pol.length == length
        assert pol.ls1 == ls1
        assert pol.ls2 == ls2
        assert pol.bs1 == bs1
        assert pol.hs1 == hs1
        assert pol.hs2 == hs2
        assert pol.cmc == cmc
        assert pol.hc == hc
        assert pol.Axv == Axv
        assert pol.Ayv == Ayv
        assert pol.Aod == Aod

    '''
        DIRECT POWER METHOD: check for reasonable behaviour of wind coefficient C_AA
    '''
    @pytest.mark.manual
    def test_wind_coeff(self):
        u_wind_speed = 0 * u.meter/u.second
        v_wind_speed = -10 * u.meter/u.second

        courses = np.linspace(0, 180, 19) * u.degree

        pol = self.create_dummy_Direct_Power_Ship('shipconfig_manualship')
        r_wind = pol.get_wind_resistance(u_wind_speed, v_wind_speed, courses)

        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
        ax.plot(courses, r_wind["wind_coeff"], color=graphics.get_colour(0), label='CAA')
        ax.set_xlabel('angle of attack (degrees)')
        ax.set_ylabel(r'$C_{AA}$')
        plt.show()

    '''
        DIRECT POWER METHOD: check for reasonable behaviour of wind resistance on polar plot
    '''
    @pytest.mark.manual
    def test_wind_resistance(self):
        u_wind_speed = 0 * u.meter/u.second
        v_wind_speed = -10  * u.meter/u.second

        courses = np.linspace(0, 180, 19)
        courses_rad = np.radians(courses)
        courses = courses * u.degree

        pol = self.create_dummy_Direct_Power_Ship('shipconfig_manualship')
        r_wind = pol.get_wind_resistance(u_wind_speed, v_wind_speed, courses)

        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

        axes[0].plot(courses_rad, r_wind['r_wind'])
        axes[0].legend()
        for ax in axes.flatten():
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.set_theta_zero_location("S")
            ax.grid(True)
        axes[1].plot(courses_rad, r_wind['r_wind'])
        axes[0].set_title("Power", va='bottom')
        axes[1].set_title("Wind resistence", va='top')

        plt.show()


    '''
        DIRECT POWER METHOD: compare wind resistance and power from the Direct Power Method to results from maripower.
        
        - relative difference of wind direction and boat course is changing in steps of 10 degrees
        - effect from wave resistance is turned of for maripower; all other resistances are considerd by maripower
    '''
    @pytest.mark.manual
    def test_compare_wind_resistance_to_maripower(self):
        lats = np.full(10, 54.9)  # 37
        lons = np.full(10, 13.2)
        courses = np.linspace(0, 360, 10) * u.degree
        courses_rad = utils.degree_to_pmpi(courses)

        time = np.full(10, datetime.strptime("2023-07-20T10:00Z", '%Y-%m-%dT%H:%MZ'))
        bs = 7.7 * u.meter/u.second

        pol_maripower = self.create_dummy_Tanker_object()
        pol_maripower.set_ship_property('WaveForcesFactor', 0)

        pol_maripower.use_depth_data = False
        pol_maripower.set_boat_speed(bs)
        ship_params_maripower = pol_maripower.get_ship_parameters(courses, lats, lons, time, None, True)
        rwind_maripower = ship_params_maripower.get_rwind()
        P_maripower = ship_params_maripower.get_power()

        pol_simple = self.create_dummy_Direct_Power_Ship('shipconfig_manualship')
        pol_simple.set_boat_speed(bs)
        ship_params_simple = pol_simple.get_ship_parameters(courses, lats, lons, time)
        r_wind_simple = ship_params_simple.get_rwind()
        P_simple = ship_params_simple.get_power()

        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

        axes[0].plot(courses_rad, rwind_maripower, label="maripower")
        axes[0].plot(courses_rad, r_wind_simple, label="Fujiwara")
        axes[0].legend(loc="upper right")
        for ax in axes.flatten():
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.set_theta_zero_location("S")
            ax.grid(True)
        axes[0].set_title("Wind resistance", va='bottom')

        axes[1].plot(courses_rad, P_maripower, label="maripower")
        axes[1].plot(courses_rad, P_simple, label="DPM")
        axes[1].set_title("Power", va='top')
        axes[0].legend(loc="upper right")
        plt.show()




