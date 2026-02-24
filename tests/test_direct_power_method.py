from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.utils.unit_conversion as utils
import WeatherRoutingTool.utils.graphics as graphics

from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat

have_maripower = False

try:
    import mariPower
    from tests.test_maripower_tanker import TestMariPowerTanker

    have_maripower = True
except ModuleNotFoundError:
    pass  # maripower installation is optional


class TestDPM:
    '''
        DIRECT POWER METHOD: check whether class variables (speed, eta_prop, power_at_sp, overload_factor) are set as
        expected and correct power and corresponding unit are returned
    '''

    @pytest.mark.parametrize("DeltaR,speed,design_power", [(5000, 6, 6502000 * 0.75)])
    def test_get_power_for_direct_power_method(self, DeltaR, speed, design_power):
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        P = DeltaR * speed / 0.63 + design_power

        Ptest = pol.get_power(5000 * u.N)
        assert P == Ptest.value
        assert 'W' == Ptest.unit

    '''
        DIRECT POWER METHOD: check whether relative angle between wind direction and course of the ship is correctly
        calculated from u_wind and v_wind
    '''

    def test_get_wind_dir(self):
        wind_dir = np.array([30, 120, 210, 300]) * u.degree
        absv = 20
        courses = np.array([10, 10, 20, 20]) * u.degree
        rel_wind_dir = np.array([20, 110, 170, 70]) * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')

        v_wind = -absv * np.cos(np.radians(wind_dir)) * u.meter / u.second
        u_wind = -absv * np.sin(np.radians(wind_dir)) * u.meter / u.second

        true_wind_dir = pol.get_wind_dir(u_wind, v_wind)
        true_wind_dir = pol.get_relative_wind_dir(courses, true_wind_dir)

        assert np.all((rel_wind_dir - true_wind_dir) < 0.0001 * u.degree)

    '''
        DIRECT POWER METHOD: check whether apparent wind speed and direction are correctly calculated for single
        values of wind speed and wind dir
    '''

    def test_get_apparent_wind(self):
        wind_dir = np.array([0, 45, 90, 135, 180]) * u.degree
        wind_speed = np.array([10, 10, 10, 10, 10]) * u.meter / u.second
        wind_speed_test = np.array([16, 14.86112, 11.66190, 7.15173, 4]) * u.meter / u.second
        wind_dir_test = np.array([0, 28.41, 59.04, 98.606, 180]) * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        wind_result = pol.get_apparent_wind(wind_speed, wind_dir)

        for i in range(0, 4):
            assert abs(wind_result['app_wind_speed'][i] - wind_speed_test[i]) < 0.01 * u.meter / u.second
            assert abs(wind_result['app_wind_angle'][i] - wind_dir_test[i]) < 0.01 * u.degree

    '''
        DIRECT POWER METHOD: check whether apparent wind speed and direction look fine on polar plot
    '''

    @pytest.mark.manual
    def test_get_apparent_wind_polar_plot(self):
        wind_dir = np.linspace(0, 180, 19) * u.degree
        wind_speed = np.full(19, 10) * u.meter / u.second

        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
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
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        pol.load_data()

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
        pol = basic_test_func.create_dummy_Direct_Power_Ship('manualship')
        pol.load_data()

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
        u_wind_speed = np.full(19, 0) * u.meter / u.second
        v_wind_speed = np.full(19, -10) * u.meter / u.second

        courses = np.linspace(0, 180, 19) * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('manualship')
        r_wind = pol.get_wind_resistance(u_wind_speed, v_wind_speed, courses)

        old_usetex = plt.rcParams['text.usetex']
        try:
            plt.rcParams['text.usetex'] = True
            fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
            ax.plot(courses, r_wind["wind_coeff"], color=graphics.get_colour(0), label='CAA')
            ax.set_xlabel('angle of attack (degrees)')
            ax.set_ylabel(r'$C_{AA}$')
            plt.show()
        finally:
            plt.rcParams['text.usetex'] = old_usetex

    '''
        DIRECT POWER METHOD: check for reasonable behaviour of wind resistance on polar plot
    '''

    @pytest.mark.manual
    def test_wind_resistance(self):
        u_wind_speed = np.full(19, 0) * u.meter / u.second
        v_wind_speed = np.full(19, -10) * u.meter / u.second

        courses = np.linspace(0, 180, 19)
        courses_rad = np.radians(courses)
        courses = courses * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('manualship')
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
    @pytest.mark.skipif(not have_maripower, reason="maripower is not installed")
    @pytest.mark.manual
    def test_compare_wind_resistance_to_maripower(self):
        lats = np.full(10, 54.9)  # 37
        lons = np.full(10, 13.2)
        courses = np.linspace(0, 360, 10) * u.degree
        courses_rad = utils.degree_to_pmpi(courses)

        time = np.full(10, datetime.strptime("2023-07-20T10:00Z", '%Y-%m-%dT%H:%MZ'))
        bs = np.full(10, 7.7) * u.meter / u.second

        pol_maripower = basic_test_func.create_dummy_Tanker_object()
        pol_maripower.set_ship_property('WaveForcesFactor', 0)

        pol_maripower.use_depth_data = False
        ship_params_maripower = pol_maripower.get_ship_parameters(courses, lats, lons, time, bs, True)
        rwind_maripower = ship_params_maripower.get_rwind()
        P_maripower = ship_params_maripower.get_power()

        pol_simple = basic_test_func.create_dummy_Direct_Power_Ship('manualship')
        ship_params_simple = pol_simple.get_ship_parameters(courses, lats, lons, time, bs)
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

    def test_dpm_via_dict_config(self):
        config = {
            'BOAT_BREADTH': 32,
            'BOAT_FUEL_RATE': 167,
            'BOAT_HBR': 30,
            'BOAT_LENGTH': 180,
            'BOAT_SMCR_POWER': 6500,
            'BOAT_SMCR_SPEED': 6,
            'BOAT_SPEED': 6,
            'WEATHER_DATA': "abc"
        }

        pol = DirectPowerBoat(init_mode="from_dict", config_dict=config)
        pol.load_data()

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

    def test_validate_parameters(self):
        """
        Test that the validate_parameters method directly identifies and raises an error for dummy values (-99).
        """
        # Create a boat object with dummy values
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        pol.Axv = -99 * u.meter * u.meter
        pol.hs1 = -99 * u.meter

        with pytest.raises(ValueError) as excinfo:
            pol.validate_parameters()

        assert "dummy values (-99)" in str(excinfo.value)
        assert "Axv" in str(excinfo.value)

    def test_valid_parameters(self):
        """
        Test that valid parameters pass validation.
        """
        config = {
            "BOAT_TYPE": "direct_power_method",
            "BOAT_BREADTH": 32,
            "BOAT_FUEL_RATE": 167,
            "BOAT_HBR": 30,
            "BOAT_LENGTH": 180,
            "BOAT_SMCR_POWER": 6502,
            "BOAT_SMCR_SPEED": 6,
            "BOAT_SPEED": 6,
            "WEATHER_DATA": "/path/to/data",
            "BOAT_HS1": 6.0,
            "BOAT_LS1": 36.0,
            "BOAT_HS2": 3.0,
            "BOAT_LS2": 54.0,
            "BOAT_CMC": -6.3,
            "BOAT_BS1": 28.8,
            "BOAT_HC": 10.0
        }

        try:
            pol = DirectPowerBoat(init_mode="from_dict", config_dict=config)
            pol.load_data()
            # If we get here, the test passes
            assert True
        except ValueError:
            # If we get here, there was an unexpected ValueError
            assert False, "Valid parameters should not raise ValueError"
