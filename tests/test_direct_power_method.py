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
    pass


class TestDPM:
    """
    DIRECT POWER METHOD: check whether class variables (speed, eta_prop, power_at_sp, overload_factor) are set as
    expected and correct power and corresponding unit are returned.
    """

    def _assert_geometry(self, pol):
        """Helper to verify standard ship geometry."""
        hbr = 30 * u.meter
        breadth = 32 * u.meter
        length = 180 * u.meter
        
        assert pol.hbr == hbr
        assert pol.breadth == breadth
        assert pol.length == length
        assert pol.ls1 == 0.2 * length
        assert pol.ls2 == 0.3 * length
        assert pol.bs1 == 0.9 * breadth
        assert pol.hs1 == 0.2 * hbr
        assert pol.hs2 == 0.1 * hbr
        assert pol.cmc == -0.035 * length
        assert pol.hc == 10 * u.meter
        assert pol.Axv == 940.8 * u.meter**2
        assert pol.Ayv == 4248 * u.meter**2
        assert pol.Aod == 378 * u.meter**2

    @pytest.mark.parametrize("DeltaR,speed,design_power", [(5000, 6, 6502000 * 0.75)])
    def test_get_power_for_direct_power_method(self, DeltaR, speed, design_power):
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        expected_p = DeltaR * speed / 0.63 + design_power

        p_test = pol.get_power(5000 * u.N)
        assert p_test.value == pytest.approx(expected_p)
        assert p_test.unit == 'W'

    def test_get_wind_dir(self):
        wind_dir = np.array([30, 120, 210, 300]) * u.degree
        absv = 20
        courses = np.array([10, 10, 20, 20]) * u.degree
        rel_wind_dir_expected = np.array([20, 110, 170, 70]) * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')

        v_wind = -absv * np.cos(np.radians(wind_dir)) * u.meter / u.second
        u_wind = -absv * np.sin(np.radians(wind_dir)) * u.meter / u.second

        true_wind_dir = pol.get_wind_dir(u_wind, v_wind)
        rel_wind_dir_actual = pol.get_relative_wind_dir(courses, true_wind_dir)

        diff = (rel_wind_dir_expected - rel_wind_dir_actual).value
        assert np.all(np.abs(diff) < 0.0001)

    def test_get_apparent_wind(self):
        wind_dir = np.array([0, 45, 90, 135, 180]) * u.degree
        wind_speed = np.array([10, 10, 10, 10, 10]) * u.meter / u.second
        
        speed_expected = np.array([16, 14.86112, 11.66190, 7.15173, 4]) * u.meter / u.second
        angle_expected = np.array([0, 28.41, 59.04, 98.606, 180]) * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        wind_result = pol.get_apparent_wind(wind_speed, wind_dir)

        # Fixed range from 0, 4 to 0, 5 to cover all test data
        for i in range(len(wind_dir)):
            assert wind_result['app_wind_speed'][i].value == pytest.approx(speed_expected[i].value, abs=0.01)
            assert wind_result['app_wind_angle'][i].value == pytest.approx(angle_expected[i].value, abs=0.01)

    @pytest.mark.manual
    def test_get_apparent_wind_polar_plot(self):
        """Image of the resulting polar plot showing vector relationship between true and apparent wind."""
        wind_dir = np.linspace(0, 180, 19) * u.degree
        wind_speed = np.full(19, 10) * u.meter / u.second

        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        wind_result = pol.get_apparent_wind(wind_speed, wind_dir)

        fig, axes = plt.subplots(subplot_kw={'projection': 'polar'})
        
        axes.plot(np.radians(wind_dir), wind_speed, label="true wind")
        axes.plot(np.radians(wind_result['app_wind_angle']), wind_result['app_wind_speed'], label="apparent wind")
        axes.legend(loc="upper right")
        axes.set_title("Wind direction", va='bottom')
        plt.show()

    def test_calculate_geometry_simple_method(self):
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        pol.load_data()
        self._assert_geometry(pol)

    def test_calculate_geometry_manual_method(self):
        pol = basic_test_func.create_dummy_Direct_Power_Ship('manualship')
        pol.load_data()

        # Values specific to 'manualship' config
        assert pol.cmc == 8.1 * u.meter
        assert pol.hc == 7.06 * u.meter
        assert pol.Axv == 716 * u.meter**2
        assert pol.Ayv == 1910 * u.meter**2
        assert pol.Aod == 529 * u.meter**2

    @pytest.mark.manual
    def test_wind_coeff(self):
        u_wind_speed = np.zeros(19) * u.meter / u.second
        v_wind_speed = np.full(19, -10) * u.meter / u.second
        courses = np.linspace(0, 180, 19) * u.degree

        pol = basic_test_func.create_dummy_Direct_Power_Ship('manualship')
        r_wind = pol.get_wind_resistance(u_wind_speed, v_wind_speed, courses)

        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
        ax.plot(courses, r_wind["wind_coeff"], color=graphics.get_colour(0), label='CAA')
        ax.set_xlabel('angle of attack (degrees)')
        ax.set_ylabel(r'$C_{AA}$')
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
        self._assert_geometry(pol)

    def test_validate_parameters(self):
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')
        pol.Axv = -99 * u.meter**2
        
        with pytest.raises(ValueError, match="dummy values \(-99\)"):
            pol.validate_parameters()

    def test_valid_parameters(self):
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
        pol = DirectPowerBoat(init_mode="from_dict", config_dict=config)
        pol.load_data()
        # Ensure it doesn't raise
        pol.validate_parameters()
