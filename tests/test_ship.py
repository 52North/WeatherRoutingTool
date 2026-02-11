from datetime import datetime
import os

import numpy as np
import xarray as xr
from astropy import units as u

import pytest
from WeatherRoutingTool.ship.ship_config import ShipConfig

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.ship.shipparams import ShipParams


class TestShip:
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

        (fuel_test, power_test, rpm_test, speed_test, rwind_test, rcalm_test, rwaves_test,
         rshallow_test, rroughness_test, wave_height_test, wave_direction_test, wave_period_test,
         u_currents_test, v_currents_test, u_wind_speed_test, v_wind_speed_test, pressure_test,
         air_temperature_test, salinity_test, water_temperature_test, status_test, message_test) \
            = sp.get_element(idx)

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
        check whether values of weather data are correctly read from file
    '''

    def test_evaluate_weather_for_direct_power_method(self):
        # dummy weather file
        dirname = os.path.dirname(__file__)
        weather_data = xr.open_dataset(os.path.join(dirname, 'data/tests_weather_data.nc'))

        time_single = datetime.strptime('2023-07-20', '%Y-%m-%d')

        # courses test file
        courses_test = np.array([0, 180, 0, 180, 180, 0]) * u.degree
        lat_test = np.array([54.3, 54.3, 54.6, 55.6, 55.9, 55.9])
        lon_test = np.array([13.3, 13.3, 13.6, 16., 16.9, 13.9])
        time_test = np.array([time_single, time_single, time_single, time_single, time_single, time_single])
        speed = np.full(6, 6) * u.meter / u.second

        # dummy course netCDF
        pol = basic_test_func.create_dummy_Direct_Power_Ship('simpleship')

        ship_params = pol.get_ship_parameters(courses_test, lat_test, lon_test, time_test, speed)
        ship_params.print()

        for i in range(0, 6):
            ship_params.wave_direction[i] = np.nan_to_num(ship_params.wave_direction[i])
            wavedir_data = weather_data['VMDR'].sel(latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
                                                    method='nearest', drop=False).fillna(0).to_numpy()

            ship_params.wave_height[i] = np.nan_to_num(ship_params.wave_height[i])
            waveheight_data = weather_data['VHM0'].sel(latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
                                                       method='nearest', drop=False).fillna(0).to_numpy()
            ship_params.wave_period[i] = np.nan_to_num(ship_params.wave_period[i])
            waveperiod_data = weather_data['VTPK'].sel(latitude=lat_test[i], longitude=lon_test[i], time=time_test[i],
                                                       method='nearest', drop=False).fillna(0).to_numpy()

            assert abs(ship_params.wave_direction[i].value - wavedir_data) < 0.0001
            assert abs(ship_params.wave_height[i].value - waveheight_data) < 0.0001
            assert abs(ship_params.wave_period[i].value - waveperiod_data) < 0.0001

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


# A minimal valid ship config to use as a base
VALID_SHIP_CONFIG = {
    "BOAT_BREADTH": 20,
    "BOAT_FUEL_RATE": 10,
    "BOAT_HBR": 5,
    "BOAT_LENGTH": 100,
    "BOAT_SMCR_POWER": 5000,
    "BOAT_SPEED": 10,
    "BOAT_SMCR_SPEED": 6,
    "WEATHER_DATA": "tests/data/reduced_testdata_weather.nc"
}


def test_valid_ship_config_initialization():
    """Tests that a valid ship config does not raise an exception."""
    try:
        ShipConfig.assign_config(init_mode='from_dict', config_dict=VALID_SHIP_CONFIG)
    except ValueError as e:
        pytest.fail(f"Valid ship config raised an unexpected ValueError: {e}")


def test_invalid_propulsion_efficiency_raises_error():
    """Tests that an out-of-range BOAT_PROPULSION_EFFICIENCY raises ValueError."""
    invalid_config = VALID_SHIP_CONFIG.copy()
    # This optional variable has a default, so we add it for the test
    invalid_config["BOAT_PROPULSION_EFFICIENCY"] = 1.1  # > 1
    with pytest.raises(ValueError, match="'BOAT_PROPULSION_EFFICIENCY' must be between 0 and 1"):
        ShipConfig.assign_config(init_mode='from_dict', config_dict=invalid_config)
