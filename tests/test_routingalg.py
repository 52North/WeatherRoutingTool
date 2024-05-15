import numpy as np

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.ship.shipparams import ShipParams


#
# test whether ship parameters for current routing step are correctly merged to previous arrays
#
def test_update_ship_params():
    fuel = np.array([[0, 1, 2], [0.1, 1.1, 2.1], [0.2, 1.2, 2.2]])
    power = np.array([[3, 4, 5], [3.1, 4.1, 5.1], [3.2, 4.2, 5.2]])
    rpm = np.array([[5, 6, 7], [5.1, 6.1, 7.1], [5.2, 6.2, 7.2]])
    speed = np.array([[8, 9, 8], [8.1, 9.1, 8.1], [8.2, 9.2, 8.2]])
    rcalm = np.array([[8.3, 9.3, 8.3], [8.13, 9.13, 8.13], [8.23, 9.23, 8.23]])
    rwind = np.array([[8.4, 9.4, 8.4], [8.14, 9.14, 8.14], [8.24, 9.24, 8.24]])
    rshallow = np.array([[8.5, 9.5, 8.5], [8.15, 9.15, 8.15], [8.25, 9.25, 8.25]])
    rroughness = np.array([[8.6, 9.6, 8.6], [8.16, 9.16, 8.16], [8.26, 9.26, 8.26]])
    rwaves = np.array([[8.7, 9.7, 8.7], [8.17, 9.17, 8.17], [8.27, 9.27, 8.27]])
    wave_height = np.array([[4.1, 4.2, 4.3], [4.11, 4.12, 4.13], [4.21, 4.22, 4.23]])
    wave_direction = np.array([[4.4, 4.5, 4.6], [4.41, 4.42, 4.43], [4.51, 4.52, 4.53]])
    wave_period = np.array([[4.7, 4.8, 4.9], [4.71, 4.72, 4.73], [4.81, 4.81, 8.82]])
    u_currents = np.array([[5.1, 5.2, 5.3], [5.11, 5.12, 5.13], [5.21, 5.22, 5.23]])
    v_currents = np.array([[5.4, 5.5, 5.6], [5.41, 5.42, 5.43], [5.51, 5.52, 5.53]])
    u_wind_speed = np.array([[7.1, 7.2, 7.3], [7.11, 7.12, 7.13], [7.21, 7.22, 7.23]])
    v_wind_speed = np.array([[7.4, 7.5, 7.6], [7.41, 7.42, 7.43], [7.51, 7.52, 7.53]])
    pressure = np.array([[5.7, 5.8, 5.9], [5.71, 5.72, 5.73], [5.81, 5.82, 5.83]])
    air_temperature = np.array([[6.1, 6.2, 6.3], [6.11, 6.12, 6.13], [6.21, 6.22, 6.23]])
    salinity = np.array([[6.4, 6.5, 6.6], [6.41, 6.42, 6.43], [6.51, 6.52, 6.53]])
    water_temperature = np.array([[6.7, 6.8, 6.9], [6.71, 6.72, 6.73], [6.81, 6.82, 6.83]])
    status = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    message = np.array([['OK', 'OK', 'Error'], ['OK', 'Error', 'Error'], ['OK', 'Error', 'OK']])
    sp = ShipParams(fuel_rate=fuel, power=power, rpm=rpm, speed=speed, r_calm=rcalm, r_wind=rwind,
                    r_shallow=rshallow, r_roughness=rroughness, r_waves=rwaves, wave_height=wave_height,
                    wave_direction=wave_direction, wave_period=wave_period, u_currents=u_currents,
                    v_currents=v_currents, u_wind_speed=u_wind_speed, v_wind_speed=v_wind_speed, pressure=pressure,
                    air_temperature=air_temperature, salinity=salinity, water_temperature=water_temperature,
                    status=status, message=message)

    fuel_single = np.array([0.01, 0.02, 0.03])
    power_single = np.array([1.01, 1.02, 1.03])
    rpm_single = np.array([2.01, 2.02, 2.03])
    speed_single = np.array([3.01, 3.02, 3.03])
    rcalm_single = np.array([4.01, 4.02, 4.03])
    rwind_single = np.array([5.01, 5.02, 5.03])
    rshallow_single = np.array([6.01, 6.02, 6.03])
    rroughness_single = np.array([7.01, 7.02, 7.03])
    rwaves_single = np.array([8.01, 8.02, 8.03])
    wave_height_single = np.array([4.01, 4.02, 4.03])
    wave_direction_single = np.array([4.04, 4.02, 4.03])
    wave_period_single = np.array([4.07, 4.08, 4.09])
    u_currents_single = np.array([5.01, 5.02, 5.03])
    v_currents_single = np.array([5.04, 5.04, 5.05])
    u_wind_speed_single = np.array([7.01, 7.02, 7.03])
    v_wind_speed_single = np.array([7.04, 7.05, 7.06])
    pressure_single = np.array([5.07, 5.08, 5.09])
    air_temperature_single = np.array([6.01, 6.02, 6.03])
    salinity_single = np.array([6.04, 6.05, 6.06])
    water_temperature_single = np.array([6.07, 6.08, 6.09])
    status_single = np.array([1, 3, 2])
    message_single = np.array(['OK', 'OK', 'OK'])

    sp_single = ShipParams(fuel_rate=fuel_single, power=power_single, rpm=rpm_single, speed=speed_single,
                           r_calm=rcalm_single, r_wind=rwind_single, r_shallow=rshallow_single,
                           r_roughness=rroughness_single, r_waves=rwaves_single, wave_height=wave_height_single,
                           wave_direction=wave_direction_single, wave_period=wave_period_single,
                           u_currents=u_currents_single, v_currents=v_currents_single,
                           u_wind_speed=u_wind_speed_single, v_wind_speed=v_wind_speed_single,
                           pressure=pressure_single, air_temperature=air_temperature_single,
                           salinity=salinity_single, water_temperature=water_temperature_single,
                           status=status_single, message=message_single)

    ra = basic_test_func.create_dummy_IsoFuel_object()
    ra.shipparams_per_step = sp
    ra.update_shipparams(sp_single)

    # fuel_test = np.array([[0.01, 0.02, 0.03], [0, 1, 2], [0.1, 1.1, 2.1], [0.2, 1.2, 2.2]])
    power_test = np.array([[1.01, 1.02, 1.03], [3, 4, 5], [3.1, 4.1, 5.1], [3.2, 4.2, 5.2]])
    rpm_test = np.array([[2.01, 2.02, 2.03], [5, 6, 7], [5.1, 6.1, 7.1], [5.2, 6.2, 7.2]])
    speed_test = np.array([[3.01, 3.02, 3.03], [8, 9, 8], [8.1, 9.1, 8.1], [8.2, 9.2, 8.2]])
    rcalm_test = np.array([[4.01, 4.02, 4.03], [8.3, 9.3, 8.3], [8.13, 9.13, 8.13], [8.23, 9.23, 8.23]])
    rwind_test = np.array([[5.01, 5.02, 5.03], [8.4, 9.4, 8.4], [8.14, 9.14, 8.14], [8.24, 9.24, 8.24]])
    rshallow_test = np.array([[6.01, 6.02, 6.03], [8.5, 9.5, 8.5], [8.15, 9.15, 8.15], [8.25, 9.25, 8.25]])
    rroughness_test = np.array([[7.01, 7.02, 7.03], [8.6, 9.6, 8.6], [8.16, 9.16, 8.16], [8.26, 9.26, 8.26]])
    rwaves_test = np.array([[8.01, 8.02, 8.03], [8.7, 9.7, 8.7], [8.17, 9.17, 8.17], [8.27, 9.27, 8.27]])
    wave_height_test = np.array([[4.01, 4.02, 4.03], [4.1, 4.2, 4.3], [4.11, 4.12, 4.13], [4.21, 4.22, 4.23]])
    wave_direction_test = np.array([[4.04, 4.02, 4.03], [4.4, 4.5, 4.6], [4.41, 4.42, 4.43], [4.51, 4.52, 4.53]])
    wave_period_test = np.array([[4.07, 4.08, 4.09], [4.7, 4.8, 4.9], [4.71, 4.72, 4.73], [4.81, 4.81, 8.82]])
    u_currents_test = np.array([[5.01, 5.02, 5.03], [5.1, 5.2, 5.3], [5.11, 5.12, 5.13], [5.21, 5.22, 5.23]])
    v_currents_test = np.array([[5.04, 5.04, 5.05], [5.4, 5.5, 5.6], [5.41, 5.42, 5.43], [5.51, 5.52, 5.53]])
    u_wind_speed_test = np.array([[7.01, 7.02, 7.03], [7.1, 7.2, 7.3], [7.11, 7.12, 7.13], [7.21, 7.22, 7.23]])
    v_wind_speed_test = np.array([[7.04, 7.05, 7.06], [7.4, 7.5, 7.6], [7.41, 7.42, 7.43], [7.51, 7.52, 7.53]])
    pressure_test = np.array([[5.07, 5.08, 5.09], [5.7, 5.8, 5.9], [5.71, 5.72, 5.73], [5.81, 5.82, 5.83]])
    air_temperature_test = np.array([[6.01, 6.02, 6.03], [6.1, 6.2, 6.3], [6.11, 6.12, 6.13], [6.21, 6.22, 6.23], ])
    salinity_test = np.array([[6.04, 6.05, 6.06], [6.4, 6.5, 6.6], [6.41, 6.42, 6.43], [6.51, 6.52, 6.53]])
    water_temperature_test = np.array([[6.07, 6.08, 6.09], [6.7, 6.8, 6.9], [6.71, 6.72, 6.73], [6.81, 6.82, 6.83]])
    status_test = np.array([[1, 3, 2], [1, 2, 3], [2, 3, 1], [3, 2, 1]])
    message_test = np.array([['OK', 'OK', 'OK'], ['OK', 'OK', 'Error'], ['OK', 'Error', 'Error'],
                             ['OK', 'Error', 'OK']])

    assert np.array_equal(power_test, ra.shipparams_per_step.get_power())
    assert np.array_equal(rpm_test, ra.shipparams_per_step.get_rpm())
    assert np.array_equal(speed_test, ra.shipparams_per_step.get_speed())
    assert np.array_equal(rcalm_test, ra.shipparams_per_step.get_rcalm())
    assert np.array_equal(rwind_test, ra.shipparams_per_step.get_rwind())
    assert np.array_equal(rshallow_test, ra.shipparams_per_step.get_rshallow())
    assert np.array_equal(rroughness_test, ra.shipparams_per_step.get_rroughness())
    assert np.array_equal(rwaves_test, ra.shipparams_per_step.get_rwaves())
    assert np.array_equal(wave_height_test, ra.shipparams_per_step.get_wave_height())
    assert np.array_equal(wave_direction_test, ra.shipparams_per_step.get_wave_direction())
    assert np.array_equal(wave_period_test, ra.shipparams_per_step.get_wave_period())
    assert np.array_equal(u_currents_test, ra.shipparams_per_step.get_u_currents())
    assert np.array_equal(v_currents_test, ra.shipparams_per_step.get_v_currents())
    assert np.array_equal(u_wind_speed_test, ra.shipparams_per_step.get_u_wind_speed())
    assert np.array_equal(v_wind_speed_test, ra.shipparams_per_step.get_v_wind_speed())
    assert np.array_equal(pressure_test, ra.shipparams_per_step.get_pressure())
    assert np.array_equal(air_temperature_test, ra.shipparams_per_step.get_air_temperature())
    assert np.array_equal(salinity_test, ra.shipparams_per_step.get_salinity())
    assert np.array_equal(water_temperature_test, ra.shipparams_per_step.get_water_temperature())
    assert np.array_equal(status_test, ra.shipparams_per_step.get_status())
    assert np.array_equal(message_test, ra.shipparams_per_step.get_message())
