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
    sp = ShipParams(fuel_rate=fuel, power=power, rpm=rpm, speed=speed, r_calm=rcalm, r_wind=rwind, r_shallow=rshallow,
                    r_roughness=rroughness, r_waves=rwaves)

    fuel_single = np.array([0.01, 0.02, 0.03])
    power_single = np.array([1.01, 1.02, 1.03])
    rpm_single = np.array([2.01, 2.02, 2.03])
    speed_single = np.array([3.01, 3.02, 3.03])
    rcalm_single = np.array([4.01, 4.02, 4.03])
    rwind_single = np.array([5.01, 5.02, 5.03])
    rshallow_single = np.array([6.01, 6.02, 6.03])
    rroughness_single = np.array([7.01, 7.02, 7.03])
    rwaves_single = np.array([8.01, 8.02, 8.03])
    sp_single = ShipParams(fuel_rate=fuel_single, power=power_single, rpm=rpm_single, speed=speed_single,
                           r_calm=rcalm_single, r_wind=rwind_single, r_shallow=rshallow_single,
                           r_roughness=rroughness_single, r_waves=rwaves_single)

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

    assert np.array_equal(power_test, ra.shipparams_per_step.get_power())
    assert np.array_equal(rpm_test, ra.shipparams_per_step.get_rpm())
    assert np.array_equal(speed_test, ra.shipparams_per_step.get_speed())
    assert np.array_equal(rcalm_test, ra.shipparams_per_step.get_rcalm())
    assert np.array_equal(rwind_test, ra.shipparams_per_step.get_rwind())
    assert np.array_equal(rshallow_test, ra.shipparams_per_step.get_rshallow())
    assert np.array_equal(rroughness_test, ra.shipparams_per_step.get_rroughness())
    assert np.array_equal(rwaves_test, ra.shipparams_per_step.get_rwaves())
