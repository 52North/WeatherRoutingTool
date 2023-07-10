import numpy as np
import pytest

import basic_test_func
from constraints.constraints import *
from ship.shipparams import ShipParams




##
# test whether ship parameters for current routing step are correctly merged to previous arrays
#
def test_update_ship_params():
    fuel=np.array([[0,1,2], [0.1,1.1,2.1],[0.2,1.2,2.2]])
    power=np.array([[3,4,5], [3.1,4.1,5.1],[3.2,4.2,5.2]])
    rpm=np.array([[5,6,7], [5.1,6.1,7.1],[5.2,6.2,7.2]])
    speed=np.array([[8,9,8], [8.1,9.1,8.1],[8.2,9.2,8.2]])
    sp = ShipParams(fuel = fuel, power = power, rpm = rpm, speed = speed)

    fuel_single = np.array([0.01,0.02,0.03])
    power_single = np.array([1.01,1.02,1.03])
    rpm_single = np.array([2.01,2.02,2.03])
    speed_single = np.array([3.01,3.02,3.03])
    sp_single = ShipParams(fuel = fuel_single, power = power_single, rpm = rpm_single, speed = speed_single)

    ra = basic_test_func.create_dummy_IsoFuel_object()
    ra.shipparams_per_step = sp
    ra.update_shipparams(sp_single)

    fuel_test = np.array([[0.01,0.02,0.03], [0,1,2], [0.1,1.1,2.1],[0.2,1.2,2.2]])
    power_test =np.array([[1.01,1.02,1.03], [3,4,5], [3.1,4.1,5.1],[3.2,4.2,5.2]])
    rpm_test =np.array([[2.01,2.02,2.03], [5,6,7], [5.1,6.1,7.1],[5.2,6.2,7.2]])
    speed_test =np.array([[3.01,3.02,3.03], [8,9,8], [8.1,9.1,8.1],[8.2,9.2,8.2]])

    assert np.array_equal(power_test, ra.shipparams_per_step.get_power())
    assert np.array_equal(rpm_test, ra.shipparams_per_step.get_rpm())
    assert np.array_equal(speed_test, ra.shipparams_per_step.get_speed())



