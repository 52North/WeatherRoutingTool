from datetime import datetime
import os

import pytest
from astropy import units as u

from WeatherRoutingTool.ship.nnmodel import NNBoat


@pytest.mark.parametrize("ang_boat,ang_wind,delta_ang_test",
                         [(0, 45, 45), (0, 315, 45), (90, 120, 30), (120, 90, 30), (270, 10, 100)])
def test_get_relative_wind_dir(ang_boat, ang_wind, delta_ang_test):
    config_dict = {
        "WEATHER_DATA": "/path/to/data",
        "DEPTH_DATA": " ",
        "BOAT_SMCR_POWER": 6502,
        "BOAT_SMCR_SPEED": 7,
        "BOAT_FUEL_RATE": 167,
        "BOAT_SPEED": 6,
        "BOAT_CORRECT_BY_NNMODEL": False,
        "BOAT_NNMODEL_PATH":
            "/home/kdemmich/1_Projekte/MariData/3_Code/blackgreywhiteboxmodelle/260202/nn_model_trial_26_rank_1.pth",
        "BOAT_TYPE": "nnmodel",

        "BOAT_DRAUGHT_AFT": 10,
        "BOAT_DRAUGHT_FORE": 10,
        "BOAT_LENGTH": 180,
        "BOAT_BREADTH": 32,
        "BOAT_HBR": 30,
        "BOAT_AXV": 716,
        "BOAT_AYV": 1910,
        "BOAT_AOD": 529.07,
        "BOAT_CMC": 8.1,
        "BOAT_HC": 7.06,
        "BOAT_UNDER_KEEL_CLEARANCE": 20
    }
    boat = NNBoat(init_mode="from_dict", config_dict=config_dict)
    delta_ang = boat.get_relative_wind_dir(ang_boat * u.degree, ang_wind * u.degree)

    assert delta_ang == delta_ang_test * u.degree
    assert delta_ang <= 180 * u.degree
    assert delta_ang >= 0 * u.degree
