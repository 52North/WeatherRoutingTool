from datetime import datetime
import os

import matplotlib.pyplot as pyplot
import numpy as np
import pytest
from astropy import units as u

from WeatherRoutingTool.ship.nnmodel import NNBoat

import tests.basic_test_func as basic_test_func


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
            "/home/kdemmich/1_Projekte/MariData/3_Code/blackgreywhiteboxmodelle/260428/ME_LOAD_NN_final_model.pth",
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


def test_wind_dir_polar_plot(plt):
    configpath = os.path.join(
        "/home/kdemmich/1_Projekte/TwinShip/5_Results/260428_Biskaya_Model_Comparison/config_KD.json")
    dirname = os.path.dirname(__file__)

    nnboat = NNBoat(file_name=configpath)
    nnboat.weather_path = os.path.join(dirname, 'data/tests_weather_data.nc')
    nnboat.courses_path = os.path.join(dirname, 'data/CoursesRoute.nc')
    nnboat.depth_path = os.path.join(dirname, 'data/tests_depth_data.nc')
    nnboat.load_data()

    boat_speed = np.full(19, 7)
    debug = True

    wind_dir = np.linspace(0, 180, 19)
    P_perc = np.full(19, -99.)

    for ipoint in range(len(wind_dir)):
        input_dict = {
            'STW': boat_speed[ipoint],  # STW
            'AP (interpolated)': 9.5,
            'FP (interpolated)': 9.5,
            'WIND_SPEED_REL': 15,
            'WIND_DIRECTION_REL': wind_dir[ipoint],  # rel_wind_direction
            'VHM0': 1.5,  # VHM0
            'rel_seaway_direction': -45.0,  # rel_seaway_direction
        }
        if debug:
            print('input_dict: ', input_dict)
        input_data = nnboat.get_input_data(input_dict)
        if debug:
            print('input_data: ', input_data)
        pred = nnboat.predict_mean(nnboat.evaluator, nnboat.model_path, input_data)
        P_perc[ipoint] = pred[0]
        if debug:
            print('prediction: ', P_perc[ipoint])

    P_perc = P_perc / 100 * 6502

    fig, axes = pyplot.subplots(1, 1, subplot_kw={'projection': 'polar'})
    wind_dir_rad = np.radians(wind_dir)
    axes.plot(wind_dir_rad, P_perc)
    axes.legend()
    axes.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axes.set_theta_zero_location("S")
    axes.grid(True)
    axes.set_title("Power", va='bottom')

    pyplot.tight_layout()
    plt.saveas = ("/home/kdemmich/1_Projekte/TwinShip/5_Results/260428_Biskaya_Model_Comparison/"
                  "summary/polar_winddir.png")
