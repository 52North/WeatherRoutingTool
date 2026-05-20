from datetime import datetime
import os

import matplotlib.pyplot as pyplot
import numpy as np
import pytest
from astropy import units as u

from WeatherRoutingTool.ship.ship_config import ShipConfig
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
    ship_config = ShipConfig.assign_config(init_mode="from_dict", config_dict=config_dict)
    boat = NNBoat(ship_config)
    delta_ang = boat.get_relative_wind_dir(ang_boat * u.degree, ang_wind * u.degree)

    assert delta_ang == delta_ang_test * u.degree
    assert delta_ang <= 180 * u.degree
    assert delta_ang >= -180 * u.degree


@pytest.mark.parametrize("ang_boat,ang_wind,delta_ang_test",
                         [(0, 45, 45), (0, 315, -45), (90, 120, 30), (120, 90, -30), (270, 10, 100)])
def test_get_relative_wind_dir_asymm(ang_boat, ang_wind, delta_ang_test):
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
    ship_config = ShipConfig.assign_config(init_mode="from_dict", config_dict=config_dict)
    boat = NNBoat(ship_config)
    delta_ang = boat.get_relative_wind_dir_asymmetric(ang_boat * u.degree, ang_wind * u.degree)

    assert delta_ang == delta_ang_test * u.degree
    assert delta_ang <= 180 * u.degree
    assert delta_ang >= -180 * u.degree


@pytest.fixture
def model_name(request):
    return request.config.getoption("--model_name")


@pytest.fixture
def model_path(request):
    return request.config.getoption("--model_path")


@pytest.fixture
def model_type(request):
    return request.config.getoption("--model_type")


@pytest.fixture
def config_dir(request):
    return request.config.getoption("--config_dir")


@pytest.fixture
def results_dir(request):
    return request.config.getoption("--results_dir")


def get_input_pars(dataset: str, wind_var: bool, wave_var: bool, speed_var: bool, npoints: int, ipoint: int):
    available_pars = {
        'STW': 6,  # STW
        'AP': 9.5,
        'FP': 9.5,
        'WIND_SPEED_REL': 5,
        'WIND_DIRECTION_REL': -45.0,  # rel_wind_direction
        'VHM0': 1.5,  # VHM0
        'rel_seaway_direction': -45.0  # rel_seaway_direction
    }

    par_list = []
    if dataset == "smallDataset_Copernicus":
        par_list = ["STW", "AP", "FP", "WIND_SPEED_REL", "WIND_DIRECTION_REL", "VHM0", "rel_seaway_direction"]
    if dataset == "smallDataset_boardWind_9features":
        par_list = ["STW", "AP", "FP", "WIND_SPEED_REL", "WIND_DIRECTION_REL", "VHM0", "rel_seaway_direction"]
    if dataset == "STWAndWaves":
        par_list = ["STW", "VHM0", "rel_seaway_direction"]
    if dataset == "STWAndWind":
        par_list = ["STW", "WIND_SPEED_REL", "WIND_DIRECTION_REL"]
    if dataset == "NNfinal":
        par_list = ["STW", "AP", "FP", "WIND_SPEED_REL", "WIND_DIRECTION_REL", "VHM0", "rel_seaway_direction"]

    input_data = {}
    for var in par_list:
        input_data[var] = available_pars[var]

        if (var == "WIND_DIRECTION_REL" and wind_var) or (var == "rel_seaway_direction" and wave_var):
            temp = np.linspace(-180, 180, npoints)
            input_data[var] = temp[ipoint]

        if (var == "STW" and speed_var):
            temp = np.linspace(5.5, 7, npoints)
            input_data[var] = temp[ipoint]

    return input_data


def test_wind_dir_polar_plot(plt, model_name, model_path, model_type, config_dir, results_dir):
    print(f"\nTesting Model: {model_name} with Path {model_path}")

    nnboat = basic_test_func.create_dummy_NNBoat(config_dir)
    nnboat.model_path = model_path
    nnboat.load_data()

    debug = True
    npoints = 37
    P_perc = np.full(npoints, -99.)

    for ipoint in range(npoints):
        input_dict = get_input_pars(
            dataset=model_type,
            wind_var=True,
            wave_var=False,
            speed_var=False,
            npoints=npoints,
            ipoint=ipoint
        )
        if debug:
            print('input_dict: ', input_dict)
        input_data = nnboat.get_input_data(input_dict)
        if debug:
            print('input_data: ', input_data)

        # call NN models using predict_mean, call GP models using predict_with_uncertainty
        pred = None
        if "GP" in model_name:
            pred, std = nnboat.predict_with_uncertainty(nnboat.evaluator, nnboat.model_path, input_data, model_type)
        else:
            pred = nnboat.predict_mean(nnboat.evaluator, nnboat.model_path, input_data, model_type)

        P_perc[ipoint] = pred[0]
        if debug:
            print('prediction: ', P_perc[ipoint])

    P_perc = P_perc / 100 * 6502

    wind_dir = np.linspace(-180, 180, npoints)
    fig, axes = pyplot.subplots(1, 1, subplot_kw={'projection': 'polar'})
    wind_dir_rad = np.radians(wind_dir)
    axes.plot(wind_dir_rad, P_perc)
    axes.legend()
    axes.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axes.set_theta_zero_location("S")
    axes.set_thetamin(-180)
    axes.set_thetamax(180)
    axes.grid(True)
    axes.set_title("Power", va='bottom')

    ticks_deg = np.arange(-180, 180, 45)
    ticks_rad = np.deg2rad(ticks_deg)
    axes.set_xticks(ticks_rad)
    axes.set_xticklabels([f"{d}°" for d in ticks_deg])

    pyplot.tight_layout()
    results_path = f"{results_dir}/{model_name}_polar_winddir.png"
    print(f"Writing figures to {results_path}")
    plt.saveas = (results_path)


def test_wave_dir_polar_plot(plt, model_name, model_path, model_type, config_dir, results_dir):
    print(f"\nTesting Model: {model_name} with Path {model_path}")

    nnboat = basic_test_func.create_dummy_NNBoat(config_dir)
    nnboat.model_path = model_path
    nnboat.load_data()

    debug = True
    npoints = 37
    P_perc = np.full(npoints, -99.)

    for ipoint in range(npoints):
        input_dict = get_input_pars(
            dataset=model_type,
            wind_var=False,
            wave_var=True,
            speed_var=False,
            npoints=npoints,
            ipoint=ipoint
        )
        if debug:
            print('input_dict: ', input_dict)
        input_data = nnboat.get_input_data(input_dict)
        if debug:
            print('input_data: ', input_data)

        # call NN models using predict_mean, call GP models using predict_with_uncertainty
        pred = None
        if "GP" in model_name:
            pred, std = nnboat.predict_with_uncertainty(nnboat.evaluator, nnboat.model_path, input_data, model_type)
        else:
            pred = nnboat.predict_mean(nnboat.evaluator, nnboat.model_path, input_data, model_type)

        P_perc[ipoint] = pred[0]
        if debug:
            print('prediction: ', P_perc[ipoint])

    P_perc = P_perc / 100 * 6502

    wave_dir = np.linspace(-180, 180, npoints)
    fig, axes = pyplot.subplots(1, 1, subplot_kw={'projection': 'polar'})
    wave_dir_rad = np.radians(wave_dir)
    axes.plot(wave_dir_rad, P_perc)
    axes.legend()
    axes.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axes.set_theta_zero_location("S")
    axes.set_thetamin(-180)
    axes.set_thetamax(180)
    axes.grid(True)
    axes.set_title("Power", va='bottom')

    ticks_deg = np.arange(-180, 180, 45)
    ticks_rad = np.deg2rad(ticks_deg)
    axes.set_xticks(ticks_rad)
    axes.set_xticklabels([f"{d}°" for d in ticks_deg])

    pyplot.tight_layout()
    results_path = f"{results_dir}/{model_name}_polar_wavedir.png"
    print(f"Writing figures to {results_path}")
    plt.saveas = (results_path)


def test_speed_dependence(plt, model_name, model_path, model_type, config_dir, results_dir):
    print(f"\nTesting Model: {model_name} with Path {model_path}")

    nnboat = basic_test_func.create_dummy_NNBoat(config_dir)
    nnboat.model_path = model_path
    nnboat.load_data()

    debug = True
    npoints = 15
    P_perc = np.full(npoints, -99.)

    for ipoint in range(npoints):
        input_dict = get_input_pars(
            dataset=model_type,
            wind_var=False,
            wave_var=False,
            speed_var=True,
            npoints=npoints,
            ipoint=ipoint
        )
        if debug:
            print('input_dict: ', input_dict)
        input_data = nnboat.get_input_data(input_dict)
        if debug:
            print('input_data: ', input_data)

        # call NN models using predict_mean, call GP models using predict_with_uncertainty
        pred = None
        if "GP" in model_name:
            pred, std = nnboat.predict_with_uncertainty(nnboat.evaluator, nnboat.model_path, input_data, model_type)
        else:
            pred = nnboat.predict_mean(nnboat.evaluator, nnboat.model_path, input_data, model_type)

        P_perc[ipoint] = pred[0]
        if debug:
            print('prediction: ', P_perc[ipoint])

    P_perc = P_perc / 100 * 6502

    speed = np.linspace(5.5, 7, npoints)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    ax.plot(speed, P_perc)

    pyplot.tight_layout()
    results_path = f"{results_dir}/{model_name}_speed_dependence.png"
    print(f"Writing figures to {results_path}")
    plt.saveas = (results_path)


@pytest.mark.parametrize("ang_boat,ang_wind,delta_ang_test",
                         [(0, 45, 45), (0, 315, -45), (90, 120, 30), (120, 90, -30), (270, 10, 100), (10, 270, -100),
                          (370, 270, -100)])
def test_get_relative_wind_dir_asymmetric(ang_boat, ang_wind, delta_ang_test):
    boat = basic_test_func.create_dummy_NNBoat()
    delta_ang = boat.get_relative_wind_dir_asymmetric(ang_boat * u.degree, ang_wind * u.degree)

    assert delta_ang == delta_ang_test * u.degree
    assert delta_ang <= 180 * u.degree
    assert delta_ang > -180 * u.degree
