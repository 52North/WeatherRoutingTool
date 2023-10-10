import datetime
import os

import numpy as np
import xarray
import pytest

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.config
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import *

from WeatherRoutingTool.constraints import constraints


def generate_dummy_constraint_list():
    pars = ConstraintPars()
    pars.resolution = 1. / 10

    constraint_list = ConstraintsList(pars)
    return constraint_list


'''
    test adding of negative constraint to ConstraintsList.negativ_constraints
'''


def test_add_neg_constraint():
    land_crossing = LandCrossing()

    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing, 'continuous')
    assert len(constraint_list.negative_constraints_continuous) == 1
    assert constraint_list.neg_cont_size == 1


'''
    test elements of is_constrained for single end point on land and in sea
'''


def test_safe_endpoint_land_crossing():
    lat = np.array([52.7, 53.04])
    lon = np.array([4.04, 5.66])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5])

    is_constrained = [False for i in range(0, lat.shape[0])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_endpoint(lat, lon, time, is_constrained)
    assert is_constrained[0] == 0
    assert is_constrained[1] == 1


'''
    test elements of is_constrained for single end point and to large wave heights
'''


def test_safe_endpoint_wave_heigth():
    lat = np.array([52.7, 53.55])
    lon = np.array([4.04, 5.45])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([11, 11])

    is_constrained = [False for i in range(0, lat.shape[0])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_endpoint(lat, lon, time, is_constrained)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 1


'''
    test elements of is_constrained for investigation of crossing land
'''


def test_safe_crossing_land_crossing():
    lat = np.array([[52.70, 53.55], [52.76, 53.45], ])
    lon = np.array([[4.04, 5.45], [5.40, 3.72]])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5])

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing(lat[1, :], lon[1, :],lat[0, :],  lon[0, :], time, is_constrained)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 0


'''
    test elements of is_constrained for investigation of crossing waves
'''


def test_safe_crossing_wave_height():
    lat = np.array([[54.07, 53.55], [54.11, 53.45], ])
    lon = np.array([[4.80, 5.45], [7.43, 3.72]])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 11])

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing(lat[1, :], lon[1, :], lat[0, :], lon[0, :], time, is_constrained)
    assert is_constrained[0] == 0
    assert is_constrained[1] == 1


def test_safe_waterdepth():
    lat = np.array([[51.16, 52.5], [52, 52], ])
    lon = np.array([[2.5, 2.5], [2.05, 2.5], ])
    time = 0
    depthfile = config.DEPTH_DATA
    map = Map(50, 0, 55, 5)
    waterdepth = WaterDepth("from_file", 20, map, depthfile)
    # waterdepth.plot_depth_map_from_file(depthfile, 50,0,55,5)

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(waterdepth)
    is_constrained = constraint_list.safe_crossing(lat[1, :],  lon[1, :], lat[0, :], lon[0, :], time, is_constrained)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 0


'''
    test shape of is_constrained
'''


def test_safe_crossing_shape_return():
    lat = np.array([[54.07, 53.55], [54.11, 53.45], ])
    lon = np.array([[4.80, 5.45], [7.43, 3.72]])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 11])

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing_discrete(lat[1, :], lat[0, :], lon[1, :], lon[0, :], time,
                                                            is_constrained)

    assert is_constrained.shape[0] == lat.shape[1]


'''
    test results for elements of is_constrained
'''


def test_check_constraints_land_crossing():
    move = {'lat2': np.array([52.70, 53.55]),
            # 1st point: land crossing (failure), 2nd point: no land crossing(success)
            'lon2': np.array([4.04, 5.45])}

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[52.76, 53.45]])
    ra.lons_per_step = np.array([[5.40, 3.72]])

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5])

    # is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = ra.check_constraints(move, constraint_list)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 0


def test_safe_crossing_continuous():
    test_case1 = [False, False, True, True, False, False, True, False, False, False]
    test_case2 = [False, True, True, False, False, False, False, False, True, False]
    is_constrained_test = [False, True, True, True, False]

    test_mod1 = RunTestContinuousChecks(test_case1)
    test_mod2 = RunTestContinuousChecks(test_case2)
    dummy_lats = [0, 0, 0, 0, 0]

    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(test_mod1, "continuous")
    constraint_list.add_neg_constraint(test_mod2, "continuous")
    is_constrained = constraint_list.safe_crossing_continuous(dummy_lats, dummy_lats, dummy_lats, dummy_lats,
                                                              dummy_lats)

    assert np.array_equal(is_constrained_test, is_constrained)
