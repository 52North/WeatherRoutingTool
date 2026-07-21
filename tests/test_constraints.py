import os
from datetime import datetime

import numpy as np
import xarray as xr
from astropy import units as u

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.config import set_up_logging
from WeatherRoutingTool.constraints.constraints import (ConstraintsList, ConstraintPars, LandCrossing,
                                                        RunTestContinuousChecks, WaterDepth, WaveHeight,
                                                        StatusCodeError, StayInTime)
from WeatherRoutingTool.utils.maps import Map

set_up_logging()

'''
    test adding of negative constraint to ConstraintsList.negativ_constraints
'''


def test_add_neg_constraint():
    land_crossing = LandCrossing()

    constraint_list = basic_test_func.generate_dummy_constraint_list()
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
    constraint_list = basic_test_func.generate_dummy_constraint_list()
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
    constraint_list = basic_test_func.generate_dummy_constraint_list()
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
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing(lat[1, :], lon[1, :], lat[0, :], lon[0, :], time, is_constrained)
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
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing(lat[1, :], lon[1, :], lat[0, :], lon[0, :], time, is_constrained)
    assert is_constrained[0] == 0
    assert is_constrained[1] == 1


def test_safe_waterdepth():
    lat = np.array([[51.16, 52.5], [52, 52], ])
    lon = np.array([[2.5, 2.5], [2.05, 2.5], ])
    time = 0
    dirname = os.path.dirname(__file__)
    depthfile = os.path.join(dirname, 'data/reduced_testdata_depth.nc')
    map = Map(50, 0, 55, 5)
    waterdepth = WaterDepth("from_file", 20, map, depthfile)
    # waterdepth.plot_depth_map_from_file(depthfile, 50,0,55,5)

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(waterdepth)
    is_constrained = constraint_list.safe_crossing(lat[1, :], lon[1, :], lat[0, :], lon[0, :], time, is_constrained)
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
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing_discrete(lat[1, :], lat[0, :], lon[1, :], lon[0, :], time,
                                                            is_constrained)

    assert is_constrained.shape[0] == lat.shape[1]


'''
    test results for elements of is_constrained
'''


def test_check_constraints_land_crossing():
    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.routing_step.init_step(
        lats_start=np.array([52.76, 53.45]),
        lons_start=np.array([5.40, 3.72]),
        courses=np.array([99, 99]) * u.degree,
        time=None
    )
    ra.routing_step.update_end_step(
        lats=np.array([52.70, 53.55]),
        lons=np.array([4.04, 5.45]),
    )

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5])

    # is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    ra.check_constraints(constraint_list)
    assert ra.routing_step.is_constrained[0] == 1
    assert ra.routing_step.is_constrained[1] == 0


def test_safe_crossing_continuous():
    test_case1 = [False, False, True, True, False, False, True, False, False, False]
    test_case2 = [False, False, True, False, False, False, False, False, True, False]
    is_constrained_test = [True, False, True, True, True]
    is_constrained_input = [False, False, False, False, False]

    weather_time_start = datetime(2023, 1, 1, 0, 0)
    weather_time_end = datetime(2023, 1, 2, 0, 0)

    # first start time is before the weather data, last one is after it, the middle one is within the timeframe
    time_start = np.array([datetime(2022, 12, 31, 23, 0),
                           datetime(2023, 1, 1, 12, 0),
                           datetime(2023, 1, 1, 12, 0),
                           datetime(2023, 1, 2, 12, 0),
                           datetime(2023, 1, 3, 0, 0)])

    test_mod1 = RunTestContinuousChecks(test_case1)
    test_mod2 = RunTestContinuousChecks(test_case2)
    stay_in_time = StayInTime()
    stay_in_time.set_time(weather_time_start, weather_time_end)
    dummy_lats = [0, 0, 0, 0, 0]

    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(test_mod1, "continuous")
    constraint_list.add_neg_constraint(test_mod2, "continuous")
    constraint_list.add_neg_constraint(stay_in_time, "continuous")
    is_constrained = constraint_list.safe_crossing_continuous(dummy_lats, dummy_lats, dummy_lats, dummy_lats,
                                                              time_start, is_constrained_input)

    assert np.array_equal(is_constrained_test, is_constrained)


'''
    test elements of is_constrained for status error of the status values = [1, 2, 3, 2, 3, 1]
    where 3 is the the error value
'''


def test_check_crossing_status_errror():
    ref_is_constrained = np.array([False, False, True, False, True, False])
    ref_lat = [1, 1, 1, 2, 2, 2]
    ref_lon = [4, 4, 4, 3, 3, 3]

    dirname = os.path.dirname(__file__)
    coursesfile = os.path.join(dirname, 'data/CoursesRouteStatus.nc')
    statusCodeError = StatusCodeError(coursesfile)

    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(statusCodeError, 'continuous')

    is_constrained = constraint_list.negative_constraints_continuous[0].check_crossing(ref_lat, ref_lon)

    assert np.array_equal(ref_is_constrained, is_constrained)


'''
    test StayInTime.check_crossing for the case that the start times of all route segments lie within the
    timeframe of the weather data -> no segment shall be constrained
'''


def test_check_crossing_stay_in_time_within_range():
    weather_time_start = datetime(2023, 1, 1, 0, 0)
    weather_time_end = datetime(2023, 1, 2, 0, 0)

    stay_in_time = StayInTime()
    stay_in_time.set_time(weather_time_start, weather_time_end)

    # start times of all route segments are within [weather_time_start, weather_time_end]
    time_start = np.array([datetime(2023, 1, 1, 3, 0),
                           datetime(2023, 1, 1, 12, 0),
                           datetime(2023, 1, 1, 23, 0)])
    lat_start = np.array([10., 11., 12.])
    lon_start = np.array([4., 5., 6.])
    lat_end = np.array([11., 12., 13.])
    lon_end = np.array([5., 6., 7.])

    ref_is_constrained = np.array([False, False, False])
    is_constrained = stay_in_time.check_crossing(lat_start, lon_start, lat_end, lon_end, time_start)

    assert np.array_equal(ref_is_constrained, is_constrained)


'''
    test StayInTime.check_crossing for the case that the start times of the route segments lie partly outside the
    timeframe of the weather data -> only the segments outside the timeframe shall be constrained
'''


def test_check_crossing_stay_in_time_partly_out_of_range():
    weather_time_start = datetime(2023, 1, 1, 0, 0)
    weather_time_end = datetime(2023, 1, 2, 0, 0)

    stay_in_time = StayInTime()
    stay_in_time.set_time(weather_time_start, weather_time_end)

    # first start time is before the weather data, last one is after it, the middle one is within the timeframe
    time_start = np.array([datetime(2022, 12, 31, 23, 0),
                           datetime(2023, 1, 1, 12, 0),
                           datetime(2023, 1, 3, 0, 0)])
    lat_start = np.array([10., 11., 12.])
    lon_start = np.array([4., 5., 6.])
    lat_end = np.array([11., 12., 13.])
    lon_end = np.array([5., 6., 7.])

    ref_is_constrained = np.array([True, False, True])
    is_constrained = stay_in_time.check_crossing(lat_start, lon_start, lat_end, lon_end, time_start)

    assert np.array_equal(ref_is_constrained, is_constrained)
