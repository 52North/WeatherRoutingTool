import datetime
import os

import numpy as np
import xarray
import pytest

import basic_test_func
import config
from constraints.constraints import *
from utils.maps import Map
from weather import *

from WeatherRoutingTool.constraints import constraints

def generate_dummy_constraint_list():
    pars = ConstraintPars()
    pars.resolution = 1./10

    constraint_list = ConstraintsList(pars)
    return constraint_list
'''
    test adding of negative constraint to ConstraintsList.negativ_constraints
'''
def test_add_neg_constraint():
    land_crossing = LandCrossing()

    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    assert len(constraint_list.negative_constraints) == 1
    assert constraint_list.neg_size == 1

'''
    test elements of is_constrained for single end point on land and in sea
'''
def test_safe_endpoint_land_crossing():
    lat = np.array([52.7, 53.04])
    lon = np.array([4.04, 5.66])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5,5])

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
    wave_height.current_wave_height = np.array([11,11])

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
    lat = np.array([
        [52.70, 53.55],
        [52.76, 53.45],
    ])
    lon = np.array([
        [4.04, 5.45],
        [5.40, 3.72]
    ])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5,5])

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing(lat[1,:], lat[0,:], lon[1,:], lon[0,:] ,time, is_constrained)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 0

'''
    test elements of is_constrained for investigation of crossing waves
'''
def test_safe_crossing_wave_height():
    lat = np.array([
        [54.07, 53.55],
        [54.11, 53.45],
    ])
    lon = np.array([
        [4.80, 5.45],
        [7.43, 3.72]
    ])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5,11])

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing(lat[1,:], lat[0,:], lon[1,:], lon[0,:] , time, is_constrained)
    assert is_constrained[0] == 0
    assert is_constrained[1] == 1

def test_safe_waterdepth():
    lat = np.array([
        [51.16, 52.5],
        [52, 52],
    ])
    lon = np.array([
        [2.5, 2.5],
        [2.05, 2.5],
    ])
    time = 0
    depthfile = config.DEPTH_DATA
    map = Map(50, 0, 55, 5)
    waterdepth = WaterDepth(depthfile, 20, map)
    #waterdepth.plot_depth_map_from_file(depthfile, 50,0,55,5)

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(waterdepth)
    is_constrained = constraint_list.safe_crossing(lat[1,:], lat[0,:], lon[1,:], lon[0,:] , time, is_constrained)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 0


'''
def test_adjust_depth_format():
    lon = np.array([2.802,2.885,5.292,354.917,355.498,358.901])
    lat = np.array([48.083, 48.166,48.249,48.332, 48.415,48.498])
    lon_test = np.array([-5.083,-4.502,-1.099,2.802,2.885,5.292])
    ds_test_file = "/home/kdemmich/MariData/Code/MariGeoRoute/Isochrone/Data/20221110/test_depth_merge.nc"

    print('len lon', lon.shape)

    a = np.random.rand(lat.shape[0], int(lat.shape[0]/2))
    b = np.random.rand(lat.shape[0], int(lat.shape[0]/2))

    print('shape a', a.shape)
    print('shape b', b.shape)

    c = np.concatenate((a,b), axis=1)
    d = np.concatenate((b,a), axis=1)

    print('c=', c)
    print('d=', d)

    data_vars = dict(
        deptho=(["latitude", "longitude"], c),
    )

    coords = dict(
        latitude=(["latitude"], lat),
        longitude=(["longitude"], lon),
    )
    attrs = dict(description="Necessary descriptions added here.")

    ds = xr.Dataset(data_vars, coords, attrs)
    ds.to_netcdf(ds_test_file)

    windfile = os.environ['BASE_PATH'] +  "/tests/data/77175d34-9006-11ed-b628-0242ac120003_Brighton_Rotterdam.nc"
    model = '2020111600'
    start_time = datetime.datetime.now()
    hours = 0

    weather = WeatherCondCMEMS(windfile, model, start_time, hours, 3)
    depth_test = weather.adjust_depth_format(ds_test_file)
    lat_test_read = depth_test['latitude'].to_numpy()
    lon_test_read = depth_test['longitude'].to_numpy()
    depth_test_read = depth_test['deptho'].to_numpy()

    assert np.allclose(lon_test_read, lon_test, 0.00000001)
    assert np.allclose(lat_test_read, lat, 0.00000001)
    assert np.allclose(d, depth_test_read, 0.00000001)
'''

def test_depth_interpolation_depth():
    debug = True
    depthfile = config.DEPTH_DATA
    windfile = os.environ['BASE_PATH'] +  "/tests/data/77175d34-9006-11ed-b628-0242ac120003_Brighton_Rotterdam.nc"

    lat = [55.0,49.60, 50.14, 50.98, 51.60,
           52.29, 53.18, 53.92, 54.26, 55.0]

    lon = [-1.04,-3.98, -2.30, 1.43, 2.65,
           2.75, 3.70, 4.98, 6.28, 0.41]

    ds_orig=xr.open_dataset(depthfile)
    if(debug): print('ds_orig', ds_orig)
    depth_orig = np.full(len(lat), -99)

    for i in range (0, len(lat)):
        #depth_orig[i] = ds_orig['deptho'].sel(latitude = lat[i], longitude = lon[i], method = "nearest").to_numpy()
        lon_test = lon[i]
        ds_rounded=ds_orig.interp(lat = lat[i], lon = lon_test, method='linear')
        depth_orig[i] = ds_rounded['z'].to_numpy()

        if(debug): print('i=' + str(i) + ': [' + str(lat[i]) + ',' + str(lon[i]) + ']=' + str(depth_orig[i]))

    map = Map(49,-4, 56, 7)
    waterdepth = WaterDepth(depthfile, 20, map)

    depth_int = waterdepth.get_current_depth(lat, lon)
    diff = (depth_int - depth_orig) < 1

    for i in range (0, len(lat)):
       if(debug): print('i=' + str(i) + ': [' + str(lat[i]) + ',' + str(lon[i]) + ']=' + str(depth_int[i]) + ', diff=' + str(diff))

    assert diff.all

'''
    test shape of is_constrained
'''
def test_safe_crossing_shape_return():
    lat = np.array([
        [54.07, 53.55],
        [54.11, 53.45],
    ])
    lon = np.array([
        [4.80, 5.45],
        [7.43, 3.72]
    ])
    time = 0

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5,11])

    is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = constraint_list.safe_crossing_discrete(lat[1,:], lat[0,:], lon[1,:], lon[0,:] , time, is_constrained)

    assert is_constrained.shape[0] == lat.shape[1]

'''
    test results for elements of is_constrained
'''
def test_check_constraints_land_crossing():
    move = {'lat2' : np.array([52.70, 53.55]),  #1st point: land crossing (failure), 2nd point: no land crossing(success)
            'lon2' : np.array([4.04, 5.45])}

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[52.76, 53.45]])
    ra.lons_per_step = np.array([[5.40, 3.72]])

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5])

    #is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)
    is_constrained = ra.check_constraints(move, constraint_list)
    assert is_constrained[0] == 1
    assert is_constrained[1] == 0

def test_connect_database():
    assert isinstance(constraints.ContinuousCheck.connect_database(), type(db.create_engine('postgresql://myuser:***@172.25.0.3/mydatabase')))


def test_safe_crossing_continuous():
    test_case1 = [False,False,True,True,False, False,True,False,False,False]
    test_case2 = [False,True,True,False,False, False,False,False,True,False]
    is_constrained_test = [False,True,True,True,False]

    test_mod1 = TestContinuousChecks(test_case1)
    test_mod2 = TestContinuousChecks(test_case2)
    dummy_lats = [0,0,0,0,0]

    constraint_list = generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(test_mod1, "continuous")
    constraint_list.add_neg_constraint(test_mod2, "continuous")
    is_constrained = constraint_list.safe_crossing_continuous(dummy_lats, dummy_lats,dummy_lats,dummy_lats,dummy_lats)

    assert np.array_equal(is_constrained_test, is_constrained)
