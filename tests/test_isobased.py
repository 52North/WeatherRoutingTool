import datetime
import os

import numpy as np
import pytest
import xarray as xr
from geovectorslib import geod

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.config
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.ship.shipparams import ShipParams

'''
    test whether IsoBased.update_position() updates current_azimuth, lats/lons_per_step, dist_per_step correctly
        - boat crosses land
'''
def test_update_position_fail():
    lat_start = 51.289444
    lon_start = 6.766667
    lat_end = 60.293333
    lon_end = 5.218056
    dist_travel = 1007.091*1000
    az = 355.113
    az_till_start = 330.558

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start,lat_start,lat_start,lat_start]])
    ra.lons_per_step = np.array([[lon_start,lon_start,lon_start,lon_start]])
    ra.azimuth_per_step = np.array([[0,0,0,0]])
    ra.dist_per_step = np.array([[0,0,0,0]])
    ra.current_variant = np.array([az,az,az,az])

    dist = np.array([dist_travel,dist_travel,dist_travel,dist_travel])

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5, 5, 5])

    # is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)

    move = ra.check_bearing(dist)
    constraints = ra.check_constraints(move, constraint_list)
    ra.update_position(move, constraints, dist)

    lats_test = np.array([
        [lat_end, lat_end, lat_end, lat_end],
        [lat_start,lat_start,lat_start,lat_start]]
    )
    lons_test = np.array([
        [lon_end, lon_end, lon_end, lon_end],
        [lon_start, lon_start, lon_start, lon_start]]
    )
    dist_test = np.array([
        [dist_travel,dist_travel,dist_travel,dist_travel],
        [0, 0, 0, 0]
    ])
    az_test = np.array([
        #[az_till_start,az_till_start,az_till_start,az_till_start],
        [az,az,az,az],
        [0,0,0,0]
    ])


    assert np.allclose(lats_test, ra.lats_per_step, 0.01)
    assert np.allclose(lons_test, ra.lons_per_step, 0.01)
    assert np.allclose(ra.current_variant,np.array([az_till_start,az_till_start,az_till_start,az_till_start]), 0.1)
    assert np.array_equal(ra.dist_per_step,dist_test)
    assert np.array_equal(ra.azimuth_per_step,az_test)

    assert np.array_equal(ra.full_dist_traveled, np.array([0,0,0,0]))


'''
    test whether IsoBased.update_position() updates current_azimuth, lats/lons_per_step, dist_per_step correctly
        - no land crossing
'''
def test_update_position_success():
    lat_start = 53.55
    lon_start = 5.45
    lat_end = 53.45
    lon_end = 3.72
    dist_travel = 1007.091*1000
    az = 355.113
    az_till_start = 330.558

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start,lat_start,lat_start,lat_start]])
    ra.lons_per_step = np.array([[lon_start,lon_start,lon_start,lon_start]])
    ra.azimuth_per_step = np.array([[0,0,0,0]])
    ra.dist_per_step = np.array([[0,0,0,0]])
    ra.current_variant = np.array([az,az,az,az])

    dist = np.array([dist_travel,dist_travel,dist_travel,dist_travel])

    land_crossing = LandCrossing()
    wave_height = WaveHeight()
    wave_height.current_wave_height = np.array([5, 5, 5, 5])

    # is_constrained = [False for i in range(0, lat.shape[1])]
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    constraint_list.add_neg_constraint(land_crossing)
    constraint_list.add_neg_constraint(wave_height)

    move = ra.check_bearing(dist)
    constraints = ra.check_constraints(move, constraint_list)
    ra.update_position(move, constraints, dist)

    no_constraints = ra.full_dist_traveled > 0
    assert np.array_equal(no_constraints, np.array([1,1,1,1]))

##
# test wheather IsoBased::checkbearing correcly sets is_last_step to True and whether the returned variables are correct
def test_check_bearing_true():

    ra = basic_test_func.create_dummy_IsoBased_object()

    lat_start = 54.87
    lon_start = 13.33
    lat_end = 54.9
    lon_end = 13.46

    az = 68.087
    az_test = np.array([az,az,az,az])
    lon_test = np.array([lon_end,lon_end,lon_end,lon_end])
    lat_test = np.array([lat_end, lat_end, lat_end, lat_end])
    dist = np.array([10000000, 10000000, 10000000, 10000])


    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.azimuth_per_step = np.array([[0, 0, 0, 0]])
    ra.dist_per_step = np.array([[0, 0, 0, 0]])
    ra.current_variant = np.array([az, az, az, az])
    ra.finish = (lat_end, lon_end)
    ra.finish_temp = ra.finish

    move = ra.check_bearing(dist)

    assert ra.is_last_step == True
    assert np.allclose(move['azi2'], az_test, 0.1)
    assert np.array_equal(move['lon2'] ,lon_test)
    assert np.array_equal(move['lat2'], lat_test)
##
# test wheather IsoBased::checkbearing correcly leaves is_last_step untouched if travelled distance is small enough
def test_check_bearing_false():

    ra = basic_test_func.create_dummy_IsoBased_object()

    lat_start = 54.87
    lon_start = 13.33
    lat_end = 54.9
    lon_end = 13.46

    az = 355.113
    az_till_start = 330.558

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.azimuth_per_step = np.array([[0, 0, 0, 0]])
    ra.dist_per_step = np.array([[0, 0, 0, 0]])
    ra.current_variant = np.array([az, az, az, az])


    dist = np.array([10000, 10000, 10000, 10000])

    move = ra.check_bearing(dist)

    assert ra.is_last_step == False

##
# test whether IsoFuel::get_delta_variables_netCDF_last_step returns correct dist and delta_time. delta_fuel can't be tested
# with sensible amount of work
def test_get_delta_variables_last_step():
    lat_start = 54.87
    lon_start = 13.33
    lat_end = 54.9
    lon_end = 13.46
    boat_speed = 20

    az = 68.087
    dist_test = np.array([8987, 8987, 8987, 8987])
    time_test = dist_test / boat_speed

    ##
    # initialise routing alg
    ra = basic_test_func.create_dummy_IsoFuel_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.azimuth_per_step = np.array([[0, 0, 0, 0]])
    ra.dist_per_step = np.array([[0, 0, 0, 0]])
    ra.time = np.array([datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()])
    ra.current_variant = np.array([az, az, az, az])
    ra.finish = (lat_end, lon_end)
    ra.finish_temp = (lat_end, lon_end)

    ##
    # initialise boat
    weatherpath = os.environ['BASE_PATH'] + '/tests/data/8e7db202-df43-11ed-b7a6-9263520eff59.nc'
    routepath = os.environ['BASE_PATH'] + 'CoursesRoute.nc'
    depthpath = os.environ['DEPTH_DATA']
    tk = Tanker(-99)
    tk.set_boat_speed(boat_speed)
    tk.init_hydro_model_Route(weatherpath, routepath, depthpath)

    ##
    # initialise wind
    wind = {'tws' : np.array([5,5,5,5]), 'twa' : np.array([0,0,0,0])}

    ship_params = tk.get_fuel_per_time_netCDF(ra.get_current_azimuth(),
                                              ra.get_current_lats(),
                                              ra.get_current_lons(),
                                              ra.time)
    ship_params.print()

    delta_time, delta_fuel, dist = ra.get_delta_variables_netCDF_last_step(ship_params, tk.boat_speed_function(wind))

    assert np.allclose(dist, dist_test, 0.1)
    assert np.allclose(delta_time, time_test, 0.1)

'''
    Test whether shapes of arrays are sensible after define_variants()
'''
def test_define_variants_array_shapes():
    nof_hdgs_segments = 4
    hdgs_increments = 1

    ra = basic_test_func.create_dummy_IsoBased_object()
    current_var = ra.get_current_azimuth()
    ra.set_variant_segments(nof_hdgs_segments, hdgs_increments)

    ra.define_variants()
    ra.print_shape()
    ra.print_current_status()
    #checking 2D arrays
    assert ra.lats_per_step.shape[1] == nof_hdgs_segments+1
    assert ra.dist_per_step.shape == ra.lats_per_step.shape
    assert ra.azimuth_per_step.shape == ra.lats_per_step.shape
    assert ra.shipparams_per_step.speed.shape == ra.lats_per_step.shape
    assert ra.shipparams_per_step.fuel.shape == ra.lats_per_step.shape

    #checking 1D arrays
    assert ra.full_time_traveled.shape[0] == nof_hdgs_segments+1
    assert ra.full_dist_traveled.shape == ra.full_time_traveled.shape
    assert ra.time.shape == ra.full_time_traveled.shape
    assert ra.full_fuel_consumed.shape == ra.full_time_traveled.shape

'''
    test whether current_variant is correctly filled in define_variants()
'''
def test_define_variants_current_variant_filling():
    start = (30, 45)
    finish = (0, 20)
    ra = basic_test_func.create_dummy_IsoBased_object()

    new_azi = geod.inverse(
        [start[0]],
        [start[1]],
        [finish[0]],
        [finish[1]]
    )

    ra.define_variants()
    ra.print_shape()
    ra.print_current_status()

    # checking current_variant
    assert ra.current_variant.shape[0] == ra.lats_per_step.shape[1]

    test_current_var = np.array([new_azi['azi1']+2, new_azi['azi1']+1, new_azi['azi1'], new_azi['azi1']-1, new_azi['azi1']-2])

    for i in range(0,test_current_var.shape[0]):
       assert test_current_var[i] == ra.current_variant[i]

'''
    test whether indices survive the pruning which maximise the total distance traveled
'''
def test_pruning_select_correct_idxs():
    nof_hdgs_segments = 8
    hdgs_increments = 1

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.set_variant_segments(nof_hdgs_segments, hdgs_increments)
    ra.define_variants()

    pruning_bins =  np.array([10,20,40,60, 80])
    ra.current_variant = np.array([15,16 ,22,23, 44,45, 71,72,74])
    ra.current_azimuth = np.array([15,16 ,22,23, 44,45, 71,72,74])
    ra.full_dist_traveled = np.array([1,5, 6,1, 2,7, 10,1,8])
    ra.full_time_traveled = np.random.rand(9)
    full_fuel_consumed = np.random.rand(1,9)
    speed_per_step = np.random.rand(1,9)

    ra.dist_per_step = np.array([ra.full_dist_traveled])
    ra.azimuth_per_step = np.array([ra.current_azimuth])

    sp = ShipParams(full_fuel_consumed, np.full(full_fuel_consumed.shape, 0), np.full(full_fuel_consumed.shape, 0), speed_per_step)
    ra.shipparams_per_step = sp

    cur_var_test = np.array([16, 22, 45, 71])
    cur_azi_test = np.array([16, 22, 45, 71])
    full_dist_test = np.array([5, 6, 7, 10])
    full_time_test = np.array([ra.full_time_traveled[1],ra.full_time_traveled[2],ra.full_time_traveled[5],ra.full_time_traveled[6]])

    full_fuel_test = np.array([full_fuel_consumed[0,1],full_fuel_consumed[0,2],full_fuel_consumed[0,5],full_fuel_consumed[0,6]])
    speed_ps_test = np.array([speed_per_step[0,1],speed_per_step[0,2],speed_per_step[0,5],speed_per_step[0,6]])
    lat_test = np.array([[30,30,30,30]])
    lon_test = np.array([[45,45,45,45]])
    time_test = np.array([datetime.date.today(),datetime.date.today(),datetime.date.today(),datetime.date.today()])

    ra.print_current_status()
    form.print_line()

    ra.pruning(True, pruning_bins)

    assert np.array_equal(cur_var_test,ra.current_variant)
    assert np.array_equal(cur_azi_test, ra.current_azimuth)
    assert np.array_equal(full_time_test, ra.full_time_traveled)
    assert np.array_equal(full_dist_test, ra.full_dist_traveled)

    assert np.array_equal(cur_azi_test,ra.azimuth_per_step[0])
    assert np.array_equal(full_dist_test, ra.dist_per_step[0])
    assert np.array_equal(full_fuel_test, ra.shipparams_per_step.fuel[0])
    assert np.array_equal(speed_ps_test, ra.shipparams_per_step.speed[0])
    assert np.array_equal(lat_test, ra.lats_per_step)
    assert np.array_equal(lon_test, ra.lons_per_step)
    assert np.array_equal(time_test, ra.time)

    #form.print_line()
    #ra.print_ra()
'''
    test shape and content of 'move' for known distance, start and end points
'''
def test_check_bearing():
    lat_start = 51.289444
    lon_start = 6.766667
    lat_end = 60.293333
    lon_end = 5.218056
    dist_travel = 1007.091*1000
    az = 355.113

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.current_variant = np.array([az,az,az,az])

    dist = np.array([dist_travel, dist_travel, dist_travel, dist_travel])

    lats_test = np.array([
        [lat_end, lat_end, lat_end, lat_end],
        [lat_start, lat_start, lat_start, lat_start]]
    )
    lons_test = np.array([
        [lon_end, lon_end, lon_end, lon_end],
        [lon_start, lon_start, lon_start, lon_start]]
    )

    ra.print_current_status()
    move = ra.check_bearing(dist)

    #print('lats_test[0]', lats_test[0])
    #print('lons_test[0]', lons_test[0])
    assert np.allclose(lats_test[0], move['lat2'], 0.01)
    assert np.allclose(lons_test[0], move['lon2'], 0.01)
