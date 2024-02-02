import os
from datetime import datetime

import numpy as np
from astropy import units as u
from geovectorslib import geod

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.constraints.constraints import LandCrossing, WaveHeight
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
    dist_travel = 1007.091 * 1000
    az = 355.113
    az_till_start = 330.558

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.course_per_step = np.array([[0, 0, 0, 0]]) * u.degree
    ra.dist_per_step = np.array([[0, 0, 0, 0]]) * u.meter
    ra.current_course = np.array([az, az, az, az]) * u.degree

    dist = np.array([dist_travel, dist_travel, dist_travel, dist_travel]) * u.meter

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

    lats_test = np.array([[lat_end, lat_end, lat_end, lat_end], [lat_start, lat_start, lat_start, lat_start]])
    lons_test = np.array([[lon_end, lon_end, lon_end, lon_end], [lon_start, lon_start, lon_start, lon_start]])
    dist_test = np.array([[dist_travel, dist_travel, dist_travel, dist_travel], [0, 0, 0, 0]]) * u.meter
    az_test = np.array([[az, az, az, az], [0, 0, 0, 0]]) * u.degree
    current_course_test = np.array([az_till_start, az_till_start, az_till_start, az_till_start]) * u.degree

    assert np.allclose(lats_test, ra.lats_per_step, 0.01)
    assert np.allclose(lons_test, ra.lons_per_step, 0.01)
    assert np.allclose(ra.current_course, current_course_test, 0.1)
    assert np.array_equal(ra.dist_per_step, dist_test)
    assert np.array_equal(ra.course_per_step, az_test)

    assert np.array_equal(ra.full_dist_traveled, np.array([0, 0, 0, 0]))


'''
    test whether IsoBased.update_position() updates current_azimuth, lats/lons_per_step, dist_per_step correctly
        - no land crossing
'''


def test_update_position_success():
    lat_start = 53.55
    lon_start = 5.45
    # lat_end = 53.45
    # lon_end = 3.72
    dist_travel = 1007.091 * 1000
    az = 355.113
    # az_till_start = 330.558

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.course_per_step = np.array([[0, 0, 0, 0]]) * u.degree
    ra.dist_per_step = np.array([[0, 0, 0, 0]]) * u.meter
    ra.current_course = np.array([az, az, az, az]) * u.degree

    dist = np.array([dist_travel, dist_travel, dist_travel, dist_travel]) * u.meter

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
    assert np.array_equal(no_constraints, np.array([1, 1, 1, 1]))


##
# test wheather IsoBased::checkbearing correcly sets route_reached_destination to True and whether the returned
# variables are correct
def test_check_bearing_true():

    ra = basic_test_func.create_dummy_IsoBased_object()

    lat_start = 54.87
    lon_start = 13.33
    lat_end = 54.9
    lon_end = 13.46

    az = 68.087
    az_test = np.array([az, az, az, az]) * u.degree
    lon_test = np.array([lon_end, lon_end, lon_end, lon_end])
    lat_test = np.array([lat_end, lat_end, lat_end, lat_end])
    dist = np.array([10000000, 10000000, 10000000, 10000]) * u.meter

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.course_per_step = np.array([[0, 0, 0, 0]]) * u.degree
    ra.dist_per_step = np.array([[0, 0, 0, 0]]) * u.meter
    ra.current_course = np.array([az, az, az, az]) * u.degree
    ra.finish = (lat_end, lon_end)
    ra.finish_temp = ra.finish

    move = ra.check_bearing(dist)
    move['azi2'] = move['azi2'] * u.degree

    assert ra.route_reached_destination is True
    assert np.allclose(move['azi2'], az_test, 0.1)
    assert np.array_equal(move['lon2'], lon_test)
    assert np.array_equal(move['lat2'], lat_test)


##
# test wheather IsoBased::checkbearing correcly leaves route_reached_destination untouched if travelled distance is
# small enough
def test_check_bearing_false():
    lat_start = 54.87
    lon_start = 13.33
    # lat_end = 54.9
    # lon_end = 13.46

    az = 355.113
    # az_till_start = 330.558

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.course_per_step = np.array([[0, 0, 0, 0]]) * u.degree
    ra.dist_per_step = np.array([[0, 0, 0, 0]]) * u.meter
    ra.current_course = np.array([az, az, az, az]) * u.degree

    dist = np.array([10000, 10000, 10000, 10000]) * u.meter

    ra.check_bearing(dist)

    assert ra.route_reached_destination is False


##
# test whether IsoFuel::get_delta_variables_netCDF_last_step returns correct dist and delta_time. delta_fuel can't be
# tested
# with sensible amount of work
def test_get_delta_variables_last_step():
    lat_start = 54.87
    lon_start = 13.33
    lat_end = 54.9
    lon_end = 13.46
    boat_speed = 20 * u.meter/u.second

    az = 68.087
    dist_test = np.array([8987, 8987, 8987, 8987]) * u.meter
    time_test = dist_test / boat_speed

    ##
    # initialise routing alg
    ra = basic_test_func.create_dummy_IsoFuel_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.course_per_step = np.array([[0, 0, 0, 0]]) * u.degree
    ra.dist_per_step = np.array([[0, 0, 0, 0]]) * u.meter
    ra.time = np.array(
        [datetime.now(), datetime.now(), datetime.now(), datetime.now()])
    ra.current_course = np.array([az, az, az, az]) * u.degree
    ra.finish = (lat_end, lon_end)
    ra.finish_temp = (lat_end, lon_end)

    ##
    # initialise boat
    tk = basic_test_func.create_dummy_Tanker_object()
    tk.speed = boat_speed
    tk.use_depth_data = False

    ship_params = tk.get_ship_parameters(ra.get_current_course(), ra.get_current_lats(), ra.get_current_lons(),
                                         ra.time, [])
    ship_params.print()

    delta_time, delta_fuel, dist = ra.get_delta_variables_netCDF_last_step(ship_params, tk.get_boat_speed())

    assert np.allclose(dist, dist_test, 0.1)
    assert np.allclose(delta_time, time_test, 0.1)


'''
    Test whether shapes of arrays are sensible after define_courses()
'''


def test_define_courses_array_shapes():
    nof_hdgs_segments = 4
    hdgs_increments = 1

    ra = basic_test_func.create_dummy_IsoBased_object()
    # current_var = ra.get_current_azimuth()
    ra.set_course_segments(nof_hdgs_segments, hdgs_increments)

    ra.define_courses()
    ra.print_shape()
    ra.print_current_status()
    # checking 2D arrays
    assert ra.lats_per_step.shape[1] == nof_hdgs_segments + 1
    assert ra.dist_per_step.shape == ra.lats_per_step.shape
    assert ra.course_per_step.shape == ra.lats_per_step.shape
    assert ra.shipparams_per_step.speed.shape == ra.lats_per_step.shape
    assert ra.shipparams_per_step.fuel_rate.shape == ra.lats_per_step.shape
    assert ra.absolutefuel_per_step.shape == ra.lats_per_step.shape

    # checking 1D arrays
    assert ra.full_time_traveled.shape[0] == nof_hdgs_segments + 1
    assert ra.full_dist_traveled.shape == ra.full_time_traveled.shape
    assert ra.time.shape == ra.full_time_traveled.shape


'''
    test whether current_course is correctly filled in define_courses()
'''


def test_define_courses_current_course_filling():
    start = (30, 45)
    finish = (0, 20)
    ra = basic_test_func.create_dummy_IsoBased_object()

    new_course = geod.inverse([start[0]], [start[1]], [finish[0]], [finish[1]])
    new_course['azi1'] = new_course['azi1']

    ra.define_courses()
    ra.print_shape()
    ra.print_current_status()

    # checking current_course
    assert ra.current_course.shape[0] == ra.lats_per_step.shape[1]

    test_current_course = np.array(
        [new_course['azi1'] + 2, new_course['azi1'] + 1, new_course['azi1'],
         new_course['azi1'] - 1, new_course['azi1'] - 2]) * u.degree

    for i in range(0, test_current_course.shape[0]):
        print('ra.current_course: ', test_current_course[i])
        assert test_current_course[i] == ra.current_course[i]


'''
    test whether indices survive the pruning which maximise the total distance traveled
'''


def test_pruning_select_correct_idxs():
    nof_hdgs_segments = 8
    hdgs_increments = 1

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.set_course_segments(nof_hdgs_segments, hdgs_increments)
    ra.define_courses()

    pruning_bins = np.array([10, 20, 40, 60, 80]) * u.degree
    ra.current_course = np.array([15, 16, 22, 23, 44, 45, 71, 72, 74]) * u.degree
    ra.full_time_traveled = np.random.rand(9)
    fuel_rate = np.random.rand(1, 9)
    speed_per_step = np.random.rand(1, 9)
    ra.full_dist_traveled = np.array([1, 5, 6, 1, 2, 7, 10, 1, 8])

    ra.dist_per_step = np.array([ra.full_dist_traveled])
    ra.course_per_step = np.array([ra.current_course]) * u.degree

    sp = ShipParams(
        fuel_rate=fuel_rate * u.kg/u.second,
        power=np.full(fuel_rate.shape, 0) * u.Watt,
        rpm=np.full(fuel_rate.shape, 0) * u.Hz,
        speed=speed_per_step * u.meter/u.second,
        r_calm=np.full(fuel_rate.shape, 0) * u.newton,
        r_wind=np.full(fuel_rate.shape, 0) * u.newton,
        r_waves=np.full(fuel_rate.shape, 0) * u.newton,
        r_shallow=np.full(fuel_rate.shape, 0) * u.newton,
        r_roughness=np.full(fuel_rate.shape, 0) * u.newton,
        status=np.full(fuel_rate.shape, 0)
    )
    ra.shipparams_per_step = sp

    cur_course_test = np.array([16, 22, 45, 71]) * u.degree
    full_dist_test = np.array([5, 6, 7, 10])
    full_time_test = np.array(
        [ra.full_time_traveled[1], ra.full_time_traveled[2], ra.full_time_traveled[5], ra.full_time_traveled[6]])

    full_fuel_test = np.array(
        [fuel_rate[0, 1], fuel_rate[0, 2], fuel_rate[0, 5], fuel_rate[0, 6]]) * u.kg/u.second
    speed_ps_test = np.array([speed_per_step[0, 1], speed_per_step[0, 2],
                              speed_per_step[0, 5], speed_per_step[0, 6]]) * u.meter/u.second
    lat_test = np.array([[30, 30, 30, 30]])
    lon_test = np.array([[45, 45, 45, 45]])
    time_single = datetime(2023, 11, 11, 11, 11)
    time_test = np.array([time_single, time_single, time_single, time_single])

    ra.print_current_status()
    form.print_line()

    ra.prune_groups = 'courses'
    ra.pruning(True, pruning_bins)

    assert np.array_equal(cur_course_test, ra.current_course)
    assert np.array_equal(full_time_test, ra.full_time_traveled)
    assert np.array_equal(full_dist_test, ra.full_dist_traveled)

    assert np.array_equal(cur_course_test, ra.course_per_step[0])
    assert np.array_equal(full_dist_test, ra.dist_per_step[0])
    assert np.array_equal(full_fuel_test, ra.shipparams_per_step.fuel_rate[0])
    assert np.array_equal(speed_ps_test, ra.shipparams_per_step.speed[0])
    assert np.array_equal(lat_test, ra.lats_per_step)
    assert np.array_equal(lon_test, ra.lons_per_step)
    assert np.array_equal(time_test, ra.time)

    # form.print_line()  # ra.print_ra()


'''
    test shape and content of 'move' for known distance, start and end points
'''


def test_check_bearing():
    lat_start = 51.289444
    lon_start = 6.766667
    lat_end = 60.293333
    lon_end = 5.218056
    dist_travel = 1007.091 * 1000
    az = 355.113

    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[lat_start, lat_start, lat_start, lat_start]])
    ra.lons_per_step = np.array([[lon_start, lon_start, lon_start, lon_start]])
    ra.current_course = np.array([az, az, az, az]) * u.degree

    dist = np.array([dist_travel, dist_travel, dist_travel, dist_travel]) * u.meter

    lats_test = np.array([[lat_end, lat_end, lat_end, lat_end], [lat_start, lat_start, lat_start, lat_start]])
    lons_test = np.array([[lon_end, lon_end, lon_end, lon_end], [lon_start, lon_start, lon_start, lon_start]])

    ra.print_current_status()
    move = ra.check_bearing(dist)

    # print('lats_test[0]', lats_test[0])
    # print('lons_test[0]', lons_test[0])
    assert np.allclose(lats_test[0], move['lat2'], 0.01)
    assert np.allclose(lons_test[0], move['lon2'], 0.01)


'''
    For a test case with two branches each with 2 route segments, routes of both branches are reaching the
    destination (dist > dist_to_dest). Test whether the routes reaching the destination are found for every branch.
'''


def test_find_every_route_reaching_destination_testtwobranches():
    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[37.67, 37.67, 37.66, 37.66], [37.42, 37.42, 37.43, 37.43]])
    ra.lons_per_step = np.array([[-123.76, -123.50, -123.32, -123.09], [-123.61, -123.61, -123.23, -123.23]])
    ra.finish = (37.53, -123.24)
    ra.current_last_step_dist = np.array([2, 2, 2, 0]) * u.meter
    ra.current_last_step_dist_to_dest = np.array([1, 1, 1, 1]) * u.meter
    ra.shipparams_per_step = ShipParams.set_default_array()
    ra.absolutefuel_per_step = np.array([[1, 0, 1, 1], [1, 1, 1, 1]]) * u.kg
    ra.print_init()

    ra.find_every_route_reaching_destination()
    assert ra.current_step_routes['st_index'][0] == 1
    assert ra.current_step_routes['st_index'][1] == 2
    assert ra.current_step_routes.shape[0] == 2
    assert ra.next_step_routes.shape[0] == 0


'''
    For a test case with two branches each with 2 route segments, routes of only one branch are reaching the
    destination (dist > dist_to_dest). Test whether the routes reaching the destination are found and written to
    current_step_routes and the routes from the branch of which none reaches the destination are passed to
    next_step_routes.
'''


def test_find_every_route_reaching_destination_testonebranch():
    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[37.67, 37.67, 37.66, 37.66], [37.42, 37.42, 37.43, 37.43]])
    ra.lons_per_step = np.array([[-123.76, -123.50, -123.32, -123.09], [-123.61, -123.61, -123.23, -123.23]])
    ra.finish = (37.53, -123.24)
    ra.current_last_step_dist = np.array([2, 2, 0, 0]) * u.meter
    ra.current_last_step_dist_to_dest = np.array([1, 1, 1, 1]) * u.meter
    ra.shipparams_per_step = ShipParams.set_default_array()
    ra.absolutefuel_per_step = np.array([[1, 0, 1, 1], [1, 1, 1, 1]]) * u.kg
    ra.print_init()

    ra.find_every_route_reaching_destination()
    assert ra.current_step_routes['st_index'][0] == 1
    assert ra.current_step_routes.shape[0] == 1
    assert ra.next_step_routes.shape[0] == 2
    assert ra.next_step_routes['st_index'][0] == 2
    assert ra.next_step_routes['st_index'][1] == 3


'''
    For a test case with two branches each with 2 route segments, routes of both branches are reaching the
    destination (dist > dist_to_dest). As the routes of the first branch split only in the last routing step and
    have been propagated to the destination, they constitute duplicates. Test whether only one of the duplicates is
    selected for the final route_list. Test also, whether the remaining routes are sorted correctly according to fuel.
'''


def test_find_routes_testduplicates():
    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[37.67, 37.67, 37.66, 37.66], [37.42, 37.42, 37.43, 37.43]])
    ra.lons_per_step = np.array([[-123.76, -123.76, -123.32, -123.09], [-123.61, -123.61, -123.23, -123.23]])

    ra.finish = (37.53, -123.24)
    ra.current_last_step_dist = np.array([2, 2, 2, 0]) * u.meter
    ra.current_last_step_dist_to_dest = np.array([1, 1, 1, 1]) * u.meter
    ra.shipparams_per_step = ShipParams.set_default_array()
    ra.shipparams_per_step.fuel_rate = np.array([[2, 2, 1, 1], [1, 1, 1, 1]]) * u.kg/u.second
    ra.absolutefuel_per_step = np.array([[2, 2, 1, 1], [1, 1, 1, 1]]) * u.kg

    # definitions necessary only for completness
    ra.shipparams_per_step.speed = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.meter/u.second
    ra.shipparams_per_step.power = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.Watt
    ra.shipparams_per_step.rpm = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.Hz
    ra.shipparams_per_step.r_wind = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.newton
    ra.shipparams_per_step.r_calm = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.newton
    ra.shipparams_per_step.r_waves = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.newton
    ra.shipparams_per_step.r_shallow = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.newton
    ra.shipparams_per_step.r_roughness = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.newton
    ra.shipparams_per_step.status = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    ra.course_per_step = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.degree
    ra.dist_per_step = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]) * u.meter

    ra.starttime_per_step = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    ra.time = np.array([0, 0, 0, 0])
    ra.path_to_route_folder = None
    ra.figure_path = None

    ra.print_init()

    ra.find_every_route_reaching_destination()
    ra.find_routes_reaching_destination_in_current_step(2)
    assert ra.current_step_routes['st_index'][0] == 0
    assert ra.current_step_routes['st_index'][1] == 2
    assert ra.current_step_routes.shape[0] == 2
    assert ra.next_step_routes.shape[0] == 0
    assert ra.route_list[0].lons_per_step[1] == -123.32
    assert ra.route_list[1].lons_per_step[1] == -123.76


def test_branch_based_pruning():
    ra = basic_test_func.create_dummy_IsoBased_object()
    ra.lats_per_step = np.array([[37.68, 37.67, 37.66, 37.65, 37.64, 37.63],
                                 [37.42, 37.42, 37.42, 37.43, 37.43, 37.43]])
    ra.lons_per_step = np.array([[-123.10, -123.76, -123.32, -123.09, -123.07, -123.06],
                                 [-123.61, -123.61, -123.61, -123.23, -123.23, -123.23]])
    ra.full_dist_traveled = np.array([1, 2, 3, 2, 3, 1])

    ra.prune_groups = 'branch'

    idxs = ra.branch_based_pruning()
    idxs_test = [2, 4]

    assert np.array_equal(np.array(idxs), np.array(idxs_test))
