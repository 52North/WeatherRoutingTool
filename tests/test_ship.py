import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import WeatherRoutingTool.config as config
import WeatherRoutingTool.utils.unit_conversion as utils

from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.ship.shipparams import ShipParams


# def test_inc():
#    pol = Tanker(2)
#    assert pol.inc(3) == 5

def get_default_Tanker():
    DEFAULT_GFS_FILE = config.BASE_PATH + '/tests/data/reduced_testdata_weather.nc'  # CMEMS needs lat: 30 to 45,
    # lon: 0 to 20
    COURSES_FILE = config.BASE_PATH + '/CoursesRoute.nc'
    DEPTH_FILE = config.DEPTH_DATA

    pol = Tanker(2)
    pol.init_hydro_model_Route(DEFAULT_GFS_FILE, COURSES_FILE, DEPTH_FILE)
    return pol


def compare_times(time64, time):
    time64 = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    time = (time - datetime.datetime(1970, 1, 1, 0, 0))
    for iTime in range(0, time.shape[0]):
        time[iTime] = time[iTime].total_seconds()
    assert np.array_equal(time64, time)


'''
    test whether lat, lon, time and courses are correctly written to course netCDF (elements and shape read from netCDF
     match properties of original array)
'''
'''

def test_get_netCDF_courses():
    lat = np.array([1., 1., 1, 2, 2, 2])
    lon = np.array([4., 4., 4, 3, 3, 3])
    courses = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # speed = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    pol = get_default_Tanker()
    time = np.array([datetime.datetime(2022, 12, 19), datetime.datetime(2022, 12, 19), datetime.datetime(2022, 12, 19),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360)])

    pol.write_netCDF_courses(courses, lat, lon, time)
    ds = xr.open_dataset(pol.courses_path)

    lat_read = ds['lat'].to_numpy()
    lon_read = ds['lon'].to_numpy()
    courses_read = ds['courses'].to_numpy()
    time_read = ds['time'].to_numpy()

    lat_ind = np.unique(lat, return_index=True)[1]
    lon_ind = np.unique(lon, return_index=True)[1]
    time_ind = np.unique(time, return_index=True)[1]
    lat = [lat[index] for index in sorted(lat_ind)]
    lon = [lon[index] for index in sorted(lon_ind)]
    time = [time[index] for index in sorted(time_ind)]
    time = np.array(time)

    assert np.array_equal(lat, lat_read)
    assert np.array_equal(lon, lon_read)
    compare_times(time_read, time)

    assert courses.shape[0] == courses_read.shape[0] * courses_read.shape[1]
    for ilat in range(0, courses_read.shape[0]):
        for iit in range(0, courses_read.shape[1]):
            iprev = ilat * courses_read.shape[1] + iit
            assert courses[iprev] == courses_read[ilat][iit]

    ds.close()
'''

'''
    test whether power is correctly extracted from courses netCDF
'''


def test_get_fuel_from_netCDF():
    lat = np.array([1.1, 2.2, 3.3, 4.4])
    it = np.array([1, 2])
    power = np.array([[1, 4], [3.4, 5.3], [2.1, 6], [1., 5.1]])
    rpm = np.array([[10, 14], [11, 15], [20, 60], [15, 5]])
    fcr = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
    rcalm = np.array([[2.2, 2.3], [2.4, 2.5], [2.6, 2.7], [2.8, 2.9]])
    rwind = np.array([[3.2, 3.3], [3.4, 3.5], [3.6, 3.7], [3.8, 3.9]])
    rshallow = np.array([[4.2, 4.3], [4.4, 4.5], [4.6, 4.7], [4.8, 4.9]])
    rwaves = np.array([[5.2, 5.3], [5.4, 5.5], [5.6, 5.7], [5.8, 5.9]])
    rroughness = np.array([[6.2, 6.3], [6.4, 6.5], [6.6, 6.7], [6.8, 6.9]])

    data_vars = dict(Power_brake=(["lat", "it"], power), RotationRate=(["lat", "it"], rpm),
                     Fuel_consumption_rate=(["lat", "it"], fcr), Calm_resistance=(["lat", "it"], rcalm),
                     Wind_resistance=(["lat", "it"], rwind), Wave_resistance=(["lat", "it"], rwaves),
                     Shallow_water_resistance=(["lat", "it"], rshallow),
                     Hull_roughness_resistance=(["lat", "it"], rroughness))

    coords = dict(lat=(["lat"], lat), it=(["it"], it), )
    attrs = dict(description="Necessary descriptions added here.")

    ds = xr.Dataset(data_vars, coords, attrs)
    print(ds)

    pol = get_default_Tanker()
    ship_params = pol.extract_params_from_netCDF(ds)
    power_test = ship_params.get_power()
    rpm_test = ship_params.get_rpm()
    fuel_test = ship_params.get_fuel()
    rcalm_test = ship_params.get_rcalm()
    rwind_test = ship_params.get_rwind()
    rshallow_test = ship_params.get_rshallow()
    rwaves_test = ship_params.get_rwaves()
    rroughness_test = ship_params.get_rroughness()

    power_ref = np.array([1, 4, 3.4, 5.3, 2.1, 6, 1., 5.1])
    rpm_ref = np.array([10, 14, 11, 15, 20, 60, 15, 5])
    fuel_ref = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    rcalm_ref = np.array([2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])
    rwind_ref = np.array([3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9])
    rshallow_ref = np.array([4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9])
    rwaves_ref = np.array([5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
    rroughness_ref = np.array([6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9])

    fuel_test = fuel_test * 3.6

    assert np.array_equal(power_test, power_ref)
    assert np.array_equal(rpm_test, rpm_ref)
    assert np.array_equal(fuel_test, fuel_ref)
    assert np.array_equal(rcalm_test, rcalm_ref)
    assert np.array_equal(rwind_test, rwind_ref)
    assert np.array_equal(rshallow_test, rshallow_ref)
    assert np.array_equal(rwaves_test, rwaves_ref)
    assert np.array_equal(rroughness_test, rroughness_ref)

    ds.close()


'''
    check return values by maripower: has there been renaming? Do the return values have a sensible order of magnitude?
'''


def test_power_consumption_returned():
    # dummy weather file
    time_single = datetime.datetime.strptime('2023-07-20', '%Y-%m-%d')

    # courses test file
    courses_test = np.array([0, 180, 0, 180, 180, 0])
    lat_test = np.array([54.3, 54.3, 54.6, 54.6, 54.9, 54.9])
    lon_test = np.array([13.3, 13.3, 13.6, 13.6, 13.9, 13.9])

    # dummy course netCDF
    pol = get_default_Tanker()
    pol.set_boat_speed(np.array([8]))
    pol.set_env_data_path(config.WEATHER_DATA)
    pol.set_courses_path(config.BASE_PATH + '/TestCoursesRoute.nc')

    time_test = np.array([time_single, time_single, time_single, time_single, time_single, time_single])
    pol.write_netCDF_courses(courses_test, lat_test, lon_test, time_test)
    ds = pol.get_fuel_netCDF()

    power = ds['Power_brake']
    rpm = ds['RotationRate']
    fuel = ds['Fuel_consumption_rate']

    assert np.all(power < 10000000)
    assert np.all(rpm < 100)

    assert np.all(fuel < 1.5)
    assert np.all(power > 1000000)
    assert np.all(rpm > 70)
    assert np.all(fuel > 0.5)


'''
    test whether single elements of fuel, power, rpm and speed are correctly returned by ShipParams.get_element(idx)
'''


def test_shipparams_get_element():
    fuel = np.array([1, 2, 3, 4])
    speed = np.array([0.1, 0.2, 0.3, 0.4])
    power = np.array([11, 21, 31, 41])
    rpm = np.array([21, 22, 23, 24])
    rwind = np.array([1.1, 1.2, 1.3, 1.4])
    rcalm = np.array([2.1, 2.2, 2.3, 2.4])
    rwaves = np.array([3.1, 3.2, 3.3, 3.4])
    rshallow = np.array([4.1, 4.2, 4.3, 4.4])
    rroughness = np.array([5.1, 5.2, 5.3, 5.4])

    sp = ShipParams(fuel=fuel, power=power, rpm=rpm, speed=speed, r_wind=rwind, r_calm=rcalm, r_waves=rwaves,
                    r_shallow=rshallow, r_roughness=rroughness)
    idx = 2

    fuel_test, power_test, rpm_test, speed_test, rwind_test, rcalm_test, rwaves_test, rshallow_test, rroughness_test \
        = sp.get_element(idx)

    assert fuel[idx] == fuel_test
    assert speed[idx] == speed_test
    assert power[idx] == power_test
    assert rpm[idx] == rpm_test
    assert rwind[idx] == rwind_test
    assert rcalm[idx] == rcalm_test
    assert rwaves[idx] == rwaves_test
    assert rshallow[idx] == rshallow_test
    assert rroughness[idx] == rroughness_test


'''
    test whether ShipParams object for single waypoint is correctly returned by ShipParams.get_single_object(idx)
'''


def test_shipparams_get_single():
    fuel = np.array([1, 2, 3, 4])
    speed = np.array([0.1, 0.2, 0.3, 0.4])
    power = np.array([11, 21, 31, 41])
    rpm = np.array([21, 22, 23, 24])
    rwind = np.array([1.1, 1.2, 1.3, 1.4])
    rcalm = np.array([2.1, 2.2, 2.3, 2.4])
    rwaves = np.array([3.1, 3.2, 3.3, 3.4])
    rshallow = np.array([4.1, 4.2, 4.3, 4.4])
    rroughness = np.array([5.1, 5.2, 5.3, 5.4])

    sp = ShipParams(fuel=fuel, power=power, rpm=rpm, speed=speed, r_wind=rwind, r_calm=rcalm, r_waves=rwaves,
                    r_shallow=rshallow, r_roughness=rroughness)
    idx = 2

    sp_test = sp.get_single_object(idx)

    assert sp_test.fuel == fuel[idx]
    assert sp_test.power == power[idx]
    assert sp_test.rpm == rpm[idx]
    assert sp_test.speed == speed[idx]
    assert sp_test.r_calm == rcalm[idx]
    assert sp_test.r_wind == rwind[idx]
    assert sp_test.r_waves == rwaves[idx]
    assert sp_test.r_shallow == rshallow[idx]
    assert sp_test.r_roughness == rroughness[idx]


'''
    test whether lat, lon, time and courses are correctly written to course netCDF (elements and shape read from netCDF
     match properties of original array)
'''


def test_get_netCDF_courses_isobased():
    lat = np.array([1., 1., 1, 2, 2, 2])
    lon = np.array([4., 4., 4, 3, 3, 3])
    courses = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # speed = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    pol = get_default_Tanker()
    time = np.array([datetime.datetime(2022, 12, 19), datetime.datetime(2022, 12, 19), datetime.datetime(2022, 12, 19),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360)])

    pol.write_netCDF_courses(courses, lat, lon, time, True)
    ds = xr.open_dataset(pol.courses_path)

    lat_read = ds['lat'].to_numpy()
    lon_read = ds['lon'].to_numpy()
    courses_read = ds['courses'].to_numpy()
    time_read = ds['time'].to_numpy()

    lat_ind = np.unique(lat, return_index=True)[1]
    lon_ind = np.unique(lon, return_index=True)[1]
    time_ind = np.unique(time, return_index=True)[1]
    lat = [lat[index] for index in sorted(lat_ind)]
    lon = [lon[index] for index in sorted(lon_ind)]
    time = [time[index] for index in sorted(time_ind)]
    time = np.array(time)

    assert np.array_equal(lat, lat_read)
    assert np.array_equal(lon, lon_read)
    compare_times(time_read, time)

    assert courses.shape[0] == courses_read.shape[0] * courses_read.shape[1]
    for ilat in range(0, courses_read.shape[0]):
        for iit in range(0, courses_read.shape[1]):
            iprev = ilat * courses_read.shape[1] + iit
            assert courses[iprev] == np.rad2deg(courses_read[ilat][iit])

    ds.close()


'''
    test whether lat, lon, time and courses are correctly written to course netCDF (elements and shape read from netCDF
     match properties of original array) for the genetic algorithm
'''


def test_get_netCDF_courses_GA():
    lat_short = np.array([1, 2, 1])
    lon_short = np.array([4, 4, 1.5])
    courses = np.array([0.1, 0.2, 0.3])

    pol = get_default_Tanker()
    time = np.array([datetime.datetime(2022, 12, 19), datetime.datetime(2022, 12, 19) + datetime.timedelta(days=180),
                     datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360)])

    pol.write_netCDF_courses(courses, lat_short, lon_short, time)
    ds = xr.open_dataset(pol.courses_path)

    lat_read = ds['lat'].to_numpy()
    lon_read = ds['lon'].to_numpy()
    courses_read = ds['courses'].to_numpy()
    time_read = ds['time'].to_numpy()

    assert np.array_equal(lat_short, lat_read)
    assert np.array_equal(lon_short, lon_read)
    compare_times(time_read, time)

    assert courses.shape[0] == courses_read.shape[0] * courses_read.shape[1]
    for ilat in range(0, courses_read.shape[0]):
        for iit in range(0, courses_read.shape[1]):
            iprev = ilat * courses_read.shape[1] + iit
            assert np.radians(courses[iprev]) == courses_read[ilat][iit]

    ds.close()


'''
    test whether lat, lon, time and courses are correctly written to course netCDF & wheather start_times_per_step
    and dist_per_step
    are correctly calculated
'''


def test_get_fuel_for_fixed_waypoints():
    bs = 6
    start_time = datetime.datetime.strptime("2023-07-20T10:00Z", '%Y-%m-%dT%H:%MZ')
    route_lats = np.array([54.9, 54.7, 54.5, 54.2])
    route_lons = np.array([13.2, 13.4, 13.7, 13.9])

    pol = get_default_Tanker()
    pol.set_boat_speed(bs)

    waypoint_dict = RouteParams.get_per_waypoint_coords(route_lons, route_lats, start_time, bs)

    ship_params = pol.get_fuel_per_time_netCDF(waypoint_dict['courses'], waypoint_dict['start_lats'],
                                               waypoint_dict['start_lons'], waypoint_dict['start_times'])
    ship_params.print()

    ds = xr.open_dataset(pol.courses_path)
    test_lat_start = ds.lat
    test_lon_start = ds.lon
    test_courses = np.rad2deg(ds.courses.to_numpy()[:, 0])
    test_time = ds.time.to_numpy()

    test_time_dt = np.full(3, datetime.datetime(1970, 1, 1, 0, 0))
    for t in range(0, 3):
        test_time_dt[t] = utils.convert_npdt64_to_datetime(test_time[t])

    ref_lat_start = np.array([54.9, 54.7, 54.5])
    ref_lon_start = np.array([13.2, 13.4, 13.7])
    ref_courses = np.array([149.958, 138.89, 158.685])
    ref_dist = np.array([25712., 29522., 35836.])
    ref_time = np.array([start_time, start_time + datetime.timedelta(seconds=ref_dist[0] / bs),
                         start_time + datetime.timedelta(seconds=ref_dist[0] / bs) + datetime.timedelta(
                             seconds=ref_dist[1] / bs)])

    assert test_lon_start.any() == ref_lon_start.any()
    assert test_lat_start.any() == ref_lat_start.any()
    assert np.allclose(test_courses, ref_courses, 0.1)
    assert utils.compare_times(test_time_dt, ref_time) is True
    assert np.allclose(waypoint_dict['dist'], ref_dist, 0.1)


'''
    test whether power and wind resistance that are returned by maripower lie on an ellipse
'''


def test_wind_force():
    lats = np.full(10, 54.9)  # 37
    lons = np.full(10, 13.2)
    courses = np.linspace(0, 360, 10)
    courses = utils.degree_to_pmpi(courses)

    time = np.full(10, datetime.datetime.strptime("2023-07-20T10:00Z", '%Y-%m-%dT%H:%MZ'))
    bs = 6

    pol = get_default_Tanker()
    pol.set_boat_speed(bs)
    # pol.write_netCDF_courses(courses, lats, lons, time)
    # ds = xr.open_dataset(pol.courses_path)
    ship_params = pol.get_fuel_per_time_netCDF(courses, lats, lons, time, True)
    power = ship_params.get_power()
    rwind = ship_params.get_rwind()

    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
    for i in range(0, courses.shape[0]):
        courses[i] = math.radians(courses[i])
    # wind_dir = math.radians(wind_dir)

    axes[0].plot(courses, power)
    axes[0].legend()
    for ax in axes.flatten():
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.set_theta_zero_location("S")
        ax.grid(True)
    axes[1].plot(courses, rwind)
    axes[0].set_title("Power", va='bottom')
    axes[1].set_title("Wind resistence", va='top')
