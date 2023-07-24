import datetime
import math
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import WeatherRoutingTool.config as config
from WeatherRoutingTool.ship.ship import Tanker


# def test_inc():
#    pol = Tanker(2)
#    assert pol.inc(3) == 5

def get_default_Tanker():
    DEFAULT_GFS_FILE = os.environ[
                           'BASE_PATH'] + '/reduced_testdata_weather.nc'  # CMEMS needs lat: 30 to 45, lon: 0 to 20
    COURSES_FILE = os.environ['BASE_PATH'] + '/CoursesRoute.nc'
    DEPTH_FILE = os.environ['DEPTH_DATA']

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
    test whether power is correctly extracted from courses netCDF
'''


def test_get_fuel_from_netCDF():
    lat = np.array([1.1, 2.2, 3.3, 4.4])
    it = np.array([1, 2])
    power = np.array([[1, 4], [3.4, 5.3], [2.1, 6], [1., 5.1]])
    rpm = np.array([[10, 14], [11, 15], [20, 60], [15, 5]])
    fcr = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])

    data_vars = dict(Power_delivered=(["lat", "it"], power), RotationRate=(["lat", "it"], rpm),
                     Fuel_consumption_rate=(["lat", "it"], fcr), )

    coords = dict(lat=(["lat"], lat), it=(["it"], it), )
    attrs = dict(description="Necessary descriptions added here.")

    ds = xr.Dataset(data_vars, coords, attrs)
    print(ds)

    pol = get_default_Tanker()
    ship_params = pol.extract_params_from_netCDF(ds)
    power_test = ship_params.get_power()
    rpm_test = ship_params.get_rpm()
    fuel_test = ship_params.get_fuel()

    power_ref = np.array([1, 4, 3.4, 5.3, 2.1, 6, 1., 5.1])
    rpm_ref = np.array([10, 14, 11, 15, 20, 60, 15, 5])
    fuel_ref = np.array([2, 3, 4, 5, 6, 7, 8, 9])

    fuel_test = fuel_test * 3.6

    assert np.array_equal(power_test, power_ref)
    assert np.array_equal(rpm_test, rpm_ref)
    assert np.array_equal(fuel_test, fuel_ref)

    ds.close()


def test_power_consumption_returned():
    # dummy weather file
    lat = np.array([54., 55, 56])
    lon = np.array([14., 15, 16])
    # time_single = np.datetime64('2023-01-23')
    time_single = datetime.datetime.strptime('2023-01-23', '%Y-%m-%d')
    time = np.array([time_single])

    '''uwind = np.array([[
        [40, 0, -40],
        [40, 0, -40],
        [40, 0, -40],
    ]])
    vwind = np.array([[
        [0,40, 0],
        [0,40, 0],
        [0,40, 0],
    ]])'''

    uwind = np.array([[[40, -40, 0], [40, -40, 0], [40, -40, 0], ]])
    vwind = np.array([[[0, 0, 40], [0, 0, 40], [0, 0, 40], ]])

    # courses test file
    courses_test = np.array([0, 180, 0, 180, 180, 0])
    # lat_test = np.array([54, 54, 55, 55, 56, 56])
    # lon_test = np.array([14, 14, 15, 15, 16, 16])
    lat_test = np.array([55, 55, 56, 56, 54, 54])
    lon_test = np.array([15, 15, 16, 16, 14, 14])

    vo = np.full(shape=(time.shape[0], lat.shape[0], lon.shape[0]), fill_value=0)

    data_vars = dict(vo=(["time", "latitude", "longitude"], vo), uo=(["time", "latitude", "longitude"], vo),
                     VHM0=(["time", "latitude", "longitude"], vo), VTPK=(["time", "latitude", "longitude"], vo),
                     VMDR=(["time", "latitude", "longitude"], vo), thetao=(["time", "latitude", "longitude"], vo),
                     so=(["time", "latitude", "longitude"], vo),
                     Temperature_surface=(["time", "latitude", "longitude"], vo),
                     Pressure_surface=(["time", "latitude", "longitude"], vo))

    coords = dict(time=(["time"], time), latitude=(["latitude"], lat), longitude=(["longitude"], lon), )

    attrs = dict(description="Necessary descriptions added here.")

    ds = xr.Dataset(data_vars, coords, attrs)
    ds['u-component_of_wind_height_above_ground'] = (['time', 'latitude', 'longitude'], uwind)
    ds['v-component_of_wind_height_above_ground'] = (['time', 'latitude', 'longitude'], vwind)

    print(ds)
    ds.to_netcdf(config.BASE_PATH + '/TestEnvData.nc')

    # dummy course netCDF
    pol = get_default_Tanker()
    pol.set_boat_speed(np.array([20]))
    pol.set_env_data_path(config.BASE_PATH + '/TestEnvData.nc')
    pol.set_courses_path(config.BASE_PATH + '/TestCoursesRoute.nc')

    time_test = np.array([time_single, time_single, time_single, time_single, time_single, time_single])
    pol.write_netCDF_courses(courses_test, lat_test, lon_test, time_test)
    ds = pol.get_fuel_netCDF()

    power = ds['Power_delivered']
    rpm = ds['RotationRate']
    fuel = ds['Fuel_consumption_rate']

    assert np.all(power < 3000000)
    assert np.all(rpm < 100)
    assert np.all(fuel < 0.8)
    assert np.all(power > 1000000)
    assert np.all(rpm > 70)
    assert np.all(fuel > 0.5)
