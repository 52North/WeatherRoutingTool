"""Weather functions."""
import numpy as np
import datetime as dt
import netCDF4
from numpy import arange
import xarray
import pygrib as pg# Provides a high-level interface to  ECCODES C library for reading GRIB files

from scipy.interpolate import RegularGridInterpolator
"""GRIB is a file format for the storage and transport of gridded meteorological data,"""
from utils import round_time


def grib_to_wind_function(filepath):
    """Vectorized wind functions from grib file.GRIB is a file format for the storage and transport of gridded meteorological data,
    such as Numerical Weather Prediction model output."""
    #grbs = pg.open(filepath)

    #u, _, _ = grbs[1].data() # U for Initial
    #v, _, _ = grbs[2].data() # V for Final
    #filepath = 'wind-router-master/386fa86c-b801-11ec-bf54-0242ac120003.nc'
    nc = netCDF4.Dataset(filepath, mode='r')
    masku = nc.variables['10u'][:]
    maskv = nc.variables['10v'][:]
    u_final = masku[0]
    v_final = maskv[0]

    a=u_final.tolist()
    a1=v_final.tolist()

    c=np.array(a)
    c1=np.array(a1)
    d=c[0]
    d1=c1[0]

    u=d
    v=d1
    print('printing u ',u.shape)
    print('printing v shape',v.shape)
    # u=u_final
    # v=v_final

    tws = np.sqrt(u * u + v * v)
    twa = 180.0 / np.pi * np.arctan2(u, v) + 180.0#arctan:This mathematical function helps user to calculate inverse tangent for all x(being the array elements

    lats_grid = np.linspace(-90, 90, 181)#Linespace : is a tool in Python for creating numeric sequences.
    lons_grid = np.linspace(0, 360, 361)

    f_twa = RegularGridInterpolator(
        (lats_grid, lons_grid),
        np.flip(np.hstack((twa, twa[:, 0].reshape(181, 1))), axis=0), #Flip. Reverse the order of elements in an array along the given axis.
    )#hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array

    f_tws = RegularGridInterpolator(
        (lats_grid, lons_grid),
        np.flip(np.hstack((tws, tws[:, 0].reshape(181, 1))), axis=0),
    )

    return {'twa': f_twa, 'tws': f_tws}


def grib_to_wind_vectors(filepath, lat1, lon1, lat2, lon2):
    """Return u-v components for given rect for visualization."""
    #filepath='wind-router-master/386fa86c-b801-11ec-bf54-0242ac120003.nc'
    nc = netCDF4.Dataset(filepath, mode='r')
    # masku = nc.variables['10u'][:]
    # maskv = nc.variables['10v'][:]
    # u_final = masku[0]
    # v_final = maskv[0]
    print('hehe',lat1)
    print(lon1)
    print(lat2)
    print(lon2)
    #
    # a = u_final.tolist()
    # a1 = v_final.tolist()
    #
    # c = np.array(a)
    # c1 = np.array(a1)
    # d = c[0]
    # d1 = c1[0]
    #
    # u = d[0]
    # v = d1[0]
    # lats_u=nc.variables['lat'][:]
    # lons_u=nc.variables['lon'][:]
    lat = nc.variables['10u'][:]
    latitude = nc.variables['10v'][:]

    # print('latitude',latitude)
    a = lat.tolist()
    # print('printing lat0',lat[0])
    # print('a',type(a))
    c = np.array(a)
    d = c[0]
    # print("c",c)
    # print("c type",type(c))
    #print('ndim',ndim(d[0]))
    e = d[0]
    # print('e',e[45:61],'end e')

    c = 0
    v = []
    for i in e[45:61]:
        v.append(i[:41])
    #print('printing u', (np.array(v)))
    u=np.array((v))

    a1 = latitude.tolist()
    # print('printing lat0',lat[0])
    # print('a',type(a))
    c1 = np.array(a1)
    d1 = c1[0]
    # print("c",c)
    # print("c type",type(c))
    #print('ndim', ndim(d1[0]))
    e1 = d1[0]
    # print('e',e[45:61],'end e')

    # c=0
    v1 = []
    for i in e[45:61]:
        v1.append(i[:41])
    #print('printing v', (np.array(v1)))
    v=np.array(v1)

    h = 45
    k = 30
    z = []
    while (h >= k):
        x = np.array(np.repeat(h, 41), np.float32)

        z.append(x)
        # print('*',np.array(z))
        # print(x)
        h = h - 1
        # o=np.reshape(x, (1, x.size))
        # print('h',np.asarray(h))
        # con = np.vstack((arr, arr1))
    # print(z)
    #print(np.array(z))
    lats_u=np.array(z)

    a = []
    for i in range(0, 41):
        a.append(i)
    print(a)
    lst = []
    # (np.array(np.repeat(a, repeats = 1),np.float32)
    for i in range(0, 16):
        lst.append((np.array(np.repeat(a, repeats=1), np.float32)))
    #print(len(np.array(lst)))
    lons_u=np.array(lst)

    # u_final = u[0]
    # v_final = v[0]
    # u = u_final
    # v = v_final
    #
    # lon = nc.variables['lon']
    # lat = nc.variables['lat']
    # npLon = np.array(lon)
    # npLat = np.array(lat)
    # u, lats_u, lons_u = nc[0].data(lat1, lat2, lon1, lon2)
    # v, lats_v, lons_v = nc[0].data(lat1, lat2, lon1, lon2)
    # print('printing u in grib to wind vector',u)
    # print('printing u in grib to wind vector',v)
    # print('printing u in grib to wind vector',npLat)
    # print('printing u in grib to wind vector',npLon)
    # lats_u=(lat=lat , lon=lon, method='nearest')
    # lats_out = arange(start=lat1, stop=60.25, step=0.25, dtype='float32')
    # lons_out = arange(start=-18.75, stop=59.0, step=0.25, dtype='float32')

    return u, v, lats_u, lons_u


def read_wind_vectors(model, hours_ahead, lat1, lon1, lat2, lon2):
    """Return wind vectors for given number of hours.

            Parameters:
                    model (dict): available forecast wind functions
                    hours_ahead (int): number of hours looking ahead
                    lats, lons: rectange defining forecast area

            Returns:
                    wind_vectors (dict):
                        model: model timestamp
                        hour: function for given forecast hour
    """
    wind_vectors = {}
    wind_vectors['model'] = model

    for i in range(hours_ahead + 1):
        filename='wind-router-master/wind-router-master/data/2019122212/test.nc'
            #filename = 'wind-router-master/data/{}/{}f{:03d}'.format(model, model, i)
        wind_vectors[i] = grib_to_wind_vectors(filename, lat1, lon1, lat2, lon2)

    return wind_vectors


def read_wind_functions(model, hours_ahead):
    """
    Read wind functions.

            Parameters:
                    model (dict): available forecast wind functions

            Returns:
                    wind_functions (dict):
                        model: model timestamp
                        model+hour: function for given forecast hour
    """
    wind_functions = {}
    wind_functions['model'] = model

    for i in range(hours_ahead + 1):

        filename='wind-router-master/wind-router-master/data/2019122212/test.nc'
            #filename = 'wind-router-master/data/{}/{}f{:03d}'.format(model, model, i)
        wind_functions[i] = grib_to_wind_function(filename)
    return wind_functions


def wind_function(winds, coordinate, time):
    """
    Vectorized TWA and TWS function from forecast.

            Parameters:
                    winds (dict): available forecast wind functions
                    coordinate (array): array of tuples (lats, lons)
                    time (datetime): time to forecast

            Returns:
                    forecast (dict):
                        twa (array): array of TWA
                        tws (array): array of TWS
    """
    model_time = dt.datetime.strptime(winds['model'], "%Y%m%d%H")
    rounded_time = round_time(time, 3600 * 3)

    timedelta = rounded_time - model_time
    forecast = int(timedelta.seconds / 3600)

    wind = winds[forecast]
    twa = wind['twa'](coordinate)
    tws = wind['tws'](coordinate)
    return {'twa': twa, 'tws': tws}
