from skimage.graph import route_through_array
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import os
from geographiclib.geodesic import Geodesic


def loadData(path):
    '''
    This function take a string as the path of the file and load the data from the file.
    Parameters:
    path: String, path of the file
    Returns:
    data: xarray, dataset
    '''
    # Path of the data file
    file = os.path.join(path,'mfwamglocep_2020120200_R20201203.nc')

    # read the dataset
    data = xr.open_dataset(file)
    #data = data.VHM0.isel(time=0)
    return data

# create a bounding box from the coordinates
# NY to Lisbon 
def get_closest(array, value):
    return np.abs(array - value).argmin()

def getBBox(lon1, lat1,lon2,lat2, data):
    bbox = ((lon1, lat1),(lon2, lat2))
    time_slice = 0
    lon_min = get_closest(data.longitude.data, bbox[0][0])
    lat_min = get_closest(data.latitude.data, bbox[0][1])
    lon_max = get_closest(data.longitude.data, bbox[1][0])
    lat_max = get_closest(data.latitude.data, bbox[1][1])
    return lon_min, lon_max, lat_min, lat_max

def cleanData(data):
    # copy the data and remove NaN
    #cost = wave_height.data
    cost = data.copy()
    #np.random.shuffle(cost)
    nan_mask = np.isnan(cost)
    cost[nan_mask] = 2* np.nanmax(cost) if np.nanmax(cost) else 0

    return cost

def findStartAndEnd(lat1, lon1, lat2, lon2, wave_height):
    # Define start and end points
    lat_NY = lat1
    lon_NY = lon1
    lat_LS = lat2
    lon_LS = lon2

    start_lon = get_closest(wave_height.longitude.data, lon_NY)
    start_lat = get_closest(wave_height.latitude.data, lat_NY)
    end_lon = get_closest(wave_height.longitude.data,lon_LS)
    end_lat = get_closest(wave_height.latitude.data,lat_LS)

    start = (start_lat, start_lon)
    end = (end_lat, end_lon)

    return start, end

def distance(route):
    geod = Geodesic.WGS84
    dists = []

    lat1 = route[0,1]
    lon1 = route[0,0]
    d = 0

    for coord in route:
        lat2 = coord[1]
        lon2 = coord[0]
        d += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        dists.append(d)
        lat1 = lat2
        lon1 = lon2
    dists = np.array(dists)/1000
    print(dists)
    return dists

def time_diffs(speed, route):
    geod = Geodesic.WGS84
    speed = speed * 1.852

    lat1 = route[0,1]
    lon1 = route[0,0]
    diffs = []

    for coord in route:
        lat2 = coord[1]
        lon2 = coord[0]
        d = geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        diffs.append(d)
        lat1 = lat2
        lon1 = lon2

    diffs = np.array(diffs) / (speed * 1000)
    print(diffs)
    return diffs