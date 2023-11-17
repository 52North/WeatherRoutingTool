from skimage.graph import route_through_array
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import os
from geographiclib.geodesic import Geodesic
import math
from WeatherRoutingTool.ship.ship import Tanker
from datetime import datetime


def loadData(path):
    """
    This function take a string as the path of the file and load the data from the file.
    Parameters:
    path: String, path of the file
    Returns:
    data: xarray, dataset
    """
    # Path of the data file
    file = os.path.join(path)

    # read the dataset
    data = xr.open_dataset(file)
    # data = data.VHM0.isel(time=0)
    return data


# create a bounding box from the coordinates
# NY to Lisbon 
def get_closest(array, value):
    return np.abs(array - value).argmin()


def getBBox(lon1, lat1,lon2,lat2, data):
    lon_min = get_closest(data.longitude.data,lon1)
    lon_max = get_closest(data.longitude.data,lon2)
    lat_min = get_closest(data.latitude.data,lat1)
    lat_max = get_closest(data.latitude.data,lat2)

    lon_min = lon_min if lon_min < lon_max else lon_max 
    lon_max = lon_max if lon_min < lon_max else lon_min
    lat_min = lat_min if lat_min < lat_max else lat_max 
    lat_max = lat_max if lat_min < lat_max else lat_min
    # print(lon_min, lon_max, lat_min, lat_max)
    return lon_min, lon_max, lat_min, lat_max


def cleanData(data):
    cost = data.copy()
    nan_mask = np.isnan(cost)
    cost[nan_mask] = 1e100* np.nanmax(cost) if np.nanmax(cost) else 0

    return cost


def findStartAndEnd(lat1, lon1, lat2, lon2, grid_points):
    # Define start and end points

    start_lon = get_closest(grid_points.longitude.data, lon1)
    start_lat = get_closest(grid_points.latitude.data, lat1)
    end_lon = get_closest(grid_points.longitude.data,lon2)
    end_lat = get_closest(grid_points.latitude.data,lat2)

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
    dists = np.array(dists)
    # print(dists)
    return dists


def time_diffs(speed, route):
    geod = Geodesic.WGS84
    # speed = speed * 1.852

    lat1 = route[0, 0]
    lon1 = route[0, 1]
    diffs = []
    d = 0
    for coord in route:
        lat2 = coord[0]
        lon2 = coord[1]
        d = d + geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        diffs.append(d)
        lat1 = lat2
        lon1 = lon2

    diffs = np.array(diffs) / speed
    # print(diffs)
    return diffs


def calculate_course_for_route(route):
    geod = Geodesic.WGS84
    courses = np.zeros(len(route)-1)
    lats = np.zeros(len(route)-1)
    lons = np.zeros(len(route)-1)

    # print(route)
    for i in range(len(route) - 1):
        # Get the coordinates of the current and next waypoints
        lat1, lon1 = route[i]
        lats[i] = lat1
        lons[i] = lon1
        lat2, lon2 = route[i+1]
        course = geod.Inverse(lat1, lon1, lat2, lon2)['azi1']
        courses[i] = course
    
    # print(courses, lats, lons)
    return courses, lats, lons

