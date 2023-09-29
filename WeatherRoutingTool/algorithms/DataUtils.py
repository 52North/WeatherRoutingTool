from skimage.graph import route_through_array
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import os
from geographiclib.geodesic import Geodesic
import math
from WeatherRoutingTool.ship.ship import Tanker
from datetime import datetime
import WeatherRoutingTool.config as config


def loadData(path):
    '''
    This function take a string as the path of the file and load the data from the file.
    Parameters:
    path: String, path of the file
    Returns:
    data: xarray, dataset
    '''
    # Path of the data file
    file = os.path.join(path)

    # read the dataset
    data = xr.open_dataset(file)
    #data = data.VHM0.isel(time=0)
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
    #print(lon_min, lon_max, lat_min, lat_max)
    return lon_min, lon_max, lat_min, lat_max

def cleanData(data):
    # copy the data and remove NaN
    #cost = wave_height.data
    cost = data.copy()
    #np.random.shuffle(cost)
    nan_mask = np.isnan(cost)
    cost[nan_mask] = 1e100* np.nanmax(cost) if np.nanmax(cost) else 0

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
    dists = np.array(dists)
    #print(dists)
    return dists

def time_diffs(speed, route):
    geod = Geodesic.WGS84
    #speed = speed * 1.852

    lat1 = route[0,1]
    lon1 = route[0,0]
    diffs = []
    d = 0
    for coord in route:
        lat2 = coord[1]
        lon2 = coord[0]
        d = d + geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        diffs.append(d)
        lat1 = lat2
        lon1 = lon2

    diffs = np.array(diffs) / (speed)
    #print(diffs)
    return diffs


def calculate_course_for_route(route, wave_height):
    courses = np.zeros(len(route)-1)
    lats = np.zeros(len(route)-1)
    lons = np.zeros(len(route)-1)
    
    #print(route)
    for i in range(len(route) - 1):
        # Get the coordinates of the current and next waypoints
        lat1, lon1 = route[i]
        lats[i] = wave_height.coords['latitude'] [lat1]
        lons[i] = wave_height.coords['longitude'] [lon1]
        lat2, lon2 = route[i+1]
        
        # Convert latitude and longitude to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate the course in radians
        delta_lon = lon2_rad - lon1_rad
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        course_rad = math.atan2(y, x)
        
        # Convert the course to degrees
        course_deg = math.degrees(course_rad)
        
        # Adjust the degrees to be in the range of 0-360
        course = (course_deg + 360) % 360
        
        # Append the course to the list
        courses[i] = course
    
    #print(courses, lats, lons)
    return courses, lats, lons

def getPower(route, wave_height):
    #base = config.BASE_PATH
    DEFAULT_GFS_FILE = config.WEATHER_DATA  # CMEMS needs lat: 30 to 45, lon: 0 to 20
    COURSES_FILE = config.COURSES_FILE
    #print(route)
    courses, lats, lons = calculate_course_for_route(route[0], wave_height)
    #print(lons.shape)

    tank = Tanker(2)
    tank.init_hydro_model_Route(DEFAULT_GFS_FILE, COURSES_FILE,'')
    dt = '2020.12.02 00:00:00' 
    dt_obj = datetime.strptime(dt, '%Y.%m.%d %H:%M:%S')
    departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')

    time = np.array([departure_time]*len(courses))
    power = tank.get_fuel_per_time_netCDF(courses, lats, lons, time)
    return power

