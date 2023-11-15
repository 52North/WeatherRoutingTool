import numpy as np
import random
from skimage.graph import route_through_array
import sys
import os

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
print(current_path)
sys.path.append(os.path.join(current_path, '..', ''))

from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.algorithms.DataUtils import *



pars = ConstraintPars()
pars.resolution = 1./10

constraint_list = ConstraintsList(pars)

land_crossing = LandCrossing()
wave_height_const = WaveHeight()
constraint_list.add_neg_constraint(land_crossing)
constraint_list.add_neg_constraint(wave_height_const)
#global wave_height
def set_data(data, cst):
    global wave_height
    wave_height = data
    global cost
    cost = cst

def is_neg_constraints(lat, lon, wh, time):  
    lat = np.array([lat])
    lon = np.array([lon])
    #print(lat,lon)

    wave_height_const.current_wave_height = np.array([wh])

    is_constrained = [False for i in range(0, lat.shape[0])]
    is_constrained = constraint_list.safe_endpoint(lat, lon, time, is_constrained)
    #print(is_constrained)
    return 0 if not is_constrained else 1

def route_const(routes):
    crossing = 0
    costs = []
    #print(routes[0][0])
    for route in routes:
        costs.append(np.sum([is_neg_constraints(wave_height.coords['latitude'][i],
                                                          wave_height.coords['longitude'][j], cost[i,j], 0) for i,j in route[0]]))
    #print(costs)
    return costs

def index_to_coords(route):
    lats = wave_height.coords['latitude'][route[:,0]]
    lons = wave_height.coords['longitude'][route[:,1]]
    route = [[x,y] for x,y in zip(lats, lons)]
    #print(type(lats))
    return lats, lons,np.array(route)

# make initial population for genetic algorithm
def population(size, src, dest, cost):
    shuffled_cost = cost.copy()
    #print(shuffled_cost)
    #routes = []
    nan_mask = np.isnan(shuffled_cost)
    routes = np.zeros((size,1), dtype=object)
    for i in range(size):
        shuffled_cost = cost.copy()
        shuffled_cost[nan_mask] = 1
        shuffled_indices = np.random.permutation(len(shuffled_cost))
        shuffled_cost = shuffled_cost[shuffled_indices]
        shuffled_cost[nan_mask] = 1e20

        route, _ = route_through_array(shuffled_cost,src, dest, fully_connected=True, geometric=False)
        routes[i][0] = np.array(route)
    return routes

def crossOver(parent1, parent2):
    src = parent1[0]
    dest = parent1[-1]
    
    intersect = np.array([x for x in parent1 if any((x == y).all() for y in parent2)])

    if len(intersect) == 0:
        return parent1, parent2
    else:
        cross_over_point = random.choice(intersect)
        idx1 = np.where((parent1 == cross_over_point).all(axis=1))[0][0]
        idx2 = np.where((parent2 == cross_over_point).all(axis=1))[0][0]
        child1 = np.concatenate((parent1[:idx1], parent2[idx2:]), axis=0)
        child2 = np.concatenate((parent2[:idx2],parent1[idx1:]), axis=0)
        #print(child1, child2)
    return child1, child2

def route_cost(routes):
    costs = []
    #print(routes[0][0])
    for route in routes:
        #print(route[0])
        costs.append(np.sum([cost[i,j] for i,j in route[0]]))
        #costs.append(np.sum([cost[i,j] * is_neg_constraints(wave_height.coords['latitude'][i],wave_height.coords['longitude'][j], cost[i,j], 0) for i,j in route[0]]))
    #print(costs)
    return costs

def power_cost(routes, boat, departure_time):
    costs = []

    for route in routes:
        fuels = getPower(route, wave_height, boat, departure_time)
        #print(fuels.get_full_fuel())
        #costs.append(np.sum([fuel for fuel in fuels.]))
        costs.append(fuels)
    #print(costs)
    return costs

def getPower(route, wave_height, boat, departure_time):
    courses, lats, lons = calculate_course_for_route(route[0], wave_height)
    npoints = len(courses)

    # FIXME: why are we passing the departure time here? Should be the times where we start from each waypoint.
    time = np.array([departure_time] * npoints)
    power = boat.get_fuel_per_time_netCDF(courses, lats, lons, time)
    full_fuel = 0

    for ipoint in range(0,npoints-1):
        time_passed = (time[ipoint + 1] - time[ipoint]).total_seconds() / 3600
        fuel_per_step = power.fuel[ipoint] * time_passed
        full_fuel = full_fuel + fuel_per_step
    return full_fuel

def mutate(route):
    source = route[0]
    destination = route[-1]
    shuffled_cost = cost.copy()
    nan_mask = np.isnan(shuffled_cost)

    path = route.copy()
    size = len(route)

    start = random.randint(1,size-2)
    end = random.randint(start,size-2)
    
    shuffled_cost = np.ones(cost.shape, dtype=np.float)
    shuffled_cost[nan_mask] = 1e20
   
    subpath, _ = route_through_array(shuffled_cost,route[start], route[end], fully_connected=True, geometric=False)
    newPath = np.concatenate((route[:start],np.array(subpath),route[end+1:]), axis=0)

    return newPath
