import numpy as np
import random
from skimage.graph import route_through_array
import sys
import os

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
print(current_path)
sys.path.append(os.path.join(current_path, '..', ''))

from constraints.constraints import *


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
    return 1 if not is_constrained else 9999999

def index_to_coords(route):
    lats = wave_height.coords['latitude'][route[:,0]]
    lons = wave_height.coords['longitude'][route[:,1]]
    route = [[x,y] for x,y in zip(lats, lons)]
    return lats, lons,np.array(route)

# make initial population for genetic algorithm
def population(size, src, dest, cost):
    shuffled_cost = cost.copy()
    #print(shuffled_cost)
    #routes = []
    routes = np.zeros((size,1), dtype=object)
    for i in range(size):
        shuffled_indices = np.random.permutation(len(shuffled_cost))
        shuffled_cost = shuffled_cost[shuffled_indices]
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
    print(costs)
    return costs
        

def mutate(route):
    source = route[0]
    destination = route[-1]

    path = route.copy()
    size = len(route)
    for i in range(1, size):
        end = random.randint(i+1,size-2)
        subpath, _ = route_through_array(cost,route[i-1], route[end+1], fully_connected=True, geometric=False)
        newPath = np.concatenate((route[:i-1],np.array(subpath),route[end+2:]), axis=0)
        #print(newPath)
        #if len(newPath) == len(set(newPath)):
        return newPath
