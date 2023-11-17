import random
from datetime import timedelta

import numpy as np
import xarray as xr
from skimage.graph import route_through_array

from WeatherRoutingTool.algorithms.data_utils import calculate_course_for_route, time_diffs
from WeatherRoutingTool.routeparams import *


class GeneticUtils:
    wave_height: xr.Dataset
    boat: None
    constraint_list: None
    departure_time: None

    def __init__(self, departure_time, grid_points, boat, constraint_list):
        self.grid_points = grid_points
        self.boat = boat
        self.constraint_list = constraint_list
        self.departure_time = departure_time

    def get_power(self, route):
        _, _, route = self.index_to_coords(route[0])
        route_dict = RouteParams.get_per_waypoint_coords(route[:, 0], route[:, 1], self.departure_time,
                                                         self.boat.boat_speed_function())

        shipparams = self.boat.get_fuel_per_time_netCDF(route_dict['courses'], route_dict['start_lats'],
                                                        route_dict['start_lons'], route_dict['start_times'])
        fuel = shipparams.get_fuel()
        fuel = (fuel / 3600) * route_dict['travel_times']

        return np.sum(fuel), shipparams

    def interpolate_grid(self, lat_int, lon_int):
        self.grid_points = self.grid_points[::lat_int, ::lon_int]

    def get_grid(self):
        return self.grid_points

    def power_cost(self, routes):
        costs = []
        for route in routes:
            fuel, _ = self.get_power(route)
            costs.append(fuel)
        return costs

    def is_neg_constraints(self, lat, lon, time):
        lat = np.array([lat])
        lon = np.array([lon])
        is_constrained = [False for i in range(0, lat.shape[0])]
        is_constrained = self.constraint_list.safe_endpoint(lat, lon, time, is_constrained)
        # print(is_constrained)
        return 0 if not is_constrained else 1

    def route_const(self, routes):
        cost = self.grid_points
        costs = []
        for route in routes:
            costs.append(np.sum([
                self.is_neg_constraints(self.grid_points.coords['latitude'][i], self.grid_points.coords['longitude'][j],
                                        cost[i, j]) for i, j in route[0]]))
        # print(costs)
        return costs

    def index_to_coords(self, route):
        lats = self.grid_points.coords['latitude'][route[:, 0]]
        lons = self.grid_points.coords['longitude'][route[:, 1]]
        route = [[x, y] for x, y in zip(lats, lons)]
        # print(type(lats))
        return lats, lons, np.array(route)

    # make initial population for genetic algorithm
    def population(self, size, src, dest):
        cost = self.grid_points.data
        shuffled_cost = cost.copy()
        nan_mask = np.isnan(shuffled_cost)
        routes = np.zeros((size, 1), dtype=object)

        debug = False

        for i in range(size):
            shuffled_cost = cost.copy()
            shuffled_cost[nan_mask] = 1
            shuffled_indices = np.random.permutation(len(shuffled_cost))
            shuffled_cost = shuffled_cost[shuffled_indices]
            shuffled_cost[nan_mask] = 1e20

            route, _ = route_through_array(shuffled_cost, src, dest, fully_connected=True, geometric=False)
            routes[i][0] = np.array(route)

        if debug:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            ax.add_feature(cf.LAND)
            ax.add_feature(cf.COASTLINE)
            for i in range(0, size):
                _, _, route = self.index_to_coords(routes[i][0])
                ax.plot(route[:, 1], route[:, 0], color="firebrick")

            plt.show()

        return routes

    def cross_over(self, parent1, parent2):
        # src = parent1[0]
        # dest = parent1[-1]
        intersect = np.array([x for x in parent1 if any((x == y).all() for y in parent2)])

        if len(intersect) == 0:
            return parent1, parent2
        else:
            cross_over_point = random.choice(intersect)
            idx1 = np.where((parent1 == cross_over_point).all(axis=1))[0][0]
            idx2 = np.where((parent2 == cross_over_point).all(axis=1))[0][0]
            child1 = np.concatenate((parent1[:idx1], parent2[idx2:]), axis=0)
            child2 = np.concatenate((parent2[:idx2], parent1[idx1:]), axis=0)  # print(child1, child2)
        return child1, child2

    def route_cost(self, routes):
        cost = self.grid_points.data
        costs = []
        for route in routes:
            costs.append(np.sum([cost[i, j] for i, j in route[0]]))
        return costs

    def mutate(self, route):
        cost = self.grid_points.data
        # source = route[0]
        # destination = route[-1]
        shuffled_cost = cost.copy()
        nan_mask = np.isnan(shuffled_cost)

        # path = route.copy()
        size = len(route)

        start = random.randint(1, size - 2)
        end = random.randint(start, size - 2)

        shuffled_cost = np.ones(cost.shape, dtype=np.float)
        shuffled_cost[nan_mask] = 1e20

        subpath, _ = route_through_array(shuffled_cost, route[start], route[end], fully_connected=True, geometric=False)
        newPath = np.concatenate((route[:start], np.array(subpath), route[end + 1:]), axis=0)

        return newPath
