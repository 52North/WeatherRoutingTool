import random

import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from skimage.graph import route_through_array

from WeatherRoutingTool.routeparams import RouteParams


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


class Population(Sampling):
    def __init__(self, src, dest, util, var_type=np.float64):
        super().__init__()
        self.var_type = var_type
        self.src = src
        self.dest = dest
        self.util = util

    def _do(self, problem, n_samples, **kwargs):
        routes = self.util.population(n_samples, self.src, self.dest)
        # print(routes.shape)
        self.X = routes
        # print(self.X.shape)
        return self.X


class GeneticCrossover(Crossover):
    def __init__(self, util, prob=1):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob
        self.util = util

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.util.cross_over(a, b)
        # print("Y:",Y)
        return Y


class GeneticMutation(Mutation):
    def __init__(self, util, prob=0.4):
        super().__init__()
        self.prob = prob
        self.util = util

    def _do(self, problem, X, **kwargs):
        offsprings = np.zeros((len(X), 1), dtype=object)
        # loop over individuals in population
        for idx, i in enumerate(X):
            # perform mutation with certain probability
            if np.random.uniform(0, 1) < self.prob:
                mutated_individual = self.util.mutate(i[0])
                # print("mutated_individual", mutated_individual, "###")
                offsprings[idx][0] = mutated_individual
        # if no mutation
            else:
                offsprings[idx][0] = i[0]
        return offsprings


class RoutingProblem(Problem):
    """
    Class definition of the weather routing problem
    """
    def __init__(self, util):
        super().__init__(n_var=1, n_obj=1, n_constr=1)
        self.util = util

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Method defined by pymoo which has to be overriden
        :param x: numpy matrix with shape (rows: number of solutions/individuals, columns: number of design variables)
        :param out:
            out['F']: function values, vector of length of number of solutions
            out['G']: constraints
        :param args:
        :param kwargs:
        :return:
        """
        # costs = route_cost(X)
        costs = self.util.power_cost(x)
        constraints = self.util.route_const(x)
        # print(costs.shape)
        out['F'] = np.column_stack([costs])
        out['G'] = np.column_stack([constraints])


def optimize(strt, end, pop_size, n_gen, n_offspring, util):
    # cost[nan_mask] = 20000000000* np.nanmax(cost) if np.nanmax(cost) else 0
    problem = RoutingProblem(util)
    algorithm = NSGA2(pop_size=pop_size,
                      sampling=Population(strt, end, util),
                      crossover=GeneticCrossover(util),
                      n_offsprings=n_offspring,
                      mutation=GeneticMutation(util),
                      eliminate_duplicates=False,
                      return_least_infeasible=False)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=True,
                   verbose=True)
    # stop = timeit.default_timer()
    # route_cost(res.X)
    return res
