import logging
import os
import random

import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from skimage.graph import route_through_array

from WeatherRoutingTool.algorithms.data_utils import GridMixin
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.routeparams import RouteParams

logger = logging.getLogger('WRT.Genetic')


class GridBasedPopulation(GridMixin, Sampling):
    """
    Make initial population for genetic algorithm based on a grid and associated cost values

    Notes on the inheritance:
     - GridMixin has to be inherited first because Sampling isn't designed for multiple inheritance
     - implemented approach: https://stackoverflow.com/a/50465583, scenario 2
     - call print(GridBasedPopulation.mro()) to see the method resolution order
    """
    def __init__(self, src, dest, grid, var_type=np.float64):
        super().__init__(grid=grid)
        self.var_type = var_type
        self.src = src
        self.dest = dest

    def _do(self, problem, n_samples, **kwargs):
        cost = self.grid.data
        shuffled_cost = cost.copy()
        nan_mask = np.isnan(shuffled_cost)
        routes = np.full((n_samples, 1), None, dtype=object)
        _, _, start_indices = self.coords_to_index([(self.src[0], self.src[1])])
        _, _, end_indices = self.coords_to_index([(self.dest[0], self.dest[1])])
        for i in range(n_samples):
            shuffled_cost = cost.copy()
            shuffled_cost[nan_mask] = 1
            shuffled_indices = np.random.permutation(len(shuffled_cost))
            shuffled_cost = shuffled_cost[shuffled_indices]
            shuffled_cost[nan_mask] = 1e20

            route, _ = route_through_array(shuffled_cost, start_indices[0], end_indices[0],
                                           fully_connected=True, geometric=False)
            # logger.debug(f"GridBasedPopulation._do: type(route)={type(route)}, route={route}")
            _, _, route = self.index_to_coords(route)
            routes[i][0] = np.array(route)

        figure_path = get_figure_path()
        if figure_path is not None:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            ax.add_feature(cf.LAND)
            ax.add_feature(cf.COASTLINE)
            for i in range(0, n_samples):
                ax.plot(routes[i, 0][:, 1], routes[i, 0][:, 0], color="firebrick")
            ax.set_xlim([-160, -115])
            ax.set_ylim([30, 60])
            plt.savefig(os.path.join(figure_path, 'genetic_algorithm_initial_population.png'))

        self.X = routes
        return self.X


class PopulationFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_population(population_type, src, dest, grid=None):
        if population_type == 'grid_based':
            population = GridBasedPopulation(src, dest, grid)
        else:
            msg = f"Population type '{population_type}' is invalid!"
            logger.error(msg)
            raise ValueError(msg)
        return population


class GeneticCrossover(Crossover):
    """
    Custom class to define genetic crossover for routes
    """
    def __init__(self, prob=1):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.cross_over(a, b)
        # print("Y:",Y)
        return Y

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


class CrossoverFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_crossover():
        crossover = GeneticCrossover()
        return crossover


class GridBasedMutation(GridMixin, Mutation):
    """
    Custom class to define genetic mutation for routes
    """
    def __init__(self, grid, prob=0.4):
        super().__init__(grid=grid)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        offsprings = np.zeros((len(X), 1), dtype=object)
        # loop over individuals in population
        for idx, i in enumerate(X):
            # perform mutation with certain probability
            if np.random.uniform(0, 1) < self.prob:
                mutated_individual = self.mutate(i[0])
                # print("mutated_individual", mutated_individual, "###")
                offsprings[idx][0] = mutated_individual
        # if no mutation
            else:
                offsprings[idx][0] = i[0]
        return offsprings

    def mutate(self, route):
        cost = self.grid.data
        shuffled_cost = cost.copy()
        nan_mask = np.isnan(shuffled_cost)

        size = len(route)
        start = random.randint(1, size - 2)
        end = random.randint(start, size - 2)

        _, _, start_indices = self.coords_to_index([(route[start][0], route[start][1])])
        _, _, end_indices = self.coords_to_index([(route[end][0], route[end][1])])

        shuffled_cost = np.ones(cost.shape, dtype=np.float)
        shuffled_cost[nan_mask] = 1e20

        subpath, _ = route_through_array(shuffled_cost, start_indices[0], end_indices[0],
                                         fully_connected=True, geometric=False)
        _, _, subpath = self.index_to_coords(subpath)
        newPath = np.concatenate((route[:start], np.array(subpath), route[end + 1:]), axis=0)
        return newPath


class MutationFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_mutation(mutation_type, grid=None):
        if mutation_type == 'grid_based':
            mutation = GridBasedMutation(grid)
        else:
            msg = f"Mutation type '{mutation_type}' is invalid!"
            logger.error(msg)
            raise ValueError(msg)
        return mutation


class RoutingProblem(ElementwiseProblem):
    """
    Class definition of the weather routing problem
    """
    boat: None
    constraint_list: None
    departure_time: None

    def __init__(self, departure_time, boat, constraint_list):
        super().__init__(n_var=1, n_obj=1, n_constr=1)
        self.boat = boat
        self.constraint_list = constraint_list
        self.departure_time = departure_time

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
        # logger.debug(f"RoutingProblem._evaluate: type(x)={type(x)}, x.shape={x.shape}, x={x}")
        fuel, _ = self.get_power(x[0])
        constraints = self.get_constraints(x[0])
        # print(costs.shape)
        out['F'] = np.column_stack([fuel])
        out['G'] = np.column_stack([constraints])

    def is_neg_constraints(self, lat, lon, time):
        lat = np.array([lat])
        lon = np.array([lon])
        is_constrained = [False for i in range(0, lat.shape[0])]
        is_constrained = self.constraint_list.safe_endpoint(lat, lon, time, is_constrained)
        # print(is_constrained)
        return 0 if not is_constrained else 1

    def get_constraints(self, route):
        # ToDo: what about time?
        constraints = np.sum([self.is_neg_constraints(lat, lon, None) for lat, lon in route])
        return constraints

    def get_power(self, route):
        route_dict = RouteParams.get_per_waypoint_coords(route[:, 1], route[:, 0], self.departure_time,
                                                         self.boat.boat_speed_function())

        shipparams = self.boat.get_fuel_per_time_netCDF(route_dict['courses'], route_dict['start_lats'],
                                                        route_dict['start_lons'], route_dict['start_times'])
        fuel = shipparams.get_fuel()
        fuel = (fuel / 3600) * route_dict['travel_times']
        return np.sum(fuel), shipparams
