import json
import logging
import os
import random
from math import ceil

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from skimage.graph import route_through_array

from WeatherRoutingTool.algorithms.data_utils import GridMixin
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.graphics import plot_genetic_algorithm_initial_population

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
        routes = np.full((n_samples, 1), None, dtype=object)
        _, _, start_indices = self.coords_to_index([(self.src[0], self.src[1])])
        _, _, end_indices = self.coords_to_index([(self.dest[0], self.dest[1])])
        for i in range(n_samples):
            shuffled_cost = self.get_shuffled_cost()
            route, _ = route_through_array(shuffled_cost, start_indices[0], end_indices[0],
                                           fully_connected=True, geometric=False)
            # logger.debug(f"GridBasedPopulation._do: type(route)={type(route)}, route={route}")
            _, _, route = self.index_to_coords(route)

            # set initial and final points from the config file, to reach exact points
            route[0] = self.src
            route[-1] = self.dest

            routes[i][0] = np.array(route)

        plot_genetic_algorithm_initial_population(self.src, self.dest, routes)
        self.X = routes
        return self.X


class FromGeojsonPopulation(Sampling):
    """
    Make initial population for genetic algorithm based on the isofuel algorithm with a ConstantFuelBoat
    """
    def __init__(self, src, dest, path_to_route_folder, var_type=np.float64):
        super().__init__()
        self.var_type = var_type
        self.src = src
        self.dest = dest
        self.path_to_route_folder = path_to_route_folder

    def _do(self, problem, n_samples, **kwargs):
        routes = np.full((n_samples, 1), None, dtype=object)
        # Routes have to be named route_1.json, route_2.json, etc.
        # See method find_routes_reaching_destination_in_current_step in isobased.py
        # ToDo: exit program with error when number of files is not equal to n_samples
        for i in range(n_samples):
            route_file = os.path.join(self.path_to_route_folder, f'route_{i+1}.json')
            try:
                route = self.read_route_from_file(route_file)
                routes[i][0] = np.array(route)
            except FileNotFoundError:
                logger.warning(f"File '{route_file}' couldn't be found. Use great circle route instead.")
                route = self.get_great_circle_route()
                routes[i][0] = np.array(route)

        plot_genetic_algorithm_initial_population(self.src, self.dest, routes)
        self.X = routes
        return self.X

    def get_great_circle_route(self, distance=100000):
        """
        Get equidistant route along great circle in the form [[lat1, lon1], [lat12, lon2], ...]
        :param distance: distance in m
        :return: route as list of lat/lon points
        """
        geod = Geodesic.WGS84
        line = geod.InverseLine(self.src[0], self.src[1], self.dest[0], self.dest[1])
        n = int(ceil(line.s13 / distance))
        route = []
        for i in range(n+1):
            s = min(distance * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            route.append([g['lat2'], g['lon2']])
        return route

    def read_route_from_file(self, route_absolute_path):
        """
        Read route from geojson file and return the coordinates in the form [[lat1, lon1], [lat12, lon2], ...]
        :param route_absolute_path: absolute path to geojson file
        :return: route as list of lat/lon points
        """
        with open(route_absolute_path) as file:
            rp_dict = json.load(file)
        route = [[feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]]
                 for feature in rp_dict['features']]
        return route


class PopulationFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_population(population_type, src, dest, path_to_route_folder=None, grid=None):
        if population_type == 'grid_based':
            if grid is None:
                msg = f"For population type '{population_type}', a grid has to be provided!"
                logger.error(msg)
                raise ValueError(msg)
            population = GridBasedPopulation(src, dest, grid)
        elif population_type == 'from_geojson':
            if (not path_to_route_folder or not os.path.isdir(path_to_route_folder) or
                    not os.access(path_to_route_folder, os.R_OK)):
                msg = f"For population type '{population_type}', a valid route path has to be provided!"
                logger.error(msg)
                raise ValueError(msg)
            population = FromGeojsonPopulation(src, dest, path_to_route_folder)
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
        size = len(route)
        start = random.randint(1, size - 2)
        end = random.randint(start, size - 2)

        _, _, start_indices = self.coords_to_index([(route[start][0], route[start][1])])
        _, _, end_indices = self.coords_to_index([(route[end][0], route[end][1])])

        shuffled_cost = self.get_shuffled_cost()
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

    def get_constraints_array(self, route: np.ndarray) -> np.ndarray:
        """
        Return constraint violation per waypoint in route

        :param route: Candidate array of waypoints
        :type route: np.ndarray
        :return: Array of constraint violations
        """

        constraints = np.array([self.is_neg_constraints(lat, lon, None) for lat, lon in route])
        return constraints

    def get_constraints(self, route):
        # ToDo: what about time?
        constraints = np.sum(self.get_constraints_array(route))
        return constraints

    def get_power(self, route):
        route_dict = RouteParams.get_per_waypoint_coords(route[:, 1], route[:, 0], self.departure_time,
                                                         self.boat.get_boat_speed())

        shipparams = self.boat.get_ship_parameters(route_dict['courses'], route_dict['start_lats'],
                                                   route_dict['start_lons'], route_dict['start_times'])
        fuel = shipparams.get_fuel_rate()
        fuel = (fuel / 3600) * route_dict['travel_times']
        return np.sum(fuel), shipparams


class RouteDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return np.array_equal(a.X[0], b.X[0])


## Repair Class

class RepairInfeasibles(Repair):
    """
    Repairs infeasible candidates of a population
    """

    def _repair_candidate(self, route: np.ndarray, constraints: np.ndarray) -> np.ndarray:
        """
        Repair the waypoints of a route when a constraint violation is found.

        NOTE: Currently replaces the waypoint with the previous one in the
        route, which is not feasible at all; need to incorporate a closest
        feasible waypoint finding method

        :param route: Candidate solution array
        :type route: np.ndarray
        :param constraints: Constraints array indicating violations with waypoints
        :type constraints: np.ndarray
        :return: Updated Candidate array
        """

        for i, c in enumerate(constraints):
            if i == 0:
                continue

            # TODO: Adjust to replace waypoint with a feasible one instead of the previous one
            if c != 0:
                route[i] = route[i - 1]
        return route

    def _do(self, problem: RoutingProblem, Z: np.ndarray, **kw) -> np.ndarray:
        """
        Fix routes when waypoints do not meet the constraints

        :param problem: Problem being solved by the algorithm
        :type problem: RoutingProblem
        :param Z: Candidate solution array
        :type Z: np.ndarray
        :return: Array of updated population
        """

        constraints = [problem.get_constraints_array(r[0]) for r in Z]

        for i, c in enumerate(constraints):
            # if any constraint is broken in the route
            if c.any():
                Z[i, 0] = self._repair_candidate(Z[i, 0].copy(), c)
        return Z
