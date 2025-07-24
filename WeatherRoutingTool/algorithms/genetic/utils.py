import json
import logging
import os
import random
from math import ceil
from pathlib import Path
from datetime import datetime

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from skimage.graph import route_through_array

from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.algorithms.data_utils import GridMixin
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.graphics import plot_genetic_algorithm_initial_population

logger = logging.getLogger('WRT.Genetic')


def save_routes(routes, path_to_route_folder: str, boat: Boat, coords_to_index=None):
    departure_time = datetime.strptime("2023-11-11T11:11Z", '%Y-%m-%dT%H:%MZ')

    path = Path(os.path.join(path_to_route_folder, 'initial_population'))
    path.mkdir(exist_ok=True)

    logger.debug(f"Saving initial population routes to {path}")

    for i, route in enumerate(routes.flatten()):

        route_dict = RouteParams.get_per_waypoint_coords(
            route[:, 1],
            route[:, 0],
            departure_time,
            boat.get_boat_speed())

        shipparams = boat.get_ship_parameters(
            route_dict['courses'],
            route_dict['start_lats'],
            route_dict['start_lons'],
            route_dict['start_times'])

        fuel = shipparams.get_fuel_rate()
        fuel = (fuel / 3600) * route_dict['travel_times']

        indices = coords_to_index([tuple(x) for x in route.tolist()]) if coords_to_index else None

        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for j, pt in enumerate(route.tolist()):
            geojson["features"].append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": pt[::-1]
                },
                "properties": {
                    "fuel": fuel[j].to_value() if j < fuel.shape[0] else 0,
                    "x-index": indices[0][j].item(),
                    "y-index": indices[1][j].item()
                },
                "id": i
            })

        with open(path / f'route_{i+1}.json', 'w') as fp:
            json.dump(geojson, fp)

    logger.info(f"Routes saved in {path}")


class GridBasedPopulation(GridMixin, Sampling):
    """
    Make initial population for genetic algorithm based on a grid and associated cost values

    Notes on the inheritance:
     - GridMixin has to be inherited first because Sampling isn't designed for multiple inheritance
     - implemented approach: https://stackoverflow.com/a/50465583, scenario 2
     - call print(GridBasedPopulation.mro()) to see the method resolution order
    """
    def __init__(self, src, dest, grid, path_to_route_folder, var_type=np.float64, boat=None):
        super().__init__(grid=grid)
        self.var_type = var_type
        self.src = src
        self.dest = dest
        self.path_to_route_folder = path_to_route_folder
        self.boat = boat

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

        save_routes(routes, self.path_to_route_folder, self.boat, coords_to_index=self.coords_to_index)
        plot_genetic_algorithm_initial_population(self.src, self.dest, routes)
        self.X = routes
        return self.X


class FromGeojsonPopulation(Sampling):
    """
    Make initial population for genetic algorithm based on the isofuel algorithm with a ConstantFuelBoat
    """
    def __init__(self, src, dest, path_to_route_folder, population_path, var_type=np.float64):
        super().__init__()
        self.var_type = var_type
        self.src = src
        self.dest = dest
        self.path_to_route_folder = path_to_route_folder
        self.population_path = population_path

    def _do(self, problem, n_samples, **kwargs):
        logger.debug(f"Population from GeoJSON routes: {self.population_path}")

        routes = np.full((n_samples, 1), None, dtype=object)
        # Routes have to be named route_1.json, route_2.json, etc.
        # See method find_routes_reaching_destination_in_current_step in isobased.py
        # ToDo: exit program with error when number of files is not equal to n_samples
        for i in range(n_samples):
            route_file = os.path.join(self.path_to_route_folder, "initial_population", f'route_{i+1}.json')
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
    def get_population(
        population_type,
        src,
        dest,
        path_to_route_folder=None,
        grid=None,
        population_path=None,
        boat=None
    ):
        if population_type == 'grid_based':
            if grid is None:
                msg = f"For population type '{population_type}', a grid has to be provided!"
                logger.error(msg)
                raise ValueError(msg)
            population = GridBasedPopulation(src, dest, grid, path_to_route_folder, boat=boat)
        elif population_type == 'from_geojson':
            if (not path_to_route_folder or not os.path.isdir(path_to_route_folder) or
                    not os.access(path_to_route_folder, os.R_OK)):
                msg = f"For population type '{population_type}', a valid route path has to be provided!"
                logger.error(msg)
                raise ValueError(msg)
            population = FromGeojsonPopulation(src, dest, path_to_route_folder, population_path)
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
            Y[0, k, 0], Y[1, k, 0] = self.pmx_cross_over(a, b)
        return Y

    def single_point_cross_over(self, parent1, parent2):
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

    def pmx_cross_over(self, parent1: np.ndarray, parent2: np.ndarray):
        """PMX crossover adapted for coordinate-based routing.

        `parent1` and `parent2` are of shape (N, D)

        where, N = number of waypoints, D = coordinate dimensions.
        """

        if len(parent1) < 3 or len(parent2) < 3:
            logger.info("Routes too short for PMX crossover, returning copies")
            return parent1.copy(), parent2.copy()

        src = parent1[0].copy()
        dest = parent1[-1].copy()
        middle1 = parent1[1:-1]
        middle2 = parent2[1:-1]

        if len(middle1) != len(middle2):
            logger.warning("Middle segments are of unequal length; skipping PMX")
            return parent1.copy(), parent2.copy()

        size = len(middle1)
        start, end = sorted(random.sample(range(size), 2))
        if start == end:
            end = min(size, start + 2)

        def pmx_single(m1, m2):
            child = [None] * size
            mapping = {}

            # Copy slice from m1
            for i in range(start, end):
                child[i] = m1[i]
                mapping[tuple(m2[i])] = tuple(m1[i])

            # Fill the rest
            for i in range(size):
                if child[i] is None:
                    candidate = tuple(m2[i])
                    while candidate in mapping:
                        candidate = mapping[candidate]
                    child[i] = np.array(candidate)

            return np.array(child)

        child1_middle = pmx_single(middle1, middle2)
        child2_middle = pmx_single(middle2, middle1)

        child1 = np.vstack([src, child1_middle, dest])
        child2 = np.vstack([src, child2_middle, dest])
        return child1, child2


class PMXCrossover(Crossover):
    """Partially Mapped Crossover"""

    def __init__(self, prob=.5):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob,)

    def _do(self, problem, X, **kw):
        n_parents, n_matings, n_var = X.shape

        # return var
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.pmx_crossover(p1.copy(), p2.copy())
        return Y

    def pmx_crossover(self, p1: np.ndarray, p2: np.ndarray):
        """Perform Partially Mapped Crossover (PMX) between two parent routes.

        Args:
            p1, p2: np.ndarray of shape (N,2) with the same set of waypoints.

        Returns:
            Two offspring np.ndarrays of shape (N,2).
        """

        if p1.shape != p2.shape:
            logging.info("PMX — Not of equal length")
            return p1, p2

        N = min(p1.shape[0], p2.shape[0])

        # Convert to lists of tuples
        parent1 = [tuple(row) for row in p1]
        parent2 = [tuple(row) for row in p2]

        # Choose crossover points
        cx1, cx2 = sorted(random.sample(range(N), 2))

        # Initialize offspring placeholders
        child1 = [None] * N
        child2 = [None] * N

        # Copy the segment
        for i in range(cx1, cx2):
            child1[i] = parent2[i]
            child2[i] = parent1[i]

        # Build mapping for the swapped segments
        mapping12 = {parent2[i]: parent1[i] for i in range(cx1, cx2)}
        mapping21 = {parent1[i]: parent2[i] for i in range(cx1, cx2)}

        def resolve(gene, segment, mapping):
            # Keep resolving until gene is not in the given segment
            while gene in segment:
                gene = mapping[gene]
            return gene

        # Fill remaining positions
        for i in range(N):
            if not (cx1 <= i < cx2):
                g1 = parent1[i]
                g2 = parent2[i]

                # If g1 is already in the swapped segment of child1, resolve via mapping12
                if g1 in child1[cx1:cx2]:
                    g1 = resolve(g1, child1[cx1:cx2], mapping12)
                child1[i] = g1

                # Likewise for child2
                if g2 in child2[cx1:cx2]:
                    g2 = resolve(g2, child2[cx1:cx2], mapping21)
                child2[i] = g2

        # Convert back to numpy arrays
        c1 = np.array(child1, dtype=p1.dtype)
        c2 = np.array(child2, dtype=p1.dtype)

        return c1, c2


class GeneticCrossover_X(Crossover):
    """
    Enhanced genetic crossover for route optimization with multiple strategies
    """
    def __init__(self, prob=1.0, crossover_type='intersection_based', grid=None):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob
        self.crossover_type = crossover_type
        self.grid = grid

    def _do(self, problem, X, **kwargs):
        # The input has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            # get the first and the second parent
            parent1, parent2 = X[0, k, 0], X[1, k, 0]

            # Apply crossover with probability
            if np.random.random() < self.prob:
                child1, child2 = self.cross_over(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            Y[0, k, 0], Y[1, k, 0] = child1, child2

        return Y

    def cross_over(self, parent1, parent2):
        """
        Main crossover method that delegates to specific crossover strategies
        """
        if self.crossover_type == 'intersection_based':
            return self._intersection_based_crossover(parent1, parent2)
        elif self.crossover_type == 'order_based':
            return self._order_based_crossover(parent1, parent2)
        elif self.crossover_type == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == 'geometric':
            return self._geometric_crossover(parent1, parent2)
        elif self.crossover_type == 'adaptive':
            return self._adaptive_crossover(parent1, parent2)
        else:
            logger.warning(f"Unknown crossover type: {self.crossover_type}. Using intersection_based.")
            return self._intersection_based_crossover(parent1, parent2)

    def _intersection_based_crossover(self, parent1, parent2):
        """
        Original intersection-based crossover with improvements
        """
        # Ensure start and end points are preserved
        src, dest = parent1[0], parent1[-1]

        # Find intersection points between routes
        intersect = np.array([x for x in parent1 if any((x == y).all() for y in parent2)])

        if len(intersect) == 0:
            # No intersection points - use midpoint crossover
            return self._midpoint_crossover(parent1, parent2)
        else:
            # Use intersection point for crossover
            cross_over_point = random.choice(intersect)
            idx1 = np.where((parent1 == cross_over_point).all(axis=1))[0][0]
            idx2 = np.where((parent2 == cross_over_point).all(axis=1))[0][0]

            child1 = np.concatenate((parent1[:idx1], parent2[idx2:]), axis=0)
            child2 = np.concatenate((parent2[:idx2], parent1[idx1:]), axis=0)

            # Ensure start and end points are correct
            child1[0], child1[-1] = src, dest
            child2[0], child2[-1] = src, dest

            return child1, child2

    def _midpoint_crossover(self, parent1, parent2):
        """
        Crossover at midpoint when no intersection points exist
        """
        src, dest = parent1[0], parent1[-1]

        # Use middle point of each route
        mid1 = len(parent1) // 2
        mid2 = len(parent2) // 2

        child1 = np.concatenate((parent1[:mid1], parent2[mid2:]), axis=0)
        child2 = np.concatenate((parent2[:mid2], parent1[mid1:]), axis=0)

        # Ensure start and end points are correct
        child1[0], child1[-1] = src, dest
        child2[0], child2[-1] = src, dest

        return child1, child2

    def _order_based_crossover(self, parent1, parent2):
        """
        Order-based crossover that preserves relative ordering of waypoints
        """
        src, dest = parent1[0], parent1[-1]

        # Create waypoint ordering based on distance from start
        def get_waypoint_order(route):
            distances = []
            for i, point in enumerate(route[1:-1]):  # Exclude start and end
                dist = np.sqrt((point[0] - src[0])**2 + (point[1] - src[1])**2)
                distances.append((dist, i+1, point))
            return sorted(distances, key=lambda x: x[0])

        order1 = get_waypoint_order(parent1)
        order2 = get_waypoint_order(parent2)

        # Create children by alternating between parent orders
        child1_waypoints = []
        child2_waypoints = []

        max_len = max(len(order1), len(order2))
        for i in range(max_len):
            if i < len(order1) and i < len(order2):
                if i % 2 == 0:
                    child1_waypoints.append(order1[i][2])
                    child2_waypoints.append(order2[i][2])
                else:
                    child1_waypoints.append(order2[i][2])
                    child2_waypoints.append(order1[i][2])
            elif i < len(order1):
                child1_waypoints.append(order1[i][2])
                child2_waypoints.append(order1[i][2])
            else:
                child1_waypoints.append(order2[i][2])
                child2_waypoints.append(order2[i][2])

        child1 = np.vstack([src, child1_waypoints, dest])
        child2 = np.vstack([src, child2_waypoints, dest])

        return child1, child2

    def _uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover that randomly selects waypoints from either parent
        """
        src, dest = parent1[0], parent1[-1]

        # Ensure both parents have same length for uniform crossover
        min_len = min(len(parent1), len(parent2))

        child1_waypoints = []
        child2_waypoints = []

        for i in range(1, min_len - 1):  # Skip start and end points
            if np.random.random() < 0.5:
                child1_waypoints.append(parent1[i])
                child2_waypoints.append(parent2[i])
            else:
                child1_waypoints.append(parent2[i])
                child2_waypoints.append(parent1[i])

        # Handle remaining waypoints
        if len(parent1) > min_len:
            child1_waypoints.extend(parent1[min_len-1:-1])
            child2_waypoints.extend(parent1[min_len-1:-1])
        elif len(parent2) > min_len:
            child1_waypoints.extend(parent2[min_len-1:-1])
            child2_waypoints.extend(parent2[min_len-1:-1])

        child1 = np.vstack([src, child1_waypoints, dest])
        child2 = np.vstack([src, child2_waypoints, dest])

        return child1, child2

    def _geometric_crossover(self, parent1, parent2):
        """
        Geometric crossover that creates waypoints as weighted averages
        """
        src, dest = parent1[0], parent1[-1]

        # Ensure both parents have same length
        min_len = min(len(parent1), len(parent2))

        child1_waypoints = []
        child2_waypoints = []

        for i in range(1, min_len - 1):  # Skip start and end points
            # Create weighted average of waypoints
            alpha = np.random.random()
            beta = 1 - alpha

            new_point1 = alpha * parent1[i] + beta * parent2[i]
            new_point2 = beta * parent1[i] + alpha * parent2[i]

            child1_waypoints.append(new_point1)
            child2_waypoints.append(new_point2)

        # Handle remaining waypoints
        if len(parent1) > min_len:
            child1_waypoints.extend(parent1[min_len-1:-1])
            child2_waypoints.extend(parent1[min_len-1:-1])
        elif len(parent2) > min_len:
            child1_waypoints.extend(parent2[min_len-1:-1])
            child2_waypoints.extend(parent2[min_len-1:-1])

        child1 = np.vstack([src, child1_waypoints, dest])
        child2 = np.vstack([src, child2_waypoints, dest])

        return child1, child2

    def _adaptive_crossover(self, parent1, parent2):
        """
        Adaptive crossover that chooses strategy based on route characteristics
        """
        # Calculate route characteristics
        len1, len2 = len(parent1), len(parent2)
        intersect_count = len([x for x in parent1 if any((x == y).all() for y in parent2)])

        # Choose crossover strategy based on characteristics
        if intersect_count > 0:
            # Use intersection-based if routes intersect
            return self._intersection_based_crossover(parent1, parent2)
        elif abs(len1 - len2) <= 2:
            # Use uniform crossover if routes are similar length
            return self._uniform_crossover(parent1, parent2)
        else:
            # Use order-based for very different routes
            return self._order_based_crossover(parent1, parent2)

    def _repair_route(self, route, src, dest):
        """
        Repair route to ensure it's valid (start/end points correct, no duplicates)
        """
        # Ensure start and end points
        route[0] = src
        route[-1] = dest

        # Remove duplicate consecutive waypoints
        unique_indices = []
        for i, point in enumerate(route):
            if i == 0 or not np.array_equal(point, route[i-1]):
                unique_indices.append(i)

        return route[unique_indices]


class CrossoverFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_crossover():
        crossover = PMXCrossover()
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
            if np.random.uniform(0, 1) < self.prob or True:
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


# Repair Population

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
