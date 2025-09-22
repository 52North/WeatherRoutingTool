from skimage.graph import route_through_array
from pymoo.core.sampling import Sampling

import numpy as np
import logging
import os.path
import os

from WeatherRoutingTool.weather import WeatherCond
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.data_utils import GridMixin
from WeatherRoutingTool.algorithms.genetic import utils

logger = logging.getLogger("WRT.genetic.population")


class Population(Sampling):
    """Base Population Class"""

    def __init__(self, default_route: list, constraints_list: list, pop_size: int):
        super().__init__()

        self.constraints_list = constraints_list
        self.n_constrained_routes = 0
        self.pop_size = pop_size

        self.src: tuple[float, float] = tuple(default_route[:-2])
        self.dst: tuple[float, float] = tuple(default_route[-2:])

    def check_validity(self, routes):
        """
        Check whether the waypoints of the routes violate constraints. Raise a warning for each single route that
        violates constraints. Raise an error, if more than 50% of all routes violate constraints.

        :param routes: array of routes for initial population
        """

        n_route = 1
        for route in routes:
            constraints = utils.get_constraints(route[0], self.constraints_list)

            if constraints:
                logger.warning("Initial Route route_" + str(n_route) + " is constrained.")
                self.n_constrained_routes += 1
                n_route += 1

        percentage_constrained = self.n_constrained_routes / self.pop_size
        assert (percentage_constrained < 0.5), "More than 50% of the initial routes are constrained"


class GridBasedPopulation(GridMixin, Population):
    """Make initial population for genetic algorithm based on a grid and associated cost values

    Notes on the inheritance:
     - GridMixin has to be inherited first because Sampling isn't designed for multiple inheritance
     - implemented approach: https://stackoverflow.com/a/50465583, scenario 2
     - call `print(GridBasedPopulation.mro())` to see the method resolution order
    """

    def __init__(self, default_route, grid, constraints_list, pop_size):
        super().__init__(default_route=default_route, grid=grid, constraints_list=constraints_list, pop_size=pop_size)

        # ----------
        self.var_type = np.float64

    def _do(self, problem, n_samples, **kw):
        self.X = routes = np.full((n_samples, 1), None, dtype=object)

        _, _, start_indices = self.coords_to_index([self.src])
        _, _, end_indices = self.coords_to_index([self.dst])

        for i in range(n_samples):
            shuffled_cost = self.get_shuffled_cost()

            route, _ = route_through_array(
                array=shuffled_cost,
                start=start_indices[0],
                end=end_indices[0],
                fully_connected=True,
                geometric=False, )

            # logger.debug(f"GridBasedPopulation._do: type(route)={type(route)}, route={route}")
            _, _, route = self.index_to_coords(route)

            if self.route_constraint_violations(np.array([self.src, *route[1:-1], self.dst])).any():
                logger.warning(f"Population {i} generation failed")

            # match first and last points to src and dst
            routes[i][0] = np.array([
                self.src, *route[1:-1], self.dst])

        self.check_validity(routes)

        return self.X

    def route_constraint_violations(self, route: np.ndarray) -> np.ndarray:
        """Check if route breaks any discrete constraints

        :param route: list of waypoints
        :dtype route: np.ndarray
        :return: Boolean array of constraint violations per waypoint
        :rtype: np.ndarray
        """

        is_constrained = self.constraints_list.safe_crossing_discrete(
            route[:-1, 0], route[:-1, 1], route[1:, 0], route[1:, 1],
            current_time=self.departure_time,
            is_constrained=[False] * (route.shape[0] - 1), )

        return np.array(is_constrained)


class FromGeojsonPopulation(Population):
    """Genetic population from a directory of routes

    NOTE:
        Routes are expected to be named in the following format: `route_{1..N}.json`

        example: `route_1.json, route_2.json, route_3.json, ...`
    """

    def __init__(self, default_route, routes_dir: str, constraints_list, pop_size):
        super().__init__(default_route=default_route, constraints_list=constraints_list, pop_size=pop_size)

        if not os.path.exists(routes_dir) or not os.path.isdir(routes_dir):
            raise FileNotFoundError("Routes directory not found")
        self.routes_dir: str = routes_dir

    def _do(self, problem, n_samples, **kw):
        logger.debug(f"Population from geojson routes: {self.routes_dir}")

        # routes are expected to be named in the following format:
        # route_{1..N}.json
        # example: route_1.json, route_2.json, route_3.json, ...

        self.X = routes = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            path = os.path.join(self.routes_dir, f"route_{i + 1}.json")

            if not os.path.exists(path):
                raise ValueError("The number of available routes for the initial population does not match the "
                                 "population size.")
            else:
                route = utils.route_from_geojson_file(path)

            assert np.array_equal(route[0], self.src), "Route not starting at source"
            assert np.array_equal(route[-1], self.dst), "Route not ending at destination"

            routes[i, 0] = np.array(route)

        self.check_validity(routes)
        return routes


class PopulationFactory:
    @staticmethod
    def get_population(
            config: Config,
            boat: Boat,
            constraints_list: ConstraintsList,
            wt: WeatherCond,
    ) -> Population:

        # wave height grid
        lat_res, lon_res = 1, 1
        wave_height = wt.ds.VHM0.isel(time=0)
        wave_height = wave_height[::lat_res, ::lon_res]

        # population
        population_type = config.GENETIC_POPULATION_TYPE
        population_size = config.GENETIC_POPULATION_SIZE

        match population_type:
            case "grid_based":
                return GridBasedPopulation(
                    default_route=config.DEFAULT_ROUTE,
                    grid=wave_height,
                    constraints_list=constraints_list,
                    pop_size=population_size
                )

            case "from_geojson":
                return FromGeojsonPopulation(
                    default_route=config.DEFAULT_ROUTE,
                    routes_dir=config.GENETIC_POPULATION_PATH,
                    constraints_list=constraints_list,
                    pop_size=population_size
                )

            case _:
                logger.error(f"Population type invalid: {population_type}")
                raise ValueError(population_type)
