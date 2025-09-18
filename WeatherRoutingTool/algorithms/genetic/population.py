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

    def __init__(self, default_route: list):
        super().__init__()

        self.src: tuple[float, float] = tuple(default_route[:-2])
        self.dst: tuple[float, float] = tuple(default_route[-2:])


class GridBasedPopulation(GridMixin, Population):
    """Make initial population for genetic algorithm based on a grid and associated cost values

    Notes on the inheritance:
     - GridMixin has to be inherited first because Sampling isn't designed for multiple inheritance
     - implemented approach: https://stackoverflow.com/a/50465583, scenario 2
     - call `print(GridBasedPopulation.mro())` to see the method resolution order
    """

    def __init__(self, default_route, grid, constraints_list, departure_time):
        super().__init__(default_route=default_route, grid=grid)

        self.constraints_list = constraints_list
        self.departure_time = departure_time

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
                print(f"Population failed {i}")

            # match first and last points to src and dst
            routes[i][0] = np.array([
                self.src, *route[1:-1], self.dst])

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

    def __init__(self, default_route, routes_dir: str):
        super().__init__(default_route=default_route)

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
            path = os.path.join(self.routes_dir, f"route_{i+1}.json")

            if not os.path.exists(path):
                logger.warning(
                    f"{path} not found. Using Great Circle route instead.")
                route = utils.great_circle_route(self.src, self.dst)
            else:
                route = utils.route_from_geojson_file(path)

            assert np.array_equal(route[0], self.src), "Route not starting at source"
            assert np.array_equal(route[-1], self.dst), "Route not ending at destination"

            routes[i, 0] = np.array(route)
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
        lat_res, lon_res = 10, 10
        wave_height = wt.ds.VHM0.isel(time=0)
        wave_height = wave_height[::lat_res, ::lon_res]

        # population
        population_type = config.GENETIC_POPULATION_TYPE

        match population_type:
            case "grid_based":
                return GridBasedPopulation(
                    default_route=config.DEFAULT_ROUTE,
                    constraints_list=constraints_list,
                    departure_time=config.DEPARTURE_TIME,
                    grid=wave_height,)

            case "from_geojson":
                return FromGeojsonPopulation(
                    default_route=config.DEFAULT_ROUTE,
                    routes_dir=config.GENETIC_POPULATION_PATH,)

            case _:
                logger.error(f"Population type invalid: {population_type}")
                raise ValueError(population_type)
