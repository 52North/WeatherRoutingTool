import logging
import os
import os.path
from math import ceil

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.sampling import Sampling
from skimage.graph import route_through_array

from WeatherRoutingTool.weather import WeatherCond
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.data_utils import GridMixin
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory
from WeatherRoutingTool.algorithms.gcrslider import GcrSliderAlgorithm

logger = logging.getLogger("WRT.genetic.population")

geod = Geodesic.WGS84


class Population(Sampling):
    """Base Class for generating the initial population."""

    def __init__(self, default_route: list, constraints_list: list, pop_size: int):
        super().__init__()

        self.constraints_list = constraints_list
        self.n_constrained_routes = 0
        self.pop_size = pop_size

        self.src: tuple[float, float] = tuple(default_route[:-2])
        self.dst: tuple[float, float] = tuple(default_route[-2:])

    def _do(self, problem, n_samples, **kw):
        X = self.generate(problem, n_samples, **kw)

        for rt, in X:
            assert tuple(rt[0]) == self.src, "Source waypoint not matching"
            assert tuple(rt[-1]) == self.dst, "Destination waypoint not matching"

        self.X = X
        return self.X

    def generate(self, problem, n_samples, **kw):
        raise NotImplementedError("Has to be implemented by child class!")

    def check_validity(self, routes):
        """
        Check whether the waypoints of the routes violate constraints. Raise a warning for each single route that
        violates constraints. Raise an error, if more than 50% of all routes violate constraints.

        :param routes: array of routes for initial population
        """

        logger.debug("Validating generated routes")

        for i, route in enumerate(routes):
            constraints = utils.get_constraints(route[0], self.constraints_list)

            if constraints:
                logger.warning(f"Initial Route route_{i + 1} is constrained.")
                self.n_constrained_routes += 1

        percentage_constrained = self.n_constrained_routes / self.pop_size

        assert (percentage_constrained < 0.5), (
            f"{self.n_constrained_routes} / {self.pop_size} constrained — "
            "More than 50% of the initial routes are constrained")


class GridBasedPopulation(GridMixin, Population):
    """Make initial population for genetic algorithm based on a grid and associated cost values

    Notes on the inheritance:
     - GridMixin has to be inherited first because Sampling isn't designed for multiple inheritance
     - implemented approach: https://stackoverflow.com/a/50465583, scenario 2
     - call `print(GridBasedPopulation.mro())` to see the method resolution order
    """

    def __init__(self, default_route, grid, constraints_list, pop_size):
        super().__init__(default_route=default_route, grid=grid, constraints_list=constraints_list, pop_size=pop_size)

        # update nan_mask with constraints_list
        # ----------
        nan_mask = np.isnan(self.grid)

        xs, ys = np.where(~nan_mask)
        fs = np.stack([grid.latitude[xs], grid.longitude[ys]], axis=1)

        rf = fs[
            np.where(utils.get_constraints_array(fs, constraints_list))]

        for lat, lon in rf:
            (latindex,), = np.where(lat == grid.latitude.values)
            (lonindex,), = np.where(lon == grid.longitude.values)

            nan_mask[latindex, lonindex] = True
        self.grid.data[nan_mask] = np.nan

        # ----------
        self.var_type = np.float64

    def generate(self, problem, n_samples, **kw):
        X = np.full((n_samples, 1), None, dtype=object)

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

            # match first and last points to src and dst
            X[i, 0] = np.array([
                self.src, *route[1:-1], self.dst])

        return X


class FromGeojsonPopulation(Population):
    """Genetic population from a directory of routes

    NOTE:
        Routes are expected to be named in the following format: `route_{1..N}.json`

        example: `route_1.json, route_2.json, route_3.json, ...`

    :param routes_dir: Directory pointing to the routes folder
    :type routes_dir: str
    """

    def __init__(self, routes_dir: str, default_route, constraints_list, pop_size):
        super().__init__(default_route=default_route, constraints_list=constraints_list, pop_size=pop_size)

        if not os.path.exists(routes_dir) or not os.path.isdir(routes_dir):
            raise FileNotFoundError("Routes directory not found")
        self.routes_dir: str = routes_dir

    def generate(self, problem, n_samples, **kw):
        logger.debug(f"Population from geojson routes: {self.routes_dir}")

        # routes are expected to be named in the following format:
        # route_{1..N}.json
        # example: route_1.json, route_2.json, route_3.json, ...

        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            path = os.path.join(self.routes_dir, f"route_{i + 1}.json")

            if not os.path.exists(path):
                raise ValueError("The number of available routes for the initial population does not match the "
                                 "population size.")
            else:
                route = utils.route_from_geojson_file(path)

            assert np.array_equal(route[0], self.src), "Route not starting at source"
            assert np.array_equal(route[-1], self.dst), "Route not ending at destination"

            X[i, 0] = np.array(route)

        return X


class IsoFuelPopulation(Population):
    """Population generation using the IsoFuel algorithm

    Produces initial routes using the IsoFuel algorithm's
    ISOCHRONE_NUMBER_OF_STEPS configuration. If the number of generated routes
    is lesser than the expected n_samples number of individuals, the last
    produced route is repeated until the required number of individuals are met
    """

    def __init__(self, config: Config, boat: Boat, default_route, constraints_list, pop_size):
        super().__init__(
            default_route=default_route,
            constraints_list=constraints_list,
            pop_size=pop_size, )

        self.departure_time = config.DEPARTURE_TIME

        self.patcher = PatchFactory.get_patcher(config=config, patch_type="isofuel_multiple_routes",
                                                application="initial population")

    def generate(self, problem, n_samples, **kw):
        routes = self.patcher.patch(self.src, self.dst, self.departure_time)

        X = np.full((n_samples, 1), None, dtype=object)

        for i, rt in enumerate(routes):
            X[i, 0] = np.array([self.src, *rt[1:-1], self.dst])

        # fallback: fill all other individuals with the same population as the last one
        for j in range(i + 1, n_samples):
            X[j, 0] = np.copy(X[j - 1, 0])
        return X


class GcrSliderPopulation(Population):

    def __init__(self, config: Config, default_route, constraints_list, pop_size):
        super().__init__(default_route=default_route, constraints_list=constraints_list, pop_size=pop_size)
        self.algo = GcrSliderAlgorithm(config)

    def generate(self, problem, n_samples, **kw):
        """
        Create an initial population with good variety of routes by calculating routes using the GCR Slider Algorithm
        with different waypoints. Each route is created with only one waypoint.
        Waypoints are created in two steps:
        1) Find the point(s) which splits the segment(s) into two new segments of equal distance.
           These points will be located at fractions of 0.5 (1st split), 0.25|0.75 (2nd split),
           0.125|0.625|0.375|0.875 (3rd split) and so forth.
        2) For each point move the point in orthogonal direction (clockwise and counter-clockwise). Increase the
           distance used to move the point incrementally.
        """
        # FIXME: how to handle already existing waypoints specified for the genetic algorithm?
        route = self.create_route()
        routes = []
        if route is not None:
            routes.append(route)
            logger.info(f"Found {len(routes)} of {n_samples} routes for initial population.")
        line = geod.InverseLine(self.algo.start[0], self.algo.start[1], self.algo.finish[0], self.algo.finish[1])
        wpt_increment_max = 0.5 * line.s13
        wpt_increment = 0.05 * line.s13
        wpt_increment_steps_max = ceil(wpt_increment_max/wpt_increment)

        element = 1
        clockwise = True
        wpt_increment_step = 1
        while len(routes) < n_samples:
            dist_fraction = self.van_der_corput_sequence(element)
            g = line.Position(dist_fraction*line.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            dist_orthogonal = wpt_increment_step * wpt_increment
            lat, lon = self.algo.move_point_orthogonally(g, dist_orthogonal, clockwise=clockwise)
            if not self.algo.is_land(lat, lon):
                route = self.create_route(lat, lon)
                if route is not None:
                    routes.append(route)
                    logger.info(f"Found {len(routes)} of {n_samples} routes for initial population.")
                else:
                    logger.info(f"Could not find a new route (dist_fraction={dist_fraction}, "
                                f"dist_orthogonal={dist_orthogonal}, clockwise={clockwise}).")
            if not clockwise:
                wpt_increment_step += 1
            clockwise = not clockwise
            if wpt_increment_step > wpt_increment_steps_max:
                element += 1
                wpt_increment_step = 1
            if element > 7:
                break

        X = np.full((n_samples, 1), None, dtype=object)
        for i, rt in enumerate(routes):
            X[i, 0] = np.array([self.src, *rt[1:-1], self.dst])

        # fallback: fill all other individuals with the same population as the last one
        for j in range(i + 1, n_samples):
            X[j, 0] = np.copy(X[j - 1, 0])
        return X

    def create_route(self, lat: float = None, lon: float = None):
        """
        :param lat: latitude of the waypoint
        :type lat: float
        :param lon: longitude of the waypoint
        :type lon: float
        """
        route = None
        if lat and lon:
            self.algo.waypoints = [[lat, lon]]
        try:
            route, _ = self.algo.execute()
            # import uuid
            # filename = f"{str(uuid.uuid4())}.geojson"
            # route.write_to_geojson(filename)
            route = [[route.lats_per_step[i], route.lons_per_step[i]] for i in range(0, len(route.lats_per_step))]
            route = np.array(route)
        except Exception:
            pass
        return route

    @staticmethod
    def van_der_corput_sequence(n: int, base: int = 2):
        """
        Returns the n-th element of the van der Corput sequence in a given base.
        Can be used to produce the sequence 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, ...
        Based on https://en.wikipedia.org/wiki/Van_der_Corput_sequence.
        """
        result = 0.0
        result_increment = 1 / base
        while n > 0:
            least_significant_digit = n % base
            result += least_significant_digit * result_increment
            n //= base
            result_increment /= base
        return result


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
                    grid=wave_height,
                    default_route=config.DEFAULT_ROUTE,
                    constraints_list=constraints_list,
                    pop_size=population_size, )

            case "isofuel":
                return IsoFuelPopulation(
                    config=config,
                    boat=boat,
                    default_route=config.DEFAULT_ROUTE,
                    constraints_list=constraints_list,
                    pop_size=population_size, )

            case "from_geojson":
                return FromGeojsonPopulation(
                    routes_dir=config.GENETIC_POPULATION_PATH,
                    default_route=config.DEFAULT_ROUTE,
                    constraints_list=constraints_list,
                    pop_size=population_size, )

            case "gcrslider":
                return GcrSliderPopulation(
                    config=config,
                    default_route=config.DEFAULT_ROUTE,
                    constraints_list=constraints_list,
                    pop_size=population_size, )

            case _:
                logger.error(f"Population type invalid: {population_type}")
                raise ValueError(population_type)
