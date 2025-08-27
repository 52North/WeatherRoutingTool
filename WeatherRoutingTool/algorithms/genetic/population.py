import json
import logging
import os
from math import ceil
from pathlib import Path
from datetime import datetime

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.sampling import Sampling
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
        :param distance: distance in m, defaults to 100000
        :type distance: int or float
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
