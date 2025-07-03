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
