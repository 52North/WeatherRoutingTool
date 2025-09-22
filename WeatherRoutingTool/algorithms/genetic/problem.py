import logging

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from WeatherRoutingTool.routeparams import RouteParams
import WeatherRoutingTool.algorithms.genetic.utils as utils

logger = logging.getLogger('WRT.Genetic')


class RoutingProblem(ElementwiseProblem):
    """GA definition of the Weather Routing Problem"""

    def __init__(self, departure_time, boat, constraint_list):
        super().__init__(n_var=1, n_obj=1, n_constr=1)
        self.boat = boat
        self.constraint_list = constraint_list
        self.departure_time = departure_time

    def _evaluate(self, x, out, *args, **kwargs):
        """Overridden function for population evaluation

        :param x: numpy matrix with shape (rows: number of solutions/individuals, columns: number of design variables)
        :type x: numpy matrix
        :param out:
            out['F']: function values, vector of length of number of solutions
            out['G']: constraints
        :type out: dict
        :param *args:
        :param **kwargs:
        """

        # logger.debug(f"RoutingProblem._evaluate: type(x)={type(x)}, x.shape={x.shape}, x={x}")
        fuel, _ = self.get_power(x[0])
        constraints = utils.get_constraints(x[0], self.constraint_list)
        # print(costs.shape)
        out['F'] = np.column_stack([fuel])
        out['G'] = np.column_stack([constraints])

    def get_power(self, route):
        route_dict = RouteParams.get_per_waypoint_coords(
            route[:, 1],
            route[:, 0],
            self.departure_time,
            self.boat.get_boat_speed(), )

        shipparams = self.boat.get_ship_parameters(
            route_dict['courses'],
            route_dict['start_lats'],
            route_dict['start_lons'],
            route_dict['start_times'], )

        fuel = shipparams.get_fuel_rate()
        fuel = (fuel / 3600) * route_dict['travel_times']
        return np.sum(fuel), shipparams
