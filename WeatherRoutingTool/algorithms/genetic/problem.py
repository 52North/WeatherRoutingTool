import datetime
import logging

import astropy.units as u
import numpy as np
from pymoo.core.problem import ElementwiseProblem

from WeatherRoutingTool.routeparams import RouteParams
import WeatherRoutingTool.algorithms.genetic.utils as utils

logger = logging.getLogger('WRT.Genetic')


class RoutingProblem(ElementwiseProblem):
    """GA definition of the Weather Routing Problem"""

    def __init__(self, departure_time, arrival_time, boat, boat_speed, constraint_list):
        super().__init__(n_var=1, n_obj=2, n_constr=1)
        self.boat = boat
        self.constraint_list = constraint_list
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.boat_speed = boat_speed
        self.boat_speed_from_arrival_time = False
        if boat_speed.value == -99.:
            self.boat_speed_from_arrival_time = True

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
        obj_dict = self.get_power(x[0])
        constraints = utils.get_constraints(x[0], self.constraint_list)
        # print(costs.shape)
        out['F'] = [np.column_stack([obj_dict["fuel_sum"]]),np.column_stack([obj_dict["time_obj"]])]
        out['G'] = np.column_stack([constraints])

    def get_power(self, route):
        debug = True

        bs = self.boat_speed

        #if self.boat_speed_from_arrival_time:
        #    bs = utils.get_speed_from_arrival_time(
        #        lons=route[:, 1],
        #        lats=route[:, 0],
        #        departure_time=self.departure_time,
        #        arrival_time=self.arrival_time,
        #    )

        route_dict = RouteParams.get_per_waypoint_coords(
            route[:, 1],
            route[:, 0],
            self.departure_time,
            bs, )

        shipparams = self.boat.get_ship_parameters(
            courses=route_dict['courses'],
            lats=route_dict['start_lats'],
            lons=route_dict['start_lons'],
            time=route_dict['start_times'],
            speed=bs,
        )

        fuel = shipparams.get_fuel_rate()
        fuel = fuel * route_dict['travel_times']
        fuel_spread = np.max(fuel)-np.min(fuel)
        if debug:
            print('max fuel: ', np.max(fuel))
            print('min fuel: ', np.min(fuel))
            print('fuel max spread: ', fuel_spread)

            print('last start_time: ', route_dict['start_times'][-1])
            print('last travel time: ', route_dict['travel_times'][-1].value)

        real_arrival_time = route_dict['start_times'][-1] + datetime.timedelta(seconds=route_dict['travel_times'][-1].value)
        time_diff = np.abs(self.arrival_time - real_arrival_time).total_seconds()
        beta = fuel_spread.value/(1800 * 1800 * 1800 * 1800)
        time_obj = beta * time_diff * time_diff * time_diff * time_diff
        if debug:
            print('departure time: ', self.departure_time)
            print('planned arrival time:', self.arrival_time)

            print('real arrival time: ', real_arrival_time)
            print('time_diff: ', time_diff)
            print('beta: ', beta)
            print('time obj.: ', time_obj)

        return {"fuel_sum": np.sum(fuel), "shipparams": shipparams, "time_obj": time_obj}


