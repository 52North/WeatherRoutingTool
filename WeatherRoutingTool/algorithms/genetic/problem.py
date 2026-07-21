import datetime
import logging

import numpy as np
from astropy import units as u
from pymoo.core.problem import ElementwiseProblem

from WeatherRoutingTool.algorithms.data_utils import get_speed_from_arrival_time
from WeatherRoutingTool.algorithms.genetic.utils import get_constraints
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
import WeatherRoutingTool.algorithms.genetic.utils as utils

logger = logging.getLogger('WRT.Genetic')


class RoutingProblem(ElementwiseProblem):
    """
    Definition of the Weather Routing Problem.

    This class defines the optimization problem for finding the best weather-dependent
    route using the pymoo framework. It handles the evaluation of fuel consumption,
    arrival time accuracy, and navigational constraints.

    :param departure_time: The time of departure.
    :type departure_time: datetime.datetime
    :param arrival_time: The desired time of arrival.
    :type arrival_time: datetime.datetime
    :param boat: Boat object for calculating fuel and power consumption.
    :type boat: Boat
    :param boat_speed: Boat speed. Only used to set self.boat_speed_from_arrival_time.
    :type boat_speed: float
    :param constraint_list: List of constraints to be checked.
    :type constraint_list: ConstraintsList
    :param objectives: dictionary of objective names and respective user weights.
    :type objectives: dict

    """

    def __init__(self,
                 departure_time: datetime.datetime,
                 arrival_time: datetime.datetime,
                 boat: Boat,
                 boat_speed: float,
                 constraint_list: ConstraintsList,
                 objectives: dict,
                 symmetric_time_objective: bool
                 ):
        n_constr = 1

        super().__init__(
            n_var=1,
            n_obj=len(objectives),
            n_constr=n_constr

        )
        self.boat = boat
        self.constraint_list = constraint_list
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.boat_speed_from_arrival_time = False
        self.objectives = objectives
        self.symmetric_time_objective = symmetric_time_objective
        # maximum allowed delay (minutes) when 'arrival_time' is an objective
        self.max_delay_minutes = 60

        if boat_speed is None:
            self.boat_speed_from_arrival_time = True

    def get_objectives(self, obj_dict: dict) -> list:
        """
        Convert dictionary of objective values into a flat list of objective values for pymoo.

        The order of the returned values is fixed (arrival_time before fuel_consumption) and must match the
        column order expected by the MCDM selection step.

        :param obj_dict: Dictionary containing calculated metrics like 'time_obj' and 'fuel_sum'.
        :type obj_dict: dict
        :return: A list of objective values, one entry per active objective.
        :rtype: list
        :raises ValueError: If no valid objectives are specified or found in the dictionary.
        """
        objective_keys = list(self.objectives.keys())
        objs = []
        if "arrival_time" in objective_keys:
            objs.append(obj_dict["time_obj"])
        if "fuel_consumption" in objective_keys:
            objs.append(obj_dict["fuel_sum"].value)

        if not objs:
            raise ValueError('Please specify an objective for the genetic algorithm.')

        return objs

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Overridden function for population evaluation.

        :param x: numpy matrix with shape (rows: number of solutions/individuals, columns: number of design variables)
        :type x: np.ndarray
        :param out:
            out['F']: function values, vector of length of number of solutions
            out['G']: constraints
        :type out: dict
        """

        # logger.debug(f"RoutingProblem._evaluate: type(x)={type(x)}, x.shape={x.shape}, x={x}")
        obj_dict = self.get_power(x[0])
        constraints = utils.get_constraints(x[0], self.constraint_list, obj_dict["start_times"])
        constraint_values = [constraints]
        # if "arrival_time" in self.objectives.keys():
        # pymoo convention: g <= 0 is feasible, g > 0 is a violation. The boat is not allowed to be late
        # by more than max_delay_minutes; being early (negative delay) is always feasible.
        # constraint_values.append(obj_dict["delay"] - self.max_delay_minutes)
        out['F'] = self.get_objectives(obj_dict)
        out['G'] = np.column_stack(constraint_values)

    def get_power(self, route: np.array) -> dict:
        """
        Calculate objective values for fuel consumption and arrival-time accuracy for a specific route.

        This method extracts speed data, calculates weather-dependent ship parameters, and determines the deviation
        from the target arrival time.

        :param route: A 2D numpy array where columns represent [latitude, longitude, speed].
        :type route: np.ndarray
        :return: A dictionary containing the total fuel consumption ('fuel_sum'), further ship parameters
          ('shipparams'), and the objective value for the arrival-time accuracy.
        :rtype: dict
        """
        debug = False

        bs = route[:, 2]
        bs = bs[:-1] * u.meter / u.second
        fuel_obj = None
        time_obj = None
        delay = None

        if self.boat_speed_from_arrival_time:
            bs = get_speed_from_arrival_time(
                lons=route[:, 1],
                lats=route[:, 0],
                departure_time=self.departure_time,
                arrival_time=self.arrival_time,
            )
            bs = np.full(route[:, 1].shape[0] - 1, bs) * u.meter / u.second

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

        if "fuel_consumption" in self.objectives.keys():
            fuel = shipparams.get_fuel_rate()
            fuel = fuel * route_dict['travel_times']
            fuel_spread = np.max(fuel) - np.min(fuel)
            fuel_obj = np.sum(fuel)

            if debug:
                print('max fuel: ', np.max(fuel))
                print('min fuel: ', np.min(fuel))
                print('fuel max spread: ', fuel_spread)
                print('fuel obj: ', fuel_obj)

                print('last start_time: ', route_dict['start_times'][-1])
                print('last travel time: ', route_dict['travel_times'][-1].value)

        if "arrival_time" in self.objectives.keys():
            real_arrival_time = route_dict['start_times'][-1] + datetime.timedelta(
                seconds=route_dict['travel_times'][-1].value)
            # signed time difference in minutes: positive means the boat arrives earlier than planned
            time_diff = (self.arrival_time - real_arrival_time).total_seconds() / 60

            # penalising delay
            if time_diff > -1 and time_diff < 0:
                time_diff = -1
            time_obj = 1. / (120 ** 4) * time_diff * time_diff * time_diff * time_diff

            # penalising being early
            if time_diff > 0 and not self.symmetric_time_objective:
                if time_diff < 1:
                    time_diff = 1
                time_obj = 100. / (120 ** 4) * time_diff * time_diff

            # penalise deviations by more than 120 min
            abs_time_diff = np.abs(time_diff)
            if abs_time_diff > 120:
                time_obj = 1000

            if debug:
                print('departure time: ', self.departure_time)
                print('planned arrival time:', self.arrival_time)

                print('real arrival time: ', real_arrival_time)
                print('time_diff: ', time_diff)
                print('time obj.: ', time_obj)

        return {"fuel_sum": fuel_obj, "shipparams": shipparams, "time_obj": time_obj, "delay": delay,
                "start_times": route_dict["start_times"]}
