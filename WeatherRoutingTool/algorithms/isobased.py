from datetime import timedelta
import logging

import numpy as np
import pandas as pd
from geovectorslib import geod
from scipy.stats import binned_statistic
from astropy import units as u

import WeatherRoutingTool.utils.unit_conversion as units
from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger('WRT.Isobased')


class RoutingStep:
    """
    Class for storing parameters that characterise a single routing step for IsoBased algorithms.

    :param lats: latitude values for the start and arrival point of the routing step
    :type: np.ndarray, rows: latitudes for start (index = 0) and arrival points (index=1), columns: latitudes for\
        different routes
    :param lons: longitude values for the start and arrival point of the routing step
    :type: np.ndarray, rows: longitudes for start (index = 0) and arrival points (index=1), columns: longitudes for\
        different routes
    :param courses: courses set at the starting point of the routing step
    :type: np.ndarray
    :param departure_time: departure times of all routes from the starting point
    :type: np.ndarray
    :param delta_time: travel time
    :type: timedelta
    :param delta_fuel: fuel consumption
    :type: float
    :param is_constrained: information on constraint violations
    :type: bool
    """

    lats: np.ndarray
    lons: np.ndarray
    courses: np.ndarray
    departure_time: np.ndarray

    delta_time: timedelta
    delta_fuel: float
    delta_dist: float
    is_constrained: np.ndarray

    def __init__(self):
        self.delta_time = None
        self.delta_fuel = None
        self.delta_dist = None
        self.is_constrained = None

        self.lats = np.array([[None]])
        self.lons = np.array([[None]])
        self.courses = np.array([None])
        self.departure_time = np.array([None])

    def update_delta_variables(self, delta_fuel: float, delta_time: timedelta, delta_dist: float) -> None:
        """Update variables for fuel consumption, travel time and travel distance."""
        self.delta_fuel = delta_fuel
        self.delta_time = delta_time
        self.delta_dist = delta_dist

    def _update_single_var(self, old_var: np.ndarray, added_var: np.ndarray, position: int) -> np.ndarray:
        var_array = np.split(old_var, 2)
        new_var = None
        if position == 0:
            var_keep = var_array[1]
            new_var = np.vstack((added_var, var_keep))
        elif position == 1:
            var_keep = var_array[0]
            new_var = np.vstack((var_keep, added_var))
        return new_var

    def _update_step(
            self,
            position: int,
            lats: np.ndarray,
            lons: np.ndarray,
            courses: np.ndarray,
            time: np.ndarray
    ) -> None:
        """
        Update class variables while routing is ongoing.

        The shape of the arguments has to match the shape that has been chosen for the initialisation.

        :param position: 0 = departure point, 1 = arrival point
        :type: int
        :param lats: new latitude values
        :type: np.ndarray
        :param lons: new longitude values
        :type: np.ndarray
        :param courses: new courses
        :type: np.ndarray
        :param time: new departure time
        :type: np.ndrarray
        """

        self.lats = self._update_single_var(self.lats, lats, position)
        self.lons = self._update_single_var(self.lons, lons, position)
        if position == 0:
            self.departure_time = time
            self.courses = courses

    def update_start_step(self, lats: np.ndarray, lons: np.ndarray, courses: np.ndarray, time: np.ndarray) -> None:
        """
        Update class variables for the departure point while routing is ongoing.

        The shape of the arguments has to match the shape that has been chosen for the initialisation.

        :param lats: new latitude values
        :type: np.ndarray
        :param lons: new longitude values
        :type: np.ndarray
        :param courses: new courses
        :type: np.ndarray
        :param time: new departure time
        :type: np.ndrarray
        """
        self._update_step(0, lats, lons, courses, time)

    def update_end_step(self, lats: np.ndarray, lons: np.ndarray) -> None:
        """
        Update class variables for the arrival point while routing is ongoing.

        The shape of the arguments has to match the shape that has been chosen for the initialisation.

        :param lats: new latitude values
        :type: np.ndarray
        :param lons: new longitude values
        :type: np.ndarray
        """
        self._update_step(1, lats, lons, None, None)

    def print(self) -> None:
        logger.info(form.get_log_step('Departure: ', 0))
        logger.info(form.get_log_step('lats: ' + str(self.lats[0]), 1))
        logger.info(form.get_log_step('lons: ' + str(self.lons[0]), 1))
        logger.info(form.get_log_step('courses: ' + str(self.courses), 1))
        logger.info(form.get_log_step('time: ' + str(self.departure_time), 1))
        logger.info(form.get_log_step('Arrival: ', 0))
        logger.info(form.get_log_step('lats: ' + str(self.lats[1]), 1))
        logger.info(form.get_log_step('lons: ' + str(self.lons[1]), 1))
        logger.info(form.get_log_step('constraints: ' + str(self.is_constrained)))

    def init_step(self, lats_start: np.ndarray, lons_start: np.ndarray, courses: np.ndarray, time: np.ndarray) -> None:
        """
        Initialise the class object at the start of each routing step.

        The arguments initialise the variables for the starting point. The variables for the arrival point are set to
        arrays containing None. The variables for the starting point can come with any shape; the shape of all other
        arrays will be adjusted, accordingly. The array for the constraint information is initialised to be 'False' for
        all routes.

        :param lats: new latitude values
        :type: np.ndarray
        :param lons: new longitude values
        :type: np.ndarray
        :param courses: new courses
        :type: np.ndarray
        :param time: new departure time
        :type: np.ndrarray
        """
        var_shape = lats_start.shape[0]
        dummy_end = np.full(var_shape, -99)

        self.lats = np.vstack((lats_start, dummy_end))
        self.lons = np.vstack((lons_start, dummy_end))
        self.courses = courses
        self.departure_time = time

        self.is_constrained = np.full(var_shape, False)

        self.delta_time = None
        self.delta_fuel = None
        self.delta_dist = None

    def update_constraints(self, constraints: np.ndarray) -> None:
        """Update the constraint information."""
        self.is_constrained = constraints

    def get_start_point(self, coord: str = "all"):
        """
        Get the coordinates of the starting point.

        :param coord: coordinate(s) that is/are requested. Can be 'lat', 'lon', 'all. Defaults to 'all'.
        :type: str

        :return: coordinate(s) of starting point
        :rtype: float or tuple in the form of (longitudes, latitudes)
        :raises ValueError: if coord is not implemented
        """
        return self._get_point(coord, 0)

    def get_end_point(self, coord: str = "all"):
        """
        Get the coordinates of the arrival point.

        :param coord: coordinate(s) that is/are requested. Can be 'lat', 'lon', 'all. Defaults to 'all'.
        :type coord: str

        :return: coordinate(s) of arrival point
        :rtype: float or tuple in the form of (longitudes, latitudes)
        :raises ValueError: if coord is not implemented
        """
        return self._get_point(coord, 1)

    def _get_point(self, coord: str = "all", position: int = 0):
        if coord == "all":
            return (self.lons[position], self.lats[position])
        elif coord == 'lat':
            return self.lats[position]
        elif coord == 'lon':
            return self.lons[position]
        else:
            raise ValueError('RoutingSteps.get_point accepts arguments "all", "lat", "lon"')

    def get_courses(self) -> np.ndarray:
        """Get courses set at starting point."""
        return self.courses

    def get_time(self) -> np.ndarray:
        """Get departure time from starting point."""
        return self.departure_time


class IsoBasedStatus():
    """
    Class to store status and error descriptions of IsoBased algorithms.

    This class defines status and error descriptions as well as error codes. At the beginning of the routing procedure,
    the state is set to "routing" and the error to "no_error".

    :params available states: pre-defined status descriptions
    :type list[str]:
    :params available errors: pre-defined error descriptions
    :type list[str]:
    :params state: current routing state
    :type str:
    :params error: error status
    :type str:
    :params needs_further_routing: information about whether further routing steps are necessary
    :type bool:
    """

    name: str
    state: str
    error: str
    needs_further_routing: bool

    available_states: list
    available_errors: dict

    def __init__(self):
        self.available_states = [
            "routing",
            "some_reached_destination",
            "all_reached_destination",
            "reached_waypoint",
            "error"
        ]
        self.available_errors = {
            'no_error': 0,
            'pruning_error': 1,
            'out_of_routes': 2,
            'destination_not_reached': 3
        }
        self.state = "routing"
        self.error = "no_error"
        self.needs_further_routing = True

    def update_state(self, state_request: str) -> None:
        """
        Updates status description.

        :raises ValueError: if status description is not implemented
        """
        state_exists = [istate for istate in self.available_states if istate == state_request]
        if not state_exists:
            raise ValueError('Wrong state requested for Isobased routing: ' + state_request)

        self.state = state_request

    def set_error_str(self, error_str: str) -> None:
        """
        Updates error state.

        :raises ValueError: if error state is not implemented.
        """
        error_exists = [ierr for ierr in self.available_errors.keys() if ierr == error_str]
        if not error_exists:
            raise ValueError('Wrong error requested for Isobased routing: ' + error_str)

        # prevent overwriting with "pruning_error" if state is "out_of_routes"
        if self.error == "out_of_routes":
            return
        self.error = error_str
        self.update_state("error")

    def get_error_code(self) -> int:
        """Returns error code. """
        return self.available_errors[self.error]

    def print(self):
        logger.info(form.get_log_step('Routing Status Report: ', 0))
        logger.info(form.get_log_step('active state: ' + self.state, 1))
        logger.info(form.get_log_step('error state: ' + self.error, 1))
        logger.info(form.get_log_step('error code: ' + str(self.get_error_code()), 1))
        logger.info(form.get_log_step('needs further routing: ' + str(self.needs_further_routing), 1))


class IsoBased(RoutingAlg):
    """
    Base class for algorithms that are based on traveling with constant fuel/time/etc.

    The class initiates the main evaluation steps that are necessary for IsoBased algorithms. The function
    execute_routing is the core of the implementation. It iterates over individual routing steps and initiates the main
    evaluations which are:
        - define a set of route segments that is to be tested (function: define_courses_per_step)
        - estimate the fuel consumption rate at the start of the route segments based on weather conditions and\
            ship type (function: estimate_fuel_consumption, calls Ship module)
        - move the ship considering that a fixed amount of fuel/time/etc can be consumed (function: move_boat)
        - evaluate possible constraints (function: check_constraints, calls Constraints module)
        - select routes that maximise/minimise the evaluation criterion (function: pruning)
    The class also considers positive constraints like waypoints that need to be passed
    (function: check_for_positive_constraints).

    The variables that charactarise a single routing step are stored in a RoutingStep object. Error and state
    descriptions are stored in an IsoBasedStatus object. For further evaluation of the error status by the user,
    an error code is returned by the main function execute_routing.

    The history of the routing procedure is stored in a selection of np.ndarrays with dimension MxN (variables with
    suffix 'per_step') and np.ndarrays with dimension N (variables with prefix 'full') whereby 'M' corresponds to the
    current number of routing steps and 'N' corresponds to the current number of courses +1. The dimensions of these
    arrays aren't static; for every routing step, a row is added on top of the matrices (functions with prefix
    'update').

    :param ncount: total number of routing steps
    :type ncount: int
    :param count: current routing step
    :type count: int
    :param start_temp: temporary starting point considering intermediate waypoints
    :type start_temp: tuple (lat, lon)
    :param finish_temp: temporary arrival point considering intermediate waypoints
    :type finish_temp: tuple (lat, lon)
    :param grc_course_temp: course of grand circle route towards temporary arrival point
    :type gcr_course_temp: tuple

    :param lats_per_step: latitudes per routing step and test route
    :type lats_per_step: (M,N) np.ndarray N=courses+1, M=steps (M decreasing)
    :param lons_per_step: longitudes per routing step and test route
    :type lons_per_step: (M,N) np.ndarray N=courses+1, M=steps (M decreasing)
    :param course_per_step: courses per routing step and test route; angle convention: 0-360°
    :type course_per_step: (M,N) np.ndarray N=courses+1, M=steps (M decreasing)
    :param dist_per_step:  geodesic distance travelled per routing step and test route
    :type dist_per_step: (M,N) np.ndarray N=courses+1, M=steps (M decreasing)
    :param shipparams_per_step: ship parameters (fuel rate, power consumption ...) per routing step and test route
    :type shipparams_per_step: ShipParams
    :param starttime_per_step: start time for every routing step
    :type starttime_per_step: (M,N) np.ndarray (datetime object) N=courses+1, M=steps (M decreasing)
    :param absolutefuel_per_step: absolute fuel consumed for every route at certain routing step
    :type absolutefuel_per_step: (M,N) np.ndarray N=courses+1, M=steps (M decreasing)

    :param full_dist_traveled: full distance traveled for every test route
    :type full_dist_traveled: (N) np.ndarray N=courses+1
    :param full_time_traveled: full travel time for every test route
    :type full_time_traveled: (N) np.ndarray N=courses+1
    :param time: current datetime for every test route
    :type time: (N) np.ndarray N=courses+1

    :param course_segments: number of course segments in the range of -180° to 180°
    :type course_segments: int
    :param course_increments_deg: increment between different courses
    :type course_increments_deg: int
    :param prune_sector_deg_half: angular range of course that is considered for pruning (only one half, 0-180°)
    :type prune_sector_deg_half: int
    :param prune_segments: number of course bins that are used for pruning
    :type prune segments: int
    :param prune_symmetry_axis: method to define pruning symmetry axis
    :type prune_symmetry_axis: str
    :param prune_groups: method to define grouping of route segments before the pruning
    :type prune_groups: str
    :param minimisation_criterion: minimisation criterion
    :type minimisation_criterion: str

    :param desired_number_of_routes: number of routes requested for multiple-routes approach
    :type desired_number_of_routes: int
    :param current_number_of_routes: current number of routes in case of multiple-routes approach
    :type current_number_of_routes: int
    :param current_step_routes:
    :type current_step_routes: pd.DataFrame
    :param next_step_routes:
    :type next_step_routes: pd.DataFrame
    :param route_list: list of routes in case of multiple-routes approach
    :type route_list: list[RouteParams]

    :param status: container for status and error information
    :type status: IsoBasedStatus
    :param routing_step: container for variables for single routing step
    :type routing_step: RoutingStep
    """

    ncount: int  # total number of routing steps
    count: int  # current routing step

    start_temp: tuple  # temporary starting point considering intermediate waypoints
    finish_temp: tuple  # temporary arrival point considering intermediate waypoints
    gcr_course_temp: tuple  # course of grand circle route towards temporary arrival point

    # (M,N) arrays to store routing history per routing step: N=courses+1, M=steps (M decreasing)
    lats_per_step: np.ndarray  # latitudes
    lons_per_step: np.ndarray  # longitudes
    course_per_step: np.ndarray  # courses (0 - 360°)
    dist_per_step: np.ndarray  # geodesic distance
    starttime_per_step: np.ndarray  # start time: datetime object
    absolutefuel_per_step: np.ndarray  # absolute fuel

    shipparams_per_step: ShipParams  # ship parameters (fuel rate, power consumption ...)

    # (N) arrays to store routing history for full routes: N=courses
    full_dist_traveled: np.ndarray  # full geodesic distance since start
    full_time_traveled: np.ndarray  # time elapsed since start
    time: np.ndarray  # current datetime

    course_segments: int  # number of course segments in the range of -180° to 180°
    course_increments_deg: int  # increment between different variants
    prune_sector_deg_half: int  # angular range of course that is considered for pruning (only one half, 0-180°)
    prune_segments: int  # number of course bins that are used for pruning
    prune_symmetry_axis: str  # method to define pruning symmetry axis
    prune_groups: str  # method to define grouping of route segments before the pruning
    minimisation_criterion: str  # minimisation criterion

    desired_number_of_routes: int  # number of routes requested for multiple-routes approach
    current_number_of_routes: int  # current number of routes in case of multiple-routes approach
    current_step_routes: pd.DataFrame
    next_step_routes: pd.DataFrame
    route_list: list  # list of routes in case of multiple-routes approach

    status: IsoBasedStatus  # container for status and error information
    routing_step: RoutingStep  # container for variables for single routing step

    def __init__(self, config):
        super().__init__(config)

        self.ncount = config.ISOCHRONE_MAX_ROUTING_STEPS
        self.count = 0

        self.lats_per_step = np.array([[self.start[0]]])
        self.lons_per_step = np.array([[self.start[1]]])
        self.course_per_step = np.array([[0]]) * u.degree
        self.dist_per_step = np.array([[0]]) * u.meter
        self.shipparams_per_step = ShipParams.set_default_array()
        self.starttime_per_step = np.array([[self.departure_time]])
        self.absolutefuel_per_step = np.array([[0]]) * u.kg

        self.time = np.array([self.departure_time])
        self.full_time_traveled = np.array([0]) * u.s
        self.full_dist_traveled = np.array([0]) * u.m

        self.finish_temp = self.finish
        self.start_temp = self.start
        self.gcr_course_temp = self.gcr_course

        self.desired_number_of_routes = config.ISOCHRONE_NUMBER_OF_ROUTES
        self.current_number_of_routes = 0
        self.current_step_routes = pd.DataFrame()
        self.next_step_routes = pd.DataFrame()
        self.route_list = []

        self.minimisation_criterion = 'squareddist_over_disttodest'

        self.set_pruning_settings(sector_deg_half=config.ISOCHRONE_PRUNE_SECTOR_DEG_HALF,
                                  seg=config.ISOCHRONE_PRUNE_SEGMENTS, prune_groups=config.ISOCHRONE_PRUNE_GROUPS,
                                  prune_symmetry_axis=config.ISOCHRONE_PRUNE_SYMMETRY_AXIS)
        self.set_course_segments(config.ROUTER_HDGS_SEGMENTS, config.ROUTER_HDGS_INCREMENTS_DEG)
        self.set_minimisation_criterion(config.ISOCHRONE_MINIMISATION_CRITERION)

        self.path_to_route_folder = config.ROUTE_PATH

        self.status = IsoBasedStatus()
        self.routing_step = RoutingStep()

    def print_init(self):
        RoutingAlg.print_init(self)
        logger.info(form.get_log_step('pruning settings', 1))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_SECTOR_DEG_HALF: ' + str(self.prune_sector_deg_half), 2))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_SEGMENTS: ' + str(self.prune_segments), 2))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_SYMMETRY_AXIS: ' + str(self.prune_symmetry_axis), 2))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_GROUPS: ' + str(self.prune_groups), 2))
        logger.info(form.get_log_step('ISOCHRONE_MINIMISATION_CRITERION: ' + str(self.minimisation_criterion), 2))
        logger.info(form.get_log_step('ROUTER_HDGS_SEGMENTS: ' + str(self.course_segments), 2))
        logger.info(form.get_log_step('ROUTER_HDGS_INCREMENTS_DEG: ' + str(self.course_increments_deg), 2))

    def print_current_status(self):
        logger.info('PRINTING ALG SETTINGS')
        logger.info('step = ' + str(self.count))
        logger.info('start' + str(self.start))
        logger.info('finish' + str(self.finish))
        logger.info('per-step variables:')
        logger.info(form.get_log_step('lats_per_step = ' + str(self.lats_per_step)))
        logger.info(form.get_log_step('lons_per_step = ' + str(self.lons_per_step)))
        logger.info(form.get_log_step('courses = ' + str(self.course_per_step)))
        logger.info(form.get_log_step('dist_per_step = ' + str(self.dist_per_step)))
        logger.info(form.get_log_step('starttime_per_step = ' + str(self.starttime_per_step)))
        logger.info(form.get_log_step('absolut_fuel_per_step = ' + str(self.absolutefuel_per_step)))

        self.shipparams_per_step.print()

        logger.info('per-course variables')
        logger.info(form.get_log_step('time =' + str(self.time)))
        logger.info(form.get_log_step('full_dist_traveled=' + str(self.full_dist_traveled)))
        logger.info(form.get_log_step('full_time_traveled = ' + str(self.full_time_traveled)))

    def print_shape(self):
        logger.info('PRINTING SHAPE')
        logger.info('per-step variables:')
        logger.info(form.get_log_step('lats_per_step = ' + str(self.lats_per_step.shape)))
        logger.info(form.get_log_step('lons_per_step = ' + str(self.lons_per_step.shape)))
        logger.info(form.get_log_step('courses = ' + str(self.course_per_step.shape)))
        logger.info(form.get_log_step('dist_per_step = ' + str(self.dist_per_step.shape)))
        logger.info(form.get_log_step('absolute_fuel_per_step = ' + str(self.absolutefuel_per_step.shape)))

        self.shipparams_per_step.print_shape()

        logger.info('per-course variables:')
        logger.info(form.get_log_step('time =' + str(self.time.shape)))
        logger.info(form.get_log_step('full_dist_traveled = ' + str(self.full_dist_traveled.shape)))
        logger.info(form.get_log_step('full_time_traveled = ' + str(self.full_time_traveled.shape)))

    def current_position(self):
        logger.info('CURRENT POSITION')
        logger.info('lats = ', self.routing_step.get_start_point('lat'))
        logger.info('lons = ', self.routing_step.get_start_point('lon'))
        logger.info('course = ', self.routing_step.get_start_point('courses'))
        logger.info('full_time_traveled = ', self.full_time_traveled)

    def define_courses(self):
        """
        Initialise variables that store the routing history for the next routing step.

        All variables that store the routing history are extended to match the dimension M = N_routes x course_segments.
        Variables for single coordinate pairs are repeated for different course segments. The routing_step object is
        initialised for the current routing step.
        """

        # branch out for multiple courses
        nof_input_routes = self.lats_per_step.shape[1]

        new_finish_one = np.repeat(self.finish_temp[0], nof_input_routes)
        new_finish_two = np.repeat(self.finish_temp[1], nof_input_routes)

        new_course = geod.inverse(self.lats_per_step[0], self.lons_per_step[0], new_finish_one, new_finish_two)

        self.lats_per_step = np.repeat(self.lats_per_step, self.course_segments + 1, axis=1)
        self.lons_per_step = np.repeat(self.lons_per_step, self.course_segments + 1, axis=1)
        self.dist_per_step = np.repeat(self.dist_per_step, self.course_segments + 1, axis=1)
        self.course_per_step = np.repeat(self.course_per_step, self.course_segments + 1, axis=1)
        self.starttime_per_step = np.repeat(self.starttime_per_step, self.course_segments + 1, axis=1)
        self.absolutefuel_per_step = np.repeat(self.absolutefuel_per_step, self.course_segments + 1, axis=1)

        self.shipparams_per_step.define_courses(self.course_segments)

        self.full_time_traveled = np.repeat(self.full_time_traveled, self.course_segments + 1, axis=0)
        self.full_dist_traveled = np.repeat(self.full_dist_traveled, self.course_segments + 1, axis=0)
        self.time = np.repeat(self.time, self.course_segments + 1, axis=0)
        self.check_course_def()

        # determine new courses - centered around gcrs X0 -> X_prev_step
        delta_hdgs = np.linspace(-self.course_segments / 2 * self.course_increments_deg,
                                 +self.course_segments / 2 * self.course_increments_deg, self.course_segments + 1)
        delta_hdgs = np.tile(delta_hdgs, nof_input_routes)

        current_course = new_course['azi1'] * u.degree  # center courses around gcr
        current_course = np.repeat(current_course, self.course_segments + 1)
        current_course = current_course - delta_hdgs
        current_course = units.cut_angles(current_course)

        self.routing_step.init_step(
            lats_start=self.lats_per_step[0],
            lons_start=self.lons_per_step[0],
            courses=current_course,
            time=self.starttime_per_step[0]
        )

    def define_initial_variants(self):
        pass

    def execute_routing(
            self, boat: Boat,
            wt: WeatherCond,
            constraints_list: ConstraintsList,
            verbose: bool = False,
            patch_count: int = 0
    ):
        """
        Core function for the initialisation of important evaluations of IsoBased algorithms.

        The function iterates over individual routing steps and initiates the main routing evaluations which are:
        - define a set of route segments that is to be tested (function: define_courses_per_step)
        - estimate the fuel consumption rate at the start of the route segments based on weather conditions and\
            ship type (function: estimate_fuel_consumption, calls Ship module)
        - move the ship considering that a fixed amount of fuel/time/etc can be consumed (function: move_boat)
        - evaluate possible constraints (function: check_constraints, calls Constraints module)
        - select routes that maximise/minimise the evaluation criterion (function: pruning)
        The class also considers positive constraints like waypoints that need to be passed
        (function: check_for_positive_constraints).

        In case of successful algorithm execution, the function returns a RouteParams object. It performs
        evaluations considering the individual routing states. It catches errors, returns an error code as well as the
        best route at the state of the routing algorithm, at which the error occurred. In case the algorithm is
        configured to find multiple routes (ISOCHRONE_NUMBER_OF_ROUTES>1), this function returns the best route while
        all routes that have been found are written to separate json files.

        :param boat: Ship object
        :type boat: Ship
        :param wt: Weather data
        :type wt: WeatherCond
        :param constraints_list: List of constraints
        :type constraints_list: ConstraintsList
        :param verbose: sets verbosity, defaults to False
        :type verbose: bool, optional
        :param patch_count: counter for calls to execute_routing to prevent figure overwriting in case of multiple calls
        :type patch_count: int, optional
        :return: calculated route
        :rtype: RouteParams
        """

        self.check_settings()
        self.check_for_positive_constraints(constraints_list)
        self.define_initial_variants()
        # start_time=time.time()
        # self.print_shape()

        # Note: self.count starts at 0
        while self.count < self.ncount:
            logger.info(form.get_line_string())
            logger.info('Step ' + str(self.count))

            self.define_courses_per_step()
            bs, ship_params = self.estimate_fuel_consumption(boat)
            self.move_boat(bs, ship_params)
            self.check_constraints(constraints_list)
            self.update(ship_params)

            # Distinguish situations where the ship reached the final destination and where it reached a waypoint
            if self.status.state == "some_reached_destination":
                self.collect_routes()
                if not self.status.needs_further_routing:
                    break

            elif self.status.state == "reached_waypoint":
                self.depart_from_waypoint(constraints_list)
                continue

            self.pruning_per_step(True)
            if (self.status.error == "pruning_error") or (self.status.error == "out_of_routes"):
                break
            self.update_fig(f'patch{patch_count}')
            self.count += 1

        route = self.terminate()
        return route, self.status.get_error_code()

    def move_boat(self, bs, ship_params):
        """
        Move boat to new position based on estimated fuel consumption (or similar).

        The travel time, fuel consumption and travel distance are estimated and stored in routing_step. Based on
        these variables, the new ship position is determined without considering constraints. If the ship can reach
        its (temporary) destination for one test route, all variables of this route are propagated to the destination.

        :param bs: boat speed
        :type bs: float
        :param ship_params: ship parameters (fuel consumption, ...)
        :type ship_params: ShipParams
        """

        debug = False
        self.routing_step.courses = units.cut_angles(self.routing_step.get_courses())
        delta_time, delta_fuel, dist = self.get_delta_variables_netCDF(ship_params, bs)
        self.routing_step.update_delta_variables(delta_fuel, delta_time, dist)
        # ToDo: remove debug variable and use logger settings instead
        if debug:
            logger.info('delta_time: ' + str(delta_time))
            logger.info('delta_fuel: ' + str(delta_fuel))
            logger.info('dist: ' + str(dist))
            logger.info('state:' + str(self.status.state))
        self.check_bearing()
        self.check_land_ahoy(ship_params, bs)

    def estimate_fuel_consumption(self, boat: Boat):
        """
        Initiate the estimation of the fuel consumption by the Ship module.

        :param boat: boat object for fuel estimation
        :type boat: Boat
        :return: boat speed, calculated ship parameters
        :rtype: float, ShipParams
        """
        bs = boat.get_boat_speed()
        bs = np.repeat(bs, (self.routing_step.get_courses().shape[0]), axis=0)

        # TODO: check whether changes on IntegrateGeneticAlgorithm should be applied here
        ship_params = boat.get_ship_parameters(
            courses=self.routing_step.get_courses(),
            lats=self.routing_step.get_start_point('lat'),
            lons=self.routing_step.get_start_point('lon'),
            time=self.routing_step.get_time(),
            speed=None,
            unique_coords=True
        )
        return bs, ship_params

    def check_constraints(self, constraint_list):
        """
        Evaluate whether the new route segments violate constraints.

        The characteristics of the new route segments are sent to the Constraints module for evaluation. The variable
        routing_step is updated accordingly.

        :param constraint_list: list of constraints
        :type constraint_list: ConstraintsList
        """

        debug = False

        is_constrained = [False for i in range(0, self.lats_per_step.shape[1])]
        if (debug):
            form.print_step('shape is_constraint before checking:' + str(len(is_constrained)), 1)

        is_constrained = constraint_list.safe_crossing(self.routing_step.get_start_point('lat'),
                                                       self.routing_step.get_start_point('lon'),
                                                       self.routing_step.get_end_point('lat'),
                                                       self.routing_step.get_end_point('lon'), self.time,
                                                       is_constrained)
        if (debug):
            form.print_step('is_constrained after checking' + str(is_constrained), 1)
        self.routing_step.update_constraints(is_constrained)

    def update(self, ship_params):
        """
        Update all variables that store the routing history for the new position considering the constraints.

        :param ship_params: ship parameters
        :type ship_params: ShipParams
        """

        self.update_position()
        self.update_time()
        self.update_fuel(ship_params.get_fuel_rate())
        self.update_shipparams(ship_params)

    def depart_from_waypoint(self, constraints_list):
        """
        Initialise variables that store the routing history when departing from an intermediate waypoint.

        :param constraints_list: list of constraints
        :type constraints_list: ConstraintsList
        """

        logger.info('Initiating pruning for intermediate waypoint at routing step' + str(self.count))
        self.final_pruning()
        self.expand_axis_for_intermediate()
        constraints_list.reached_positive()
        self.finish_temp = constraints_list.get_current_destination()
        self.start_temp = constraints_list.get_current_start()
        self.gcr_course_temp = self.calculate_gcr(self.start_temp, self.finish_temp) * u.degree
        self.status.update_state("routing")

        logger.info('Initiating routing for next segment going from ' + str(self.start_temp) + ' to ' + str(
            self.finish_temp))
        self.update_fig('p')
        self.count += 1

    def collect_routes(self):
        """
        Collect all routes if any has been propagated to destination.

        In case of standard algorithm execution (desired_number_of_routes = 1), the status is updated accordingly.

        In case the algorithm has been configured to find multiple routes (desired_number_of_routes > 1):
        - it is checked whether the requested number of routes has been found
        - all route found in the current routing step are written to file
        - the status is updated accordingly
        - the algorithm is configured to find further routes if necessary
        """
        logger.info('Initiating last step at routing step ' + str(self.count))

        if self.desired_number_of_routes == 1:
            self.status.needs_further_routing = False
            self.status.update_state("all_reached_destination")
        else:
            # TODO: delete this if unnessessary
            if self.current_number_of_routes >= self.desired_number_of_routes:
                raise ValueError("Something very strange happening here! Take a look.")
            self.find_every_route_reaching_destination()
            number_of_possible_routes = self.current_number_of_routes + self.current_step_routes.shape[0]

            # if the number of routes aimed at is larger than the number of routes reaching the distination
            # in this step, collect all routes, otherwise collect only as many as required
            if self.desired_number_of_routes <= number_of_possible_routes:
                remaining_routes = self.desired_number_of_routes - self.current_number_of_routes
                self.find_routes_reaching_destination_in_current_step(remaining_routes)
                self.status.update_state("all_reached_destination")
                self.status.needs_further_routing = False
            else:
                self.find_routes_reaching_destination_in_current_step(number_of_possible_routes)
                if self.next_step_routes.shape[0] == 0:
                    logger.warning('No routes left for execution, terminating!')
                    self.status.set_error_str('out_of_routes')
                    self.status.needs_further_routing = False
                    return

                # organise routes for next step
                self.set_next_step_routes()
                self.status.update_state('routing')

    def check_land_ahoy(self, ship_params, bs):
        """
        Check whether any of the test routes can reach the destination.

        For every test route, the travel distance of the current routing step is compared to the distance to the
        (temporary) destination. If the latter distance is smaller, this very route is propagated to the destination.

        """
        if (self.status.state == "some_reached_destination") or (self.status.state == "reached_waypoint"):
            delta_time_last_step, delta_fuel_last_step, dist_last_step = \
                self.get_delta_variables_netCDF_last_step(ship_params, bs)
            if (self.status.state == "some_reached_destination"):
                for i in range(len(self.bool_arr_reached_final)):
                    if self.bool_arr_reached_final[i]:
                        self.routing_step.delta_time[i] = delta_time_last_step[i]
                        self.routing_step.delta_fuel[i] = delta_fuel_last_step[i]
                        self.routing_step.delta_dist[i] = dist_last_step[i]
            else:
                self.routing_step.delta_time = delta_time_last_step
                self.routing_step.delta_fuel = delta_fuel_last_step
                self.routing_step.delta_dist = dist_last_step

    #        self.routing_step.update_delta_variables(delta_fuel, delta_time, dist)

    def find_every_route_reaching_destination(self):
        # ToDo: move to IsoFuel algorithm
        """
        Collect all test routes that can be propagated to the destination (multiple-routes approach).

        This function collects all routes reaching the destination in the current routing step. It
        - creates a pd.DataFrame with the latitudes, longitudes, travel distance, distance to destination and fuel
            consumption of the current routing segments
        - stores the index order of the original 'per_step' arrays for later reference as axis with name 'st_index'
        - groups the variables of the DataFrame according to matching starting points of the routing segment
        - stores *only* the best routes per branch that reach the destination in the 'current_step_routes' dataframe.
        - stores *all* routes that do not reach the destination in the 'next_step_routes' dataframe for further
          evaluation

        """

        df_current_last_step = pd.DataFrame()
        df_current_last_step['st_lat'] = self.lats_per_step[1, :]
        df_current_last_step['st_lon'] = self.lons_per_step[1, :]
        df_current_last_step['dist'] = self.current_last_step_dist.value  # pandas struggles with units
        df_current_last_step['dist_dest'] = self.current_last_step_dist_to_dest.value
        df_current_last_step['fuel'] = self.absolutefuel_per_step[0, :].value

        len_df = df_current_last_step.shape[0]

        df_current_last_step.set_index(pd.RangeIndex(start=0, stop=len_df), inplace=True)
        df_current_last_step.rename_axis('st_index', inplace=True)
        df_current_last_step = df_current_last_step.reset_index()
        df_current_last_step.set_index(['st_lat', 'st_lon'], inplace=True, drop=False)
        df_grouped_by_routes_has_same_origin = df_current_last_step.groupby(level=['st_lat', 'st_lon'])

        unique_origins = df_grouped_by_routes_has_same_origin.groups.keys()

        self.current_step_routes = pd.DataFrame()
        self.next_step_routes = pd.DataFrame()

        for unique_key in unique_origins:
            specific_route_group = df_grouped_by_routes_has_same_origin.get_group(unique_key)

            df_reaching_destination = specific_route_group[
                specific_route_group['dist'] >= specific_route_group['dist_dest']
                ]
            num_rows = df_reaching_destination.shape[0]

            if num_rows > 0:
                min_fuel = df_reaching_destination['fuel'].min()
                rows_min_fuel = df_reaching_destination[df_reaching_destination['fuel'] == min_fuel]
                # Make sure we keep only one route (ToDo: which one do we want to keep?)
                row_min_fuel = rows_min_fuel.drop_duplicates('fuel')
                self.current_step_routes = pd.concat([self.current_step_routes, row_min_fuel], ignore_index=True)
            else:
                self.next_step_routes = pd.concat([self.next_step_routes, specific_route_group], ignore_index=True)

    def find_routes_reaching_destination_in_current_step(self, remaining_routes=0):
        # ToDo: move to IsoFuel algorithm
        """
        Rank routes that reach the destination and write them to file (multiple-routes approach).

        The number of routes that are selected is specified by the variable remaining_routes. If the figure path
        is set, the routes are plotted.

        :param remaining_routes: specifies how many routes shall be selected
        :type remaining_routes: int, optional
        """

        current_step_routes_sort_by_fuel = self.current_step_routes.sort_values(by=['fuel'])
        route_df = current_step_routes_sort_by_fuel['st_index'].head(remaining_routes)

        for idxs in route_df:
            self.current_number_of_routes = self.current_number_of_routes + 1
            route_object = self.make_route_object(idxs)
            self.check_status(route_object.ship_params_per_step.get_status(), str(self.current_number_of_routes))
            self.route_list.append(route_object)
            if self.path_to_route_folder is not None:
                route_object.return_route_to_API(self.path_to_route_folder + '/' +
                                                 'route_' + str(self.current_number_of_routes) + ".json")
            if self.figure_path is not None:
                self.plot_routes(idxs)

    def make_route_object(self, idxs):
        """
        Generates RouteParams object from 'per_step' variables based on index of test route (multiple-routes approach).

        :param idxs: index of test route
        :type idxs: int
        """
        # ToDo: very similar to IsoFuel.final_pruning -> harmonize

        try:
            lats_per_step = self.lats_per_step[:, idxs]
            lons_per_step = self.lons_per_step[:, idxs]
            course_per_step = self.course_per_step[:, idxs]
            dist_per_step = self.dist_per_step[:, idxs]
            shipparams_per_step = self.shipparams_per_step.get_reduced_2D_object(idxs=idxs)

            starttime_per_step = self.starttime_per_step[:, idxs]
            time = self.time[idxs]

            lats_per_step = np.flip(lats_per_step, 0)
            lons_per_step = np.flip(lons_per_step, 0)
            course_per_step = np.flip(course_per_step, 0)
            dist_per_step = np.flip(dist_per_step, 0)
            starttime_per_step = np.flip(starttime_per_step, 0)

            shipparams_per_step.flip()

        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

        route = RouteParams(count=self.count, start=self.start,
                            finish=self.finish, gcr=self.full_dist_traveled,
                            route_type='min_time_route', time=time,
                            lats_per_step=lats_per_step,
                            lons_per_step=lons_per_step,
                            course_per_step=course_per_step,
                            dists_per_step=dist_per_step,
                            starttime_per_step=starttime_per_step,
                            ship_params_per_step=shipparams_per_step
                            )
        self.prune_groups = 'branch'
        return route

    def plot_routes(self, idxs):
        """
        Plot every complete individual route that is reaching the destination (multiple-routes approach).

        :param idxs: loop index
        :type idxs: int
        """

        fig = self.fig
        fig, ax = graphics.generate_basemap(self.fig, self.depth, self.start,
                                            self.finish)

        lats_per_step = self.lats_per_step[:, idxs]
        lons_per_step = self.lons_per_step[:, idxs]

        route, = ax.plot(lons_per_step,
                         lats_per_step, color="firebrick")

        route_ensemble = []
        route_ensemble.append(route)

        route.set_xdata(lons_per_step)
        route.set_ydata(lats_per_step)
        fig.canvas.draw()
        fig.canvas.flush_events()

        final_path = self.figure_path + '/fig' + str(
            self.count) + '_route_' + str(idxs) + '.png'
        logger.info('Save updated figure to ' + final_path)
        plt.savefig(final_path)

    def set_next_step_routes(self):
        """
        Update all arrays of test routes that need to be further processed (multiple-routes approach).
        """

        # sorting order matters here????
        idxs = self.next_step_routes['st_index']
        # Return a trimmed isochrone
        try:
            self.lats_per_step = self.lats_per_step[:, idxs]
            self.lons_per_step = self.lons_per_step[:, idxs]
            self.course_per_step = self.course_per_step[:, idxs]
            self.dist_per_step = self.dist_per_step[:, idxs]
            self.absolutefuel_per_step = self.absolutefuel_per_step[:, idxs]
            self.shipparams_per_step.select(idxs)

            self.starttime_per_step = self.starttime_per_step[:, idxs]

            self.full_dist_traveled = self.full_dist_traveled[idxs]
            self.full_time_traveled = self.full_time_traveled[idxs]
            self.time = self.time[idxs]
        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def revert_to_previous_step(self):
        """
        Revert arrays to previous routing step if all routes are constrained for current step (multiple-routes approach)
        """

        last_idx = len(self.lats_per_step)
        col = len(self.lats_per_step[0])
        self.update_fig('p')
        try:
            self.lats_per_step = self.lats_per_step[1:last_idx, :]
            self.lons_per_step = self.lons_per_step[1:last_idx, :]
            self.course_per_step = self.course_per_step[1:last_idx, :]
            self.dist_per_step = self.dist_per_step[1:last_idx, :]
            self.starttime_per_step = self.starttime_per_step[1:last_idx, :]
            self.absolutefuel_per_step = self.absolutefuel_per_step[1:last_idx, :]
            self.shipparams_per_step.get_reduced_2D_object(row_start=1,
                                                           row_end=last_idx,
                                                           col_start=0, col_end=col,
                                                           idxs=None)
            col_len = len(self.lats_per_step[0])
            self.full_dist_traveled = np.full(col_len, -99)
            self.full_time_traveled = np.full(col_len, -99)
            self.time = np.full(col_len, -99)

        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def routes_from_previous_step(self):
        """
        Collect routes for previous step if all routes are constrained for current step (multiple-routes approach).

        When all routes are constrained, unique routes until the current constrained
        routing step are found here. Then, the unique routes are written into json files
        and plotted.
        """

        df_current_last_step = pd.DataFrame()
        df_current_last_step['st_lat'] = self.lats_per_step[1, :]
        df_current_last_step['st_lon'] = self.lons_per_step[1, :]
        df_current_last_step['fuel'] = self.shipparams_per_step.get_fuel_rate()[0, :].value

        len_df = df_current_last_step.shape[0]

        df_current_last_step.set_index(pd.RangeIndex(start=0, stop=len_df),
                                       inplace=True)
        df_current_last_step.rename_axis('st_index', inplace=True)
        df_current_last_step = df_current_last_step.reset_index()
        df_current_last_step.set_index(['st_lat', 'st_lon'], inplace=True,
                                       drop=False)
        df_grouped_by_routes_has_same_origin = df_current_last_step.groupby(
            level=['st_lat', 'st_lon'])

        unique_origins = df_grouped_by_routes_has_same_origin.groups.keys()

        current_step_routes = pd.DataFrame()

        for unique_key in unique_origins:
            specific_route_group = df_grouped_by_routes_has_same_origin.get_group(
                unique_key)
            row_min_fuel = specific_route_group.drop_duplicates(subset=['fuel'])
            current_step_routes = pd.concat(
                [current_step_routes, row_min_fuel],
                ignore_index=True)
        current_step_routes_sort_by_fuel = current_step_routes.sort_values(by=['fuel'])
        route_df = current_step_routes_sort_by_fuel['st_index']

        for idx in route_df:
            route_object = self.make_route_object(idx)
            self.route_list.append(route_object)
            if self.path_to_route_folder is not None:
                route_object.return_route_to_API(self.path_to_route_folder + '/' + 'route_' + str(idx) + ".json")
            if self.figure_path is not None:
                self.plot_routes(idx)

    def update_shipparams(self, ship_params_single_step):
        """Update ShipParams object. """
        new_rpm = np.vstack((ship_params_single_step.get_rpm(), self.shipparams_per_step.get_rpm()))
        new_power = np.vstack((ship_params_single_step.get_power(), self.shipparams_per_step.get_power()))
        new_speed = np.vstack((ship_params_single_step.get_speed(), self.shipparams_per_step.get_speed()))
        new_rwind = np.vstack((ship_params_single_step.get_rwind(), self.shipparams_per_step.get_rwind()))
        new_rcalm = np.vstack((ship_params_single_step.get_rcalm(), self.shipparams_per_step.get_rcalm()))
        new_rwaves = np.vstack((ship_params_single_step.get_rwaves(), self.shipparams_per_step.get_rwaves()))
        new_rshallow = np.vstack((ship_params_single_step.get_rshallow(), self.shipparams_per_step.get_rshallow()))
        new_rroughness = np.vstack(
            (ship_params_single_step.get_rroughness(), self.shipparams_per_step.get_rroughness()))
        new_wave_height = np.vstack((ship_params_single_step.get_wave_height(),
                                     self.shipparams_per_step.get_wave_height()))
        new_wave_direction = np.vstack((ship_params_single_step.get_wave_direction(),
                                        self.shipparams_per_step.get_wave_direction()))
        new_wave_period = np.vstack((ship_params_single_step.get_wave_period(),
                                     self.shipparams_per_step.get_wave_period()))
        new_u_currents = np.vstack((ship_params_single_step.get_u_currents(),
                                    self.shipparams_per_step.get_u_currents()))
        new_v_currents = np.vstack((ship_params_single_step.get_v_currents(),
                                    self.shipparams_per_step.get_v_currents()))
        new_u_wind_speed = np.vstack((ship_params_single_step.get_u_wind_speed(),
                                      self.shipparams_per_step.get_u_wind_speed()))
        new_v_wind_speed = np.vstack((ship_params_single_step.get_v_wind_speed(),
                                      self.shipparams_per_step.get_v_wind_speed()))
        new_pressure = np.vstack((ship_params_single_step.get_pressure(), self.shipparams_per_step.get_pressure()))
        new_air_temperature = np.vstack((ship_params_single_step.get_air_temperature(),
                                         self.shipparams_per_step.get_air_temperature()))
        new_salinity = np.vstack((ship_params_single_step.get_salinity(),
                                  self.shipparams_per_step.get_salinity()))
        new_water_temperature = np.vstack((ship_params_single_step.get_water_temperature(),
                                           self.shipparams_per_step.get_water_temperature()))
        new_status = np.vstack((ship_params_single_step.get_status(), self.shipparams_per_step.get_status()))
        new_message = np.vstack((ship_params_single_step.get_message(), self.shipparams_per_step.get_message()))

        self.shipparams_per_step.set_rpm(new_rpm)
        self.shipparams_per_step.set_power(new_power)
        self.shipparams_per_step.set_speed(new_speed)
        self.shipparams_per_step.set_rwind(new_rwind)
        self.shipparams_per_step.set_rcalm(new_rcalm)
        self.shipparams_per_step.set_rwaves(new_rwaves)
        self.shipparams_per_step.set_rshallow(new_rshallow)
        self.shipparams_per_step.set_rroughness(new_rroughness)
        self.shipparams_per_step.set_wave_height(new_wave_height)
        self.shipparams_per_step.set_wave_direction(new_wave_direction)
        self.shipparams_per_step.set_wave_period(new_wave_period)
        self.shipparams_per_step.set_u_currents(new_u_currents)
        self.shipparams_per_step.set_v_currents(new_v_currents)
        self.shipparams_per_step.set_u_wind_speed(new_u_wind_speed)
        self.shipparams_per_step.set_v_wind_speed(new_v_wind_speed)
        self.shipparams_per_step.set_pressure(new_pressure)
        self.shipparams_per_step.set_air_temperature(new_air_temperature)
        self.shipparams_per_step.set_salinity(new_salinity)
        self.shipparams_per_step.set_water_temperature(new_water_temperature)
        self.shipparams_per_step.set_status(new_status)
        self.shipparams_per_step.set_message(new_message)

    def check_course_def(self):
        """ Perform sanity checks for 'per_step' variables. """
        if (not ((self.lats_per_step.shape[1] == self.lons_per_step.shape[1]) and (
                self.lats_per_step.shape[1] == self.course_per_step.shape[1]) and (
                         self.lats_per_step.shape[1] == self.dist_per_step.shape[1]))):
            raise 'define_courses: number of columns not matching!'

        if (not ((self.lats_per_step.shape[0] == self.lons_per_step.shape[0]) and (
                self.lats_per_step.shape[0] == self.course_per_step.shape[0]) and (
                         self.lats_per_step.shape[0] == self.dist_per_step.shape[0]) and (
                         self.lats_per_step.shape[0] == (self.count + 1)))):
            raise ValueError(
                'define_courses: number of rows not matching! count = ' + str(self.count) + ' lats per step ' + str(
                    self.lats_per_step.shape[0]))

    def get_pruned_indices_statistics(
            self,
            bin_stat: np.ndarray,
            bin_edges: np.ndarray,
            trim: bool
    ) -> list[int]:
        """
        Collect routes whose travel distance matches the maximum bin entries in the pruning histogram.

        :param bin_stat: bin content of the pruning histogram which is the maximum travel distance per bin
        :type bin_stat: np.ndarray
        :param bin_edges: bin edges of the pruning histogram
        :type bin_edges: np.ndarray
        :param trim: omit bins with zero bin content for the pruning
        :type trim: bool
        :return: list of indices of best routes
        :rtype: list[int]
        """

        idxs = []

        if trim:
            for i in range(len(bin_edges) - 1):
                try:
                    if (bin_stat[i] == 0):
                        continue
                    idxs.append(np.where(self.full_dist_traveled == bin_stat[i])[0][0])
                except IndexError:
                    pass
            idxs = list(set(idxs))
        else:
            for i in range(len(bin_edges) - 1):
                idxs.append(np.where(self.full_dist_traveled == bin_stat[i])[0])
            idxs = list(set([item for subl in idxs for item in subl]))

        return idxs

    def pruning(self, trim: bool, bins: np.ndarray) -> None:
        """
        Perform pruning.

        Call the methods for larger-direction based, courses-based and branch-based pruning which determine the
        indices of the routes which perform best after this routing step. Slice the arrays that store the history of
        the routing procedure such that only the best routes survive.

        :param trim: omit bins with zero bin content for the pruning, defaults to True
        :type trim: bool, optional
        :param bins: bin edges for the pruning
        :type bins: np.ndarray
        :raises ValueError: if no routes are available for the pruning
        :raises IndexError: if any array can not be sliced according to the indices that have been found
        """

        debug = False
        valid_pruning_segments = -99

        # ToDo: use logger.debug and args.debug
        if debug:
            print('binning for pruning', bins)
            print('current courses', self.routing_step.get_courses())
            print('full_dist_traveled', self.full_dist_traveled)
            print('courses per step ', self.course_per_step)

        is_pruned = False
        prune_concept = self.prune_groups
        if self.prune_groups == 'multiple_routes':
            prune_concept = self.tune_multiple_routes_prunig()

        if prune_concept == 'larger_direction':
            logger.info('Executing larger-direction-based pruning.')
            bin_stat, bin_edges, bin_number = self.larger_direction_based_pruning(bins)
            idxs = self.get_pruned_indices_statistics(bin_stat, bin_edges, trim)
            is_pruned = True
        if prune_concept == 'courses':
            logger.info('Executing courses-based pruning.')
            bin_stat, bin_edges, bin_number = self.courses_based_pruning(bins)
            idxs = self.get_pruned_indices_statistics(bin_stat, bin_edges, trim)
            is_pruned = True
        if prune_concept == 'branch':
            logger.info('Executing branch-based pruning.')
            idxs = self.branch_based_pruning()
            is_pruned = True

        if not is_pruned:
            raise ValueError('The selected pruning option is not available!')

        # ToDo: use logger.debug and args.debug
        if debug:
            print('full_dist_traveled', self.full_dist_traveled)
            print('Indexes that passed', idxs)

        valid_pruning_segments = len(idxs)
        if (valid_pruning_segments == 0):
            logger.error(' All pruning segments fully constrained for step ' + str(self.count) + '!')
            self.status.set_error_str('pruning_error')
            return
        elif (valid_pruning_segments < self.prune_segments * 0.1):
            logger.warning(' More than 90% of pruning segments constrained for step ' + str(self.count) + '!')
        elif (valid_pruning_segments < self.prune_segments * 0.5):
            logger.warning(' More than 50% of pruning segments constrained for step ' + str(self.count) + '!')

        # Return a trimmed isochrone
        try:
            self.lats_per_step = self.lats_per_step[:, idxs]
            self.lons_per_step = self.lons_per_step[:, idxs]
            self.course_per_step = self.course_per_step[:, idxs]
            self.dist_per_step = self.dist_per_step[:, idxs]
            self.absolutefuel_per_step = self.absolutefuel_per_step[:, idxs]
            self.starttime_per_step = self.starttime_per_step[:, idxs]

            self.shipparams_per_step.select(idxs)

            self.full_dist_traveled = self.full_dist_traveled[idxs]
            self.full_time_traveled = self.full_time_traveled[idxs]
            self.time = self.time[idxs]

            self.routing_step.lats = self.routing_step.lats[:, idxs]
            self.routing_step.lons = self.routing_step.lons[:, idxs]
            self.routing_step.courses = self.routing_step.courses[idxs]
            self.routing_step.departure_time = self.routing_step.departure_time[idxs]

        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def tune_multiple_routes_prunig(self) -> str:
        """
        Tune pruning group for multiple-routes approach

        :return: pruning group
        :rtype: str
        """
        prune_concept = 'branch'
        # use larger-direction-based pruning on a regular basis to get more diversity
        if (self.count % 10 == 0) or (self.count % 10 == 1) or (self.count < 5):
            prune_concept = "larger_direction"

        # prohibit larger-direction-based pruning close to destination to prevent decreasing diversity
        mean_travel_dist = np.mean(self.full_dist_traveled)
        if mean_travel_dist / self.gcr_dist > 0.8:
            prune_concept = "branch"
        return prune_concept

    def courses_based_pruning(self, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform courses-based pruning

        A histogram is filled with the maximum travel distance (argument: statistic = np.nanmax) in dependence of bins
        of all courses of the route segments in the current routing step. The bin content, bin edges and the bin numbers
        are returned.

        :param bins: bin edges of the histogram
        :type bins: np.ndarray
        :return: bin content, bin edges and bin numbers
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        bin_stat, bin_edges, bin_number = binned_statistic(self.routing_step.get_courses().value,
                                                           self.full_dist_traveled,
                                                           statistic=np.nanmax, bins=bins)
        return bin_stat, bin_edges, bin_number

    def larger_direction_based_pruning(self, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform larger-direction-based pruning.

        Define an angle referred to as 'larger direction' which is the azimuthal angle from the starting point of the
        test routes towards the current position. A histogram is filled with the maximum travel distance
        (argument: statistic = np.nanmax) in dependence of bins of the larger direction. The bin content, bin edges and
        the bin numbers are returned.

        :param bins: bin edges of the histogram
        :type bins: np.ndarray
        :return: bin content, bin edges and bin numbers
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        start_lats = np.repeat(self.start_temp[0], self.lats_per_step.shape[1])
        start_lons = np.repeat(self.start_temp[1], self.lons_per_step.shape[1])
        larger_direction = geod.inverse(start_lats, start_lons, self.lats_per_step[0], self.lons_per_step[0])
        larger_direction = larger_direction['azi1']
        bin_stat, bin_edges, bin_number = binned_statistic(larger_direction, self.full_dist_traveled,
                                                           statistic=np.nanmax, bins=bins)
        return bin_stat, bin_edges, bin_number

    def branch_based_pruning(self) -> np.ndarray:
        """
        Perform branch-based pruning.

        Group routes according to the starting points of the last routing segment, i.e. routes that originate from the
        origin in the last step form a group called "branch". The indices of the routes with maximum travel distance
        for every branch are collected in an array which is returned.

        :return: indices of routes with maximum distance per branch
        :rtype: np.ndarray
        """

        df_current_last_step = pd.DataFrame()
        df_current_last_step['st_lat'] = self.lats_per_step[1, :]
        df_current_last_step['st_lon'] = self.lons_per_step[1, :]
        df_current_last_step['dist'] = self.full_dist_traveled

        len_df = df_current_last_step.shape[0]

        df_current_last_step.set_index(pd.RangeIndex(start=0, stop=len_df), inplace=True)
        df_current_last_step.rename_axis('st_index', inplace=True)
        df_current_last_step = df_current_last_step.reset_index()
        df_current_last_step.set_index(['st_lat', 'st_lon'], inplace=True, drop=False)
        df_grouped_by_routes_has_same_origin = df_current_last_step.groupby(level=['st_lat', 'st_lon'])

        unique_origins = df_grouped_by_routes_has_same_origin.groups.keys()
        idxs = []

        for unique_key in unique_origins:
            specific_route_group = df_grouped_by_routes_has_same_origin.get_group(unique_key)

            max_dist = specific_route_group['dist'].max()
            if max_dist == 0.:
                continue

            max_dist_indxs = specific_route_group[specific_route_group['dist'] == max_dist]['st_index']
            max_dist_indxs = max_dist_indxs.values
            max_dist_indxs = max_dist_indxs.astype('int32')

            idxs.append(max_dist_indxs[0])
        return idxs

    def pruning_per_step(self, trim: bool = True) -> None:
        """
        Initiate pruning. Decide between pruning methods with different symmetry axis.
        """
        if self.prune_symmetry_axis == 'gcr':
            self.pruning_gcr_centered(trim)
        else:
            self.pruning_headings_centered(trim)

    def pruning_gcr_centered(self, trim: bool = True) -> None:
        """
        Initiate pruning with the grand circle route as the symmetry axis.

        First, the symmetry axis for the binning of the pruning is determined. To do so, it is assumed that the ship
        travels from the starting point (or last intermediate waypoint) of the routes in the direction of the azimuthal
        angle gcr_course_temp for the mean full travel distance of all routes. The azimuthal angle of the waypoint that
        is found in this way towards the destination (or next intermediate waypoint) is defined as the symmetry axis of
        the pruning i.e. the bins are centered around it. These bins are fed into the function 'pruning' for further
        evaluation.

        :param trim: omit bins with zero bin content for the pruning, defaults to True
        :type trim: bool, optional
        """

        # ToDo: use logger.debug and args.debug
        debug = False
        if debug:
            logger.info('Pruning... Pruning symmetry axis defined by gcr')

        start_lats = np.repeat(self.start_temp[0], self.lats_per_step.shape[1])
        start_lons = np.repeat(self.start_temp[1], self.lons_per_step.shape[1])
        full_travel_dist = geod.inverse(start_lats, start_lons, self.lats_per_step[0], self.lons_per_step[0])
        mean_dist = np.mean(full_travel_dist['s12'])
        gcr_point = geod.direct([self.start_temp[0]], [self.start_temp[1]], self.gcr_course_temp.value, mean_dist)

        new_course = geod.inverse(gcr_point['lat2'], gcr_point['lon2'], [self.finish_temp[0]], [self.finish_temp[1]])
        new_course['azi1'] = new_course['azi1'] * u.degree

        # ToDo: use logger.debug and args.debug
        if debug:
            print('current mean end point: (' + str(gcr_point['lat2']) + ',' + str(gcr_point['lon2']) + ')')
            print('current temporary start: ', self.start)
            print('current temporary destination: ', self.finish_temp)
            print('mean course', new_course['azi1'])

            fig = self.fig
            self.ax.remove()
            fig, self.ax = graphics.generate_basemap(fig, self.depth, self.start, self.finish)

            # plot symmetry axis and boundaries of pruning area
            symmetry_axis = geod.direct([self.start_temp[0]], [self.start_temp[1]], new_course['azi1'], 1000000)
            lower_bound = geod.direct([self.start_temp[0]], [self.start_temp[1]],
                                      new_course['azi1'] - self.prune_sector_deg_half, 1000000)
            upper_bound = geod.direct([self.start_temp[0]], [self.start_temp[1]],
                                      new_course['azi1'] + self.prune_sector_deg_half, 1000000)

            self.ax.plot([self.start_temp[1], symmetry_axis["lon2"]],
                         [self.start_temp[0], symmetry_axis["lat2"]], color="blue")
            self.ax.plot([self.start_temp[1], lower_bound["lon2"]],
                         [self.start_temp[0], lower_bound["lat2"]], color="blue")
            self.ax.plot([self.start_temp[1], upper_bound["lon2"]],
                         [self.start_temp[0], upper_bound["lat2"]], color="blue")

            if self.figure_path is not None:
                final_path = self.figure_path + '/fig' + str(self.count) + '_gcr_symmetry_axis.png'
                logger.info('Saving updated figure to ' + str(final_path))
                plt.savefig(final_path)

        # define pruning area
        azi0s = np.repeat(new_course['azi1'], self.prune_segments + 1)

        delta_hdgs = np.linspace(-self.prune_sector_deg_half, +self.prune_sector_deg_half, self.prune_segments + 1)

        bins = units.cut_angles(azi0s - delta_hdgs)
        bins = np.sort(bins)

        self.pruning(trim, bins)

    def pruning_headings_centered(self, trim: bool = True) -> None:
        """
        Initiate pruning with a symmetry axis that is determined from the spread of courses.

        First, the symmetry axis for the binning of the pruning is determined as the median of the courses of
        all route segments from the current routing step. Bins are centered around this symmetry axis and the resulting
        binning is fed into the function 'pruning' for further evaluation.

        :param trim: omit bins with zero bin content for the pruning, defaults to True
        :type trim: bool, optional
        """

        # ToDo: use logger.debug and args.debug
        debug = False
        if debug:
            print('Pruning... Pruning symmetry axis defined by median of considered courses.')

        # propagate current end points towards temporary destination
        non_zero_idxs = np.where(self.full_dist_traveled != 0)[0]
        cat_lats = self.lats_per_step[0][non_zero_idxs]
        cat_lons = self.lons_per_step[0][non_zero_idxs]
        new_finish_one = np.repeat(self.finish_temp[0], cat_lats.shape[0])
        new_finish_two = np.repeat(self.finish_temp[1], cat_lats.shape[0])

        new_course = geod.inverse(cat_lats, cat_lons, new_finish_one, new_finish_two)
        mean_course = np.median(new_course['azi1']) * u.degree

        if debug:
            print('mean course: ', mean_course)
            # plot symmetry axis and boundaries of pruning area
            symmetry_axis = geod.direct([self.start_temp[0]], [self.start_temp[1]],
                                        mean_course, 1000000)
            lower_bound = geod.direct([self.start_temp[0]], [self.start_temp[1]],
                                      mean_course - self.prune_sector_deg_half, 1000000)
            upper_bound = geod.direct([self.start_temp[0]], [self.start_temp[1]],
                                      mean_course + self.prune_sector_deg_half, 1000000)

            self.ax.plot([self.start_temp[1], symmetry_axis["lon2"]],
                         [self.start_temp[0], symmetry_axis["lat2"]], color="blue")
            self.ax.plot([self.start_temp[1], lower_bound["lon2"]],
                         [self.start_temp[0], lower_bound["lat2"]], color="blue")
            self.ax.plot([self.start_temp[1], upper_bound["lon2"]],
                         [self.start_temp[0], upper_bound["lat2"]], color="blue")

            if self.figure_path is not None:
                final_path = self.figure_path + '/fig' + str(self.count) + '_median.png'
                logger.info('Saving updated figure to ' + final_path)
                plt.savefig(final_path)

        # define pruning area
        bins = units.get_angle_bins(mean_course - self.prune_sector_deg_half,
                                    mean_course + self.prune_sector_deg_half, self.prune_segments + 1)

        bins = np.sort(bins)

        # ToDo: use logger.debug and args.debug
        if debug:
            print('bins: ', bins)

        self.pruning(trim, bins)

    def define_courses_per_step(self):
        self.define_courses()

    def set_pruning_settings(self, sector_deg_half, seg, prune_groups, prune_symmetry_axis='gcr'):
        self.prune_sector_deg_half = sector_deg_half * u.degree
        self.prune_segments = seg
        self.prune_groups = prune_groups
        self.prune_symmetry_axis = prune_symmetry_axis

    def set_minimisation_criterion(self, min_str):
        self.minimisation_criterion = min_str

    def set_course_segments(self, seg, inc):
        self.course_segments = seg
        self.course_increments_deg = inc * u.degree

    def get_current_lats(self):
        return self.lats_per_step[0, :]

    def get_current_lons(self):
        return self.lons_per_step[0, :]

    def get_current_speed(self):
        return self.speed_per_step[0]

    def get_wind_functions(self, wt):
        debug = False
        winds = wt.get_wind_function((self.get_current_lats(), self.get_current_lons()), self.time[0])
        # ToDo: use logger.debug and args.debug
        if debug:
            print('obtaining wind function for position: ', self.get_current_lats(), self.get_current_lons())
            print('time', self.time[0])
            print('winds', winds)
        return winds

    def check_settings(self):
        if (self.course_segments / 2 * self.course_increments_deg >= self.prune_sector_deg_half):
            raise ValueError(
                'Prune sector does not contain all courses. Please adjust settings. (course_segments=' + str(
                    self.course_segments) + ', course_increments_deg=' + str(
                    self.course_increments_deg) + ', prune_sector_deg_half=' + str(self.prune_sector_deg_half))
        if ((self.course_segments % 2) != 0):
            raise ValueError(
                'Please provide an even number of course segments, you chose: ' + str(self.course_segments))

        if ((self.prune_segments % 2) != 0):
            raise ValueError('Please provide an even number of prune segments, you chose: ' + str(self.prune_segments))

    def get_final_index(self):
        idx = np.argmax(self.full_dist_traveled)
        return idx

    def terminate(self, **kwargs):
        """TODO: add description

        :return: Calculated route as a RouteParams object ready to be returned to the user
        :rtype: RouteParams
        """
        self.status.print()

        if self.status.state == "routing":
            self.status.set_error_str("destination_not_reached")
            self.count -= 1

        if self.status.error == "pruning_error":
            if self.count > 0:
                self.count = self.count - 1
                self.revert_to_previous_step()

        if self.desired_number_of_routes == 1:
            # if a single route is requested, return a single route.
            self.final_pruning()
        elif self.route_list:
            # if multiple routes are requested and the list of routes is filled, return list of routes.
            self.route_list.sort(key=lambda x: x.get_full_fuel())
            return self.route_list[0]
        else:
            # if multiple routes are requested and the list of routes is emtpy, return route that minimised fuel
            # of current or previous step.
            if self.status.error == "pruning_error":
                self.routes_from_previous_step()
            self.final_pruning()

        self.check_status(self.shipparams_per_step.get_status(), 'minimum')

        super().terminate()

        self.lats_per_step = np.flip(self.lats_per_step, 0)
        self.lons_per_step = np.flip(self.lons_per_step, 0)
        self.course_per_step = np.flip(self.course_per_step, 0)
        self.dist_per_step = np.flip(self.dist_per_step, 0)
        self.starttime_per_step = np.flip(self.starttime_per_step, 0)
        self.shipparams_per_step.flip()

        route = RouteParams(count=self.count, start=self.start, finish=self.finish, gcr=self.full_dist_traveled,
                            route_type='min_time_route',
                            time=self.full_time_traveled,
                            lats_per_step=self.lats_per_step[:],
                            lons_per_step=self.lons_per_step[:],
                            course_per_step=self.course_per_step[:],
                            dists_per_step=self.dist_per_step[:],
                            starttime_per_step=self.starttime_per_step[:],
                            ship_params_per_step=self.shipparams_per_step
                            )

        return route

    def check_status(self, shipparams_per_step_status, route_name):
        success_array = []
        success_array = np.where(shipparams_per_step_status == 1)  # Status 1=OK
        if success_array == 0:
            logger.info('0% of status values of the route segments are successful for Route ' + route_name + '!')
            return
        success_percentage = (len(success_array[0]) / (len(shipparams_per_step_status) - 1)) * 100
        logger.info("{:.2f}".format(success_percentage) + '% of status values '
                                                          'of the route segments are successful for Route '
                    + route_name + '!')

    def update_time(self):
        self.full_time_traveled += self.routing_step.delta_time
        self.time += timedelta(seconds=self.routing_step.delta_time)

    def check_bearing(self):
        """
        TODO: add description
        :return:
        """

        debug = False

        dist = self.routing_step.delta_dist
        ncourses = self.get_current_lons().shape[0]
        dist_to_dest = geod.inverse(self.get_current_lats(), self.get_current_lons(),
                                    np.full(ncourses, self.finish_temp[0]), np.full(ncourses, self.finish_temp[1]))
        dist_to_dest["s12"] = dist_to_dest["s12"] * u.meter
        dist_to_dest["azi1"] = dist_to_dest["azi1"] * u.degree
        # ToDo: use logger.debug and args.debug
        if debug:
            print('dist_to_dest:', dist_to_dest['s12'])
            # print('dist traveled:', dist)

        reaching_dest = np.any(dist_to_dest['s12'] < dist)

        move = geod.direct(self.get_current_lats(), self.get_current_lons(),
                           self.routing_step.get_courses().value, dist.value)

        if reaching_dest:
            reached_final = (self.finish_temp[0] == self.finish[0]) & (self.finish_temp[1] == self.finish[1])

            if debug:
                print('reaching final:', reached_final)

            new_lat = np.full(ncourses, self.finish_temp[0])
            new_lon = np.full(ncourses, self.finish_temp[1])

            if reached_final:
                self.status.update_state('some_reached_destination')
                self.current_last_step_dist = dist.copy()
                self.current_last_step_dist_to_dest = dist_to_dest['s12']

                self.bool_arr_reached_final = dist_to_dest['s12'] < dist

                for i in range(len(self.bool_arr_reached_final)):
                    if self.bool_arr_reached_final[i]:
                        move['azi2'][i] = dist_to_dest['azi1'][i].value
                        move['lat2'][i] = new_lat[i]
                        move['lon2'][i] = new_lon[i]
            else:
                self.status.update_state('reached_waypoint')
                move['azi2'] = dist_to_dest['azi1'].value
                move['lat2'] = new_lat
                move['lon2'] = new_lon

        self.routing_step.update_end_step(lats=move['lat2'], lons=move['lon2'])

    def update_position(self):
        """
        Update the current position of the ship
        """

        debug = False
        end_step_lon = self.routing_step.get_end_point('lon')
        end_step_lat = self.routing_step.get_end_point('lat')
        dist = self.routing_step.delta_dist
        is_constrained = self.routing_step.is_constrained

        self.lats_per_step = np.vstack((end_step_lat, self.lats_per_step))
        self.lons_per_step = np.vstack((end_step_lon, self.lons_per_step))
        self.dist_per_step = np.vstack((dist, self.dist_per_step))
        self.course_per_step = np.vstack((self.routing_step.get_courses(), self.course_per_step))
        self.routing_step.update_end_step(
            lats=end_step_lat,
            lons=end_step_lon
        )

        # ToDo: use logger.debug and args.debug
        if debug:
            print('path of this step' +
                  str(end_step_lat) + str(end_step_lon))
            print('dist_per_step', self.dist_per_step)
            print('dist', dist)

        start_lats = np.repeat(self.start_temp[0], self.lats_per_step.shape[1])
        start_lons = np.repeat(self.start_temp[1], self.lons_per_step.shape[1])
        travel_dist = geod.inverse(start_lats, start_lons, end_step_lat, end_step_lon)  # calculate full distance
        end_lats = np.repeat(self.finish_temp[0], self.lats_per_step.shape[1])
        end_lons = np.repeat(self.finish_temp[1], self.lons_per_step.shape[1])
        dist_to_dest = geod.inverse(end_step_lat, end_step_lon, end_lats, end_lons)  # calculate full distance

        # traveled, azimuth of gcr connecting start and new position
        # self.current_variant = gcrs['azi1']
        # self.current_azimuth = gcrs['azi1']
        # gcrs['s12'][is_constrained] = 0
        travel_dist['s12'][is_constrained] = 0

        concatenated_distance = np.sum(self.dist_per_step, axis=0)
        concatenated_distance[is_constrained] = 0

        if np.all(dist_to_dest['s12']) > 0:
            if self.minimisation_criterion == 'squareddist_over_disttodest':
                self.full_dist_traveled = travel_dist['s12'] * travel_dist['s12'] / dist_to_dest['s12']
            if self.minimisation_criterion == 'dist':
                self.full_dist_traveled = travel_dist['s12']
        else:
            self.full_dist_traveled = travel_dist['s12']
        # ToDo: use logger.debug and args.debug
        if debug:
            print('full_dist_traveled:', self.full_dist_traveled)

    def update_fuel(self, fuel_rate):
        self.shipparams_per_step.set_fuel_rate(np.vstack((fuel_rate, self.shipparams_per_step.get_fuel_rate())))
        self.absolutefuel_per_step = np.vstack((self.routing_step.delta_fuel, self.absolutefuel_per_step))

    def get_delta_variables(self, boat, wind, bs):
        pass

    def get_delta_variables_netCDF_last_step(self, boat, wind, bs):
        pass

    def init_fig(self, water_depth, map_size, showDepth=True):
        if self.figure_path is None:
            return
        self.showDepth = showDepth
        plt.rcParams['font.size'] = graphics.get_standard('font_size')
        self.fig, self.ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        self.ax.axis('off')
        self.ax.xaxis.set_tick_params(labelsize='large')

        if (self.showDepth):
            # decrease resolution and extend of depth data to prevent memory issues when plotting
            # FIXME: double check boundary settings (set exact to trim for debugging)
            ds_depth = water_depth.depth_data.coarsen(latitude=10, longitude=10, boundary='trim').mean()
            ds_depth_coarsened = ds_depth.compute()

            self.depth = ds_depth_coarsened.where(
                (ds_depth_coarsened.latitude > map_size.lat1) & (ds_depth_coarsened.latitude < map_size.lat2) &
                (ds_depth_coarsened.longitude > map_size.lon1) & (ds_depth_coarsened.longitude < map_size.lon2) &
                (ds_depth_coarsened.z < 0), drop=True)

        self.fig, self.ax = graphics.generate_basemap(self.fig, self.depth, self.start, self.finish)

        final_path = self.figure_path + '/fig0.png'
        logger.info('Save start figure to ' + final_path)
        plt.savefig(final_path)

    def update_fig(self, status):
        if self.figure_path is None:
            return
        fig = self.fig
        self.ax.remove()
        fig, self.ax = graphics.generate_basemap(fig, self.depth, self.start, self.finish)

        latitudes = self.lats_per_step.copy()
        longitudes = self.lons_per_step.copy()

        latitudes_T = latitudes.T
        longitudes_T = longitudes.T
        # Plotting each route
        for lat_segment, lon_segment in zip(latitudes_T, longitudes_T):
            self.ax.plot(lon_segment, lat_segment, color="firebrick", linestyle='-', linewidth=0.6)
            # fig.canvas.draw()
            # fig.canvas.flush_events()

        if self.status.error == "pruning_error":
            final_path = self.figure_path + '/fig' + str(self.count) + status + '_error.png'
        else:
            final_path = self.figure_path + '/' + status + '_fig' + str(self.count) + '.png'
        logger.info('Save updated figure to ' + final_path)
        plt.savefig(final_path)

    def expand_axis_for_intermediate(self):
        self.lats_per_step = np.expand_dims(self.lats_per_step, axis=1)
        self.lons_per_step = np.expand_dims(self.lons_per_step, axis=1)
        self.course_per_step = np.expand_dims(self.course_per_step, axis=1)
        self.dist_per_step = np.expand_dims(self.dist_per_step, axis=1)
        self.starttime_per_step = np.expand_dims(self.starttime_per_step, axis=1)
        self.absolutefuel_per_step = np.expand_dims(self.absolutefuel_per_step, axis=1)

        self.shipparams_per_step.expand_axis_for_intermediate()

    def final_pruning(self):
        pass

    def update_dist(self, delta_time, bs):
        pass

    def check_for_positive_constraints(self, constraint_list):
        have_pos_points = constraint_list.have_positive()
        if not have_pos_points:
            self.finish_temp = self.finish
            self.start_temp = self.start
            self.gcr_course_temp = self.gcr_course
            return

        constraint_list.init_positive_lists(self.start, self.finish)
        self.finish_temp = constraint_list.get_current_destination()
        self.start_temp = constraint_list.get_current_start()
        self.gcr_course_temp = self.calculate_gcr(self.start_temp, self.finish_temp) * u.degree

        logger.info('Currently going from')
        logger.info(self.start_temp)
        logger.info('to')
        logger.info(self.finish_temp)

    def check_destination(self):
        destination_lats = self.lats_per_step[0]
        destination_lons = self.lons_per_step[0]

        arrived_at_destination = (destination_lats == self.finish[0]) & (destination_lons == self.finish[1])
        if not arrived_at_destination:
            logger.error('Did not arrive at destination! Need further routing steps or lower resolution.')

    def check_positive_power(self):
        negative_power = self.absolutefuel_per_step.sum() < 0
        if negative_power.any():
            logging.error('Have negative values for power consumption. Needs to be checked!')
