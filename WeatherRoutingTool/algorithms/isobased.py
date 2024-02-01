import logging
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geovectorslib import geod
from scipy.stats import binned_statistic
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.unit_conversion as units
from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger('WRT.Isobased')


class IsoBased(RoutingAlg):
    ncount: int  # total number of routing steps
    count: int  # current routing step

    route_reached_destination: bool  # True everytime one route (or more) reaches the destination in a routing step
    route_reached_waypoint: bool

    start_temp: tuple  # changes if intermediate waypoints are used
    finish_temp: tuple  # changes if intermediate waypoints are used
    gcr_course_temp: tuple

    '''
       All variables that are named *_per_step constitute (M,N) arrays, whereby N corresponds to the number of
       courses (plus 1) and
       M corresponds to the number of routing steps.

       At the start of each routing step 'count', the element(s) at the position 'count' of the following arrays
       correspond to
       properties of the point of departure of the respective routing step. This means that for 'count = 0' the
       elements of
       lats_per_step and lons_per_step correspond to the coordinates of the departure point of the whole route.
       The first
       elements of the attributes
           - course_per_step
           - dist_per_step
           - speed_per_step
       are 0 to satisfy this definition.
   '''

    lats_per_step: np.ndarray  # lats: (M,N) array, N=headings+1, M=steps (M decreasing)
    lons_per_step: np.ndarray  # longs: (M,N) array, N=headings+1, M=steps
    course_per_step: np.ndarray  # course (0 - 360°)
    dist_per_step: np.ndarray  # geodesic distance traveled per time stamp:
    shipparams_per_step: ShipParams  # object storing ship parameters (fuel rate, power consumption ...)
    starttime_per_step: np.ndarray  # start time for every routing step (datetime object)
    absolutefuel_per_step: np.ndarray   # (kg)

    current_course: np.ndarray  # current course (0-360°)

    # the lenght of the following arrays depends on the number of courses (course segments)
    full_dist_traveled: np.ndarray  # full geodesic distance since start for all courses
    full_time_traveled: np.ndarray  # time elapsed since start for all courses
    time: np.ndarray  # current datetime for all courses

    course_segments: int  # number of course segments in the range of -180° to 180°
    course_increments_deg: int  # increment between different variants
    prune_sector_deg_half: int  # angular range of course that is considered for pruning (only one half, 0-180°)
    prune_segments: int  # number of course bins that are used for pruning
    prune_symmetry_axis: str  # method to define pruning symmetry axis
    prune_groups: str   # method to define grouping of route segments before the pruning
    minimisation_criterion: str  # minimisation criterion

    desired_number_of_routes: int
    current_number_of_routes: int
    current_step_routes: pd.DataFrame
    next_step_routes: pd.DataFrame
    route_list: list
    pruning_error: bool

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

        self.route_reached_destination = False
        self.route_reached_waypoint = False
        self.pruning_error = False

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
        logger.info('step = ', self.count)
        logger.info('start', self.start)
        logger.info('finish', self.finish)
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
        logger.info('lats = ', self.current_lats)
        logger.info('lons = ', self.current_lons)
        logger.info('course = ', self.current_course)
        logger.info('full_time_traveled = ', self.full_time_traveled)

    def define_courses(self):
        # branch out for multiple headings
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

        # determine new headings - centered around gcrs X0 -> X_prev_step
        delta_hdgs = np.linspace(-self.course_segments / 2 * self.course_increments_deg,
                                 +self.course_segments / 2 * self.course_increments_deg, self.course_segments + 1)
        delta_hdgs = np.tile(delta_hdgs, nof_input_routes)

        self.current_course = new_course['azi1'] * u.degree  # center courses around gcr
        self.current_course = np.repeat(self.current_course, self.course_segments + 1)
        self.current_course = self.current_course - delta_hdgs
        self.current_course = units.cut_angles(self.current_course)

    def define_initial_variants(self):
        pass

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        """
        Progress one isochrone with pruning/optimising route for specific time segment

            Parameters:
                iso1 (Isochrone) - starting isochrone
                start_point (tuple) - starting point of the route
                end_point (tuple) - end point of the route
                x1_coords (tuple) - tuple of arrays (lats, lons)
                x2_coords (tuple) - tuple of arrays (lats, lons)
                boat (dict) - boat profile
                winds (dict) - wind functions
                start_time (datetime) - start time
                delta_time (float) - time to move in seconds
                params (dict) - isochrone calculation parameters

            Returns:
                iso (Isochrone) - next isochrone
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
            self.move_boat_direct(wt, boat, constraints_list)

            # Distinguish situations where the ship reached the final destination and where it reached a waypoint
            if self.route_reached_destination:
                logger.info('Initiating last step at routing step ' + str(self.count))

                if self.desired_number_of_routes > 1 and self.current_number_of_routes < self.desired_number_of_routes:
                    self.find_every_route_reaching_destination()
                    number_of_possible_routes = self.current_number_of_routes + self.current_step_routes.shape[0]

                    if self.desired_number_of_routes <= number_of_possible_routes:
                        remaining_routes = self.desired_number_of_routes - self.current_number_of_routes
                        self.find_routes_reaching_destination_in_current_step(remaining_routes)
                        break
                    else:
                        self.find_routes_reaching_destination_in_current_step(number_of_possible_routes)
                        if self.next_step_routes.shape[0] == 0:
                            logger.warning('No routes left for execution, terminating!')
                            break

                        self.set_next_step_routes()
                        self.pruning_per_step(True)
                        if self.pruning_error:
                            break
                        self.route_reached_destination = False
                        self.update_fig('p')
                        self.count += 1
                        continue
                else:
                    break

            elif self.route_reached_waypoint:
                logger.info('Initiating pruning for intermediate waypoint at routing step' + str(self.count))
                self.final_pruning()
                self.expand_axis_for_intermediate()
                constraints_list.reached_positive()
                self.finish_temp = constraints_list.get_current_destination()
                self.start_temp = constraints_list.get_current_start()
                self.gcr_course_temp = self.calculate_gcr(self.start_temp, self.finish_temp) * u.degree
                self.route_reached_waypoint = False

                logger.info('Initiating routing for next segment going from ' + str(self.start_temp) + ' to ' + str(
                    self.finish_temp))
                self.update_fig('p')
                self.count += 1
                continue

            self.pruning_per_step(True)

            if self.pruning_error:
                break
            else:
                self.update_fig('p')
                self.count += 1

        if self.pruning_error and self.count > 0:
            self.count = self.count - 1
            self.revert_to_previous_step()

        # ToDo: harmonize with above/merge with loop over routing steps
        if self.desired_number_of_routes == 1:
            self.final_pruning()
            route = self.terminate()
            return route
        else:
            if not self.route_list:
                if self.pruning_error:
                    self.routes_from_previous_step()
                self.final_pruning()
                route = self.terminate()
                return route
            else:
                self.route_list.sort(key=lambda x: x.get_full_fuel())
                return self.route_list[0]

    def move_boat_direct(self, wt: WeatherCond, boat: Boat, constraint_list: ConstraintsList):
        """
        calculate new boat position for current time step based on wind and boat function
        """
        # get wind speed (tws) and angle (twa)
        debug = False

        # get boat speed
        bs = boat.get_boat_speed()
        bs = np.repeat(bs, (self.get_current_course().shape[0]), axis=0)

        # TODO: check whether changes on IntegrateGeneticAlgorithm should be applied here
        ship_params = boat.get_ship_parameters(self.get_current_course(), self.get_current_lats(),
                                               self.get_current_lons(), self.time, [], True)
        units.cut_angles(self.current_course)

        # ship_params.print()

        delta_time, delta_fuel, dist = self.get_delta_variables_netCDF(ship_params, bs)
        # ToDo: remove debug variable and use logger settings instead
        if debug:
            logger.info('delta_time: ', delta_time)
            logger.info('delta_fuel: ', delta_fuel)
            logger.info('dist: ', dist)
            logger.info('route_reached_destination:', self.route_reached_destination)

        move = self.check_bearing(dist)

        if debug:
            logger.info('move:', move)

        if self.route_reached_destination or self.route_reached_waypoint:
            delta_time_last_step, delta_fuel_last_step, dist_last_step = \
                self.get_delta_variables_netCDF_last_step(ship_params, bs)
            if self.route_reached_destination:
                for i in range(len(self.bool_arr_reached_final)):
                    if self.bool_arr_reached_final[i]:
                        delta_time[i] = delta_time_last_step[i]
                        delta_fuel[i] = delta_fuel_last_step[i]
                        dist[i] = dist_last_step[i]
            else:
                delta_time = delta_time_last_step
                delta_fuel = delta_fuel_last_step
                dist = dist_last_step

        is_constrained = self.check_constraints(move, constraint_list)

        self.update_position(move, is_constrained, dist)
        self.update_time(delta_time)
        self.update_fuel(delta_fuel, ship_params.get_fuel_rate())
        self.update_shipparams(ship_params)

    def find_every_route_reaching_destination(self):
        """
        This function finds routes reaching the destination in the current last step of routing.
        First, it creates a dataframe with origin point of the current route segments.
        The route segments are grouped according to the origin point.
        'dist' is the distance that could be travelled with available amount of fuel.
        'dist_dest' is the distance from origin point to the destination.
        'st_index' is storing the same index order of other nd arrays such as self.lats_per_step
         before grouping. So that, later it is referred in find_routes_reaching_destination_in_current_step function.
        (acts as a key from the dataframe to other arrays such as self.lats_per_step )

        Routes from the current step reaching the destination are stored in 'current_step_routes' dataframe.
        Only the one route segment per branch originating from the same origin point that minimize the fuel is stored
        in current_step_routes. Routes which are not reaching the destination in the current step are stored in
        'next_step_routes' dataframe. In this case, all routes originating from the same origin point are stored in
        the dataframe.
        """
        df_current_last_step = pd.DataFrame()
        df_current_last_step['st_lat'] = self.lats_per_step[1, :]
        df_current_last_step['st_lon'] = self.lons_per_step[1, :]
        df_current_last_step['dist'] = self.current_last_step_dist
        df_current_last_step['dist_dest'] = self.current_last_step_dist_to_dest
        df_current_last_step['fuel'] = self.absolutefuel_per_step[0, :]

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
        """
        In this function, different routes obtained from 'find_every_route_reaching_destination'
        and stored in current_step_routes dataframe are sorted by minimum fuel.
        The number of routes that are selected is specified by the variable remaining_routes.
        """
        current_step_routes_sort_by_fuel = self.current_step_routes.sort_values(by=['fuel'])
        route_df = current_step_routes_sort_by_fuel['st_index'].head(remaining_routes)

        for idxs in route_df:
            self.current_number_of_routes = self.current_number_of_routes + 1
            route_object = self.make_route_object(idxs)
            self.route_list.append(route_object)
            if self.path_to_route_folder is not None:
                route_object.return_route_to_API(self.path_to_route_folder + '/' +
                                                 'route_' + str(self.current_number_of_routes) + ".json")
            if self.figure_path is not None:
                self.plot_routes(idxs)

    def make_route_object(self, idxs):

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
        Plot every complete individual route that is reaching the destination
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
        Updating all arrays according to the indices of the routes that need to be further
        processed in the next routing step
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

            self.current_course = self.current_course[idxs]
            self.full_dist_traveled = self.full_dist_traveled[idxs]
            self.full_time_traveled = self.full_time_traveled[idxs]
            self.time = self.time[idxs]
        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def revert_to_previous_step(self):
        """
        In this function, when all routes are constrained, the arrays are set
        back to previous step to provide meaningful error message.
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
            self.current_course = np.full(col_len, -99)
            self.full_dist_traveled = np.full(col_len, -99)
            self.full_time_traveled = np.full(col_len, -99)
            self.time = np.full(col_len, -99)

        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def routes_from_previous_step(self):
        '''
        When all routes are constrained, unique routes until the current constrained
        routing step are found here. Then, the unique routes are written into json files
        and plotted.
        '''
        df_current_last_step = pd.DataFrame()
        df_current_last_step['st_lat'] = self.lats_per_step[1, :]
        df_current_last_step['st_lon'] = self.lons_per_step[1, :]
        df_current_last_step['fuel'] = self.shipparams_per_step.get_fuel()[0, :]

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
        new_rpm = np.vstack((ship_params_single_step.get_rpm(), self.shipparams_per_step.get_rpm()))
        new_power = np.vstack((ship_params_single_step.get_power(), self.shipparams_per_step.get_power()))
        new_speed = np.vstack((ship_params_single_step.get_speed(), self.shipparams_per_step.get_speed()))
        new_rwind = np.vstack((ship_params_single_step.get_rwind(), self.shipparams_per_step.get_rwind()))
        new_rcalm = np.vstack((ship_params_single_step.get_rcalm(), self.shipparams_per_step.get_rcalm()))
        new_rwaves = np.vstack((ship_params_single_step.get_rwaves(), self.shipparams_per_step.get_rwaves()))
        new_rshallow = np.vstack((ship_params_single_step.get_rshallow(), self.shipparams_per_step.get_rshallow()))
        new_rroughness = np.vstack(
            (ship_params_single_step.get_rroughness(), self.shipparams_per_step.get_rroughness()))

        self.shipparams_per_step.set_rpm(new_rpm)
        self.shipparams_per_step.set_power(new_power)
        self.shipparams_per_step.set_speed(new_speed)
        self.shipparams_per_step.set_rwind(new_rwind)
        self.shipparams_per_step.set_rcalm(new_rcalm)
        self.shipparams_per_step.set_rwaves(new_rwaves)
        self.shipparams_per_step.set_rshallow(new_rshallow)
        self.shipparams_per_step.set_rroughness(new_rroughness)

    def check_course_def(self):
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

    def get_pruned_indices_statistics(self, bin_stat, bin_edges, bin_number, trim):
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

    def pruning(self, trim, bins):
        debug = False
        valid_pruning_segments = -99

        # ToDo: use logger.debug and args.debug
        if debug:
            print('binning for pruning', bins)
            print('current courses', self.current_course)
            print('full_dist_traveled', self.full_dist_traveled)

        is_pruned = False
        if self.prune_groups == 'larger_direction':
            logger.info('Executing larger-direction-based pruning.')
            bin_stat, bin_edges, bin_number = self.larger_direction_based_pruning(bins)
            idxs = self.get_pruned_indices_statistics(bin_stat, bin_edges, bin_number, trim)
            is_pruned = True
        if self.prune_groups == 'courses':
            logger.info('Executing courses-based pruning.')
            bin_stat, bin_edges, bin_number = self.courses_based_pruning(bins)
            idxs = self.get_pruned_indices_statistics(bin_stat, bin_edges, bin_number, trim)
            is_pruned = True
        if self.prune_groups == 'branch':
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
            self.pruning_error = True
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
            self.shipparams_per_step.select(idxs)

            self.starttime_per_step = self.starttime_per_step[:, idxs]

            self.current_course = self.current_course[idxs]
            self.full_dist_traveled = self.full_dist_traveled[idxs]
            self.full_time_traveled = self.full_time_traveled[idxs]
            self.time = self.time[idxs]
        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def courses_based_pruning(self, bins):
        bin_stat, bin_edges, bin_number = binned_statistic(self.current_course, self.full_dist_traveled,
                                                           statistic=np.nanmax, bins=bins)
        return bin_stat, bin_edges, bin_number

    def larger_direction_based_pruning(self, bins):
        start_lats = np.repeat(self.start_temp[0], self.lats_per_step.shape[1])
        start_lons = np.repeat(self.start_temp[1], self.lons_per_step.shape[1])
        larger_direction = geod.inverse(start_lats, start_lons, self.lats_per_step[0], self.lons_per_step[0])
        larger_direction = larger_direction['azi1']
        bin_stat, bin_edges, bin_number = binned_statistic(larger_direction, self.full_dist_traveled,
                                                           statistic=np.nanmax, bins=bins)
        return bin_stat, bin_edges, bin_number

    def branch_based_pruning(self):
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

    def pruning_per_step(self, trim=True):
        if self.prune_symmetry_axis == 'gcr':
            self.pruning_gcr_centered(trim)
        else:
            self.pruning_headings_centered(trim)

    def pruning_gcr_centered(self, trim=True):
        '''
        For every pruning segment, select the route that maximises the distance towards the starting point (or last
        intermediate waypoint). All other routes are discarded. The symmetry axis of the pruning segments is defined
        based on the gcr
        of the current 'mean' position towards the (temporary) destination.
        '''
        # ToDo: use logger.debug and args.debug
        debug = False
        if debug:
            print('Pruning... Pruning symmetry axis defined by gcr')

        # Calculate the auxiliary coordinate for the definition of pruning symmetry axis. The route is propagated
        # towards the coordinate
        # which is reached if one travels from the starting point (or last intermediate waypoint) in the direction
        # of the course defined by the distance between the start point and the destination for the mean distance
        # travelled
        # during the current routing step.
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

    def pruning_headings_centered(self, trim=True):
        '''
        For every pruning segment, select the route that maximises the distance towards the starting point (or last
        intermediate waypoint). All other routes are discarded. The symmetry axis of the pruning segments is given by
        the median of all considered courses.
        '''
        # ToDo: use logger.debug and args.debug
        debug = False
        if debug:
            print('Pruning... Pruning symmetry axis defined by median of considered headings.')

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

    def get_current_course(self):
        return self.current_course

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

    def update_time(self, delta_time):
        self.full_time_traveled += delta_time
        self.time += timedelta(seconds=delta_time)

    def check_bearing(self, dist):
        """
        ToDo: add description
        :param dist:
        :return:
        """
        debug = False

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
                           self.current_course.value, dist.value)

        if reaching_dest:
            reached_final = (self.finish_temp[0] == self.finish[0]) & (self.finish_temp[1] == self.finish[1])

            if debug:
                print('reaching final:', reached_final)

            new_lat = np.full(ncourses, self.finish_temp[0])
            new_lon = np.full(ncourses, self.finish_temp[1])

            if reached_final:
                self.route_reached_destination = True
                self.current_last_step_dist = dist.copy()
                self.current_last_step_dist_to_dest = dist_to_dest['s12']

                self.bool_arr_reached_final = dist_to_dest['s12'] < dist

                for i in range(len(self.bool_arr_reached_final)):
                    if self.bool_arr_reached_final[i]:
                        move['azi2'][i] = dist_to_dest['azi1'][i].value
                        move['lat2'][i] = new_lat[i]
                        move['lon2'][i] = new_lon[i]
            else:
                self.route_reached_waypoint = True
                move['azi2'] = dist_to_dest['azi1'].value
                move['lat2'] = new_lat
                move['lon2'] = new_lon

        return move

    def check_constraints(self, move, constraint_list):
        debug = False

        is_constrained = [False for i in range(0, self.lats_per_step.shape[1])]
        if (debug):
            form.print_step('shape is_constraint before checking:' + str(len(is_constrained)), 1)
        is_constrained = constraint_list.safe_crossing(self.lats_per_step[0], self.lons_per_step[0], move['lat2'],
                                                       move['lon2'], self.time, is_constrained)
        if (debug):
            form.print_step('is_constrained after checking' + str(is_constrained), 1)
        return is_constrained

    def update_position(self, move, is_constrained, dist):
        debug = False
        self.lats_per_step = np.vstack((move['lat2'], self.lats_per_step))
        self.lons_per_step = np.vstack((move['lon2'], self.lons_per_step))
        self.dist_per_step = np.vstack((dist, self.dist_per_step))
        self.course_per_step = np.vstack((self.current_course, self.course_per_step))

        # ToDo: use logger.debug and args.debug
        if debug:
            print('path of this step' +  # str(move['lat1']) +
                  # str(move['lon1']) +
                  str(move['lat2']) + str(move['lon2']))
            print('dist_per_step', self.dist_per_step)
            print('dist', dist)

        start_lats = np.repeat(self.start_temp[0], self.lats_per_step.shape[1])
        start_lons = np.repeat(self.start_temp[1], self.lons_per_step.shape[1])
        travel_dist = geod.inverse(start_lats, start_lons, move['lat2'], move['lon2'])  # calculate full distance
        end_lats = np.repeat(self.finish_temp[0], self.lats_per_step.shape[1])
        end_lons = np.repeat(self.finish_temp[1], self.lons_per_step.shape[1])
        dist_to_dest = geod.inverse(move['lat2'], move['lon2'], end_lats, end_lons)  # calculate full distance

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

    def update_fuel(self, delta_fuel, fuel_rate):
        self.shipparams_per_step.set_fuel_rate(np.vstack((fuel_rate, self.shipparams_per_step.get_fuel_rate())))
        self.absolutefuel_per_step = np.vstack((delta_fuel, self.absolutefuel_per_step))

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
        route_ensemble = []
        self.ax.remove()
        fig, self.ax = graphics.generate_basemap(fig, self.depth, self.start, self.finish)

        count_routeseg = self.lats_per_step.shape[1]

        for iRoute in range(0, count_routeseg):
            route, = self.ax.plot(self.lons_per_step[:, 0], self.lats_per_step[:, 0], color="firebrick")
            route_ensemble.append(route)

        for iRoute in range(0, count_routeseg):
            route_ensemble[iRoute].set_xdata(self.lons_per_step[:, iRoute])
            route_ensemble[iRoute].set_ydata(self.lats_per_step[:, iRoute])
            fig.canvas.draw()
            fig.canvas.flush_events()
        if self.pruning_error:
            final_path = self.figure_path + '/fig' + str(self.count) + status + '_error.png'
        else:
            final_path = self.figure_path + '/fig' + str(self.count) + status + '.png'
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
