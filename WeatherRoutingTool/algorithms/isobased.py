import datetime as dt
import logging

import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import matplotlib.pyplot as plt
from geovectorslib import geod
from global_land_mask import globe
from scipy.stats import binned_statistic

import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.formatting as form
import WeatherRoutingTool.utils.unit_conversion as units
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger('WRT.Isobased')


class IsoBased(RoutingAlg):
    is_last_step: bool
    is_pos_constraint_step: bool

    start_temp: tuple
    finish_temp: tuple
    gcr_azi_temp: tuple

    '''
           All variables that are named *_per_step constitute (M,N) arrays, whereby N corresponds to the number of
           variants (plus 1) and
           M corresponds to the number of routing steps.

           At the start of each routing step 'count', the element(s) at the position 'count' of the following arrays
           correspond to
           properties of the point of departure of the respective routing step. This means that for 'count = 0' the
           elements of
           lats_per_step and lons_per_step correspond to the coordinates of the departure point of the whole route.
           The first
           elements of the attributes
               - azimuth_per_step
               - dist_per_step
               - speed_per_step
           are 0 to satisfy this definition.
       '''

    lats_per_step: np.ndarray  # lats: (M,N) array, N=headings+1, M=steps (M decreasing)    #
    lons_per_step: np.ndarray  # longs: (M,N) array, N=headings+1, M=steps
    azimuth_per_step: np.ndarray  # heading
    dist_per_step: np.ndarray  # geodesic distance traveled per time stamp:
    shipparams_per_step: ShipParams
    starttime_per_step: np.ndarray

    current_azimuth: np.ndarray  # current azimuth
    current_variant: np.ndarray  # current variant

    # the lenght of the following arrays depends on the number of variants (variant segments)
    full_dist_traveled: np.ndarray  # full geodesic distance since start for all variants
    full_time_traveled: np.ndarray  # time elapsed since start for all variants
    full_fuel_consumed: np.ndarray
    time: np.ndarray  # current datetime for all variants

    variant_segments: int  # number of variant segments in the range of -180° to 180°
    variant_increments_deg: int
    expected_speed_kts: int
    prune_sector_deg_half: int  # angular range of azimuth that is considered for pruning (only one half)
    prune_segments: int  # number of azimuth bins that are used for pruning
    prune_gcr_centered: bool
    prune_bearings: bool
    minimisation_criterion: str

    def __init__(self, start, finish, departure_time, figurepath=""):
        super().__init__(start, finish, departure_time, figurepath)

        self.lats_per_step = np.array([[start[0]]])
        self.lons_per_step = np.array([[start[1]]])
        self.azimuth_per_step = np.array([[None]])
        self.dist_per_step = np.array([[0]])
        sp = ShipParams.set_default_array()
        self.shipparams_per_step = sp
        self.starttime_per_step = np.array([[departure_time]])

        self.time = np.array([departure_time])
        self.full_time_traveled = np.array([0])
        self.full_fuel_consumed = np.array([0])
        self.full_dist_traveled = np.array([0])

        self.current_variant = self.current_azimuth
        self.is_last_step = False
        self.is_pos_constraint_step = False

        self.finish_temp = self.finish
        self.start_temp = self.start
        self.gcr_azi_temp = self.gcr_azi

        self.minimisation_criterion = 'squareddist_over_disttodest'

    def print_init(self):
        RoutingAlg.print_init(self)
        logger.info(form.get_log_step('pruning settings', 1))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_SECTOR_DEG_HALF: ' + str(self.prune_sector_deg_half), 2))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_SEGMENTS: ' + str(self.prune_segments), 2))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_GCR_CENTERED: ' + str(self.prune_gcr_centered), 2))
        logger.info(form.get_log_step('ISOCHRONE_PRUNE_BEARING: ' + str(self.prune_bearings), 2))
        logger.info(form.get_log_step('ISOCHRONE_MINIMISATION_CRITERION: ' + str(self.minimisation_criterion), 2))
        logger.info(form.get_log_step('ROUTER_HDGS_SEGMENTS: ' + str(self.variant_segments), 2))
        logger.info(form.get_log_step('ROUTER_HDGS_INCREMENTS_DEG: ' + str(self.variant_increments_deg), 2))

    def print_current_status(self):
        print('PRINTING ALG SETTINGS')
        print('step = ', self.count)
        print('start', self.start)
        print('finish', self.finish)
        print('per-step variables:')
        print(form.get_log_step('lats_per_step = ' + str(self.lats_per_step)))
        print(form.get_log_step('lons_per_step = ' + str(self.lons_per_step)))
        print(form.get_log_step('variants = ' + str(self.azimuth_per_step)))
        print(form.get_log_step('dist_per_step = ' + str(self.dist_per_step)))
        print(form.get_log_step('starttime_per_step = ' + str(self.starttime_per_step)))

        self.shipparams_per_step.print()

        print('per-variant variables')
        print(form.get_log_step('time =' + str(self.time)))
        print(form.get_log_step('full_dist_traveled=' + str(self.full_dist_traveled)))
        print(form.get_log_step('full_time_traveled = ' + str(self.full_time_traveled)))
        print(form.get_log_step('full_fuel_consumed = ' + str(self.full_fuel_consumed)))

    def print_shape(self):
        print('PRINTING SHAPE')
        print('per-step variables:')
        print(form.get_log_step('lats_per_step = ' + str(self.lats_per_step.shape)))
        print(form.get_log_step('lons_per_step = ' + str(self.lons_per_step.shape)))
        print(form.get_log_step('azimuths = ' + str(self.azimuth_per_step.shape)))
        print(form.get_log_step('dist_per_step = ' + str(self.dist_per_step.shape)))

        self.shipparams_per_step.print_shape()

        print('per-variant variables:')
        print(form.get_log_step('time =' + str(self.time.shape)))
        print(form.get_log_step('full_dist_traveled = ' + str(self.full_dist_traveled.shape)))
        print(form.get_log_step('full_time_traveled = ' + str(self.full_time_traveled.shape)))
        print(form.get_log_step('full_fuel_consumed = ' + str(self.full_fuel_consumed.shape)))

    def current_position(self):
        print('CURRENT POSITION')
        print('lats = ', self.current_lats)
        print('lons = ', self.current_lons)
        print('azimuth = ', self.current_azimuth)
        print('full_time_traveled = ', self.full_time_traveled)

    def define_variants(self):
        # branch out for multiple headings
        nof_input_routes = self.lats_per_step.shape[1]

        new_finish_one = np.repeat(self.finish_temp[0], nof_input_routes)
        new_finish_two = np.repeat(self.finish_temp[1], nof_input_routes)

        new_azi = geod.inverse(self.lats_per_step[0], self.lons_per_step[0], new_finish_one, new_finish_two)

        self.lats_per_step = np.repeat(self.lats_per_step, self.variant_segments + 1, axis=1)
        self.lons_per_step = np.repeat(self.lons_per_step, self.variant_segments + 1, axis=1)
        self.dist_per_step = np.repeat(self.dist_per_step, self.variant_segments + 1, axis=1)
        self.azimuth_per_step = np.repeat(self.azimuth_per_step, self.variant_segments + 1, axis=1)
        self.starttime_per_step = np.repeat(self.starttime_per_step, self.variant_segments + 1, axis=1)

        self.shipparams_per_step.define_variants(self.variant_segments)

        self.full_time_traveled = np.repeat(self.full_time_traveled, self.variant_segments + 1, axis=0)
        self.full_fuel_consumed = np.repeat(self.full_fuel_consumed, self.variant_segments + 1, axis=0)
        self.full_dist_traveled = np.repeat(self.full_dist_traveled, self.variant_segments + 1, axis=0)
        self.time = np.repeat(self.time, self.variant_segments + 1, axis=0)
        self.check_variant_def()

        # determine new headings - centered around gcrs X0 -> X_prev_step
        delta_hdgs = np.linspace(-self.variant_segments / 2 * self.variant_increments_deg,
                                 +self.variant_segments / 2 * self.variant_increments_deg, self.variant_segments + 1)
        delta_hdgs = np.tile(delta_hdgs, nof_input_routes)

        self.current_variant = new_azi['azi1']  # center courses around gcr
        self.current_variant = np.repeat(self.current_variant, self.variant_segments + 1)
        self.current_variant = self.current_variant - delta_hdgs
        self.current_variant = units.cut_angles(self.current_variant)

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
        for i in range(self.ncount):
            logger.info(form.get_line_string())
            logger.info('Step ' + str(i))

            self.define_variants_per_step()
            self.move_boat_direct(wt, boat, constraints_list)
            if self.is_last_step:
                logger.info('Initiating last step at routing step ' + str(self.count))
                break

            if self.is_pos_constraint_step:
                logger.info('Initiating pruning for intermediate waypoint at routing step' + str(self.count))
                self.final_pruning()
                self.expand_axis_for_intermediate()
                constraints_list.reached_positive()
                self.finish_temp = constraints_list.get_current_destination()
                self.start_temp = constraints_list.get_current_start()
                self.gcr_azi_temp = self.calculate_gcr(self.start_temp, self.finish_temp)
                self.is_pos_constraint_step = False

                logger.info('Initiating routing for next segment going from ' + str(self.start_temp) + ' to ' + str(
                    self.finish_temp))
                # self.update_fig('p')
                continue

            # if i>9:
            # self.update_fig('bp')
            self.pruning_per_step(
                True)  # form.print_current_time('move_boat: Step=' + str(i), start_time)  # if i>9:  #
            self.update_fig('p')

        self.final_pruning()
        route = self.terminate()
        return route

    def move_boat_direct(self, wt: WeatherCond, boat: Boat, constraint_list: ConstraintsList):
        """
                calculate new boat position for current time step based on wind and boat function
            """

        # get wind speed (tws) and angle (twa)
        debug = False

        # get boat speed
        bs = boat.boat_speed_function()
        bs = np.repeat(bs, (self.get_current_azimuth().shape[0]), axis=0)

	# TODO: check whether changes on IntegrateGeneticAlgorithm should be applied here
        ship_params = boat.get_fuel_per_time_netCDF(self.get_current_azimuth(), self.get_current_lats(),
                                                    self.get_current_lons(), self.time, True)
        units.cut_angles(self.current_variant)

        # ship_params.print()

        delta_time, delta_fuel, dist = self.get_delta_variables_netCDF(ship_params, bs)
        if (debug):
            print('delta_time: ', delta_time)
            print('delta_fuel: ', delta_fuel)
            print('dist: ', dist)
            print('is_last_step:', self.is_last_step)

        move = self.check_bearing(dist)

        if (debug):
            print('move:', move)

        if (self.is_last_step or self.is_pos_constraint_step):
            delta_time, delta_fuel, dist = self.get_delta_variables_netCDF_last_step(ship_params, bs)

        is_constrained = self.check_constraints(move, constraint_list)

        self.update_position(move, is_constrained, dist)
        self.update_time(delta_time)
        self.update_fuel(delta_fuel)
        self.update_shipparams(ship_params)
        self.count += 1

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

    def check_variant_def(self):
        if (not ((self.lats_per_step.shape[1] == self.lons_per_step.shape[1]) and (
                self.lats_per_step.shape[1] == self.azimuth_per_step.shape[1]) and (
                         self.lats_per_step.shape[1] == self.dist_per_step.shape[1]))):
            raise 'define_variants: number of columns not matching!'

        if (not ((self.lats_per_step.shape[0] == self.lons_per_step.shape[0]) and (
                self.lats_per_step.shape[0] == self.azimuth_per_step.shape[0]) and (
                         self.lats_per_step.shape[0] == self.dist_per_step.shape[0]) and (
                         self.lats_per_step.shape[0] == (self.count + 1)))):
            raise ValueError(
                'define_variants: number of rows not matching! count = ' + str(self.count) + ' lats per step ' + str(
                    self.lats_per_step.shape[0]))

    def pruning(self, trim, bins, larger_direction_based=True):
        debug = False
        valid_pruning_segments = -99

        if (debug):
            print('binning for pruning', bins)
            print('current courses', self.current_variant)
            print('full_dist_traveled', self.full_time_traveled)

        idxs = []

        bin_stat = None
        bin_edges = None
        bin_number = None
        if larger_direction_based:
            bin_stat, bin_edges, bin_number = self.larger_direction_based_pruning(bins)
        else:
            bin_stat, bin_edges, bin_number = self.courses_based_pruning(bins)

        if trim:
            for i in range(len(bin_edges) - 1):
                try:
                    if (bin_stat[i] == 0):
                        # form.print_step('Pruning: sector ' + str(i) + 'is null (binstat[i])=' + str(bin_stat[i]) +
                        # 'full_dist_traveled=' + str(self.full_dist_traveled))
                        continue
                    idxs.append(np.where(self.full_dist_traveled == bin_stat[i])[0][0])
                except IndexError:
                    pass
            idxs = list(set(idxs))
        else:
            for i in range(len(bin_edges) - 1):
                idxs.append(np.where(self.full_dist_traveled == bin_stat[i])[0])
            idxs = list(set([item for subl in idxs for item in subl]))

        if (debug):
            print('full_dist_traveled', self.full_dist_traveled)
            print('Indexes that passed', idxs)

        valid_pruning_segments = len(idxs)
        if (valid_pruning_segments == 0):
            logger.error(' All pruning segments fully constrained for step ' + str(self.count) + '!')
        elif (valid_pruning_segments < self.prune_segments * 0.1):
            logger.warning(' More than 90% of pruning segments constrained for step ' + str(self.count) + '!')
        elif (valid_pruning_segments < self.prune_segments * 0.5):
            logger.warning(' More than 50% of pruning segments constrained for step ' + str(self.count) + '!')

        # Return a trimmed isochrone
        try:
            self.lats_per_step = self.lats_per_step[:, idxs]
            self.lons_per_step = self.lons_per_step[:, idxs]
            self.azimuth_per_step = self.azimuth_per_step[:, idxs]
            self.dist_per_step = self.dist_per_step[:, idxs]
            self.shipparams_per_step.select(idxs)

            self.starttime_per_step = self.starttime_per_step[:, idxs]

            self.current_azimuth = self.current_variant[idxs]
            self.current_variant = self.current_variant[idxs]
            self.full_dist_traveled = self.full_dist_traveled[idxs]
            self.full_time_traveled = self.full_time_traveled[idxs]
            self.full_fuel_consumed = self.full_fuel_consumed[idxs]
            self.time = self.time[idxs]
        except IndexError:
            raise Exception('Pruned indices running out of bounds.')

    def courses_based_pruning(self, bins):
        bin_stat, bin_edges, bin_number = binned_statistic(self.current_variant, self.full_dist_traveled,
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

    def pruning_per_step(self, trim=True):
        if self.prune_gcr_centered:
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

        debug = False
        if debug:
            print('Pruning... Pruning symmetry axis defined by gcr')

        # Calculate the auxiliary coordinate for the definition of pruning symmetry axis. The route is propagated
        # towards the coordinate
        # which is reached if one travels from the starting point (or last intermediate waypoint) in the direction
        # of the azimuth defined by the distance between the start point and the destination for the mean distance
        # travelled
        # during the current routing step.
        start_lats = np.repeat(self.start_temp[0], self.lats_per_step.shape[1])
        start_lons = np.repeat(self.start_temp[1], self.lons_per_step.shape[1])
        full_travel_dist = geod.inverse(start_lats, start_lons, self.lats_per_step[0], self.lons_per_step[1])
        mean_dist = np.mean(full_travel_dist['s12'])
        gcr_point = geod.direct([self.start_temp[0]], [self.start_temp[1]], self.gcr_azi_temp, mean_dist)

        new_azi = geod.inverse(gcr_point['lat2'], gcr_point['lon2'], [self.finish_temp[0]], [self.finish_temp[1]])

        if (debug):
            print('current mean end point: (' + str(gcr_point['lat2']) + ',' + str(gcr_point['lon2']) + ')')
            print('current temporary destination: ', self.finish_temp)
            print('mean azimuth', new_azi['azi1'])

        # define pruning area
        azi0s = np.repeat(new_azi['azi1'], self.prune_segments + 1)

        delta_hdgs = np.linspace(-self.prune_sector_deg_half, +self.prune_sector_deg_half, self.prune_segments + 1)

        bins = units.cut_angles(azi0s - delta_hdgs)
        bins = np.sort(bins)

        if self.prune_bearings:
            self.pruning(trim, bins, False)
        else:
            if ((self.ncount % 10) < 3) and (self.ncount > 10):
                self.pruning(trim, bins, True)
            else:
                self.pruning(trim, bins, False)

    def pruning_headings_centered(self, trim=True):
        '''
        For every pruning segment, select the route that maximises the distance towards the starting point (or last
        intermediate waypoint). All other routes are discarded. The symmetry axis of the pruning segments is given by
        the median of all considered courses.
        '''

        debug = False
        if debug:
            print('Pruning... Pruning symmetry axis defined by median of considered headings.')

        # propagate current end points towards temporary destination
        nof_input_routes = self.lats_per_step.shape[1]
        new_finish_one = np.repeat(self.finish_temp[0], nof_input_routes)
        new_finish_two = np.repeat(self.finish_temp[1], nof_input_routes)

        new_azi = geod.inverse(self.lats_per_step[0], self.lons_per_step[0], new_finish_one, new_finish_two)

        # sort azimuths and select (approximate) median
        new_azi_sorted = np.sort(new_azi['azi1'])
        meadian_indx = int(np.round(new_azi_sorted.shape[0] / 2))

        if debug:
            print('sorted azimuths: ', new_azi_sorted)
            print('median index: ', meadian_indx)

        mean_azimuth = new_azi_sorted[meadian_indx]

        if debug:
            # plot symmetry axis and boundaries of pruning area
            symmetry_axis = geod.direct([self.lats_per_step[1][meadian_indx]], [self.lons_per_step[1][meadian_indx]],
                                        mean_azimuth, 1000000)
            lower_bound = geod.direct([self.lats_per_step[1][meadian_indx]], [self.lons_per_step[1][meadian_indx]],
                                      mean_azimuth - self.prune_sector_deg_half, 1000000)
            upper_bound = geod.direct([self.lats_per_step[1][meadian_indx]], [self.lons_per_step[1][meadian_indx]],
                                      mean_azimuth + self.prune_sector_deg_half, 1000000)

            self.ax.plot([self.lons_per_step[1][meadian_indx], symmetry_axis["lon2"]],
                         [self.lats_per_step[1][meadian_indx], symmetry_axis["lat2"]], color="blue")
            self.ax.plot([self.lons_per_step[1][meadian_indx], lower_bound["lon2"]],
                         [self.lats_per_step[1][meadian_indx], lower_bound["lat2"]], color="blue")
            self.ax.plot([self.lons_per_step[1][meadian_indx], upper_bound["lon2"]],
                         [self.lats_per_step[1][meadian_indx], upper_bound["lat2"]], color="blue")

            if self.figure_path is not None:
                final_path = self.figure_path + '/fig' + str(self.count) + '_median.png'
                print('Saving updated figure to ', final_path)
                plt.savefig(final_path)

        # define pruning area
        bins = units.get_angle_bins(mean_azimuth - self.prune_sector_deg_half,
                                    mean_azimuth + self.prune_sector_deg_half, self.prune_segments + 1)

        bins = np.sort(bins)

        if debug:
            print('bins: ', bins)

        if self.prune_bearings:
            self.pruning(trim, bins, False)
        else:
            if ((self.ncount % 10) < 3) and (self.ncount > 10):
                self.pruning(trim, bins, True)
            else:
                self.pruning(trim, bins, False)

    def define_variants_per_step(self):
        self.define_variants()

    def set_pruning_settings(self, sector_deg_half, seg, prune_bearings=False, prune_gcr_centered=True):
        self.prune_sector_deg_half = sector_deg_half
        self.prune_segments = seg
        self.prune_bearings = prune_bearings
        self.prune_gcr_centered = prune_gcr_centered

    def set_minimisation_criterion(self, min_str):
        self.minimisation_criterion = min_str

    def set_variant_segments(self, seg, inc):
        self.variant_segments = seg
        self.variant_increments_deg = inc

    def get_current_azimuth(self):
        return self.current_variant

    def get_current_lats(self):
        return self.lats_per_step[0, :]

    def get_current_lons(self):
        return self.lons_per_step[0, :]

    def get_current_speed(self):
        return self.speed_per_step[0]

    def get_wind_functions(self, wt):
        debug = False
        winds = wt.get_wind_function((self.get_current_lats(), self.get_current_lons()), self.time[0])
        if (debug):
            print('obtaining wind function for position: ', self.get_current_lats(), self.get_current_lons())
            print('time', self.time[0])
            print('winds', winds)
        return winds

    def check_settings(self):
        if (self.variant_segments / 2 * self.variant_increments_deg >= self.prune_sector_deg_half):
            raise ValueError(
                'Prune sector does not contain all variants. Please adjust settings. (variant_segments=' + str(
                    self.variant_segments) + ', variant_increments_deg=' + str(
                    self.variant_increments_deg) + ', prune_sector_deg_half=' + str(self.prune_sector_deg_half))
        if ((self.variant_segments % 2) != 0):
            raise ValueError(
                'Please provide an even number of variant segments, you chose: ' + str(self.variant_segments))

        if ((self.prune_segments % 2) != 0):
            raise ValueError('Please provide an even number of prune segments, you chose: ' + str(self.prune_segments))

    def get_final_index(self):
        idx = np.argmax(self.full_dist_traveled)
        return idx

    def terminate(self):
        super().terminate()

        self.lats_per_step = np.flip(self.lats_per_step, 0)
        self.lons_per_step = np.flip(self.lons_per_step, 0)
        self.azimuth_per_step = np.flip(self.azimuth_per_step, 0)
        self.dist_per_step = np.flip(self.dist_per_step, 0)
        self.starttime_per_step = np.flip(self.starttime_per_step, 0)
        self.shipparams_per_step.flip()

        time = round(self.full_time_traveled / 3600, 2)
        route = RouteParams(count=self.count, start=self.start, finish=self.finish, gcr=self.full_dist_traveled,
                            route_type='min_time_route', time=time, lats_per_step=self.lats_per_step[:],
                            lons_per_step=self.lons_per_step[:], azimuths_per_step=self.azimuth_per_step[:],
                            dists_per_step=self.dist_per_step[:], starttime_per_step=self.starttime_per_step[:],
                            ship_params_per_step=self.shipparams_per_step)

        return route

    def update_time(self, delta_time):
        self.full_time_traveled += delta_time
        self.time += dt.timedelta(seconds=delta_time)

    def check_bearing(self, dist):
        debug = False

        nvariants = self.get_current_lons().shape[0]
        dist_to_dest = geod.inverse(self.get_current_lats(), self.get_current_lons(),
                                    np.full(nvariants, self.finish_temp[0]), np.full(nvariants, self.finish_temp[1]))
        if (debug):
            print('dist_to_dest:', dist_to_dest['s12'])
            print('dist traveled:', dist)

        reaching_dest = np.any(dist_to_dest['s12'] < dist)

        if (debug):
            print('reaching dest:', reaching_dest)

        if (reaching_dest):
            reached_final = (self.finish_temp[0] == self.finish[0]) & (self.finish_temp[1] == self.finish[1])

            if (debug):
                print('reaching final:', reached_final)

            new_lat = np.full(nvariants, self.finish_temp[0])
            new_lon = np.full(nvariants, self.finish_temp[1])

            if reached_final:
                self.is_last_step = True
            else:
                self.is_pos_constraint_step = True

            return {'azi2': dist_to_dest['azi1'], 'lat2': new_lat, 'lon2': new_lon,
                    'iterations': -99}  # compare to  'return {'lat2': lat2, 'lon2': lon2, 'azi2': azi2,
            # 'iterations':  # iterations}' by geod.direct

        move = geod.direct(self.get_current_lats(), self.get_current_lons(), self.current_variant, dist)
        # form.print_step('move=' + str(move),1)
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
        self.azimuth_per_step = np.vstack((self.current_variant, self.azimuth_per_step))

        if (debug):
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
        if (debug):
            print('full_dist_traveled:', self.full_dist_traveled)

    def update_fuel(self, delta_fuel):
        self.shipparams_per_step.set_fuel(np.vstack((delta_fuel, self.shipparams_per_step.get_fuel())))
        for i in range(0, self.full_fuel_consumed.shape[0]):
            self.full_fuel_consumed[i] += delta_fuel[i]

    def get_delta_variables(self, boat, wind, bs):
        pass

    def get_delta_variables_netCDF_last_step(self, boat, wind, bs):
        pass

    def init_fig(self, water_depth, map_size, showDepth=True):
        if self.figure_path is None:
            return
        self.showDepth = showDepth
        plt.rcParams['font.size'] = 20
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.axis('off')
        self.ax.xaxis.set_tick_params(labelsize='large')

        if (self.showDepth):
            # decrease resolution and extend of depth data to prevent memory issues when plotting
            ds_depth = water_depth.depth_data.coarsen(latitude=10, longitude=10, boundary='exact').mean()
            ds_depth_coarsened = ds_depth.compute()

            self.depth = ds_depth_coarsened.where(
                (ds_depth_coarsened.latitude > map_size.lat1) & (ds_depth_coarsened.latitude < map_size.lat2) & (
                        ds_depth_coarsened.longitude > map_size.lon1) & (ds_depth_coarsened.longitude < map_size.lon2) & (
                        ds_depth_coarsened.depth < 0), drop=True)

        self.generate_basemap()

        final_path = self.figure_path + '/fig0.png'
        logger.info('Save start figure to ' + final_path)
        plt.savefig(final_path)

    def generate_basemap(self):

        self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())
        if (self.showDepth):
            level_diff = 10
            cp = self.depth['depth'].plot.contourf(ax=self.ax, levels=np.arange(-100, 0, level_diff),
                                                   transform=ccrs.PlateCarree())
            self.fig.colorbar(cp, ax=self.ax, shrink=0.7, label='Wassertiefe (m)', pad=0.1)

            self.fig.subplots_adjust(left=0.1, right=1.2, bottom=0, top=1, wspace=0, hspace=0)

        self.ax.add_feature(cf.LAND)
        self.ax.add_feature(cf.COASTLINE)
        self.ax.gridlines(draw_labels=True)

        self.ax.plot(self.start[1], self.start[0], marker="o", markerfacecolor="orange", markeredgecolor="orange",
                     markersize=10)
        self.ax.plot(self.finish[1], self.finish[0], marker="o", markerfacecolor="orange", markeredgecolor="orange",
                     markersize=10)

        gcr = graphics.get_gcr_points(self.start[0], self.start[1], self.finish[0], self.finish[1], n_points=10)
        lats_gcr = [x[0] for x in gcr]
        lons_gcr = [x[1] for x in gcr]
        self.ax.plot(lons_gcr, lats_gcr, color="orange")
        plt.title('')

    def update_fig(self, status):
        if self.figure_path is None:
            return
        fig = self.fig
        route_ensemble = []
        self.ax.remove()
        self.generate_basemap()

        count_routeseg = self.lats_per_step.shape[1]

        for iRoute in range(0, count_routeseg):
            route, = self.ax.plot(self.lons_per_step[:, 0], self.lats_per_step[:, 0], color="firebrick")
            route_ensemble.append(route)

        for iRoute in range(0, count_routeseg):
            route_ensemble[iRoute].set_xdata(self.lons_per_step[:, iRoute])
            route_ensemble[iRoute].set_ydata(self.lats_per_step[:, iRoute])
            fig.canvas.draw()
            fig.canvas.flush_events()

        final_path = self.figure_path + '/fig' + str(self.count) + status + '.png'
        logger.info('Save updated figure to ' + final_path)
        plt.savefig(final_path)

    def expand_axis_for_intermediate(self):
        self.lats_per_step = np.expand_dims(self.lats_per_step, axis=1)
        self.lons_per_step = np.expand_dims(self.lons_per_step, axis=1)
        self.azimuth_per_step = np.expand_dims(self.azimuth_per_step, axis=1)
        self.dist_per_step = np.expand_dims(self.dist_per_step, axis=1)
        self.starttime_per_step = np.expand_dims(self.starttime_per_step, axis=1)

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
            self.gcr_azi_temp = self.gcr_azi
            return

        constraint_list.init_positive_lists(self.start, self.finish)
        self.finish_temp = constraint_list.get_current_destination()
        self.start_temp = constraint_list.get_current_start()
        self.gcr_azi_temp = self.calculate_gcr(self.start_temp, self.finish_temp)

        print('Currently going from')
        print(self.start_temp)
        print('to')
        print(self.finish_temp)

    def check_destination(self):
        destination_lats = self.lats_per_step[0]
        destination_lons = self.lons_per_step[0]

        arrived_at_destination = (destination_lats == self.finish[0]) & (destination_lons == self.finish[1])
        if not arrived_at_destination:
            logger.error('Did not arrive at destination! Need further routing steps or lower resolution.')

    def check_positive_power(self):
        negative_power = self.full_fuel_consumed < 0
        if negative_power.any():
            logging.error('Have negative values for power consumption. Needs to be checked!')
