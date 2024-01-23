import datetime as dt
import logging

from geovectorslib import geod
import numpy as np

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.routeparams import RouteParams

logger = logging.getLogger('WRT.routingalg')


class IsoFuel(IsoBased):
    delta_fuel: float

    def __init__(self, config):
        self.delta_fuel = config.DELTA_FUEL
        super().__init__(config)

    def print_init(self):
        IsoBased.print_init(self)
        logger.info(form.get_log_step('Fuel minimisation, delta power: ' + str(self.delta_fuel), 1))

    def check_isochrones(self, route: RouteParams):
        logger.info('To be implemented')

    def get_dist(self, bs, delta_time):
        dist = delta_time * bs
        return dist

    # calculate time [s] from boat speed and distance
    def get_time(self, bs, dist):
        time = dist / bs
        return time

    ##
    # returns fuel (= power) [W], dist [m], delta_time [s], delta_fuel [Ws]
    def get_delta_variables(self, boat, wind, bs):
        fuel = boat.get_fuel_per_time(self.get_current_course(), wind)
        delta_time = self.delta_fuel / fuel
        dist = self.get_dist(bs, delta_time)

        # print('delta_fuel=' + str(fuel) + ' , delta_time=' + str(delta_time) + ' , dist=' + str(dist))
        delta_fuel = np.repeat(self.delta_fuel, wind['twa'].shape)

        return delta_time, delta_fuel, dist

    ##
    # returns fuel (= power) [W], dist [m], delta_time [s], delta_fuel [Ws]
    def get_delta_variables_netCDF(self, ship_params, bs):
        fuel = ship_params.get_fuel()

        delta_time = self.delta_fuel / fuel
        dist = self.get_dist(bs, delta_time)

        # print('delta_fuel=' + str(fuel) + ' , delta_time=' + str(delta_time) + ' , dist=' + str(dist))
        delta_fuel = np.repeat(self.delta_fuel, bs.shape)

        # self.determine_timespread(delta_time)

        return delta_time, delta_fuel, dist

    ##
    # returns fuel (= power) [W], dist [m], delta_time [s], delta_fuel [Ws]
    def get_delta_variables_netCDF_last_step(self, ship_params, bs):
        fuel = ship_params.get_fuel()
        dist = geod.inverse(self.get_current_lats(), self.get_current_lons(),
                            np.full(self.get_current_lats().shape, self.finish_temp[0]),
                            np.full(self.get_current_lons().shape, self.finish_temp[1]))
        delta_time = self.get_time(bs, dist['s12'])
        delta_fuel = fuel * delta_time

        return delta_time, delta_fuel, dist['s12']

    def determine_timespread(self, delta_time):
        stddev = np.std(delta_time)
        mean = np.mean(delta_time)
        logger.info('delta_time', delta_time / 3600)
        logger.info('spread of time: ' + str(mean / 3600) + '+-' + str(stddev / 3600))

    def update_time(self, delta_time):
        if not ((self.full_time_traveled.shape == delta_time.shape) and (self.time.shape == delta_time.shape)):
            raise ValueError('shapes of delta_time, time and full_time_traveled not matching!')
        for i in range(0, self.full_time_traveled.shape[0]):
            self.full_time_traveled[i] += delta_time[i]
            self.time[i] += dt.timedelta(seconds=delta_time[i])
        self.starttime_per_step = np.vstack((self.time, self.starttime_per_step))

    def final_pruning(self):
        # ToDo: use logger.debug and args.debug
        debug = False
        if debug:
            print('Final IsoFuel Pruning...')
            print('full_fuel_consumed:', self.full_fuel_consumed)

        idxs = np.argmin(self.full_fuel_consumed)

        if debug:
            print('idxs', idxs)

        # Return a trimmed isochrone
        try:
            self.lats_per_step = self.lats_per_step[:, idxs]
            self.lons_per_step = self.lons_per_step[:, idxs]
            self.course_per_step = self.course_per_step[:, idxs]
            self.dist_per_step = self.dist_per_step[:, idxs]
            self.starttime_per_step = self.starttime_per_step[:, idxs]
            self.shipparams_per_step.select(idxs)

            self.current_course = self.current_course[idxs]
            self.full_dist_traveled = self.full_dist_traveled[idxs]
            self.full_time_traveled = self.full_time_traveled[idxs]
            self.full_fuel_consumed = self.full_fuel_consumed[idxs]
            self.time = self.time[idxs]
        except IndexError:
            raise Exception('Pruned indices running out of bounds.')
