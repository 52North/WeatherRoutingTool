import datetime as dt
import logging
import time

import numpy as np
import matplotlib
from geovectorslib import geod
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger('WRT.routingalg')


class RoutingAlg():
    """
        Isochrone data structure with typing.
                Parameters:
                    count: int  (routing step)
                    start: tuple    (lat,long at start)
                    finish: tuple   (lat,lon and end)
                    gcr_azi: float (initial gcr heading)
                    lats1, lons1, azi1, s12: (M, N) arrays, N=headings+1, M=number of steps+1 (decreasing step number)
                    azi0, s0: (M, 1) vectors without history
                    time1: current datetime
                    elapsed: complete elapsed timedelta
        """
    ncount: int  # total number of routing steps
    count: int  # current routing step

    start: tuple  # lat, lon at start
    finish: tuple  # lat, lon at end
    departure_time: dt.datetime
    gcr_azi: float  # azimut of great circle route

    fig: matplotlib.figure
    route_ensemble: list
    figure_path: str

    def __init__(self, start, finish, departure_time, figure_path=None):
        self.count = 0
        self.start = start
        self.finish = finish
        self.departure_time = departure_time

        gcr = self.calculate_gcr(start, finish)
        self.current_azimuth = gcr
        self.gcr_azi = gcr

        self.figure_path = figure_path

    def init_fig(self):
        pass

    def update_fig(self):
        pass

    def print_init(self):
        logger.info('Initialising routing:')
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish), 1))
        logger.info(form.get_log_step('start time ' + str(self.departure_time), 1))

    def print_current_status(self):
        pass

    def set_steps(self, steps):
        self.ncount = steps

    def calculate_gcr(self, start, finish):
        gcr = geod.inverse([start[0]], [start[1]], [finish[0]], [
            finish[1]])  # calculate distance between start and end according to Vincents approach, return dictionary
        return gcr['azi1']

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        pass

    def check_for_positive_constraints(self, constraint_list):
        pass

    def terminate(self):
        logger.info(form.get_line_string())
        logger.info('Terminating...')

        self.check_destination()
        self.check_positive_power()
        pass

    def check_destination(self):
        pass

    def check_positive_power(self):
        pass
