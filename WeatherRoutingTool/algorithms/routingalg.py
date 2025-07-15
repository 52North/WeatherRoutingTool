from datetime import datetime

import matplotlib
from astropy import units as u
from geovectorslib import geod
from matplotlib.figure import Figure

from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.environmental_data.weather import WeatherCond

logger = logging.getLogger('WRT.routingalg')


class RoutingAlg:
    """
        Isochrone data structure with typing.
                Parameters:
                    count: int  (routing step)
                    start: tuple    (lat,long at start)
                    finish: tuple   (lat,lon and end)
                    gcr_course: float (initial course)
                    lats1, lons1, azi1, s12: (M, N) arrays, N=headings+1, M=number of steps+1 (decreasing step number)
                    azi0, s0: (M, 1) vectors without history
                    time1: current datetime
                    elapsed: complete elapsed timedelta
    """

    start: tuple  # lat, lon at start
    finish: tuple  # lat, lon at end
    departure_time: datetime
    gcr_course: float  # course of great circle route (0 - 360°)

    fig: matplotlib.figure
    route_ensemble: list
    figure_path: str

    def __init__(self, config):
        lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE
        self.start = (lat_start, lon_start)
        self.finish = (lat_end, lon_end)
        self.departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')

        gcr = self.calculate_gcr(self.start, self.finish)
        self.current_course = gcr * u.degree
        self.gcr_course = gcr * u.degree

        self.figure_path = get_figure_path()
        plt.switch_backend("Agg")

    def init_fig(self, **kwargs):
        pass

    def update_fig(self):
        pass

    def print_init(self):
        logger.info('Initialising routing:')
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish), 1))
        logger.info(form.get_log_step('start time ' + str(self.departure_time), 1))

    def print_current_status(self):
        pass

    def calculate_gcr(self, start, finish):
        gcr = geod.inverse([start[0]], [start[1]], [finish[0]], [
            finish[1]])  # calculate distance between start and end according to Vincents approach, return dictionary
        return gcr['azi1']

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        pass

    def check_for_positive_constraints(self, constraint_list):
        pass

    def terminate(self, **kwargs):
        logger.info(form.get_line_string())
        logger.info('Terminating...')

        self.check_destination()
        self.check_positive_power()
        pass

    def check_destination(self):
        pass

    def check_positive_power(self):
        pass
