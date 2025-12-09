from datetime import datetime

import matplotlib
from astropy import units as u
from geovectorslib import geod
from matplotlib.figure import Figure

from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger('WRT.routingalg')


class RoutingAlg:
    """
    Mother class of all routing algorithms defining basic attributes and methods
    """

    start: tuple  # lat, lon at start
    finish: tuple  # lat, lon at end
    departure_time: datetime
    arrival_time: datetime
    gcr_course: float  # azimuthal angle of great circle route (0 - 360°)
    gcr_dist: float  # distance of great circle route

    fig: matplotlib.figure
    route_ensemble: list
    figure_path: str
    map_ext: Map

    boat_speed: float

    def __init__(self, config):
        lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE

        lat_1, lon1, lat2, lon2 = config.DEFAULT_MAP
        self.map_ext = Map(lat_1, lon1, lat2, lon2)
        self.start = (lat_start, lon_start)
        self.finish = (lat_end, lon_end)
        self.departure_time = config.DEPARTURE_TIME
        self.arrival_time = config.ARRIVAL_TIME

        self.gcr_course, self.gcr_dist = self.calculate_gcr(self.start, self.finish)
        self.gcr_course = self.gcr_course * u.degree

        self.figure_path = get_figure_path()
        plt.switch_backend("Agg")

        self.boat_speed = config.BOAT_SPEED * u.meter/u.second

    def get_boat_speed(self, dists=None):
        return self.boat_speed

    def init_fig(self, **kwargs):
        pass

    def update_fig(self):
        pass

    def clear_figure_path(self):
        self.figure_path = None

    def set_figure_path(self, path: str):
        self.figure_path = path

    def print_init(self):
        logger.info('Initialising routing:')
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish), 1))
        logger.info(form.get_log_step('start time ' + str(self.departure_time), 1))

    def print_current_status(self):
        pass

    def calculate_gcr(self, start, finish):
        """
        Calculate distance between start and end according to Vincenty's approach, return dictionary
        """
        gcr = geod.inverse([start[0]], [start[1]], [finish[0]], [
            finish[1]])
        return gcr['azi1'], gcr['s12']

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
