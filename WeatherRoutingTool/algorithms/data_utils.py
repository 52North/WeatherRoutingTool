import numpy as np
import xarray as xr
from astropy import units as u
from geographiclib.geodesic import Geodesic

from WeatherRoutingTool.routeparams import RouteParams


def get_closest(array, value):
    """
    Determine and return index of the value in the array which is closest to the given value.
    If there are multiple values in the array with the same distance to the value, the first/smallest index is used.
    :param array: array used to search in
    :type array: numpy.ndarray
    :param value: value for which the closest value in the array should be found
    :type value: numeric
    :return: index
    :rtype: numpy.int64
    """
    return np.abs(array - value).argmin()


def distance(route):
    """Calculates the accumulated geodesic distance along a route.

    .. warning::
        This function is not imported or called anywhere in the main WRT
        package. It appears to be unused. Consider removing it in a future
        clean-up (see issue: orphaned helpers in data_utils.py).
        Unit tests in tests/test_data_utils.py::TestDistance act as a
        regression guard until a removal decision is made.

    :param route: Array of waypoints with columns [lat, lon].
    :type route: numpy.ndarray, shape (n, 2)
    :return: Accumulated geodesic distance from the first waypoint (m).
             Element 0 is always 0. Element i is the total distance from
             waypoint 0 to waypoint i.
    :rtype: numpy.ndarray
    """

    geod = Geodesic.WGS84
    dists = []

    lat1 = route[0, 0]
    lon1 = route[0, 1]
    d = 0

    for coord in route:
        lat2 = coord[0]
        lon2 = coord[1]
        d += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        dists.append(d)
        lat1 = lat2
        lon1 = lon2
    dists = np.array(dists)
    # print(dists)
    return dists


def time_diffs(speed, route):
    """Calculates the accumulated travel time along a route at a constant speed.

    .. warning::
        This function is not imported or called anywhere in the main WRT
        package. It appears to be unused. Consider removing it in a future
        clean-up (see issue: orphaned helpers in data_utils.py).
        Unit tests in tests/test_data_utils.py::TestTimeDiffs act as a
        regression guard until a removal decision is made.

    :param speed: Constant boat speed in m/s.
    :type speed: float
    :param route: Array of waypoints with columns [lat, lon].
    :type route: numpy.ndarray, shape (n, 2)
    :return: Accumulated travel time from the first waypoint (s).
             Element 0 is always 0. Element i is the total time from
             waypoint 0 to waypoint i.
    :rtype: numpy.ndarray
    """
    geod = Geodesic.WGS84
    # speed = speed * 1.852

    lat1 = route[0, 0]
    lon1 = route[0, 1]
    diffs = []
    d = 0
    for coord in route:
        lat2 = coord[0]
        lon2 = coord[1]
        d = d + geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        diffs.append(d)
        lat1 = lat2
        lon1 = lon2

    diffs = np.array(diffs) / speed
    # print(diffs)
    return diffs


def get_speed_from_arrival_time(lons, lats, departure_time, arrival_time) -> u.Quantity:
    """
    Calculate boat speed based on coordinates, departure and arrival time for a route array.

    :param lons: longitudes
    :type lons: np.array
    :param lats: latitudes
    :type lats: np.array
    :param departure_time: departure time
    :type departure_time: datetime object
    :param arrival_time: arrival time
    :type arrival_time: datetime object
    :return: boat speed in m/s (scalar/float)
    :rtype: u.Quantity

    """
    dummy_speed = 6 * u.meter / u.second
    route_dict = RouteParams.get_per_waypoint_coords(
        lons,
        lats,
        departure_time,
        dummy_speed, )

    full_travel_distance = np.sum(route_dict['dist'])
    time_diff = arrival_time - departure_time
    bs = full_travel_distance / (time_diff.total_seconds() * u.second)
    return bs


class GridMixin:
    """TODO: add class description
    _summary_

    """
    grid: xr.Dataset

    def __init__(self, grid, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = grid

    def index_to_coords(self, points_as_indices):
        lats = self.grid.coords['latitude'][[lat_index for lat_index, lon_index in points_as_indices]].values.tolist()
        lons = self.grid.coords['longitude'][[lon_index for lat_index, lon_index in points_as_indices]].values.tolist()
        route = [[x, y] for x, y in zip(lats, lons)]
        return lats, lons, route

    def coords_to_index(self, points_as_coords):
        lats = [get_closest(self.grid.latitude.data, lat) for lat, lon in points_as_coords]
        lons = [get_closest(self.grid.longitude.data, lon) for lat, lon in points_as_coords]
        route = [[x, y] for x, y in zip(lats, lons)]
        return lats, lons, route

    def get_shuffled_cost(self):
        cost = self.grid.data
        shuffled_cost = cost.copy()
        nan_mask = np.isnan(shuffled_cost)  # corresponds, e.g., to land pixels
        shuffled_cost[nan_mask] = np.nanmean(cost)

        # shuffle first along South-North (latitude), then along West-East (longitude) axis
        rng = np.random.default_rng()
        shuffled_cost = rng.permutation(shuffled_cost, axis=0)
        shuffled_cost = rng.permutation(shuffled_cost, axis=1)

        # assign very high weights to nan values (land pixels)
        shuffled_cost[nan_mask] = 1e20
        return shuffled_cost
