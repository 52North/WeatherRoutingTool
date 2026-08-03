import numpy as np
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
    """TODO: Where is this function used?
    Calculates the accumulated distance along a route

    :param route: Some kind of route
    :type route: TODO: add type of route
    :return: Accumulated distance along a route
    :rtype: np.array
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
    # TODO: Where is this function used?
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
