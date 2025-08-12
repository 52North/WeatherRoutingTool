import numpy as np
import xarray as xr
from geographiclib.geodesic import Geodesic


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
