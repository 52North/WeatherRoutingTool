import copy
import json
import logging
from typing import Optional

import numpy as np
from astropy import units as u
from geographiclib.geodesic import Geodesic
from pymoo.core.duplicate import ElementwiseDuplicateElimination

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.routeparams import RouteParams

logger = logging.getLogger("WRT.genetic")


def gcr_distance(src, dst) -> float:
    """Return the Great Circle distance between src and dst

    :param src: Source coords as (lat, lon)
    :type src: tuple[float, float]
    :param dst: Destination coords as (lat, lon)
    :type dst: tuple[float, float]
    :return: Distance between src and dst in meters
    :rtype: float
    """

    geod = Geodesic.WGS84

    rs = geod.Inverse(*src[:-1], *dst[:-1])
    return rs["s12"]


def great_circle_distance(src, dst) -> float:
    """Measure great circle distance between src and dst waypoints

    :param src: Source waypoint as (lat, lon) pair
    :type src: tuple[float, float]
    :param dst: Destination waypoint as (lat, lon) pair
    :type dst: tuple[float, float]
    :return: Great circle distance between src and dst
    :rtype: float
    """

    geod: Geodesic = Geodesic.WGS84
    dist = geod.Inverse(*src, *dst)
    return dist["s12"]


def geojson_from_route(
        route: list[tuple[float, float]],
        save_to: Optional[str] = None
) -> dict:
    """Generate geojson from list of waypoints

    :param route: List of waypoints
    :type route: list[tuple[float, float]]
    :param save_to: Specify path to save the geojson to
    :type save_to: str
    :return: Geojson dictionary
    :rtype: dict
    """

    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, rt in enumerate(route):
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(x) for x in rt[::-1]],
            },
            "properties": {"id": i},
            # for compatibility
            "id": i,
        })

    if save_to is not None:
        with open(save_to, 'w') as fp:
            json.dump(geojson, fp)
    return geojson


def get_constraints_array(route: np.ndarray, constraint_list) -> np.ndarray:
    """Return constraint violation per waypoint in route

    :param route: Candidate array of waypoints
    :type route: np.ndarray
    :return: Array of constraint violations
    """
    lat = route[:, 0]
    lon = route[:, 1]
    is_constrained = [False for i in range(0, lat.shape[0] - 1)]

    lat_start = lat[:-1]
    lat_end = lat[1:]
    lon_start = lon[:-1]
    lon_end = lon[1:]

    is_constrained = constraint_list.safe_crossing(lat_start, lon_start, lat_end, lon_end, None, is_constrained)
    return is_constrained


def get_constraints(route, constraint_list):
    """Get sum of constraint violations of all waypoints of the provided route

    :param route: List of waypoints
    :type route: np.ndarray
    :param constraints_list: List of constraints configured by the config
    :type constraints_list: ConstraintsList
    """

    # ToDo: what about time?
    constraints = np.sum(get_constraints_array(route, constraint_list))
    return constraints


def route_from_geojson(dt: dict) -> list[tuple[float, float]]:
    """Parse list of waypoints from geojson dict

    :param dt: Geojson dictionary
    :type dt: dict
    :return: List of waypoints as (lat, lon) pair
    :rtype: list[tuple[float, float]]
    """

    waypoints = [
        ft["geometry"]["coordinates"][::-1] for ft in dt["features"]
    ]
    speed_info = [
        [ft["properties"]["speed"]["value"]] for ft in dt["features"]
    ]
    route = np.hstack((waypoints, speed_info))

    return route


def route_from_geojson_file(path: str) -> list[tuple[float, float]]:
    """Parse route from geojson file

    :param path: Path to geojson file
    :type path: str
    :return: List of waypoints as a (lat, lon) pair
    :rtype: list[tuple[float, float]]
    """

    with open(path) as fp:
        dt = json.load(fp)

    return route_from_geojson(dt)


def get_hist_values_from_route(route: np.array, departure_time):
    lats = route[:, 0]
    lons = route[:, 1]
    speed = route[:, 2]
    speed = speed[:-1] * u.meter / u.second

    waypoint_coords = RouteParams.get_per_waypoint_coords(
        route_lons=lons,
        route_lats=lats,
        start_time=departure_time,
        bs=speed, )
    dist = waypoint_coords['dist']

    hist_values = graphics.get_hist_values_from_widths(dist, speed, "speed")
    return hist_values


def check_speed_dif(speed_arr: np.ndarray, max_diff: float) -> list[int]:
    """
    Identify indices where the speed difference between consecutive points exceeds a limit.

    This function iterates through the speed array and compares each element
    with the previous one. If the absolute difference is greater than the
    specified threshold, both the current and previous indices are added to the output list.

    :param speed_arr: array containing speed values
    :type speed_arr: np.ndarray
    :param max_diff: the maximum allowed difference between consecutive speeds
    :return: a sorted list of unique indice pairs for which speed violations occurred
    """

    viol_list = []
    debug = False

    previous = speed_arr[0]
    for i in range(speed_arr.shape[0] - 1):
        speed = speed_arr[i]
        diff = abs(previous - speed)
        if debug:
            print('diff:', diff)
        if diff > max_diff:
            viol_list.append(i)
            viol_list.append(i - 1)
        previous = speed

    if debug:
        print("before duplicate removal viol_list: ", viol_list)
    viol_list = list(set(viol_list))
    if debug:
        print("returning viol_list: ", viol_list)

    return viol_list


def smoothen_speed_rec(speed_arr: np.ndarray, viol_list: list[int], n_calls: int) -> tuple[np.ndarray, int]:
    r"""
        Perform a single pass of weighted averaging on consecutive speed values violating the maximum speed difference.

        Updates values at the provided indices by averaging them with their
        immediate neighbors. It uses a weighted formula:
        $$(2 \times current + lower + upper) / n\_smooth$$

        :param speed_arr: the original array of speed values
        :param viol_list: list of indices identified as having excessive speed differences
        :param n_calls: the current recursion/iteration count
        :raises Exception: if ``n_calls`` exceeds the hardcoded limit of 40
        :return: a tuple containing the updated (smoothened) array and the incremented call count
    """
    arr_smooth = copy.deepcopy(speed_arr)
    max_calls = 40
    debug = False

    if debug:
        print('Call: ', n_calls)

    if n_calls > max_calls:
        raise Exception("Too many calls to smoothen")

    for ispeed in viol_list:
        lower = 0.
        upper = 0.
        n_smooth = 4

        if ispeed > 0:
            lower = speed_arr[ispeed - 1]
        else:
            n_smooth -= 1
        if (ispeed < speed_arr.shape[0] - 1) and (speed_arr[ispeed + 1] != -99):
            upper = speed_arr[ispeed + 1]
        else:
            n_smooth -= 1
        arr_smooth[ispeed] = (speed_arr[ispeed] * 2 + lower + upper) / n_smooth

        if debug:
            print('    lower: ', lower)
            print('    upper: ', upper)
            print('    orig: ', speed_arr[ispeed])
            print('    av: ', arr_smooth[ispeed])

    n_calls += 1
    return arr_smooth, n_calls


def smoothen_speed(speed_arr: np.ndarray, max_diff: float) -> np.ndarray:
    """
        Iteratively smoothen a speed array until all consecutive differences are within limits.

        This acts as the main controller that repeatedly checks for violations
        and applies the smoothing algorithm until all consecutive differences are within limits or
        the maximum iteration limit is reached.

        :param speed_arr: the initial array of speed values to be processed
        :param max_diff: the allowed difference for speed consecutive speed values
        :return: the final smoothened array where no differences exceed ``max_diff``
    """
    viol_list = check_speed_dif(speed_arr, max_diff)
    n_calls = 0

    while viol_list != []:
        speed_arr, n_calls = smoothen_speed_rec(speed_arr, viol_list, n_calls)
        viol_list = check_speed_dif(speed_arr, max_diff)

    return speed_arr


# ----------
class RouteDuplicateElimination(ElementwiseDuplicateElimination):
    """Custom duplicate elimination strategy for routing problem."""

    def is_equal(self, a, b):
        return np.array_equal(a.X[0], b.X[0])
