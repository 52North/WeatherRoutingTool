from pymoo.core.duplicate import ElementwiseDuplicateElimination

from geographiclib.geodesic import Geodesic
import numpy as np

from typing import Optional
import functools
import json
import math


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

    rs = geod.Inverse(*src, *dst)
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


# constraints
def is_neg_constraints(lat, lon, time, constraint_list):
    """Check if the given point is constrained by the constraints_list

    :param lat: Latitude of the point
    :type lat: float
    :param lat: Longitude of the point
    :type lat: float
    :param time: Datetime of the provided point data
    :type time: datetime.datetime
    :param constraints_list: Constraints list passed in by the config
    :type constraints_list: ConstraintsList
    """

    lat = np.array([lat])
    lon = np.array([lon])
    is_constrained = [False for i in range(0, lat.shape[0])]
    is_constrained = constraint_list.safe_endpoint(lat, lon, time, is_constrained)
    # print(is_constrained)
    return 0 if not is_constrained else 1


def get_constraints_array(route: np.ndarray, constraint_list) -> np.ndarray:
    """Return constraint violation per waypoint in route

    :param route: Candidate array of waypoints
    :type route: np.ndarray
    :return: Array of constraint violations
    """

    constraints = np.array([
        is_neg_constraints(lat, lon, None, constraint_list) for lat, lon in route])
    return constraints


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

    route = [
        ft["geometry"]["coordinates"][::-1]
        for ft in dt["features"]
    ]

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


# ----------
class RouteDuplicateElimination(ElementwiseDuplicateElimination):
    """Custom duplicate elimination strategy for routing problem."""

    def is_equal(self, a, b):
        return np.array_equal(a.X[0], b.X[0])
