# Heuristic algorithm to find a route between two points which does not cross land.
# It does not consider weather, travel time, etc.
# The algorithm is explained in Kuhlemann & Tierney (2020): “A genetic algorithm for finding realistic sea routes
# considering the weather”, doi:10.1007/s10732-020-09449-7
import json
import logging
import math
import os.path
import sys
from typing import TypedDict

import shapely
from geographiclib.geodesic import Geodesic
from geographiclib.geodesicline import GeodesicLine
from global_land_mask import is_land as is_land_global_land_mask
from shapely import line_interpolate_point, LineString, Point

log_level = logging.DEBUG
# log_level = logging.INFO
log_format='%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s'
formatter = logging.Formatter(log_format)
logging.basicConfig(stream=sys.stdout, format=log_format)
logger = logging.getLogger()
logger.setLevel(log_level)

geod = Geodesic.WGS84
GeodesicOutmask = TypedDict('GeodesicOutmask', {'lat1': float, 'lat2': float, 'a12': float, 's12': float,
                                                'azi2': float, 'azi1': float, 'lon1': float, 'lon2': float})


def is_land(point: Point, buffer: float = 1000, angle_step: float = 30) -> bool:
    """
    Check if the point is on land.
    If a buffer is specified, check also the surrounding of the point.
    The check is done by sampling and checking points on a circle with
    the given buffer distance as radius in predefined angular steps.

    :param point:
    :type point: shapely.Point
    :param buffer: buffer distance to check for land
    :type buffer: int or float
    :param angle_step:
    :type angle_step: int or float
    :return:
    :rtype: bool
    """
    # FIXME: could be improved by checking also points on the line between the original point and the point moved
    #  by the specific distance. With the current implementation islands could be ignored.
    if is_land_global_land_mask(point.y, point.x):
        return True
    if buffer > 0:
        for angle in [i*angle_step for i in range(math.ceil(360/angle_step))]:
            p = geod.Direct(point.y, point.x, angle, buffer)
            if is_land_global_land_mask(p['lat2'], p['lon2']):
                return True
    return False


def has_point_on_land(
        line: GeodesicLine,
        interval: float = 1000,
        buffer: float = 1000,
        angle_step: float = 30
) -> bool:
    """
    Check if the line has a point on land.

    :param line:
    :type line: geographiclib.geodesicline.GeodesicLine
    :param interval: interval at which points along the line are checked
    :type interval: int or float
    :param buffer:
    :type buffer: int or float
    :param angle_step:
    :type angle_step: int or float
    :return:
    :rtype: bool
    """
    n = int(math.ceil(line.s13 / interval))
    for i in range(0, n+1):
        s = min(interval * i, line.s13)
        g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        if is_land(Point(g['lon2'], g['lat2']), buffer, angle_step):
            return True
    return False


def get_midpoint(line: GeodesicLine) -> GeodesicOutmask:
    """
    :param line:
    :type line: geographiclib.geodesicline.GeodesicLine
    """
    return line.Position(line.s13 / 2, Geodesic.STANDARD | Geodesic.LONG_UNROLL)


def interpolate_equal_distance(route: list[Point], distance: float = 0.1, normalized: bool = True) -> list[Point]:
    """
    :param route:
    :type route: list[list[Point]]
    :param distance:
    :type distance: int or float
    :param normalized:
    :type normalized: bool
    """
    line = LineString(route)
    if normalized:
        length = 1
    else:
        length = line.length
    steps = [i * distance for i in range(math.ceil(length/distance)+1)]
    route_interpolated = line_interpolate_point(line, steps, normalized=normalized).tolist()
    return route_interpolated


def move_point_perpendicular(point: GeodesicOutmask, distance: int | float, clockwise: bool = True) -> Point:
    """
    Move point perpendicular to the geodesic it originated from. The ...
    :param point:
    :type point: GeodesicOutmask
    :param distance:
    :type distance: int or float
    :param clockwise:
    :type clockwise: bool
    """
    if clockwise:
        new_point = geod.Direct(point['lat2'], point['lon2'], point['azi2'] + 90, distance)
        return Point(new_point['lon2'], new_point['lat2'])
    else:
        new_point = geod.Direct(point['lat2'], point['lon2'], point['azi2'] - 90, distance)
        return Point(new_point['lon2'], new_point['lat2'])


def split_segments(
        start: Point,
        end: Point,
        threshold: float,
        distance_move: float,
        point_sequence: list[Point] = None,
        distance_to_land: float = 1000,
        angle_step: float = 30
) -> list[Point]:
    """
    Check if a segment crosses land. Divide segment if it does. The segment is divided at its midpoint.
    If the midpoint is on land it is moved incrementally by a defined distance until it is not on land.

    :param start: start point of segment
    :type start: shapely.Point
    :param end: end point of segment
    :type end: shapely.Point
    :param threshold: segment length below which no further division is done to prevent the algorithm to run forever
    :type threshold: int or float
    :param distance_move: distance used to move midpoint perpendicular
    :type distance_move: int or float
    :param point_sequence:
    :type point_sequence: list[shapely.Point]
    :param distance_to_land: distance used to check if a point is close to land
    :type distance_to_land: int or float
    :param angle_step: angle interval used to check if a point is close to land
    :type angle_step: int or float
    """
    logger.info(f"Check segment from {start} to {end}.")
    if point_sequence is None:
        raise Exception("Start point wasn't added.")
    line = geod.InverseLine(start.y, start.x, end.y, end.x)
    if has_point_on_land(line):
        logger.info("Segment crosses land. Split segment at midpoint.")
        if line.s13 > threshold:
            midpoint = get_midpoint(line)
            over_land = is_land(Point(midpoint['lon2'], midpoint['lat2']), distance_to_land, angle_step)
            logger.info(f"Midpoint {midpoint}.")
            logger.info(f"Midpoint over land: {over_land}.")
            i = 1
            new_point = Point(midpoint['lon2'], midpoint['lat2'])
            while over_land:
                logger.debug(f"Iteration {i}: move midpoint perpendicular by {i*distance_move/1000} km.")
                dist = i * distance_move
                p1 = move_point_perpendicular(midpoint, dist)
                p1_over_land = is_land(p1, distance_to_land, angle_step)
                p2 = move_point_perpendicular(midpoint, dist, False)
                p2_over_land = is_land(p2, distance_to_land, angle_step)
                if not p1_over_land and p2_over_land:
                    new_point = p1
                    break
                elif p1_over_land and not p2_over_land:
                    new_point = p2
                    break
                elif not p1_over_land and not p2_over_land:
                    # If both points are on water check which of them is better, i.e. the connections of the point to
                    # the start and end are on water as well.
                    p1_to_start_cuts_land = has_point_on_land(geod.InverseLine(p1.y, p1.x, start.y, start.x))
                    p1_to_end_cuts_land = has_point_on_land(geod.InverseLine(p1.y, p1.x, end.y, end.x))
                    p2_to_start_cuts_land = has_point_on_land(geod.InverseLine(p2.y, p2.x, start.y, start.x))
                    p2_to_end_cuts_land = has_point_on_land(geod.InverseLine(p2.y, p2.x, end.y, end.x))
                    if (p1_to_start_cuts_land + p1_to_end_cuts_land) > (p2_to_start_cuts_land + p2_to_end_cuts_land):
                        new_point = p2
                    else:
                        new_point = p1
                    break
                logger.debug("New point still on land.")
                i += 1
            logger.info(f"Found new point {new_point}.")
            # Note: the order of the following lines is important to have the correct order of points in the route
            split_segments(start, new_point, threshold, distance_move, point_sequence, distance_to_land, angle_step)
            point_sequence.append(new_point)
            split_segments(new_point, end, threshold, distance_move, point_sequence, distance_to_land, angle_step)
        else:
            logger.info(f"Segment length below threshold: {line.s13} m < {threshold} m. No splitting.")
    else:
        logger.info("Segment doesn't cross land.")
    return point_sequence


def initial_route_generation(
        start: Point,
        end: Point,
        threshold: float,
        distance: float,
        filename: str = None,
        waypoints: list[list[Point]] = None,
        distance_to_land: float = 1000,
        angle_step: float = 30,
        interpolate: bool = True,
        interp_dist: float = 0.1,
        interp_normalized: bool = True
) -> list[list[Point]]:
    """
    Initial route generation after Kuhlemann and Tierney (2020)

    :param start:
    :type start: shapely.Point
    :param end:
    :type end: shapely.Point
    :param threshold: define a segment length below which no further division is done to prevent the algorithm to
      run forever
    :type threshold: int or float
    :param distance: distance used to move midpoint perpendicular
    :type distance: int or float
    :param filename:
    :type filename: str
    :param waypoints:
    :type waypoints: list[list[Point]]
    :param distance_to_land: distance used to check if a point is close to land
    :type distance_to_land: int or float
    :param angle_step: angle interval used to check if a point is close to land
    :type angle_step: int or float
    :param interpolate: interpolate the found route at an equal distance
    :type interpolate: bool
    :param interp_dist: interpolation distance in meter
    :type interp_dist: int or float
    :param interp_normalized:
    :type interp_normalized: bool
    :return:
    :rtype: list[list[Point]]
    """
    routes = []
    if waypoints is not None:
        for route_idx in range(len(waypoints)):
            logger.info("----------------------------------")
            logger.info(f"Generate route {route_idx}")
            point_sequence = [start]
            fixed_points = [start] + waypoints[route_idx] + [end]
            for waypoint_idx in range(len(fixed_points)-1):
                split_segments(fixed_points[waypoint_idx], fixed_points[waypoint_idx+1],
                               threshold, distance, point_sequence, distance_to_land, angle_step)
                point_sequence.append(fixed_points[waypoint_idx+1])
            if interpolate:
                point_sequence = interpolate_equal_distance(point_sequence, interp_dist, interp_normalized)
            routes.append(point_sequence)
            if filename is not None:
                root, ext = os.path.splitext(filename)
                filename_new = f"{root}_{route_idx}{ext}"
                logger.info(f"Save route to {filename_new}")
                to_geojson(point_sequence, filename_new)
    else:
        point_sequence = [start]
        split_segments(start, end, threshold, distance, point_sequence, distance_to_land, angle_step)
        point_sequence.append(end)
        if interpolate:
            point_sequence = interpolate_equal_distance(point_sequence, interp_dist, interp_normalized)
        routes.append(point_sequence)
        if filename is not None:
            logger.info(f"Save route to {filename}")
            to_geojson(point_sequence, filename)
    return routes


def to_geojson(points: list[Point], filename: str = None):
    d = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": { "id" : i },
                "geometry": json.loads(shapely.to_geojson(points[i]))
            }
            for i in range(len(points))
        ]
    }
    if filename:
        with open(filename, 'w') as f:
            json.dump(d, f, indent=2)
    return d


class TestScenario:

    def __init__(self, start: Point, end: Point, waypoints: list[list[Point]] = None):
        self.start = start
        self.end = end
        self.waypoints = waypoints


def get_test_scenario(name: str):
    match name:
        case "northsea":
            return TestScenario(Point(2, 54), Point(6, 56))
        case "sardinia":
            return TestScenario(Point(7, 40), Point(11, 40))
        case "marseille-cyprus":
            return TestScenario(Point(5.2, 43), Point(33.7, 34.8))
        case "barcelona-alexandria":
            return TestScenario(Point(2.3, 41), Point(32, 32))
        case "perth-brisbane":
            return TestScenario(Point(114, -32), Point(155, -27))
        case "brest-lisbon":
            return TestScenario(Point(-5, 48.2), Point(-9.5, 38.5),
                                [[Point(-10, 43)], [Point(-5.5, 45.5), Point(-10, 41)]])


def main():
    # Warning: depending on the scenario and the configured parameters (e.g. distance_move) the program might run into
    #  a recursion error (maximum recursion depth)!
    # print(sys.getrecursionlimit())  # default: 1000
    # sys.setrecursionlimit(100000)
    scenario = get_test_scenario("brest-lisbon")
    distance_move = 10000  # in m
    min_distance = 5000  # in m
    filename = "initial_route.geojson"
    distance_to_land = 1000  # in m, set to 0 to check only if the point is exactly on land
    angle_step = 30  # in deg
    interpolate = False
    try:
        initial_route_generation(scenario.start, scenario.end, min_distance, distance_move, filename, scenario.waypoints,
                                 distance_to_land, angle_step, interpolate)
    except RecursionError as err:
        logger.error(err)


if __name__ == '__main__':
    main()
