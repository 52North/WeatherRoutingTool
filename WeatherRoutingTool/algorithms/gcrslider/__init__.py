import json
import logging
import math
from copy import deepcopy
from hashlib import sha256
from typing import TypedDict

import numpy as np
from astropy import units as u
from geographiclib.geodesic import Geodesic
from geographiclib.geodesicline import GeodesicLine
from global_land_mask import is_land as is_land_global_land_mask
from shapely import line_interpolate_point, LineString

from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger("WRT.gcrslider")

geod = Geodesic.WGS84
GeodesicOutmask = TypedDict('GeodesicOutmask', {'lat1': float, 'lat2': float, 'a12': float, 's12': float,
                                                'azi2': float, 'azi1': float, 'lon1': float, 'lon2': float})


class GcrSliderAlgorithm(RoutingAlg):
    """
    Heuristic algorithm to find a route between two points which does not cross land based on the great circle route.
    The algorithm connects start and end point. If the connecting line cuts land a new waypoint is added in the
    middle of the line. If the new waypoint is on land it is moved orthogonally until it is on water. This process
    continues until no segment cuts land.

    References:
    - Kuhlemann, S., & Tierney, K. (2020). A genetic algorithm for finding realistic sea routes considering the weather.
      Journal of Heuristics, 26(6), 801-825.
    """
    def __init__(self, config):
        """
        :param config:
        :type config: WeatherRoutingTool.config.Config
        :return: None
        :rtype: NoneType
        """
        super().__init__(config)
        self.distance_move = config.GCR_SLIDER_DISTANCE_MOVE
        self.threshold = config.GCR_SLIDER_THRESHOLD
        self.land_buffer = config.GCR_SLIDER_LAND_BUFFER
        self.angle_step = config.GCR_SLIDER_ANGLE_STEP
        self.cluster_distance = 10000
        self.dynamic_parameters = config.GCR_SLIDER_DYNAMIC_PARAMETERS
        self.interpolate = config.GCR_SLIDER_INTERPOLATE
        self.interp_dist = config.GCR_SLIDER_INTERP_DIST
        self.interp_normalized = config.GCR_SLIDER_INTERP_NORMALIZED
        self.postprocessing = False
        self.waypoints = config.INTERMEDIATE_WAYPOINTS
        self.sequence_id = 0
        self.found_id = 0
        self.max_nof_points = config.GCR_SLIDER_MAX_POINTS
        self.points = {
            self.get_point_id(self.start): {
                'seq_id': self.sequence_id,
                'found_id': self.found_id,
                'point': self.start
            }
        }

    def adjust_parameters(self, line_length: float):
        """
        :param line_length: line length in m
        :type line_length: float
        """
        self.distance_move = 0.05 * line_length

    def create_single_cluster(self, nucleus: tuple[float, float], points: dict, cluster_points: dict):
        new_cluster_points = {}
        for k, v in list(points.items()):
            d = geod.Inverse(nucleus[0], nucleus[1], v['point'][0], v['point'][1])["s12"]
            if d < self.cluster_distance:
                new_cluster_points[k] = v
                del points[k]
        cluster_points.update(new_cluster_points)
        for k, v in new_cluster_points.items():
            self.create_single_cluster(v['point'], points, cluster_points)

    def create_clusters(self):
        points_copy = deepcopy(self.points)
        points_copy = dict(sorted(points_copy.items(), key=lambda item: item[1]['seq_id']))
        clusters = []
        while len(points_copy) > 1:
            cluster_points = {list(points_copy.keys())[0]: list(points_copy.values())[0]}
            self.create_single_cluster(
                list(points_copy.values())[0]['point'], dict(list(points_copy.items())[1:]), cluster_points)
            # if len(cluster_points) > 2:
            if len(cluster_points) > 1:
                clusters.append(cluster_points)
                for k in cluster_points.keys():
                    del points_copy[k]
            else:
                del points_copy[list(points_copy.keys())[0]]
        return clusters

    def execute(self) -> tuple[RouteParams, int]:
        self.sequence_id = 0
        self.found_id = 0
        self.points = {
            self.get_point_id(self.start): {
                'seq_id': self.sequence_id,
                'found_id': self.found_id,
                'point': self.start
            }
        }
        if len(self.waypoints) > 0:
            fixed_points = [self.start] + self.waypoints + [self.finish]
            for waypoint_idx in range(len(fixed_points)-1):
                self.split_segments(fixed_points[waypoint_idx], fixed_points[waypoint_idx+1])
                point_id = self.get_point_id(fixed_points[waypoint_idx+1])
                self.sequence_id += 1
                self.found_id += 1
                self.points[point_id] = {
                    'seq_id': self.sequence_id,
                    'found_id': self.found_id,
                    'point': fixed_points[waypoint_idx+1]
                }
        else:
            self.split_segments(self.start, self.finish)
            point_id = self.get_point_id(self.finish)
            self.sequence_id += 1
            self.found_id += 1
            self.points[point_id] = {
                'seq_id': self.sequence_id,
                'found_id': self.found_id,
                'point': self.finish
            }

        if self.sequence_id != self.found_id:
            raise RuntimeError(f"Sequence id ({self.sequence_id}) and found id ({self.found_id}) have to be equal!")

        if self.postprocessing:
            point_sequence = self.postprocess()
        else:
            point_sequence = [p['point'] for p in sorted(self.points.values(), key=lambda v: v['seq_id'])]
        if self.interpolate:
            point_sequence = self.interpolate_equal_distance(point_sequence)
        lats, lons = list(zip(*point_sequence))

        ship_params = ShipParams(
            speed=np.zeros(len(point_sequence) - 1) * u.meter / u.second,
            fuel_rate=np.zeros(len(point_sequence) - 1) * u.kg / u.second,
            power=np.zeros(len(point_sequence) - 1) * u.Watt,
            rpm=np.zeros(len(point_sequence) - 1) * 1 / u.minute,
            r_calm=np.zeros(len(point_sequence) - 1) * u.newton,
            r_wind=np.zeros(len(point_sequence) - 1) * u.newton,
            r_waves=np.zeros(len(point_sequence) - 1) * u.newton,
            r_shallow=np.zeros(len(point_sequence) - 1) * u.newton,
            r_roughness=np.zeros(len(point_sequence) - 1) * u.newton,
            wave_height=np.zeros(len(point_sequence) - 1) * u.meter,
            wave_direction=np.zeros(len(point_sequence) - 1) * u.radian,
            wave_period=np.zeros(len(point_sequence) - 1) * u.second,
            u_currents=np.zeros(len(point_sequence) - 1) * u.meter / u.second,
            v_currents=np.zeros(len(point_sequence) - 1) * u.meter / u.second,
            u_wind_speed=np.zeros(len(point_sequence) - 1) * u.meter / u.second,
            v_wind_speed=np.zeros(len(point_sequence) - 1) * u.meter / u.second,
            pressure=np.zeros(len(point_sequence) - 1) * u.kg / u.meter / u.second ** 2,
            air_temperature=np.zeros(len(point_sequence) - 1) * u.deg_C,
            salinity=np.zeros(len(point_sequence) - 1) * u.dimensionless_unscaled,
            water_temperature=np.zeros(len(point_sequence) - 1) * u.deg_C,
            status=np.zeros(len(point_sequence) - 1),
            message=np.array([""] * (len(point_sequence) - 1))
        )

        route = RouteParams(
            count=len(point_sequence) - 2,
            start=self.start,
            finish=self.finish,
            gcr=None,
            route_type='route_gcr_slider',
            time=0,
            lats_per_step=lats,
            lons_per_step=lons,
            course_per_step=0,
            dists_per_step=[0] * len(point_sequence),
            starttime_per_step=[0] * len(point_sequence),
            ship_params_per_step=ship_params,
        )
        return route, 0

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False
                        ) -> tuple[RouteParams, int]:
        return self.execute()

    def has_point_on_land(
            self,
            line: GeodesicLine,
            interval: float = 1000
    ) -> bool:
        """
        Check if the line has a point on land.

        :param line:
        :type line: geographiclib.geodesicline.GeodesicLine
        :param interval: interval at which points along the line are checked
        :type interval: int or float
        :return:
        :rtype: bool
        """
        n = int(math.ceil(line.s13 / interval))
        for i in range(0, n+1):
            s = min(interval * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            if self.is_land(g['lat2'], g['lon2']):
                return True
        return False

    def get_midpoint(self, line: GeodesicLine) -> GeodesicOutmask:
        """
        :param line:
        :type line: geographiclib.geodesicline.GeodesicLine
        """
        return line.Position(line.s13 / 2, Geodesic.STANDARD | Geodesic.LONG_UNROLL)

    def get_point_id(self, point: tuple[float, float], decimals: int = 7) -> str:
        p = (round(point[0], decimals), round(point[1], decimals))
        return sha256(json.dumps(p).encode('utf-8')).hexdigest()

    def interpolate_equal_distance(self, route: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        :param route: [(lat1, lon1), (lat2, lon2), ...]
        :type route: list[tuple[float, float]]
        """
        line = LineString([(x, y) for y, x in route])
        if self.interp_normalized:
            length = 1
        else:
            length = line.length
        steps = [i * self.interp_dist for i in range(math.ceil(length / self.interp_dist) + 1)]
        route_interpolated = line_interpolate_point(line, steps, normalized=self.interp_normalized).tolist()
        return [(p.y, p.x) for p in route_interpolated]

    def is_land(self, lat: float, lon: float) -> bool:
        """
        Check if the point is on land.
        If a buffer is specified, check also the surrounding of the point.
        The check is done by sampling and checking points on a circle with
        the given buffer distance as radius in predefined angular steps.

        :param lat: latitude
        :type lat: float
        :param lon: longitude
        :type lon: float
        :return:
        :rtype: bool
        """
        # FIXME: could be improved by checking also points on the line between the original point and the point moved
        #  by the specific distance. With the current implementation islands could be ignored.
        if is_land_global_land_mask(lat, lon):
            return True
        if self.land_buffer > 0:
            for angle in [i*self.angle_step for i in range(math.ceil(360/self.angle_step))]:
                p = geod.Direct(lat, lon, angle, self.land_buffer)
                if is_land_global_land_mask(p['lat2'], p['lon2']):
                    return True
        return False

    def move_point_orthogonally(self, point: GeodesicOutmask, distance: float, clockwise: bool = True
                                ) -> tuple[float, float]:
        """
        Move point orthogonally to the geodesic it originated from.
        :param point:
        :type point: GeodesicOutmask
        :param distance:
        :type distance: int or float
        :param clockwise:
        :type clockwise: bool
        """
        if clockwise:
            new_point = geod.Direct(point['lat2'], point['lon2'], point['azi2'] + 90, distance)
            return new_point['lat2'], new_point['lon2']
        else:
            new_point = geod.Direct(point['lat2'], point['lon2'], point['azi2'] - 90, distance)
            return new_point['lat2'], new_point['lon2']

    def postprocess(self) -> list[tuple[float, float]]:
        # clusters = self.create_clusters()
        # FIXME: continue implementation...
        return []

    def split_segments(self, start: tuple[float, float], end: tuple[float, float]):
        """
        Check if a segment crosses land. Divide segment if it does. The segment is divided at its midpoint.
        If the midpoint is on land it is moved incrementally by a defined distance until it is not on land.

        :param start: start point of segment
        :type start: tuple[float, float]
        :param end: end point of segment
        :type end: tuple[float, float]
        """
        logger.info(f"Check segment from {start} to {end}.")
        line = geod.InverseLine(start[0], start[1], end[0], end[1])
        if self.has_point_on_land(line):
            logger.info("Segment crosses land. Split segment at midpoint.")
            if line.s13 > self.threshold:
                if self.dynamic_parameters:
                    self.adjust_parameters(line.s13)
                midpoint = self.get_midpoint(line)
                over_land = self.is_land(midpoint['lat2'], midpoint['lon2'])
                logger.info(f"Midpoint {midpoint}.")
                logger.info(f"Midpoint over land: {over_land}.")
                i = 1
                new_point = (midpoint['lat2'], midpoint['lon2'])
                while over_land:
                    logger.debug(f"Iteration {i}: move midpoint orthogonally by {i*self.distance_move/1000} km.")
                    dist = i * self.distance_move
                    lat1, lon1 = self.move_point_orthogonally(midpoint, dist)
                    p1_over_land = self.is_land(lat1, lon1)
                    lat2, lon2 = self.move_point_orthogonally(midpoint, dist, False)
                    p2_over_land = self.is_land(lat2, lon2)
                    if not p1_over_land and p2_over_land:
                        new_point = (lat1, lon1)
                        break
                    elif p1_over_land and not p2_over_land:
                        new_point = (lat2, lon2)
                        break
                    elif not p1_over_land and not p2_over_land:
                        # If both points are on water check which of them is better, i.e. the connections of the point
                        # to the start and end are on water as well.
                        p1_to_start_cuts_land = self.has_point_on_land(geod.InverseLine(lat1, lon1, start[0], start[1]))
                        p1_to_end_cuts_land = self.has_point_on_land(geod.InverseLine(lat1, lon1, end[0], end[1]))
                        p2_to_start_cuts_land = self.has_point_on_land(geod.InverseLine(lat2, lon2, start[0], start[1]))
                        p2_to_end_cuts_land = self.has_point_on_land(geod.InverseLine(lat2, lon2, end[0], end[1]))
                        if ((p1_to_start_cuts_land + p1_to_end_cuts_land) >
                                (p2_to_start_cuts_land + p2_to_end_cuts_land)):
                            new_point = (lat2, lon2)
                        else:
                            new_point = (lat1, lon1)
                        break
                    logger.debug("New point still on land.")
                    i += 1
                logger.info(f"Found new point {new_point}.")
                # Note: the order of the following lines is important to have the correct order of points in the route
                point_id = self.get_point_id(new_point)
                self.found_id += 1
                if self.found_id > self.max_nof_points:
                    raise RuntimeError(f"Reached maximum number of allowed points ({self.max_nof_points}).")
                self.points[point_id] = {'point': new_point, 'found_id': self.found_id}
                self.split_segments(start, new_point)
                self.sequence_id += 1
                self.points[point_id].update({'seq_id': self.sequence_id})
                logger.debug(f"Add point {new_point} with route sequence id: {self.sequence_id}")
                self.split_segments(new_point, end)
            else:
                logger.info(f"Segment length below threshold: {line.s13} m < {self.threshold} m. No splitting.")
        else:
            logger.info("Segment doesn't cross land.")
