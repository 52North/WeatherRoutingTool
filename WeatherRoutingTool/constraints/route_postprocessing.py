import logging
import math
import os
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import sqlalchemy
from astropy import units as u
from geovectorslib import geod
from shapely.geometry import box, LineString, Point
from shapely.ops import polygonize_full

from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat

logger = logging.getLogger('WRT.RoutePostprocessing')


class RoutePostprocessing:
    """
    Currently RoutePostprocessing is focused on Traffic Separation Scheme.
    In the future, it should be integrated into a more general approach.
    """
    route: RouteParams
    lats_per_step: list
    lons_per_step: list
    starttime_per_step: list
    ship_speed: list
    boat: Boat

    def __init__(self, min_fuel_route, boat, db_engine=None):
        self.set_data(min_fuel_route, boat)
        self.ship = boat  # Alias for compatibility with code/tests expecting 'ship'

        if db_engine is not None:
            self.engine = db_engine
        else:
            self.host = os.getenv("WRT_DB_HOST")
            self.database = os.getenv("WRT_DB_DATABASE")
            self.user = os.getenv("WRT_DB_USERNAME")
            self.password = os.getenv("WRT_DB_PASSWORD")
            self.schema = os.getenv("POSTGRES_SCHEMA")
            self.port = os.getenv("WRT_DB_PORT")
            self.engine = self.connect_database()

    def set_data(self, route, boat):
        self.route = route
        self.lats_per_step = route.lats_per_step
        self.lons_per_step = route.lons_per_step

        self.starttime_per_step = route.starttime_per_step
        self.boat = boat
        self.ship_speed = self.boat.get_boat_speed()

    def post_process_route(self):
        route_bbx = self.get_route_bbox()
        route_segments_gdf = self.create_route_segments()
        seamark_gdf = self.retrieve_seamark_data(route_bbx, self.engine)

        intersecting_route_node_list = self.find_seamark_intersections(route_segments_gdf, seamark_gdf)
        no_of_intersections = len(intersecting_route_node_list)

        if no_of_intersections:
            if self.is_start_or_finish_node_in_separation_zone(
                    route_segments_gdf, seamark_gdf):
                route_postprocessed = self.route
                return route_postprocessed

            first_route_seg_gdf = route_segments_gdf[0:intersecting_route_node_list[0]]
            last_route_node_intersecting = intersecting_route_node_list[len(intersecting_route_node_list) - 1]
            first_node_index_of_last_route_seg = last_route_node_intersecting + 1
            last_route_seg_gdf = route_segments_gdf[first_node_index_of_last_route_seg:]

            separation_lanes_data_gdf = self.retrieve_seperation_lane_data(route_bbx, self.engine)

            if first_route_seg_gdf.empty or last_route_seg_gdf.empty:
                # Handle scenarios when there is an intersection in the first segment of the route
                # or the last segment of the route or in both first and last segments of the route
                last_node_of_route_seg = self.find_first_node_of_route_seg(route_segments_gdf)
                separation_lane_gdf = self.find_seperation_lane_to_follow(last_node_of_route_seg,
                                                                          separation_lanes_data_gdf)
                final_route = self.connect_route_segments(first_route_seg_gdf, separation_lane_gdf,
                                                          last_route_seg_gdf, route_segments_gdf)
            else:
                last_node_of_first_route_seg = self.find_last_node_of_route_seg(first_route_seg_gdf)
                first_node_of_last_route_seg = self.find_first_node_of_route_seg(last_route_seg_gdf)
                is_valid_90_crossing, separation_lane_segment = self.check_valid_crossing(separation_lanes_data_gdf,
                                                                                          last_node_of_first_route_seg,
                                                                                          first_node_of_last_route_seg)
                if is_valid_90_crossing:
                    x_point_on_lane, y_point_on_lane = self.find_point_from_perpendicular_angle(
                                                            last_node_of_first_route_seg, separation_lane_segment)
                    perpendicular_line_segment = LineString([last_node_of_first_route_seg,
                                                             (x_point_on_lane, y_point_on_lane)])
                    end_node_x, end_node_y = self.find_point_from_perpendicular_angle(first_node_of_last_route_seg,
                                                                                      perpendicular_line_segment)
                    extended_line = LineString([last_node_of_first_route_seg, (x_point_on_lane, y_point_on_lane),
                                                (end_node_x, end_node_y)])
                    extended_line_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=[extended_line])
                    extended_line_gdf.rename_geometry('geom', inplace=True)
                    connecting_line = self.create_last_connecting_line(last_route_seg_gdf,
                                                                       extended_line_gdf.iloc[0])
                    connecting_line_to_last_route_seg_gdf = gpd.GeoDataFrame(geometry=[connecting_line],
                                                                             crs=first_route_seg_gdf.crs)
                    connecting_line_to_last_route_seg_gdf.rename_geometry('geom', inplace=True)
                    final_route = self.connect_route_segments(first_route_seg_gdf,
                                                              connecting_line_to_last_route_seg_gdf.iloc[0],
                                                              last_route_seg_gdf)

                else:
                    separation_lane_gdf = self.find_seperation_lane_to_follow(last_node_of_first_route_seg,
                                                                              separation_lanes_data_gdf)
                    final_route = self.connect_route_segments(first_route_seg_gdf, separation_lane_gdf,
                                                              last_route_seg_gdf)
            starttime_list = self.recalculate_starttime_per_node(final_route)

            route_postprocessed = self.terminate(self.lons_per_step, self.lats_per_step,
                                                 starttime_list, self.ship_speed)

        else:
            logger.info(' Route postprocessing is not continued as the route does '
                        'not intersect any Traffic Separation Schemes')
            route_postprocessed = self.route

        return route_postprocessed

    def connect_database(self):
        engine = sqlalchemy.create_engine("postgresql://{user}:{pwd}@{host}:{port}/{db}".
                                          format(user=self.user, pwd=self.password, host=self.host,
                                                 db=self.database, port=self.port))
        return engine

    def create_route_segments(self):
        """
        Create relevant line segments from the nodes(coordinates) of the minimum fuel route
        """
        route_segments = []
        timestamp = []
        for i in range(len(self.lons_per_step) - 1):
            segment = LineString([(self.lons_per_step[i], self.lats_per_step[i]),
                                  (self.lons_per_step[i + 1], self.lats_per_step[i + 1])])
            route_segments.append(segment)

            timestamp.append(self.starttime_per_step[i])

        # Add LineString segments to a new GeoDataFrame
        route_segments_gdf = gpd.GeoDataFrame({'timestamp': timestamp}, geometry=route_segments, crs="EPSG:4326")
        return route_segments_gdf

    def get_route_bbox(self):
        min_lat = min(self.lats_per_step) - 0.5
        min_lon = min(self.lons_per_step) - 0.5
        max_lat = max(self.lats_per_step) + 0.5
        max_lon = max(self.lons_per_step) + 0.5

        bbox = box(min_lon, min_lat, max_lon, max_lat)
        return bbox

    def query_data(self, query, engine):
        gdf_seamark = gpd.read_postgis(query, engine)
        return gdf_seamark

    def retrieve_seamark_data(self, bbox_wkt, engine):
        """
        Retrieve all the seamark objects within the bounding box
        """
        query = "SELECT *,linestring AS geom FROM " + self.schema + ".ways " \
                "WHERE  (tags -> 'seamark:type'='separation_boundary' OR tags -> 'seamark:type'='separation_line' " \
                "OR tags -> 'seamark:type'='separation_zone' OR tags -> 'seamark:type'='separation_lane' " \
                "OR tags -> 'seamark:type'='inshore_traffic_zone')" \
                "AND ST_Intersects(linestring, ST_GeomFromText('{}', 4326))".format(bbox_wkt)
        seamark_gdf = self.query_data(query, engine)
        return seamark_gdf

    def find_seamark_intersections(self, route_segments_gdf, seamark_gdf):
        """
        First, find intersection points of route segments with any seamark object.
        If any intersection is available, then iterate over each route segment and the intersection points
        to identify which route segment is getting intersected with the intersection point.
        Then, the relevant route segment is added to the list of indices of intersected route segments

        :param route_segments_gdf: geodataframe of ship route segments
        :param seamark_gdf: geodataframe of all seamark TSS objects within the bounding box of the route
        :returns: The list of route segment indices which intersects the TSS seamark objects
        """
        intersection_gdf = gpd.overlay(route_segments_gdf, seamark_gdf,
                                       how='intersection', keep_geom_type=False)

        intersected_route_indices_list = []
        for index_seg, seg in route_segments_gdf.iterrows():
            for index_in, intersecting_point in intersection_gdf.iterrows():
                # All floating point calculations are limited by the machine epsilon(round off method).
                # The intersected points are interpolated from geometries of the linestrings, and
                # they are exact only when there's a right angle. All of the DE-9IM predicates like 'intersects'
                # requires exact node. Therefore, 2 strategies can be applied in this scenario. Setting a buffer
                # or test the distance between the geometries.
                # Tolerance: 1° is approx 110km. Hence, 10^-12 of 110 km is 110 picometers.
                EPS = 1e-12
                dist = intersecting_point.geometry.distance(seg.geometry) < EPS
                if dist:
                    intersected_route_indices_list.append(index_seg)

        return intersected_route_indices_list

    def is_start_or_finish_node_in_separation_zone(self, route_segments_gdf, seamark_gdf):
        """
        Find whether seamark TSS objects contains the starting or ending node of the
        route. If contains, the route is not forwarded for postprocessing
        """
        first_route_node = self.find_first_node_of_route_seg(route_segments_gdf)
        last_route_node = self.find_last_node_of_route_seg(route_segments_gdf)
        result, cuts, dangles, invalids = polygonize_full(seamark_gdf['geom'])
        contains_first_node = first_route_node.intersects(result)
        contains_last_node = last_route_node.intersects(result)

        if contains_first_node:
            logger.warning(' Route postprocessing is not continued as the start point is '
                           'inside the Traffic Separation Zone!')
            return True
        elif contains_last_node:
            logger.warning(' Route postprocessing is not continued as the finish point is '
                           'inside the Traffic Separation Zone!')
            return True
        else:
            return False

    def find_last_node_of_route_seg(self, first_route_seg_gdf):
        """
        Find the node of the route segments before the first intersecting point
        """
        last_line_geom = first_route_seg_gdf.tail(1).geometry.get_coordinates()
        last_node_geom = last_line_geom.tail(1)
        last_node = Point(last_node_geom.x, last_node_geom.y)
        return last_node

    def find_first_node_of_route_seg(self, route_seg_gdf):
        """
        Find the node of the route segments after the last intersecting point
        """
        first_line_geom = route_seg_gdf.head(1).geometry.get_coordinates()
        first_node_geom = first_line_geom.head(1)
        first_node = Point(first_node_geom.x, first_node_geom.y)
        return first_node

    def retrieve_seperation_lane_data(self, bbox_wkt, engine):
        query = "SELECT *,linestring AS geom FROM public.ways WHERE  ( tags -> 'seamark:type'='separation_lane') " \
                "AND ST_Intersects(linestring, ST_GeomFromText('{}', 4326))".format(bbox_wkt)
        separation_lanes_gdf = self.query_data(query, engine)
        return separation_lanes_gdf

    def find_seperation_lane_to_follow(self, last_node, seperation_lanes_gdf):
        """
        The direction of the separation lane is determined by the order of the points in the linestring.
        Hence, it is assumed the nearest starting point of the separation lane
        (when in parallel and opposite directions) need to be followed.
        This is achieved by finding the distances between the route node before the first intersection and the starting
        node of the each separation lane. Then the separation lane having the minimum distance is selected.

        :param last_node: the node of the route segments before the first intersecting point
        :param seperation_lanes_gdf: geodataframe of separation lanes
        :returns: geodataframe of the separation lane need to be followed
        """
        dist_list = []
        for line in seperation_lanes_gdf.geom:
            seamark_lane_segment = line
            x, y = seamark_lane_segment.xy
            seamark_starting_node = Point(x[0], y[0])
            dist = last_node.distance(seamark_starting_node)
            dist_list.append(dist)

        min_dist_index = dist_list.index(min(dist_list))
        seperation_lane_gdf = seperation_lanes_gdf.loc[min_dist_index]
        return seperation_lane_gdf

    def connect_route_segments(self, first_route_seg_gdf, separation_lane_gdf,
                               last_route_seg_gdf, route_segments_gdf=None):
        # Create new geometries for new connecting LineStrings before and after separation lanes
        if first_route_seg_gdf.empty:
            first_connecting_seg_geom = self.create_first_connecting_seg_from_node(route_segments_gdf,
                                                                                   separation_lane_gdf)
            first_connecting_seg_gdf = gpd.GeoDataFrame(geometry=[first_connecting_seg_geom],
                                                        crs=first_route_seg_gdf.crs)
            first_connecting_seg_gdf['timestamp'] = [self.starttime_per_step[0]]
        else:
            first_connecting_seg_geom = self.create_first_connecting_seg(first_route_seg_gdf, separation_lane_gdf)
            first_connecting_seg_gdf = gpd.GeoDataFrame(geometry=[first_connecting_seg_geom],
                                                        crs=first_route_seg_gdf.crs)

        if last_route_seg_gdf.empty:
            last_connecting_seg_geom = self.create_last_connecting_line_from_node(route_segments_gdf,
                                                                                  separation_lane_gdf)
        else:
            last_connecting_seg_geom = self.create_last_connecting_line(last_route_seg_gdf, separation_lane_gdf)

        separation_lane_geom = separation_lane_gdf.geom

        # Create new GeoDataFrame with the new connecting LineStrings before and after separation lanes

        seperation_lane_seg_gdf = gpd.GeoDataFrame(geometry=[separation_lane_geom],
                                                   crs=last_route_seg_gdf.crs)
        last_connecting_seg_gdf = gpd.GeoDataFrame(geometry=[last_connecting_seg_geom],
                                                   crs=last_route_seg_gdf.crs)

        # Append the new GeoDataFrame to the existing one
        route_first_connecting_gdf = gpd.GeoDataFrame(pd.concat([first_route_seg_gdf, first_connecting_seg_gdf],
                                                                ignore_index=True), crs=first_route_seg_gdf.crs)
        route_separation_lane_gdf = gpd.GeoDataFrame(pd.concat([route_first_connecting_gdf, seperation_lane_seg_gdf],
                                                               ignore_index=True), crs=first_route_seg_gdf.crs)
        route_last_connecting_gdf = gpd.GeoDataFrame(pd.concat([route_separation_lane_gdf, last_connecting_seg_gdf],
                                                               ignore_index=True), crs=first_route_seg_gdf.crs)
        final_route = gpd.GeoDataFrame(pd.concat([route_last_connecting_gdf, last_route_seg_gdf], ignore_index=True),
                                       crs=last_route_seg_gdf.crs)
        return final_route

    def create_first_connecting_seg(self, first_route_seg_gdf, separation_lane_gdf):
        """
        Build the LineString between the route node before the first intersection and
        the first node of the separation lane segment
        """
        x, y = separation_lane_gdf.geom.xy
        first_node_of_separation_lane = Point(x[0], y[0])
        last_node_of_first_route_seg = self.find_last_node_of_route_seg(first_route_seg_gdf)
        first_connecting_seg = LineString([last_node_of_first_route_seg, first_node_of_separation_lane])
        return first_connecting_seg

    def create_first_connecting_seg_from_node(self, route_segment_gdf,
                                              separation_lane_gdf):
        x, y = separation_lane_gdf.geom.xy
        first_node_of_separation_lane = Point(x[0], y[0])
        first_node_of_route_seg = self.find_first_node_of_route_seg(route_segment_gdf)
        first_connecting_seg = LineString([first_node_of_route_seg, first_node_of_separation_lane])
        return first_connecting_seg

    def create_last_connecting_line(self, last_route_seg_gdf, separation_lane_gdf):
        """
        Build the LineString between the last node of the separtion lane and the first route node
        after the last intersection
        """
        x, y = separation_lane_gdf.geom.xy
        last_index = len(x)-1
        last_node_of_separation_lane = Point(x[last_index], y[last_index])
        first_node_of_last_route_seg = self.find_first_node_of_route_seg(last_route_seg_gdf)
        last_connecting_seg = LineString([last_node_of_separation_lane, first_node_of_last_route_seg])
        return last_connecting_seg

    def create_last_connecting_line_from_node(self, route_segment_gdf, separation_lane_gdf):
        """
        Build the LineString between the last node of the separtion lane and the first route node
        after the last intersection
        """
        x, y = separation_lane_gdf.geom.xy
        last_index = len(x)-1
        last_node_of_separation_lane = Point(x[last_index], y[last_index])
        last_node_of_route_seg = self.find_last_node_of_route_seg(route_segment_gdf)
        last_connecting_seg = LineString([last_node_of_separation_lane, last_node_of_route_seg])
        return last_connecting_seg

    def recalculate_starttime_per_node(self, final_route):
        """
        To recalculate the start time of the new route segments, first a new integer index is set
        to final route segments dataframe. Then, the index of first Not available Timestamp
        value is searched and start calculating the new time taken from that index to rest
        of the dataframe
        """
        self.lats_per_step = []
        self.lons_per_step = []
        start_times_list = []

        for index, route_segment in final_route.iterrows():
            line = route_segment['geometry']

            for point in line.coords:
                self.lons_per_step.append(point[0])
                self.lats_per_step.append(point[1])

            if index < len(final_route) - 1:
                # Check if the end point of the current LineString is the same as the start point of the next LineString
                if line.coords[-1] == final_route['geometry'].iloc[index + 1].coords[0]:
                    # If they match, remove the duplicate node
                    self.lons_per_step.pop()
                    self.lats_per_step.pop()

        start_time = final_route.loc[0, 'timestamp']
        start_time_datetime_obj = start_time.to_pydatetime()
        start_times_list.append(start_time_datetime_obj)
        array_len = len(self.lons_per_step)

        first_index = 1
        for index in range(first_index, array_len):
            current_timestamp = self.calculate_timsestamp(self.lats_per_step, self.lons_per_step,
                                                          start_times_list, index, self.ship_speed)
            start_times_list.append(current_timestamp)

        return start_times_list

    def calculate_timsestamp(self, lat, lon, start_times, node_index, speed):
        """
        Calculate the time taken using time = distance / ship speed of the previous
        route segment and then added the new time taken into previous timestamp to
        get the new start time
        """
        previous_step_index = node_index - 1
        previous_timestamp = start_times[previous_step_index]
        gcd = geod.inverse([lat[previous_step_index]], [lon[previous_step_index]],
                           [lat[node_index]], [lon[node_index]])
        dist = gcd['s12']
        time_taken_for_current_step = dist[0] / speed.value
        current_timestamp = previous_timestamp + timedelta(seconds=int(time_taken_for_current_step))
        return current_timestamp

    def terminate(self, route_lons, route_lats, starttime_list, boat_speed):
        """
        Find the courses from route_dict to calculate the ship_parameters
        """
        route_lons_np = np.array(route_lons)
        route_lats_np = np.array(route_lats)
        start_time_np = np.array(starttime_list)
        route_dict = RouteParams.get_per_waypoint_coords(route_lons_np,
                                                         route_lats_np,
                                                         start_time_np[0],
                                                         boat_speed)
        start_times = route_dict['start_times']
        start_times_datetime64 = np.array(start_times, dtype='datetime64[ns]')
        ship_params = self.boat.get_ship_parameters(route_dict['courses'], route_dict['start_lats'],
                                                    route_dict['start_lons'], start_times_datetime64)
        npoints = len(self.lats_per_step) - 1

        start = (self.lats_per_step[0], self.lons_per_step[0])
        last_index = len(self.lats_per_step)-1
        finish = (self.lats_per_step[last_index], self.lons_per_step[last_index])

        travel_times = route_dict['travel_times']
        courses = route_dict['courses']
        dists = route_dict['dist']
        start_times = route_dict['start_times']
        arrival_time = start_times[-1] + timedelta(seconds=dists[-1].value / boat_speed.value)

        travel_times = np.append(travel_times, -99 * u.second)
        courses = np.append(courses, -99 * u.degree)
        dists = np.append(dists, -99 * u.meter)
        start_times = np.append(start_times, arrival_time)

        route = RouteParams(count=npoints-1,
                            start=start,
                            finish=finish,
                            gcr=None,
                            route_type='min_time_route',
                            time=travel_times[-1],
                            lats_per_step=route_lats,
                            lons_per_step=route_lons,
                            course_per_step=courses[-1],
                            dists_per_step=dists[-1],
                            starttime_per_step=start_times,
                            ship_params_per_step=ship_params)

        return route

    def find_point_from_perpendicular_angle(self, start_node,
                                            segment):
        """
        Find the intersecting point on a line segment which it makes a perpendicular
        angle from a given point
        :param start_node: given point
        :param segment: given line segment
        :returns: coordinates of the point on the line which  makes a right angle
        from the given point
        """
        line_x, line_y = segment.xy
        x_start = line_x[0]
        y_start = line_y[0]
        x_end = line_x[1]
        y_end = line_y[1]
        xp = start_node.x
        yp = start_node.y

        slope = self.calculate_slope(x_start, y_start, x_end, y_end)
        m = (-1) / slope
        x = (m * xp - yp - slope * x_start + y_start) / (m - slope)
        y = m * x - m * xp + yp
        return x, y

    def check_valid_crossing(self, separation_lanes_data_gdf,
                             last_node_of_first_route_seg,
                             first_node_of_last_route_seg):
        """
        This checks whether the straight line starting from the point before the intersection
        and ending from the point after the last intersection makes an angle between 60 to 120
        with respect to the nearest separation lane of the starting point.
        """
        intersecting_route_seg_geom = LineString(
            [(last_node_of_first_route_seg.x, last_node_of_first_route_seg.y),
             (first_node_of_last_route_seg.x, first_node_of_last_route_seg.y)])
        intersecting_route_seg_gdf = gpd.GeoDataFrame(
            geometry=[intersecting_route_seg_geom],
            crs=separation_lanes_data_gdf.crs)
        intersecting_separation_lanes = gpd.overlay(intersecting_route_seg_gdf,
                                                    separation_lanes_data_gdf,
                                                    how='intersection',
                                                    keep_geom_type=False)
        if len(intersecting_separation_lanes) == 0:
            return False, None
        merged_separation_lanes_data_df = pd.merge(
            intersecting_separation_lanes, separation_lanes_data_gdf, on='id',
            how='left')
        dist_to_separation_lane = last_node_of_first_route_seg.distance(
            merged_separation_lanes_data_df['geom'])
        min_dist_indx = dist_to_separation_lane.idxmin()
        separation_lane = merged_separation_lanes_data_df.iloc[[min_dist_indx]]
        angle_current_crossing, separation_lane_segment = self.calculate_angle_of_current_crossing(
            last_node_of_first_route_seg, first_node_of_last_route_seg,
            separation_lane, intersecting_route_seg_geom)
        if angle_current_crossing >= 60 and angle_current_crossing <= 120:
            return True, separation_lane_segment
        else:
            return False, None

    def calculate_slope(self, x1, y1, x2, y2):
        """
        Calculate the slope of a line from the two given points on the same line
        """
        slope = (y2 - y1) / (x2 - x1)
        return slope

    def calculate_angle_from_slope(self, s1, s2):
        """
        Calculate angle between two lines when the slopes of the two lines are given
        """
        if s1*s2 == -1:
            angle = 90.0
        else:
            angle = math.degrees(math.atan((s1 - s2) / (1 + (s1 * s2))))
        return angle

    def calculate_angle_of_current_crossing(self, start_node, end_node,
                                            separation_lane_gdf,
                                            intersecting_route_seg_geom):
        """
        Calculate the angle between the straight line which starts from the point before first
        intersection and ends after the last intersection with respect to the intersecting
        separation lane segment. Separation lane can contains multiple line segments.
        :param start_node: starting node of the straight line
        :param end_node: ending node of the straight line
        :param separation_lane_gdf: separation lanes
        :param intersecting_route_seg_geom: route segment which is intersecting the separation
        lane
        """

        slope_route = self.calculate_slope(start_node.x, start_node.y,
                                           end_node.x, end_node.y)

        for line in separation_lane_gdf.geom:
            for i in range(len(line.coords) - 1):
                separation_lane_segment = LineString(
                    [line.coords[i], line.coords[i + 1]])
                if separation_lane_segment.intersects(intersecting_route_seg_geom):
                    point_x, point_y = separation_lane_segment.xy
                    slope_separation_lane = self.calculate_slope(point_x[0],
                                                                 point_y[0],
                                                                 point_x[1],
                                                                 point_y[1])
                    angle_current_crossing = self.calculate_angle_from_slope(
                        slope_route,
                        slope_separation_lane)
                    return abs(angle_current_crossing), separation_lane_segment
