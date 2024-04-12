import os
from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
import sqlalchemy
from geovectorslib import geod
from shapely.geometry import box, LineString, Point

from WeatherRoutingTool.routeparams import RouteParams


class RoutePostprocessing:
    """
    Currently RoutePostprocessing is focused on Traffic Separation Scheme.
    In the future, it should be integrated into a more general approach.
    """
    route: RouteParams
    lats_per_step: list
    lons_per_step: list
    starttime_per_step: list
    ship_speed: float

    def __init__(self, min_fuel_route=None, boat_speed=None):
        self.route = min_fuel_route
        self.ship_speed = boat_speed
        self.set_data(self.route)

        self.host = os.getenv("WRT_DB_HOST")
        self.database = os.getenv("WRT_DB_DATABASE")
        self.user = os.getenv("WRT_DB_USERNAME")
        self.password = os.getenv("WRT_DB_PASSWORD")
        self.schema = os.getenv("POSTGRES_SCHEMA")
        self.port = os.getenv("WRT_DB_PORT")
        self.engine = self.connect_database()

        self.post_process_route(self.engine)

    def set_data(self, route):
        self.lats_per_step = route.lats_per_step
        self.lons_per_step = route.lons_per_step
        self.starttime_per_step = route.starttime_per_step

    def post_process_route(self, engine):
        route_bbx = self.get_route_bbox()
        route_segments_gdf = self.create_route_segments()
        seamark_gdf = self.retrieve_seamark_data(route_bbx, engine)
        intersecting_route_node_list = self.find_seamark_intersections(route_segments_gdf, seamark_gdf)
        if len(intersecting_route_node_list):
            first_route_seg_gdf = route_segments_gdf[0:intersecting_route_node_list[0]]
            last_route_node_intersecting = intersecting_route_node_list[len(intersecting_route_node_list) - 1]
            first_node_idx_of_last_route_seg = last_route_node_intersecting + 1
            last_route_seg_gdf = route_segments_gdf[first_node_idx_of_last_route_seg:]

            separation_lanes_data_gdf = self.retrieve_seperation_lane_data(route_bbx, engine)
            last_node_of_first_route_seg = self.find_last_node_of_first_route_seg(first_route_seg_gdf)
            separation_lane_gdf = self.find_seperation_lane_to_follow(last_node_of_first_route_seg,
                                                                      separation_lanes_data_gdf)

            final_route = self.connect_route_segments(first_route_seg_gdf, separation_lane_gdf,
                                                      last_route_seg_gdf)
            self.recalculate_starttime_per_node(final_route)

            # return postprocess routes to Maripower

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
                "OR tags -> 'seamark:type'='separation_zone' OR tags -> 'seamark:type'='separation_lane')" \
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
                # ToDo: Check the sensitivity of the buffer
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

    def find_last_node_of_first_route_seg(self, first_route_seg_gdf):
        """
        Find the node of the route segments before the first intersecting point
        """
        last_line_geom = first_route_seg_gdf.tail(1).geometry.get_coordinates()
        last_node_geom = last_line_geom.tail(1)
        last_node = Point(last_node_geom.x, last_node_geom.y)
        return last_node

    def find_first_node_of_last_route_seg(self, last_route_seg_gdf):
        """
        Find the node of the route segments after the last intersecting point
        """
        first_line_geom = last_route_seg_gdf.head(1).geometry.get_coordinates()
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
        (when in parallel and opposite directions) need to be followed
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

    def connect_route_segments(self, first_route_seg_gdf, separation_lane_gdf, last_route_seg_gdf):
        # Create new geometries for new connecting LineStrings before and after separation lanes
        first_connecting_seg_geom = self.create_first_connecting_seg(first_route_seg_gdf, separation_lane_gdf)
        last_connecting_seg_geom = self.create_last_connecting_line(last_route_seg_gdf, separation_lane_gdf)
        separation_lane_geom = separation_lane_gdf.geom

        # Create new GeoDataFrame with the new connecting LineStrings before and after separation lanes
        first_connecting_seg_gdf = gpd.GeoDataFrame(geometry=[first_connecting_seg_geom],
                                                    crs=first_route_seg_gdf.crs)
        seperation_lane_seg_gdf = gpd.GeoDataFrame(geometry=[separation_lane_geom],
                                                   crs=last_route_seg_gdf.crs)
        last_connecting_seg_gdf = gpd.GeoDataFrame(geometry=[last_connecting_seg_geom],
                                                   crs=last_route_seg_gdf.crs)

        # Append the new GeoDataFrame to the existing one
        route_first_connecting_gdf = gpd.GeoDataFrame(
            pd.concat([first_route_seg_gdf, first_connecting_seg_gdf], ignore_index=True),
            crs=first_route_seg_gdf.crs)
        route_separation_lane_gdf = gpd.GeoDataFrame(
            pd.concat([route_first_connecting_gdf, seperation_lane_seg_gdf],
                      ignore_index=True), crs=first_route_seg_gdf.crs)
        route_last_connecting_gdf = gpd.GeoDataFrame(
            pd.concat([route_separation_lane_gdf, last_connecting_seg_gdf], ignore_index=True),
            crs=first_route_seg_gdf.crs)
        final_route = gpd.GeoDataFrame(
            pd.concat([route_last_connecting_gdf, last_route_seg_gdf],
                      ignore_index=True), crs=last_route_seg_gdf.crs)

        return final_route

    def create_first_connecting_seg(self, first_route_seg_gdf, separation_lane_gdf):
        """
        Build the LineString between the route node before the first intersection and
        the first node of the separation lane segment
        """
        x, y = separation_lane_gdf.geom.xy
        first_node_of_separation_lane = Point(x[0], y[0])
        last_node_of_first_route_seg = self.find_last_node_of_first_route_seg(first_route_seg_gdf)
        first_connecting_seg = LineString([last_node_of_first_route_seg, first_node_of_separation_lane])
        return first_connecting_seg

    def create_last_connecting_line(self, last_route_seg_gdf, separation_lane_gdf):
        """
        Build the LineString between the last node of the separtion lane and the first route node
        after the last intersection
        """
        x, y = separation_lane_gdf.geom.xy
        last_index = len(x)-1
        last_node_of_separation_lane = Point(x[last_index], y[last_index])
        first_node_of_last_route_seg = self.find_first_node_of_last_route_seg(last_route_seg_gdf)
        first_connecting_seg = LineString([last_node_of_separation_lane, first_node_of_last_route_seg])
        return first_connecting_seg

    def recalculate_starttime_per_node(self, final_route):
        """
        To recalculate the start time of the new route segments, first a new integer index is set
        to final route segments dataframe. Then, the index of first Not available Timestamp
        value is searched and start calculating the new time taken from that index to rest
        of the dataframe
        """
        self.lats_per_step = []
        self.lons_per_step = []
        self.starttime_per_step = []

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

        self.starttime_per_step.append(start_time)
        array_len = len(self.lons_per_step)

        first_index = 1
        for index in range(first_index, array_len):
            current_timestamp = self.calculate_timsestamp(self.lats_per_step, self.lons_per_step,
                                                          self.starttime_per_step, index, self.ship_speed)
            self.starttime_per_step.append(current_timestamp)

    def calculate_timsestamp(self, lat, lon, starttime, node_index, speed):
        """
        Calculate the time taken using time = distance / ship speed of the previous
        route segment and then added the new time taken into previous timestamp to
        get the new start time
        """
        previous_step_index = node_index - 1
        previous_timestamp = starttime[previous_step_index]
        gcd = geod.inverse([lat[previous_step_index]], [lon[previous_step_index]],
                           [lat[node_index]], [lon[node_index]])
        dist = gcd['s12']
        time_taken_for_current_step = dist[0] / speed
        current_timestamp = previous_timestamp + timedelta(seconds=int(time_taken_for_current_step))

        return current_timestamp
