from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import sqlalchemy as db
from astropy import units as u
from shapely.geometry import box, LineString, Point

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.constraints.route_postprocessing import RoutePostprocessing
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.shipparams import ShipParams

test_seamark_gdf = gpd.GeoDataFrame(
    columns=["tags", "geometry"],
    data=[
        [
            {"seamark:type": "separation_boundary"},
            LineString([(12, 0), (11, 3)]),
        ],
        [
            {"seamark:type": "separation_boundary"},
            LineString([(11, 3), (5, 9)]),
        ],
        [
            {"seamark:type": "separation_boundary"},
            LineString([(5, 9), (0, 6)]),
        ],
        [
            {"seamark:type": "separation_line"},
            LineString([(11, 3), (8, 6), (5, 9)]),
        ],
        [
            {"seamark:type": "separation_line"},
            LineString([(17, 7), (14, 10), (10, 14)]),
        ],
        [
            {"seamark:type": "separation_zone"},
            LineString(
                [
                    (15, 6),
                    (9, 12),
                    (7, 10),
                    (13, 4),
                    (15, 6)
                ]
            ),
        ],
        [
            {"seamark:type": "separation_lane"},
            LineString(
                [
                    (6, 10),
                    (9, 7),
                    (12, 4)

                ]
            ),
        ],
        [
            {"seamark:type": "separation_lane"},
            LineString(
                [
                    (16, 6),
                    (13, 9),
                    (9, 13)
                ]
            ),
        ]
    ],
)
# Create engine using SQLite
engine = db.create_engine("sqlite:///gdfDB.sqlite")


class TestRoutePostprocessing:
    test_seamark_gdf.to_file(f'{"gdfDB.sqlite"}', driver="SQLite", layer="seamark", overwrite=True)

    def generate_test_route_postprocessing_obj(self):
        lats = np.array([40, 50, 60, 70])
        lons = np.array([4, 5, 6, 7])
        dist = np.array([100, 200, 150]) * u.meter
        start_time = np.array([datetime(2022, 12, 19),
                               datetime(2022, 12, 19) + timedelta(hours=1),
                               datetime(2022, 12, 19) + timedelta(hours=2),
                               datetime(2022, 12, 19) + timedelta(hours=3)])
        dummy = np.array([0, 0, 0, 0])

        sp = ShipParams(
            fuel_rate=dummy * u.kg / u.second,
            power=dummy * u.Watt,
            rpm=dummy * u.Hz,
            speed=dummy * u.m / u.s,
            r_calm=dummy * u.N,
            r_wind=dummy * u.N,
            r_waves=dummy * u.N,
            r_shallow=dummy * u.N,
            r_roughness=dummy * u.N,
            wave_height=dummy * u.meter,
            wave_direction=dummy * u.radian,
            wave_period=dummy * u.second,
            u_currents=dummy * u.meter / u.second,
            v_currents=dummy * u.meter / u.second,
            u_wind_speed=dummy * u.meter / u.second,
            v_wind_speed=dummy * u.meter / u.second,
            pressure=dummy * u.kg / u.meter / u.second ** 2,
            air_temperature=dummy * u.deg_C,
            salinity=dummy * u.dimensionless_unscaled,
            water_temperature=dummy * u.deg_C,
            status=3,
            message="",
        )

        rp = RouteParams(
            count=2,
            start=(lons[0], lats[0]),
            finish=(lons[-1], lats[-1]),
            gcr=None,
            route_type='test',
            time=dummy,
            lats_per_step=lats,
            lons_per_step=lons,
            course_per_step=dummy,
            dists_per_step=dist,
            starttime_per_step=start_time,
            ship_params_per_step=sp
        )
        boat = basic_test_func.create_dummy_Direct_Power_Ship("simpleship")
        boat_speed = 6 * u.meter/u.second
        with engine.connect() as conn:
            postprocessed_route = RoutePostprocessing(rp, boat, boat_speed, db_engine=conn.connection)
        return postprocessed_route

    def test_create_route_segments(self):
        test_gdf = gpd.GeoDataFrame(columns=["timestamp", "geometry"],
                                    data=[[datetime(2024, 5, 17, 8), LineString([(1, 2), (2, 4)])],
                                          [datetime(2024, 5, 17, 9), LineString([(2, 4), (3, 6)])],
                                          [datetime(2024, 5, 17, 10), LineString([(3, 6), (4, 8)])]])

        rpp = self.generate_test_route_postprocessing_obj()
        rpp.lats_per_step = [2, 4, 6, 8]
        rpp.lons_per_step = [1, 2, 3, 4]
        rpp.starttime_per_step = [datetime(2024, 5, 17, 8), datetime(2024, 5, 17, 9), datetime(2024, 5, 17, 10)]

        route_gdf = rpp.create_route_segments()
        pd.testing.assert_frame_equal(route_gdf, test_gdf)

    def test_get_route_box(self):
        test_bbox = box((1 - 0.5), (2 - 0.5), (4 + 0.5), (8 + 0.5))
        rpp = self.generate_test_route_postprocessing_obj()
        rpp.lats_per_step = [2, 4, 6, 8]
        rpp.lons_per_step = [1, 2, 3, 4]

        bbox = rpp.get_route_bbox()
        assert isinstance(bbox, type(test_bbox))
        assert bbox == test_bbox

    def test_query_data(self):
        test_query = "SELECT *, geometry as geom From seamark"
        rpp = self.generate_test_route_postprocessing_obj()
        with engine.connect() as conn:
            gdf_seamark = rpp.query_data(test_query, engine=conn.connection)
        assert isinstance(gdf_seamark, type(test_seamark_gdf))
        type_list = [type(geometry) for geometry in gdf_seamark["geom"]]
        assert set(type_list).intersection([LineString]), "Geometry type error"

    def test_find_seamark_intersections(self):
        test_list = [1, 3]
        route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[[datetime(2024, 5, 17, 8), LineString([(16, 1), (13, 2)])],
                  [datetime(2024, 5, 17, 9), LineString([(13, 2), (7, 3)])],
                  [datetime(2024, 5, 17, 10), LineString([(7, 3), (4, 5)])],
                  [datetime(2024, 5, 17, 11), LineString([(4, 5), (2, 11)])]
                  ]
        )
        rpp = self.generate_test_route_postprocessing_obj()
        intersect_indx_list = rpp.find_seamark_intersections(route_gdf, test_seamark_gdf)
        assert intersect_indx_list == test_list

    def test_is_start_or_finish_node_in_separation_zone(self):
        test_val = True
        route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[
                [datetime(2024, 5, 17, 10), LineString([(12, 7), (9, 10)])],
                [datetime(2024, 5, 17, 11), LineString([(9, 10), (5, 14)])]])
        test_seamark_gdf.set_geometry('geometry', inplace=True)
        test_seamark_gdf.rename_geometry('geom', inplace=True)

        rpp = self.generate_test_route_postprocessing_obj()
        is_true = rpp.is_start_or_finish_node_in_separation_zone(route_gdf, test_seamark_gdf)

        assert is_true == test_val

    def test_find_first_node_of_route_seg(self):
        test_fisrt_node = Point(16, 1)
        route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],

            data=[
                [datetime(2024, 5, 17, 9), LineString([(16, 1), (13, 2)])],
                [datetime(2024, 5, 17, 10), LineString([(13, 2), (7, 3)])],
                [datetime(2024, 5, 17, 11), LineString([(7, 3), (4, 5)])],
                [datetime(2024, 5, 17, 12), LineString([(4, 5), (2, 11)])]
            ]
        )
        rpp = self.generate_test_route_postprocessing_obj()
        first_node = rpp.find_first_node_of_route_seg(route_gdf)

        assert first_node == test_fisrt_node

    def test_find_last_node_of_route_seg(self):
        test_last_node = Point(2, 11)
        route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[
                [datetime(2024, 5, 17, 9), LineString([(16, 1), (13, 2)])],
                [datetime(2024, 5, 17, 10), LineString([(13, 2), (7, 3)])],
                [datetime(2024, 5, 17, 11), LineString([(7, 3), (4, 5)])],
                [datetime(2024, 5, 17, 12), LineString([(4, 5), (2, 11)])]
            ]
        )
        rpp = self.generate_test_route_postprocessing_obj()
        last_node = rpp.find_last_node_of_route_seg(route_gdf)

        assert last_node == test_last_node

    def test_find_separation_lane_to_follow(self):
        seperation_lane_dict = {'tags': {"seamark:type": "separation_lane"},
                                'geom': LineString([(16, 6), (13, 9), (9, 13)])}
        test_separation_lane = pd.Series(seperation_lane_dict)

        separation_lanes = gpd.GeoDataFrame(columns=["tags", "geom"],
                                            data=[[{"seamark:type": "separation_lane"},
                                                   LineString([(16, 6), (13, 9), (9, 13)])],
                                                  [{"seamark:type": "separation_lane"},
                                                   LineString([(6, 10), (9, 7), (12, 4)])]])
        test_last_node = Point(16, 1)
        rpp = self.generate_test_route_postprocessing_obj()
        separation_lane = rpp.find_seperation_lane_to_follow(test_last_node,
                                                             separation_lanes)
        pd.testing.assert_series_equal(separation_lane, test_separation_lane, check_names=False)

    def test_connect_route_segments(self):
        test_connected_seg = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[[datetime(2024, 5, 17, 9), LineString([(16, 1), (13, 2)])],
                  [datetime(2024, 5, 17, 10), LineString([(13, 2), (12, 2)])],
                  [datetime(2024, 5, 17, 9), LineString([(12, 2), (16, 6)])],
                  [datetime(2024, 5, 17, 9), LineString([(16, 6), (13, 9), (9, 13)])],
                  [datetime(2024, 5, 17, 9), LineString([(9, 13), (8, 15)])],
                  [datetime(2024, 5, 17, 9), LineString([(8, 15), (7, 17)])],
                  [datetime(2024, 5, 17, 10), LineString([(7, 17), (9, 20)])]])

        first_route_gdf = gpd.GeoDataFrame(columns=["timestamp", "geometry"],
                                           data=[[datetime(2024, 5, 17, 9), LineString([(16, 1), (13, 2)])],
                                                 [datetime(2024, 5, 17, 10), LineString([(13, 2), (12, 2)])]])

        separation_lanes_dict = {'tags': {"seamark:type": "separation_lane"},
                                 'geom': LineString([(16, 6), (13, 9), (9, 13)])}
        separation_lanes_series = pd.Series(separation_lanes_dict)

        last_route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[
                [datetime(2024, 5, 17, 9), LineString([(8, 15), (7, 17)])],
                [datetime(2024, 5, 17, 10), LineString([(7, 17), (9, 20)])]
            ]
        )
        rpp = self.generate_test_route_postprocessing_obj()

        connected_df = rpp.connect_route_segments(first_route_gdf, separation_lanes_series, last_route_gdf)

        assert all(test_connected_seg["geometry"] == connected_df["geometry"])
        assert isinstance(connected_df, gpd.GeoDataFrame)
        assert not connected_df.empty

    def test_create_first_connecting_seg(self):
        test_connected_seg = LineString([(12, 2), (16, 6)])

        first_route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[[datetime(2024, 5, 17, 9), LineString([(16, 1), (13, 2)])],
                  [datetime(2024, 5, 17, 10), LineString([(13, 2), (12, 2)])]])

        separation_lanes_dict = {'tags': {"seamark:type": "separation_lane"},
                                 'geom': LineString(
                                     [(16, 6), (13, 9), (9, 13)])}
        separation_lanes_series = pd.Series(separation_lanes_dict)

        rpp = self.generate_test_route_postprocessing_obj()
        first_connecting_seg = rpp.create_first_connecting_seg(first_route_gdf, separation_lanes_series)

        assert first_connecting_seg == test_connected_seg

    def test_create_first_connecting_seg_from_node(self):
        test_connected_seg = LineString([(15, 5), (16, 6)])
        first_route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[[datetime(2024, 5, 17, 10), LineString([(15, 5), (13, 10)])],
                  [datetime(2024, 5, 17, 9), LineString([(13, 10), (13, 2)])],
                  [datetime(2024, 5, 17, 9), LineString([(13, 2), (11, 13)])], ])

        separation_lanes_dict = {'tags': {"seamark:type": "separation_lane"},
                                 'geom': LineString(
                                     [(16, 6), (13, 9), (9, 13)])}
        separation_lanes_series = pd.Series(separation_lanes_dict)

        rpp = self.generate_test_route_postprocessing_obj()
        seg_from_node = rpp.create_first_connecting_seg_from_node(first_route_gdf, separation_lanes_series)

        assert seg_from_node == test_connected_seg

    def test_create_last_connecting_line(self):
        test_connecting_seg = LineString([(9, 13), (8, 15)])

        separation_lanes_dict = {'tags': {"seamark:type": "separation_lane"},
                                 'geom': LineString([(16, 6), (13, 9), (9, 13)])}
        separation_lanes_series = pd.Series(separation_lanes_dict)
        last_route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[
                [datetime(2024, 5, 17, 9), LineString([(8, 15), (7, 17)])],
                [datetime(2024, 5, 17, 10), LineString([(7, 17), (9, 20)])]
            ]
        )

        rpp = self.generate_test_route_postprocessing_obj()
        last_connecting_seg = rpp.create_last_connecting_line(last_route_gdf, separation_lanes_series)

        assert last_connecting_seg == test_connecting_seg

    def test_create_last_connecting_line_from_node(self):
        test_last_connecting_seg = LineString([(9, 13), (9, 20)])

        last_route_gdf = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[[datetime(2024, 5, 17, 9), LineString([(9, 11), (8, 15)])],
                  [datetime(2024, 5, 17, 9), LineString([(8, 15), (7, 17)])],
                  [datetime(2024, 5, 17, 10), LineString([(7, 17), (9, 20)])]])
        separation_lanes_dict = {'tags': {"seamark:type": "separation_lane"},
                                 'geom': LineString(
                                     [(16, 6), (13, 9), (9, 13)])}
        separation_lanes_series = pd.Series(separation_lanes_dict)

        rpp = self.generate_test_route_postprocessing_obj()
        last_connecting_seg = rpp.create_last_connecting_line_from_node(last_route_gdf, separation_lanes_series)

        assert last_connecting_seg == test_last_connecting_seg

    def test_recalculate_starttime_per_node(self):
        final_route = gpd.GeoDataFrame(
            columns=["timestamp", "geometry"],
            data=[[datetime(2024, 5, 17, 9), LineString([(16, 1), (13, 2)])],
                  [datetime(2024, 5, 17, 10), LineString([(13, 2), (12, 2)])]])

        rpp = self.generate_test_route_postprocessing_obj()
        rpp.ship_speed = 6 * u.meter / u.second
        time_list = rpp.recalculate_starttime_per_node(final_route)
        # test_list = [(datetime(2024, 5, 17, 9), datetime(2024, 5, 18, 1, 16, 57), datetime(2024, 5, 18, 6, 26,  7))]
        # delta time 16:16:57, 05:09:10
        # [datetime.datetime(2024, 5, 17, 9, 0), datetime.datetime(2024, 5, 18, 1, 16, 52),
        # datetime.datetime(2024, 5, 18, 6, 25, 54)]

        for timestamp in time_list:
            assert isinstance(timestamp, datetime)

    def test_find_point_from_perpendicular_angle(self):
        test_point = Point(1, 1)

        start_node = Point(0, 2)
        segment = LineString([(0, 0), (2, 2)])

        rpp = self.generate_test_route_postprocessing_obj()
        x, y = rpp.find_point_from_perpendicular_angle(start_node, segment)
        rpp_point = Point(x, y)

        assert test_point == rpp_point

    def test_check_valid_crossing(self):
        test_valid_crossing = True

        separation_line_1 = LineString([(0, 2), (2, 0)])
        separation_line_2 = LineString([(0, 3), (3, 0)])

        separation_lane_gdf = gpd.GeoDataFrame({'geom': [separation_line_1, separation_line_2]},
                                               geometry='geom')
        separation_lane_gdf['id'] = [0, 1]
        last_node_of_fisrt_routing_seg = Point(0, 0)
        fisrt_node_of_last_routing_seg = Point(4, 4)

        rpp = self.generate_test_route_postprocessing_obj()
        valid_crossing, segment = rpp.check_valid_crossing(separation_lane_gdf, last_node_of_fisrt_routing_seg,
                                                           fisrt_node_of_last_routing_seg)
        assert test_valid_crossing == valid_crossing

    def test_calculate_slope(self):
        test_slope = 1

        x1, y1, x2, y2 = 1, 1, 2, 2
        rpp = self.generate_test_route_postprocessing_obj()
        slope = rpp.calculate_slope(x1, y1, x2, y2)

        assert test_slope == slope

    def test_calculate_angle_from_slope(self):
        test_angle = 90.0
        s1 = 2
        s2 = (-1) / 2
        rpp = self.generate_test_route_postprocessing_obj()
        angle1 = rpp.calculate_angle_from_slope(s1, s2)

        assert test_angle == angle1

        test_angle = 45.0
        s1 = 1
        s2 = 0
        angle2 = rpp.calculate_angle_from_slope(s1, s2)

        assert test_angle == angle2

    def test_calculate_angle_of_current_crossing(self):
        test_angle = 90.0
        start_node = Point(0, 0)
        end_node = Point(4, 4)

        separation_line_1 = LineString([(0, 2), (2, 0)])
        separation_line_2 = LineString([(0, 3), (3, 0)])

        separation_lane_gdf = gpd.GeoDataFrame(
            {'geom': [separation_line_1, separation_line_2]},
            geometry='geom')
        separation_lane_gdf['id'] = [0, 1]

        intersecting_route_seg_geom = LineString([(0, 2), (2, 0)])
        rpp = self.generate_test_route_postprocessing_obj()
        angle, segment = rpp.calculate_angle_of_current_crossing(start_node, end_node,
                                                                 separation_lane_gdf, intersecting_route_seg_geom)
        assert test_angle == angle


engine.dispose
