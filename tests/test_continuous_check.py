import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from WeatherRoutingTool.constraints.constraints import ContinuousCheck
from WeatherRoutingTool.utils.maps import Map
import tests.basic_test_func as basic_test_func


@pytest.mark.usefixtures("continuous_check_database")
class TestContinuousCheck:

    def test_set_map_bbox(self, continuous_check_database):
        lat1 = 1
        lat2 = 5
        lon1 = 2
        lon2 = 7

        # shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)
        test_bbox = box(2, 1, 7, 5)

        map_bbx = Map(lat1, lon1, lat2, lon2)
        with continuous_check_database.connect() as conn:
            continuouscheck_obj = ContinuousCheck(db_engine=conn.connection)
        continuous_bbox_wkt = continuouscheck_obj.set_map_bbox(map_bbx)

        assert continuous_bbox_wkt == test_bbox.wkt

    def test_query_nodes(self, continuous_check_database):
        with continuous_check_database.connect() as conn:
            seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(
                db_engine=conn.connection)
            gdf = seamark_obj.query_nodes(conn.connection,
                                          "SELECT *,geometry as geom FROM nodes")

        point = {"col1": ["name1", "name2"],
                 "geometry": [Point(1, 2), Point(2, 1)]}
        point_df = gpd.GeoDataFrame(point)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
        assert not gdf.empty, "GeoDataFrame is empty."
        assert isinstance(gdf, type(point_df)), "GeoDataFrame Type Error"
        for geom in point_df["geometry"]:
            assert isinstance(geom, Point), "Point Instantiation Error"
        print("point type checked")

    def test_query_ways(self, continuous_check_database):
        with continuous_check_database.connect() as conn:
            seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(
                db_engine=conn.connection)
            gdf = seamark_obj.query_ways(conn.connection,
                                         "SELECT *, geometry AS geom FROM ways")

        line = {"col1": ["name1", "name2"],
                "geometry": [LineString([(1, 2), (3, 4)]), LineString([(2, 1), (5, 6)])]}
        line_df = gpd.GeoDataFrame(line)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert not gdf.empty, "GeoDataFrame is empty."
        assert isinstance(gdf, type(line_df))
        # assert type(gdf) == type(line_df), "GeoDataFrame Type Error"
        for geom in gdf["geom"]:
            assert isinstance(geom, LineString), "LineString Instantiation Error"
        print("Linestring type checked")

    def test_concat_nodes_ways(self, continuous_check_database):
        """
        Test for checking if table with  ways and nodes includes geometries (Point, LineString)
        """
        with continuous_check_database.connect() as conn:
            seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(
                            db_engine=conn.connection)
            concat_all = seamark_obj.concat_nodes_ways(db_engine=conn.connection,
                                                       query=["SELECT *, geometry as geom FROM nodes",
                                                              "SELECT *, geometry AS geom FROM ways"])

        # Create points and linestrings dummy data

        point1 = {
            "tags": [{"seamark:type": "separation_line"}],
            "geometry": [Point(1, 2)],
        }
        point1_df = gpd.GeoDataFrame(point1)

        line1 = {
            "tags": [{"seamark:type": "separation_line"}, {"seamark:type": "separation_lane"}],
            "geometry": [LineString([(7, 8), (5, 9)]), LineString([(24.6575999, 59.6085725), (24.7026512, 59.5505585)])]
        }
        line1_df = gpd.GeoDataFrame(line1)
        concat_df_all = pd.concat([point1_df, line1_df])

        assert isinstance(concat_all, gpd.GeoDataFrame)
        assert not concat_all.empty, "GeoDataFrame is empty."
        assert isinstance(concat_all, type(concat_df_all))
        type_list = [type(geometry) for geometry in concat_all["geom"]]
        assert set(type_list).intersection([Point, LineString]), "Geometry type error"

    def test_set_STRETree(self, continuous_check_database):
        with continuous_check_database.connect() as conn:
            seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(
                            db_engine=conn.connection)
            test_query = ["SELECT *, geometry as geom FROM nodes",
                          "SELECT *, geometry AS geom FROM ways"]
            concat_tree = seamark_obj.set_STRTree(db_engine=conn.connection, query=test_query)

        # Create points and linestrings dummy data
        point1 = {
            "tags": [{"seamark:type": "separation_line"}],
            "geometry": [Point(1, 2)],
        }
        point1_df = gpd.GeoDataFrame(point1)

        line1 = {
            "tags": [{"seamark:type": "separation_line"},
                     {"seamark:type": "separation_lane"}],
            "geometry": [LineString([(7, 8), (5, 9)]), LineString(
                [(24.6575999, 59.6085725), (24.7026512, 59.5505585)])]
        }
        line1_df = gpd.GeoDataFrame(line1)

        concat_df_all = pd.concat([point1_df, line1_df])
        test_concat_tree = STRtree(concat_df_all["geometry"])

        assert isinstance(concat_tree, STRtree)
        assert isinstance(concat_tree, type(test_concat_tree))
        assert not (concat_tree.geometries.size == 0), "ndarray is empty."

    def test_check_crossing(self, continuous_check_database):
        lat_start = np.array((54.192091, 54.1919199, 54.1905871, 54.189601, 1))
        lon_start = np.array((6.3732417, 6.3593333, 6.3310833, 6.3182992, 1))
        lat_end = np.array((48.92595, 48.02595, 48.12595, 48.22595, 48.42595, 2))
        lon_end = np.array((12.01631, 12.04631, 12.05631, 12.08631, 2))

        test_crossing_list = [True, True, True, True, False]

        with continuous_check_database.connect() as conn:
            seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(db_engine=conn.connection)
            test_query = ["SELECT *, geometry as geom FROM nodes",
                          "SELECT *, geometry AS geom FROM ways"]

            concat_tree = seamark_obj.set_STRTree(db_engine=conn.connection, query=test_query)
        seamark_obj.concat_tree = concat_tree

        # returns a list of tuples(shapelySTRTree, predicate, result_array, bool type)
        check_list = seamark_obj.check_crossing(lat_start=lat_start, lon_start=lon_start,
                                                lat_end=lat_end, lon_end=lon_end)

        for i in range(len(check_list)):
            print(check_list[i])
            assert isinstance(check_list[i], bool)
        assert test_crossing_list == check_list

    def test_set_landpolygon_STRTree(self, continuous_check_database, test_land_polygons_gdf):
        with continuous_check_database.connect() as conn:
            landpolygoncrossing_obj = basic_test_func.create_dummy_landpolygonsCrossing_object(conn.connection)
            test_query = "SELECT *,geometry as geom from land_polygons"
            landpolygon_tree = landpolygoncrossing_obj.set_landpolygon_STRTree(
                db_engine=conn.connection, query=test_query)

        test_land_polygons_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 1, 3, 5)])
        test_concat_tree = STRtree(test_land_polygons_gdf["geometry"])

        assert isinstance(landpolygon_tree, STRtree)
        assert isinstance(landpolygon_tree, type(test_concat_tree))
        assert not (landpolygon_tree.geometries.size == 0), "ndarray is empty."

    def test_check_land_crossing(self, continuous_check_database, test_land_polygons_gdf):
        lat_start = np.array((47, 49.5, 48, 48, 47))
        lon_start = np.array((4.5, 6, 7, 9, 5))
        lat_end = np.array((52, 50.7, 50.5, 50, 50.2))
        lon_end = np.array((4.5, 7.5, 10, 10, 5))
        test_list = [True, True, True, False, True]

        with continuous_check_database.connect() as conn:
            landpolygoncrossing_obj = basic_test_func.create_dummy_landpolygonsCrossing_object(
                                        conn.connection)
        landpolygoncrossing_obj.land_polygon_STRTree = STRtree(test_land_polygons_gdf["geometry"])
        check_list = landpolygoncrossing_obj.check_crossing(
            lat_start=lat_start, lon_start=lon_start, lat_end=lat_end, lon_end=lon_end
        )

        assert test_list == check_list
        for i in range(len(check_list)):
            assert isinstance(check_list[i], bool)
