import geopandas as gpd
import numpy as np
import pandas as pd
import sqlalchemy as db
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from WeatherRoutingTool.constraints.constraints import ContinuousCheck
from WeatherRoutingTool.utils.maps import Map
import tests.basic_test_func as basic_test_func


# Create dummy GeoDataFrames
test_nodes_gdf = gpd.GeoDataFrame(
    columns=["tags", "geometry"],
    data=[
        [{"waterway": "lock_gate", "seamark:type": "gate"}, Point(5, 15)],
        [{"seamark:type": "harbour"}, Point(9.91950, 57.06081)],
        [{"seamark:type": "buoy_cardinal"}, Point(12.01631, 48.92595)],
        [{"seamark:type": "separation_boundary"}, Point(12.01631, 48.92595)],
        [{"seamark:type": "separation_crossing"}, Point(12.01631, 48.92595)],
        [{"seamark:type": "separation_lane"}, Point(12.01631, 48.92595)],
        [{"seamark:type": "separation_roundabout"}, Point(12.01631, 48.92595)],
        [{"seamark:type": "separation_zone"}, Point(12.01631, 48.92595)],
        [{"seamark:type": "restricted_area"}, Point(12.01631, 48.92595)],
    ],
)

test_ways_gdf = gpd.GeoDataFrame(
    columns=["tags", "geometry"],
    data=[
        [
            {"seamark:type": "separation_boundary"},
            LineString([(3.2738333, 51.8765), (3.154833, 51.853667)]),
        ],
        [
            {"seamark:type": "separation_crossing"},
            LineString(
                [
                    (6.3732417, 54.192091),
                    (6.3593333, 54.1919199),
                    (6.3310833, 54.1905871),
                    (6.3182992, 54.189601),
                ]
            ),
        ],
        [
            {"seamark:type": "separation_lane"},
            LineString([(24.6575999, 59.6085725), (24.7026512, 59.5505585)]),
        ],
        [
            {"seamark:type": "separation_roundabout"},
            LineString(
                [
                    (27.9974563, 43.186327),
                    (27.998524, 43.1864565),
                    (27.9995173, 43.186412),
                    (28.0012373, 43.1859232),
                    (28.0020059, 43.1854689),
                    (28.0025203, 43.1850186),
                    (28.0029253, 43.1845006),
                    (28.0032216, 43.1838693),
                    (27.9947856, 43.1813859),
                    (27.9944414, 43.1819034),
                    (27.9941705, 43.1826993),
                    (27.9941723, 43.1835194),
                    (27.9944142, 43.1842511),
                    (27.9947709, 43.1848037),
                    (27.9953623, 43.1853841),
                    (27.9961109, 43.1858589),
                    (27.9974563, 43.186327),
                ]
            ),
        ],
        [
            {"seamark:type": "separation_zone"},
            LineString(
                [
                    (-1.9830398, 49.2927514),
                    (-1.9830233, 49.2925889),
                    (-1.9828257, 49.2924815),
                    (-1.9827145, 49.2925089),
                    (-1.9828543, 49.2927771),
                    (-1.9830398, 49.2927514),
                ]
            ),
        ],
        [
            {"seamark:type": "restricted_area"},
            LineString(
                [
                    (12.3569916, 47.9186626),
                    (12.3567217, 47.9188108),
                    (12.3564934, 47.9189565),
                    (12.3564734, 47.9191199),
                    (12.3565413, 47.9192803),
                    (12.3568636, 47.919524),
                    (12.3571719, 47.9196858),
                    (12.3575482, 47.9197593),
                    (12.3579399, 47.9198024),
                    (12.3587152, 47.9200541),
                    (12.3594448, 47.9203064),
                    (12.3596907, 47.9203917),
                    (12.3599993, 47.9204654),
                    (12.3604107, 47.9205391),
                    (12.3608174, 47.9205554),
                    (12.3610279, 47.9205224),
                    (12.3612053, 47.9204511),
                    (12.3614394, 47.9201326),
                    (12.3616484, 47.9198195),
                    (12.3616249, 47.9196335),
                    (12.361631, 47.9194503),
                    (12.3616174, 47.9193071),
                    (12.36156, 47.9192435),
                    (12.3614394, 47.9191936),
                    (12.3611173, 47.9191633),
                    (12.3609535, 47.9190676),
                    (12.3607335, 47.9189749),
                    (12.3604259, 47.918891),
                    (12.3595763, 47.9187613),
                    (12.3587674, 47.9185358),
                    (12.3584371, 47.9183784),
                    (12.3582044, 47.9182997),
                    (12.3579056, 47.918306),
                    (12.3576587, 47.9183381),
                    (12.3573105, 47.9184692),
                    (12.3569916, 47.9186626),
                ]
            ),
        ],
    ],
)

test_land_polygons_gdf = gpd.GeoDataFrame(geometry=[box(4.056342603541809, 49.06378892560051,
                                                        8.748316591073674, 51.19862259935186)])

# Create engine using SQLite
engine = db.create_engine("sqlite:///gdfDB.sqlite")


class TestContinuousCheck:
    # write geodataframe into spatialite database
    test_nodes_gdf.to_file(
        f'{"gdfDB.sqlite"}', driver="SQLite", layer="nodes", overwrite=True
    )

    test_ways_gdf.to_file(
        f'{"gdfDB.sqlite"}', driver="SQLite", layer="ways", overwrite=True
    )

    test_land_polygons_gdf.to_file(
        f'{"gdfDB.sqlite"}', driver="SQLite", layer="land_polygons",
        overwrite=True
    )

    def test_set_map_bbox(self):
        lat1 = 1
        lat2 = 5
        lon1 = 2
        lon2 = 7

        # shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)
        test_bbox = box(2, 1, 7, 5)

        map_bbx = Map(lat1, lon1, lat2, lon2)
        continuouscheck_obj = ContinuousCheck(db_engine=engine)
        continuous_bbox_wkt = continuouscheck_obj.set_map_bbox(map_bbx)

        assert continuous_bbox_wkt == test_bbox.wkt

    def test_query_nodes(self):
        seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(db_engine=engine)
        gdf = seamark_obj.query_nodes(engine,
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

    def test_query_ways(self):
        seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(db_engine=engine)
        gdf = seamark_obj.query_ways(engine,
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

    def test_concat_nodes_ways(self):
        """
        Test for checking if table with  ways and nodes includes geometries (Point, LineString)
        """
        seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(
            db_engine=engine)
        concat_all = seamark_obj.concat_nodes_ways(
            db_engine=engine,
            query=[
                "SELECT *, geometry as geom FROM nodes",
                "SELECT *, geometry AS geom FROM ways"
            ]
        )

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

    def test_set_STRETree(self):
        seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(
            db_engine=engine)
        test_query = ["SELECT *, geometry as geom FROM nodes",
                      "SELECT *, geometry AS geom FROM ways"]
        concat_tree = seamark_obj.set_STRTree(db_engine=engine, query=test_query)

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

    def test_check_crossing(self):
        lat_start = np.array((54.192091, 54.1919199, 54.1905871, 54.189601, 1))
        lon_start = np.array((6.3732417, 6.3593333, 6.3310833, 6.3182992, 1))
        lat_end = np.array((48.92595, 48.02595, 48.12595, 48.22595, 48.42595, 2))
        lon_end = np.array((12.01631, 12.04631, 12.05631, 12.08631, 2))

        test_crossing_list = [True, True, True, True, False]

        seamark_obj = basic_test_func.create_dummy_SeamarkCrossing_object(db_engine=engine)
        test_query = ["SELECT *, geometry as geom FROM nodes",
                      "SELECT *, geometry AS geom FROM ways"
                      ]
        concat_tree = seamark_obj.set_STRTree(db_engine=engine, query=test_query)
        seamark_obj.concat_tree = concat_tree

        # returns a list of tuples(shapelySTRTree, predicate, result_array, bool type)
        check_list = seamark_obj.check_crossing(lat_start=lat_start, lon_start=lon_start,
                                                lat_end=lat_end, lon_end=lon_end)

        for i in range(len(check_list)):
            print(check_list[i])
            assert isinstance(check_list[i], bool)
        assert test_crossing_list == check_list

    def test_set_landpolygon_STRTree(self):
        landpolygoncrossing_obj = basic_test_func.create_dummy_landpolygonsCrossing_object(engine)
        test_query = "SELECT *,geometry as geom from land_polygons"
        landpolygon_tree = landpolygoncrossing_obj.set_landpolygon_STRTree(db_engine=engine, query=test_query)

        test_land_polygons_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 1, 3, 5)])
        test_concat_tree = STRtree(test_land_polygons_gdf["geometry"])

        assert isinstance(landpolygon_tree, STRtree)
        assert isinstance(landpolygon_tree, type(test_concat_tree))
        assert not (landpolygon_tree.geometries.size == 0), "ndarray is empty."

    def test_check_land_crossing(self):
        lat_start = np.array((47, 49.5, 48, 48, 47))
        lon_start = np.array((4.5, 6, 7, 9, 5))
        lat_end = np.array((52, 50.7, 50.5, 50, 50.2))
        lon_end = np.array((4.5, 7.5, 10, 10, 5))
        test_list = [True, True, True, False, True]

        landpolygoncrossing_obj = basic_test_func.create_dummy_landpolygonsCrossing_object(
            engine)
        landpolygoncrossing_obj.land_polygon_STRTree = STRtree(test_land_polygons_gdf["geometry"])
        check_list = landpolygoncrossing_obj.check_crossing(
            lat_start=lat_start, lon_start=lon_start, lat_end=lat_end, lon_end=lon_end
        )

        assert test_list == check_list
        for i in range(len(check_list)):
            assert isinstance(check_list[i], bool)


# Closing engine
engine.dispose()
