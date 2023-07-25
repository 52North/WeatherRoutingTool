import numpy
import sqlalchemy as db
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPolygon, box, Polygon
from WeatherRoutingTool.constraints.constraints import ContinuousCheck, SeamarkCrossing, LandPolygonsCrossing

# Create engine using SQLite
engine = db.create_engine("sqlite:///gdfDB.sqlite")

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


class TestContinuousCheck:
    """
    Contains various functions to test data connection,
    gathering and use for obtaining spatial relations
    for the continuous check in the negative constraints
    """

    # write geodataframe into spatialite database (nodes)
    test_nodes_gdf.to_file(
        f'{"gdfDB.sqlite"}', driver="SQLite", layer="nodes", overwrite=True
    )

    # write geodataframe into spatialite database (ways)
    test_ways_gdf.to_file(
        f'{"gdfDB.sqlite"}', driver="SQLite", layer="ways", overwrite=True
    )

    test_land_polygons_gdf.to_file(
        f'{"gdfDB.sqlite"}', driver="SQLite", layer="land_polygons", overwrite=True
    )

    def test_connect_database(self):
        """
        Test for checking the engine object and if the connection with the db is estabilished
        """

        # parameters might change when creating the engine
        self.connection = engine.connect()

        assert isinstance(
            ContinuousCheck().connect_database(), type(engine)
        ), "Engine Instantiation Error"

    def test_query_nodes(self):
        check = SeamarkCrossing()
        gdf = check.query_nodes(engine=engine, query="SELECT *,geometry as geom FROM nodes")

        # gdf = gpd.read_postgis(sql="SELECT * FROM nodes",con=engine,geom_col="geometry")

        point = {"col1": ["name1", "name2"], "geometry": [Point(1, 2), Point(2, 1)]}
        point_df = gpd.GeoDataFrame(point)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "SELECT * FROM nodes" == check.query[0]
        # assert type(check.con) == type(check.connect_database())
        assert len(gdf) > 0
        assert not gdf.empty, "GeoDataFrame is empty."
        assert type(gdf) == type(point_df), "GeoDataFrame Type Error"
        for geom in point_df["geometry"]:
            assert isinstance(geom, Point), "Point Instantiation Error"
        print("point type checked")

    def test_query_ways(self):
        check = SeamarkCrossing()
        gdf = check.query_ways(
            engine=engine, query="SELECT *, geometry AS geom FROM ways"
        )

        line = {
            "col1": ["name1", "name2"],
            "geometry": [LineString([(1, 2), (3, 4)]), LineString([(2, 1), (5, 6)])],
        }
        line_df = gpd.GeoDataFrame(line)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert not gdf.empty, "GeoDataFrame is empty."
        assert type(gdf) == type(line_df), "GeoDataFrame Type Error"
        for geom in gdf["geom"]:
            assert isinstance(geom, LineString), "LineString Instantiation Error"
        print("Linestring type checked")

    def test_gdf_seamark_combined_nodes(self):
        """
        Test for checking if gdf nodes is gdf and geometry is Point type
        """
        nodes_concat = SeamarkCrossing().gdf_seamark_combined_nodes(
            engine,
            query="SELECT *, geometry as geom FROM nodes",
            seamark_object=["nodes"],
            seamark_list=["separation_zone", "separation_line"],
        )

        # Create points dummy data
        point1 = {
            "tags": [{"seamark:type": "separation_line"}],
            "geometry": [Point(1, 2)],
        }

        point1_df = gpd.GeoDataFrame(point1)
        point2 = {
            "tags": [{"seamark:type": "separation_zone"}],
            "geometry": [Point(6, 2)],
        }
        point2_df = gpd.GeoDataFrame(point2)

        concat_df = pd.concat([point1_df, point2_df])

        assert isinstance(nodes_concat, gpd.GeoDataFrame)
        assert not nodes_concat.empty, "GeoDataFrame is empty."
        assert type(concat_df) == type(nodes_concat)

        nodes_tags = [item["seamark:type"] for item in list(nodes_concat["tags"])]
        test_tags = [item["seamark:type"] for item in list(concat_df["tags"])]

        assert len(set(nodes_tags).intersection(set(test_tags))) > 0, "no intersection"

        for geom in nodes_concat["geom"]:
            assert isinstance(geom, Point), "Point Instantiation Error"

    def test_gdf_seamark_combined_ways(self):
        """
        Test for checking if gdf ways is gdf and geometry is LineString type
        """
        lines_concat = SeamarkCrossing().gdf_seamark_combined_ways(
            engine=engine,
            query="SELECT *, geometry AS geom FROM ways",
            seamark_object=["ways"],
            seamark_list=["separation_zone", "separation_line"],
        )

        # Create linestrings dummy data
        line1 = {
            "tags": [{"seamark:type": "separation_line"}],
            "geometry": [LineString([(7, 8), (5, 9)])],
        }
        line1_df = gpd.GeoDataFrame(line1)
        line2 = {
            "tags": [{"seamark:type": "separation_zone"}],
            "geometry": [LineString([(1, 2), (3, 4)])],
        }
        line2_df = gpd.GeoDataFrame(line2)
        concat_df = pd.concat([line1_df, line2_df])

        assert isinstance(lines_concat, gpd.GeoDataFrame)
        assert not lines_concat.empty, "GeoDataFrame is empty."
        assert type(concat_df) == type(lines_concat)
        lines_tags = [item["seamark:type"] for item in list(lines_concat["tags"])]
        test_tags = [item["seamark:type"] for item in list(concat_df["tags"])]
        assert len(set(lines_tags).intersection(set(test_tags))) > 0, "no intersection"
        for geom in lines_concat["geom"]:
            assert isinstance(geom, LineString), "Linestring Instantiation Error"

    def test_concat_nodes_ways(self):
        """
        Test for checking if table with  ways and nodes includes geometries (Point, LineString)
        """
        concat_all = SeamarkCrossing().concat_nodes_ways(
            query=[
                "SELECT *, geometry as geom FROM nodes",
                "SELECT *, geometry AS geom FROM ways",
            ],
            engine=engine,
        )

        # Create points and linestrings dummy data
        point1 = {
            "tags": [{"seamark:type": "separation_line"}],
            "geometry": [Point(1, 2)],
        }
        point1_df = gpd.GeoDataFrame(point1)

        line1 = {
            "tags": [{"seamark:type": "separation_line"}],
            "geometry": [LineString([(7, 8), (5, 9)])],
        }
        line1_df = gpd.GeoDataFrame(line1)

        concat_df_all = pd.concat([point1_df, line1_df])

        assert isinstance(concat_all, gpd.GeoDataFrame)
        assert not concat_all.empty, "GeoDataFrame is empty."

        assert type(concat_all) == type(concat_df_all)

        type_list = [type(geometry) for geometry in concat_all["geom"]]
        assert set(type_list).intersection([Point, LineString]), "Geometry type error"

    def test_gdf_seamark_combined_nodes_ways(self):
        """
        Test for checking if combined gdf (nodes and ways) includes geometries (Point, LineString) and is a gdf
        """

        # gdf with nodes and linestrings
        nodes_lines_concat = SeamarkCrossing().gdf_seamark_combined_nodes_ways(
            engine=engine,
            query=[
                "SELECT *,geometry AS geom FROM nodes",
                "SELECT *, geometry AS geom FROM ways",
            ],
            seamark_object=["nodes", "ways"],
            seamark_list=["separation_zone", "separation_line"],
        )

        # creating dummy data point
        nodes1 = {
            "tags": [
                {"seamark:type": "separation_line"},
                {"seamark:type": "separation_zone"},
                {""},
            ],
            "geometry": [Point(1, 2), Point(4, 7), None],
        }
        nodes_df = gpd.GeoDataFrame(nodes1)

        ways1 = {
            "tags": [
                {"seamark:type": "separation_line"},
                {"seamark:type": "separation_zone"},
                {"seamark:type": "restricted_area"},
                {""},
            ],
            "geometry": [
                LineString([(7, 8), (5, 9)]),
                LineString([(16, 14), (8, 13)]),
                LineString([(2, 11), (7, 15)]),
                None,
            ],
        }

        ways_df = gpd.GeoDataFrame(ways1)

        concat_df = pd.concat([nodes_df, ways_df])

        assert isinstance(nodes_lines_concat, gpd.GeoDataFrame)
        assert not nodes_lines_concat.empty, "GeoDataFrame is empty."
        assert type(concat_df) == type(nodes_lines_concat), "DataFrame type error"

        nodes_lines_tags = [
            item["seamark:type"] for item in list(nodes_lines_concat["tags"])
        ]
        test_tags = [
            item["seamark:type"]
            for item in concat_df["tags"]
            if item is not None and "seamark:type" in item
        ]
        assert (
                len(set(nodes_lines_tags).intersection(set(test_tags))) > 0
        ), "no intersection"

        type_list = [type(geometry) for geometry in nodes_lines_concat["geom"]]
        assert set(type_list).intersection([Point, LineString]), "Geometry type error"

    def test_check_crossing(self):
        """
        Test for checking if different spatial relations (intersection, contain, touches ...) are being returned
        """

        # route = {
        #     "tags": [{"seamark:type": "separation_line"}],
        #     "geom": [LineString([(7, 8), (5, 9)])],
        # }
        # route_df = gpd.GeoDataFrame(route)

        # 9.91950, 57.06081)],
        # [{"seamark:type": "buoy_cardinal"}, Point(12.01631, 48.92595)],
        # [{"seamark:type": "separation_boundary"}, Point(12.01631, 48.92595)],
        # [{"seamark:type": "separation_crossing"}, Point(12.01631, 48.92595

        lat_start = numpy.array((54.192091, 54.1919199, 54.1905871, 54.189601))
        lon_start = numpy.array((6.3732417, 6.3593333, 6.3310833, 6.3182992))
        lat_end = numpy.array((48.92595, 48.02595, 48.12595, 48.22595, 48.42595))
        lon_end = numpy.array((12.01631, 12.04631, 12.05631, 12.08631))
        # returns a list of tuples(shapelySTRTree, predicate, result_array, bool type)
        check_list = SeamarkCrossing().check_crossing(
            lat_start=lat_start, lon_start=lon_start, lat_end=lat_end, lon_end=lon_end,
            engine=engine,
            query=[
                "SELECT * , geometry as geom FROM nodes",
                "SELECT *, geometry AS geom FROM ways",
            ],
            seamark_object=["nodes", "ways"],
            seamark_list=["separation_zone", "separation_line"],
        )

        for i in range(len(check_list)):
            assert (type(check_list[i]) == bool)

    def test_query_land_polygons(self):
        gdf = LandPolygonsCrossing().query_land_polygons(engine=engine,
                                                         query="SELECT *,geometry as geom from land_polygons")
        print(gdf)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert not gdf.empty, 'empty geodataframe'

        for geom in gdf["geom"].to_list():
            assert type(geom) == Polygon or type(geom) == LineString or type(
                geom) == MultiPolygon, "Geometry Instantiation Error"

    def test_check_land_crossing(self):
        lat_start = numpy.array((54.192091, 54.1919199, 54.1905871, 54.189601))
        lon_start = numpy.array((6.3732417, 6.3593333, 6.3310833, 6.3182992))
        lat_end = numpy.array((48.92595, 48.02595, 48.12595, 48.22595, 48.42595))
        lon_end = numpy.array((12.01631, 12.04631, 12.05631, 12.08631))
        # returns a list of tuples(shapelySTRTree, predicate, result_array, bool type)
        check_list = LandPolygonsCrossing().check_crossing(
            lat_start=lat_start, lon_start=lon_start, lat_end=lat_end, lon_end=lon_end,
            engine=engine,
            query="SELECT *,geometry as geom from land_polygons"
        )

        for i in range(len(check_list)):
            assert (type(check_list[i]) == bool)


# Closing engine
engine.dispose()
