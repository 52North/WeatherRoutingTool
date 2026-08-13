"""
Pytest configuration and shared fixtures for WeatherRoutingTool tests.

This module provides reusable fixtures for test setup and teardown,
replacing ad-hoc helper patterns with proper pytest fixtures.
"""

import pytest
import geopandas as gpd
import sqlalchemy as db
from shapely.geometry import LineString, Point, box


@pytest.fixture(scope="class")
def test_nodes_gdf():
    """
    Fixture providing test nodes GeoDataFrame for seamark/constraint tests.

    :return: GeoDataFrame with test node data
    :rtype: gpd.GeoDataFrame
    """
    return gpd.GeoDataFrame(
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


@pytest.fixture(scope="class")
def test_ways_gdf():
    """
    Fixture providing test ways GeoDataFrame for seamark/constraint tests.

    :return: GeoDataFrame with test ways data
    :rtype: gpd.GeoDataFrame
    """
    return gpd.GeoDataFrame(
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


@pytest.fixture(scope="class")
def test_land_polygons_gdf():
    """
    Fixture providing test land polygons GeoDataFrame for constraint tests.

    :return: GeoDataFrame with test land polygon data
    :rtype: gpd.GeoDataFrame
    """
    return gpd.GeoDataFrame(
        geometry=[box(4.056342603541809, 49.06378892560051, 8.748316591073674, 51.19862259935186)]
    )


@pytest.fixture(scope="class")
def continuous_check_database(tmp_path_factory, test_nodes_gdf, test_ways_gdf, test_land_polygons_gdf):
    """
    Fixture providing a temporary SQLite database for ContinuousCheck tests.

    Creates database in temporary directory with nodes, ways, and land_polygons layers.
    Database is automatically cleaned up after test class completes.

    :param tmp_path_factory: pytest fixture for creating temporary directories
    :param test_nodes_gdf: GeoDataFrame with test nodes
    :type test_nodes_gdf: gpd.GeoDataFrame
    :param test_ways_gdf: GeoDataFrame with test ways
    :type test_ways_gdf: gpd.GeoDataFrame
    :param test_land_polygons_gdf: GeoDataFrame with test land polygons
    :type test_land_polygons_gdf: gpd.GeoDataFrame
    :yields: Database engine connected to temporary database
    :rtype: sqlalchemy.engine.Engine
    """
    # Create temporary directory for this test class
    tmp_dir = tmp_path_factory.mktemp("continuous_check_db")
    db_path = tmp_dir / "gdfDB.sqlite"

    # Create database engine
    engine = db.create_engine(f"sqlite:///{db_path}")

    # Write GeoDataFrames to database
    test_nodes_gdf.to_file(str(db_path), driver="SQLite", layer="nodes", overwrite=True)
    test_ways_gdf.to_file(str(db_path), driver="SQLite", layer="ways", overwrite=True)
    test_land_polygons_gdf.to_file(str(db_path), driver="SQLite", layer="land_polygons", overwrite=True)

    yield engine

    # Cleanup (engine disposal)
    engine.dispose()
    # tmp_path_factory automatically cleans up the temporary directory


@pytest.fixture(scope="class")
def test_seamark_gdf():
    """
    Fixture providing test seamark GeoDataFrame for route postprocessing tests.

    :return: GeoDataFrame with test seamark data
    :rtype: gpd.GeoDataFrame
    """
    return gpd.GeoDataFrame(
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
                LineString([(15, 6), (9, 12), (7, 10), (13, 4), (15, 6)]),
            ],
            [
                {"seamark:type": "separation_lane"},
                LineString([(6, 10), (9, 7), (12, 4)]),
            ],
            [
                {"seamark:type": "separation_lane"},
                LineString([(16, 6), (13, 9), (9, 13)]),
            ],
        ],
    )


@pytest.fixture(scope="class")
def route_postprocessing_database(tmp_path_factory, test_seamark_gdf):
    """
    Fixture providing a temporary SQLite database for RoutePostprocessing tests.

    Creates database in temporary directory with seamark layer.
    Database is automatically cleaned up after test class completes.

    :param tmp_path_factory: pytest fixture for creating temporary directories
    :param test_seamark_gdf: GeoDataFrame with test seamark data
    :type test_seamark_gdf: gpd.GeoDataFrame
    :yields: Database engine connected to temporary database
    :rtype: sqlalchemy.engine.Engine
    """
    # Create temporary directory for this test class
    tmp_dir = tmp_path_factory.mktemp("route_postprocessing_db")
    db_path = tmp_dir / "gdfDB.sqlite"

    # Create database engine
    engine = db.create_engine(f"sqlite:///{db_path}")

    # Write GeoDataFrame to database
    test_seamark_gdf.to_file(str(db_path), driver="SQLite", layer="seamark", overwrite=True)

    yield engine

    # Cleanup (engine disposal)
    engine.dispose()
    # tmp_path_factory automatically cleans up the temporary directory
