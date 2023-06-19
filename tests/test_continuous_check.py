import os
import sqlalchemy as db
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import sys
from shapely.geometry import LineString, Point
import pytest
from shapely.geometry import Point
from shapely import STRtree
from pathlib import Path

# Load the environment variables from the .env file
load_dotenv()

os.chdir(
    Path(__file__).resolve().parent.parent
)  # os.path.dirname(__file__))

# current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(os.getcwd(), ""))


from constraints.constraints import *

engine = db.create_engine(
    "postgresql://{user}:{pw}@{host}/{db}".format(
        user="myuser", pw="mypassword", host="172.30.0.3", db="mydatabase", port="5434"
    )
)


class TestContinuousCheck:
    """
    include test methods for the continuous check in the negative constraints
    """

    def test_connect_database(self):
        """
        test for checking the engine object and if the connection with the db is estabilished
        """

        # parameters might change when creating the engine
        self.connection = engine.connect()

        # returns 1 if the connection is estabilished and 0 otherwise
        status = self.connection.connection.connection.status

        assert isinstance(
            ContinuousCheck().connect_database(),
            type(db.create_engine("postgresql://myuser:***@172.25.0.3/mydatabase")),
        ), "Engine Instantiation Error"
        assert status == 1, "Connection not estalished"

    def test_query_nodes(self):
        """
        test for checking if table nodes is gdf and geometry is Point type
        """
        gdf = ContinuousCheck().query_nodes(
            engine, query="SELECT * FROM nodes LIMIT 100"
        )
        # gdf = gpd.read_postgis('select * from nodes', engine)
        # gdf['tstamp'] = gdf['tstamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        gdf = gdf[gdf["geom"] != None]

        point = {"col1": ["name1", "name2"], "geometry": [Point(1, 2), Point(2, 1)]}
        point_df = gpd.GeoDataFrame(point)
        assert type(gdf) == type(point_df), "GeoDataFrame Type Error"
        for geom in point_df["geometry"]:
            assert isinstance(geom, Point), "Point Instantiation Error"
        print("point type checked")

    def test_query_ways(self):
        """
        test for checking if table nodes is gdf and geometry is Linestring type
        """

        # # Use geopandas to read the SQL query into a dataframe from postgis
        # gdf = gpd.read_postgis("SELECT *, linestring AS geom FROM ways", engine)
        # # read timestamp type data as string
        # gdf['tstamp']=gdf['tstamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        gdf = ContinuousCheck().query_ways(
            engine, query="SELECT *, linestring AS geom FROM ways LIMIT 100"
        )
        gdf = gdf[gdf["geom"] != None]

        line = {
            "col1": ["name1", "name2"],
            "geometry": [LineString([(1, 2), (3, 4)]), LineString([(2, 1), (5, 6)])],
        }
        line_df = gpd.GeoDataFrame(line)
        assert type(gdf) == type(line_df), "GeoDataFrame Type Error"
        for geom in gdf["geom"]:
            assert isinstance(geom, LineString), "LineString Instantiation Error"
        print("Linestring type checked")
        
        

    def test_gdf_seamark_combined_nodes(self):
        """
        test for checking if table nodes is gdf and geometry is Linestring type
        """
        nodes_concat = ContinuousCheck().gdf_seamark_combined_nodes(
            engine,
            query="SELECT * FROM nodes",
            seamark_object=["nodes"],
            seamark_list=["separation_zone", "separation_line"],
        )
        
        point1 = {"tags": [{'seamark:type':'separation_line'}], "geometry": [Point(1, 2)]}
        
        point1_df = gpd.GeoDataFrame(point1)
        point2 = {"tags": [{'seamark:type':'separation_zone'}], "geometry": [Point(6, 2)]}
        point2_df = gpd.GeoDataFrame(point2)
        
        concat_df= pd.concat([point1_df, point2_df])
        
        assert isinstance(nodes_concat,gpd.GeoDataFrame)
        assert type(concat_df) == type(nodes_concat) 
        
        nodes_tags= [item['seamark:type'] for item in list(nodes_concat['tags'])]
        test_tags= [item['seamark:type'] for item in list(concat_df['tags'])]
        
        assert len(set(nodes_tags).intersection(set(test_tags))) > 0 ,'no intersection'
        
        #assert concat_df['tags'].values in nodes_concat['tags'].values
        
        for geom in nodes_concat["geom"]:
            assert isinstance(geom, Point), "Point Instantiation Error"


    def test_gdf_seamark_combined_ways(self):
        """
        test for checking if table nodes is gdf and geometry is Linestring type
        """
        lines_concat = ContinuousCheck().gdf_seamark_combined_ways(
            engine,
            query="SELECT *, linestring AS geom FROM ways",
            seamark_object=["ways"],
            seamark_list=["separation_zone", "separation_line"],
        )
        line1 = {"tags": [{'seamark:type': 'separation_line'}], "geometry": [LineString([(7, 8), (5, 9)])]}
        line1_df = gpd.GeoDataFrame(line1)
        line2 = {"tags": [{'seamark:type': 'separation_zone'}], "geometry": [LineString([(1, 2), (3, 4)])]}
        line2_df = gpd.GeoDataFrame(line2)
        concat_df = pd.concat([line1_df, line2_df])

        assert isinstance(lines_concat, gpd.GeoDataFrame)
        assert type(concat_df) == type(lines_concat)
        lines_tags = [item['seamark:type'] for item in list(lines_concat['tags'])]
        test_tags = [item['seamark:type'] for item in list(concat_df['tags'])]
        assert len(set(lines_tags).intersection(set(test_tags))) > 0, 'no intersection'
        # assert concat_df['tags'].values in nodes_concat['tags'].values
        for geom in lines_concat["geom"]:
            assert isinstance(geom, LineString), "Linestring Instantiation Error"


    def test_concat_nodes_ways(self):
        """
            test for checking if table nodes is gdf and geometry is Linestring type
            https://shapely.readthedocs.io/en/stable/geometry.html
        """
        concat_all = ContinuousCheck().concat_nodes_ways(
            query=["SELECT * FROM nodes","SELECT *, linestring AS geom FROM ways"],
            engine = engine
        )

        point1 = {"tags": [{'seamark:type': 'separation_line'}], "geometry": [Point(1, 2)]}
        point1_df = gpd.GeoDataFrame(point1)

        line1 = {"tags": [{'seamark:type': 'separation_line'}], "geometry": [LineString([(7, 8), (5, 9)])]}
        line1_df = gpd.GeoDataFrame(line1)

        concat_df_all = pd.concat([point1_df, line1_df])
        #concat_df_all = concat_df_all[concat_df_all["geom"] != None]


        assert isinstance(concat_all, gpd.GeoDataFrame)

        assert type(concat_all) == type(concat_df_all)

        type_list = [type(geometry) for geometry in concat_all['geom']]
        assert set(type_list).intersection([Point, LineString]), 'Geometry type error'

    def test_gdf_seamark_combined_nodes_ways():
        """
        test for checking if table nodes is gdf and geometry is Linestring type
        """

        #gdf with nodes and linestrings
        nodes_lines_concat = ContinuousCheck().gdf_seamark_combined_nodes_ways(
            engine,
            query=["SELECT * FROM nodes","SELECT *, linestring AS geom FROM ways"],
            seamark_object=["nodes", "ways"],
            seamark_list=["separation_zone", "separation_line"],
        )

        #creating dummy data point
        point1 = {"tags": [{'seamark:type': 'separation_line'}], "geometry": [Point(1, 2)]}

        point1_df = gpd.GeoDataFrame(point1)
        point2 = {"tags": [{'seamark:type': 'separation_zone'}], "geometry": [Point(6, 2)]}
        point2_df = gpd.GeoDataFrame(point2)

        concat_df = pd.concat([point1_df, point2_df])

        # creating dummy data linestring
        line1 = {"tags": [{'seamark:type': 'separation_line'}], "geometry": [LineString([(7, 8), (5, 9)])]}
        line1_df = gpd.GeoDataFrame(line1)
        line2 = {"tags": [{'seamark:type': 'separation_zone'}], "geometry": [LineString([(1, 2), (3, 4)])]}
        line2_df = gpd.GeoDataFrame(line2)
        concat_df = pd.concat([line1_df, line2_df])

        assert isinstance(lines_concat, gpd.GeoDataFrame)
        assert type(concat_df) == type(lines_concat)
        lines_tags = [item['seamark:type'] for item in list(lines_concat['tags'])]
        test_tags = [item['seamark:type'] for item in list(concat_df['tags'])]
        assert len(set(lines_tags).intersection(set(test_tags))) > 0, 'no intersection'
        # assert concat_df['tags'].values in nodes_concat['tags'].values
        for geom in lines_concat["geom"]:
            assert isinstance(geom, LineString), "Linestring Instantiation Error"
