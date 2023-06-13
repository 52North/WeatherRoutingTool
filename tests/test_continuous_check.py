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
# Load the environment variables from the .env file
load_dotenv()

os.chdir('/home/igor/projects/maridata/MariGeoRoute/WeatherRoutingTool')#os.path.dirname(__file__))

#current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(os.getcwd(),''))


from constraints.constraints import *

engine = db.create_engine('postgresql://{user}:{pw}@{host}/{db}'
                          .format(user='myuser',
                                  pw='mypassword',
                                  host='172.26.0.2',
                                  db='mydatabase',
                                  port='5434'))


class TestContinuousCheck:
    '''
        include test methods for the continuous check in the negative constraints
    '''

    def test_connect_database(self):
        '''
            test for checking the engine object and if the connection with the db is estabilished
        '''

        # parameters might change when creating the engine
        self.connection = engine.connect()

        # returns 1 if the connection is estabilished and 0 otherwise
        status = self.connection.connection.connection.status

        assert isinstance(ContinuousCheck().connect_database(),
                          type(db.create_engine('postgresql://myuser:***@172.25.0.3/mydatabase'))), 'Engine Instantiation Error'
        assert status == 1, 'Connection not estalished'



    def test_query_nodes(self):
        '''
            test for checking if table nodes is gdf and geometry is Point type
        '''

        gdf = gpd.read_postgis('select * from nodes', engine)
        gdf['tstamp'] = gdf['tstamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        gdf = gdf[gdf['geom'] != None]

        point = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        point_df = gpd.GeoDataFrame(point)
        assert type(gdf) == type(point_df), 'GeoDataFrame Type Error'
        for geom in point_df['geometry']:
            assert isinstance(geom, Point), 'Point Instantiation Error'
        print('point type checked')


    def test_query_ways(self):
        '''
            test for checking if table nodes is gdf and geometry is Linestring type
        '''


        # Use geopandas to read the SQL query into a dataframe from postgis
        gdf = gpd.read_postgis("SELECT *, linestring AS geom FROM ways", engine)
        # read timestamp type data as string
        gdf['tstamp']=gdf['tstamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        gdf = gdf[gdf['geom'] != None]

        line = {'col1': ['name1', 'name2'],
                'geometry': [LineString([(1, 2), (3, 4)]), LineString([(2, 1), (5, 6)])]}
        line_df = gpd.GeoDataFrame(line)
        assert type(gdf)== type(line_df), 'GeoDataFrame Type Error'
        for geom in gdf['geom']:
            assert isinstance(geom, LineString), 'LineString Instantiation Error'
        print('Linestring type checked')
