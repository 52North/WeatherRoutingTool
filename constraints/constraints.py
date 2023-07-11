import datetime as dt
import logging
import time

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from global_land_mask import globe
import ast

import utils.graphics as graphics
import utils.formatting as form
from routeparams import RouteParams
from utils.maps import Map
from weather import WeatherCond

## used as a part of the continuouscheck class ##
import os
import sqlalchemy as db
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from shapely.geometry import Point,LineString
from shapely import STRtree

# Load the environment variables from the .env file
load_dotenv()

logger = logging.getLogger("WRT.Constraints")


##
# Constraint: Main class for handling of constraints
# PositiveConstraint: handling of constraints where the ship NEEDS to take a certain route (e.g. waterways)
# NegativeConstraint: handling of constraints where the ship MUST NOT pass a certain area (too low water depth, too high waves, danger areas...)
# NegativeConstraintFrom weather: negative constraint which needs information from the weather (this includes depth information which are stored in the netCDF weather file)
# ConstraintPars: class that initialises ConstraintList
# ConstraintList: list of constraints
#
# constraints implemented so far: LandCrossing (prohibit land crossing), WaterDepth (prohibit crossing of areas with too low water depth), StayOnMap (prohibit leaving the area
#           for which the weather data has been obtained
#
# Inclusion in routing algorithm:
#       1) initialise all individual constraints that shall be considered (for example check Isochrones/execute_routing.py)
#       2) initialise ConstraintList object and add all constraints that shall be considered (for example check Isochrones/execute_routing.py)
#       3) during the routing procedure, check for ConstraintList.safe_crossing(lat_start, lat_end, lon_start, lon_end, time) which looks for constraints in
#           between starting point and destination;
#           alternatively it can also be checked for a single point whether a constraint is hit via ConstraintList.safe_endpoint(lat, lon, time)


class Constraint:
    name: str
    message: str
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray

    # resource_type: int

    def __init__(self, name):
        self.name = name

    # def get_resource_type(self):
    #    return self.resource_type

    def print_constraint_message(self):
        print(self.message)
        pass

    def constraint_on_point(self, lat, lon, time):
        pass

    def print_debug(self, message):
        print(self.name + str(": ") + str(message))

    def print_info(self):
        pass

    def plot_route_in_constraint(self, route: RouteParams, colour, fig, ax):
        fig, ax = self.plot_constraint(fig, ax)
        route.plot_route(ax, graphics.get_colour(0), "")
        return ax

    def plot_constraint(self):
        pass


class PositiveConstraint(Constraint):
    def __init__(self, name):
        Constraint.__init__(self, name)

    def get_points(self):
        pass


class PositiveConstraintPoint(PositiveConstraint):
    coord: tuple

    def __init__(self, lat, lon):
        self.coord = (lat, lon)

        super().__init__("OverWaypoint")

    def get_points(self):
        return self.coord


class NegativeContraint(Constraint):
    def __init__(self, name):
        Constraint.__init__(self, name)
        self.message = "At least one point discarded as "


class NegativeConstraintFromWeather(NegativeContraint):
    wt: WeatherCond

    def __init__(self, name, weather):
        NegativeContraint.__init__(self, name)
        self.wt = weather

    def check_weather(self, lat, lon, time):
        pass


class ConstraintPars:
    resolution: int
    bCheckEndPoints: bool
    bCheckCrossing: bool

    def __init__(self):
        self.resolution = 1.0 / 20
        self.bCheckEndPoints = True
        self.bCheckCrossing = True

    def print(self):
        logger.info("Print settings of Constraint Pars:")
        logger.info(form.get_log_step("resolution=" + str(self.resolution), 1))
        logger.info(
            form.get_log_step("bCheckEndPoints=" + str(self.bCheckEndPoints), 1)
        )


class ConstraintsList:
    pars: ConstraintPars
    positive_constraints: list
    negative_constraints_discrete: list
    negative_constraints_continuous: list
    neg_dis_size: int
    neg_cont_size: int
    pos_size: int

    positive_point_dict: dict
    current_positive: int

    constraints_crossed: list
    weather: WeatherCond

    def __init__(self, pars):
        self.pars = pars
        self.positive_constraints = []
        self.negative_constraints_discrete = []
        self.negative_constraints_continuous = []
        self.constraints_crossed = []
        self.neg_dis_size = 0
        self.neg_cont_size = 0
        self.pos_size = 0

    def print_constraints_crossed(self):
        print("Discarding point as:")
        for iConst in range(0, len(self.constraints_crossed)):
            form.print_step(str(self.constraints_crossed[iConst]), 1)

    def print_settings(self):
        self.pars.print()
        self.print_active_constraints()

    def print_active_constraints(self):
        for Const in self.negative_constraints_continuous:
            Const.print_info()

        for Const in self.negative_constraints_discrete:
            Const.print_info()

        for Const in self.positive_constraints:
            Const.print_info()

    def have_positive(self):
        if self.pos_size > 0:
            return True
        else:
            return False

    def have_negative(self):
        if self.neg_size > 0:
            return True
        else:
            return False

    def init_positive_lists(self, start, finish):
        lat = []
        lon = []
        lat.append(start[0])
        lon.append(start[1])
        for iconst in self.positive_constraints:
            lat.append(iconst.get_points()[0])
            lon.append(iconst.get_points()[1])

        lat.append(finish[0])
        lon.append(finish[1])

        print("Tuple of positive constraints:")
        print("lat: ", lat)
        print("lon: ", lon)

        self.positive_point_dict = {"lat": lat, "lon": lon}
        self.current_positive = 0

    def reached_positive(self):
        self.current_positive = self.current_positive + 1

    def get_current_destination(self):
        finish_lat = self.positive_point_dict["lat"][self.current_positive + 1]
        finish_lon = self.positive_point_dict["lon"][self.current_positive + 1]
        return (finish_lat, finish_lon)

    def get_current_start(self):
        start_lat = self.positive_point_dict["lat"][self.current_positive]
        start_lon = self.positive_point_dict["lon"][self.current_positive]
        return (start_lat, start_lon)

    def shall_I_pass(self, lat, lon, time):
        is_constrained = [False for i in range(0, lat.shape[1])]

        if self.pars.bCheckCrossing:
            is_constrained = self.safe_crossing(lat, lon, time)
        elif self.pars.bCheckEndPoints:
            is_constrained = self.safe_endpoint(lat, lon, time, is_constrained)
        if is_constrained.any():
            self.print_constraints_crossed()

    def split_route(self):
        pass

    ##
    # Check whether there is a constraint on the space-time point defined by lat, lon, time. To do so, the code loops
    # over all Constraints added to the ConstraintList
    def safe_endpoint(self, lat, lon, current_time, is_constrained):
        debug = False

        for iConst in range(0, self.neg_dis_size):
            is_constrained_temp = self.negative_constraints_discrete[iConst].constraint_on_point(
                lat, lon, current_time
            )
            if is_constrained_temp.any():
                self.constraints_crossed.append(
                    self.negative_constraints_discrete[iConst].message
                )
            if debug:
                print("is_constrained_temp: ", is_constrained_temp)
                print("is_constrained: ", is_constrained)
                # form.print_current_time('constraint execution', start_time)

            is_constrained += is_constrained_temp
        # if (is_constrained.any()) & (debug): self.print_constraints_crossed()
        return is_constrained

    def safe_crossing(self, lat_start, lat_end, lon_start, lon_end, current_time, is_constrained):
        is_constrained_discrete = self.safe_crossing_discrete(lat_start, lat_end, lon_start, lon_end, current_time, is_constrained)
        is_constrained_continuous = self.safe_crossing_continuous(lat_start, lat_end, lon_start, lon_end, current_time)
        is_constrained = is_constrained + is_constrained_discrete + is_constrained_continuous
        return is_constrained

    def safe_crossing_continuous(self, lat_start, lat_end, lon_start, lon_end, current_time):
        debug = True
        is_constrained = []

        if debug:
            print('Entering continuous checks')
            print('Length of latitudes: ' + str(len(lat_start)))
        for constr in self.negative_constraints_continuous:
            is_constrained_temp = constr.check_crossing(lat_start, lat_end, lon_start, lon_end, current_time)
            print('is_constrained_temp: ', is_constrained_temp)
            is_constrained = is_constrained + is_constrained_temp
            print('is_constrained: ', is_constrained)

        print('is_constrained_final: ', is_constrained)
        return is_constrained

    ##
    # Check whether there is a constraint on the way from a starting point (lat_start, lon_start) to the destination (lat_end, lon_end).
    # To do so, the code segments the travel distance into steps (step length given by ConstraintPars.resolution) and loops through all these steps
    # calling ConstraintList.safe_endpoint()
    def safe_crossing_discrete(
            self, lat_start, lat_end, lon_start, lon_end, current_time, is_constrained
    ):
        debug = False

        delta_lats = (lat_end - lat_start) * self.pars.resolution
        delta_lons = (lon_end - lon_start) * self.pars.resolution
        x0 = lat_start
        y0 = lon_start

        # if (debug):
        # form.print_step('Constraints: Moving from (' + str(lat_start) + ',' + str(lon_start) + ') to (' + str(
        #        lat_end) + ',' + str(lon_end), 0)

        nSteps = int(1.0 / self.pars.resolution)
        for iStep in range(0, nSteps):
            x = x0 + delta_lats
            y = y0 + delta_lons

            is_constrained = self.safe_endpoint(x, y, current_time, is_constrained)
            x0 = x
            y0 = y

        if debug:
            lat_start_constrained = lat_start[is_constrained == 1]
            lon_start_constrained = lon_start[is_constrained == 1]
            lat_end_constrained = lat_end[is_constrained == 1]
            lon_end_constrained = lon_end[is_constrained == 1]

            if lat_start_constrained.shape[0] > 0:
                form.print_step("transitions constrained:", 1)
            for i in range(0, lat_start_constrained.shape[0]):
                form.print_step(
                    "["
                    + str(lat_start_constrained[i])
                    + ","
                    + str(lon_start_constrained[i])
                    + "] to ["
                    + str(lat_end_constrained[i])
                    + ","
                    + str(lon_end_constrained[i])
                    + "]",
                    2,
                )

        # if not ((round(x0.all,8) == round(self.lats_per_step[0, :].all) and (x0.all == self.lons_per_step[0, :].all)):
        #    exc = 'Did not check destination, only checked lat=' + str(x0) + ', lon=' + str(y0)
        #    raise ValueError(exc)

        if not np.allclose(x, lat_end):
            raise Exception(
                "Constraints.land_crossing(): did not reach latitude of destination!"
            )
        if not np.allclose(y, lon_end):
            raise Exception(
                "Constraints.land_crossing(): did not reach longitude of destination!"
            )

        return is_constrained

    def add_pos_constraint(self, constraint):
        self.positive_constraints.append(constraint)
        self.pos_size += 1

    def add_neg_constraint(self, constraint, option = 'discrete'):
        if option == 'discrete':
            self.negative_constraints_discrete.append(constraint)
            self.neg_dis_size += 1
            return

        if option == 'continuous':
            self.negative_constraints_continuous.append(constraint)
            self.neg_cont_size += 1
            return

        raise ValueError('You chose to add a negetive constraint with option ' + option + '. However only options -discrete- and -continuous- are implemented ')

    def check_weather(self):
        pass


class LandCrossing(NegativeContraint):
    def __init__(self):
        NegativeContraint.__init__(self, "LandCrossing")
        self.message += "crossing land!"
        # self.resource_type = 0

    def constraint_on_point(self, lat, lon, time):
        # self.print_debug('checking point: ' + str(lat) + ',' + str(lon))
        return globe.is_land(lat, lon)

    def print_info(self):
        logger.info(form.get_log_step("no land crossing", 1))


class WaveHeight(NegativeConstraintFromWeather):
    current_wave_height: np.ndarray
    max_wave_height: float

    def __init__(self):
        NegativeContraint.__init__(self, "WaveHeight")
        self.message += "waves are to high!"
        # self.resource_type = 0
        self.current_wave_height = np.array([-99])
        self.max_wave_height = 10

    def constraint_on_point(self, lat, lon, time):
        # self.print_debug('checking point: ' + str(lat) + ',' + str(lon))
        self.check_weather(lat, lon, time)
        # print('current_wave_height:', self.current_wave_height)
        return self.current_wave_height > self.max_wave_height

    def print_info(self):
        logger.info(
            form.get_log_step(
                "maximum wave height=" + str(self.max_wave_height) + "m", 1
            )
        )


class WaterDepth(NegativeContraint):
    map: Map
    depth_data: xr
    current_depth: np.ndarray
    min_depth: float

    def __init__(self, depth_path, drougth, map, rename = True):
        NegativeContraint.__init__(self, "WaterDepth")
        self.message += "water not deep enough!"

        ds_depth = xr.open_dataset(
            depth_path, chunks={"time": "500MB"}, decode_times=False
        )

        if rename:
            self.depth_data = ds_depth.rename(z="depth", lat="latitude", lon="longitude")
        else:
            self.depth_data = ds_depth

        self.current_depth = np.array([-99])
        self.min_depth = drougth
        self.map = map

    def set_drought(self, depth):
        self.min_depth = depth

    def constraint_on_point(self, lat, lon, time):
        self.check_depth(lat, lon, time)
        returnvalue = self.current_depth > -self.min_depth
        # form.print_step('current_depth:' + str(self.current_depth), 1)
        return returnvalue

    def check_depth(self, lat, lon, time):
        lat_da = xr.DataArray(lat, dims="dummy")
        lon_da = xr.DataArray(lon, dims="dummy")
        rounded_ds = self.depth_data["depth"].interp(
            latitude=lat_da, longitude=lon_da, method="linear"
        )
        self.current_depth = rounded_ds.to_numpy()

    def print_info(self):
        logger.info(
            form.get_log_step("minimum water depth=" + str(self.min_depth) + "m", 1)
        )

    def get_current_depth(self, lat, lon):
        self.check_depth(lat, lon, None)
        return self.current_depth

    def plot_depth_map_from_file(self, path):
        level_diff = 10

        ds_depth = xr.open_dataset(path)
        depth = ds_depth["z"].where(
            (ds_depth.lat > self.map.lat1)
            & (ds_depth.lat < self.map.lat2)
            & (ds_depth.lon > self.map.lon1)
            & (ds_depth.lon < self.map.lon2)
            & (ds_depth.z < 0),
            drop=True,
        )

        # depth = ds_depth['deptho'].where((ds_depth.latitude > lat_start) & (ds_depth.latitude < lat_end) & (ds_depth.longitude > lon_start) & (ds_depth.longitude < lon_end),drop=True) #.where((ds_depth.deptho>-100) & (ds_depth.deptho<0) )

        fig, ax = plt.subplots(figsize=(12, 10))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        depth.plot.contourf(
            ax=ax,
            # extend='max',
            # levels=np.arange(-60, 0, 0.1),  high-resolution contours
            levels=np.arange(-100, 0, level_diff),
            transform=ccrs.PlateCarree(),
        )

        fig.subplots_adjust(
            left=0.05, right=1, bottom=0.05, top=0.95, wspace=0, hspace=0
        )
        ax.add_feature(cf.LAND)
        ax.add_feature(cf.COASTLINE)
        ax.gridlines(draw_labels=True)

        plt.show()

    def plot_constraint(self, fig, ax):
        level_diff = 10
        plt.rcParams["font.size"] = 20
        ax.axis("off")
        ax.xaxis.set_tick_params(labelsize="large")

        ds_depth = self.depth_data.coarsen(
            latitude=10, longitude=10, boundary="exact"
        ).mean()
        ds_depth_coarsened = ds_depth.compute()

        self.depth_data = ds_depth_coarsened.where(
            (ds_depth_coarsened.latitude > self.map.lat1)
            & (ds_depth_coarsened.latitude < self.map.lat2)
            & (ds_depth_coarsened.longitude > self.map.lon1)
            & (ds_depth_coarsened.longitude < self.map.lon2)
            & (ds_depth_coarsened.depth < 0),
            drop=True,
        )

        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cp = self.depth_data["depth"].plot.contourf(
            ax=ax, levels=np.arange(-100, 0, level_diff), transform=ccrs.PlateCarree()
        )
        fig.colorbar(cp, ax=ax, shrink=0.7, label="Wassertiefe (m)", pad=0.1)

        fig.subplots_adjust(left=0.1, right=1.2, bottom=0, top=1, wspace=0, hspace=0)
        ax.add_feature(cf.LAND)
        ax.add_feature(cf.COASTLINE)
        ax.gridlines(draw_labels=True)
        plt.title("")

        return fig, ax


class StayOnMap(NegativeContraint):
    lat1: float
    lon1: float
    lat2: float
    lon2: float

    def __init__(self):
        NegativeContraint.__init__(self, "StayOnMap")
        self.message += "leaving wheather map!"
        # self.resource_type = 0

    def constraint_on_point(self, lat, lon, time):
        # self.print_debug('checking point: ' + str(lat) + ',' + str(lon))
        is_on_map = (
                (lat > self.lat2)
                + (lat < self.lat1)
                + (lon > self.lon2)
                + (lon < self.lon1)
        )
        return is_on_map

    def print_info(self):
        logger.info(form.get_log_step("stay on wheather map", 1))

    def set_map(self, lat1, lon1, lat2, lon2):
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2


class ContinuousCheck(NegativeContraint):
    """
    Contains various functions to test data connection,
    gathering and use for obtaining spatial relations
    for the continuous check in the negative constraints

    Attributes
    ----------
    
    host : str
    database : str
    user : str
    password : str
    port : str
        returns values from .env  to be passed in the engine of the db

    predicates : list
        Possible spatial relations to be tested when considering the constraints

    tags : list
        Values of the seamark tags that need to be considered
    """

    def __init__(self):
        NegativeContraint.__init__(self, "ContinuousChecks")
        self.host = os.getenv("HOST")
        self.database = os.getenv("DATABASE")
        self.user = os.getenv("USER")
        self.password = os.getenv("PASSWORD")
        self.port = os.getenv("PORT")
        self.land_polygons = "land_polygons"
        self.query=["SELECT * FROM nodes","SELECT *, linestring AS geom FROM ways"]
        self.predicates = ["intersects", "contains", "touches", "crosses", "overlaps"]
        self.tags = [
            "separation_zone",
            "separation_line",
            #"separation_lane",
            "restricted_area",
            #"separation_roundabout",
        ]

    def print_info(self):
        logger.info(form.get_log_step("no seamarks crossing", 1))

    def connect_database(self):
        """
        Connect to the database

        Parameters
        ----------
        No arguments


        Returns
        ----------
        Engine of PostgreSQL
        """

        # Connect to the PostgreSQL database using SQLAlchemy
        engine = db.create_engine(
            "postgresql://{user}:{pw}@{host}/{db}".format(
                user=self.user,
                pw=self.password,
                host=self.host,
                db=self.database,
                port=self.port,
            )
        )
        return engine

    def query_nodes(self, engine, query="SELECT * FROM nodes"):
        """
        Create new GeoDataFrame using public.nodes table in the query

        Parameters
        ----------
        engine : sqlalchemy engine
            engine object

        query : str
            sql query for table nodes

        Returns
        ----------
        gdf : GeoDataFrame
            gdf including all the features from public.nodes table
        """

        # Define SQL query to retrieve list of tables
        # sql_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        query = query

        # Use geopandas to read the SQL query into a dataframe from postgis
        gdf = gpd.read_postgis(query, engine, geom_col="geom")

        # Eliminate none values
        gdf = gdf[gdf["geom"] != None]

        # read timestamp type data as string
        # gdf['tstamp']=gdf['tstamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return gdf

    def query_ways(self, engine, query="SELECT *, linestring AS geom FROM ways"):
        """
        Create new GeoDataFrame using public.ways table in the query

        Parameters
        ----------
        engine : sqlalchemy engine
            engine object

        query : str
            sql query for table ways

        Returns
        ----------
        gdf : GeoDataFrame
            gdf including all the features from public.ways table
        """

        # Define SQL query to retrieve list of tables
        # sql_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        query = query

        # Use geopandas to read the SQL query into a dataframe from postgis
        gdf = gpd.read_postgis(query, engine, geom_col="geom")

        # Eliminate none values
        gdf = gdf[gdf["geom"] != None]

        # read timestamp type data as string
        # gdf['tstamp']=gdf['tstamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return gdf

    def gdf_seamark_combined_nodes(
            self, engine, query, seamark_object=list, seamark_list=list
    ):
        """
        Create new GeoDataFrame with specified seamark tags

        Parameters
        ----------
        engine : sqlalchemy engine
            engine object

        query : str
            sql query for table nodes

        seamark_object : list
            value nodes (which table to be considered)

        seamark_list : list
            list of all the tags that must be considered for filtering specific seamark objects


        Returns
        ----------
        gdf_concat : GeoDataFrame
            gdf including all the features with specified seamark tags using nodes OSM element
        """

        ##################### optional for the moment ######################
        if ("nodes" in seamark_object) and all(
                element in self.tags for element in seamark_list
        ):
            gdf = self.query_nodes(engine, query)
            gdf_list = []
            for i in range(0, len(seamark_list)):
                if type(gdf["tags"][i]) == str:
                    gdf["tags"] = gdf["tags"].apply(ast.literal_eval)
                    gdf1 = gdf[
                        gdf["tags"].apply(lambda x: seamark_list[i] in x.values())
                    ]
                    gdf_list.append(gdf1)
                else:
                    gdf1 = gdf[
                        gdf["tags"].apply(lambda x: seamark_list[i] in x.values())
                    ]
                    gdf_list.append(gdf1)
            gdf_concat = pd.concat(gdf_list)

        return gdf_concat

    def gdf_seamark_combined_ways(
            self, engine, query, seamark_object=list, seamark_list=list
    ):
        """
         Create new GeoDataFrame with specified seamark tags

         Parameters
         ----------
         engine : sqlalchemy engine
             engine object

         query : str
             sql query for table ways

         seamark_object : list
             value ways (which table to be considered)

         seamark_list : list
             list of all the tags that must be considered for filtering specific seamark objects


         Returns
         ----------
         gdf_concat : GeoDataFrame
             gdf including all the features with specified seamark tags using ways OSM element
         """

        if ("ways" in seamark_object) and all(
                element in self.tags for element in seamark_list
        ):
            gdf = self.query_ways(engine, query)
            gdf_list = []
            for i in range(0, len(seamark_list)):
                if type(gdf["tags"][i]) == str:
                    gdf["tags"] = gdf["tags"].apply(ast.literal_eval)
                    gdf1 = gdf[
                        gdf["tags"].apply(lambda x: seamark_list[i] in x.values())
                    ]
                    gdf_list.append(gdf1)
                else:
                    gdf1 = gdf[
                        gdf["tags"].apply(lambda x: seamark_list[i] in x.values())
                    ]
                    gdf_list.append(gdf1)
            gdf_concat = pd.concat(gdf_list)

        return gdf_concat

    def concat_nodes_ways(self, query, engine):
        """
         Create new GeoDataFrame using public.ways and public.nodes table together in the query

         Parameters
         ----------
         query : str
             sql query for table ways

        engine : sqlalchemy engine
             engine object

         Returns
         ----------
         gdf_all : GeoDataFrame
             gdf including all the features from public.ways and public.nodes table
         """

        # consider the scenario for a tag present in nodes and ways at the same time
        for query in query:
            if "nodes" in query:
                gdf_nodes = self.query_nodes(engine, query)
            elif "ways" in query:
                gdf_ways = self.query_ways(engine, query)
            else:
                print("false query passed")

        # checks if there are repeated values in both gdfs
        if gdf_nodes.overlaps(gdf_ways).values.sum() == 0:
            gdf_all = pd.concat([gdf_nodes, gdf_ways])
        else:
            gdf_all = pd.concat([gdf_nodes, gdf_ways]).drop_duplicates(
                subset="id", keep="first"
            )

        return gdf_all

    def gdf_seamark_combined_nodes_ways(
            self, engine, query=list, seamark_object=list, seamark_list=list
    ):
        """
         Create new GeoDataFrame with specified seamark tags

         Parameters
         ----------
         engine : sqlalchemy engine
             engine object

         query : list
             list of str for the sql query for table nodes and ways

         seamark_object : list
            value nodes, ways (which table to be considered)

         seamark_list : list
             list of all the tags that must be considered for filtering specific seamark objects


         Returns
         ----------
         gdf_concat : GeoDataFrame
             gdf including all the features with specified seamark tags using nodes and ways OSM element
         """
        # gdf_concat = gpd.GeoDataFrame()

        if (
                ("nodes" in seamark_object)
                and ("ways" in seamark_object)
                and all(element in self.tags for element in seamark_list)
        ):
            gdf = ContinuousCheck().concat_nodes_ways(query, engine)

            gdf_list = []
            for i in range(0, len(seamark_list)):
                if type(gdf["tags"].iloc[i]) == str:
                    gdf["tags"] = gdf["tags"].apply(ast.literal_eval)
                    gdf1 = gdf[
                        gdf["tags"].apply(lambda x: seamark_list[i] in x.values())
                    ]
                    gdf_list.append(gdf1)
                else:
                    gdf1 = gdf[
                        gdf["tags"].apply(lambda x: seamark_list[i] in x.values())
                    ]
                    gdf_list.append(gdf1)

            gdf_concat = pd.concat(gdf_list)
        else:
            gdf_concat = gpd.GeoDataFrame()
            logger.info("Check the seamark object and seamark tag list")

        return gdf_concat

    def check_crossing(self, lat_start, lon_start, lat_end, lon_end, time=None):  #best way to go (keep just these arguments)
        """
         Check if certain route crosses specified seamark objects

         Parameters
         ----------
         lat_start : np.array
            array of all origin latitudes of routing segments

        lon_start : np.array
            array of all origin longitudes of routing segments

         lat_end : np.array
            array of all destination latitudes of routing segments

        lon_end : np.array
            array of all destination longitudes of routing segments

         time : datetime.datetime (optional argument)

         Returns
         ----------
         query_tree : list
             bool of spatial relation result (True or False)
         """

        query_tree = []

        concat_gdf = self.gdf_seamark_combined_nodes_ways(engine=self.connect_database(),query=self.query,seamark_object=["nodes","ways"],seamark_list=self.tags)
        lines = []

        # generating the LineString geometry from start and end point
        for i in len(lat_start):
            start_point = Point(lat_start[i],lon_start[i])
            end_point = Point(lat_end[i],lon_end[i])
            line = LineString([start_point,end_point])
            lines.append(line)

        # creating geospatial dataframe objects from linestring geometries
        route_df = gpd.GeoDataFrame(lines)

        # checking the spatial relations using shapely.STRTree spatial indexing method
        for predicate in self.predicates:
            concat_df = concat_gdf
            tree = STRtree(concat_df["geom"])
            geom_object = tree.query(route_df["geom"], predicate=predicate).tolist()

            # checks if there is spatial relation between routes and seamarks objects
            if geom_object == [[], []] or geom_object == []:
                # if route is not constrained
                query_tree.append(False)
            else:
                # if route is constrained
                query_tree.append(True)

        # returns a list bools (spatial relation)
        return query_tree

    """
    query_tree = []


        for predicate in self.predicates:
            concat_df = self.gdf_seamark_combined_nodes_ways(
                engine, query, seamark_object, seamark_list
            )
            tree = STRtree(concat_df["geom"])
            geom_object = tree.query(route_df["geom"], predicate=predicate).tolist()

            if geom_object == [[], []] or geom_object == []:
                print(f"no {predicate} for the geometry objects with {geom_object}")
            else:
                print(f"{predicate} for the geometry object with {geom_object}")
                query_tree.append((tree, predicate, geom_object, True))

        # returns a list of tuples(shapelySTRTree, predicate, result_array, bool type)
        return query_tree"""
