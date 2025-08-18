import os
import logging

import cartopy.crs as ccrs
import cartopy.feature as cf
import datacube
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
import xarray as xr
from global_land_mask import globe
from shapely.geometry import Point, LineString, box
from shapely.strtree import STRtree

import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.formatting as form
from maridatadownloader import DownloaderFactory
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger("WRT.Constraints")


class Constraint:
    """
    Main class for handling of constraints. Constraints implemented so far:
    LandCrossing (prohibit land crossing), WaterDepth (prohibit crossing of areas with too low water depth),
    StayOnMap (prohibit leaving the area for which the weather data has been obtained)
    """

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
        logger.info(self.message)
        pass

    def constraint_on_point(self, lat, lon, time):
        pass

    def print_debug(self, message):
        logger.debug(self.name + str(": ") + str(message))

    def print_info(self):
        pass

    def plot_route_in_constraint(self, route: RouteParams, colour, fig, ax):
        fig, ax = self.plot_constraint(fig, ax)
        route.plot_route(ax, graphics.get_colour(0), "")
        return ax

    def plot_constraint(self):
        pass


class PositiveConstraint(Constraint):
    """
    Handling of constraints where the ship NEEDS to take a certain route (e.g. waterways)
    """

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

    def print_info(self):
        logger.info(form.get_log_step("intermediate waypoints activated for: " + str(self.coord), 1))


class NegativeContraint(Constraint):
    """
    Handling of constraints where the ship MUST NOT pass a certain area (too low water depth,
    too high waves, danger areas...)
    """

    def __init__(self, name):
        Constraint.__init__(self, name)
        self.message = "At least one point discarded as "


class NegativeConstraintFromWeather(NegativeContraint):
    """
    Negative constraint which needs information from the weather (this includes depth
    information which are stored in the netCDF weather file)
    """

    wt: WeatherCond

    def __init__(self, name, weather):
        NegativeContraint.__init__(self, name)
        self.wt = weather

    def check_weather(self, lat, lon, time):
        pass


class ConstraintPars:
    """
    Class that initialises ConstraintList
    """

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
        logger.info(form.get_log_step("bCheckEndPoints=" + str(self.bCheckEndPoints), 1))


class ConstraintsListFactory:
    """
    Initialise ConstraintList object and add all constraints that shall be considered
    """

    def __init__(self):
        pass

    @staticmethod
    def get_constraints_list(constraints_string_list, **kwargs):
        pars = ConstraintPars()
        constraints_list = ConstraintsList(pars)
        is_stay_on_map = False

        if 'land_crossing_global_land_mask' in constraints_string_list:
            land_crossing = LandCrossing()
            constraints_list.add_neg_constraint(land_crossing)

        if 'land_crossing_polygons' in constraints_string_list:
            map_size = kwargs.get('map_size')
            land_crossing_polygons = LandPolygonsCrossing(map_size)
            constraints_list.add_neg_constraint(land_crossing_polygons, 'continuous')

        if 'water_depth' in constraints_string_list:
            if ('data_mode' not in kwargs) or ('min_depth' not in kwargs) or ('depthfile' not in kwargs) or (
                    'map_size' not in kwargs):
                raise ValueError(
                    'To use the depth constraint module, you need to provide the data mode for the download, '
                    'the boat draught, the map size and the path to the depth file.')
            data_mode = kwargs.get('data_mode')
            min_depth = kwargs.get('min_depth')
            map_size = kwargs.get('map_size')
            depthfile = kwargs.get('depthfile')
            water_depth = WaterDepth(data_mode, min_depth, map_size, depthfile)
            constraints_list.add_neg_constraint(water_depth)

        if 'status_error' in constraints_string_list:
            courses_path = kwargs.get('courses_path')
            status_error = StatusCodeError(courses_path)
            constraints_list.add_neg_constraint(status_error, 'continuous')

        if 'on_map' in constraints_string_list:
            if 'map_size' not in kwargs:
                raise ValueError('To use the on-map constraint module, you need to providethe map size.')
            map_size = kwargs.get('map_size')
            on_map = StayOnMap()
            on_map.set_map(map_size.lat1, map_size.lon1, map_size.lat2, map_size.lon2)
            constraints_list.add_neg_constraint(on_map)
            is_stay_on_map = True

        if 'seamarks' in constraints_string_list:
            if is_stay_on_map:
                seamarks = SeamarkCrossing(is_stay_on_map, map_size)
            else:
                seamarks = SeamarkCrossing()
            constraints_list.add_neg_constraint(seamarks, 'continuous')

        if 'via_waypoints' in constraints_string_list:
            if 'waypoints' not in kwargs:
                raise ValueError('To use the waypoints constraint module, you need to provide the waypoints.')
            waypoints = kwargs.get('waypoints')
            for (lat, lon) in waypoints:
                wp = PositiveConstraintPoint(lat, lon)
                constraints_list.add_pos_constraint(wp)

        constraints_list.print_settings()
        return constraints_list


class ConstraintsList:
    """
    List of constraints. During the routing procedure, you can check for ConstraintList.safe_crossing(lat_start,
    lat_end, lon_start, lon_end, time) which looks for constraints in between starting point and destination.
    Alternatively it can also be checked for a single point whether a constraint is hit via
    ConstraintList.safe_endpoint(lat, lon, time).
    """

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
        logger.info("Discarding point as:")
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

        logger.info(form.get_log_step('Tuple of positive constraints:', 0))
        logger.info(form.get_log_step('lat:' + str(lat), 1))
        logger.info(form.get_log_step('lon:' + str(lon), 1))

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
        return start_lat, start_lon

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

    def safe_endpoint(self, lat, lon, current_time, is_constrained):
        """
        Check whether there is a constraint on the space-time point defined by lat, lon, time. To do so, the code loops
        over all Constraints added to the ConstraintList

        :param lat: Latitude of point to check
        :type lat: numpy.ndarray or float
        :param lon: Longitude of point to check
        :type lon: numpy.ndarray or float
        :param current_time: Time of point to check
        :type current_time: datetime
        :param is_constrained: List of booleans for every constraint stating if the point is constraint by it
        :type is_constrained: list[bool]
        :return: is_constrained
        :rtype: list[bool]
        """

        for iConst in range(0, self.neg_dis_size):
            is_constrained_temp = self.negative_constraints_discrete[iConst].constraint_on_point(lat, lon, current_time)
            if is_constrained_temp.any():
                self.constraints_crossed.append(self.negative_constraints_discrete[iConst].message)
            is_constrained += is_constrained_temp
        return is_constrained

    def safe_crossing(self, lat_start, lon_start, lat_end, lon_end, current_time, is_constrained):
        is_constrained_discrete = is_constrained
        is_constrained_continuous = is_constrained
        is_constrained_discrete = self.safe_crossing_discrete(lat_start, lon_start, lat_end, lon_end, current_time,
                                                              is_constrained)
        is_constrained_continuous = self.safe_crossing_continuous(lat_start, lon_start, lat_end, lon_end,
                                                                  is_constrained)

        # TO BE UPDATED
        is_constrained_array = np.array(is_constrained) | np.array(is_constrained_discrete) \
                                                        | np.array(is_constrained_continuous)
        is_constrained = is_constrained_array.tolist()
        return is_constrained

    def safe_crossing_continuous(self, lat_start, lon_start, lat_end, lon_end, is_constrained):
        """TODO: add description
        _summary_

        :param lat_start: Latitude of start point of section to check
        :type lat_start: ndarray or float
        :param lon_start: Longitude of start point of section to check
        :type lon_start: numpy.ndarray or float
        :param lat_end: Latitude of end point of section to check
        :type lat_end: numpy.ndarray or float
        :param lon_end: Longitude of end point of section to check
        :type lon_end: numpy.ndarray or float
        :param is_constrained: List of booleans for every constraint stating if the section is constraint by it
        :type is_constrained: list[bool]
        :return: is_constrained.tolist()
        :rtype: list[bool]
        """

        is_constrained = np.array(is_constrained)

        logger.debug('Entering continuous checks')
        logger.debug('Length of latitudes: ' + str(len(lat_start)))

        for constr in self.negative_constraints_continuous:
            is_constrained_temp = constr.check_crossing(lat_start, lon_start, lat_end, lon_end)
            is_constrained = np.array(is_constrained) | np.array(is_constrained_temp)

        return is_constrained.tolist()

    def safe_crossing_discrete(self, lat_start, lon_start, lat_end, lon_end, current_time, is_constrained):
        """
        Check whether there is a constraint on the way from a starting point (lat_start, lon_start) to the destination
        (lat_end, lon_end).
        To do so, the code segments the travel distance into steps (step length given by ConstraintPars.resolution) and
        loops through all these steps calling ConstraintList.safe_endpoint())

        :param lat_start: Latitude of start point of section to check
        :type lat_start: numpy.ndarray or float
        :param lon_start: Longitude of start point of section to check
        :type lon_start: numpy.ndarray or float
        :param lat_end: Latitude of end point of section to check
        :type lat_end: numpy.ndarray or float
        :param lon_end: Longitude of end point of section to check
        :type lon_end: numpy.ndarray or float
        :param current_time:Time of point to check
        :type current_time: datetime
        :param is_constrained: List of booleans for every constraint stating if the section is constraint by it
        :type is_constrained: list[bool]
        :return: is_constrained.tolist()
        :rtype: list[bool]
        """

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
                    "[" + str(lat_start_constrained[i]) + "," + str(lon_start_constrained[i]) + "] to [" + str(
                        lat_end_constrained[i]) + "," + str(lon_end_constrained[i]) + "]", 2, )

        # if not ((round(x0.all,8) == round(self.lats_per_step[0, :].all) and (x0.all == self.lons_per_step[0, :].all)):
        #    exc = 'Did not check destination, only checked lat=' + str(x0) + ', lon=' + str(y0)
        #    raise ValueError(exc)

        if not np.allclose(x, lat_end):
            raise Exception("Constraints.land_crossing(): did not reach latitude of destination!")
        if not np.allclose(y, lon_end):
            raise Exception("Constraints.land_crossing(): did not reach longitude of destination!")

        return is_constrained

    def add_pos_constraint(self, constraint):
        self.positive_constraints.append(constraint)
        self.pos_size += 1

    def add_neg_constraint(self, constraint, option='discrete'):
        if option == 'discrete':
            self.negative_constraints_discrete.append(constraint)
            self.neg_dis_size += 1
            return

        if option == 'continuous':
            self.negative_constraints_continuous.append(constraint)
            self.neg_cont_size += 1
            return

        raise ValueError(
            'You chose to add a negetive constraint with option ' + option + '. However only options -discrete- and '
                                                                             '-continuous- are implemented ')

    def check_weather(self):
        pass


class LandCrossing(NegativeContraint):
    """
    Constraint such that the boat cannot cross land
    """

    def __init__(self):
        NegativeContraint.__init__(self, "LandCrossing")
        self.message += "crossing land!"  # self.resource_type = 0

    def constraint_on_point(self, lat, lon, time):
        # self.print_debug('checking point: ' + str(lat) + ',' + str(lon))
        return globe.is_land(lat, lon)

    def print_info(self):
        logger.info(form.get_log_step("no land crossing", 1))


class StatusCodeError(NegativeContraint):
    """
    Negative constraint for points where mariPower returns a status code of 3 (=error).
    FIXME: currently, this constraint is added as 'continuous' constraint, but a 'discrete' constraint would
        be more suitable/intuitive. However, this cannot be used at the moment because discrete constraints are checked
        on intermediate points between two consecutive routing points and the status code is not available for these.
    """
    courses_path: str

    def __init__(self, courses_path):
        NegativeContraint.__init__(self, 'StatusErrorCode')
        self.message = 'At least one point discarded as routes have error status! '
        self.courses_path = courses_path

    def load_data_from_file(self, courses_path):
        routeData = xr.open_dataset(courses_path)
        status = routeData['Status'].to_numpy().flatten()
        lats = np.repeat(routeData.lat.values, len(routeData.it_course))
        lons = np.repeat(routeData.lon.values, len(routeData.it_course))
        routeData.close()
        return status, lats, lons

    def check_crossing(self, lat_start=None, lon_start=None, lat_end=None, lon_end=None, current_time=None):
        status, lats_netcdf, lon_netcdf = self.load_data_from_file(self.courses_path)
        # Double-check coordinates
        assert (lats_netcdf == lat_start).all()
        assert (lon_netcdf == lon_start).all()
        # status error = 3, warning = 2, OK = 1
        is_status_error = (status == 3)
        return is_status_error


class WaveHeight(NegativeConstraintFromWeather):
    """
    Constraint such that the boat can't cross a certain wave hight
    """

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
        # logger.info('current_wave_height:', self.current_wave_height)
        return self.current_wave_height > self.max_wave_height

    def print_info(self):
        logger.info(form.get_log_step("maximum wave height=" + str(self.max_wave_height) + "m", 1))


class WaterDepth(NegativeContraint):
    """
    Constraint such that the boat can't cross over a minimal water depth
    """

    map_size: Map
    depth_data: xr  # the xarray.Dataset is expected to have a variable called "z" (as in the original ETOPO dataset)
    current_depth: np.ndarray
    min_depth: float

    def __init__(self, data_mode, min_depth, map_size, depth_path=''):
        NegativeContraint.__init__(self, 'WaterDepth')
        self.message += 'water not deep enough!'
        self.current_depth = np.array([-99])
        self.min_depth = min_depth
        self.map_size = map_size

        self.depth_data = None

        if data_mode == 'odc':
            self.depth_data = self.load_data_ODC(depth_path, 'global_relief', measurements=['z'])
        elif data_mode == 'automatic':
            self.depth_data = self.load_data_automatic(depth_path)
        elif data_mode == 'from_file':
            self.depth_data = self.load_data_from_file(depth_path)
        else:
            raise ValueError('Option "' + data_mode + '" not implemented for download of depth data!')

    def load_data_ODC(self, depth_path, product_name, measurements=None):
        """
        Load depth data from ODC

        :param depth_path: Path to where depth data should be written to
        :type depth_path: str
        :param product_name: Name of the ODC dataset
        :type product_name: str
        :param measurements: A list of band names to load from the product_name dataset, defaults to None
        :type measurements: list[str], optional
        :raises ValueError: product_name is not known on ODC
        :raises KeyError: measurement in measurements is not known on ODC
        :return: Depth data loaded from ODC
        :rtype: xarray.Dataset
        """
        logger.info(form.get_log_step('Obtaining depth data from ODC', 0))

        dc = datacube.Datacube()

        if product_name not in list(dc.list_products().index):
            raise ValueError(f"{product_name} is not known in the Open Data Cube instance")

        if measurements is None:
            measurements = list(dc.list_measurements().loc[product_name].index)
        else:
            # Check if requested measurements are available in ODC (measurements or aliases)
            measurements_odc = list(dc.list_measurements().loc[product_name].index)
            aliases_odc = [alias for aliases_per_var in list(dc.list_measurements().loc[product_name]['aliases']) for
                           alias in aliases_per_var]
            for measurement in measurements:
                if (measurement not in measurements_odc) and (measurement not in aliases_odc):
                    raise KeyError(f"{measurement} is not a valid measurement for odc product {product_name}")

        res_x = 30 / 3600  # 30 arc seconds to degrees
        res_y = 30 / 3600  # 30 arc seconds to degrees
        query = {'resolution': (res_x, res_y), 'align': (res_x / 2, res_y / 2),
                 'latitude': (self.map_size.lat1, self.map_size.lat2),
                 'longitude': (self.map_size.lon1, self.map_size.lon2), 'output_crs': 'EPSG:4326',
                 'measurements': measurements}
        ds_datacube = dc.load(product=product_name, **query).drop('time')
        if self._has_scaling(ds_datacube):
            ds_datacube = self._scale(ds_datacube)
        # Note: if depth_path already exists, the file will be overwritten!
        self._to_netcdf(ds_datacube, depth_path)
        return ds_datacube

    def load_data_automatic(self, depth_path):
        """
        Load data from NCEI

        :param depth_path: Path to where depth data should be written to
        :type depth_path: str
        :return: Depth data loaded from NCEI
        :rtype: xarray.Dataset
        """

        logger.info(form.get_log_step('Automatic download of depth data', 0))

        downloader = DownloaderFactory.get_downloader(downloader_type='xarray', platform='etoponcei')
        depth_data = downloader.download()
        depth_data_chunked = depth_data.chunk(chunks={"latitude": "100MB", "longitude": "100MB"})
        depth_data_chunked = depth_data_chunked.sel(latitude=slice(self.map_size.lat1, self.map_size.lat2),
                                                    longitude=slice(self.map_size.lon1, self.map_size.lon2))
        # Note: if depth_path already exists, the file will be overwritten!
        self._to_netcdf(depth_data_chunked, depth_path)
        return depth_data_chunked

    def load_data_from_file(self, depth_path):
        """
        Load depth data from given file

        :param depth_path: Path to the depth data
        :type depth_path: str
        :return: Depth data loaded from file
        :rtype: xarray.Dataset
        """

        # FIXME: if this loads the whole file into memory, apply subsetting already here
        logger.info(form.get_log_step('Downloading data from file: ' + depth_path, 0))
        ds_depth = None
        if graphics.get_figure_path():
            ds_depth = xr.open_dataset(depth_path, chunks={"time": "500MB"}, decode_times=False)
        else:
            ds_depth = xr.open_dataset(depth_path)
        return ds_depth

    def set_draught(self, depth):
        self.min_depth = depth

    def constraint_on_point(self, lat, lon, time):
        self.check_depth(lat, lon, time)
        return_value = self.current_depth > -self.min_depth
        # form.print_step('current_depth:' + str(self.current_depth), 1)
        return return_value

    def check_depth(self, lat, lon, time):
        lat_da = xr.DataArray(lat, dims="dummy")
        lon_da = xr.DataArray(lon, dims="dummy")
        rounded_ds = self.depth_data["z"].interp(latitude=lat_da, longitude=lon_da, method="linear")
        self.current_depth = rounded_ds.to_numpy()

    def print_info(self):
        logger.info(form.get_log_step("minimum water depth=" + str(self.min_depth) + "m", 1))

    def get_current_depth(self, lat, lon):
        self.check_depth(lat, lon, None)
        return self.current_depth

    def plot_depth_map_from_file(self, path):
        level_diff = 10

        ds_depth = xr.open_dataset(path)
        depth = ds_depth["z"].where((ds_depth.lat > self.map_size.lat1) & (ds_depth.lat < self.map_size.lat2) & (
                ds_depth.lon > self.map_size.lon1) & (ds_depth.lon < self.map_size.lon2) & (ds_depth.z < 0),
                                    drop=True, )

        # depth = ds_depth['deptho'].where((ds_depth.latitude > lat_start) & (ds_depth.latitude < lat_end) & (
        # ds_depth.longitude > lon_start) & (ds_depth.longitude < lon_end),drop=True) #.where((ds_depth.deptho>-100)
        # & (ds_depth.deptho<0) )

        fig, ax = plt.subplots(figsize=(12, 10))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        depth.plot.contourf(ax=ax,  # extend='max',
                            # levels=np.arange(-60, 0, 0.1),  high-resolution contours
                            levels=np.arange(-100, 0, level_diff), transform=ccrs.PlateCarree(), )

        fig.subplots_adjust(left=0.05, right=1, bottom=0.05, top=0.95, wspace=0, hspace=0)
        ax.add_feature(cf.LAND)
        ax.add_feature(cf.COASTLINE)
        ax.gridlines(draw_labels=True)

        plt.show()

    def plot_constraint(self, fig, ax):
        level_diff = 10
        plt.rcParams["font.size"] = 20
        ax.axis("off")
        ax.xaxis.set_tick_params(labelsize="large")

        ds_depth = self.depth_data.coarsen(latitude=10, longitude=10, boundary="exact").mean()
        ds_depth_coarsened = ds_depth.compute()

        self.depth_data = ds_depth_coarsened.where(
            (ds_depth_coarsened.latitude > self.map_size.lat1) & (ds_depth_coarsened.latitude < self.map_size.lat2) & (
                    ds_depth_coarsened.longitude > self.map_size.lon1) & (
                    ds_depth_coarsened.longitude < self.map_size.lon2) & (ds_depth_coarsened.z < 0), drop=True, )

        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cp = self.depth_data["z"].plot.contourf(ax=ax, levels=np.arange(-100, 0, level_diff),
                                                transform=ccrs.PlateCarree())
        fig.colorbar(cp, ax=ax, shrink=0.7, label="Wassertiefe (m)", pad=0.1)

        fig.subplots_adjust(left=0.1, right=1.2, bottom=0, top=1, wspace=0, hspace=0)
        ax.add_feature(cf.LAND)
        ax.add_feature(cf.COASTLINE)
        ax.gridlines(draw_labels=True)
        plt.title("")

        return fig, ax

    def _has_scaling(self, dataset):
        """
        Check if any of the included data variables has a scale_factor or add_offset

        :param dataset: Dataset that should be checked on scaling
        :type dataset: xarray.Dataset
        :return: If dataset has scaling
        :rtype: bool
        """

        for var in dataset.data_vars:
            if 'scale_factor' in dataset[var].attrs or 'add_offset' in dataset[var].attrs:
                return True
        return False

    def _scale(self, dataset):
        # FIXME: decode_cf also scales the nodata values, e.g. -32767 -> -327.67
        return xr.decode_cf(dataset)

    def _to_netcdf(self, dataset, file_out):
        """
        Customized method to fix 'AttributeError: NetCDF: String match to name in use' error
        References:
            - https://github.com/pydata/xarray/issues/2822
            - https://github.com/Unidata/netcdf4-python/issues/1020
        """

        if '_NCProperties' in dataset.attrs:
            del dataset.attrs['_NCProperties']
        dataset.to_netcdf(file_out)


class StayOnMap(NegativeContraint):
    """
    Constraint such that the boat can't leave the map that has weather data available
    """

    lat1: float
    lon1: float
    lat2: float
    lon2: float

    def __init__(self):
        NegativeContraint.__init__(self, "StayOnMap")
        self.message += "leaving wheather map!"  # self.resource_type = 0

    def constraint_on_point(self, lat, lon, time):
        # self.print_debug('checking point: ' + str(lat) + ',' + str(lon))
        is_on_map = ((lat > self.lat2) + (lat < self.lat1) + (lon > self.lon2) + (lon < self.lon1))
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
    Contains various functions to test data connection, gathering and use for obtaining spatial relations
    for the continuous check in the negative constraints
    """

    host: str
    database: str
    user: str
    password: str
    port: str  # returns values from .env  to be passed in the engine of the db

    predicates: list  # Possible spatial relations to be tested when considering the constraints

    tags: list  # Values of the seamark tags that need to be considered

    engine: sqlalchemy.engine

    def __init__(self, db_engine=None):
        NegativeContraint.__init__(self, "ContinuousChecks")
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

    def print_info(self):
        logger.info(form.get_log_step("no seamarks crossing", 1))

    def connect_database(self):
        """
        Connect to the database
        """

        # Connect to the PostgreSQL database using SQLAlchemy
        engine = sqlalchemy.create_engine(
            "postgresql://{user}:{pwd}@{host}:{port}/{db}".format(user=self.user, pwd=self.password, host=self.host,
                                                                  db=self.database, port=self.port))
        return engine

    def set_map_bbox(self, map_size):
        if map_size.lon1 <= map_size.lon2:
            min_lon = map_size.lon1
            max_lon = map_size.lon2
        else:
            min_lon = map_size.lon2
            max_lon = map_size.lon1

        if map_size.lat1 <= map_size.lat2:
            min_lat = map_size.lat1
            max_lat = map_size.lat2
        else:
            min_lat = map_size.lat2
            max_lat = map_size.lat1

        bbox = box(min_lon, min_lat, max_lon, max_lat)
        bbox_wkt = bbox.wkt
        logger.debug('BBox in WKT: ', bbox_wkt)
        return bbox_wkt


class RunTestContinuousChecks(ContinuousCheck):
    def __init__(self, test_dict):
        NegativeContraint.__init__(self, "ContinuousChecks")
        self.test_result_dict = test_dict

    def print_info(self):
        logger.info(form.get_log_step("adding test module for ContinuousChecks", 1))

    def connect_database(self):
        pass

    def check_crossing(self, lat_start, lon_start, lat_end, lon_end, time=None):
        result_length = len(lat_start)
        res = []

        for ires in range(0, result_length):
            res.append(self.test_result_dict[ires])

        return res


class SeamarkCrossing(ContinuousCheck):
    """
    Contains various functions to test data connection, gathering and use for obtaining spatial relations
    for the continuous check in the negative constraints
    """

    host: str
    database: str
    user: str
    password: str
    port: str  # returns values from .env  to be passed in the engine of the db

    predicates: list  # Possible spatial relations to be tested when considering the constraints

    tags: list  # Values of the seamark tags that need to be considered

    concat_tree: STRtree

    def __init__(self, is_stay_on_map=None, map_size=None, db_engine=None):
        super().__init__(db_engine=db_engine)

        if db_engine is None:
            seamark_query = self.build_seamark_query(is_stay_on_map, map_size)
            self.concat_tree = self.set_STRTree(db_engine=self.engine, query=seamark_query)

    def build_seamark_query(self, is_stay_on_map=None, map_size=None):
        """TODO: add description
        _summary_

        :param is_stay_on_map: _description_, defaults to None
        :type is_stay_on_map: _type_, optional
        :param map_size: _description_, defaults to None
        :type map_size: _type_, optional
        :return: _description_
        :rtype: _type_
        """

        tags = "'seamark:type'='restricted_area'"
        tags_category = ['safety', 'nature_reserve', 'bird_sanctuary',
                         'game_reserve', 'seal_sanctuary',
                         'degaussing_range',
                         'military', 'historic_wreck',
                         'navigational_aid_safety',
                         'minefield', 'swimming', 'waiting',
                         'research', 'dredging', 'fish_sanctuary',
                         'ecological_reserve', 'no_wake', 'swinging',
                         'water_skiing', 'essa', 'pssa']

        category_clause = " OR ".join([f"tags -> 'seamark:restricted_area:category'='{value}'" for value
                                       in tags_category])

        if is_stay_on_map:
            bbox_wkt = self.set_map_bbox(map_size)
            query = ["SELECT * FROM " + self.schema + ".nodes "
                     "WHERE ST_Intersects(geom, ST_GeomFromText('{}', 4326))".format(bbox_wkt)
                     + f" AND ({category_clause} OR tags -> " + tags + ")",
                     "SELECT *, linestring AS geom FROM " + self.schema + ".ways "
                     "WHERE ST_Intersects(linestring, ST_GeomFromText('{}', 4326))".format(bbox_wkt)
                     + f" AND ({category_clause} OR tags -> " + tags + ")"]
            logger.debug(query)
        else:
            query = ["SELECT * FROM " + self.schema + ".nodes "
                     f"WHERE ({category_clause} OR tags -> " + tags + ")",
                     "SELECT *, linestring AS geom FROM " + self.schema + ".ways "
                     f"WHERE ({category_clause} OR tags -> " + tags + ")"]
            logger.debug(query)
        return query

    def set_STRTree(self, db_engine=None, query=None):
        concat_gdf = self.concat_nodes_ways(db_engine, query)
        concat_tree = STRtree(concat_gdf["geom"])

        logger.debug(f'PRINT CONCAT DF {concat_gdf}')
        return concat_tree

    def query_nodes(self, db_engine, query=None):
        """
        Create new GeoDataFrame using public.nodes table in the query

        :param db_engine: sqlalchemy engine
        :type db_engine: sqlalchemy.engine.Engine
        :param query: sql query for table nodes, defaults to None
        :type query: str, optional
        :return: gdf including all the features from public.nodes table
        :rtype: GeoDataFrame
        """

        # Define SQL query to retrieve list of tables
        # sql_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        gdf = gpd.read_postgis(con=db_engine, sql=query, geom_col="geom", crs="epsg:4326")
        gdf = gdf[gdf["geom"] != None]
        return gdf

    def query_ways(self, db_engine, query):
        """
        Create new GeoDataFrame using public.nodes table in the query

        :param db_engine: sqlalchemy engine
        :type db_engine: sqlalchemy.engine.Engine
        :param query: sql query for table ways, defaults to None
        :type query: str, optional
        :return: gdf including all the features from public.nodes table
        :rtype: GeoDataFrame
        """

        gdf = gpd.read_postgis(con=db_engine, sql=query, geom_col="geom", crs="epsg:4326")
        gdf = gdf[gdf["geom"] != None]
        return gdf

    def concat_nodes_ways(self, db_engine, query):
        """
        Create new GeoDataFrame using public.ways and public.nodes table together in the query

        :param query: sql query for table ways
        :type query: list
        :param engine: sqlalchemy engine
        :type engine: sqlalchemy.engine.Engine
        :return: gdf including all the features from public.ways and public.nodes table
        :rtype: geopandas.GeoDataFrame
        """
        if "nodes" in query[0] and "ways" in query[1]:
            gdf_nodes = self.query_nodes(db_engine, query[0])
            gdf_ways = self.query_ways(db_engine, query[1])

            if (not gdf_nodes.empty) & (not gdf_ways.empty):
                gdf_nodes.reset_index(drop=True, inplace=True)
                gdf_ways.reset_index(drop=True, inplace=True)

                combined_gdf = pd.concat([gdf_nodes, gdf_ways], ignore_index=True)
                return combined_gdf
            elif (not gdf_nodes.empty) & gdf_ways.empty:
                return gdf_nodes
            else:
                return gdf_ways
        else:
            return "false query passed"

    def check_crossing(self, lat_start, lon_start, lat_end, lon_end):
        """
        Check if certain route crosses specified seamark objects

        :param lat_start: array of all origin latitudes of routing segments
        :type lat_start: numpy.ndarray
        :param lon_start: array of all origin longitudes of routing segments
        :type lon_start: numpy.ndarray
        :param lat_end: array of all destination latitudes of routing segments
        :type lat_end: numpy.ndarray
        :param lon_end: array of all destination longitudes of routing segments
        :type lon_end: numpy.ndarray
        :return: list of spatial relation result (True or False)
        :rtype: list[bool]
        """

        query_tree = []
        if self.concat_tree is not None:
            for i in range(len(lat_start)):
                start_point = Point(lon_start[i], lat_start[i])
                end_point = Point(lon_end[i], lat_end[i])
                line = LineString([start_point, end_point])
                route_df = gpd.GeoDataFrame(geometry=[line])

                geom_object = self.concat_tree.query(route_df["geometry"], predicate='intersects').tolist()

                if geom_object == [[], []] or geom_object == []:
                    # if route is not constrained
                    query_tree.append(False)
                    logger.debug(f'NO CROSSING for  {line} in the query tree: {query_tree} ')
                else:
                    # if route is constrained
                    query_tree.append(True)
                    logger.debug(f'CROSSING for  {line} in the query tree: {query_tree} ')

            # returns a list bools (spatial relation)
            return query_tree


class LandPolygonsCrossing(ContinuousCheck):
    """
    Use the 'LandPolygonsCrossing' constraint cautiously.
    This class is yet to be tested.
    """
    land_polygon_STRTree = None

    def __init__(self, map_size=None, db_engine=None):
        super().__init__(db_engine=db_engine)
        self.map_size = map_size

        if db_engine is None:
            landpolygon_query = self.build_landpolygon_query(map_size)
            self.land_polygon_STRTree = self.set_landpolygon_STRTree(self.engine, landpolygon_query)

    def build_landpolygon_query(self, map_size):
        bbox_wkt = self.set_map_bbox(map_size)
        query = "SELECT *,wkb_geometry as geom FROM public.land_polygons " \
                "WHERE ST_Intersects(wkb_geometry, ST_GeomFromText('{}', 4326))".\
                format(bbox_wkt)
        return query

    def set_landpolygon_STRTree(self, db_engine=None, query=None):
        land_polygon_gdf = self.query_land_polygons(db_engine, query)
        land_STRTree = STRtree(land_polygon_gdf["geom"])
        return land_STRTree

    def query_land_polygons(self, db_engine, query):
        """
        Create new GeoDataFrame using public.ways table in the query

        :param engine: sqlalchemy engine
        :type engine: sqlalchemy.engine.Engine
        :param query: sql query for table ways
        :type query: str
        :return: gdf including all the features from public.ways table
        :rtype: geopandas.GeoDataFrame
        """

        gdf = gpd.read_postgis(sql=query, con=db_engine, geom_col="geom")  # .drop(columns=["GEOMETRY"])
        gdf = gdf[gdf["geom"] != None]
        return gdf

    def check_crossing(self, lat_start, lon_start, lat_end, lon_end):
        """
        Check if certain route crosses specified seamark objects

        :param lat_start: array of all origin latitudes of routing segments
        :type lat_start: numpy.ndarray
        :param lon_start: array of all origin longitudes of routing segments
        :type lon_start: numpy.ndarray
        :param lat_end: array of all destination latitudes of routing segments
        :type lat_end: numpy.ndarray
        :param lon_end: array of all destination longitudes of routing segments
        :type lon_end: numpy.ndarray
        :return: list of spatial relation result (True or False)
        :rtype: list[bool]
        """

        query_tree = []
        if self.land_polygon_STRTree is not None:
            # generating the LineString geometry from start and end point
            for i in range(len(lat_start)):
                start_point = Point(lon_start[i], lat_start[i])
                end_point = Point(lon_end[i], lat_end[i])
                line = LineString([start_point, end_point])

                route_df = gpd.GeoDataFrame(geometry=[line])
                geom_object = self.land_polygon_STRTree.query(route_df["geometry"], predicate="intersects").tolist()

                # checks if there is spatial relation between routes and seamarks objects
                if geom_object == [[], []] or geom_object == []:
                    # if route is not constrained
                    query_tree.append(False)
                else:
                    # if route is constrained
                    query_tree.append(True)
            return query_tree
