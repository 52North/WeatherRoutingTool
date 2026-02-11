import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from geographiclib.geodesic import Geodesic

from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList, ConstraintsListFactory, WaterDepth
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond
from WeatherRoutingTool.weather_factory import WeatherFactory

logger = logging.getLogger("WRT.genetic.patcher")


class PatcherBase:
    """Base class for route patching"""

    def __init__(self, *args, **kwargs):
        pass

    def patch(self, src: tuple, dst: tuple, departure_time: datetime = None):
        """Obtain waypoints between `src` and `dst`.

        :param src: Source coords as (lat, lon)
        :type src: tuple[float, float]
        :param dst: Destination coords as (lat, lon)
        :type dst: tuple[float, float]
        :param departure_time: Departure time
        :type departure_time: datetime
        """
        raise NotImplementedError("This patching method is not implemented.")


class SingletonBase(type):
    """
    TODO: make this thread-safe
    Base class for Singleton implementation of patcher methods.

    This is the implementation of a metaclass for those classes for which only a single instance shall be available
    during runtime.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


# patcher variants
# ----------
class GreatCircleRoutePatcher(PatcherBase):
    """Produce a set of waypoints along the Great Circle Route between src and dst.

    The same speed as the speed at `src` is added to every waypoint.

    :param dist: Dist between each waypoint in the Great Circle Route
    :type dist: float
    """
    dist: float

    def __init__(self, dist: float = 10_000.0):
        super().__init__()

        # variables
        self.dist = dist

    def patch(self,
              src: tuple[float, float, float],
              dst: tuple[float, float, float],
              departure_time: datetime = None,
              npoints=None,
              ) -> np.ndarray:
        """Generate equi-distant waypoints across the Great Circle Route from src to
        dst

        :param src: Source waypoint as (lat, lon, v) triple
        :type src: tuple[float, float, float]
        :param dst: Destination waypoint as (lat, lon, v) triple
        :type dst: tuple[float, float, float]
        :return: List of waypoints along the great circle (lat, lon, v)
        :rtype: np.array[tuple[float, float, float]]
        """

        geod: Geodesic = Geodesic.WGS84
        line = geod.InverseLine(*src[:-1], *dst[:-1])
        speed = src[2]

        if not npoints == None:
            self.dist = line.s13 / npoints
        else:
            npoints = int(math.ceil(line.s13 / self.dist))

        route = []
        for i in range(npoints + 1):
            s = min(self.dist * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            route.append((g['lat2'], g['lon2'], speed))

        return np.array([src, *route[1:-1], dst])


class IsofuelPatcher(PatcherBase):
    """Use the IsoFuel algorithm to produce a route between src and dst.

    Intuition behind having this as a class:
    1. The Isofuel path finding component can be quite expensive during the
        preparation stage (defining map, loading data, etc.). Having setup and
        execution as separate components could help speed things up.

    :param config: Configuration for the run
    :type config: Config
    :param n_routes: Type of response expected. Either "single" or "multiple"
    :type n_routes: str
    """

    n_routes: str
    patch_count: int
    config: Config
    config_boat_dict: dict

    wt: WeatherCond
    boat: Boat
    water_depth: WaterDepth
    constraints_list: ConstraintsList

    def __init__(self, base_config: Config, n_routes: str = "single") -> None:
        super().__init__()

        # variables
        self.n_routes = n_routes
        self.patch_count = 0

        # setup components
        self.config = base_config
        self._setup_configuration()
        wt, boat, water_depth, constraints_list = self._setup_components()

        self.wt: WeatherCond = wt
        self.boat: Boat = boat
        self.water_depth: WaterDepth = water_depth
        self.constraints_list: ConstraintsList = constraints_list

        self.patchfn_gcr = PatchFactory.get_patcher(
            patch_type="gcr_singleton",
            config=self.config,
            application="Isofuel patcher"
        )

    def _setup_configuration(self) -> Config:
        """ Setup configuration for generation of a single or multiple routes with the IsofuelPatcher.

        The configuration is based on the general config file. Based on n_routes, this configuration is overwritten
        by the configuration files for the IsofuelPatcher for single and multiple routes.


        :return: config object
        :rtype: Config
        """
        cfg_select = self.config.model_dump(
            include=[
                "DEFAULT_ROUTE",
                "DEPARTURE_TIME",
                "DEFAULT_MAP",

                "COURSES_FILE",
                "DEPTH_DATA",
                "WEATHER_DATA",
                "ROUTE_PATH",
                "BOAT_UNDER_KEEL_CLEARANCE",
                "BOAT_DRAUGHT_AFT",
                "BOAT_DRAUGHT_FORE"
            ], )

        cfg_path = Path(os.path.dirname(__file__)) / "configs" / "config.isofuel_single_route.json"
        if self.n_routes == "multiple":
            cfg_path = Path(os.path.dirname(__file__)) / "configs" / "config.isofuel_multiple_routes.json"

        with cfg_path.open() as fp:
            dt = json.load(fp)

        cfg = Config.model_validate({**dt, **cfg_select})

        # combine patcher configuration and ship parameters from base configuration relevant for constraints
        ship_config_base = ShipConfig.assign_config(Path(self.config.CONFIG_PATH))
        cfg_ship_base = ship_config_base.model_dump(
            include=[
                "BOAT_UNDER_KEEL_CLEARANCE",
                "BOAT_DRAUGHT_AFT",
                "BOAT_DRAUGHT_FORE"
            ],
        )
        self.config_boat_dict = cfg_ship_base

        # set config path to patcher configuration
        cfg.CONFIG_PATH = cfg_path
        self.config = cfg
        print('self.config: ', cfg)
        return

    def _setup_components(self) -> tuple[WeatherCond, Boat, WaterDepth, ConstraintsList]:
        """
        Initialise the modules for weather conditions, fuel consumption and constraints that are necessary for the
        execution of the Isofuel algorithm.

        :return: modules for weather conditions, fuel consumption & constraints
        :rtype: tuple(WeatherCond, Boat, WaterDepth, ConstraintsList)
        """

        # *******************************************
        # basic settings
        config = self.config
        windfile = config.WEATHER_DATA
        depthfile = config.DEPTH_DATA
        time_resolution = config.DELTA_TIME_FORECAST
        time_forecast = config.TIME_FORECAST
        lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
        departure_time = config.DEPARTURE_TIME
        default_map = Map(lat1, lon1, lat2, lon2)

        # *******************************************
        # initialise weather
        wt = WeatherFactory.get_weather(
            config._DATA_MODE_WEATHER,
            windfile,
            departure_time,
            time_forecast,
            time_resolution,
            default_map, )

        # *******************************************
        # initialise boat
        boat = ShipFactory.get_ship(config)
        boat.under_keel_clearance = self.config_boat_dict["BOAT_UNDER_KEEL_CLEARANCE"] * u.meter
        boat.draught_aft = self.config_boat_dict['BOAT_DRAUGHT_AFT'] * u.meter
        boat.draught_fore = self.config_boat_dict['BOAT_DRAUGHT_FORE'] * u.meter

        # *******************************************
        # initialise constraints
        water_depth = WaterDepth(
            config._DATA_MODE_DEPTH,
            boat.get_required_water_depth(),
            default_map,
            depthfile, )

        constraints_list = ConstraintsListFactory.get_constraints_list(
            constraints_string_list=config.CONSTRAINTS_LIST,
            data_mode=config._DATA_MODE_DEPTH,
            min_depth=boat.get_required_water_depth(),
            map_size=default_map,
            depthfile=depthfile,
            waypoints=config.INTERMEDIATE_WAYPOINTS,
            courses_path=config.COURSES_FILE, )

        return wt, boat, water_depth, constraints_list

    def patch(self,
              src: tuple[float, float, float],
              dst: tuple[float, float, float],
              departure_time: datetime = None
              ):
        """
        Produce a set of waypoints between src and dst using the IsoFuel algorithm.

        The same speed as the speed at `src` is added to every waypoint.

        :param src: Source waypoint as (lat, lon, speed) triple
        :type src: tuple[float, float, float]
        :param dst: Destination waypoint as (lat, lon, speed) triple
        :type dst: tuple[float, float, float]
        :param departure_time: departure time from src
        :type departure_time: datetime
        :return: List of waypoints or list of multiple routes connecting src and dst
        :rtype: np.array[tuple[float, float]] or list[np.array[tuple[float, float]]]
        """
        self.patch_count += 1

        cfg = self.config.model_copy(update={
            "DEFAULT_ROUTE": [*src[:-1], *dst[:-1]],
            "DEPARTURE_TIME": departure_time
        })

        # make Isofuel algorithm run in quite mode
        original_log_level = logging.getLogger().level
        if original_log_level > logging.DEBUG:
            logging.getLogger().setLevel(logging.ERROR)

        # alg is defined in method because route_list is defined per instance,
        # which wouldn't work well when we want to "generate" multiple times
        alg = IsoFuel(cfg)
        alg.path_to_route_folder = None
        alg.clear_figure_path()
        if original_log_level == logging.DEBUG:
            alg.init_fig(water_depth=self.water_depth, map_size=Map(*self.config.DEFAULT_MAP))

        min_fuel_route, err_code = alg.execute_routing(
            boat=self.boat,
            wt=self.wt,
            constraints_list=self.constraints_list,
            patch_count=self.patch_count)

        # reactivate original logging level
        logging.getLogger().setLevel(original_log_level)

        # fall-back to gcr patching if Isofuel algorithm can not provide valid results
        if err_code > 0:
            logger.debug('Falling back to gcr patching!')
            return self.patchfn_gcr.patch(src, dst, departure_time)

        speed = np.full(min_fuel_route.lons_per_step.shape, src[2])

        # single route
        if self.n_routes == "single":
            return np.stack([min_fuel_route.lats_per_step, min_fuel_route.lons_per_step, speed], axis=1)

        # list of routes
        if not alg.route_list:
            raise RuntimeError("The Isofuel algorithm couldn't find any route")

        routes = []

        for rt in alg.route_list:
            routes.append(np.stack([rt.lats_per_step, rt.lons_per_step, speed], axis=1))
        return routes


class GreatCircleRoutePatcherSingleton(GreatCircleRoutePatcher, metaclass=SingletonBase):
    """Implementation class for GreatCircleRoutePatcher that allows only a single instance."""

    def __init__(self, dist: float = 10_000.0):
        super().__init__(dist)


class IsofuelPatcherSingleton(IsofuelPatcher, metaclass=SingletonBase):
    """Implementation class for IsofuelPatcher that allows only a single instance."""

    def __init__(self, base_config, n_routes: str = "single"):
        super().__init__(base_config, n_routes)


# factory
# ----------
class PatchFactory:
    @staticmethod
    def get_patcher(
            patch_type: str,
            application: str = 'application undefined',
            config: Config = None
    ) -> PatcherBase:

        if patch_type == "gcr_singleton":
            logger.debug(f'Setting patch type of genetic algorithm for {application} to "gcr_singleton".')
            return GreatCircleRoutePatcherSingleton()

        if patch_type == "isofuel_singleton":
            logger.debug(f'Setting patch type of genetic algorithm for {application} to "isofuel_singleton".')
            return IsofuelPatcherSingleton(base_config=config)

        if patch_type == "isofuel_multiple_routes":
            logger.debug(f'Setting patch type of genetic algorithm for {application} to "isofuel_multiple_routes".')
            return IsofuelPatcher(base_config=config, n_routes="multiple")

        if patch_type == "gcr":
            logger.debug(f'Setting patch type of genetic algorithm for {application} to "gcr".')
            return GreatCircleRoutePatcher()

        raise NotImplementedError(f'The patch type {patch_type} is not implemented.')
