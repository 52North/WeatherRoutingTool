from typing import Union

from geographiclib.geodesic import Geodesic
import numpy as np

from pathlib import Path
from datetime import datetime
import functools
import math
import os

from WeatherRoutingTool.weather import WeatherCond
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.constraints.constraints import ConstraintsList

from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory, WaterDepth


# base class
# ----------
class PatcherBase:
    def patch(self, src, dst):
        """Patch a route between `src` and `dst`

        :param src: Source coords as (lat, lon)
        :type src: tuple[float, float]
        :param dst: Destination coords as (lat, lon)
        :type dst: tuple[float, float]
        """

        pass


# patcher variants
# ----------
class GreatCircleRoutePatcher(PatcherBase):
    """Produce a set of waypoints along the Great Circle Route between src and dst

    :param dist: Dist between each waypoint in the Great Circle Route
    :type dist: float
    """

    def __init__(self, dist: float = 100_000.0):
        super().__init__()

        # variables
        self.dist = dist

    def patch(self, src, dst, departure_time: datetime = None):
        """Generate equi-distant waypoints across the Great Circle Route from src to
        dst

        :param src: Source waypoint as (lat, lon) pair
        :type src: tuple[float, float]
        :param dst: Destination waypoint as (lat, lon) pair
        :type dst: tuple[float, float]
        :param distance: Distance between waypoints generated
        :type distance: float
        :return: List of waypoints along the great circle (lat, lon)
        :rtype: list[tuple[float, float]]
        """

        geod: Geodesic = Geodesic.WGS84
        line = geod.InverseLine(*src, *dst)
        n = int(math.ceil(line.s13 / self.dist))
        route = []
        for i in range(n + 1):
            s = min(self.dist * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            route.append((g['lat2'], g['lon2']))
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

    def _setup_components(self, config):
        """
        Execute route optimization based on the user-defined configuration.
        After a successful run the final route is saved into the configured folder.

        :param config: validated configuration
        :type config: WeatherRoutingTool.config.Config
        :return: None
        """
        # prof = cProfile.Profile()
        # prof.enable()

        # *******************************************
        # basic settings
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
        # boat = ConstantFuelBoat(init_mode="from_dict", config_dict=config.model_dump())

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

    def __init__(self, config: Config, n_routes: str = "single"):
        super().__init__()

        # variables
        self.n_routes = n_routes
        self.config = config

        # setup components
        wt, boat, water_depth, constraints_list = self._setup_components(self.config)

        self.wt: WeatherCond = wt
        self.boat: Boat = boat
        self.water_depth: WaterDepth = water_depth
        self.constraints_list: ConstraintsList = constraints_list

    def patch(self, src, dst, departure_time: datetime = None):
        cfg = self.config.model_copy(update={
            "DEFAULT_ROUTE": [*src, *dst],
            "DEPARTURE_TIME": departure_time
        })

        # alg is defined in method because route_list is defined per instance,
        # which wouldn't work well when we want to "generate" multiple times
        alg = IsoFuel(cfg)
        alg.path_to_route_folder = None

        min_fuel_route, err_code = alg.execute_routing(
            boat=self.boat,
            wt=self.wt,
            constraints_list=self.constraints_list, )

        # single route
        if self.n_routes == "single":
            return np.stack([min_fuel_route.lats_per_step, min_fuel_route.lons_per_step], axis=1)

        # list of routes
        if not alg.route_list:
            raise RuntimeError("The Isofuel algorithm couldn't find any route")

        routes = []

        for rt in alg.route_list:
            routes.append(np.stack([rt.lats_per_step, rt.lons_per_step], axis=1))
        return routes

    @classmethod
    def for_multiple_routes(cls, default_map, **kw):
        """Class constructor for multiple routes generation"""

        cfg = Config.assign_config(
            path=Path(os.path.dirname(__file__)) / "configs" / "config.isofuel_multiple_routes.json")

        cfg.DEFAULT_MAP = default_map

        for k, v in kw.items():
            setattr(cfg, k, v)

        return cls(cfg, n_routes="multiple")

    @classmethod
    def for_single_route(cls, default_map, **kw):
        """Class constructor for single route generation"""

        cfg = Config.assign_config(
            path=Path(os.path.dirname(__file__)) / "configs" / "config.isofuel_single_route.json")

        cfg.DEFAULT_MAP = default_map

        for k, v in kw.items():
            setattr(cfg, k, v)

        return cls(cfg, n_routes="single")
