import json
import logging
import math
import os
import threading
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

# Specialized logger for patcher operations
logger = logging.getLogger("WRT.genetic.patcher")

class PatcherBase:
    """Base class for route patching."""
    def __init__(self, *args, **kwargs):
        pass

    def patch(self, src: tuple, dst: tuple):
        """Obtain waypoints between `src` and `dst`."""
        raise NotImplementedError("This patching method is not implemented.")

class SingletonBase(type):
    """
    A thread-safe and fork-safe Singleton metaclass.
    
    Uses the Double-Checked Locking pattern for efficiency and 
    'os.register_at_fork' to prevent deadlocks in multi-processing environments.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # First check (non-locked) for performance
        if cls not in cls._instances:
            with cls._lock:
                # Second check (locked) to prevent race conditions
                if cls not in cls._instances:
                    logger.debug(f"Initializing unique Singleton instance for {cls.__name__}")
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

# --- Fork-Safety Logic ---
def _reinitialize_singleton_lock():
    """Reset the lock after a process fork to avoid inherited deadlocks."""
    SingletonBase._lock = threading.Lock()
    logger.debug("SingletonBase lock re-initialized for child process.")

if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reinitialize_singleton_lock)

# --- Patcher Variants ---

class GreatCircleRoutePatcher(PatcherBase):
    """Produce waypoints along the Great Circle Route."""
    def __init__(self, dist: float = 10_000.0):
        super().__init__()
        self.dist = dist

    def patch(self, src: tuple, dst: tuple, departure_time: datetime = None, npoints=None) -> np.ndarray:
        geod: Geodesic = Geodesic.WGS84
        line = geod.InverseLine(*src, *dst)

        if npoints is not None:
            self.dist = line.s13 / npoints
        else:
            npoints = int(math.ceil(line.s13 / self.dist))

        route = []
        for i in range(npoints + 1):
            s = min(self.dist * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            route.append((g['lat2'], g['lon2']))

        return np.array([src, *route[1:-1], dst])

class IsofuelPatcher(PatcherBase):
    """Use the IsoFuel algorithm to produce a route between src and dst."""
    def __init__(self, base_config: Config, n_routes: str = "single") -> None:
        super().__init__()
        self.n_routes = n_routes
        self.patch_count = 0
        self.config = base_config
        
        self._setup_configuration()
        wt, boat, water_depth, constraints_list = self._setup_components()

        self.wt = wt
        self.boat = boat
        self.water_depth = water_depth
        self.constraints_list = constraints_list

        # Internal patcher for fallback
        self.patchfn_gcr = PatchFactory.get_patcher(
            patch_type="gcr_singleton",
            config=self.config,
            application="Isofuel patcher fallback"
        )

    def _setup_configuration(self) -> None:
        cfg_select = self.config.model_dump(
            include={
                "DEFAULT_ROUTE", "DEPARTURE_TIME", "DEFAULT_MAP",
                "COURSES_FILE", "DEPTH_DATA", "WEATHER_DATA",
                "ROUTE_PATH", "BOAT_UNDER_KEEL_CLEARANCE",
                "BOAT_DRAUGHT_AFT", "BOAT_DRAUGHT_FORE"
            }
        )

        cfg_dir = Path(os.path.dirname(__file__)) / "configs"
        cfg_path = cfg_dir / "config.isofuel_single_route.json"
        if self.n_routes == "multiple":
            cfg_path = cfg_dir / "config.isofuel_multiple_routes.json"

        with cfg_path.open() as fp:
            dt = json.load(fp)

        # Merge base config with patcher-specific JSON
        self.config = Config.model_validate({**dt, **cfg_select})
        self.config.CONFIG_PATH = str(cfg_path)
        
        # Load ship specific configurations
        ship_config_base = ShipConfig.assign_config(Path(self.config.CONFIG_PATH))
        self.config_boat_dict = ship_config_base.model_dump(
            include={"BOAT_UNDER_KEEL_CLEARANCE", "BOAT_DRAUGHT_AFT", "BOAT_DRAUGHT_FORE"}
        )

    def _setup_components(self) -> tuple[WeatherCond, Boat, WaterDepth, ConstraintsList]:
        cfg = self.config
        departure_time = cfg.DEPARTURE_TIME
        default_map = Map(*cfg.DEFAULT_MAP)

        wt = WeatherFactory.get_weather(
            cfg._DATA_MODE_WEATHER, cfg.WEATHER_DATA, departure_time,
            cfg.TIME_FORECAST, cfg.DELTA_TIME_FORECAST, default_map
        )

        boat = ShipFactory.get_ship(cfg)
        boat.under_keel_clearance = self.config_boat_dict["BOAT_UNDER_KEEL_CLEARANCE"] * u.meter
        boat.draught_aft = self.config_boat_dict['BOAT_DRAUGHT_AFT'] * u.meter
        boat.draught_fore = self.config_boat_dict['BOAT_DRAUGHT_FORE'] * u.meter

        water_depth = WaterDepth(
            cfg._DATA_MODE_DEPTH, boat.get_required_water_depth(),
            default_map, cfg.DEPTH_DATA
        )

        constraints_list = ConstraintsListFactory.get_constraints_list(
            constraints_string_list=cfg.CONSTRAINTS_LIST,
            data_mode=cfg._DATA_MODE_DEPTH,
            min_depth=boat.get_required_water_depth(),
            map_size=default_map,
            depthfile=cfg.DEPTH_DATA,
            waypoints=cfg.INTERMEDIATE_WAYPOINTS,
            courses_path=cfg.COURSES_FILE
        )

        return wt, boat, water_depth, constraints_list

    def patch(self, src, dst, departure_time: datetime = None):
        self.patch_count += 1
        cfg = self.config.model_copy(update={
            "DEFAULT_ROUTE": [*src, *dst],
            "DEPARTURE_TIME": departure_time
        })

        # Temporary silencing of non-critical logs during execution
        original_log_level = logging.getLogger().level
        if original_log_level > logging.DEBUG:
            logging.getLogger().setLevel(logging.ERROR)

        alg = IsoFuel(cfg)
        alg.path_to_route_folder = None
        alg.clear_figure_path()
        
        if original_log_level == logging.DEBUG:
            alg.init_fig(water_depth=self.water_depth, map_size=Map(*self.config.DEFAULT_MAP))

        min_fuel_route, err_code = alg.execute_routing(
            boat=self.boat, wt=self.wt,
            constraints_list=self.constraints_list,
            patch_count=self.patch_count
        )

        logging.getLogger().setLevel(original_log_level)

        if err_code > 0:
            logger.debug('Isofuel failed; falling back to GCR.')
            return self.patchfn_gcr.patch(src, dst, departure_time)

        if self.n_routes == "single":
            return np.stack([min_fuel_route.lats_per_step, min_fuel_route.lons_per_step], axis=1)

        if not alg.route_list:
            raise RuntimeError("Isofuel algorithm could not find any routes.")

        return [np.stack([rt.lats_per_step, rt.lons_per_step], axis=1) for rt in alg.route_list]

# --- Singleton Implementations ---

class GreatCircleRoutePatcherSingleton(GreatCircleRoutePatcher, metaclass=SingletonBase):
    pass

class IsofuelPatcherSingleton(IsofuelPatcher, metaclass=SingletonBase):
    pass

# --- Factory ---

class PatchFactory:
    @staticmethod
    def get_patcher(patch_type: str, application: str = 'undefined', config: Config = None) -> PatcherBase:
        logger.debug(f'Requesting patcher "{patch_type}" for {application}.')
        
        if patch_type == "gcr_singleton":
            return GreatCircleRoutePatcherSingleton()
        if patch_type == "isofuel_singleton":
            return IsofuelPatcherSingleton(base_config=config)
        if patch_type == "isofuel_multiple_routes":
            return IsofuelPatcher(base_config=config, n_routes="multiple")
        if patch_type == "gcr":
            return GreatCircleRoutePatcher()

        raise NotImplementedError(f'Patch type "{patch_type}" is not supported.')