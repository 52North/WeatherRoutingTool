import os

from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList, ConstraintPars, \
    SeamarkCrossing, LandPolygonsCrossing
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship_config import ShipConfig

try:
    import mariPower
    from WeatherRoutingTool.ship.maripower_tanker import MariPowerTanker
except ModuleNotFoundError:
    pass  # maripower installation is optional


def generate_dummy_constraint_list():
    pars = ConstraintPars()
    pars.resolution = 1. / 10

    constraint_list = ConstraintsList(pars)
    return constraint_list


def create_dummy_IsoBased_object():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    config = Config.assign_config(configpath)

    ra = IsoBased(config)
    return ra


def create_dummy_IsoFuel_object():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    config = Config.assign_config(configpath)

    ra = IsoFuel(config)
    return ra


def create_dummy_SeamarkCrossing_object(db_engine):
    seamark_obj = SeamarkCrossing(db_engine=db_engine)
    return seamark_obj


def create_dummy_landpolygonsCrossing_object(db_engine):
    landpolygoncrossing_obj = LandPolygonsCrossing(db_engine=db_engine)
    return landpolygoncrossing_obj


def create_dummy_Tanker_object():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')

    ship_config = ShipConfig.assign_config(path=configpath)
    pol = MariPowerTanker(ship_config)
    pol.weather_path = os.path.join(dirname, 'data/tests_weather_data.nc')
    pol.courses_path = os.path.join(dirname, 'data/CoursesRoute.nc')
    pol.use_depth_data = True
    pol.depth_path = os.path.join(dirname, 'data/tests_depth_data.nc')
    pol.load_data()
    return pol


def create_dummy_Direct_Power_Ship(ship_config_path):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests_' + ship_config_path + '.json')
    dirname = os.path.dirname(__file__)

    ship_config = ShipConfig.assign_config(path=configpath)
    pol = DirectPowerBoat(ship_config)
    pol.weather_path = os.path.join(dirname, 'data/tests_weather_data.nc')
    pol.courses_path = os.path.join(dirname, 'data/CoursesRoute.nc')
    pol.depth_path = os.path.join(dirname, 'data/tests_depth_data.nc')
    pol.load_data()
    return pol
