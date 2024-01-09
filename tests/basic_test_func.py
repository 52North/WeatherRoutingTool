import os

from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList, ConstraintPars
from WeatherRoutingTool.ship.ship import Tanker


def generate_dummy_constraint_list():
    pars = ConstraintPars()
    pars.resolution = 1. / 10

    constraint_list = ConstraintsList(pars)
    return constraint_list


def create_dummy_IsoBased_object():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    config = Config(file_name=configpath)

    ra = IsoBased(config)
    return ra


def create_dummy_IsoFuel_object():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    config = Config(file_name=configpath)

    ra = IsoFuel(config)
    return ra


def create_dummy_Tanker_object():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    config = Config(file_name=configpath)

    dirname = os.path.dirname(__file__)
    config.WEATHER_DATA = os.path.join(dirname, 'data/reduced_testdata_weather.nc')
    config.COURSES_FILE = os.path.join(dirname, 'data/CoursesRoute.nc')
    config.DEPTH_FILE = os.path.join(dirname, 'data/reduced_testdata_depth.nc')

    pol = Tanker(config)
    return pol
