import os

from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList, ConstraintPars, \
    SeamarkCrossing, LandPolygonsCrossing


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


def create_dummy_SeamarkCrossing_object(db_engine):
    seamark_obj = SeamarkCrossing(db_engine=db_engine)
    return seamark_obj


def create_dummy_landpolygonsCrossing_object(db_engine):
    landpolygoncrossing_obj = LandPolygonsCrossing(db_engine=db_engine)
    return landpolygoncrossing_obj
