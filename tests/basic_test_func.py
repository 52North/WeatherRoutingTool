import datetime
import os

import xarray
import pytest

from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import *


def generate_dummy_constraint_list():
    pars = ConstraintPars()
    pars.resolution = 1. / 10

    constraint_list = ConstraintsList(pars)
    return constraint_list


def create_dummy_IsoBased_object():

    print('Reading correct IsoBased XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx')

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
