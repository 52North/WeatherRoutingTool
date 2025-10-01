import os
from datetime import datetime
from pathlib import Path

from WeatherRoutingTool.algorithms.genetic.patcher import PatcherBase, GreatCircleRoutePatcher, IsofuelPatcher, \
    GreatCircleRoutePatcherSingleton, IsofuelPatcherSingleton
from WeatherRoutingTool.config import Config


def test_isofuelpatcher_singleton():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    src = [38.851, 4.066]
    dst = [37.901, 8.348]

    departure_time = datetime(2025, 4, 1, 12, 11)
    pt_one = IsofuelPatcherSingleton(config)
    pt_two = IsofuelPatcherSingleton(config)

    pt_one.patch(src, dst, departure_time)

    assert id(pt_two) == id(pt_one)


def test_isofuelpatcher_no_singleton():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    src = [38.851, 4.066]
    dst = [37.901, 8.348]

    departure_time = datetime(2025, 4, 1, 12, 11)
    pt_one = IsofuelPatcher(config)
    pt_two = IsofuelPatcher(config)

    pt_one.patch(src, dst, departure_time)

    assert id(pt_two) != id(pt_one)
