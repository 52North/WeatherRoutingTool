import copy
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from WeatherRoutingTool.algorithms.genetic.patcher import PatcherBase, GreatCircleRoutePatcher, IsofuelPatcher, \
    GreatCircleRoutePatcherSingleton, IsofuelPatcherSingleton
from WeatherRoutingTool.algorithms.genetic.mutation import RandomWalkMutation
import  WeatherRoutingTool.utils.graphics as graphics
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

'''
   sanity test for output for genetic.mutation.RandomPlateauMutation.mutate():
   - does the shape of the output route matrix resemble the shape of the input route matrix
   - do the starting and end points of all routes match with the input routes
'''
def test_random_walk_mutation():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    np.random.seed(1)

    mt = RandomWalkMutation(config, )

    route1 = np.array([
        [35.199, 15.490],
        [34.804, 16.759],
        [34.447, 18.381],
        [34.142, 18.763],
        [33.942, 21.080],
        [33.542, 23.024],
        [33.408, 24.389],
        [33.166, 26.300],
        [32.937, 27.859],
        [32.737, 28.859],
    ])
    route2 = np.array([
        [35.199, 16.490],
        [34.804, 17.759],
        [34.447, 19.381],
        [34.142, 19.763],
        [33.942, 22.080],
        [33.542, 23.024],
        [33.408, 24.389],
        [33.166, 25.300],
        [32.937, 26.859],
        [32.737, 27.859],
    ])
    X=np.array([[route1],[route2]])

    old_route = copy.deepcopy(X)
    new_route = mt.mutate(None, X, )

    # plot figure with original and mutated routes
    fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
    fig, ax = graphics.generate_basemap(
        fig = fig,
        depth = None,
        start=(35.199, 15.490),
        finish=(32.937, 27.859),
        title='',
        show_depth=False,
        show_gcr=False
    )
    ax.plot(old_route[0, 0] [:, 1], old_route[0, 0][:, 0], color="firebrick")
    ax.plot(new_route[0, 0] [:, 1], new_route[0, 0][:, 0], color="blue")
    ax.plot(old_route[1, 0] [:, 1], old_route[1, 0][:, 0], color="firebrick")
    ax.plot(new_route[1, 0] [:, 1], new_route[1, 0][:, 0], color="blue")

    assert old_route.shape == new_route.shape
    for i_route in range(old_route.shape[0]):
        assert np.array_equal(old_route[i_route,0][-1,:], new_route[i_route,0][-1,:])
        assert np.array_equal(old_route[i_route, 0][0, :], new_route[i_route, 0][0, :])

'''
    test whether routes are returned as they are if they are too short
'''
def test_random_walk_mutation_refusal():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    np.random.seed(1)

    mt = RandomWalkMutation(config, )

    route1 = np.array([
        [35.199, 15.490],
        [34.804, 16.759],
        [34.447, 18.381],
        [34.142, 18.763],
        [33.942, 21.080],
        [33.542, 23.024],
        [33.408, 24.389],
        [33.166, 26.300],
    ])
    route2 = np.array([
        [35.199, 16.490],
        [34.804, 17.759],
        [34.447, 19.381],
        [34.142, 19.763],
        [33.942, 22.080],
        [33.542, 23.024],
        [33.408, 24.389],
        [33.166, 25.300],
    ])
    X = np.array([[route1], [route2]])

    old_route = copy.deepcopy(X)
    new_route = mt.mutate(None, X, )

    assert np.array_equal(old_route, new_route)
