import copy
import os

import matplotlib.pyplot as plt
import pytest

from WeatherRoutingTool.algorithms.genetic_utils import GridBasedMutation, GeneticCrossover
from WeatherRoutingTool.utils import graphics


def get_xy_from_tuples(route):
    x = [x[0] for x in route]
    y = [x[1] for x in route]
    return x, y


@pytest.mark.parametrize(
    ("route"),
    [
        [(35.3, 16.1), (35.17, 17.2), (35.1, 18.2), (35.0, 19.1), (35.0, 20.3)]
    ],
)
@pytest.mark.genetic
class TestGenetic:

    @pytest.mark.manual
    @pytest.mark.skip(reason="GridBasedMutation needs a grid parameter that we can't easily create in the test")
    def test_mutate_move(self, route):
        orig_route = route.copy()
        plt.rcParams['text.usetex'] = False
        
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        try:
            fig, ax = graphics.generate_basemap(fig, None, (35.3, 16.1), (35.0, 20.3), '', False)
        except Exception as e:
            print(f"Basemap generation failed: {e}")
            ax = fig.add_subplot(111)

        x_route, y_route = get_xy_from_tuples(route)
        x_origroute, y_origroute = get_xy_from_tuples(orig_route)

        ax.plot(y_route, x_route, color="blue")
        ax.plot(y_origroute, x_origroute, color="orange")

        from WeatherRoutingTool.utils.graphics import get_figure_path
        figure_path = get_figure_path() or os.getcwd()
        output_file = os.path.join(figure_path, "test_mutate_move.png")
        plt.savefig(output_file)
        plt.close()

    @pytest.mark.skip(reason="GA needs to be reviewed.")
    @pytest.mark.manual
    def test_crossover_noint(self, route):
        route1 = [(35.3, 16.1), (36.17, 18.2), (36.1, 19.2), (36.0, 20.1), (35.0, 20.3)]
        route2 = route

        cross = GeneticCrossover()
        child1, child2 = cross.crossover_noint(route1, route2)

        plt.rcParams['text.usetex'] = False
        
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        try:
            fig, ax = graphics.generate_basemap(fig, None, (35.3, 16.1), (35.0, 20.3), '', False)
        except Exception as e:
            print(f"Basemap generation failed: {e}")
            ax = fig.add_subplot(111)

        x_route1, y_route1 = get_xy_from_tuples(route1)
        x_route2, y_route2 = get_xy_from_tuples(route2)
        x_child1, y_child1 = get_xy_from_tuples(child1)
        x_child2, y_child2 = get_xy_from_tuples(child2)

        ax.plot(y_route1, x_route1, color="blue", label='parent 1', linewidth=3)
        ax.plot(y_route2, x_route2, color="orange", label='parent 2', linewidth=3)
        ax.plot(y_child1, x_child1, color="red", label='child 1', linestyle='dashed', linewidth=3)
        ax.plot(y_child2, x_child2, color="green", label='child 2', linestyle='dashed', linewidth=3)

        ax.legend()
        figure_path = graphics.get_figure_path() or os.getcwd()
        output_file = os.path.join(figure_path, "test_crossover_noint.png")
        plt.savefig(output_file)
        plt.close()
