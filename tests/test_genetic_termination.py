import pytest

from WeatherRoutingTool.algorithms.genetic import Genetic


class _EmptyResult:
    F = None
    X = None


def test_genetic_terminate_raises_on_empty_result():
    genetic = Genetic.__new__(Genetic)

    with pytest.raises(RuntimeError, match=r"no feasible solution"):
        genetic.terminate(_EmptyResult(), problem=None)
