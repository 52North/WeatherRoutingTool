from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem


DEPARTURE_TIME = datetime(2025, 4, 1, 11, 11)
ARRIVAL_TIME = DEPARTURE_TIME + timedelta(hours=10)
BOAT_SPEED = 6 * u.meter / u.second
BOAT_SPEED_UNSET = -99. * u.meter / u.second


def make_route():
    lats = np.array([38.192, 38.8, 39.4, 40.0, 40.6, 41.349])
    lons = np.array([13.392, 11.5, 9.6, 7.7, 5.0, 2.188])
    speeds = np.full(len(lats), BOAT_SPEED.value)
    return np.column_stack([lats, lons, speeds])


@pytest.fixture(scope="module")
def boat():
    return basic_test_func.create_dummy_Direct_Power_Ship("simpleship")


@pytest.fixture(scope="module")
def constraint_list():
    return basic_test_func.generate_dummy_constraint_list()


@pytest.fixture(scope="module")
def problem_fixed_speed(boat, constraint_list):
    return RoutingProblem(
        departure_time=DEPARTURE_TIME,
        arrival_time=ARRIVAL_TIME,
        boat=boat,
        boat_speed=BOAT_SPEED,
        constraint_list=constraint_list,
    )


@pytest.fixture(scope="module")
def problem_arrival_time(boat, constraint_list):
    return RoutingProblem(
        departure_time=DEPARTURE_TIME,
        arrival_time=ARRIVAL_TIME,
        boat=boat,
        boat_speed=BOAT_SPEED_UNSET,
        constraint_list=constraint_list,
    )


class TestRoutingProblemInit:
    def test_fixed_speed_flag_false(self, problem_fixed_speed):
        assert problem_fixed_speed.boat_speed_from_arrival_time is False

    def test_arrival_time_flag_true(self, problem_arrival_time):
        assert problem_arrival_time.boat_speed_from_arrival_time is True


class TestGetPowerFixedSpeed:
    def test_returns_tuple(self, problem_fixed_speed):
        route = make_route()
        result = problem_fixed_speed.get_power(route)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fuel_is_scalar(self, problem_fixed_speed):
        route = make_route()
        fuel, _ = problem_fixed_speed.get_power(route)
        assert np.isscalar(fuel) or fuel.ndim == 0

    def test_fuel_is_positive(self, problem_fixed_speed):
        route = make_route()
        fuel, _ = problem_fixed_speed.get_power(route)
        assert fuel.value > 0

    def test_shipparams_returned(self, problem_fixed_speed):
        route = make_route()
        _, shipparams = problem_fixed_speed.get_power(route)
        assert shipparams is not None

    def test_fuel_reproducible(self, problem_fixed_speed):
        route = make_route()
        fuel1, _ = problem_fixed_speed.get_power(route)
        fuel2, _ = problem_fixed_speed.get_power(route)
        assert np.isclose(fuel1.value, fuel2.value)

    def test_longer_route_more_fuel(self, problem_fixed_speed):
        short_route = make_route()[:3]
        short_speeds = np.full(3, BOAT_SPEED.value)
        short_route = np.column_stack([short_route[:, 0], short_route[:, 1], short_speeds])

        full_fuel, _ = problem_fixed_speed.get_power(make_route())
        short_fuel, _ = problem_fixed_speed.get_power(short_route)
        assert full_fuel.value > short_fuel.value


class TestGetPowerArrivalTime:
    def test_returns_tuple(self, problem_arrival_time):
        route = make_route()
        result = problem_arrival_time.get_power(route)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fuel_is_positive(self, problem_arrival_time):
        route = make_route()
        fuel, _ = problem_arrival_time.get_power(route)
        assert fuel.value > 0


class TestEvaluate:
    def _run_evaluate(self, problem, fuel_value=5000.0, constraint_value=0):
        route = make_route()
        x = np.array([route], dtype=object)
        out = {}
        with patch.object(problem, 'get_power', return_value=(fuel_value, MagicMock())), \
             patch('WeatherRoutingTool.algorithms.genetic.problem.get_constraints', return_value=constraint_value):
            problem._evaluate(x, out)
        return out

    def test_out_F_and_G_populated(self, problem_fixed_speed):
        out = self._run_evaluate(problem_fixed_speed)
        assert 'F' in out
        assert 'G' in out

    def test_out_F_is_positive(self, problem_fixed_speed):
        out = self._run_evaluate(problem_fixed_speed, fuel_value=5000.0)
        assert out['F'].flat[0] > 0

    def test_out_G_is_zero_when_no_constraints(self, problem_fixed_speed):
        out = self._run_evaluate(problem_fixed_speed, constraint_value=0)
        assert out['G'].flat[0] == 0
