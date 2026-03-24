"""
Unit tests for RoutingProblem.get_power

Tests verify that get_power returns correct structure and physically meaningful
values given a valid route input, assuming current results are correct.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load the shared test config used across the test suite."""
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    return Config.assign_config(Path(configpath))


@pytest.fixture
def ship_config():
    """Load the shared ship config used across the test suite."""
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.tests.json')
    return ShipConfig.assign_config(Path(configpath))


@pytest.fixture
def boat(ship_config):
    """Initialise a DirectPowerBoat from the ship config."""
    return DirectPowerBoat(ship_config)


@pytest.fixture
def constraint_list():
    """Return a dummy constraint list (no active constraints)."""
    return basic_test_func.generate_dummy_constraint_list()


@pytest.fixture
def routing_problem(config, boat, constraint_list):
    """Return a RoutingProblem instance ready for unit testing."""
    departure_time = datetime(2025, 4, 1, 12, 0)
    arrival_time   = datetime(2025, 4, 2, 12, 0)
    objectives     = {"arrival_time": 1.5, "fuel_consumption": 1.5}

    return RoutingProblem(
        departure_time=departure_time,
        arrival_time=arrival_time,
        boat=boat,
        boat_speed=config.BOAT_SPEED,
        constraint_list=constraint_list,
        objectives=objectives,
    )


@pytest.fixture
def dummy_route():
    """
    Minimal valid route array with columns [lat, lon, speed].
    Route goes from Mediterranean (35N, 15E) to (32N, 28E) — same region
    used by existing tests in test_genetic.py.
    """
    return np.array([
        [35.199, 15.490, 7.0],
        [34.804, 16.759, 7.0],
        [34.447, 18.381, 7.0],
        [34.142, 18.763, 7.0],
        [33.942, 21.080, 7.0],
        [33.542, 23.024, 7.0],
        [33.408, 24.389, 7.0],
        [33.166, 26.300, 7.0],
        [32.937, 27.859, 7.0],
        [32.737, 28.859, 7.0],
    ])


# ---------------------------------------------------------------------------
# Tests for RoutingProblem.get_power
# ---------------------------------------------------------------------------

class TestGetPower:
    """Unit tests for RoutingProblem.get_power assuming current results are correct."""

    def test_get_power_returns_dict(self, routing_problem, dummy_route):
        """get_power must return a dictionary."""
        result = routing_problem.get_power(dummy_route)
        assert isinstance(result, dict)

    def test_get_power_dict_has_required_keys(self, routing_problem, dummy_route):
        """Returned dict must contain 'fuel_sum', 'shipparams', and 'time_obj'."""
        result = routing_problem.get_power(dummy_route)
        assert "fuel_sum"    in result
        assert "shipparams"  in result
        assert "time_obj"    in result

    def test_get_power_fuel_sum_is_positive(self, routing_problem, dummy_route):
        """Total fuel consumption must be a positive quantity."""
        result = routing_problem.get_power(dummy_route)
        fuel = result["fuel_sum"]
        assert fuel is not None
        assert fuel.value > 0, f"Expected positive fuel, got {fuel}"

    def test_get_power_fuel_sum_has_mass_unit(self, routing_problem, dummy_route):
        """Fuel sum must carry a mass unit (kg)."""
        result = routing_problem.get_power(dummy_route)
        fuel = result["fuel_sum"]
        assert fuel.unit.is_equivalent(u.kg), (
            f"Expected fuel unit equivalent to kg, got {fuel.unit}"
        )

    def test_get_power_time_obj_is_positive(self, routing_problem, dummy_route):
        """Arrival-time objective must be a positive scalar."""
        result = routing_problem.get_power(dummy_route)
        time_obj = result["time_obj"]
        assert time_obj is not None
        assert time_obj > 0, f"Expected positive time_obj, got {time_obj}"

    def test_get_power_shipparams_not_none(self, routing_problem, dummy_route):
        """ShipParams object must not be None."""
        result = routing_problem.get_power(dummy_route)
        assert result["shipparams"] is not None

    def test_get_power_deterministic(self, routing_problem, dummy_route):
        """Calling get_power twice with the same route must return identical fuel values."""
        result_one = routing_problem.get_power(dummy_route)
        result_two = routing_problem.get_power(dummy_route)
        assert result_one["fuel_sum"].value == pytest.approx(
            result_two["fuel_sum"].value, rel=1e-6
        ), "get_power is not deterministic for identical inputs"

    def test_get_power_higher_speed_increases_fuel(self, routing_problem, dummy_route):
        """A faster route should consume more fuel (sanity / physics check)."""
        slow_route = dummy_route.copy()
        fast_route = dummy_route.copy()
        slow_route[:, 2] = 3.0   # 3 m/s
        fast_route[:, 2] = 9.0   # 9 m/s

        fuel_slow = routing_problem.get_power(slow_route)["fuel_sum"]
        fuel_fast = routing_problem.get_power(fast_route)["fuel_sum"]

        assert fuel_fast.value > fuel_slow.value, (
            f"Expected faster route to burn more fuel: "
            f"fast={fuel_fast:.2f}, slow={fuel_slow:.2f}"
        )
