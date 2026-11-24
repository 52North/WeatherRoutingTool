from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u

import WeatherRoutingTool.algorithms.genetic.problem as genetic_problem
import WeatherRoutingTool.algorithms.genetic.utils as genetic_utils
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.errors import WRTConfigError


@pytest.fixture
def problem_factory():
    def create_problem(objectives=None, boat_speed=6 * u.meter / u.second):
        return RoutingProblem(
            departure_time=datetime(2025, 4, 1, 11, 11),
            arrival_time=datetime(2025, 4, 1, 21, 11),
            boat=MagicMock(),
            boat_speed=boat_speed,
            constraint_list=MagicMock(),
            objectives=objectives if objectives is not None else {
                "arrival_time": 1.5,
                "fuel_consumption": 1.5,
            },
            symmetric_time_objective=True,
        )

    return create_problem

@pytest.fixture
def mock_route_data():
    departure = datetime(2025, 4, 1, 11, 11)

    return {
        "start_lats": np.array([10.0, 11.0]),
        "start_lons": np.array([20.0, 21.0]),
        "courses": np.array([90.0, 90.0]),
        "start_times": [
            departure,
            departure + timedelta(seconds=10),
        ],
        "travel_times": np.array([10.0, 20.0]) * u.second,
    }
@pytest.fixture
def time_objective_for_difference(mock_route_data, monkeypatch):
    actual_arrival = (
        mock_route_data["start_times"][-1]
        + timedelta(seconds=mock_route_data["travel_times"][-1].value)
    )

    route = np.array([
        [10.0, 20.0, 6.0],
        [11.0, 21.0, 6.0],
        [12.0, 22.0, 6.0],
    ])

    monkeypatch.setattr(
        RouteParams,
        "get_per_waypoint_coords",
        MagicMock(return_value=mock_route_data),
    )

    def calculate(problem, difference_minutes):
        problem.arrival_time = actual_arrival + timedelta(
            minutes=difference_minutes
        )
        problem.boat.get_ship_parameters.return_value = MagicMock()

        return problem.get_power(route)["time_obj"]

    return calculate




class TestRoutingProblemInit:

    def test_fixed_speed_disables_arrival_time_speed(self, problem_factory):
        problem = problem_factory()

        assert problem.boat_speed_from_arrival_time is False

    def test_missing_boat_speed_enables_arrival_time_speed(self, problem_factory):
        problem = problem_factory(boat_speed=None)

        assert problem.boat_speed_from_arrival_time is True

    def test_objectives_are_stored(self, problem_factory):
        objectives = {"fuel_consumption": 1.5}
        problem = problem_factory(objectives=objectives)

        assert problem.objectives == objectives




class TestGetObjectives:

    def test_raises_error_when_objectives_are_empty(self, problem_factory):
        problem = problem_factory(objectives={})

        with pytest.raises(WRTConfigError):
            problem.get_objectives({})

    def test_returns_arrival_time_and_fuel_in_expected_order(self, problem_factory):
        problem = problem_factory()
        obj_dict = {
            "time_obj": 12.5,
            "fuel_sum": 250 * u.kg,
        }

        result = problem.get_objectives(obj_dict)

        assert result == [12.5, 250]

    def test_returns_only_arrival_time_objective(self, problem_factory):
        problem = problem_factory(objectives={"arrival_time": 1.5})
        obj_dict = {
            "time_obj": 12.5,
            "fuel_sum": None,
        }

        result = problem.get_objectives(obj_dict)

        assert result == [12.5]

    def test_returns_only_fuel_consumption_objective(self, problem_factory):
        problem = problem_factory(objectives={"fuel_consumption": 1.5})
        obj_dict = {
            "time_obj": None,
            "fuel_sum": 250 * u.kg,
        }

        result = problem.get_objectives(obj_dict)

        assert result == [250]

    def test_raises_error_when_no_valid_objective_is_configured(self, problem_factory):
        problem = problem_factory(objectives={"unknown_objective": 1.0})

        with pytest.raises(WRTConfigError):
            problem.get_objectives({})
class TestGetPower:

    def test_calculates_fuel_consumption(self, problem_factory, mock_route_data, monkeypatch):
        problem = problem_factory(objectives={"fuel_consumption": 1.5})

        monkeypatch.setattr(
            RouteParams,
            "get_per_waypoint_coords",
            MagicMock(return_value=mock_route_data),
        )

        shipparams = MagicMock()
        shipparams.get_fuel_rate.return_value = np.array([2.0, 3.0]) * u.kg / u.second
        problem.boat.get_ship_parameters.return_value = shipparams

        route = np.array([[
            [10.0, 20.0, 6.0],
            [11.0, 21.0, 6.0],
            [12.0, 22.0, 6.0],
        ]])

        result = problem.get_power(route)

        assert result["fuel_sum"] == 80 * u.kg
        assert result["time_obj"] is None
        assert result["shipparams"] is shipparams

    def test_calculates_zero_time_objective_on_exact_arrival(
        self, problem_factory, mock_route_data, monkeypatch
    ):
        departure = datetime(2025, 4, 1, 11, 11)
        arrival = departure + timedelta(seconds=30)

        problem = problem_factory(objectives={"arrival_time": 1.5})
        problem.arrival_time = arrival

        monkeypatch.setattr(
            RouteParams,
            "get_per_waypoint_coords",
            MagicMock(return_value=mock_route_data),
        )

        problem.boat.get_ship_parameters.return_value = MagicMock()

        route = np.array([
            [10.0, 20.0, 6.0],
            [11.0, 21.0, 6.0],
            [12.0, 22.0, 6.0],
        ])

        result = problem.get_power(route)

        assert result["time_obj"] == 0
        assert result["fuel_sum"] is None
    def test_calculates_symmetric_time_objective_for_early_arrival(
        self, problem_factory, mock_route_data, monkeypatch
    ):
        departure = datetime(2025, 4, 1, 11, 11)
        arrival = departure + timedelta(seconds=90)

        problem = problem_factory(objectives={"arrival_time": 1.5})
        problem.arrival_time = arrival

        monkeypatch.setattr(
            RouteParams,
            "get_per_waypoint_coords",
            MagicMock(return_value=mock_route_data),
        )

        problem.boat.get_ship_parameters.return_value = MagicMock()

        route = np.array([
            [10.0, 20.0, 6.0],
            [11.0, 21.0, 6.0],
            [12.0, 22.0, 6.0],
        ])

        result = problem.get_power(route)

        expected = (1.0 / (120 ** 4)) * (1.0 ** 4)
        assert result["time_obj"] == pytest.approx(expected)


    def test_calculates_non_symmetric_time_objective_for_early_arrival(
        self, problem_factory, mock_route_data, monkeypatch
    ):
        departure = datetime(2025, 4, 1, 11, 11)
        arrival = departure + timedelta(seconds=90)

        problem = problem_factory(objectives={"arrival_time": 1.5})
        problem.arrival_time = arrival
        problem.symmetric_time_objective = False

        monkeypatch.setattr(
            RouteParams,
            "get_per_waypoint_coords",
            MagicMock(return_value=mock_route_data),
        )

        problem.boat.get_ship_parameters.return_value = MagicMock()

        route = np.array([
            [10.0, 20.0, 6.0],
            [11.0, 21.0, 6.0],
            [12.0, 22.0, 6.0],
        ])

        result = problem.get_power(route)

        expected = (100.0 / (120 ** 4)) * (1.0 ** 2)
        assert result["time_obj"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("difference_minutes", "expected"),
        [
            (120, 1.0),
            (-120, 1.0),
            (120.01, 1000),
            (-120.01, 1000),
        ],
    )
    def test_time_objective_at_and_beyond_120_minute_boundary(
        self,
        problem_factory,
        time_objective_for_difference,
        difference_minutes,
        expected,
    ):
        problem = problem_factory(objectives={"arrival_time": 1.5})

        result = time_objective_for_difference(
            problem, difference_minutes
        )

        assert result == pytest.approx(expected)

    def test_late_arrival_under_one_minute_uses_one_minute_penalty(
        self, problem_factory, time_objective_for_difference
    ):
        problem = problem_factory(objectives={"arrival_time": 1.5})

        result = time_objective_for_difference(problem, -0.5)

        assert result == pytest.approx(1.0 / (120 ** 4))

    def test_early_arrival_under_one_minute_uses_one_minute_penalty(
        self, problem_factory, time_objective_for_difference
    ):
        problem = problem_factory(objectives={"arrival_time": 1.5})
        problem.symmetric_time_objective = False

        result = time_objective_for_difference(problem, 0.5)

        assert result == pytest.approx(100.0 / (120 ** 4))



    def test_uses_arrival_time_to_calculate_boat_speed(
        self, problem_factory, mock_route_data, monkeypatch
    ):
        problem = problem_factory(
            objectives={"fuel_consumption": 1.5},
            boat_speed=None,
        )

        speed_mock = MagicMock(return_value=6 * u.meter / u.second)
        monkeypatch.setattr(
            genetic_problem,
            "get_speed_from_arrival_time",
            speed_mock,
        )

        monkeypatch.setattr(
            RouteParams,
            "get_per_waypoint_coords",
            MagicMock(return_value=mock_route_data),
        )

        problem.boat.get_ship_parameters.return_value = MagicMock()

        route = np.array([
            [10.0, 20.0, 6.0],
            [11.0, 21.0, 6.0],
            [12.0, 22.0, 6.0],
        ])

        problem.get_power(route)

        speed_mock.assert_called_once()

        actual_kwargs = speed_mock.call_args.kwargs

        np.testing.assert_array_equal(actual_kwargs["lons"], route[:, 1])
        np.testing.assert_array_equal(actual_kwargs["lats"], route[:, 0])

        assert actual_kwargs["departure_time"] == problem.departure_time
        assert actual_kwargs["arrival_time"] == problem.arrival_time

        actual_speed = problem.boat.get_ship_parameters.call_args.kwargs["speed"]

        np.testing.assert_allclose(
            actual_speed.to_value(u.meter / u.second),
            [6.0, 6.0],
        )
class TestEvaluate:

    def test_evaluate_sets_objectives_and_constraints(self, problem_factory, monkeypatch):
        problem = problem_factory(
            objectives={
                "arrival_time": 1.5,
                "fuel_consumption": 1.5,
            }
        )

        start_times = [
            datetime(2025, 4, 1, 11, 11),
            datetime(2025, 4, 1, 11, 11, 10),
        ]

        obj_dict = {
            "time_obj": 12.5,
            "fuel_sum": 250 * u.kg,
            "start_times": start_times,
        }

        monkeypatch.setattr(problem, "get_power", MagicMock(return_value=obj_dict))

        constraints_mock = MagicMock(return_value=2)
        monkeypatch.setattr(genetic_utils, "get_constraints", constraints_mock)

        out = {}
        route = np.array([
            [10.0, 20.0, 6.0],
            [11.0, 21.0, 6.0],
            [12.0, 22.0, 6.0],
        ])

        problem._evaluate(route, out)

        assert out["F"] == [12.5, 250]
        np.testing.assert_array_equal(out["G"], np.array([[2]]))

        constraints_mock.assert_called_once()

        actual_route, actual_constraint_list, actual_start_times = (
            constraints_mock.call_args.args
        )

        np.testing.assert_array_equal(actual_route, route[0])
        assert actual_constraint_list is problem.constraint_list
        assert actual_start_times == start_times
