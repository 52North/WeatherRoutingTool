import numpy as np
import pytest
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem


def test_get_power():
    """
    Basic structural test to ensure get_power returns a tuple (fuel, shipparams).
    """

    # Create dummy object without calling full __init__
    rp = RoutingProblem.__new__(RoutingProblem)

    rp.boat_speed_from_arrival_time = False
    rp.departure_time = None
    rp.arrival_time = None

 
    class DummyShipParams:
        def get_fuel_rate(self):
            return np.array([1.0, 1.0])

    class DummyBoat:
        def get_ship_parameters(self, **kwargs):
            return DummyShipParams()

    rp.boat = DummyBoat()

    from unittest.mock import patch

    route = np.array([
        [10.0, 20.0, 5.0],
        [11.0, 21.0, 5.0],
        [12.0, 22.0, 5.0],
    ])

    with patch(
        "WeatherRoutingTool.algorithms.genetic.problem.RouteParams.get_per_waypoint_coords"
    ) as mock_routeparams:

        mock_routeparams.return_value = {
            "courses": None,
            "start_lats": None,
            "start_lons": None,
            "start_times": None,
            "travel_times": np.array([10.0, 20.0]),
        }

        total_fuel, shipparams = rp.get_power(route)

        assert total_fuel == pytest.approx(30.0)
        assert shipparams is not None
