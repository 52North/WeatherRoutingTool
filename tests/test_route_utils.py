import numpy as np
from geographiclib.geodesic import Geodesic

from WeatherRoutingTool.algorithms.data_utils import distance, time_diffs


def test_distance_two_point_meridian():
    """
    Route: (0,0) -> (1,0)
    Expected distance ≈ geodesic distance between the two points.
    """
    route = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])

    dists = distance(route)

    expected = Geodesic.WGS84.Inverse(0, 0, 1, 0)['s12']

    assert np.isclose(dists[-1], expected, rtol=1e-4)


def test_time_diffs_constant_speed():
    """
    Same route but convert distance to time using speed.
    """
    route = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])

    speed = 10.0  # m/s

    times = time_diffs(speed, route)

    expected_dist = Geodesic.WGS84.Inverse(0, 0, 1, 0)['s12']
    expected_time = expected_dist / speed

    assert np.isclose(times[-1], expected_time, rtol=1e-4)