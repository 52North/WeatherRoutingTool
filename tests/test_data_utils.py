from datetime import datetime, timedelta

import numpy as np
import pytest
from astropy import units as u
from geographiclib.geodesic import Geodesic

from WeatherRoutingTool.algorithms.data_utils import (
    distance,
    get_closest,
    get_speed_from_arrival_time,
)


@pytest.fixture
def wgs84_geodesic():
    """Fixture providing the WGS84 geodesic model."""
    return Geodesic.WGS84


# --- Tests for get_closest ---


@pytest.mark.parametrize(
    "array, value, expected_idx",
    [
        (np.array([1.0, 2.0, 5.0, 10.0]), 4.8, 2),
        (np.array([1.0, 2.0, 5.0, 10.0]), 1.0, 0),
        (np.array([1.0, 2.0, 5.0, 10.0]), 12.0, 3),
        (np.array([-10.0, -5.0, 0.0, 5.0]), -4.2, 1),
        (np.array([10.0, 20.0]), 15.0, 0),  # ties resolve to the first index
    ],
)
def test_get_closest(array, value, expected_idx):
    """Test finding index of closest value in a 1D array."""
    assert get_closest(array, value) == expected_idx


# --- Tests for distance ---


def test_distance_single_point():
    """A single coordinate should yield an array with distance 0.0."""
    route = np.array([[10.0, 20.0]])
    dists = distance(route)

    assert isinstance(dists, np.ndarray)
    assert dists.shape == (1,)
    assert dists[0] == pytest.approx(0.0)


def test_distance_zero_movement():
    """Identical coordinates should produce zero accumulated distance."""
    route = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    dists = distance(route)

    assert np.allclose(dists, np.array([0.0, 0.0, 0.0]))


def test_distance_along_meridian_known_geodesic(wgs84_geodesic):
    """Test 2-point meridian route matches analytical WGS84 distance."""
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 1.0, 0.0
    expected_s12 = wgs84_geodesic.Inverse(lat1, lon1, lat2, lon2)["s12"]

    route = np.array([[lat1, lon1], [lat2, lon2]])
    dists = distance(route)

    assert dists.shape == (2,)
    assert dists[0] == pytest.approx(0.0)
    assert dists[1] == pytest.approx(expected_s12)


def test_distance_accumulated_monotonic(wgs84_geodesic):
    """Test multi-point route strictly accumulates geodesic distances."""
    route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    dists = distance(route)

    seg1 = wgs84_geodesic.Inverse(0.0, 0.0, 1.0, 0.0)["s12"]
    seg2 = wgs84_geodesic.Inverse(1.0, 0.0, 2.0, 0.0)["s12"]
    expected = np.array([0.0, seg1, seg1 + seg2])

    assert np.allclose(dists, expected)


# --- Tests for get_speed_from_arrival_time ---


def test_get_speed_from_arrival_time():
    """Test speed calculation across coordinates and known time delta."""
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 0.0])
    departure = datetime(2026, 1, 1, 0, 0, 0)
    # 10 hours travel time
    arrival = departure + timedelta(hours=10)

    speed = get_speed_from_arrival_time(lons, lats, departure, arrival)

    assert isinstance(speed, u.Quantity)
    assert speed.unit == (u.meter / u.second)
    # 1 deg latitude is ~110,574 m; over 36,000s, speed is ~3.07 m/s
    assert speed.value == pytest.approx(3.07, rel=1e-1)
