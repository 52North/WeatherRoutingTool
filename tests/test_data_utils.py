import pytest
import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from unittest.mock import patch

from WeatherRoutingTool.algorithms.data_utils import get_closest, get_speed_from_arrival_time

@pytest.fixture
def sample_array():
    return np.array([10.5, 20.0, 30.5, 40.0, 50.5])

@pytest.fixture
def route_lons():
    return np.array([10.0, 11.0, 12.0])

@pytest.fixture
def route_lats():
    return np.array([50.0, 51.0, 52.0])

@pytest.fixture
def departure_time():
    return datetime(2023, 1, 1, 12, 0, 0)

@pytest.fixture
def arrival_time():
    return datetime(2023, 1, 2, 12, 0, 0) # 24 hours later

@pytest.mark.parametrize(
    "target_value, expected_index",
    [
        (10.0, 0),    # Closest to 10.5
        (20.1, 1),    # Closest to 20.0
        (45.0, 3),    # Closter to 40.0 than 50.5
        (100.0, 4),   # Beyond max
        (0.0, 0),     # Below min
    ]
)
def test_get_closest(sample_array, target_value, expected_index):
    assert get_closest(sample_array, target_value) == expected_index

@pytest.mark.parametrize(
    "dist_values, expected_speed_mps",
    [
        ([50000, 50000], 100000 / (24 * 3600)), # 100km over 24 hours
        ([100000, 200000], 300000 / (24 * 3600)), 
        ([0, 0], 0.0),
    ]
)
@patch('WeatherRoutingTool.algorithms.data_utils.RouteParams.get_per_waypoint_coords')
def test_get_speed_from_arrival_time(mock_get_coords, route_lons, route_lats, departure_time, arrival_time, dist_values, expected_speed_mps):
    mock_get_coords.return_value = {
        'dist': np.array(dist_values) * u.meter
    }
    
    speed = get_speed_from_arrival_time(route_lons, route_lats, departure_time, arrival_time)
    
    # Check if the mock was called correctly
    mock_get_coords.assert_called_once()
    
    # The expected speed should be close
    assert np.isclose(speed.value, expected_speed_mps)
    assert speed.unit == u.meter / u.second
