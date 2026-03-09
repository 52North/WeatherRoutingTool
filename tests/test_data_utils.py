"""
Unit tests for WeatherRoutingTool/algorithms/data_utils.py

Covers:
- get_closest()
- distance()
- time_diffs()
- get_speed_from_arrival_time()
- GridMixin.index_to_coords()
- GridMixin.coords_to_index()
- GridMixin.get_shuffled_cost()
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from WeatherRoutingTool.algorithms.data_utils import (
    GridMixin,
    distance,
    get_closest,
    get_speed_from_arrival_time,
    time_diffs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equatorial_route():
    """Three waypoints along the equator: (lat=0, lon=0), (lat=0, lon=1), (lat=0, lon=3).

    Geodesic distances (WGS-84):
      segment 0→1 (1 deg lon): ~111 319.49 m
      segment 1→2 (2 deg lon): ~222 638.98 m
      accumulated:              [0, 111319.49, 333958.47]
    """
    return np.array([
        [0.0, 0.0],   # lat=0, lon=0
        [0.0, 1.0],   # lat=0, lon=1
        [0.0, 3.0],   # lat=0, lon=3
    ])


@pytest.fixture
def single_segment_route():
    """Two waypoints – one segment of exactly 1 degree longitude on the equator (~111 319.49 m)."""
    return np.array([
        [0.0, 0.0],
        [0.0, 1.0],
    ])


@pytest.fixture
def simple_grid():
    """A tiny 3×4 xarray Dataset with latitude and longitude coordinates."""
    lats = np.array([10.0, 20.0, 30.0])
    lons = np.array([100.0, 110.0, 120.0, 130.0])
    data = np.arange(12, dtype=float).reshape(3, 4)

    ds = xr.Dataset(
        {"cost": (["latitude", "longitude"], data)},
        coords={"latitude": lats, "longitude": lons},
    )
    return ds


@pytest.fixture
def grid_mixin_instance(simple_grid):
    """Concrete GridMixin subclass that carries the test grid."""

    class ConcreteGrid(GridMixin):
        def __init__(self, grid):
            self.grid = grid

    return ConcreteGrid(simple_grid)


# ---------------------------------------------------------------------------
# Tests: get_closest
# ---------------------------------------------------------------------------

class TestGetClosest:
    """Unit tests for get_closest()."""

    def test_exact_match_returns_correct_index(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert get_closest(arr, 3.0) == 2

    def test_closest_below(self):
        arr = np.array([0.0, 10.0, 20.0, 30.0])
        # 14 is closer to 10 (index 1) than to 20 (index 2)
        assert get_closest(arr, 14.0) == 1

    def test_closest_above(self):
        arr = np.array([0.0, 10.0, 20.0, 30.0])
        # 16 is closer to 20 (index 2) than to 10 (index 1)
        assert get_closest(arr, 16.0) == 2

    def test_tie_returns_first_index(self):
        """When two elements are equidistant, the smaller (first) index is returned."""
        arr = np.array([0.0, 10.0, 20.0])
        # 15 is equidistant between index 1 (10) and index 2 (20)
        assert get_closest(arr, 15.0) == 1

    def test_single_element_array(self):
        arr = np.array([42.0])
        assert get_closest(arr, 99.0) == 0

    def test_negative_values(self):
        arr = np.array([-30.0, -10.0, 0.0, 10.0])
        assert get_closest(arr, -11.0) == 1

    @pytest.mark.parametrize("value,expected_idx", [
        (1.0, 0),
        (2.0, 1),
        (3.0, 2),
        (4.0, 3),
        (5.0, 4),
    ])
    def test_parametrized_exact_matches(self, value, expected_idx):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert get_closest(arr, value) == expected_idx


# ---------------------------------------------------------------------------
# Tests: distance
# ---------------------------------------------------------------------------

class TestDistance:
    """Unit tests for distance().

    Note: distance() is currently unused in the main WRT package.
    See the warning in its docstring. These tests act as a regression guard.
    """

    # Known geodesic distances (WGS-84):
    #   1 degree longitude on equator ≈ 111 319.49 m
    #   accumulated for route [(0,0),(0,1),(0,3)] = [0, 111319.49, 333958.47]

    SEG1 = 111319.49   # m  (0°→1° lon, lat=0)
    SEG12 = 333958.47  # m  (0°→3° lon, lat=0; accumulated)

    def test_first_element_is_zero(self, equatorial_route):
        """Accumulated distance at the start waypoint must be zero."""
        dists = distance(equatorial_route)
        assert dists[0] == pytest.approx(0.0, abs=1e-3)

    def test_output_length_equals_number_of_waypoints(self, equatorial_route):
        dists = distance(equatorial_route)
        assert len(dists) == len(equatorial_route)

    def test_returns_numpy_array(self, equatorial_route):
        assert isinstance(distance(equatorial_route), np.ndarray)

    def test_accumulated_distance_first_segment(self, equatorial_route):
        """After the first segment (1 degree lon on equator) distance ≈ 111 319 m."""
        dists = distance(equatorial_route)
        assert dists[1] == pytest.approx(self.SEG1, rel=1e-4)

    def test_accumulated_distance_second_segment(self, equatorial_route):
        """After the second segment (2 more degrees lon) total ≈ 333 958 m."""
        dists = distance(equatorial_route)
        assert dists[2] == pytest.approx(self.SEG12, rel=1e-4)

    def test_distances_are_monotonically_non_decreasing(self, equatorial_route):
        dists = distance(equatorial_route)
        assert np.all(np.diff(dists) >= 0)

    def test_single_segment_route(self, single_segment_route):
        dists = distance(single_segment_route)
        assert len(dists) == 2
        assert dists[0] == pytest.approx(0.0, abs=1e-3)
        assert dists[1] == pytest.approx(self.SEG1, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: time_diffs
# ---------------------------------------------------------------------------

class TestTimeDiffs:
    """Unit tests for time_diffs().

    Note: time_diffs() is currently unused in the main WRT package.
    See the warning in its docstring. These tests act as a regression guard.
    """

    # At speed = 10 m/s the expected accumulated times for equatorial_route are:
    #   [0, 111319.49/10, 333958.47/10] = [0, 11131.95, 33395.85]

    SPEED = 10.0  # m/s
    T1 = 111319.49 / 10   # s after first segment
    T12 = 333958.47 / 10  # s after second segment

    def test_first_element_is_zero(self, equatorial_route):
        diffs = time_diffs(self.SPEED, equatorial_route)
        assert diffs[0] == pytest.approx(0.0, abs=1e-3)

    def test_output_length_equals_number_of_waypoints(self, equatorial_route):
        diffs = time_diffs(self.SPEED, equatorial_route)
        assert len(diffs) == len(equatorial_route)

    def test_returns_numpy_array(self, equatorial_route):
        assert isinstance(time_diffs(self.SPEED, equatorial_route), np.ndarray)

    def test_time_after_first_segment(self, equatorial_route):
        diffs = time_diffs(self.SPEED, equatorial_route)
        assert diffs[1] == pytest.approx(self.T1, rel=1e-4)

    def test_time_after_second_segment(self, equatorial_route):
        diffs = time_diffs(self.SPEED, equatorial_route)
        assert diffs[2] == pytest.approx(self.T12, rel=1e-4)

    def test_time_diffs_scale_inversely_with_speed(self, equatorial_route):
        """Doubling speed must halve all time differences."""
        diffs_slow = time_diffs(self.SPEED, equatorial_route)
        diffs_fast = time_diffs(self.SPEED * 2, equatorial_route)
        np.testing.assert_allclose(diffs_slow, diffs_fast * 2, rtol=1e-5)

    def test_time_diffs_are_monotonically_non_decreasing(self, equatorial_route):
        diffs = time_diffs(self.SPEED, equatorial_route)
        assert np.all(np.diff(diffs) >= 0)


# ---------------------------------------------------------------------------
# Tests: get_speed_from_arrival_time
# ---------------------------------------------------------------------------

class TestGetSpeedFromArrivalTime:
    """Unit tests for get_speed_from_arrival_time().

    Strategy: one segment on the equator (1 degree longitude ≈ 111 319.49 m).
    Choosing a time window of 111 319.49 s gives an expected speed of 1 m/s.
    Choosing a time window of 111 319.49 / 6 s gives an expected speed of 6 m/s.
    """

    # Distance for 1 degree longitude on equator (WGS-84)
    DIST_1DEG = 111319.49  # m

    @pytest.fixture
    def two_waypoint_lons_lats(self):
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    @pytest.fixture
    def departure(self):
        return datetime(2023, 1, 1, 0, 0, 0)

    def test_returns_astropy_quantity(self, two_waypoint_lons_lats, departure):
        lons, lats = two_waypoint_lons_lats
        arrival = departure + timedelta(seconds=self.DIST_1DEG)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert isinstance(result, u.Quantity)

    def test_speed_unit_is_meters_per_second(self, two_waypoint_lons_lats, departure):
        lons, lats = two_waypoint_lons_lats
        arrival = departure + timedelta(seconds=self.DIST_1DEG)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.unit.physical_type == 'speed'

    def test_speed_value_1_m_per_s(self, two_waypoint_lons_lats, departure):
        """When travel time equals distance in seconds, speed must be 1 m/s."""
        lons, lats = two_waypoint_lons_lats
        arrival = departure + timedelta(seconds=self.DIST_1DEG)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.to(u.meter / u.second).value == pytest.approx(1.0, rel=1e-3)

    def test_speed_value_6_m_per_s(self, two_waypoint_lons_lats, departure):
        """When travel time equals distance/6 seconds, speed must be 6 m/s."""
        lons, lats = two_waypoint_lons_lats
        arrival = departure + timedelta(seconds=self.DIST_1DEG / 6)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.to(u.meter / u.second).value == pytest.approx(6.0, rel=1e-3)

    def test_speed_doubles_when_time_halved(self, two_waypoint_lons_lats, departure):
        """Halving the available time window must double the required speed."""
        lons, lats = two_waypoint_lons_lats
        arrival_slow = departure + timedelta(seconds=self.DIST_1DEG)
        arrival_fast = departure + timedelta(seconds=self.DIST_1DEG / 2)

        speed_slow = get_speed_from_arrival_time(lons, lats, departure, arrival_slow)
        speed_fast = get_speed_from_arrival_time(lons, lats, departure, arrival_fast)

        assert speed_fast.value == pytest.approx(speed_slow.value * 2, rel=1e-3)

    @pytest.mark.parametrize("speed_factor", [1, 2, 5, 10])
    def test_parametrized_speed_factors(self, two_waypoint_lons_lats, departure, speed_factor):
        """Speed must equal distance / travel_time for several speed factors."""
        lons, lats = two_waypoint_lons_lats
        travel_time_s = self.DIST_1DEG / speed_factor
        arrival = departure + timedelta(seconds=travel_time_s)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.to(u.meter / u.second).value == pytest.approx(float(speed_factor), rel=1e-3)


# ---------------------------------------------------------------------------
# Tests: GridMixin
# ---------------------------------------------------------------------------

class TestGridMixin:
    """Unit tests for GridMixin methods."""

    # simple_grid layout (latitude × longitude):
    #   lats = [10, 20, 30]          (indices 0, 1, 2)
    #   lons = [100, 110, 120, 130]  (indices 0, 1, 2, 3)

    def test_index_to_coords_returns_correct_lat(self, grid_mixin_instance):
        """index_to_coords must map (lat_idx=1) → lat=20."""
        lats, lons, route = grid_mixin_instance.index_to_coords([(1, 0)])
        assert lats[0] == pytest.approx(20.0)

    def test_index_to_coords_returns_correct_lon(self, grid_mixin_instance):
        """index_to_coords must map (lon_idx=2) → lon=120."""
        lats, lons, route = grid_mixin_instance.index_to_coords([(0, 2)])
        assert lons[0] == pytest.approx(120.0)

    def test_index_to_coords_route_matches_lats_lons(self, grid_mixin_instance):
        lats, lons, route = grid_mixin_instance.index_to_coords([(0, 1), (2, 3)])
        assert route[0] == [lats[0], lons[0]]
        assert route[1] == [lats[1], lons[1]]

    def test_index_to_coords_multiple_points(self, grid_mixin_instance):
        points = [(0, 0), (1, 1), (2, 2)]
        lats, lons, route = grid_mixin_instance.index_to_coords(points)
        assert len(lats) == 3
        assert len(lons) == 3
        assert len(route) == 3

    def test_coords_to_index_exact_match(self, grid_mixin_instance):
        """coords_to_index must return index 1 for lat=20 (exact match)."""
        lat_indices, lon_indices, route = grid_mixin_instance.coords_to_index([(20.0, 110.0)])
        assert lat_indices[0] == 1   # lat=20 → index 1
        assert lon_indices[0] == 1   # lon=110 → index 1

    def test_coords_to_index_nearest_neighbor(self, grid_mixin_instance):
        """coords_to_index must snap to the nearest grid point."""
        # 14.9 is closer to 10 (idx 0) than to 20 (idx 1)
        lat_indices, _, _ = grid_mixin_instance.coords_to_index([(14.9, 100.0)])
        assert lat_indices[0] == 0

    def test_coords_to_index_route_matches_indices(self, grid_mixin_instance):
        lat_indices, lon_indices, route = grid_mixin_instance.coords_to_index([(10.0, 100.0), (30.0, 130.0)])
        assert route[0] == [lat_indices[0], lon_indices[0]]
        assert route[1] == [lat_indices[1], lon_indices[1]]

    def test_get_shuffled_cost_preserves_shape(self, simple_grid):
        """get_shuffled_cost must return an array with the same shape as the grid data."""

        class ConcreteGrid(GridMixin):
            def __init__(self, grid):
                self.grid = grid.cost  # pass DataArray, not Dataset

        obj = ConcreteGrid(simple_grid)
        shuffled = obj.get_shuffled_cost()
        assert shuffled.shape == simple_grid.cost.shape

    def test_get_shuffled_cost_replaces_nans_with_high_weight(self, simple_grid):
        """NaN cells in cost must become 1e20 in the shuffled output."""
        grid_with_nan = simple_grid.copy(deep=True)
        grid_with_nan["cost"].values[0, 0] = np.nan

        class ConcreteGrid(GridMixin):
            def __init__(self, grid):
                self.grid = grid.cost

        obj = ConcreteGrid(grid_with_nan)
        shuffled = obj.get_shuffled_cost()
        assert np.any(shuffled == 1e20)

    def test_get_shuffled_cost_no_nans_in_result_for_nan_positions(self, simple_grid):
        """After shuffling, positions that were NaN must not remain NaN."""
        grid_with_nan = simple_grid.copy(deep=True)
        grid_with_nan["cost"].values[1, 1] = np.nan

        class ConcreteGrid(GridMixin):
            def __init__(self, grid):
                self.grid = grid.cost

        obj = ConcreteGrid(grid_with_nan)
        shuffled = obj.get_shuffled_cost()
        assert not np.any(np.isnan(shuffled))
