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


@pytest.fixture
def equatorial_route():
    return np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 3.0],
    ])


@pytest.fixture
def single_segment_route():
    return np.array([
        [0.0, 0.0],
        [0.0, 1.0],
    ])


@pytest.fixture
def simple_grid():
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
    class ConcreteGrid(GridMixin):
        def __init__(self, grid):
            self.grid = grid

    return ConcreteGrid(simple_grid)


class TestGetClosest:

    def test_exact_match_returns_correct_index(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert get_closest(arr, 3.0) == 2

    def test_closest_below(self):
        arr = np.array([0.0, 10.0, 20.0, 30.0])
        assert get_closest(arr, 14.0) == 1

    def test_closest_above(self):
        arr = np.array([0.0, 10.0, 20.0, 30.0])
        assert get_closest(arr, 16.0) == 2

    def test_tie_returns_first_index(self):
        arr = np.array([0.0, 10.0, 20.0])
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


class TestDistance:

    SEG1 = 111319.49
    SEG12 = 333958.47

    def test_first_element_is_zero(self, equatorial_route):
        dists = distance(equatorial_route)
        assert dists[0] == pytest.approx(0.0, abs=1e-3)

    def test_output_length_equals_number_of_waypoints(self, equatorial_route):
        dists = distance(equatorial_route)
        assert len(dists) == len(equatorial_route)

    def test_returns_numpy_array(self, equatorial_route):
        assert isinstance(distance(equatorial_route), np.ndarray)

    def test_accumulated_distance_first_segment(self, equatorial_route):
        dists = distance(equatorial_route)
        assert dists[1] == pytest.approx(self.SEG1, rel=1e-4)

    def test_accumulated_distance_second_segment(self, equatorial_route):
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


class TestTimeDiffs:

    SPEED = 10.0
    T1 = 111319.49 / 10
    T12 = 333958.47 / 10

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
        diffs_slow = time_diffs(self.SPEED, equatorial_route)
        diffs_fast = time_diffs(self.SPEED * 2, equatorial_route)
        np.testing.assert_allclose(diffs_slow, diffs_fast * 2, rtol=1e-5)

    def test_time_diffs_are_monotonically_non_decreasing(self, equatorial_route):
        diffs = time_diffs(self.SPEED, equatorial_route)
        assert np.all(np.diff(diffs) >= 0)


class TestGetSpeedFromArrivalTime:

    DIST_1DEG = 111319.49

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
        lons, lats = two_waypoint_lons_lats
        arrival = departure + timedelta(seconds=self.DIST_1DEG)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.to(u.meter / u.second).value == pytest.approx(1.0, rel=1e-3)

    def test_speed_value_6_m_per_s(self, two_waypoint_lons_lats, departure):
        lons, lats = two_waypoint_lons_lats
        arrival = departure + timedelta(seconds=self.DIST_1DEG / 6)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.to(u.meter / u.second).value == pytest.approx(6.0, rel=1e-3)

    def test_speed_doubles_when_time_halved(self, two_waypoint_lons_lats, departure):
        lons, lats = two_waypoint_lons_lats
        arrival_slow = departure + timedelta(seconds=self.DIST_1DEG)
        arrival_fast = departure + timedelta(seconds=self.DIST_1DEG / 2)
        speed_slow = get_speed_from_arrival_time(lons, lats, departure, arrival_slow)
        speed_fast = get_speed_from_arrival_time(lons, lats, departure, arrival_fast)
        assert speed_fast.value == pytest.approx(speed_slow.value * 2, rel=1e-3)

    @pytest.mark.parametrize("speed_factor", [1, 2, 5, 10])
    def test_parametrized_speed_factors(self, two_waypoint_lons_lats, departure, speed_factor):
        lons, lats = two_waypoint_lons_lats
        travel_time_s = self.DIST_1DEG / speed_factor
        arrival = departure + timedelta(seconds=travel_time_s)
        result = get_speed_from_arrival_time(lons, lats, departure, arrival)
        assert result.to(u.meter / u.second).value == pytest.approx(float(speed_factor), rel=1e-3)


class TestGridMixin:

    def test_index_to_coords_returns_correct_lat(self, grid_mixin_instance):
        lats, lons, route = grid_mixin_instance.index_to_coords([(1, 0)])
        assert lats[0] == pytest.approx(20.0)

    def test_index_to_coords_returns_correct_lon(self, grid_mixin_instance):
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
        lat_indices, lon_indices, route = grid_mixin_instance.coords_to_index([(20.0, 110.0)])
        assert lat_indices[0] == 1
        assert lon_indices[0] == 1

    def test_coords_to_index_nearest_neighbor(self, grid_mixin_instance):
        lat_indices, _, _ = grid_mixin_instance.coords_to_index([(14.9, 100.0)])
        assert lat_indices[0] == 0

    def test_coords_to_index_route_matches_indices(self, grid_mixin_instance):
        lat_indices, lon_indices, route = grid_mixin_instance.coords_to_index([(10.0, 100.0), (30.0, 130.0)])
        assert route[0] == [lat_indices[0], lon_indices[0]]
        assert route[1] == [lat_indices[1], lon_indices[1]]

    def test_get_shuffled_cost_preserves_shape(self, simple_grid):
        class ConcreteGrid(GridMixin):
            def __init__(self, grid):
                self.grid = grid.cost

        obj = ConcreteGrid(simple_grid)
        shuffled = obj.get_shuffled_cost()
        assert shuffled.shape == simple_grid.cost.shape

    def test_get_shuffled_cost_replaces_nans_with_high_weight(self, simple_grid):
        grid_with_nan = simple_grid.copy(deep=True)
        grid_with_nan["cost"].values[0, 0] = np.nan

        class ConcreteGrid(GridMixin):
            def __init__(self, grid):
                self.grid = grid.cost

        obj = ConcreteGrid(grid_with_nan)
        shuffled = obj.get_shuffled_cost()
        assert np.any(shuffled == 1e20)

    def test_get_shuffled_cost_no_nans_in_result_for_nan_positions(self, simple_grid):
        grid_with_nan = simple_grid.copy(deep=True)
        grid_with_nan["cost"].values[1, 1] = np.nan

        class ConcreteGrid(GridMixin):
            def __init__(self, grid):
                self.grid = grid.cost

        obj = ConcreteGrid(grid_with_nan)
        shuffled = obj.get_shuffled_cost()
        assert not np.any(np.isnan(shuffled))
