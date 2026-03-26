import pytest

from WeatherRoutingTool.utils.maps import Map


@pytest.fixture
def basic_map():
    return Map(50.0, 5.0, 55.0, 15.0)


class TestMapInit:
    def test_attributes_set(self, basic_map):
        assert basic_map.lat1 == 50.0
        assert basic_map.lon1 == 5.0
        assert basic_map.lat2 == 55.0
        assert basic_map.lon2 == 15.0

    def test_values_stored_as_float(self):
        m = Map(10, 20, 30, 40)
        assert isinstance(m.lat1, float)
        assert isinstance(m.lon1, float)
        assert isinstance(m.lat2, float)
        assert isinstance(m.lon2, float)

    def test_negative_coords(self):
        m = Map(-33.9, -70.6, -30.0, -65.0)
        assert m.lat1 == -33.9
        assert m.lon1 == -70.6


class TestExtendVariable:
    def test_min_subtracts(self, basic_map):
        assert basic_map.extend_variable(10.0, "min", 2) == 8.0

    def test_max_adds(self, basic_map):
        assert basic_map.extend_variable(10.0, "max", 2) == 12.0

    def test_invalid_type_raises(self, basic_map):
        with pytest.raises(ValueError):
            basic_map.extend_variable(10.0, "center", 2)

    def test_zero_width(self, basic_map):
        assert basic_map.extend_variable(5.0, "min", 0) == 5.0
        assert basic_map.extend_variable(5.0, "max", 0) == 5.0


class TestGetWideenedMap:
    def test_returns_map_instance(self, basic_map):
        assert isinstance(basic_map.get_widened_map(1), Map)

    def test_widens_by_width(self, basic_map):
        widened = basic_map.get_widened_map(2)
        assert widened.lat1 == 48.0
        assert widened.lon1 == 3.0
        assert widened.lat2 == 57.0
        assert widened.lon2 == 17.0

    def test_zero_width_unchanged(self, basic_map):
        widened = basic_map.get_widened_map(0)
        assert widened.lat1 == basic_map.lat1
        assert widened.lon1 == basic_map.lon1
        assert widened.lat2 == basic_map.lat2
        assert widened.lon2 == basic_map.lon2


class TestGetVarTuple:
    def test_returns_tuple(self, basic_map):
        assert isinstance(basic_map.get_var_tuple(), tuple)

    def test_correct_order(self, basic_map):
        assert basic_map.get_var_tuple() == (50.0, 55.0, 5.0, 15.0)

    def test_length(self, basic_map):
        assert len(basic_map.get_var_tuple()) == 4
