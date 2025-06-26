import pytest
from WeatherRoutingTool.config import Config


class TestConfig:
    @pytest.fixture(scope="class")
    def minimal_config(self):
        """A minimal valid config dictionary to use as a base for tests."""
        return {
            "DEFAULT_MAP": [30, -155, 65, -115],
            "DEFAULT_ROUTE": [37.416, -123.607, 58.760, -151.494],
            "DEPARTURE_TIME": "2023-11-11T11:11Z",
            "DEPTH_DATA": "path/to/depth_data",
            "ROUTE_PATH": "path/to/route.json",
            "WEATHER_DATA": "path/to/weather_data"
        }

    def test_valid_config_initialization(self, minimal_config):
        """Tests that a valid config does not raise an exception."""
        try:
            Config(init_mode='from_dict', config_dict=minimal_config)
        except ValueError as e:
            pytest.fail(f"Valid config raised an unexpected ValueError: {e}")

    def test_invalid_map_latitude_raises_error(self, minimal_config):
        """Tests that an out-of-range latitude in DEFAULT_MAP raises ValueError."""
        invalid_config = minimal_config.copy()
        invalid_config["DEFAULT_MAP"] = [30, -155, 95, -115]  # lat_max > 90
        with pytest.raises(ValueError, match="Latitudes in DEFAULT_MAP must be between -90 and 90"):
            Config(init_mode='from_dict', config_dict=invalid_config)

    def test_invalid_departure_time_format_raises_error(self, minimal_config):
        """Tests that an invalid DEPARTURE_TIME format raises ValueError."""
        invalid_config = minimal_config.copy()
        invalid_config["DEPARTURE_TIME"] = "2023/11/11 11:11"
        with pytest.raises(ValueError, match="DEPARTURE_TIME must be in the format 'yyyy-mm-ddThh:mmZ'"):
            Config(init_mode='from_dict', config_dict=invalid_config)

    def test_negative_delta_fuel_raises_error(self, minimal_config):
        """Tests that a negative DELTA_FUEL raises ValueError."""
        invalid_config = minimal_config.copy()
        invalid_config["DELTA_FUEL"] = -100
        with pytest.raises(ValueError, match="DELTA_FUEL must be a positive number"):
            Config(init_mode='from_dict', config_dict=invalid_config)

    def test_non_even_router_hdgs_segments_raises_error(self, minimal_config):
        """Tests that a non-even ROUTER_HDGS_SEGMENTS raises ValueError."""
        invalid_config = minimal_config.copy()
        invalid_config["ROUTER_HDGS_SEGMENTS"] = 31
        with pytest.raises(ValueError, match="ROUTER_HDGS_SEGMENTS must be a positive even integer"):
            Config(init_mode='from_dict', config_dict=invalid_config)
