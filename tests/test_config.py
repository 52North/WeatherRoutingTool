import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from geovectorslib import geod

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.utils.errors import ERROR_MESSAGES, WRTConfigError, WRTErrorCodes


def load_example_config():
    config_path = Path(__file__).parent / "config.tests.json"
    with config_path.open("r") as f:
        return json.load(f), config_path


def assert_config_error(excinfo, detail):
    """Check that a WRTConfigError carries the correct error code, message prefix and detail."""
    err = excinfo.value
    prefix = f"[E{WRTErrorCodes.CONFIG_INVALID:03d}] {ERROR_MESSAGES[WRTErrorCodes.CONFIG_INVALID].format(detail='')}"

    assert err.code == WRTErrorCodes.CONFIG_INVALID
    assert str(err).startswith(prefix.strip())
    assert detail in str(err)


def test_assign_config_from_json():
    _, config_path = load_example_config()
    config = Config.assign_config(path=config_path, init_mode="from_file")

    assert isinstance(config, Config)
    assert config.CONFIG_PATH == config_path
    assert config.DEPARTURE_TIME is not None
    assert config.ISOCHRONE_PRUNE_SYMMETRY_AXIS == 'gcr'


def test_assign_config_from_dict():
    config_data, _ = load_example_config()
    config = Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert isinstance(config, Config)
    assert config.DEPARTURE_TIME is not None
    assert config.ISOCHRONE_PRUNE_SYMMETRY_AXIS == 'gcr'


def test_invalid_time_raises_error():
    config_data, _ = load_example_config()
    config_data["DEPARTURE_TIME"] = "2023-11-11T1111Z"

    detail = "'DEPARTURE_TIME/ARRIVAL_TIME' must be in format YYYY-MM-DDTHH:MMZ"
    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, detail)
    # the full message is fixed: error code + generic message + list of validation errors
    full_detail = f"1 validation error(s):\n  DEPARTURE_TIME: {detail}"
    expected = (f"[E{WRTErrorCodes.CONFIG_INVALID:03d}] "
                f"{ERROR_MESSAGES[WRTErrorCodes.CONFIG_INVALID].format(detail=full_detail)}")
    assert str(excinfo.value) == expected


def test_multiple_validation_errors_are_collected():
    config_data, _ = load_example_config()
    config_data["DEPARTURE_TIME"] = "2023-11-11T1111Z"
    config_data["ROUTER_HDGS_SEGMENTS"] = 3
    config_data["DELTA_FUEL"] = -1

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "3 validation error(s)")
    assert "DEPARTURE_TIME:" in str(excinfo.value)
    assert "ROUTER_HDGS_SEGMENTS:" in str(excinfo.value)
    assert "DELTA_FUEL:" in str(excinfo.value)


def test_invalid_path_raises_error(tmp_path):
    config_data, _ = load_example_config()
    config_data["ROUTE_PATH"] = str(tmp_path / "nonexistent.nc")

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "Path doesn't exist")


def test_speedy_alg_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["ALGORITHM_TYPE"] = "speedy_isobased"

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "If 'ALGORITHM_TYPE' is 'speedy_isobased', 'BOAT_TYPE' has to be 'speedy_isobased'.")


def test_genetic_shortest_route_boat_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["ALGORITHM_TYPE"] = "genetic_shortest_route"

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(
        excinfo, "If 'ALGORITHM_TYPE' is 'genetic_shortest_route', 'BOAT_TYPE' has to be 'speedy_isobased'.")


def test_speedy_boat_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["BOAT_TYPE"] = "speedy_isobased"
    config_data["ALGORITHM_TYPE"] = "isofuel"

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "'BOAT_TYPE'='speedy_isobased' can only be used together with "
                                 "'ALGORITHM_TYPE'='genetic_shortest_route' and 'ALGORITHM_TYPE'='speedy_isobased'.")


def test_invalid_route_raises_error():
    config_data, _ = load_example_config()
    config_data["DEFAULT_ROUTE"] = [54.15, 13.15, 54.56, 200]

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "lon_end must be between -180 and 180")


def test_negative_delta_fuel_raises_error():
    config_data, _ = load_example_config()
    config_data["DELTA_FUEL"] = -100
    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "'DELTA_FUEL' must be greater than zero")


def test_non_even_router_hdgs_segments_raises_error():
    config_data, _ = load_example_config()
    config_data["ROUTER_HDGS_SEGMENTS"] = 31
    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "'ROUTER_HDGS_SEGMENTS' must be a positive even integer")


def test_route_map_compatibility():
    config_data, _ = load_example_config()
    config_data["DEFAULT_ROUTE"] = [36.192, 13.392, 41.349, 17.188]
    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "is outside the defined map bounds")


def test_weather_start_time_compatibility():
    config_data, _ = load_example_config()
    config_data["DEPARTURE_TIME"] = "2022-04-01T11:11Z"
    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "Weather data does not cover the full routing time range.")


@pytest.mark.parametrize("boat_speed,arrival_time,mut_type,cross_type,ierr", [
    (7, "2025-12-07T00:00Z", "waypoints", "waypoints", 0),
    (None, None, "waypoints", "waypoints", 1),
    (7, None, "waypoints", "waypoints", 2),
    (None, "2025-12-07T00:00Z", "waypoints", "waypoints", 2),
])
def test_boat_speed_arrival_time_waypoint_optimisation_failure(boat_speed, arrival_time, mut_type, cross_type, ierr):
    config_data, _ = load_example_config()
    config_data["BOAT_SPEED"] = boat_speed
    config_data["ARRIVAL_TIME"] = arrival_time
    config_data["GENETIC_MUTATION_TYPE"] = mut_type
    config_data["GENETIC_CROSSOVER_TYPE"] = cross_type
    config_data["ALGORITHM_TYPE"] = "genetic"
    error_str_list = [
        "Please specify EITHER the boat speed OR the arrival time but not both.",
        "Please specify EITHER the boat speed OR the arrival time.",
        "Optimisation for arrival-time accuracy is meaningless for pure waypoint optimisation."
    ]

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, error_str_list[ierr])


@pytest.mark.parametrize("boat_speed,arrival_time,mut_type,cross_type", [
    (7, None, "waypoints", "waypoints"),
    (None, "2025-12-07T00:00Z", "waypoints", "waypoints"),
])
def test_boat_speed_arrival_time_waypoint_optimisation_success(boat_speed, arrival_time, mut_type, cross_type, ):
    config_data, _ = load_example_config()
    config_data["BOAT_SPEED"] = boat_speed
    config_data["ARRIVAL_TIME"] = arrival_time
    config_data["GENETIC_MUTATION_TYPE"] = mut_type
    config_data["GENETIC_CROSSOVER_TYPE"] = cross_type
    config_data["ALGORITHM_TYPE"] = "genetic"
    config_data["GENETIC_OBJECTIVES"] = {"fuel_consumption": 1.}

    Config.assign_config(init_mode="from_dict", config_dict=config_data)


@pytest.mark.parametrize("boat_speed,arrival_time,mut_type,cross_type", [
    (7, None, "random", "random"),
    (None, "2025-12-07T00:00Z", "random", "random"),
    (None, "2025-12-07T00:00Z", "waypoints", "random"),
    (None, "2025-12-07T00:00Z", "random", "waypoints"),
])
def test_boat_speed_arrival_time_speed_optimisation_failure(boat_speed, arrival_time, mut_type, cross_type, ):
    config_data, _ = load_example_config()
    config_data["BOAT_SPEED"] = boat_speed
    config_data["ARRIVAL_TIME"] = arrival_time
    config_data["GENETIC_MUTATION_TYPE"] = mut_type
    config_data["GENETIC_CROSSOVER_TYPE"] = cross_type
    config_data["ALGORITHM_TYPE"] = "genetic"

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "Please provide a valid arrival time and boat speed.")


@pytest.mark.parametrize("boat_speed,arrival_time,mut_type,cross_type", [
    (7, "2025-12-07T00:00Z", "random", "random"),
    (7, "2025-12-07T00:00Z", "random", "waypoints"),
    (7, "2025-12-07T00:00Z", "waypoints", "random"),
])
def test_boat_speed_arrival_time_speed_optimisation_success(boat_speed, arrival_time, mut_type, cross_type, ):
    config_data, _ = load_example_config()
    config_data["BOAT_SPEED"] = boat_speed
    config_data["ARRIVAL_TIME"] = arrival_time
    config_data["GENETIC_MUTATION_TYPE"] = mut_type
    config_data["GENETIC_CROSSOVER_TYPE"] = cross_type
    config_data["ALGORITHM_TYPE"] = "genetic"

    Config.assign_config(init_mode="from_dict", config_dict=config_data)


def _gcr_travel_time(config_data, boat_speed=6.0):
    """Travel time on the great circle route of DEFAULT_ROUTE at the given boat speed [m/s]."""
    lat_start, lon_start, lat_end, lon_end = config_data["DEFAULT_ROUTE"]
    gcr_distance = float(geod.inverse([lat_start], [lon_start], [lat_end], [lon_end])['s12'][0])
    return timedelta(seconds=gcr_distance / boat_speed)


def test_travel_time_feasibility_success():
    """A time window that exactly matches the great circle travel time must validate."""
    config_data, _ = load_example_config()
    departure = datetime.strptime(config_data["DEPARTURE_TIME"], '%Y-%m-%dT%H:%MZ')
    # Available time equals the minimum required travel time -> still feasible.
    config_data["ARRIVAL_TIME"] = departure + _gcr_travel_time(config_data)

    config = Config.assign_config(init_mode="from_dict", config_dict=config_data)
    assert isinstance(config, Config)


def test_travel_time_feasibility_failure():
    """A time window shorter than the great circle travel time must raise an error."""
    config_data, _ = load_example_config()
    departure = datetime.strptime(config_data["DEPARTURE_TIME"], '%Y-%m-%dT%H:%MZ')
    # Only half of the required travel time is available -> infeasible.
    config_data["ARRIVAL_TIME"] = departure + _gcr_travel_time(config_data) / 2

    with pytest.raises(WRTConfigError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert_config_error(excinfo, "cannot be travelled within the configured time window")
