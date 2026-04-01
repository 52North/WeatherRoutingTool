import json
import pytest
from pathlib import Path
from WeatherRoutingTool.config import Config


def load_example_config():
    config_path = Path(__file__).parent / "config.tests.json"
    with config_path.open("r") as f:
        return json.load(f), config_path


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
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'DEPARTURE_TIME/ARRIVAL_TIME' must be in format YYYY-MM-DDTHH:MMZ" in str(excinfo.value)


def test_invalid_path_raises_error(tmp_path):
    config_data, _ = load_example_config()
    config_data["ROUTE_PATH"] = str(tmp_path / "nonexistent.nc")

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "Path doesn't exist" in str(excinfo.value)


def test_speedy_alg_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["ALGORITHM_TYPE"] = "speedy_isobased"

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "If 'ALGORITHM_TYPE' is 'speedy_isobased', 'BOAT_TYPE' has to be 'speedy_isobased'." in str(excinfo.value)


def test_genetic_shortest_route_boat_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["ALGORITHM_TYPE"] = "genetic_shortest_route"

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "If 'ALGORITHM_TYPE' is 'genetic_shortest_route', 'BOAT_TYPE' has to be 'speedy_isobased'." in str(
        excinfo.value)


def test_speedy_boat_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["BOAT_TYPE"] = "speedy_isobased"
    config_data["ALGORITHM_TYPE"] = "isofuel"

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'BOAT_TYPE'='speedy_isobased' can only be used together with "
    "'ALGORITHM_TYPE'='genetic_shortest_route' and 'ALGORITHM_TYPE'='speedy_isobased'." in str(excinfo.value)


def test_invalid_route_raises_error():
    config_data, _ = load_example_config()
    config_data["DEFAULT_ROUTE"] = [54.15, 13.15, 54.56, 200]

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "lon_end must be between -180 and 180" in str(excinfo.value)


def test_negative_delta_fuel_raises_error():
    config_data, _ = load_example_config()
    config_data["DELTA_FUEL"] = -100
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'DELTA_FUEL' must be greater than zero" in str(excinfo.value)


def test_non_even_router_hdgs_segments_raises_error():
    config_data, _ = load_example_config()
    config_data["ROUTER_HDGS_SEGMENTS"] = 31
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'ROUTER_HDGS_SEGMENTS' must be a positive even integer" in str(excinfo.value)


def test_route_map_compatibility():
    config_data, _ = load_example_config()
    config_data["DEFAULT_ROUTE"] = [36.192, 13.392, 41.349, 17.188]
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "is outside the defined map bounds" in str(excinfo.value)


def test_weather_start_time_compatibility():
    config_data, _ = load_example_config()
    config_data["DEPARTURE_TIME"] = "2022-04-01T11:11Z"
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "Weather data does not cover the full routing time range." in str(excinfo.value)


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

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert error_str_list[ierr] in str(excinfo.value)


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
    config_data["GENETIC_OBJECTIVES"] = {"fuel_consumption" : 1.}

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

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "Please provide a valid arrival time and boat speed." in str(excinfo.value)


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
