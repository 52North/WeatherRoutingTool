import logging
import re
from pathlib import Path

import numpy as np
import pytest

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.execute_routing import execute_routing
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship_config import ShipConfig

DATA_DIR = Path(__file__).parent / "data"
CONFIG_PATH = Path(__file__).parent / "config.regression.json"
GOLDEN_ROUTE_PATH = DATA_DIR / "regression_min_fuel_route.json"
GOLDEN_LOG_PATH = DATA_DIR / "regression_log.txt"

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", re.MULTILINE)
DURATION_RE = re.compile(r"Time after minimisation: [0-9.]+")


def _normalize_log(text, route_path):
    """
    Strip everything from the log that is expected to differ between two
    otherwise-identical runs (timestamps, the tmp_path used for this run, and
    the wall-clock optimisation duration) so only behaviourally relevant log
    content is compared. A raw byte-for-byte diff of a timestamped log with an
    absolute tmp path baked in would make this test permanently flaky.
    """
    text = TIMESTAMP_RE.sub("", text)
    text = text.replace(str(route_path), "<ROUTE_PATH>")
    text = DURATION_RE.sub("Time after minimisation: <duration>", text)
    return text


def _assert_routes_match(actual: RouteParams, expected: RouteParams):
    assert actual.count == expected.count
    assert actual.start == expected.start
    assert actual.finish == expected.finish

    np.testing.assert_allclose(actual.lats_per_step, expected.lats_per_step, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.lons_per_step, expected.lons_per_step, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.course_per_step, expected.course_per_step, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual.dists_per_step, expected.dists_per_step, rtol=1e-4, atol=1.0)

    actual_sp = actual.ship_params_per_step
    expected_sp = expected.ship_params_per_step
    np.testing.assert_allclose(
        actual_sp.get_fuel_rate().value, expected_sp.get_fuel_rate().value, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        actual_sp.get_power().value, expected_sp.get_power().value, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(
        actual_sp.get_speed().value, expected_sp.get_speed().value, rtol=1e-4, atol=1e-3)


@pytest.mark.regression
def test_genetic_algorithm_regression(tmp_path):
    """
    End-to-end regression test for the genetic routing algorithm.

    Runs a small, deterministic (fixed GENETIC_RANDOM_SEED) genetic-algorithm
    optimisation against the existing Mediterranean test fixtures
    (tests/data/tests_weather_data.nc, tests_depth_data.nc) and compares the
    resulting route and log output against checked-in golden files. This
    exercises the full pipeline (population, crossover, mutation, repair,
    fuel/power model, constraints) end-to-end, rather than each part in
    isolation, so it catches regressions that unit tests would miss.

    Output is written to pytest's tmp_path so the run leaves nothing behind
    in the repository.
    """
    config = Config.assign_config(CONFIG_PATH)
    config.ROUTE_PATH = tmp_path
    ship_config = ShipConfig.assign_config(CONFIG_PATH)

    log_path = tmp_path / "info.log"
    set_up_logging(info_log_file=str(log_path))
    # set_up_logging() relies on logging.basicConfig() to raise the effective
    # level of the root logger (and via inheritance, 'WRT') to INFO. Under
    # pytest the root logger already has a handler (pytest's own log
    # capture), so basicConfig() is a no-op and INFO records are silently
    # dropped. Set the root level explicitly instead of the 'WRT' logger's
    # own level, so that WeatherRoutingTool.algorithms.genetic.patcher's
    # "quiet mode" (which temporarily raises the *root* logger's level to
    # ERROR while running nested Isofuel sub-routes for population sampling
    # and repair) still works the same way it does outside of pytest.
    logging.getLogger().setLevel(logging.INFO)

    execute_routing(config, ship_config)

    # Read the log now, before RouteParams.from_file() below adds its own
    # "Reading N coordinate pairs from file" INFO records to the same file.

    print('tmp_path: ', tmp_path)
    actual_log = _normalize_log(log_path.read_text(), tmp_path)
    (tmp_path / "actual_log.txt").write_text(actual_log)

    actual_route_path = tmp_path / "min_fuel_route.json"
    assert actual_route_path.exists(), "genetic algorithm did not produce a route file"

    actual_route = RouteParams.from_file(actual_route_path)
    expected_route = RouteParams.from_file(GOLDEN_ROUTE_PATH)
    _assert_routes_match(actual_route, expected_route)

    expected_log = GOLDEN_LOG_PATH.read_text()
    assert actual_log == expected_log
