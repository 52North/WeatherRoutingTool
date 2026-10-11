import argparse
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.execute_routing import execute_routing
from WeatherRoutingTool.ship.ship_config import ShipConfig


def run_genetic_example(config_file=None):
    """
    Run an example route optimization using the genetic algorithm.

    :param config_file: Path to configuration JSON file.
    """
    if config_file is None:
        default_config = os.path.join(project_root, 'tests', 'config.regression.json')
        template_config = os.path.join(project_root, 'config.template.json')
        config_file = default_config if os.path.exists(default_config) else template_config

    print(f"Loading configuration from: {config_file}")
    set_up_logging()

    config = Config.assign_config(config_file)
    ship_config = ShipConfig.assign_config(config_file)

    execute_routing(config, ship_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm routing example.")
    parser.add_argument(
        "-f", "--file",
        help="Path to configuration JSON file (defaults to tests/config.regression.json).",
        default=None,
    )
    args = parser.parse_args()
    run_genetic_example(args.file)

