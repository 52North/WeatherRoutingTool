# Weather Routing Tool (WRT)

[![CI](https://github.com/52North/WeatherRoutingTool/workflows/CI/badge.svg)](https://github.com/52North/WeatherRoutingTool/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python tool for optimizing ship routes based on fuel consumption in different weather conditions.

**Documentation:** https://52north.github.io/WeatherRoutingTool/

**Introduction:** [WRT-sandbox](https://github.com/52North/WRT-sandbox) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/52North/WRT-sandbox.git/HEAD?urlpath=%2Fdoc%2Ftree%2FNotebooks/execute-WRT.ipynb)

## Features

- **Multiple routing algorithms**: Isofuel, genetic algorithms, and isochrone-based methods
- **Weather integration**: Uses real-time weather data (wind, waves, currents) for route optimization
- **Constraint handling**: Avoid land, shallow waters, restricted areas, and more
- **Multiple ship models**: Support for different vessel types and power models
- **Flexible configuration**: JSON-based configuration with environment variable overrides
- **CLI & API**: Use as a command-line tool or integrate into your Python projects

## Installation

### Requirements

- Python 3.11 or higher
- System dependencies:
  - **Linux**: `libgeos-dev`, `libproj-dev`, `proj-data`, `proj-bin`
  - **macOS**: `geos`, `proj` (via Homebrew)
  - **Windows**: Usually handled automatically by pip

### Install from source

```bash
# Clone the repository
git clone https://github.com/52North/WeatherRoutingTool.git
cd WeatherRoutingTool

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install via pip (when published)

```bash
pip install WeatherRoutingTool
```

## Quick Start

### 1. Create a configuration file

Create a `config.json` file with your routing parameters:

```json
{
  "ALGORITHM_TYPE": "isofuel",
  "BOAT_TYPE": "direct_power_method",
  "CONSTRAINTS_LIST": ["land_crossing_global_land_mask", "on_map"],
  "DEFAULT_ROUTE": [50.0, 10.0, 52.0, 15.0],
  "DEFAULT_MAP": [49.0, 9.0, 53.0, 16.0],
  "DEPARTURE_TIME": "2024-01-15T12:00Z",
  "TIME_FORECAST": 72,
  "WEATHER_DATA": "path/to/weather.nc",
  "DEPTH_DATA": "path/to/depth.nc",
  "ROUTE_PATH": "./output"
}
```

### 2. Run the routing tool

#### Using the CLI

```bash
# Basic usage
wrt --config config.json

# With verbose logging
wrt --config config.json --verbose

# Save logs to file
wrt --config config.json --log-file routing.log
```

#### Using Python API

```python
from pathlib import Path
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.routingalg_factory import RoutingAlgFactory
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory
from WeatherRoutingTool.utils.maps import Map

# Load configuration
config = Config.assign_config(path=Path("config.json"), init_mode='from_json')

# Initialize components
default_map = Map(*config.DEFAULT_MAP)
weather = WeatherFactory.get_weather(
    data_mode=config._DATA_MODE_WEATHER,
    weatherfile=config.WEATHER_DATA,
    departure_time=config.DEPARTURE_TIME,
    time_forecast=config.TIME_FORECAST,
    delta_time_forecast=config.DELTA_TIME_FORECAST,
    default_map=default_map
)

constraints = ConstraintsListFactory.get_constraints_list(
    config.CONSTRAINTS_LIST,
    map_size=default_map
)

boat = ShipFactory.get_boat(config.BOAT_TYPE, config)
algorithm = RoutingAlgFactory.get_algorithm(config.ALGORITHM_TYPE, config)

# Execute routing
algorithm.execute_routing(boat, weather, constraints, verbose=True)
algorithm.terminate()
```

### 3. View results

The optimized route will be saved as a GeoJSON file in the directory specified by `ROUTE_PATH`.

## Configuration

See the [full configuration documentation](https://52north.github.io/WeatherRoutingTool/configuration.html) for all available options.

### Environment Variables

You can override configuration values using environment variables with the `WRT_` prefix:

```bash
export WRT_ALGORITHM_TYPE=genetic
export WRT_ROUTE_PATH=/path/to/output
export WRT_WEATHER_DATA=/path/to/weather.nc
```

## Development

### Setup development environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Run tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=WeatherRoutingTool --cov-report=html

# Run specific test markers
pytest -m genetic
pytest -m maripower
```

### Code quality

```bash
# Format code
black --line-length=120 .
ruff --line-length=120 --fix .

# Type checking
mypy WeatherRoutingTool/

# Run all pre-commit checks
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Funding

|                                                                                      Project/Logo                                                                                      | Description                                                                                                                                                                                                                                                                                 |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [<img alt="MariData" align="middle" width="267" height="50" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/maridata_logo.png"/>](https://www.maridata.org/) | MariGeoRoute is funded by the German Federal Ministry of Economic Affairs and Energy (BMWi)[<img alt="BMWi" align="middle" width="144" height="72" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/bmwi_logo_en.png" style="float:right"/>](https://www.bmvi.de/) |
|               [<img alt="TwinShip" align="middle" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/twinship_logo.png"/>](https://twin-ship.eu/)               | Co-funded by the European Union’s Horizon Europe programme under grant agreement No. 101192583                                                                                                                                                                                              |
