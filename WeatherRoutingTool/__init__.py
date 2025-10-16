"""Weather Routing Tool - Ship route optimization under weather conditions."""

from WeatherRoutingTool._version import __version__, __version_info__

import WeatherRoutingTool.algorithms
import WeatherRoutingTool.config
import WeatherRoutingTool.constraints
import WeatherRoutingTool.routeparams
import WeatherRoutingTool.ship
import WeatherRoutingTool.utils
import WeatherRoutingTool.weather

__all__ = [
    "__version__",
    "__version_info__",
    "algorithms",
    "config",
    "constraints",
    "routeparams",
    "ship",
    "utils",
    "weather",
]
