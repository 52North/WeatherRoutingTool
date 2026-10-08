# Weather Routing Tool (WRT)

A tool to perform optimization of ship routes based on fuel consumption in different weather conditions.

## Installation

### Minimal Installation (Core Features)
For basic weather routing functionality without visualization or advanced features:

```bash
pip install WeatherRoutingTool
```

### Installation with Optional Features

#### With Visualization Support
For plotting and visualization capabilities:
```bash
pip install WeatherRoutingTool[viz]
```

#### With Geospatial Features
For advanced geospatial analysis:
```bash
pip install WeatherRoutingTool[geospatial]
```

#### With Genetic Algorithm
For optimization using genetic algorithms:
```bash
pip install WeatherRoutingTool[genetic]
```

#### With Data Processing
For large dataset processing:
```bash
pip install WeatherRoutingTool[data]
```

#### With External Data Download
For downloading weather data:
```bash
pip install WeatherRoutingTool[download]
```

#### Full Installation
For all features (equivalent to current installation):
```bash
pip install WeatherRoutingTool[all]
```

#### Development Installation
For contributors and developers:
```bash
pip install WeatherRoutingTool[dev]
```

### Dependency Groups

- **Core**: `numpy`, `pandas`, `xarray`, `scipy`, `pydantic`, `astropy` - Essential for basic functionality
- **Visualization**: `matplotlib`, `seaborn`, `cartopy` - Plotting and mapping
- **Geospatial**: `geopandas`, `shapely`, `geographiclib`, `geovectorslib`, `global_land_mask` - Advanced geographic features
- **Data Processing**: `dask`, `datacube`, `netcdf4` - Large dataset handling
- **Download**: `maridatadownloader` - External weather data downloading
- **Genetic**: `pymoo` - Genetic algorithm optimization
- **Image**: `scikit-image`, `Pillow` - Image processing

Documentation: https://52north.github.io/WeatherRoutingTool/

Introduction: [WRT-sandbox](https://github.com/52North/WRT-sandbox) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/52North/WRT-sandbox.git/HEAD?urlpath=%2Fdoc%2Ftree%2FNotebooks/execute-WRT.ipynb)

## Funding

|                                                                                      Project/Logo                                                                                      | Description                                                                                                                                                                                                                                                                                 |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [<img alt="MariData" align="middle" width="267" height="50" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/maridata_logo.png"/>](https://www.maridata.org/) | MariGeoRoute is funded by the German Federal Ministry of Economic Affairs and Energy (BMWi)[<img alt="BMWi" align="middle" width="144" height="72" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/bmwi_logo_en.png" style="float:right"/>](https://www.bmvi.de/) |
|               [<img alt="TwinShip" align="middle" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/twinship_logo.png"/>](https://twin-ship.eu/)               | Co-funded by the European Union’s Horizon Europe programme under grant agreement No. 101192583                                                                                                                                                                                              |
