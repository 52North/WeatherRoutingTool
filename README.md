# Weather Routing Tool (WRT)

A tool to perform optimization of ship routes based on fuel consumption in different weather conditions.

Documentation: https://52north.github.io/WeatherRoutingTool/

Introduction: [WRT-sandbox](https://github.com/52North/WRT-sandbox) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/52North/WRT-sandbox.git/HEAD?urlpath=%2Fdoc%2Ftree%2FNotebooks/execute-WRT.ipynb)

 ## Known Issues and Troubleshooting

### Speedy Isobased Routing Stops Early with Test Data

When running the Weather Routing Tool using the `speedy_isobased` algorithm together with the provided test weather and depth datasets, the routing process may terminate before reaching the destination.

**Observed behavior:**
- Routing initializes correctly and progresses through multiple isochrone steps
- Warnings such as *"More than 50% of pruning segments constrained"* may appear
- The routing eventually aborts with messages indicating that all pruning segments are constrained
- A partial route output may be generated, but no complete route reaches the destination

**Why this happens:**
This can occur due to a combination of strict pruning settings and routing constraints (such as land crossing, water depth, and map bounds) when applied to the limited resolution of the included test datasets. As a result, all possible routing branches may become constrained early in the optimization process.

**What you can try:**
- Relax pruning parameters (for example, reduce the number of prune segments or adjust pruning sector angles)
- Temporarily disable or relax certain constraints for testing
- Use higher-resolution or alternative weather and depth datasets
- Increase the maximum number of allowed routing steps

Related issue: #105

## Funding

|                                                                                      Project/Logo                                                                                      | Description                                                                                                                                                                                                                                                                                 |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [<img alt="MariData" align="middle" width="267" height="50" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/maridata_logo.png"/>](https://www.maridata.org/) | MariGeoRoute is funded by the German Federal Ministry of Economic Affairs and Energy (BMWi)[<img alt="BMWi" align="middle" width="144" height="72" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/bmwi_logo_en.png" style="float:right"/>](https://www.bmvi.de/) |
|               [<img alt="TwinShip" align="middle" src="https://github.com/52North/WeatherRoutingTool/blob/main/docs/_static/twinship_logo.png"/>](https://twin-ship.eu/)               | Co-funded by the European Union’s Horizon Europe programme under grant agreement No. 101192583                                                                                                                                                                                              |
