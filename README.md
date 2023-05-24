# ISOCHRONE CODE


## Installation instructions
<ol>
  <li> 
The routing tool can be installed by doing  

```sh
pip install . 
```

in the directory that contains the file setup.py. The installation requires the separate installation of the
package mariPower. All other dependencies are installed automatically.
  </li>
  <li>
    For standalone execution, download weather data for the required time period from [here](https://maridata.dev.52north.org/EnvDataAPI/) in netCDF format. The parameters that need to be selected for the routing procedure are the following:
    <ul>
      <li> u-component_of_wind_sigma (u-component of wind @ sigma level) </li>
      <li> v-component_of_wind_sigma (v-component of wind @ sigma level) </li>
      <li> vo (northward velocity) </li>
      <li> uo (eastward velocity) </li>
      <li> VHMO (wave significant height @ sea surface)</li>
      <li> VMDR (wave direction @ sea surface)</li>
      <li> thetao (potential temperature) </li>
      <li> Pressure_surface (pressure at the water surface) </li>
      <li> so (salinity) </li>
    </ul>
  </li>
  <li>
    For standalone execution, download data on the water depth from [here](https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/30s/30s_bed_elev_netcdf/catalog.html?dataset=globalDatasetScan/ETOPO2022/30s/30s_bed_elev_netcdf/ETOPO_2022_v1_30s_N90W180_bed.nc).
  </li>
  <li> 
    Define the environment variables which are read by config.py in the sections 'File paths' and 'Boat settings' (e.g. in a separate .env file)
  </li>
  <li> 
    Adjust the start and endpoint of the route as well as the departure time using the variables 'DEFAULT_ROUTE' and 'START_TIME'. The variable 'DEFAULT_MAP' needs to be set to 
    a map size that encompasses the final route. The boat speed and drought can be configured via the variables 'BOAT_SPEED' and 'BOAT_DROUGHT'.
  </li>
  <li>
    Initiate the routing procedure by executing the file 'execute_routing.py': 

```sh
python execute_routing.py 
```
  </li>
</ol>

## Logging
The routing tool writes log output using the python package logging. Information about basic settings are written to a file which is specified by the environment variable 'INFO_LOG_FILE'. Warnings and performance information are
written to the file which is specified by the environment variable 'PERFORMANCE_LOG_FILE'. Further debug information are written to stdout.

## References
- https://github.com/omdv/wind-router
- [Henry H.T. Chen's PhD Thesis](http://resolver.tudelft.nl/uuid:a6112879-4298-40a6-91c7-d9a431a674c7)
- Modeling and Optimization Algorithms in Ship Weather Routing, doi:10.1016/j.enavi.2016.06.004
- Optimal Ship Weather Routing Using Isochrone Method on the Basis of Weather Changes, doi:10.1061/40932(246)435
- Karin, Todd. Global Land Mask. October 5, 2020. doi:10.5281/zenodo.4066722
- [GFS grib2 filter](https://nomads.ncep.noaa.gov/)
- [Boat polars - 1](https://jieter.github.io/orc-data/site/)
- [Boat polars - 2](https://l-36.com/polar_polars.php)
- https://en.wikisource.org/wiki/The_American_Practical_Navigator/Chapter_1
- https://gist.github.com/jeromer/2005586
- https://towardsdatascience.com/calculating-the-bearing-between-two-geospatial-coordinates-66203f57e4b4
- https://www.youtube.com/watch?v=DeFZ6AHtYxg
- https://www.movable-type.co.uk/scripts/latlong.html
- https://gis.stackexchange.com/questions/425515/converting-between-lat-long-azimuth-and-distance-heading
- https://geopy.readthedocs.io/en/stable/
- https://www.siranah.de/html/sail020f.html
- https://github.com/hakola/marine-traffic-modelling
- http://www.movable-type.co.uk/scripts/latlong.html?from=48.955550,-122.05169&to=48.965496,-122.072989
- https://geographiclib.sourceforge.io/html/python/code.html#geographiclib.geodesic.Geodesic.Inverse
- https://mathsathome.com/calculating-bearings/ 
## Funding

| Project/Logo | Description |
| :-------------: | :------------- |
| [<img alt="MariData" align="middle" width="267" height="50" src="https://52north.org/delivery/MariData/img/maridata_logo.png"/>](https://www.maridata.org/) | MariGeoRoute is funded by the German Federal Ministry of Economic Affairs and Energy (BMWi)[<img alt="BMWi" align="middle" width="144" height="72" src="https://52north.org/delivery/MariData/img/bmwi_logo_en.png" style="float:right"/>](https://www.bmvi.de/) |
