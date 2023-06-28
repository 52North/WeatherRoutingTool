# ISOCHRONE CODE

## To Do
- [ ] add figures for definition of basic variables
- [ ] improve the definition of heading/azimuthal angle
- [ ] sketch the basic steps of the routing procedure

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
      <li> u-component_of_wind_height_above_ground (u-component of wind @ Specified height level above ground) </li>
      <li> v-component_of_wind_height_above_ground (v-component of wind @ Specified height level above ground) </li>
      <li> vo (northward velocity) </li>
      <li> uo (eastward velocity) </li>
      <li> VHMO (wave significant height @ sea surface)</li>
      <li> VMDR (wave direction @ sea surface)</li>
      <li> thetao (potential temperature) </li>
      <li> Pressure_surface (pressure at the water surface) </li>
      <li> Temperature_surface (temperature at the water surface) </li>
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

## Isochrone Algorithm
### Parameter definitions for configuration
pruning = the process of chosing the route that maximises the distance for one routing segment
heading/course/azimuth/variants = the angular distance towards North on the grand circle route 
route segment = distance of one starting point of the routing step to one end point

ISOCHRONE_PRUNE_SEGMENTS = number of segments that are used for the pruning process
ISOCHRONE_PRUNE_SECTOR_DEG_HALF = angular range of azimuth angle that is considered for pruning (only one half of it!)
ROUTER_HDGS_SEGMENTS = total number of courses/azimuths/headings that are considered per coordinate pair for every routing step
ROUTER_HDGS_INCREMENTS_DEG = angular distance between two adjacent routing segments


### Variable definitions
lats_per_step: (M,N) array of latitudes for different routes (shape N=headings+1) and routing steps (shape M=steps,decreasing)
lons_per_step: (M,N) array of longitude for different routes (shape N=headings+1) and routing steps (shape M=steps,decreasing)

### Communication between mariPower and the WRT
Information is transfered via a netCDF file between the WRT and mariPower. The coordinate pairs, courses, the ship speed and the time for which the power estimation needs to be performed are written to this file by the WRT. This information is read by mariPower, the calculation of the ship parameters is performed and the corresponding results are added as separate variables to the xarray dataset. The structure of the xarray dataset after the ship parameters have been written is the following:

```sh
Dimensions:                    (it_pos: 2, it_course: 3)
Coordinates:
   * it_pos                    (it_pos) int64 1 2
   * it_course                 (it_course) int64 1 2 3 
Data variables:
    courses                    (it_pos, it_course) float64 ...
    speed                      (it_pos, it_course) int64 ...
    lat                        (it_pos) float64 ...
    lon                        (it_pos) float64 ...
    time                       (it_pos) datetime64[ns] ...
    Power_delivered            (it_pos, it_course) float64 ...
    RotationRate               (it_pos, it_course) float64 ...
    Fuel_consumption_rate      (it_pos, it_course) float64 ...
    Calm_resistance            (it_pos, it_course) float64 ...
    Wind_resistance            (it_pos, it_course) float64 ...
    Wave_resistance            (it_pos, it_course) float64 ...
    Shallow_water_resistance   (it_pos, it_course) float64 ...
    Hull_roughness_resistance  (it_pos, it_course) float64 ...
```

The coordinates ``` it_pos ``` and ```it_course ``` are iterators for the coordinate pairs and the courses that need to be checked per coordinate pair, respectively. The function in the WRT that writes the route parameters to the netCDF file is called ``` ship.write_netCDF_courses ```. Following up on this, the function ``` get_fuel_netCDF``` in the WRT calls the function ``` PredictPowerOrSpeedRoute ``` in mariPower which itself initiates the calcualation of the ship parameters. The netCDF file is overwritten by the WRT for every routing step s.t. the size of the file is not increasing during the routing process. 

Both for the isofuel algorithm and the genetic algorithm the same structure of the netCDF file is used. However, due to the different concepts of the algorithms, the entity of points that is send for calculation in one request differes between both algorithms. For the isofuel algorithm, all coordinate pairs and courses that are considered for a single routing step are passed to mariPower in a single request (see Fig. XXX). For the genetic algorithm all points and courses for a closed route are passed in a single request (see Fig. XXX).

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
