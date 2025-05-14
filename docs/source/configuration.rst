.. _configuration:

Configuration
=============

Configuration of the Weather Routing Tool can be done by providing a json file. An example is given by `config.example.json`.

The configuration file has to be provided when calling the Weather Routing Tool from the command line:


.. code-block:: shell

    python3 WeatherRoutingTool/cli.py -f <path>/config.json

Additionally, it's possible to define files for logging (separately for info and warning level) and if debugging mode should be used.
Check the help text to get an overview of all CLI arguments:

.. code-block:: shell

    $ python WeatherRoutingTool/cli.py --help
    usage: cli.py [-h] -f FILE [--warnings-log-file WARNINGS_LOG_FILE] [--info-log-file INFO_LOG_FILE] [--debug DEBUG] [--filter-warnings FILTER_WARNINGS]

    Weather Routing Tool

    options:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Config file name (absolute path)
      --warnings-log-file WARNINGS_LOG_FILE
                            Logging file name (absolute path) for warnings and above.
      --info-log-file INFO_LOG_FILE
                            Logging file name (absolute path) for info and above.
      --debug DEBUG         Enable debug mode. <True|False>. Defaults to 'False'.
      --filter-warnings FILTER_WARNINGS
                            Filter action. <default|error|ignore|always|module|once>.Defaults to 'default'.

Some variables have to be set using environment variables (see below).

Config file
-----------

The following lists contain information on each variable which can be set.

**Required variables** (no default values provided):

- ``COURSES_FILE``: path to file that acts as intermediate storage for courses per routing step
- ``DEFAULT_MAP``: bbox in which route optimization is performed (lat_min, lon_min, lat_max, lon_max)
- ``DEFAULT_ROUTE``: start and end point of the route (lat_start, lon_start, lat_end, lon_end)
- ``DEPARTURE_TIME``: start time of travelling, format: 'yyyy-mm-ddThh:mmZ'
- ``DEPTH_DATA``: path to depth data (Attention: if ``DATA_MODE`` is ``automatic`` or ``odc``, this file will be overwritten!)
- ``ROUTE_PATH``: path to json file to which the route will be written
- ``WEATHER_DATA``: path to weather data (Attention: if ``DATA_MODE`` is ``automatic`` or ``odc``, this file will be overwritten!)

**Recommended variables** (default values provided but might be inaccurate/unsuitable):

- ``BOAT_DRAUGHT_AFT``: aft draught (draught at rudder) in m
- ``BOAT_DRAUGHT_FORE``: fore draught (draught at forward perpendicular) in m
- ``BOAT_ROUGHNESS_DISTRIBUTION_LEVEL``: numeric value (default: 1)
- ``BOAT_ROUGHNESS_LEVEL``: numeric value (default: 1)
- ``BOAT_SPEED``: in m/s
- ``DATA_MODE``: options: 'automatic', 'from_file', 'odc'

**Optional variables** (default values provided and don't need to be changed normally):

- ``ALGORITHM_TYPE``: options: 'isofuel'
- ``CONSTRAINTS_LIST``: options: 'land_crossing_global_land_mask', 'land_crossing_polygons', 'seamarks', 'water_depth', 'on_map', 'via_waypoints', 'status_error'
- ``DELTA_FUEL``: amount of fuel per routing step (kg)
- ``DELTA_TIME_FORECAST``: time resolution of weather forecast (hours)
- ``FACTOR_CALM_WATER``: multiplication factor for the calm water resistance model
- ``FACTOR_WAVE_FORCES``: multiplication factor for the added resistance in waves model
- ``FACTOR_WIND_FORCES``: multiplication factor for the added resistance in wind model
- ``GENETIC_MUTATION_TYPE``: type for mutation (options: 'grid_based')
- ``GENETIC_NUMBER_GENERATIONS``: number of generations for genetic algorithm
- ``GENETIC_NUMBER_OFFSPRINGS``: number of offsprings for genetic algorithm
- ``GENETIC_POPULATION_SIZE``: population size for genetic algorithm
- ``GENETIC_POPULATION_TYPE``: type for initial population (options: 'grid_based', 'from_geojson')
- ``INTERMEDIATE_WAYPOINTS``: [[lat_one,lon_one], [lat_two,lon_two] ... ]
- ``ISOCHRONE_MAX_ROUTING_STEPS``: maximum number of routing steps. Applies also if more than one route is searched!
- ``ISOCHRONE_MINIMISATION_CRITERION``: options: 'dist', 'squareddist_over_disttodest'
- ``ISOCHRONE_NUMBER_OF_ROUTES``: integer specifying how many routes should be searched (default: 1)
- ``ISOCHRONE_PRUNE_GROUPS``: can be 'courses', 'larger_direction', 'branch'
- ``ISOCHRONE_PRUNE_SECTOR_DEG_HALF``: half of the angular range of azimuth angle considered for pruning; not used for branch-based pruning
- ``ISOCHRONE_PRUNE_SEGMENTS``: total number of azimuth bins used for pruning in prune sector; not used for branch-based pruning
- ``ISOCHRONE_PRUNE_SYMMETRY_AXIS``: symmetry axis for pruning. Can be 'gcr' or 'headings_based'; not used for branch-based pruning
- ``ROUTER_HDGS_INCREMENTS_DEG``: increment of headings
- ``ROUTER_HDGS_SEGMENTS``: total number of headings (put even number!!); headings are oriented around the great circle from current point to (temporary - i.e. next waypoint if used) destination
- ``ROUTE_POSTPROCESSING``: enable route postprocessing to follow the Traffic Separation Scheme in route postprocessing
- ``SHIP_TYPE``: options: 'CBT', 'SAL'
- ``TIME_FORECAST``: forecast hours weather

Environment variables
---------------------

Credentials for the Copernicus Marine Environment Monitoring Service (CMEMS) to download weather/ocean data:

- ``CMEMS_USERNAME``
- ``CMEMS_PASSWORD``

If not provided ``DATA_MODE='automatic'`` cannot be used.

Configuration parameters for the database which stores OpenSeaMap data:

- ``WRT_DB_HOST``
- ``WRT_DB_PORT``
- ``WRT_DB_DATABASE``
- ``WRT_DB_USERNAME``
- ``WRT_DB_PASSWORD``

If not provided the 'land_crossing_polygons' and 'seamarks' options of ``CONSTRAINTS_LIST`` and ``ROUTE_POSTPROCESSING=True`` cannot be used.

Path for storing figures (mainly for debugging purposes):

- ``WRT_FIGURE_PATH``

If not set or the path doesn't exist or access rights are wrong, no figures will be saved.

You can define the environment variables in a separate .env file and call the provided shell script:

.. code-block:: shell

    source <path-to-WRT>/load_wrt.sh

Logging and Debugging
---------------------

All log messages are sent to stdout by default. In addition, info and warning logs can be saved separately to file.
Debugging mode can be enabled (disabled by default) which sets the stream (stdout) logging level to debug.

The top-level logger is named "WRT". Child loggers are following the scheme "WRT.<child-name>".
They inherit the top-level loggers' logging level.

Input data
----------

Depending on the power/fuel consumption model used, different sets of environmental data are needed. The data described below are needed for the usage of **mariPower**.

There are three general options on how to provide the necessary input data:

1. The easiest option is to set the config parameter ``DATA_MODE='automatic'``. To use it, valid CMEMS credentials have to be configured using system environment variables (see above). In this case, the WRT will automatically download the necessary weather and ocean data for the chosen temporal and spatial extent and store it in the file specified by the config variable ``WEATHER_DATA``. Moreover, water depth data from [NOAA](https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/30s/30s_bed_elev_netcdf/catalog.html?dataset=globalDatasetScan/ETOPO2022/30s/30s_bed_elev_netcdf/ETOPO_2022_v1_30s_N90W180_bed.nc) is downloaded and stored in the file specified by the config variable ``DEPTH_DATA``.

2. It is also possible to prepare two NetCDF files containing the weather and ocean data and the water depth data and pointing the WRT to these files using the same config variables as before. To do so set ``DATA_MODE='from_file'``. Be sure the temporal and spatial extent is consistent with the other config variables. The `maridatadownloader <https://github.com/52North/maridatadownloader>`_ - which is used by the WRT - can facilitate the preparation.

3. A third option is to set up an `Open Data Cube (ODC) <https://www.opendatacube.org/>`_ instance. To use it set ``DATA_MODE='odc'``. In this case, the data will be extracted from ODC and also stored in the two files as described before.

Be sure that the water depth data is available and configured correctly in order to use the ``water_depth`` option of ``CONSTRAINTS_LIST``.

The following parameters are downloaded automatically or need to be prepared:

- u-component_of_wind_height_above_ground (u-component of wind @ Specified height level above ground)
- v-component_of_wind_height_above_ground (v-component of wind @ Specified height level above ground)
- vtotal (Northward total velocity: Eulerian + Waves + Tide)
- utotal (Eastward total velocity: Eulerian + Waves + Tide)
- VHMO (spectral significant wave height @ sea surface)
- VMDR (mean wave direction @ sea surface)
- VTPK (wave period at spectral peak)
- thetao (potential temperature)
- Pressure_reduced_to_MSL_msl (pressure reduced to mean sea level)
- Temperature_surface (temperature at the water surface)
- so (salinity)

.. figure:: /_static/sequence_diagram_installation_workflow.png
   :alt: sequence_diagram_installation_workflow

   Fig. 1: Basic installation workflow for the WeatherRoutingTool.
