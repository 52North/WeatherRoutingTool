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

The following lists contain information on each variable which can be set. The categorisation into required, recommended and optional variables has been chosen such that the requirements of the default settings for the algorithm type (Isofuel algorithm) and the fuel consumption model (Direct Power Method) are met. 

**Required variables** (no default values provided):

- ``DEFAULT_MAP``: bbox in which route optimization is performed (lat_min, lon_min, lat_max, lon_max)
- ``DEFAULT_ROUTE``: start and end point of the route (lat_start, lon_start, lat_end, lon_end)
- ``DEPARTURE_TIME``: start time of travelling, format: 'yyyy-mm-ddThh:mmZ'
- ``DEPTH_DATA``: path to depth data (Attention: if ``DATA_MODE`` is ``automatic`` or ``odc``, this file will be overwritten!)
- ``ROUTE_PATH``: path to json file to which the route will be written
- ``WEATHER_DATA``: path to weather data (Attention: if ``DATA_MODE`` is ``automatic`` or ``odc``, this file will be overwritten!)
- ``BOAT_BREADTH``: ship breadth (m)
- ``BOAT_FUEL_RATE``: fuel rate at service propulsion point (g/kWh)
- ``BOAT_HBR``: height of top superstructure (bridge etc.) (m)
- ``BOAT_LENGTH``: overall length (m)
- ``BOAT_SMCR_POWER``: Specific Maximum Continuous Rating power (kWh)
- ``BOAT_SMCR_SPEED``: average speed at SMCR power (m/s)
- ``BOAT_SPEED``: boat speed (m/s)

**Required variables in specific cases** (no default values provided):

- ``COURSES_FILE``: path to file that acts as intermediate storage for courses per routing step; it is required if ``BOAT_TYPE="CBT"``

**Recommended variables**:

- ``BOAT_TYPE``: fuel consumption modul; options: 'direct_power_method', 'CBT' (maripower), 'SAL' (maripower), 'speedy_isobased' (The latter shall only for testing; default: 'direct_power_method')
- ``BOAT_ROUGHNESS_DISTRIBUTION_LEVEL``: numeric value (default: 1)
- ``BOAT_ROUGHNESS_LEVEL``: numeric value (default: 1)

**Optional variables** (default values provided and don't need to be changed normally):

- ``AIR_MASS_DENSITY``: mass density of air used for direct power method (default: 1.2225 kg/m^3) 
- ``BOAT_AOD``: lateral projected area of superstructures etc. on deck (m)
- ``BOAT_AXV``: area of maximum transverse section exposed to the winds (m)
- ``BOAT_AYV``: projected lateral area above the waterline (m)
- ``BOAT_BS1``: breadth of substructure (m)
- ``BOAT_CMC``: horizontal distance from midship section to centre of lateral projected area AYV (m)
- ``BOAT_DRAUGHT_AFT``: aft draught (draught at rudder, default: 10m) 
- ``BOAT_DRAUGHT_FORE``: fore draught (draught at forward perpendicular, default: 10m) 
- ``BOAT_HC``: height of waterline to centre of lateral projected area Ayv (m)
- ``BOAT_HS1``: height of substructure 1 assumed for simple geometry (m)
- ``BOAT_HS2``: height of substructure 2 assumed for simple geometry (m)
- ``BOAT_LS1``: length of substructure 1 assumed for simple geometry (m)
- ``BOAT_LS2``: length of substructure 2 assumed for simple geometry (m)
- ``BOAT_OVERLOAD_FACTOR``: overload factor used for direct power method (default: 0)
- ``BOAT_PROPULSION_EFFICIENCY``: propulsion efficiency coefficient in ideal conditions (default: 0.63)
- ``BOAT_FACTOR_CALM_WATER``: multiplication factor for the calm water resistance model of maripower (default: 1)
- ``BOAT_FACTOR_WAVE_FORCES``: multiplication factor for added resistance in waves model of maripower (default: 1)
- ``BOAT_FACTOR_WIND_FORCES``: multiplication factor for the added resistance in wind model of maripower (default: 1)
- ``BOAT_UNDER_KEEL_CLEARANCE``: vertical distance between keel and ground (default: 20m)
- ``ALGORITHM_TYPE``: options: 'isofuel', 'genetic', 'speedy_isobased' (The latter shall only for testing; default: 'direct_power_method'; default: 'isofuel')
- ``CONSTRAINTS_LIST``: options: 'land_crossing_global_land_mask', 'land_crossing_polygons', 'seamarks', 'water_depth', 'on_map', 'via_waypoints', 'status_error' (default: ['land_crossing_global_land_mask', 'water_depth', 'on_map'])
- ``DELTA_FUEL``: amount of fuel per routing step (default: 3000kg)
- ``DELTA_TIME_FORECAST``: time resolution of weather forecast (default: 3h)
- ``FACTOR_CALM_WATER``: multiplication factor for the calm water resistance model
- ``FACTOR_WAVE_FORCES``: multiplication factor for the added resistance in waves model
- ``FACTOR_WIND_FORCES``: multiplication factor for the added resistance in wind model
- ``GENETIC_MUTATION_TYPE``: type for mutation (options: 'grid_based')
- ``GENETIC_NUMBER_GENERATIONS``: number of generations for genetic algorithm (default: 20)
- ``GENETIC_NUMBER_OFFSPRINGS``: number of offsprings for genetic algorithm (default: 2)
- ``GENETIC_POPULATION_SIZE``: population size for genetic algorithm (default: 20)
- ``GENETIC_POPULATION_TYPE``: type for initial population (options: 'grid_based', 'from_geojson'; default: 'grid_based')
- ``GENETIC_REPAIR_TYPE``: repair strategy for genetic algorithm (options: 'waypoints_infill', 'constraint_violation', 'no_repair', default: 'waypoints_infill' and 'constraint_violation')
- ``GENETIC_MUTATION_TYPE``: options: 'random', 'rndm_walk', 'rndm_plateau', 'route_blend', 'no_mutation' (default: 'random')
- ``GENETIC_CROSSOVER_PATCHER``: patching strategy for crossover (options: 'gcr', 'isofuel', default: 'isofuel')
- ``GENETIC_FIX_RANDOM_SEED`` options: True, False (default: False)
- ``INTERMEDIATE_WAYPOINTS``: coordinates for intermediate waypoints [[lat_one,lon_one], [lat_two,lon_two] ... ] (default: [])
- ``ISOCHRONE_MAX_ROUTING_STEPS``: maximum number of routing steps. Applies also if more than one route is searched! (default: 100)
- ``ISOCHRONE_MINIMISATION_CRITERION``: options: 'dist', 'squareddist_over_disttodest' (default: 'squareddist_over_disttodest')
- ``ISOCHRONE_NUMBER_OF_ROUTES``: integer specifying how many routes should be searched (default: 1)
- ``ISOCHRONE_PRUNE_GROUPS``: can be 'courses', 'larger_direction', 'branch' (default: 'larger_direction')
- ``ISOCHRONE_PRUNE_SECTOR_DEG_HALF``: half of the angular range of azimuth angle considered for pruning; not used for branch-based pruning (default: 91)
- ``ISOCHRONE_PRUNE_SEGMENTS``: total number of azimuth bins used for pruning in prune sector; not used for branch-based pruning (default: 20)
- ``ISOCHRONE_PRUNE_SYMMETRY_AXIS``: symmetry axis for pruning. Can be 'gcr' or 'headings_based'; not used for branch-based pruning (default: 'gcr')
- ``ROUTER_HDGS_INCREMENTS_DEG``: increment of headings (default: 6)
- ``ROUTER_HDGS_SEGMENTS``: total number of headings (put even number!!); headings are oriented around the great circle from current point to (temporary - i.e. next waypoint if used) destination (default: 30)
- ``ROUTE_POSTPROCESSING``: enable route postprocessing to follow the Traffic Separation Scheme in route postprocessing (default: False)
- ``TIME_FORECAST``: forecast hours weather (default: 90h)

Environment variables
---------------------

Credentials for the Copernicus Marine Environment Monitoring Service (CMEMS) to download weather/ocean data:

- ``CMEMS_USERNAME``
- ``CMEMS_PASSWORD``

If not provided ``DATA_MODE='automatic'`` cannot be used.

Configuration parameters for the database which stores OpenSeaMap data (optional):

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

Output data
-----------

The characteristics of the most optimal route(s) that has been/have been found by the routing algorithm are written to a json file. Thereby, a route is a collection of individual route segments for which the ship is assumed to travel under constant environmental conditions as well as constant ship parameters. The characteristics of a route segment are always tied to the starting coordinates of the route segment when written to json file. Therefore, all parameters but time and coordinates are set to arbitrary values (-99) for the last entry in the output file. The following parameters are written to file:

- coordinates; format: [lon, lat]
- time; format: 'yyyy-mm-dd hh:mm:ss'
- speed (m/s)
- engine power (kW)
- fuel consumption (mt/h)
- fuel type 
- propeller revolution (Hz)
- calm water resistance (N)
- wind resistance (N)
- wave resistance (N)
- shallow water resistance (N)
- hull roughness resistance (N)
- status; potential status code for validity of hydrodynamic modelling 
- wave height (m)
- wave direction (radian)
- wave period (s)
- u component of ocean currents (m/s)
- v component of ocean currents (m/s)
- u component of wind speed (m/s)
- v component of wind speed (m/s)
- air pressure (Pa)
- air temperature (°C)
- water temperature (°C)
- salinity 
