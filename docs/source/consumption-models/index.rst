.. _consumption_models:

Consumption models
==================

The WRT provides a simple built-in generic consumption model and a plugin framework that allows to use custom models. The simple consumption model is based on the Direct Power Method and should be used by users who don't have access to any ship specific or more accurate generic consumption models. Go to the `Direct Power Method`_ section or the `Plugin Framework`_ section for details.

Within research projects, several custom consumption models have been implemented for specific ships using both physics-based and data-driven approaches. In the context of the research project `MariData <https://maridata.org/en/start_en>`_ where the WRT was originally implemented, the power/fuel modelling framework **mariPower** was developed by project partners from the `Institute of Fluid Dynamics and Ship Theory <https://www.tuhh.de/fds/home>`_ of the Hamburg University of Technology. The mariPower package is closed-source software and this will most likely not change in the future. For users with access to mariPower, more details on the installation and the communication between the consumption model and the WRT can be found in the `respective section <mariPower_>`_ below.

Direct Power Method
------------------------------------------------------------------

To enable users to test the WRT functionality without further installation of software for consumption modelling, we have started the implementation of a simple consumption model based on the **Direct Power Method**  as described in the `ITTC - Recommended Procedures and Guidelines <https://www.ittc.info/media/9874/75-04-01-011.pdf>`_.

In case users would like to quickly test the WRT functionality, they can activate our implementation of the Direct Power method. This can be done by using the configuration ``SHIP_TYPE='direct_power_method'`` and specifying the mandatory config parameters:

- ``'BOAT_SMCR_POWER'``: power at maximum continuous rating (kWh)
- ``'BOAT_FUEL_RATE'``: fuel rate at the working point (g/kWh)
- ``'BOAT_LENGTH'``: boat length overall (m)
- ``'BOAT_BREADTH'``: boat breadth (m)
- ``'BOAT_HBR'``: boat height (m)

This will estimate the power consumption based on several assumptions and a simple model. It is assumed, that the ship travels at a fixed working point of propeller power and ship speed. The value pair that is considered corresponds to 75 % of the SMCR power and the ship speed. Both, the SMCR power and the ship speed need to be provided by the user in the config file and it is left to the user to select sensible values for this working point. The power consumption is then calculated as the sum of 75 % SMCR power and the power consumption caused by the added resistance due to environmental conditions. The power consumption due to added resistance is estimated using the Direct Power Method  as described in the `ITTC - Recommended Procedures and Guidelines <https://www.ittc.info/media/9874/75-04-01-011.pdf>`_. The power consumption that has been calculated for the fixed working point is then extrapolated towards the true ship speed using the dependency

:math:`P ~ v^3.`

Currently, only the added resistance due to wind is considered using the regression formula by `Fujiwara et al <https://www.nmri.go.jp/archives/institutes/marine_renewable_energy/marine_energy_research/staff/fujiwara/fujiwarapdf/2009-TPC-553.pdf>`_; the consideration of waves is planned for the future. Of course, results will only be rough estimates, but this simple model enables the user to quickly test functionality and performance of the code and get some first ideas of possible routes.

Plugin Framework
------------------------------------------------------------------

The plugin framework makes use of Python `Entry Points <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_. It is not necessary to change any code within the WRT itself. Plugins can be implemented as separate Python packages.

The basic structure of such a package can look like this:

.. code-block:: text

    my_ship_plugin/
    |-- my_ship_plugin/
    |   `-- __init__.py
    |-- pyproject.toml
    `-- README.md

The plugin must provide a new class which inherits from the abstract base class `Boat` in `ship.py <https://github.com/52North/WeatherRoutingTool/blob/main/WeatherRoutingTool/ship/ship.py>`_.

The file `__init__.py` might look similar to the following:

.. code-block:: shell

    from WeatherRoutingTool.ship.ship import Boat
    from WeatherRoutingTool.ship.shipparams import ShipParams


    class MyPluginBoat(Boat):

        def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False) -> ShipParams:
            # do custom calculations
            return ShipParams(...)

Note that it is planned to refactor the handling of the metocean data which will also affect the ship implementations. Thus, it is expected that the interface of the `Boat` class will change in the future.

In the `pyproject.toml`, we can now define the new entry point. Here is a minimum example:

.. code-block:: toml

    [build-system]
    requires = ["setuptools"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "my_ship_plugin"
    version = "0.1"
    authors = [
      { name="", email="" },
    ]
    description = "my ship plugin"
    readme = "README.md"
    requires-python = ">=3.12"
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
    ]
    dependencies = [
        "WeatherRoutingTool@git+https://github.com/52North/WeatherRoutingTool"
    ]

    [project.entry-points."WeatherRoutingTool.ship.ship.Boat"]
    my_ship_plugin = "my_ship_plugin:MyPluginBoat"  # <entry_point_name> = "<package_name>:<class_name>"

    [tool.setuptools.packages.find]
    include = ["my_ship_plugin*"]

After the plugin has been installed inside of the chosen environment of the WRT, the new model can be referenced in the configuration JSON file like this:

.. code-block:: json

    {
        ...
        "BOAT_TYPE": "my_ship_plugin",
        ...
    }

The WRT automatically detects available plugins. Note that the string has to match the name of the entry point.

mariPower
------------------------------------------------------------------
The mariPower package allows to predict engine power and fuel consumption under various environmental conditions for specific ships that have been investigated in the MariData project. More details about the package and the project as a whole can be found in  `this publication <https://proceedings.open.tudelft.nl/imdc24/article/view/875>`_. For users with access to mariPower, the software can be installed as described below:

- clone the repository
- change to the folder: ``cd maripower``
- install mariPower: ``pip install .`` or ``pip install -e .``

In the following, the communication between mariPower and the WRT shall be described. Information is transferred via a netCDF file between the WRT and mariPower of which the file path can be set via the config variable ``COURSES_FILE``. The coordinate pairs, courses, the ship speed and the time for which the power estimation needs to be performed are written to this file by the WRT. This information is read by mariPower, the calculation of the ship parameters is performed and the corresponding results are added as separate variables to the xarray dataset. The structure of the xarray dataset after the ship parameters have been written is the following:

.. code-block:: shell

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

The coordinates ``it_pos`` and ``it_course`` are iterators for the coordinate pairs and the courses that need to be checked per coordinate pair, respectively. The function in the WRT that writes the route parameters to the netCDF file is called ``ship.write_netCDF_courses``. Following up on this, the function ``get_fuel_netCDF`` in the WRT calls the function ``PredictPowerOrSpeedRoute`` in mariPower which itself initiates the calculation of the ship parameters. The netCDF file is overwritten by the WRT for every routing step s.t. the size of the file is not increasing during the routing process.

.. figure:: /_static/fuel_request_isobased.png
   :alt: fuel_request_isobased
   :width: 400
   :align: center

   Fig.2 Schema to visualise which coordinate pairs are send in a combined request to mariPower for fuel estimation in case of the isofuel algorithm. All coordinate pairs marked by orange filled circles are send for the second routing step. Coordinate pairs marked with blue filled circles are endpoints after the first routing step that survived the pruning.

.. figure:: /_static/fuel_request_GA.png
   :alt: fuel_request_GA
   :width: 200
   :align: center

   Fig.3 Schema to visualise which coordinate pairs are send in a combined request to mariPower for fuel estimation in case of the genetic algorithm. All coordinate pairs marked by the same colour are send in one request.

Both for the isofuel algorithm and the genetic algorithm the same structure of the netCDF file is used. However, due to the different concepts of the algorithms, the entity of points that is send for calculation in one request differs between both algorithms. For the isofuel algorithm, all coordinate pairs and courses that are considered for a single routing step are passed to mariPower in a single request (see Fig. 2). For the genetic algorithm all points and courses for a closed route are passed in a single request (see Fig. 3).
