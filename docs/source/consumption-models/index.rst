.. _consumption_models:

Consumption models
==================


Fuel estimation -- The communication between mariPower and the WRT
------------------------------------------------------------------

Information is transferred via a netCDF file between the WRT and mariPower of which the file path can be set via the config variable ``COURSES_FILE``. The coordinate pairs, courses, the ship speed and the time for which the power estimation needs to be performed are written to this file by the WRT. This information is read by mariPower, the calculation of the ship parameters is performed and the corresponding results are added as separate variables to the xarray dataset. The structure of the xarray dataset after the ship parameters have been written is the following:

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

   Fig.2 Schema to visualise which coordinate pairs are send in a combined request to mariPower for fuel estimation in case of the isofuel algorithm. All coordinate pairs marked by orange filled circles are send for the second routing step. Coordinate pairs marked with blue filled circles are endpoints after the first routing step that survived the pruning.

.. figure:: /_static/fuel_request_GA.png
   :alt: fuel_request_GA

   Fig.3 Schema to visualise which coordinate pairs are send in a combined request to mariPower for fuel estimation in case of the genetic algorithm. All coordinate pairs marked by the same colour are send in one request.

Both for the isofuel algorithm and the genetic algorithm the same structure of the netCDF file is used. However, due to the different concepts of the algorithms, the entity of points that is send for calculation in one request differs between both algorithms. For the isofuel algorithm, all coordinate pairs and courses that are considered for a single routing step are passed to mariPower in a single request (see Fig. 2). For the genetic algorithm all points and courses for a closed route are passed in a single request (see Fig. 3).
