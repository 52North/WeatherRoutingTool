.. _conventions:

Conventions
===========

Coordinates
-----------

* latitude: -90° - 90°
* longitude: -180° - 180°
* headings: 0° - 360°, angular difference between North and the ship's direction, angles are going in the negative mathematical direction (clockwise)

Units
-----

Apart from one exception, the WRT uses SI units for internal calculations. Only angles are handled in degrees as defined in the paragraph above. All input variables that carry a unit are converted according to these definitions. For the output -- i.e. when a route is written to a json file -- the engine power is converted to kW and the fuel rate to mt/h, where mt refers to *metric ton*.

The WRT uses the package `astropy <https://docs.astropy.org/en/stable/units/>`_ for the convenient handling of units.

Logging
-------

The routing tool writes log output using the python package logging.
Information about basic settings are written to a file which is specified by the environment variable ``INFO_LOG_FILE``. Warnings and performance information are written to the file which is specified by the environment variable ``PERFORMANCE_LOG_FILE``.
Further debug information are written to stdout.