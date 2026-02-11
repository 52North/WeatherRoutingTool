.. _getting_started:

Getting started
===============

After successful :doc:`installation` the next step is to prepare a configuration file (see :doc:`configuration` for details) and run the Weather Routing Tool. The easiest way is to use the command line:

.. code-block:: shell

    python3 WeatherRoutingTool/cli.py -f <path>/config.json

Alternatively, you can directly use the Python package:

.. code-block:: python

    from WeatherRoutingTool.execute_routing import execute_routing
    from WeatherRoutingTool.config import Config

    config = Config.assign_config("config.json")
    execute_routing(config)

There is also a Jupyter Notebook (https://github.com/52North/WRT-sandbox) which allows to play with a basic setup.

