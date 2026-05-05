.. _software_profiling:


Software profiling
==================


Runtime monitoring
------------------

The runtime can be analyzed using `cProfile <https://docs.python.org/3/library/profile.html#module-cProfile>`_ from Python's standard library.

To do this add the following code lines in `execute_routing.py <https://github.com/52North/WeatherRoutingTool/blob/main/WeatherRoutingTool/execute_routing.py>`_:

.. code-block:: python

    import cProfile

    def execute_routing(config, ship_config):
        prof = cProfile.Profile()
        prof.enable()

        ...

        prof.disable()
        prof.dump_stats('wrt_run.prof')

The result can be visualized using `snakeviz <https://jiffyclub.github.io/snakeviz/>`_:

.. code-block:: shell

    snakeviz wrt_run.prof

Memory monitoring
-----------------

TBD