.. _installation:

Installation
============

Steps:

- clone the repository: ``git clone https://github.com/52North/WeatherRoutingTool.git``
- change to the folder: ``cd WeatherRoutingTool``
- [recommended] create and activate a virtual environment, e.g.
  - ``python3 -m venv "venv"``
  - ``source venv/bin/activate``
- install the WRT: ``pip install .`` or in editable mode (recommended for development) ``pip install -e .``

**Power/fuel consumption framework**

In order to get high-quality results, a suitable power/fuel modelling framework should be used as it is the core of any weather routing optimization.

The WRT was originally implemented within the research project `MariData <https://maridata.org/en/start_en>`_. Within this project the power/fuel modelling framework **mariPower** was developed by project partners from the `Institute of Fluid Dynamics and Ship Theory <https://www.tuhh.de/fds/home>`_ of the Hamburg University of Technology.
The mariPower package allows to predict engine power and fuel consumption under various environmental conditions for specific ships investigated in the project. More details about the package and the project as a whole can be found in the following publication: https://proceedings.open.tudelft.nl/imdc24/article/view/875.
The mariPower package is closed source software and this will most likely not change in the future. However, as power demand varies from ship to ship users have to provide code for making power predictions on their own suitable to the use case.

For users with access to mariPower:

- clone the repository
- change to the folder: ``cd maripower``
- install mariPower: ``pip install .`` or ``pip install -e .``

For users without access to mariPower:

One way to quickly test the WRT without mariPower is to use the configuration ``ALGORITHM_TYPE='speedy_isobased'`` and specifying the config parameter ``'CONSTANT_FUEL_RATE'``. This will assume a constant fuel rate of the vessel in any condition (high waves, low waves, etc.) and at any time. Of course, results will be highly inaccurate, but it is a way to quickly try and test the code and get some first ideas of possible routes.

New ships with their own power/fuel model can be integrated by implementing a new `ship class <https://github.com/52North/WeatherRoutingTool/blob/main/WeatherRoutingTool/ship/ship.py>`_ and using it in the config.

In the future, it would be interesting to integrate general empirical formulas for power demand which provide reasonable results based on a small set of general ship parameters provided by the user like length and draft of the ship. However, for optimal results it is necessary to have a power prediction framework specifically designed for the ship(s) investigated. This could be based, e.g., on physical laws or data driven.