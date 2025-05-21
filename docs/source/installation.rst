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

One way to test the WRT without mariPower is to use the configuration ``SHIP_TYPE='direct_power_method'`` and specifying the mandatory config parameters

- ``'BOAT_SMCR_POWER'``: power at maximum continuous rating (kWh)
- ``'BOAT_FUEL_RATE'``: fuel rate at the working point (g/kWh)
- ``'BOAT_LENGTH'``: boat length overall (m)
- ``'BOAT_BREADTH'``: boat breadth (m)
- ``'BOAT_HBR'``: boat height (m)

This will estimate the power consumption based on several assumptions and a simple model. It is assumed, that the ship travels at a fixed working point of propeller power and ship speed. The value pair that is considered corresponds to 75 % of the SMCR power and the ship speed. Both, the SMCR power and the ship speed need to be provided by the user in the config file and it is left to the user to select sensible values for this working point. The power consumption is then calculated as the sum of 75 % SMCR power and the power consumption caused by the added resistance due to environmental conditions. The power consumption due to added resistance is estimated using the Direct Power Method  as described in the `ITTC - Recommended Procedures and Guidelines <https://www.ittc.info/media/9874/75-04-01-011.pdf>`_. Currently, only the added resistance due to wind is considered using the regression formula by `Fujiwara et al <https://www.nmri.go.jp/archives/institutes/marine_renewable_energy/marine_energy_research/staff/fujiwara/fujiwarapdf/2009-TPC-553.pdf>`_; the consideration of waves is planned for the future. Of course, results will only be rough estimates, but this simple model enables the user to quickly test functionality and performance of the code and get some first ideas of possible routes.

New ships with their own power/fuel model can be integrated by implementing a new `ship class <https://github.com/52North/WeatherRoutingTool/blob/main/WeatherRoutingTool/ship/ship.py>`_ and using it in the config.

In the future, it would be interesting to integrate general empirical formulas for power demand which provide reasonable results based on a small set of general ship parameters provided by the user like length and draft of the ship. However, for optimal results it is necessary to have a power prediction framework specifically designed for the ship(s) investigated. This could be based, e.g., on physical laws or data driven.