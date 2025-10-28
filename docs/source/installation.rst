.. _installation:

Installation
============

Steps:

- clone the repository: ``git clone https://github.com/52North/WeatherRoutingTool.git``
- change to the folder: ``cd WeatherRoutingTool``
- [recommended] create and activate a virtual environment, e.g.
  - ``python3 -m venv "venv"``
  - ``source venv/bin/activate``
- install the WRT: ``pip install . && pip install --no-deps -r requirements-without-deps.txt`` or in editable mode (recommended for development) ``pip install -e . && pip install --no-deps -r requirements-without-deps.txt``

The part `pip install --no-deps -r requirements-without-deps.txt` is necessary because of a dependency issue (see https://github.com/52North/WeatherRoutingTool/issues/8). We might implement a different solution in the future making the installation easier/cleaner.

**Power/fuel consumption framework**

In order to get high-quality results, a suitable power/fuel modelling framework should be used as it is the core of any weather routing optimization. Please check the respective section of our documentation for more information as the installation of dedicated software might be necessary for your application. 
