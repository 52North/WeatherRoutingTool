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

In order to get high-quality results, a suitable power/fuel modelling framework should be used as it is the core of any weather routing optimization. Please check the respective section of our documentation for more information as the installation of dedicated software might be necessary for your application. 
