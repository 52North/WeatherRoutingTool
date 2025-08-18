UML Class diagrams
=====================

This section provides comprehensive UML class diagrams for the WeatherRoutingTool package,
organized by functional modules to show inheritance relationships and class hierarchies.

Algorithm Classes
-------------------------------

The routing algorithms follow a clear inheritance pattern with ``RoutingAlg`` as the base class:

.. inheritance-diagram::
   WeatherRoutingTool.algorithms.genetic
   WeatherRoutingTool.algorithms.isobased
   WeatherRoutingTool.algorithms.isochrone
   WeatherRoutingTool.algorithms.isofuel
   WeatherRoutingTool.algorithms.routingalg
   WeatherRoutingTool.algorithms.routingalg_factory
   :top-classes: object
   :parts: 1
   :private-bases:

Ship and Boat Classes
-----------------------------------

The ship modeling follows a base ``Boat`` class with specialized implementations:

.. inheritance-diagram::
   WeatherRoutingTool.ship.ship
   WeatherRoutingTool.ship.direct_power_boat
   WeatherRoutingTool.ship.maripower_tanker
   WeatherRoutingTool.ship.ship_factory
   :top-classes: object
   :parts: 1
   :private-bases:

Weather Classes
----------------------

Weather handling classes for different data sources and processing:

.. inheritance-diagram::
   WeatherRoutingTool.weather
   WeatherRoutingTool.weather_factory
   :top-classes: object
   :parts: 1
   :private-bases:

Constraint Classes
-------------------------

Constraint handling for route validation and safety:

.. inheritance-diagram::
   WeatherRoutingTool.constraints.constraints
   :top-classes: object
   :parts: 1
   :private-bases:

Configuration Classes
---------------------

Configuration management using Pydantic models:

.. inheritance-diagram::
   WeatherRoutingTool.config
   WeatherRoutingTool.ship.ship_config
   :top-classes: pydantic.BaseModel
   :parts: 1
   :caption: Configuration Classes
