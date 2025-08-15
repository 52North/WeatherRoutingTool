UML Class Diagrams
==================

This section provides comprehensive UML class diagrams for the WeatherRoutingTool package, 
organized by functional modules to show inheritance relationships and class hierarchies.

Algorithm Inheritance Hierarchy
-------------------------------

The routing algorithms follow a clear inheritance pattern with ``RoutingAlg`` as the base class:

.. inheritance-diagram::
   WeatherRoutingTool.algorithms.routingalg
   WeatherRoutingTool.algorithms.isobased
   WeatherRoutingTool.algorithms.isochrone
   WeatherRoutingTool.algorithms.isofuel
   WeatherRoutingTool.algorithms.genetic
   :top-classes: WeatherRoutingTool.algorithms.routingalg.RoutingAlg
   :parts: 1
   :caption: Routing Algorithm Class Hierarchy

Ship and Boat Inheritance Hierarchy
-----------------------------------

The ship modeling follows a base ``Boat`` class with specialized implementations:

.. inheritance-diagram::
   WeatherRoutingTool.ship.ship
   WeatherRoutingTool.ship.direct_power_boat
   WeatherRoutingTool.ship.maripower_tanker
   :top-classes: WeatherRoutingTool.ship.ship.Boat
   :parts: 1
   :caption: Ship/Boat Class Hierarchy

Weather System Classes
----------------------

Weather handling classes for different data sources and processing:

.. inheritance-diagram::
   WeatherRoutingTool.weather
   :top-classes: object
   :parts: 1
   :caption: Weather System Classes

Configuration Classes
---------------------

Configuration management using Pydantic models:

.. inheritance-diagram::
   WeatherRoutingTool.config
   WeatherRoutingTool.ship.ship_config
   :top-classes: pydantic.BaseModel
   :parts: 1
   :caption: Configuration Classes

Constraint System Classes
-------------------------

Constraint handling for route validation and safety:

.. inheritance-diagram::
   WeatherRoutingTool.constraints.constraints
   :top-classes: object
   :parts: 1
   :caption: Constraint System Classes

Complete System Overview
------------------------

Full inheritance diagram showing all major classes in the system:

.. inheritance-diagram::
   WeatherRoutingTool.algorithms.routingalg
   WeatherRoutingTool.algorithms.isobased
   WeatherRoutingTool.algorithms.isochrone
   WeatherRoutingTool.algorithms.isofuel
   WeatherRoutingTool.algorithms.genetic
   WeatherRoutingTool.ship.ship
   WeatherRoutingTool.ship.direct_power_boat
   WeatherRoutingTool.ship.maripower_tanker
   WeatherRoutingTool.weather
   WeatherRoutingTool.config
   WeatherRoutingTool.ship.ship_config
   WeatherRoutingTool.constraints.constraints
   WeatherRoutingTool.routeparams
   WeatherRoutingTool.ship.shipparams
   :top-classes: object
   :parts: 2
   :caption: Complete WeatherRoutingTool Class Hierarchy

Factory Pattern Classes
-----------------------

Factory classes for creating algorithm and ship instances:

.. inheritance-diagram::
   WeatherRoutingTool.algorithms.routingalg_factory
   WeatherRoutingTool.ship.ship_factory
   WeatherRoutingTool.weather_factory
   :top-classes: object
   :parts: 1
   :caption: Factory Pattern Classes

Genetic Algorithm Components
---------------------------

Specialized classes for the genetic algorithm implementation:

.. inheritance-diagram::
   WeatherRoutingTool.algorithms.genetic_utils
   :top-classes: object
   :parts: 1
   :caption: Genetic Algorithm Components