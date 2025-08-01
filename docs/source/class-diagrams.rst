.. _class-diagrams:

Detailed Class Diagrams
========================

This document provides detailed class diagrams and architectural overviews for the WeatherRoutingTool.

Core Architecture Overview
---------------------------

The WeatherRoutingTool follows a modular architecture with clear separation of concerns:

.. graphviz::
   :caption: High-Level Architecture Diagram

   digraph architecture {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       
       CLI [label="CLI Interface\n(cli.py)"];
       Config [label="Configuration\n(config.py)"];
       Execute [label="Route Execution\n(execute_routing.py)"];
       
       subgraph cluster_algorithms {
           label="Routing Algorithms";
           style=filled;
           fillcolor=lightgreen;
           RoutingAlg [label="RoutingAlg\n(Base Class)"];
           IsoBased [label="IsoBased"];
           IsoFuel [label="IsoFuel"];
           Genetic [label="Genetic Algorithm"];
       }
       
       subgraph cluster_ships {
           label="Ship Models";
           style=filled;
           fillcolor=lightyellow;
           Boat [label="Boat\n(Base Class)"];
           DirectPower [label="DirectPowerBoat"];
           MariPower [label="MariPowerTanker"];
           ConstantFuel [label="ConstantFuelBoat"];
       }
       
       subgraph cluster_weather {
           label="Weather System";
           style=filled;
           fillcolor=lightcoral;
           WeatherCond [label="WeatherCond"];
           WeatherFactory [label="WeatherFactory"];
       }
       
       subgraph cluster_constraints {
           label="Constraints";
           style=filled;
           fillcolor=lightgray;
           Constraints [label="Constraint Classes"];
       }
       
       CLI -> Config;
       CLI -> Execute;
       Execute -> RoutingAlg;
       Execute -> Boat;
       Execute -> WeatherCond;
       Execute -> Constraints;
       RoutingAlg -> IsoBased;
       RoutingAlg -> Genetic;
       IsoBased -> IsoFuel;
       Boat -> DirectPower;
       Boat -> MariPower;
       Boat -> ConstantFuel;
   }

Algorithm Class Relationships
-----------------------------

.. graphviz::
   :caption: Detailed Algorithm Class Diagram

   digraph algorithms {
       rankdir=TB;
       node [shape=record, fontsize=10];
       
       RoutingAlg [label="{RoutingAlg|+ start: tuple\\l+ finish: tuple\\l+ departure_time: datetime\\l|+ execute_routing()\\l+ terminate()\\l+ check_destination()\\l+ check_positive_power()\\l}"];
       
       IsoBased [label="{IsoBased|+ count: int\\l+ ncount: int\\l+ route_reached_destination: bool\\l+ pruning_error: bool\\l|+ move_boat_direct()\\l+ pruning_per_step()\\l+ update_position()\\l+ update_fuel()\\l}"];
       
       IsoFuel [label="{IsoFuel|+ delta_fuel: float\\l|+ get_delta_variables()\\l+ get_dist()\\l+ get_time()\\l}"];
       
       Genetic [label="{Genetic|+ pop_size: int\\l+ n_offsprings: int\\l+ n_generations: int\\l|+ optimize()\\l+ plot_running_metric()\\l+ plot_population()\\l}"];
       
       Isochrone [label="{IsoChrone|+ delta_time: int\\l|+ check_isochrones()\\l+ get_dist()\\l}"];
       
       RoutingAlg -> IsoBased [arrowhead=empty];
       RoutingAlg -> Genetic [arrowhead=empty];
       IsoBased -> IsoFuel [arrowhead=empty];
       IsoBased -> Isochrone [arrowhead=empty];
   }

Ship Class Relationships
-------------------------

.. graphviz::
   :caption: Ship Class Hierarchy with Methods

   digraph ships {
       rankdir=TB;
       node [shape=record, fontsize=10];
       
       Boat [label="{Boat|+ speed: float\\l+ under_keel_clearance: float\\l+ draught_aft: float\\l+ draught_fore: float\\l|+ get_ship_parameters()\\l+ get_required_water_depth()\\l+ evaluate_weather()\\l+ approx_weather()\\l}"];
       
       DirectPowerBoat [label="{DirectPowerBoat|+ smcr_power: float\\l+ fuel_rate: float\\l+ length: float\\l+ breadth: float\\l+ Axv, Ayv, Aod: float\\l|+ get_power()\\l+ calculate_resistance()\\l+ evaluate_resistance()\\l+ interpolate_to_true_speed()\\l}"];
       
       MariPowerTanker [label="{MariPowerTanker|+ hydro_model: object\\l+ courses_path: str\\l|+ write_netCDF_courses()\\l+ get_fuel_netCDF()\\l+ extract_params_from_netCDF()\\l}"];
       
       ConstantFuelBoat [label="{ConstantFuelBoat|+ fuel_rate: float\\l|+ get_ship_parameters()\\l}"];
       
       ShipParams [label="{ShipParams|+ fuel_rate: array\\l+ power: array\\l+ speed: array\\l+ r_wind, r_waves: array\\l+ environmental data\\l|+ get_*() methods\\l+ set_*() methods\\l+ define_courses()\\l}"];
       
       Boat -> DirectPowerBoat [arrowhead=empty];
       Boat -> MariPowerTanker [arrowhead=empty];
       Boat -> ConstantFuelBoat [arrowhead=empty];
       Boat -> ShipParams [arrowhead=diamond, label="uses"];
   }

Configuration System
--------------------

.. graphviz::
   :caption: Configuration Class Structure

   digraph config {
       rankdir=TB;
       node [shape=record, fontsize=10];
       
       BaseModel [label="{pydantic.BaseModel|+ field validation\\l+ serialization\\l}"];
       
       Config [label="{Config|+ ALGORITHM_TYPE: str\\l+ BOAT_TYPE: str\\l+ DEFAULT_ROUTE: list\\l+ DEPARTURE_TIME: datetime\\l+ CONSTRAINTS_LIST: list\\l+ routing parameters\\l+ file paths\\l|+ validate_config()\\l+ assign_config()\\l}"];
       
       ShipConfig [label="{ShipConfig|+ BOAT_SPEED: float\\l+ BOAT_LENGTH: float\\l+ BOAT_BREADTH: float\\l+ BOAT_DRAUGHT_*: float\\l+ power parameters\\l+ resistance parameters\\l|+ validation methods\\l}"];
       
       BaseModel -> Config [arrowhead=empty];
       BaseModel -> ShipConfig [arrowhead=empty];
   }

Weather System Architecture
---------------------------

.. graphviz::
   :caption: Weather Data Processing Flow

   digraph weather {
       rankdir=LR;
       node [shape=box, style="rounded,filled"];
       
       DataSources [label="Data Sources\n• NetCDF Files\n• Automatic Download\n• OpenDataCube\n• Fake Data", fillcolor=lightblue];
       
       WeatherFactory [label="WeatherFactory\n• Factory Pattern\n• Data Mode Selection", fillcolor=lightgreen];
       
       WeatherCond [label="WeatherCond\n• Base Class\n• Time/Space Handling", fillcolor=lightyellow];
       
       Implementations [label="Implementations\n• WeatherCondFromFile\n• WeatherCondEnvAutomatic\n• WeatherCondODC\n• FakeWeather", fillcolor=lightcoral];
       
       Processing [label="Weather Processing\n• Interpolation\n• Unit Conversion\n• Validation", fillcolor=lightgray];
       
       DataSources -> WeatherFactory;
       WeatherFactory -> WeatherCond;
       WeatherCond -> Implementations;
       Implementations -> Processing;
   }

Constraint System
-----------------

.. graphviz::
   :caption: Constraint Validation Flow

   digraph constraints {
       rankdir=TB;
       node [shape=record, fontsize=10];
       
       ConstraintsList [label="{ConstraintsList|+ constraints: list\\l+ waypoints: list\\l|+ safe_crossing()\\l+ reached_positive()\\l+ get_current_destination()\\l}"];
       
       Constraint [label="{Constraint|+ name: str\\l|+ safe_crossing()\\l+ is_violating()\\l}"];
       
       LandCrossing [label="{LandCrossing|+ global_land_mask\\l|+ check_land_crossing()\\l}"];
       
       WaterDepth [label="{WaterDepth|+ min_depth: float\\l+ depth_data\\l|+ check_depth()\\l}"];
       
       Seamarks [label="{Seamarks|+ polygons\\l|+ check_seamarks()\\l}"];
       
       ViaWaypoints [label="{ViaWaypoints|+ waypoints: list\\l|+ check_waypoints()\\l}"];
       
       ConstraintsList -> Constraint [arrowhead=diamond, label="contains"];
       Constraint -> LandCrossing [arrowhead=empty];
       Constraint -> WaterDepth [arrowhead=empty];
       Constraint -> Seamarks [arrowhead=empty];
       Constraint -> ViaWaypoints [arrowhead=empty];
   }

Data Flow Architecture
----------------------

.. graphviz::
   :caption: Complete Data Flow Through the System

   digraph dataflow {
       rankdir=TB;
       node [shape=ellipse, style=filled];
       
       Input [label="Configuration\nInput", fillcolor=lightblue];
       WeatherData [label="Weather\nData", fillcolor=lightcoral];
       DepthData [label="Depth\nData", fillcolor=lightgray];
       
       node [shape=box, style="rounded,filled"];
       
       ConfigValidation [label="Config\nValidation", fillcolor=lightgreen];
       ShipModel [label="Ship Model\nInitialization", fillcolor=lightyellow];
       WeatherLoad [label="Weather\nLoading", fillcolor=lightcoral];
       ConstraintSetup [label="Constraint\nSetup", fillcolor=lightgray];
       
       AlgorithmExec [label="Algorithm\nExecution", fillcolor=orange];
       RouteOptim [label="Route\nOptimization", fillcolor=gold];
       
       Output [label="Optimized\nRoute", fillcolor=lightpink];
       
       Input -> ConfigValidation;
       WeatherData -> WeatherLoad;
       DepthData -> ConstraintSetup;
       
       ConfigValidation -> ShipModel;
       ConfigValidation -> ConstraintSetup;
       WeatherLoad -> AlgorithmExec;
       ShipModel -> AlgorithmExec;
       ConstraintSetup -> AlgorithmExec;
       
       AlgorithmExec -> RouteOptim;
       RouteOptim -> Output;
   }
