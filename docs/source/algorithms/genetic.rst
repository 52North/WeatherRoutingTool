.. genetic-algorithm:

Genetic Algorithm
=================

The Weather Routing Tool makes use of `pymoo <https://pymoo.org/>`__
as the supporting library for the Genetic algorithm’s implementation. The Genetic algorithm
  - considers weather and constraints and
  - can be used for waypoint optimisation as well as combined waypoint and boat speed optimisation (DOF: waypoints and/or speed) and
  - can be used for optimisation of fuel consumption and arrival-time accuracy (objectives: fuel consumption and/or arrival-time accuracy).
Adding functionality for sole speed optimisation for a fixed route is planned for the near future.


The Different Run Modes
-----------------------
*Degrees of Freedom*

The DOF can be specified by setting the config variables ``GENETIC_MUTATION_TYPE`` and ``GENETIC_CROSSOVER_TYPE``.

- pure waypoint optimisation: In case pure waypoint optimisation is requested, both config variables need to be set to ``"waypoints"``.
  ``GENETIC_MUTATION_TYPE`` can also be ``"rndm_walk"``, ``"rndm_plateau"`` or ``"route_blend"``. The boat speed is taken from the
  user input to ``BOAT_SPEED`` and is left constant.

- pure speed optimisation (NOT YET IMPLEMENTED!): In case pure speed optimisation is requested,
  both config variables need to be set to ``"speed"``. ``GENETIC_MUTATION_TYPE`` can also be ``"percentage_change_speed"`` or ``"gaussian_speed"``.
  The boat speed of the initial population is read from the user input to ``BOAT_SPEED``. The waypoints of the route to be
  optimised are read from a GeoJSON file. Only speed optimisation of a single route is allowed, meaning only one GeoJSON file can
  be provided as initial population.

- waypoint and speed optimisation: Any other combination of both config variables result in mixed speed and waypoint optimisation. The initial
  population differs in waypoints but is generated with constant speed from the user input to ``BOAT_SPEED``. All generation methods for the initial
  population are allowed.

*Objectives*

The objectives can be specified by setting the config variable ``GENETIC_OBJECTIVES``. Currently only the optimisation of the total fuel
consumption (``"fuel_consumption"``) and/or arrival-time accuracy (``"arrival_time"``) is possible. In case fuel consumption shall be optimised, the algorithm minimises the total amount of fuel that is consumed for
a route. In case the arrival-time accuracy shall be optimised, the algorithm minimises the following function of the real arrival time (t_real)
and the planned arrival time (t_planned):

:math:`(t_{planned} - t_{real})^4.`

Along with the objective keys, integer weights are to be specified that rank the objectives according to their importance.
E.g. ``GENETIC_OBJECTIVES={"fuel_consumption": 2, "arrival_time": 1}`` refers to optimisation of fuel consumption and arrival-time
accuracy with an emphasis on fuel-consumption optimisation. In case both objectives
are to be considered of equal importance, the mean values of the maximum possible rank shall be provided e.g.
``GENETIC_OBJECTIVES={"fuel_consumption": 1.5, "arrival_time": 1.5}``


General Concept
---------------

The genetic algorithm follows these steps (a, b):

1. Population Generation aka. Sampling

2. Reproduction: Repeat over ``GENETIC_NUMBER_GENERATIONS`` —

   a. Selection
   b. Crossover
   c. Mutation

3. Post-processing

   a. Repair
   b. Remove Duplicates

3. Fitness Evaluation


The repeating component of the algorithm is repeated until a
**termination condition** is met. The termination condition can be
either the specified maximum number of generations (config.
``GENETIC_NUMBER_GENERATIONS``) or pre-mature termination when no valid
offsprings are produced in a generation.

Weather Routing Tool’s genetic algorithms implementation is defined
in the Genetic(RoutingAlg) class definition. It utilizes the `NSGAII
algorithm <https://pymoo.org/algorithms/moo/nsga2.html>`__ for
multi-objective optimization.

A **genetic individual** for the routing problem is modelled as a
sequence of waypoints starting from the provided *source* to the
*destination*.

References.

(a) https://github.com/anyoptimization/pymoo/blob/6e1cb8833269eb87c41045d5a4d689624dd46d48/pymoo/core/mating.py#L19
(b) https://github.com/anyoptimization/pymoo/blob/6e1cb8833269eb87c41045d5a4d689624dd46d48/pymoo/core/infill.py#L22

..

1. Initial Population Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   The Initial Population critically influences the performance of the
   genetic algorithm. In general, the more diverse the population, the
   better is the genetic algorithm’s performance because the algorithm
   has a higher chance of reaching **global optima**.

   The ideal population generation is a combination of various
   population generation methods, including:

   a. *Grid Based Population*

      This approach uses a deterministic approach to produce an initial population:

      i. Breaks down the map into a set of waypoints
      ii. Shuffles the weather condition map on these waypoints to get a shuffled grid
      iii. Generates a path from source to destination using *skimage*\ ’s ``route_through_array`` \
           method to find a plausible route
      iv. Repeats this process ``GENETIC_NUMBER_GENERATIONS`` times to get a population pool

   b. *Isofuel Population*

      This method utilizes the Isofuel algorithm to generate a set of
      routes that reach the destination. The Isofuel algorithm can be
      initialised with the ``ISOCHRONE_NUMBER_OF_ROUTES`` configuration to
      generate multiple possible routes.

   c. *Static Routes*

      These are previously generated *GeoJSON* files stored in a directory.
      These can either be manually generated or saved from another
      algorithm.

      The system can read the directory by configuring
      ``GENETIC_POPULATION_TYPE`` to “from_geojson” and setting the
      ``GENETIC_POPULATION_PATH`` value to a directory with the routes.

      Note:
       The routes are expected to be named in the following format:
       **route\_{1..N}.json**
       for example; **route_1.json, route_2.json, route_3.json, …**

       Fallback: If a **route\_{i}.json** file does not exist, the system
       falls back to generating a Great Circle Route from source to
       destination.

..

2. Reproduction
^^^^^^^^^^^^^^^

a. Selection

   The **Tournament Selection** process produces N (in our case N=2)
   high fitness individuals that are to undergo crossover and mutation

b. Crossover

   Crossover aims to produce two offspring from two parents such that
   the offspring explore a route that’s a combination of the two of
   parents.

   When a crossover operation fails to produce feasible offspring, we
   can either (1) Repair the offspring in the Repair section of the code
   or, (2) Return the parents as is to negate this reproduction process
   and redo from **Selection**. If ``GENETIC_REPAIR_TYPE`` is set to any valid repair strategy, Weather Routing Tool’s
   ``OffspringRejectionCrossover`` will accept all crossover attempts. If, however, ``GENETIC_REPAIR_TYPE``
   is set to ``no_repair``, crossovers will be rejected if they fail to produce feasible offsprings
   through the following algorithm:

   I.  Generate offsprings using a child class' implementation of the
       crossover function

   II. Check if offsprings violate discrete constraints

       A. if True — refuse both offsprings, and return the parents
       B. if False — return offsprings

   The following crossover types are implementations of ``OffspringRejectionCrossover``. For every crossover scenario,
   the algorithm chooses on a random basis which of the two approaches is executed.

   *Single Point Crossover*

      *Single Point Crossover* is a simple approach to crossover where a
      **single point of crossover** is picked at random from both of the
      parents, and a route is patched from the *crossover point of parent
      1* to the *crossover point of parent 2* and vice versa. The route patcher for the crossover
      can be chosen via the config variable ``GENETIC_CROSSOVER_PATCHER``.

      .. figure:: /_static/algorithm_genetic/single_point_crossover.png

   *Two Point Crossover*

      *Two Point Crossover* utilizes two random points such that the patched
      path avoids any object that produces a constraint violation in between. As for Single Point Crossover,
      the route patcher can be chosen via the config variable ``GENETIC_CROSSOVER_PATCHER``.

      .. figure:: /_static/algorithm_genetic/two_point_crossover.png



c. Mutation

   Mutation produces unexpected variability in the initial route to
   introduce diversity and improve the chances of the optimum route
   reaching global optima.

   As for ``OffspringRejectionCrossover``, the base class ``MutationConstraintRejection`` rejects or accepts
   offspring based on the config variable ``GENETIC_REPAIR_TYPE``. The user can choose from different mutation approaches
   by setting the config variable ``GENETIC_MUTATION_TYPE``. For the setting ``random``, the algorithm
   chooses for every mutation scenario whether route-blend or random-plateau mutation is executed. The following single
   mutation stategies are available:

   *Random Walk Mutation*

      When looking at the waypoints as belonging to a grid, the Random Walk
      Mutation moves a random waypoint to one of its N-4 neighbourhood
      positions. Can be selected via ``GENETIC_MUTATION_TYPE=rndm_walk``.

      .. figure:: /_static/algorithm_genetic/random_walk_mutation.png

   *Random Plateau Mutation*

      A set of four waypoints is selected:

        - a *plateau center* that is chosen on a random basis,
        - two *plateau edges* which are the waypoints ``plateau_size``/2 waypoints before and behind the plateau center,
        - two *connectors* which are the waypoints ``plateau_slope`` before and behind the plateau edges.

      The plateau edges are moved in the same direction to one of their N-4 neighbourhood positions as for random-walk
      mutation. A *plateau* is drawn by connecting the plateau edges to the connectors and to each other via great circle
      routes.
      Can be selected via ``GENETIC_MUTATION_TYPE=rndm_plateau``.

      .. figure:: /_static/algorithm_genetic/random_plateau_mutation.png


   *Route Blend Mutation*

      This process converts a sub path into a smoother route using a
      smoothing function such as Bezier Curves or by replacing a few
      waypoints using the Great Circle Route. Can be selected via ``GENETIC_MUTATION_TYPE=route_blend``.

      .. figure:: /_static/algorithm_genetic/route_blend_mutation.png



..

3. Post-processing
^^^^^^^^^^^^^^^^^^

a. Repair

   The Repair classes play the role of normalizing routes and fixing constraints
   violations. The current implementation executes two repair processes in the
   following order:

   Methods to repair routes are enlisted in the Route Patching section below.

   *WaypointsInfillRepair*

   Repairs routes by infilling them with equi-distant waypoints when adjacent
   points are farther than the specified distance resolution (gcr_dist)

   This avoids long-distance jumps that may lead to impractical and unfeasible routes.

   .. figure:: /_static/algorithm_genetic/waypoints_infill_repair.png

   *ConstraintViolationRepair*

   Repairs routes by identifying waypoints that are undergoing a constraint
   violation and finds a route around the points using the IsoFuel algorithm
   (See the *IsoFuel Patcher* in the **Route Patching** section below.)

   .. figure:: /_static/algorithm_genetic/constraints_violation_repair.png

   Note — Repair class’ ``_do`` method takes in a population object and
   returns a population object, in both cases the size of the population
   should be the same as the one mentioned in the config (config.
   ``GENETIC_POPULATION_SIZE``)

b. Duplicates Removal

   Pymoo gets rid of duplicate individuals in a population to maintain
   the diversity in the population pool. This specific function works by
   filtering out population individuals which are the same, thus passing
   on only non-repeating individuals to the next step.

   Note — If duplicates remove all individuals, the entire reproduction
   process is repeated. Repeats can occur a maximum of a 100 times,
   after which the genetic algorithm reaches **early termination**.

..

4. Fitness Evaluation
^^^^^^^^^^^^^^^^^^^^^

   **RoutingProblem** is Weather Routing Tool’s implementation of the
   route optimization problem necessary for defining the evaluation
   criteria for the routing problem.

   The ``_evaluate`` function measures the provided **individual**\ ’s
   fitness F and the constraints G .

   - Fitness (F) — is a list of floats representing the fitness evaluation
     of the **individual** *per objective* (fuel, distance, etc.)

   - Constraints (G) — is a list of floats represents the total constraint
     violations per constraint (specified by the ``constraints_list`` value)

Route Patching
--------------


   Route Patching is an important concept that comes up as a necessity
   across the genetic implementation. This system has uses within
   Crossover, Mutation, and Repair functions.

   The purpose of a Route Patcher is to find a **valid feasible route**
   from point A to point B, *without* necessarily optimising the
   produced sub-path.

   A Route Patcher works well if

      (a) it produces valid feasible routes *and*
      (b) if it can find novel ways to connect waypoints.

   Weather Routing Tool’s Route Patcher uses the following ways to
   connect waypoints:

1. *Great Circle Route*

   Produce a granular route along the great circle distance connecting
   the two points.

   *Advantages —*

      Produces the shortest best route from point A to point B.

   *Disadvantages —*

      It cannot handle complex route navigation, e.g., if there’s a
      landmass in between the waypoints, the patched route will violate constraints and \
      will be discarded during evaluation. \
      It is left to the calling function to update the waypoints.

2. *Isofuel Algorithm*

   Produce an optimum sub-route using the Isofuel algorithm.

   *Advantages —*

      Produces an optimal route navigating complexities.

   *Disadvantages —*

      Can be very slow and can fail based on the isofuel configuration. In case of failing, the
      algorithm will fall back to patching via the great circle route.

   *Can be used if —*

      We parallelize the execution of the Isofuel algorithm to speed up the
      process.


**Implementation Notes:**

The intuition behind having Route Patching implementations setup as
classes follows the following:
   a. Route patching can be quite expensive during both the preparation
   (defining map, loading configs, etc.) and the execution stage (patching
   between point A and point B). An Object Oriented implementation of the same
   helps separate the two processes, avoids redundancy and can contribute to the
   overall speed in the longer run.

   b. Implementation consistency makes it easier to swap between different
   Patching implementations and maintains clean code


Multi-Objective Optimisation
----------------------------


Useful References
-----------------

- https://pymoo.org/index.html

- Monitoring convergence —

  - https://pymoo.org/getting_started/part_4.html
  - https://ieeexplore.ieee.org/document/9185546
