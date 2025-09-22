.. genetic-algorithm:

Genetic Algorithm
=================


Foundation
----------

The Weather Routing Tool makes use of `pymoo <https://pymoo.org/>`__
as the supporting library for the Genetic algorithm’s implementation.

General Concept
---------------

The genetic algorithm follows these steps (a, b):

1. Population Generation aka. Sampling

2. Fitness Evaluation

3. Repeat over ``GENETIC_NUMBER_GENERATIONS`` —

   a. Selection
   b. Crossover
   c. Mutation
   d. Repair
   e. Remove Duplicates

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

Preparation
^^^^^^^^^^^

I. Initial Population Generation

   The Initial Population critically influences the performance of the
   genetic algorithm. In general, the more diverse the population, the
   better is the genetic algorithm’s performance because the algorithm
   has a higher chance of reaching **global optima**.

   The ideal population generation is a combination of various
   population generation methods, including:

   a. *Grid Based Population*

      This approach uses a deterministic approach to produce an initial
        population:

      i. Breaks down the map into a set of waypoints
      ii. Shuffles the weather condition map on these waypoints to get a
         shuffled grid
      iii. Generates a path from source to destination using *skimage*\ ’s
         ``route_through_array`` method to find a plausible route
      iv. Repeats this process ``GENETIC_NUMBER_GENERATIONS`` times to get a
         population pool

   b. *Border Following Population (Yet to implement)*

      This helps generate extreme waypoints which follow the boundary of
      the map and feasible constraints.

   c. *Isofuel Population*

      This method utilizes the Isofuel algorithm to generate a set of
      routes that reach the destination. The Isofuel algorithm can be
      initialised with the ``ISOCHRONE_NUMBER_OF_ROUTES`` configuration to
      generate multiple possible routes.

   d. *Static Routes*

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

2. Fitness Evaluation

   **RoutingProblem** is Weather Routing Tool’s implementation of the
   route optimization problem necessary for defining the evaluation
   criteria for the routing problem.

   The ``_evaluate`` function measures the provided **individual**\ ’s
   fitness F and the constraints G .

   - Fitness (F) — is a list of floats representing the fitness evaluation
     of the **individual** *per objective* (fuel, distance, etc.)

   - Constraints (G) — is a list of floats represents the total constraint
     violations per constraint (specified by the ``constraints_list`` value)

Reproduction
^^^^^^^^^^^^

3. Selection

   The **Tournament Selection** process produces N (in our case N=2)
   high fitness individuals that are to undergo crossover and mutation

4. Crossover

   Crossover aims to produce two offspring from two parents such that
   the offspring explore a route that’s a combination of the two of
   parents.

   When a crossover operation fails to produce feasible offspring, we
   can either (1) Repair the offspring in the Repair section of the code
   or, (2) Return the parents as is to negate this reproduction process
   and redo from **Selection**.

   Weather Routing Tool’s OffspringRejectionCrossover base class chooses
   to dismiss the crossover when it fails to produce feasible offsprings
   through the following algorithm:

   I.  Generate offsprings using a child class' implementation of the
       crossover function

   II. Check if offsprings violate discrete constraints

       A. if True — refuse both offsprings, and return the parents
       B. if False — return offsprings

   The following crossover types are implementations of the same:

   a. *Single Point Crossover*

      *Single Point Crossover* is a simple approach to crossover where a
      **single point of crossover** is picked at random from both of the
      parents, and a route is patched from the *crossover point of parent
      1* to the *crossover point of parent 2* and vice versa.

      .. figure:: /_static/algorithm_genetic/single_point_crossover.png

   b. *Two Point Crossover*

      *Two Point Crossover* utilizes two random points such that the
      patched path avoids any object that produces a constraint violation
      in between.

      The choice of the random points don’t always produce the right
      crossover points for which we make use of **Patching** (look at the
      *Route Patching* section)

      .. figure:: /_static/algorithm_genetic/two_point_crossover.png

5. Mutation

   Mutation produces unexpected variability in the initial route to
   introduce diversity and improve the chances of the optimum route
   reaching global optima.

   The Weather Routing Tool considers the following few Mutation
   approaches:

   a. *Random Walk Mutation*

      When looking at the waypoints as belonging to a grid, the Random Walk
      Mutation moves a random waypoint to one of its N-4 neighbourhood
      positions.

      .. figure:: /_static/algorithm_genetic/random_walk_mutation.png

   b. *Route Blend Mutation (Not yet Implemented)*

      This process converts a sub path into a smoother route using a
      smoothing function such as Bezier Curves or by replacing a few
      waypoints using the Great Circle Route.

      .. figure:: /_static/algorithm_genetic/route_blend_mutation.png

..

Post-processing
^^^^^^^^^^^^^^^

6. Repair

   The Repair class is meant to fix infeasible individuals in a
   population, and return the entire fixed population. Useful for
   patching paths which have a clear violation.

   Methods to repair routes are enlisted in the **Route Patching**
   section below.

   Note — Repair class’ ``_do`` method takes in a population object and
   returns a population object, in both cases the size of the population
   should be the same as the one mentioned in the config (config.
   ``GENETIC_POPULATION_SIZE``)

7. Duplicate Removal

   Pymoo gets rid of duplicate individuals in a population to maintain
   the diversity in the population pool. This specific function works by
   filtering out population individuals which are the same, thus passing
   on only non-repeating individuals to the next step.

   Note — If duplicates remove all individuals, the entire reproduction
   process is repeated. Repeats can occur a maximum of a 100 times,
   after which the genetic algorithm reaches **early termination**.

Concepts
--------

Route Patching
^^^^^^^^^^^^^^

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
      landmass in between the waypoints. It is left to the calling function
      to update the waypoints.

2. *Isofuel Algorithm*

   Produce an optimum sub-route using the Isofuel algorithm.

   *Advantages —*

      Produces an optimal route navigating complexities.

   *Disadvantages —*

      Can be very slow and can fail based on the isofuel configuration.

   *Can be used if —*

      We parallelize the execution of the Isofuel algorithm to speed up the
      process.

Ideal and Nadir points
^^^^^^^^^^^^^^^^^^^^^^

   Monitoring convergence of the genetic algorithm for explainability
   can be achieved by graphing the fitness values of the execution of
   the algorithm.

   Ideal and Nadir points represent the most and least optimum points of
   the pareto solutions respectively. These can be difficult to measure
   prior to the execution of the problem but certain assumptions can be
   made:

   - **Ideal Point** represents the best solution the algorithm can
     achieve, which we can assume, for our complex problem to be either the
     *fitness of the isofuel algorithm’s solution* or the *deterministic
     solution to the grid based approach* can be used to model the
     performance.

   - **Nadir Point** represents the least optimum points of the Pareto
     solutions. A deterministically measured optimal route produced by the
     *recombination of the waypoints of the initial population* can be
     assumed to be the nadir point; if the genetic algorithm results in a
     worse configuration, it indicates a problem.

..

Performance over 10 iterations
------------------------------

.. figure:: /_static/algorithm_genetic/population_figures.png

Config Parameters
-----------------

1. ``GENETIC_NUMBER_GENERATIONS`` — Max number of generations

2. ``GENETIC_NUMBER_OFFSPRINGS`` — Number of offsprings

3. ``GENETIC_POPULATION_SIZE`` — Population size of the genetic algorithm

4. ``GENETIC_POPULATION_TYPE`` — Population generation method for the
   genetic algorithm

   a. ``GENETIC_POPULATION_PATH`` — Path to population directory when
      ``GENETIC_POPULATION_TYPE`` is “\ *from_geojson*\ ”


Useful References
-----------------

- https://pymoo.org/index.html

- Monitoring convergence —

  - https://pymoo.org/getting_started/part_4.html
  - https://ieeexplore.ieee.org/document/9185546
