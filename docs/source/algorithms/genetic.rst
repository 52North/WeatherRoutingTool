.. genetic-algorithm:

Genetic Algorithm
=================

General concept
---------------

Five phases:

#. Initial population

    * Consists of candidate solutions (also called individuals) which have a set of properties (also called parameters, variables, genes)
#. Fitness function (evaluation)
#. Selection
#. Crossover
#. Mutation

Abort criteria:

* Maximum number of generations
* Satisfactory fitness level has been reached

Routing Problem
---------------

Phases:

#. Initial population

    * ``route_through_array``
#. Fitness function (evaluation)

    * mariPower
#. Selection
#. Crossover

    * only routes which cross geometrically are used for crossover
#. Mutation

    * in principle random but can be restricted

Useful links
------------
* https://pymoo.org/index.html
* monitoring convergence (e.g. using Running Matrix):

    * https://pymoo.org/getting_started/part_4.html
    * https://ieeexplore.ieee.org/document/9185546

Variable definitions: debug output
----------------------------------
* ``n_gen``: current generation
* ``n_nds``: number of non-dominating solutions
* ``cv_min``: minimum constraint violation
* ``cv_avg``: average constraint violation
* ``eps``: epsilon?
* ``indicator``: indicator to monitor algorithm performance; can be Hypervolume, Running Metric ...

General Notes
-------------
* ``res.F = None``: quick-and-dirty hack possible by passing ``return_least_infeasible = True`` to init function of NSGAII
* chain of function calls until ``RoutingProblem._evaluate()`` is called:

    * core/algorithms.run -> core/algorithms.next -> core/evaluator.eval -> core/evaluator._eval
* chain of function calls until crossover/mutation/selection are called:

    * core/algorithms.run -> core/algorithms.next -> core/algorithm.infill -> algorithms/base/genetic._infill
