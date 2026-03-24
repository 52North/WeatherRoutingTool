.. algorithm-overview:

Algorithm overview
==================

The implementations of the individual algorithms meet different levels of sophistication. In particular, the algorithms
differ in the possible *degrees of freedom (DOF)* that can be manipulated (e.g. boat speed, waypoints) and the *objectives* that
can be optimised (fuel, arrival-time accuracy, distance). The table below summarises the possible run modes for all available algorithms.
Information on the configurations for the different run modes can be found in the sections that describe the functionality of each algorithm in detail.

+------------------------+-------------------------------------------------+-----------------------------------------+---------+-------------+
| Algorithm / Feature    | Degrees of Freedom                              | Objectives                              | Weather | Constraints |
+========================+=======================+=========================+=========================================+=========+=============+
| Genetic                | speed, waypoints, speed & waypoints             | fuel consumption, arrival-time accuracy | Yes     | Yes         |
+------------------------+-------------------------------------------------+-----------------------------------------+---------+-------------+
| Isofuel                | waypoints                                       | fuel consumption                        | Yes     | Yes         |
+------------------------+-------------------------------------------------+-----------------------------------------+---------+-------------+
| GCR Slider             | waypoints                                       | distance                                | No      | Partially   |
+------------------------+-------------------------------------------------+-----------------------------------------+---------+-------------+
| Dijkstra               | waypoints                                       | distance                                | No      | Partially   |
+------------------------+-------------------------------------------------+-----------------------------------------+---------+-------------+

