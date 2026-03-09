.. algorithm-overview:

Algorithm overview
==================

The implementations of the individual algorithms meet different levels of sophistication. Currently, the
implementation of the genetic algorithm is the most sophisticated as it considers weather and constraints and can be
used for waypoint optimisation as well as combined waypoint and boat speed optimisation. Adding functionality
for sole speed optimisation for a fixed route is planned for the near future.

The table below summarises the level of sophistication for all available algorithms. Information on the configurations
for the different run modes can be found in the sections that describe the functionality of each algorithm in detail.

+------------------------+-----------------------+-------------------------+-------------------------------+---------+-------------+
| Algorithm / Feature    | Waypoint optimization | Ship speed optimization | Waypoint & speed optimization | Weather | Constraints |
+========================+=======================+=========================+===============================+=========+=============+
| Genetic                | Yes                   | No                      | Yes                           | Yes     | Yes         |
+------------------------+-----------------------+-------------------------+-------------------------------+---------+-------------+
| Isofuel                | Yes                   | No                      | No                            | Yes     | Yes         |
+------------------------+-----------------------+-------------------------+-------------------------------+---------+-------------+
| GCR Slider             | Yes                   | No                      | No                            | No      | Partially   |
+------------------------+-----------------------+-------------------------+-------------------------------+---------+-------------+
| Dijkstra               | Yes                   | No                      | No                            | No      | Partially   |
+------------------------+-----------------------+-------------------------+-------------------------------+---------+-------------+

