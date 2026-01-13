.. gcr-slider-algorithm:

GCR Slider Algorithm
=================

!!!Warning: experimental!!!

General Concept
---------------

The GCR (great circle route) slider algorithm is an heuristic algorithm to find a route between two points which
does not cross land. The algorithm connects start and end point with the great circle route. If the connecting line
cuts land a new waypoint is added in the middle of the line. If the new waypoint is on land it is moved orthogonally
until it is on water. This process continues until no segment cuts land.

There are a few config variables starting with `GCR_SLIDER_` which can be used to tune the algorithm. Have a look into
the :doc:`../configuration` section for details.

The basic idea of this algorithm has been described in:

Kuhlemann, S., & Tierney, K. (2020). A genetic algorithm for finding realistic sea routes considering the weather.
Journal of Heuristics, 26(6), 801-825.

