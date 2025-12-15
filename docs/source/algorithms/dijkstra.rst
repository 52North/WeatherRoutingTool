.. dijkstra-algorithm:

Dijkstra Algorithm
=================

!!!Warning: experimental!!!

General Concept
---------------

The Dijkstra algorithm can be used to find the shortest path from a source node to a target node in a weighted graph
with non-negative edge weights (see [2]).

The `DijkstraGlobalLandMask` class is creating a graph using the global land mask grid from the `global-land-mask <https://github.com/toddkarin/global-land-mask>`_
package. Each grid point (in the provided geographic bounding box) is a node in the graph and each node is connected to
its (diagonal, horizontal and vertical) neighbors by an edge. The depth of neighbors can be specified using the `DIJKSTRA_NOF_NEIGHBORS`
config variable. By default, only the direct closest neighbors are used (`DIJKSTRA_NOF_NEIGHBORS=1`).
However, note that this might lead to non-optimal routes (the resulting route is not the shortest distance route) due
to staircase effects! Thus, higher values of `DIJKSTRA_NOF_NEIGHBORS` are recommended which, on the other hand, will
increase computation time and memory usage.

The path to the global land mask file has to be specified via `DIJKSTRA_MASK_FILE`. If the global-land-mask package is
installed it is not necessary to download the `file <https://github.com/toddkarin/global-land-mask/blob/master/global_land_mask/globe_combined_mask_compressed.npz>`_.
The file should already be inside the package. You can check the path, e.g., with `find ~ -type f -name globe_combined_mask_compressed.npz`.

In the future, the framework could be expanded by integrating further graphs, especially non grid-generated graphs.

Useful References
-----------------

1. https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
2. https://networkx.org/documentation/stable/reference/algorithms/shortest_paths/dijkstra.html
3. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html#networkx.algorithms.shortest_paths.generic.shortest_path
