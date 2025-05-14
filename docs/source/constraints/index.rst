.. constraints:

Constraints
===========

The constraints module
----------------------

The input parameters
--------------------

.. figure:: /_static/constraint_arguments.png
   :alt: constraint_arguments

   Fig.4: Figure for illustrating the concept of passing the information on the routing segments that are to be checked by the constraint module to the respective function. Variable names printed in orange correspond to the naming scheme for the first routing step while variables printed in blue correspond to the naming scheme for the second routing step.

As described above [ToDo], the constraint module can be used to check constraints for a complete routing segment. Thereby, several routing segments can be processed in only one request. This means that for the genetic algorithm, only one request needs to be performed for every route that is considered in a single generation and for the isofuel algorithm, only one request needs to be performed for every single routing step. This implementation minimises computation time and is achieved by passing arrays of latitudes and longitudes to the constraint module i.e. if the constraint module is called like this

.. code-block:: python

   safe_crossing(lat_start, lat_end, lon_start, lon_end)


then, the arguments ``lat_start``, ``lat_end``, ``lon_start`` and ``lon_end`` correspond to arrays for which every element characterises a different routing segment. Thus the length of the arrays is equal to the number of routing segments that are to be checked. While for the genetic algorithm, the separation of a closed route into different routing segments is rather simple, the separation for the isofuel algorithm is more complex. This is, why the passing of the latitudes and longitudes shall be explained in more detail for the isofuel algorithm in the following.

Let's consider only two routing steps of the form that is sketched in Fig. XXX. The parameters that are passed to the constraints module for the first routing step are the latitudes and longitudes of start and end points for the routing segments `a` to `e` which are

- :math:`lat\_start = (lat\_start_{abcde}, lat\_start_{abcde}, lat\_start_{abcde}, lat\_start_{abcde}, lat\_start_{abcde})`
- :math:`lat\_end = (lat\_end_{a}, lat\_end_{b}, lat\_end_{c}, lat\_end_{d}, lat\_end_{e})`
- :math:`lon\_start = (lon\_start_{abcde}, lon\_start_{abcde}, lon\_start_{abcde}, lon\_start_{abcde}, lon\_start_{abcde})`
- :math:`lon\_end = (lon\_end_{a}, lon\_end_{b}, lon\_end_{c}, lon\_end_{d}, lon\_end_{e})`

i.e. since the start coordinates are matching for all routing segments, the elements for the start latitudes and longitudes are all the same.<br>
The arguments that are passed for the second routing step are the start and end coordinates of the routing segments :math:`\alpha` to :math:`\zeta`:

- :math:`lat\_start = (lat\_start_{\alpha\beta\gamma}, lat\_start_{\alpha\beta\gamma}, lat\_start_{\alpha\beta\gamma}, lat\_start_{\delta\epsilon\zeta}, lat\_start_{\delta\epsilon\zeta}, lat\_start_{\delta\epsilon\zeta})`
- :math:`lat\_end = (lat\_end_{\alpha}, lat\_end_{\beta}, lat\_end_{\gamma}, lat\_end_{\delta}, lat\_end_{\epsilon}, lat\_end_{\zeta})`
- :math:`lon\_start = (lon\_start_{\alpha\beta\gamma}, lon\_start_{\alpha\beta\gamma}, lon\_start_{\alpha\beta\gamma}, lon\_start_{\delta\epsilon\zeta}, lon\_start_{\delta\epsilon\zeta},lon\_start_{\delta\epsilon\zeta})`
- :math:`lon\_end =  (lon\_end_{\alpha}, lon\_end_{\beta}, lon\_end_{\gamma}, lon\_end_{\delta}, lon\_end_{\epsilon}, lon\_end_{\zeta})`

i.e. the latitudes of the end points from the first routing step are now the start coordinates of the current routing step. In contrast to the first routing step, the start coordinates of the second routing step differ for several route segments.

Route Postprocessing
--------------------

When the optional config variable ``ROUTE_POSTPROCESSING`` is enabled, the route is forwarded for postprocessing to follow Traffic Separation Scheme (TSS) rules.
Pgsnapshot schema with Osmosis were used to import OpenSeaMap data into the PostGIS+PostgreSQL database to retrieve TSS related data. The key OpenSeaMap TSS tags considered for route postprocessing are ``inshore_traffic_zone``, ``separation_boundary``, ``separation_lane``, ``separation_boundary`` and ``separation_line``.
The primary TSS rules have been addressed in the current development phase are:
1. If the current route is crossing any Inshore Traffic Zone or other TTS element, then the route should enter and leave the nearest separation lane which is heading to the direction of destination.

.. figure:: /_static/follow_separation_lane.png
   :alt: follow_separation_lane

2. If the current route is intersecting the Traffic Separation Lanes and the angle between the route nodes before the intersection and after the intersection is between 60° to 120°, the new route segment is introduced as it is perpendicular to the separation lane and extends towards the last route segment, perpendicularly.

.. figure:: /_static/right_angle_crossing.png
   :alt: right_angle_crossing

Furthermore, if the starting node or the ending node is located inside a traffic separation zone, route postprocessing is not further executed.

Useful links:
-------------

* https://en.wikipedia.org/wiki/Traffic_separation_scheme
* https://wiki.openstreetmap.org/wiki/Seamarks/Seamark_Objects
* Szlapczynski, Rafal. (2012). Evolutionary approach to ship's trajectory planning within Traffic Separation Schemes. Polish Maritime Research. 19. DOI: `10.2478/v10012-012-0002-x <https://www.researchgate.net/publication/271052992_Evolutionary_approach_to_ship's_trajectory_planning_within_Traffic_Separation_Schemes>`_