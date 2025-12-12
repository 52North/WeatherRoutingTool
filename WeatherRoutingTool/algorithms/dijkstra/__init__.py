import json
import logging
from itertools import product
from math import ceil, gcd
from typing import Generator

import networkx as nx
import numpy as np
from astropy import units as u
from geographiclib.geodesic import Geodesic
from geographiclib.geodesicline import GeodesicLine
from global_land_mask import is_land
from shapely import Point, to_geojson

from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger("WRT.dijkstra")

geod = Geodesic.WGS84


def range_including_last_element(iterable, step: int):
    length = len(iterable)
    if length == 0:
        return
    last_index_visited = -1
    for i in range(0, length, step):
        yield i
        last_index_visited = i
    if last_index_visited != length - 1:
        yield length-1


def points_to_geojson(points: list[tuple[float, float]], filename: str = None, step: int = 10):
    d = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": i},
                "geometry": json.loads(
                    to_geojson(Point(points[i][0], points[i][1]))
                )
            }
            for i in range_including_last_element(points, step)
        ]
    }
    if filename:
        with open(filename, 'w') as f:
            json.dump(d, f, indent=2)
    return d


class DijkstraGlobalLandMask(RoutingAlg):
    """
    Grd-based Dijkstra algorithm implementation.
    The graph is created using the global land mask (https://github.com/toddkarin/global-land-mask) grid
    by connecting each grid point to a configurable number of neighbors.
    """
    mask: np.array
    longitude: np.array
    latitude: np.array
    res_lat: float
    res_lon: float

    def __init__(self, config):
        """
        :param config:
        :type config: WeatherRoutingTool.config.Config
        :return: None
        :rtype: NoneType
        """
        super().__init__(config)
        self.config = config
        self.nof_neighbors = config.DIJKSTRA_NOF_NEIGHBORS
        self.step = config.DIJKSTRA_STEP
        self.interval = 1000
        self.read_mask(config.DIJKSTRA_MASK_FILE)

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        method = 'dijkstra'
        graph = self.get_graph()
        logger.info(f"Find shortest route using the {method} algorithm")
        # The start and end points might not be on the grid or in the graph, respectively.
        # Thus, we search the closest point/node.
        start_lon = self.longitude[np.argmin(np.abs(self.longitude - self.start[1]))]
        start_lat = self.latitude[np.argmin(np.abs(self.latitude - self.start[0]))]
        end_lon = self.longitude[np.argmin(np.abs(self.longitude - self.finish[1]))]
        end_lat = self.latitude[np.argmin(np.abs(self.latitude - self.finish[0]))]
        path = nx.shortest_path(graph, (start_lon, start_lat), (end_lon, end_lat), weight='distance', method=method)
        # path_length = nx.shortest_path_length(graph, (start_lon, start_lat), (end_lon, end_lat), weight='distance', method=method)  # noqa: E501

        path_sub = [path[i] for i in range_including_last_element(path, self.step)]
        lons, lats = list(zip(*path_sub))

        ship_params = ShipParams(
            speed=np.zeros(len(path_sub)-1) * u.meter/u.second,
            fuel_rate=np.zeros(len(path_sub)-1) * u.kg/u.second,
            power=np.zeros(len(path_sub)-1) * u.Watt,
            rpm=np.zeros(len(path_sub)-1) * 1/u.minute,
            r_calm=np.zeros(len(path_sub)-1) * u.newton,
            r_wind=np.zeros(len(path_sub)-1) * u.newton,
            r_waves=np.zeros(len(path_sub)-1) * u.newton,
            r_shallow=np.zeros(len(path_sub)-1) * u.newton,
            r_roughness=np.zeros(len(path_sub)-1) * u.newton,
            wave_height=np.zeros(len(path_sub)-1) * u.meter,
            wave_direction=np.zeros(len(path_sub)-1) * u.radian,
            wave_period=np.zeros(len(path_sub)-1) * u.second,
            u_currents=np.zeros(len(path_sub)-1) * u.meter/u.second,
            v_currents=np.zeros(len(path_sub)-1) * u.meter/u.second,
            u_wind_speed=np.zeros(len(path_sub)-1) * u.meter/u.second,
            v_wind_speed=np.zeros(len(path_sub)-1) * u.meter/u.second,
            pressure=np.zeros(len(path_sub)-1) * u.kg/u.meter/u.second**2,
            air_temperature=np.zeros(len(path_sub)-1) * u.deg_C,
            salinity=np.zeros(len(path_sub)-1) * u.dimensionless_unscaled,
            water_temperature=np.zeros(len(path_sub)-1) * u.deg_C,
            status=np.zeros(len(path_sub)-1),
            message=np.array([""]*(len(path_sub)-1))
        )

        route = RouteParams(
            count=len(path_sub) - 2,
            start=self.start,
            finish=self.finish,
            gcr=None,
            route_type='min_distance_route_dijkstra',
            time=0,
            lats_per_step=lats,
            lons_per_step=lons,
            course_per_step=0,
            dists_per_step=[0]*len(path_sub),
            starttime_per_step=[0]*len(path_sub),
            ship_params_per_step=ship_params,
        )
        return route, 0

    def get_neighbors_to_connect(self) -> Generator[tuple[int, int], None, None]:
        """
        Get neighboring nodes the node in question should be connected to by an edge.
        Self-loops and axial (horizontal and vertical) and diagonal multiples are disregarded.
        """
        neighbors = list(product(range(-self.nof_neighbors, self.nof_neighbors+1, 1), repeat=2))
        for ii, jj in neighbors:
            if gcd(abs(ii), abs(jj)) != 1:
                pass
            else:
                yield ii, jj

    def get_graph(self) -> nx.Graph:
        """
        Create an undirected graph without multiple (parallel) edges.
        """
        # FIXME: might need to reload the mask
        self.subset_bbox()
        logger.info(f"Create graph for bounding box {self.map_ext.get_var_tuple()}")
        graph = nx.Graph()
        neighbors = list(self.get_neighbors_to_connect())
        # ToDo: speed up by looping over every second lon/lat value?
        for xx in range(len(self.longitude)):
            for yy in range(len(self.latitude)):
                # point on land
                if not self.mask[yy, xx]:
                    continue
                lon = float(self.longitude[xx])
                lat = float(self.latitude[yy])
                # connect the point to its neighbors
                for ii, jj in neighbors:
                    # point outside grid
                    if (xx+ii < 0) or (xx+ii > len(self.longitude)-1) or (yy+jj < 0) or (yy+jj > len(self.latitude)-1):
                        continue
                    lon_ = float(self.longitude[xx+ii])
                    lat_ = float(self.latitude[yy+jj])
                    # point on land (included in "has_point_on_land")
                    # if not self.mask[yy + jj, xx + ii]:
                    #     continue
                    # start/end point or points along the edge on land
                    if self.has_point_on_land(geod.InverseLine(lat, lon, lat_, lon_)):
                        continue
                    # ToDo: check if this is really necessary or if networkX.Graph is doing this check implicitly
                    if graph.has_edge((lon, lat), (lon_, lat_)):
                        continue
                    distance = geod.Inverse(lat, lon, lat_, lon_)['s12']
                    # nodes are added automatically if not already in the graph
                    graph.add_edge((lon, lat), (lon_, lat_), distance=distance)
        # self.save_graph(graph, "dijkstra_graph.geojson")
        return graph

    def has_point_on_land(
            self,
            line: GeodesicLine,
    ) -> bool:
        """
        Check if the line has a point on land. The check is done for points along the line with the configured interval
        and always includes the start and end point of the line.
        :param line:
        :type line: geographiclib.geodesicline.GeodesicLine
        :return:
        :rtype: bool
        """
        n = int(ceil(line.s13 / self.interval))
        for i in range(0, n+1):
            s = min(self.interval * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            if is_land(g['lat2'], g['lon2']):
                return True
        return False

    def read_mask(self, mask_file):
        # Mask shape: 21600 (lat) * 43200 (lon) = 933,120,000
        # Water equals to True, land equals to False
        npz_file = np.load(mask_file)
        self.mask = npz_file['mask']
        self.longitude = npz_file['lon']  # -180,..., 180
        self.latitude = npz_file['lat']  # 90,..., -90
        self.res_lat = abs(self.latitude[0] - self.latitude[1])
        self.res_lon = abs(self.longitude[0] - self.longitude[1])

    def save_graph(self, graph, filename, step=1):
        points_to_geojson([node for node in graph.nodes], filename, step=step)

    def subset_bbox(self):
        min_lon_idx = np.argmin(np.abs(self.longitude - self.map_ext.lon1))
        max_lon_idx = np.argmin(np.abs(self.longitude - self.map_ext.lon2))
        min_lat_idx = np.argmin(np.abs(self.latitude - self.map_ext.lat1))
        max_lat_idx = np.argmin(np.abs(self.latitude - self.map_ext.lat2))
        self.longitude = self.longitude[min(min_lon_idx, max_lon_idx):max(min_lon_idx, max_lon_idx)]
        self.latitude = self.latitude[min(min_lat_idx, max_lat_idx):max(min_lat_idx, max_lat_idx)]
        self.mask = self.mask[min(min_lat_idx, max_lat_idx):max(min_lat_idx, max_lat_idx),
                              min(min_lon_idx, max_lon_idx):max(min_lon_idx, max_lon_idx)]
