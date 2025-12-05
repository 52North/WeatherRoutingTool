import json
from itertools import product

import networkx as nx
import numpy as np
from geographiclib.geodesic import Geodesic
from shapely import Point, to_geojson

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


class DijkstraAlgorithm:

    def __init__(self, mask_filename: str):
        # Mask shape: 21600 (lat) * 43200 (lon) = 933,120,000
        # Water equals to True, land equals to False
        npz_file = np.load(mask_filename)
        self.mask = npz_file['mask']
        self.longitude = npz_file['lon']  # -180,..., 180
        self.latitude = npz_file['lat']  # 90,..., -90
        self.res_lat = abs(self.latitude[0] - self.latitude[1])
        self.res_lon = abs(self.longitude[0] - self.longitude[1])
        self.bbox = (min(self.longitude), min(self.latitude), min(self.longitude), min(self.latitude))

    def execute_routing(self, start: tuple[float, float], end: tuple[float, float],
                        bbox: tuple[float, float, float, float] = None, method='dijkstra',
                        no_neighbors: int = 1):
        if bbox:
            # FIXME: might need to reload the mask
            self.subset_bbox(bbox)
        # The start and end points might not be on the grid or in the graph, respectively.
        # Thus, we search the closest point/node.
        start_lon = self.longitude[np.argmin(np.abs(self.longitude - start[0]))]
        start_lat = self.latitude[np.argmin(np.abs(self.latitude - start[1]))]
        end_lon = self.longitude[np.argmin(np.abs(self.longitude - end[0]))]
        end_lat = self.latitude[np.argmin(np.abs(self.latitude - end[1]))]
        print(f"Create graph for bounding box {bbox}")
        graph = nx.Graph()
        neighbors = list(range(-no_neighbors, no_neighbors+1, 1))
        # ToDo: speed up by looping over every second lon/lat value?
        for xx in range(len(self.longitude)):
            for yy in range(len(self.latitude)):
                # point on land
                if not self.mask[yy, xx]:
                    continue
                lon = float(self.longitude[xx])
                lat = float(self.latitude[yy])
                for ii, jj in product(neighbors, neighbors):
                    # FIXME: check land crossing if no_neighbors > 1
                    if ii == jj == 0:
                        continue
                    # point outside grid
                    if  (xx+ii < 0) or (xx+ii > len(self.longitude)-1) or (yy+jj < 0) or (yy+jj > len(self.latitude)-1):
                        continue
                    # point on land
                    if not self.mask[yy + jj, xx + ii]:
                        continue
                    lon_ = float(self.longitude[xx+ii])
                    lat_ = float(self.latitude[yy+jj])
                    if graph.has_edge((lon, lat), (lon_, lat_)):
                        continue
                    distance = geod.Inverse(lat, lon, lat_, lon_)['s12']
                    # nodes are added automatically if not already in the graph
                    graph.add_edge((lon, lat), (lon_, lat_), distance=distance)
        # self.save_graph(graph, "graph_denmark.geojson")
        print(f"Find shortest route using the {method} algorithm")
        path = nx.shortest_path(graph, (start_lon, start_lat), (end_lon, end_lat), weight='distance', method=method)
        # path_length = nx.shortest_path_length(graph, (start_lon, start_lat), (end_lon, end_lat), weight='distance', method=method)
        return path

    def save_graph(self, graph, filename, step=1):
        points_to_geojson([node for node in graph.nodes], filename, step=step)

    def subset_bbox(self, bbox: tuple[float, float, float, float]):
        self.bbox = bbox
        min_lon, min_lat, max_lon, max_lat = self.bbox
        min_lon_idx = np.argmin(np.abs(self.longitude - min_lon))
        max_lon_idx = np.argmin(np.abs(self.longitude - max_lon))
        min_lat_idx = np.argmin(np.abs(self.latitude - min_lat))
        max_lat_idx = np.argmin(np.abs(self.latitude - max_lat))
        self.longitude = self.longitude[min(min_lon_idx, max_lon_idx):max(min_lon_idx, max_lon_idx)]
        self.latitude = self.latitude[min(min_lat_idx, max_lat_idx):max(min_lat_idx, max_lat_idx)]
        self.mask = self.mask[min(min_lat_idx, max_lat_idx):max(min_lat_idx, max_lat_idx),
                              min(min_lon_idx, max_lon_idx):max(min_lon_idx, max_lon_idx)]


if __name__ == "__main__":
    # https://github.com/toddkarin/global-land-mask/blob/master/global_land_mask/globe_combined_mask_compressed.npz
    input_filename = "globe_combined_mask_compressed.npz"
    output_filename = "shortest_route_denmark.geojson"
    # Expected order: (lon_min, lat_min, lon_max, lat_max)
    bbox = (2, 53, 12, 58)
    # bbox = (10.5, 56.97, 11.6, 57.55)
    # Expected order: (longitude, latitude)
    start = (11, 57)
    end = (3, 54)
    # end = (11, 57.5)

    algo = DijkstraAlgorithm(input_filename)
    shortest_path = algo.execute_routing(start, end, bbox, no_neighbors=1)
    points_to_geojson(shortest_path, output_filename, step=1)
