# Warning:
# The minimum cost path provided by this script does not correspond to the shortest-distance path!
# The cost grid is equally-spaced (0,008333 degrees), thus longitudes are not equidistant but their distance is reduced
# towards the poles. Other problems might be caused by staircase effects.
import json
import time

import numpy as np
from global_land_mask import lat_to_index, lon_to_index
from shapely import Point, to_geojson
from skimage.graph import route_through_array
import matplotlib.pyplot as plt

import WeatherRoutingTool.utils.unit_conversion as unit


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


def indices_to_geojson(point_indices: list[tuple[int, int]], lon, lat, filename: str = None, step: int = 10):
    d = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": i},
                "geometry": json.loads(
                    to_geojson(Point(lon[point_indices[i][1]], lat[point_indices[i][0]]))
                )
            }
            for i in range_including_last_element(point_indices, step)
        ]
    }
    if filename:
        with open(filename, 'w') as f:
            json.dump(d, f, indent=2)
    return d


if __name__ == "__main__":

    start_time = time.time()
    # https://github.com/toddkarin/global-land-mask/blob/master/global_land_mask/globe_combined_mask_compressed.npz
    input_filename = "globe_combined_mask_compressed.npz"

    # Expected order: [lat, lon]
    start = [52.286, 3.342]
    end = [57.679, 11.293]

    # reach of the bounding box
    start_lat, start_lon, end_lat, end_lon = (50, -2, 60, 14.257)

    debug = False

    # ------------------------------------

    # Mask shape: 21600 * 43200 = 933,120,000
    # Water equals to True/1, land equals to False/0
    npz_file = np.load(input_filename)
    mask = npz_file['mask']
    mask = mask.astype(int)
    mask[mask == 0] = -1

    lon = npz_file['lon']
    lat = npz_file['lat']
    lat_mirror = np.flip(lat)   # inverse order needed for unit.get_coord_index

    # calculate correct coordinate indices for the slicing
    map_start_lat_orig, map_end_lat_orig = unit.get_coord_index(end_lat, start_lat, lat_mirror)
    map_start_lat = len(lat) - map_end_lat_orig - 1  # consider ordering of lat values from 90° - -90°
    map_end_lat = len(lat) - map_start_lat_orig - 1
    map_start_lon, map_end_lon = unit.get_coord_index(start_lon, end_lon, lon)
    if debug:
        print('lon_start_test. ', lon[map_start_lon])
        print('lon_end_test: ', lon[map_end_lon])
        print('lat_start_test. ', lat[map_start_lat])
        print('lat_end_test: ', lat[map_end_lat])
        print('lat_start_test_mirror: ', lat_mirror[map_start_lat_orig])
    assert lat_mirror[map_start_lat_orig] == lat[map_end_lat]
    assert lat_mirror[map_end_lat_orig] == lat[map_start_lat]

    # perform the slicing
    lon = lon[map_start_lon:map_end_lon + 1]
    lat = lat[map_end_lat:map_start_lat + 1]
    lat_mirror = np.flip(lat)

    mask = mask[map_end_lat:map_start_lat + 1, map_start_lon:map_end_lon + 1]
    if debug:
        print('mask.shape: ', mask.shape)
        print('lat.shape: ', lat.shape)
        print('lon.shape: ', lon.shape)

    assert mask.shape[0] == lat.shape[0]
    assert mask.shape[1] == lon.shape[0]

    # calculate correct coord indices for start and end point after the slicing
    route_start_lat_orig, route_end_lat_orig = unit.get_coord_index(end[0], start[0], lat_mirror)
    route_start_lat = len(lat) - route_end_lat_orig - 1
    route_end_lat = len(lat) - route_start_lat_orig - 1
    route_start_lon, route_end_lon = unit.get_coord_index(start[1], end[1], lon)

    assert lat_mirror[route_start_lat_orig] == lat[route_end_lat]
    assert lat_mirror[route_end_lat_orig] == lat[route_start_lat]

    if debug:
        print('route_start_lat: ', lat[route_start_lat])
        print('route_end_lat: ', lat[route_end_lat])
        print('route_start_lon: ', lon[route_start_lon])
        print('route_end_lon: ', lon[route_end_lon])

    start_idx = [route_start_lat, route_start_lon]
    end_idx = [route_end_lat, route_end_lon]

    # execute route-finding algorithm
    indices, weight = route_through_array(mask, start_idx, end_idx, fully_connected=True, geometric=True)
    indices_to_geojson(indices, lon, lat, "minimum_cost_path.geojson")

    end_time = time.time()
