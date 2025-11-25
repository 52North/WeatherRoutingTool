import json

import numpy as np
from global_land_mask import lat_to_index, lon_to_index
from shapely import Point, to_geojson
from skimage.graph import route_through_array


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
    # https://github.com/toddkarin/global-land-mask/blob/master/global_land_mask/globe_combined_mask_compressed.npz
    input_filename = "globe_combined_mask_compressed.npz"

    # Mask shape: 21600 * 43200 = 933,120,000
    # Water equals to True/1, land equals to False/0
    npz_file = np.load(input_filename)
    mask = npz_file['mask']
    mask = mask.astype(int)
    mask[mask == 0] = 1000
    lon = npz_file['lon']
    lat = npz_file['lat']

    # Expected order: [lat, lon]
    start = [41, 2.3]
    end = [32, 32]
    start_idx = [lat_to_index(start[0]), lon_to_index(start[1])]
    end_idx = [lat_to_index(end[0]), lon_to_index(end[1])]

    indices, weight = route_through_array(mask, start_idx, end_idx, fully_connected=True, geometric=True)
    indices_to_geojson(indices, lon, lat, "minimum_cost_path.geojson")
