"""Utility functions."""
import datetime
import time
from json import JSONEncoder

import numpy


def get_line_string():
    return '---------------------------------------------------'


def print_line():
    print(get_line_string())


def print_step(stepnote, istep=0):
    step = "   " * (istep + 1) + stepnote
    print(step)


def get_log_step(stepnote, istep=0):
    step = "    " * (istep + 1) + stepnote
    return step


def print_current_time(function: str, start_time: time.time):
    time_passed = time.time() - start_time
    print('Time after ' + function + ':' + str(time_passed))


def get_point_from_string(point):
    lat, lon = point.split(',')
    return float(lat), float(lon)


def get_bbox_from_string(bbox):
    if bbox == '-99':
        return 0., 0., 0., 0.
    lat_start, lon_start, lat_end, lon_end = bbox.split(',')
    return float(lat_start), float(lon_start), float(lat_end), float(lon_end)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime.date, datetime.datetime)):
            obj_str = obj.strftime("%Y-%m-%d %H:%M:%S")
            return obj_str
        if isinstance(obj, numpy.int64):
            return str(obj)
        if isinstance(obj, numpy.int32):
            return str(obj)
        return JSONEncoder.default(self, obj)
