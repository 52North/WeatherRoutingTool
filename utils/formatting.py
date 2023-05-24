"""Utility functions."""
import datetime
import time
from json import JSONEncoder

import numpy


def print_line():
    print('---------------------------------------------------')

def print_step(stepnote, istep=0):
    step = "   " * (istep+1) + stepnote
    print(step)

def get_log_step(stepnote, istep=0):
    step = "   " * (istep+1) + stepnote
    return step

def print_current_time(function : str, start_time : time.time):
    time_passed=time.time()-start_time
    print('Time after ' + function + ':' + str(time_passed))

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return str(obj)
        if isinstance(obj, numpy.int64):
            return str(obj)
        return JSONEncoder.default(self, obj)
