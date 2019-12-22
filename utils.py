import sys
import pickle
import functools

import numpy as np

def active_persist_to_file(filename):
    def decorator(original_func):
        cache = None

        @functools.wraps(original_func)
        def new_func(*params):
            nonlocal cache

            if cache is None:
                try:
                    cache = pickle.load(open(filename, 'rb'))
                except (IOError, ValueError):
                    cache = {}

            if params not in cache:
                cache[params] = original_func(*params)
                try:
                    pickle.dump(cache, open(filename, 'wb'))
                except:
                    sys.stderr.write("Couldn't save {}!".format(filename))
            return cache[params]

        return new_func

    return decorator

def persist_to_file(filename):
    def decorator(original_func):

        @functools.wraps(original_func)
        def new_func(*params):
            # print('Loading', params, 'from', filename.format(*params))
            try:
                cache = pickle.load(open(filename.format(*params), 'rb'))
            except (IOError, ValueError):
                cache = {}

            if params not in cache:
                cache[params] = original_func(*params)
                try:
                    pickle.dump(cache, open(filename.format(*params), 'wb'))
                except:
                    sys.stderr.write("Couldn't save {}!".format(filename.format(*params)))
            return cache[params]

        return new_func

    return decorator

def normalize(vec):
    return vec / np.linalg.norm(vec)
