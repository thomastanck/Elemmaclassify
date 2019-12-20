import pickle

def persist_to_file(file_name):
    def decorator(original_func):
        try:
            cache = pickle.load(open(file_name, 'rb'))
        except (IOError, ValueError):
            cache = {}

        def new_func(param):
            if param not in cache:
                cache[param] = original_func(param)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[param]

        return new_func

    return decorator

import numpy as np

def normalize(vec):
    return vec / np.linalg.norm(vec)
