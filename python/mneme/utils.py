import json
import time
from dataclasses import fields
from functools import wraps

from mneme.profile import gpu_profile_start, gpu_profile_stop


def cond_time(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, result, config, *args, profile: bool = False, **kwargs):
            if not profile:
                return func(self, result, config, *args, **kwargs)
            start = time.perf_counter()
            fresult = func(self, result, config, *args, **kwargs)
            end = time.perf_counter()
            setattr(result, attr_name, end - start)
            return fresult

        return wrapper

    return decorator


def cond_gpu_time(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(
            self, result, config, kernel_name, *args, profile: bool = False, **kwargs
        ):
            if not profile:
                return func(self, result, config, kernel_name, *args, **kwargs)
            correlation_id = gpu_profile_start(kernel_name + ".kd")
            fresult = func(self, result, config, kernel_name, *args, **kwargs)
            measurements = gpu_profile_stop(correlation_id)
            setattr(result, attr_name, measurements)
            return fresult

        return wrapper

    return decorator


class MnemeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
