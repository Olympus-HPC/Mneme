import time
from functools import wraps

from mneme.profile import gpu_profile_start, gpu_profile_stop


def cond_time(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, exp, *args, profile: bool = False, **kwargs):
            if not profile:
                return func(self, exp, *args, **kwargs)
            start = time.perf_counter()
            result = func(self, exp, *args, **kwargs)
            end = time.perf_counter()
            setattr(exp, attr_name, end - start)
            return result

        return wrapper

    return decorator


def cond_gpu_time(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, exp, kernel_name, *args, profile: bool = False, **kwargs):
            if not profile:
                print("Selected to not profile, skipping")
                return func(self, exp, kernel_name, *args, **kwargs)
            print("Profiling")
            correlation_id = gpu_profile_start(kernel_name + ".kd")
            result = func(self, exp, kernel_name, *args, **kwargs)
            measurements = gpu_profile_stop(correlation_id)
            print(f"Profile returned following measurements {measurements}")
            setattr(exp, attr_name, measurements)
            return result

        return wrapper

    return decorator
