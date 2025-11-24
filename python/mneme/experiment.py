import hashlib
import json
import math

import numpy as np


def _analyze_measurements(times):
    """
    times: list or array of measurements (after dropping warm-ups)
    """
    arr = np.array(times, dtype=float)
    N = len(arr)

    # 1. Standard deviation
    std_val = np.std(arr, ddof=1)  # sample std (ddof=1)

    # 2. Arithmetic average (mean)
    avg_val = np.mean(arr)

    # 3. Median (robust "mean value")
    median_val = np.median(arr)

    # 4. R = ratio x_(N-1) / x_(2) where x are sorted samples
    sorted_arr = np.sort(arr)
    if N >= 4:
        R = sorted_arr[-2] / sorted_arr[1]
        R_percent = (R - 1.0) * 100.0
    else:
        R = None
        R_percent = None

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    return (std_val, avg_val, median_val, R, R_percent, iqr, iqr / median_val, q1, q3)

# `Experiment` represents a single configuration of a kernel replay: a specific
# combination of specialization flags, launch-bounds settings, LLVM pass
# pipelines, and codegen options. It also stores all metadata produced during
# replay, including:
#   • optimization/codegen timings and object size
#   • execution-time statistics (median, std, IQR, robustness metrics)
#   • resource usage (registers, local/const memory)
#   • bookkeeping information (start/end timestamps, GPU id, commit ids)
#
# Experiments are uniquely identified by a stable hash of their configuration
# fields. They can be serialized to dictionaries (for database storage) and
# reconstructed from them. In short, this class is the data model used by
# Mneme to describe *what* configuration was run and *what* performance and
# resource metrics were observed.
class Experiment:
    def __init__(self, **kwargs):
        self.specialize = kwargs.pop("specialize")
        self.max_threads = kwargs.pop("max_threads")
        self.min_blocks_per_sm = kwargs.pop("min_blocks_per_sm")
        self.specialize_dims = kwargs.pop("specialize_dims")
        self.passes = kwargs.pop("passes")
        self.prune = kwargs.pop("prune")
        self.internalize = kwargs.pop("internalize")
        self.codegen_opt = kwargs.pop("codegen_opt")
        self.codegen_method = kwargs.pop("codegen_method")
        self._device_arch = kwargs.pop("device_arch")

        self._opt_time = kwargs.pop("opt_time", None)
        self._codegen_time = kwargs.pop("codegen_time", None)
        self._verified = kwargs.pop("verified", None)
        self._obj_size = kwargs.pop("obj_size", None)
        self._exec_time_std = kwargs.pop("exec_time_std", None)
        self._exec_time_avg = kwargs.pop("exec_time_avg", None)
        self._exec_time_median = kwargs.pop("exec_time_median", None)
        self._exec_time_r = kwargs.pop("exec_time_r", None)
        self._exec_time_rp = kwargs.pop("exec_time_rp", None)
        self._exec_time_iqr = kwargs.pop("exec_time_iqr", None)
        self._exec_time_iqrp = kwargs.pop("exec_time_iqrp", None)
        self._exec_time_q25 = kwargs.pop("exec_time_q25", None)
        self._exec_time_q75 = kwargs.pop("exec_time_q75", None)
        self._executed = kwargs.pop("executed", False)
        self._failed = kwargs.pop("failed", False)
        self._start_id = kwargs.pop("start_id", -1)
        self._commit_id = kwargs.pop("commit_id", -1)
        self._start_time = kwargs.pop("start_time", 0)
        self._end_time = kwargs.pop("end_time", 0)
        self._gpu_id = kwargs.pop("gpu_id", 0)
        self._const_mem = kwargs.pop("const_mem", -1)
        self._local_mem = kwargs.pop("local_mem", -1)
        self._reg_usage = kwargs.pop("reg_usage", -1)

    @property
    def const_mem(self):
        return self._const_mem

    @const_mem.setter
    def const_mem(self, value):
        self._const_mem = value

    @property
    def local_mem(self):
        return self._local_mem

    @local_mem.setter
    def local_mem(self, value):
        self._local_mem = value

    @property
    def reg_usage(self):
        return self._reg_usage

    @reg_usage.setter
    def reg_usage(self, value):
        self._reg_usage = value

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self._gpu_id = value

    @property
    def start_id(self):
        return self._start_id

    @start_id.setter
    def start_id(self, value):
        self._start_id = value

    @property
    def commit_id(self):
        return self._commit_id

    @commit_id.setter
    def commit_id(self, value):
        self._commit_id = value

    @property
    def exec_time(self):
        return self._exec_time_median

    @exec_time.setter
    def exec_time(self, value):
        if not isinstance(value, list):
            raise TypeError(
                f"Optimization time expects a list of values {value}, {type(value)}"
            )
        (
            self._exec_time_std,
            self._exec_time_avg,
            self._exec_time_median,
            self._exec_time_r,
            self._exec_time_rp,
            self._exec_time_iqr,
            self._exec_time_iqrp,
            self._exec_time_q25,
            self._exec_time_q75,
        ) = _analyze_measurements(value[2:])

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def obj_size(self):
        return self._obj_size

    @obj_size.setter
    def obj_size(self, value):
        self._obj_size = value

    @property
    def verified(self):
        return self._verified

    @verified.setter
    def verified(self, value):
        self._verified = value

    @property
    def codegen_time(self):
        return self._codegen_time

    @codegen_time.setter
    def codegen_time(self, value):
        self._codegen_time = value

    @property
    def opt_time(self):
        return self._opt_time

    @opt_time.setter
    def opt_time(self, value):
        self._opt_time = value

    @property
    def executed(self):
        return self._executed

    @executed.setter
    def executed(self, value):
        self._executed = value

    @property
    def failed(self):
        return self._failed

    @failed.setter
    def failed(self, value):
        self._failed = value

    @property
    def passes(self) -> str:
        return self._passes

    @passes.setter
    def passes(self, value):
        self._passes = value

    @property
    def specialize(self):
        return self._specialize

    @specialize.setter
    def specialize(self, value):
        if isinstance(value, str):
            if value.lower() == "true":
                self._specialize = True
            else:
                self._specialize = False
        else:
            self._specialize = bool(value)

    @property
    def specialize_dims(self):
        return self._specialize_dims

    @specialize_dims.setter
    def specialize_dims(self, value):
        if isinstance(value, str):
            if value.lower() == "true":
                self._specialize_dims = True
            else:
                self._specialize_dims = False
        else:
            self._specialize_dims = bool(value)

    @property
    def max_threads(self):
        return self._max_threads

    @max_threads.setter
    def max_threads(self, value):
        if value is None:
            self._max_threads = None
        else:
            self._max_threads = int(value)

    @property
    def min_blocks_per_sm(self):
        return self._min_blocks_per_sm

    @min_blocks_per_sm.setter
    def min_blocks_per_sm(self, value):
        if value is None:
            self._min_blocks_per_sm = None
        else:
            self._min_blocks_per_sm = int(value)

    @property
    def prune(self):
        return self._prune

    @prune.setter
    def prune(self, value):
        if isinstance(value, str):
            if value.lower() == "true":
                self._prune = True
            else:
                self._prune = False
        else:
            self._prune = bool(value)

    @property
    def internalize(self):
        return self._internalize

    @internalize.setter
    def internalize(self, value):
        if isinstance(value, str):
            if value.lower() == "true":
                self._internalize = True
            else:
                self._internalize = False
        else:
            self._internalize = bool(value)

    @property
    def codegen_opt(self):
        return self._codegen_opt

    @codegen_opt.setter
    def codegen_opt(self, value):
        if value is None:
            self._codegen_opt = None
        else:
            self._codegen_opt = int(value)

    @property
    def codegen_method(self):
        return self._codegen_method

    @codegen_method.setter
    def codegen_method(self, value):
        self._codegen_method = value

    def hash(self):
        hasher = hashlib.sha256()
        hasher.update(str(self._specialize).encode("utf-8"))
        hasher.update(str(self._max_threads).encode("utf-8"))
        hasher.update(str(self._min_blocks_per_sm).encode("utf-8"))
        hasher.update(str(self._specialize_dims).encode("utf-8"))
        hasher.update(str(self._passes).encode("utf-8"))
        hasher.update(str(self._prune).encode("utf-8"))
        hasher.update(str(self._internalize).encode("utf-8"))
        hasher.update(str(self._codegen_opt).encode("utf-8"))
        hasher.update(str(self._codegen_method).encode("utf-8"))
        hasher.update(str(self._device_arch).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)

    def to_dict(self):
        data = {}
        data["specialize"] = self._specialize
        data["max_threads"] = self._max_threads
        data["min_blocks_per_sm"] = self._min_blocks_per_sm
        data["specialize_dims"] = self._specialize_dims
        data["passes"] = self._passes
        data["prune"] = self._prune
        data["internalize"] = self._internalize
        data["codegen_opt"] = self._codegen_opt
        data["codegen_method"] = self._codegen_method
        data["device_arch"] = self._device_arch
        data["failed"] = self.failed
        data["start_time"] = self._start_time
        data["end_time"] = self._end_time
        data["commit_id"] = self._commit_id
        data["start_id"] = self._start_id
        data["gpu_id"] = self._gpu_id

        data["opt_time"] = self._opt_time
        data["codegen_time"] = self._codegen_time
        data["verified"] = self._verified
        data["obj_size"] = self._obj_size
        data["exec_time_std"] = self._exec_time_std
        data["exec_time_avg"] = self._exec_time_avg
        data["exec_time_median"] = self._exec_time_median
        data["exec_time_r"] = self._exec_time_r
        data["exec_time_rp"] = self._exec_time_rp
        data["exec_time_iqr"] = self._exec_time_iqr
        data["exec_time_iqrp"] = self._exec_time_iqrp
        data["exec_time_q25"] = self._exec_time_q25
        data["exec_time_q75"] = self._exec_time_q75
        data["executed"] = self._executed
        data["hash"] = self.hash()
        data["reg_usage"] = self._reg_usage
        data["const_mem"] = self._const_mem
        data["local_mem"] = self._local_mem
        return data

    def dump(self):
        print(json.dumps(self.to_dict(), indent=6))
