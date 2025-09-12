import os
import sys
from ctypes import POINTER, c_char_p, c_uint64

from mneme.logging import logger

from .llvm import ffi
from .llvm.common import _decode_string, _encode_string
from .llvm.utils import get_profile_library

profile_lib = None


def _init_profile():
    # We need to lazily import mneme.profile. This is because hsa initializes the tool
    # at library loading (static global initialization) time. We need this initialization to
    # happen after forking the workers. So, we wrap the loading when we start measuring.
    global profile_lib
    if profile_lib is None:
        profile_lib = ffi.load_library_so(get_profile_library())

        profile_lib.MnemePy_startProfile.argtypes = [c_char_p]
        profile_lib.MnemePy_startProfile.restype = c_uint64

        profile_lib.MnemePy_stopProfile.argtypes = [
            c_uint64,
            POINTER(c_uint64),
            c_uint64,
        ]

        profile_lib.MnemePy_getNumRecords.argtypes = [
            c_uint64,
        ]
        profile_lib.MnemePy_getNumRecords.restype = c_uint64

        profile_lib.MnemePy_initProfiler.argtypes = []


def gpu_profile_start(kernel_name: str):
    if profile_lib is None:
        raise RuntimeError("Profile library is not initialized")
    return int(profile_lib.MnemePy_startProfile(_encode_string(kernel_name)))


def gpu_profile_stop(correlation_id: int):
    if profile_lib is None:
        raise RuntimeError("Profile library is not initialized")

    num_records = profile_lib.MnemePy_getNumRecords(correlation_id)
    logger.debug(f"Profiler contains {num_records} records")
    arr = (c_uint64 * num_records)()

    records = profile_lib.MnemePy_stopProfile(correlation_id, arr, num_records)
    return [int(r) for r in arr]


def init_profiler():
    _init_profile()
    # profile_lib.MnemePy_initProfiler()
