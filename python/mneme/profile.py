from ctypes import POINTER, c_char_p, c_uint64

from .llvm import ffi
from .llvm.common import _decode_string, _encode_string

ffi.lib.MnemePy_startProfile.argtypes = [c_char_p]
ffi.lib.MnemePy_startProfile.restype = c_uint64

ffi.lib.MnemePy_stopProfile.argtypes = [
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
]

ffi.lib.MnemePy_getNumRecords.argtypes = [
    c_uint64,
]
ffi.lib.MnemePy_getNumRecords.restype = c_uint64


def gpu_profile_start(kernel_name: str):
    return int(ffi.lib.MnemePy_startProfile(_encode_string(kernel_name)))


def gpu_profile_stop(correlation_id: int):
    num_records = ffi.lib.MnemePy_getNumRecords(correlation_id)
    arr = (c_uint64 * num_records)()

    records = ffi.lib.MnemePy_stopProfile(correlation_id, arr, num_records)
    return [int(r) for r in arr]
