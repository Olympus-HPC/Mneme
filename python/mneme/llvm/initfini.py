from ctypes import c_uint

from llvm4ml.binding import ffi


def initialize():
    ffi.lib.LLVMPY_Initialize();


# =============================================================================
# Set function FFI

ffi.lib.LLVMPY_GetVersionInfo.restype = c_uint


def _version_info():
    v = []
    x = ffi.lib.LLVMPY_GetVersionInfo()
    while x:
        v.append(x & 0xff)
        x >>= 8
    return tuple(reversed(v))


llvm_version_info = _version_info()
