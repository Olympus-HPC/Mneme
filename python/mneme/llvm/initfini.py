# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ctypes import c_uint

from . import ffi


def initialize():
    ffi.lib.LLVMPY_Initialize()


# =============================================================================
# Set function FFI

ffi.lib.LLVMPY_GetVersionInfo.restype = c_uint


def _version_info():
    v = []
    x = ffi.lib.LLVMPY_GetVersionInfo()
    while x:
        v.append(x & 0xFF)
        x >>= 8
    return tuple(reversed(v))


llvm_version_info = _version_info()
