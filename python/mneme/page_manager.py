"""
ctypes wrapper for Mneme virtual address space (page manager).

This module provides a thin Python binding around the native Mneme
page manager, responsible for managing a virtual address space used
during record/replay.

Notes:
  - c_uintptr_t is resolved dynamically to match the host pointer size.
  - Lifetime is managed via ffi.ObjectRef; disposal is explicit.
  - This is an internal, low-level API not intended for direct user use.
"""

import ctypes
from ctypes import c_int, c_uint64, c_void_p

from .llvm import ffi

if hasattr(ctypes, "c_uintptr_t"):
    c_uintptr_t = ctypes.c_uintptr_t
else:
    if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_ulonglong):
        c_uintptr_t = ctypes.c_ulonglong
    elif ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_ulong):
        c_uintptr_t = ctypes.c_ulong
    elif ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_uint):
        c_uintptr_t = ctypes.c_uint
    else:
        raise RuntimeError("Could not determine appropriate type for c_uintptr_t")

ffi.lib.MnemePY_initializePageManager.argtypes = [c_int, c_uintptr_t, c_uint64]
ffi.lib.MnemePY_initializePageManager.restype = c_void_p

ffi.lib.MnemePY_getPageManagerVAStart.argtypes = [c_void_p]
ffi.lib.MnemePY_getPageManagerVAStart.restype = c_uintptr_t

ffi.lib.MnemePY_DisposePageManager.argtypes = [c_void_p]


class PageManagerRef(ffi.ObjectRef):
    """
    Handle to a native Mneme Page Manager.

    This object owns a device-specific virtual address space used during
    replay. It is created once per device and disposed explicitly.
    """

    def __init__(self, device_id, va_addr: str, va_size: int):
        self._device_id = device_id
        self._va_addr = int(va_addr, 16)
        self._va_size = va_size
        super(PageManagerRef, self).__init__(
            ffi.lib.MnemePY_initializePageManager(
                self._device_id, self._va_addr, va_size
            )
        )

    def _dispose(self):
        ffi.lib.MnemePY_DisposePageManager(self)

    @property
    def va_start(self) -> int:
        return int(ffi.lib.MnemePY_getPageManagerVAStart(self))
