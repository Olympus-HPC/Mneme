from .llvm import ffi
from ctypes import c_void_p, c_uint64
import ctypes

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

ffi.lib.MnemePY_initializePageManager.argtypes = [c_uintptr_t, c_uint64]
ffi.lib.MnemePY_initializePageManager.restype = c_void_p

ffi.lib.MnemePY_DisposePageManager.argtypes = [c_void_p]


class PageManagerRef(ffi.ObjectRef):
    def __init__(self, va_addr: str, va_size: int):
        self._va_addr = int(va_addr, 16)
        self._va_size = va_size
        super(PageManagerRef, self).__init__(
            ffi.lib.MnemePY_initializePageManager(self._va_addr, va_size)
        )

    def _dispose(self):
        ffi.lib.MnemePY_DisposePageManager(self)
