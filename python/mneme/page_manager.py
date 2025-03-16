from .llvm import ffi
from ctypes import c_void_p, c_uint64, c_uintptr_t

ffi.lib.MnemePY_initializePageManager.argtypes = [c_uintptr_t, c_uint64]
ffi.lib.MnemePY_initializePageManager.restype = c_void_p

ffi.lib.MnemePY_DisposePageManager.argtypes = [c_void_p]


class PageManager(ffi.ObjectRef):
    def __init__(self, va_addr: int, va_size: int):
        super(PageManager, self).__init__(ffi.lib.MnemePY_initializePageManager())
        self._va_addr = va_addr
        self._va_size = va_size

    def _dispose(self):
        ffi.lib.MnemePY_DisposePageManager(self)
