from ctypes import c_size_t

from . import ffi

ffi.lib.LLVMPY_DisposeMemBuffer.argtypes = [ffi.LLVMMemBufferRef]

ffi.lib.LLVMPY_GetMemBufferSize.argtypes = [ffi.LLVMMemBufferRef]
ffi.lib.LLVMPY_GetMemBufferSize.restype = c_size_t


class MemBufferRef(ffi.ObjectRef):
    def __init__(self, buffer_ptr):
        super(MemBufferRef, self).__init__(buffer_ptr)

    def _dispose(self):
        self._capi.LLVMPY_DisposeMemBuffer(self)

    def get_size(self):
        return int(ffi.lib.LLVMPY_GetMemBufferSize(self))
