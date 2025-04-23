from . import ffi

ffi.lib.LLVMPY_DisposeMemBuffer.argtypes = [ffi.LLVMMemBufferRef]


class MemBufferRef(ffi.ObjectRef):
    def __init__(self, buffer_ptr):
        super(MemBufferRef, self).__init__(buffer_ptr)

    def _dispose(self):
        self._capi.LLVMPY_DisposeMemBuffer(self)
