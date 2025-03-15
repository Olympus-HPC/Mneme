from .llvm import ffi
from .llvm.buffer import MemBufferRef
from .llvm.common import _decode_string, _encode_string

import weakref
from ctypes import c_void_p, c_char_p, Structure, c_uint

ffi.lib.MnemePY_getDeviceObject.argtypes = [ffi.LLVMMemBufferRef]
ffi.lib.MnemePY_getDeviceObject.restype = c_void_p

ffi.lib.MnemePY_DisposeDeviceObject.argtypes = [c_void_p]

ffi.lib.MnemePY_getKernelFunctionFromImage.argtypes = [c_void_p, c_char_p]
ffi.lib.MnemePY_getKernelFunctionFromImage.restype = c_void_p


class dim3(Structure):
    _fields_ = [("x", c_uint), ("y", c_uint), ("z", c_uint)]

    def __init__(self, x=1, y=1, z=1):
        super().__init__(x, y, z)

    def __repr__(self):
        return f"dim3({self.x}, {self.y}, {self.z})"


ffi.lib.MnemePY_launchKernelFunction.argtypes = [c_void_p, dim3, dim3]


class DeviceFunction(ffi.ObjectRef):
    def __init__(self, func_ptr, module_ref, kernel_name):
        super(DeviceFunction, self).__init__(func_ptr)
        self._module_ref = weakref.ref(module_ref)
        self._kernel_name = kernel_name
        self.valid = True

    def invalidate(self):
        """Marks the function as invalid if the module is unloaded."""
        self.valid = False

    def _dispose(self):
        self.valid = False

    def __del__(self):
        """Ensure cleanup."""
        self.invalidate()

    def launch(self, grid_dim: dim3, block_dim: dim3):
        if self == c_void_p(None):
            raise RuntimeError(
                "hipFunction_t is NULL. The module may have been unloaded."
            )

        if self._module_ref == c_void_p(None):
            raise RuntimeError("hipModule is NULL. The module may have been unloaded.")

        if not self.valid:
            raise RuntimeError("Cannot launch function: module was unloaded.")
        ffi.lib.MnemePY_launchKernelFunction(self, grid_dim, block_dim)


class DeviceModule(ffi.ObjectRef):
    def __init__(self, module_ptr):
        super(DeviceModule, self).__init__(module_ptr)
        self._functions = weakref.WeakSet()

    def _dispose(self):
        self._capi.MnemePY_DisposeDeviceObject(self)
        for func in self._functions:
            func.invalidate()
        self._functions.clear()

    @classmethod
    def from_MemBuffer(cls, buffer: MemBufferRef):
        if not isinstance(buffer, MemBufferRef):
            raise TypeError(
                f"Expecting type of MemBufferRef instead got {type(buffer)}"
            )
        return cls(ffi.lib.MnemePY_getDeviceObject(buffer))

    def get_function(self, kernel_name: str):
        func = ffi.lib.MnemePY_getKernelFunctionFromImage(
            self,
            _encode_string(kernel_name),
        )

        dev_func = DeviceFunction(func, self, kernel_name)
        self._functions.add(dev_func)
        return dev_func
