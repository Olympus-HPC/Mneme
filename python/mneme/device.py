from .llvm import ffi
from .llvm.buffer import MemBufferRef
from .llvm.common import _decode_string, _encode_string

from .recorded_execution import MnemeRecordStateRef

import weakref
from ctypes import POINTER, c_void_p, c_char_p, Structure, c_uint, c_float, c_int
from .mneme_types import dim3


ffi.lib.MnemePY_getDeviceObject.argtypes = [ffi.LLVMMemBufferRef]
ffi.lib.MnemePY_getDeviceObject.restype = c_void_p

ffi.lib.MnemePY_DisposeDeviceObject.argtypes = [c_void_p]

ffi.lib.MnemePY_getKernelFunctionFromImage.argtypes = [c_void_p, c_char_p]
ffi.lib.MnemePY_getKernelFunctionFromImage.restype = c_void_p

ffi.lib.MnemePy_getDeviceArch.argtypes = []
ffi.lib.MnemePy_getDeviceArch.restype = c_char_p


ffi.lib.MnemePY_launchKernelFunction.argtypes = [c_void_p, dim3, dim3]


ffi.lib.MnemePy_profile.argtypes = [
    c_void_p,
    c_void_p,
    dim3,
    dim3,
    MnemeRecordStateRef,
    MnemeRecordStateRef,
    c_int,
    c_int,
    POINTER(c_float),
]


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

    def profile(
        self,
        grid_dim: dim3,
        block_dim: dim3,
        prologue_state: MemBufferRef,
        epilogue_state: MemBufferRef,
        shared_mem_size: int,
        iterations=5,
    ):
        arr = (c_float * iterations)()
        # Set argument types
        DevMod = self._module_ref()
        if DevMod is None:
            raise RuntimeError("Device Module has been garbage collected")

        ffi.lib.MnemePy_profile(
            DevMod,
            self,
            grid_dim,
            block_dim,
            prologue_state,
            epilogue_state,
            shared_mem_size,
            iterations,
            arr,
        )
        return [float(v) for v in arr]


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


def get_device_arch():
    return str(ffi.lib.MnemePy_getDeviceArch().decode())
