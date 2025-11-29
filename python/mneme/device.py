import weakref
from ctypes import POINTER, Structure, c_char_p, c_float, c_int, c_uint, c_void_p

from .llvm import ffi
from .llvm.buffer import MemBufferRef
from .llvm.common import _decode_string, _encode_string
from .mneme_types import dim3
from .recorded_execution import MnemeRecordStateRef

ffi.lib.MnemePY_getDeviceObject.argtypes = [ffi.LLVMMemBufferRef]
ffi.lib.MnemePY_getDeviceObject.restype = c_void_p

ffi.lib.MnemePY_DisposeDeviceObject.argtypes = [c_void_p]

ffi.lib.MnemePY_getKernelFunctionFromImage.argtypes = [c_void_p, c_char_p]
ffi.lib.MnemePY_getKernelFunctionFromImage.restype = c_void_p

ffi.lib.MnemePy_getDeviceArch.argtypes = []
ffi.lib.MnemePy_getDeviceArch.restype = c_char_p


ffi.lib.MnemePy_launchKernelFunction.argtypes = [c_void_p, dim3, dim3]

ffi.lib.MnemePy_getDeviceCount.argtypes = []
ffi.lib.MnemePy_getDeviceCount.restype = c_int


ffi.lib.MnemePy_setDevice.argtypes = [c_int]


ffi.lib.MnemePy_profile.argtypes = [
    c_void_p,
    c_void_p,
    dim3,
    dim3,
    MnemeRecordStateRef,
    MnemeRecordStateRef,
    c_int,
    c_int,
]


ffi.lib.MnemePy_getRegisterUsage.argtypes = [c_void_p]
ffi.lib.MnemePy_getRegisterUsage.restype = c_int
ffi.lib.MnemePy_getLocalMemUsage.argtypes = [c_void_p]
ffi.lib.MnemePy_getLocalMemUsage.restype = c_int
ffi.lib.MnemePy_getConstMemUsage.argtypes = [c_void_p]
ffi.lib.MnemePy_getConstMemUsage.restype = c_int


class DeviceFunction(ffi.ObjectRef):
    def __init__(self, func_ptr, module_ref, kernel_name):
        super(DeviceFunction, self).__init__(func_ptr)
        self._module_ref = weakref.ref(module_ref)
        self._kernel_name = kernel_name
        self.valid = True
        self._local_mem = None
        self._reg_usage = None
        self._const_mem = None

    @property
    def local_mem(self):
        if self._local_mem is None:
            self._local_mem = ffi.lib.MnemePy_getLocalMemUsage(self)
        return self._local_mem

    @property
    def const_mem(self):
        if self._const_mem is None:
            self._const_mem = ffi.lib.MnemePy_getConstMemUsage(self)
        return self._const_mem

    @property
    def reg_usage(self):
        if self._reg_usage is None:
            self._reg_usage = ffi.lib.MnemePy_getRegisterUsage(self)
        return self._reg_usage

    def invalidate(self):
        """Marks the function as invalid if the module is unloaded."""
        self.valid = False

    def _dispose(self):
        self.valid = False

    def __del__(self):
        """Ensure cleanup."""
        self.invalidate()

    def launch(self, grid_dim: dim3, block_dim: dim3):
          # Correct NULL check
        if self._ptr is None or self._ptr.value is None:
            raise RuntimeError("hipFunction_t is NULL. The module may have been unloaded.")

        mod = self._module_ref()
        if mod is None:
            raise RuntimeError("hipModule is NULL. The module may have been unloaded.")

        if not self.valid:
            raise RuntimeError("Cannot launch function: module was unloaded.")
        ffi.lib.MnemePy_launchKernelFunction(self, grid_dim, block_dim)

    def profile(
        self,
        grid_dim: dim3,
        block_dim: dim3,
        prologue_state: MemBufferRef,
        epilogue_state: MemBufferRef,
        shared_mem_size: int,
        iterations=5,
    ):
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
        )
        return


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


def get_device_count():
    return int(ffi.lib.MnemePy_getDeviceCount())


def set_device(dev_id: int):
    ffi.lib.MnemePy_setDevice(dev_id)


def get_max_blocks_per_sm():
    return 8
