# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Device runtime bindings for Mneme (Python ↔ native runtime).

This module provides thin ctypes/FFI wrappers around the Mneme native runtime
for loading device code objects and launching / profiling kernels.

Main abstractions
-----------------
- :class:`DeviceModule` loads a device object from a :class:`MemBufferRef` and
  provides access to kernel entry points.
- :class:`DeviceFunction` represents a kernel function handle and exposes
  ``launch`` and ``profile`` operations, along with basic resource-usage
  queries (registers, local/const memory).

Lifetime notes
--------------
The native runtime owns device-side resources. This module mirrors those
resources via ``ffi.ObjectRef`` and uses weak references to enforce correct
cleanup order:

- A :class:`DeviceFunction` keeps only a weakref to its parent module.
- When a :class:`DeviceModule` is disposed/unloaded, it invalidates all
  functions obtained from it to prevent use-after-free.
"""

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
    """
    Handle to a device kernel function.

    A DeviceFunction is obtained from a :class:`DeviceModule` via
    :meth:`DeviceModule.get_function`. It provides:

      - :meth:`launch` for a direct kernel launch (grid/block only)
      - :meth:`profile` for record/replay execution with prologue/epilogue state
      - Resource-usage queries via :attr:`reg_usage`, :attr:`local_mem`,
        and :attr:`const_mem`

    Notes
    -----
    Instances are tied to the lifetime of the parent :class:`DeviceModule`.
    If the module is unloaded, the function becomes invalid and further usage
    raises an error.
    """

    def __init__(self, func_ptr, module_ref, kernel_name):
        """
        Parameters
        ----------
        func_ptr
            Native function handle returned by the Mneme runtime.
        module_ref : DeviceModule
            Parent module that owns the function. A weak reference is stored.
        kernel_name : str
            Kernel symbol name (used for debugging/logging and attribution).
        """
        super(DeviceFunction, self).__init__(func_ptr)
        self._module_ref = weakref.ref(module_ref)
        self._kernel_name = kernel_name
        self.valid = True
        self._local_mem = None
        self._reg_usage = None
        self._const_mem = None

    @property
    def local_mem(self):
        """
        Local memory usage for this kernel (bytes), as reported by the runtime.

        The value is cached after the first query.

        Returns
        -------
        int
            Local memory usage in bytes.
        """
        if self._local_mem is None:
            self._local_mem = ffi.lib.MnemePy_getLocalMemUsage(self)
        return self._local_mem

    @property
    def const_mem(self):
        """
        Constant memory usage for this kernel (bytes), as reported by the runtime.

        The value is cached after the first query.

        Returns
        -------
        int
            Constant memory usage in bytes.
        """
        if self._const_mem is None:
            self._const_mem = ffi.lib.MnemePy_getConstMemUsage(self)
        return self._const_mem

    @property
    def reg_usage(self):
        """
        Register usage for this kernel (registers per thread), as reported by the runtime.

        The value is cached after the first query.

        Returns
        -------
        int
            Register usage per thread.
        """
        if self._reg_usage is None:
            self._reg_usage = ffi.lib.MnemePy_getRegisterUsage(self)
        return self._reg_usage

    def invalidate(self):
        """
        Mark this function handle as invalid.

        This is called when the owning :class:`DeviceModule` is unloaded/disposed,
        preventing use-after-free of the underlying native function handle.
        """
        self.valid = False

    def _dispose(self):
        """
        Dispose hook for ffi.ObjectRef.

        This implementation marks the function as invalid. The parent module owns
        native resources, so function disposal is primarily a logical invalidation.
        """
        self.valid = False

    def __del__(self):
        """
        Best-effort cleanup on garbage collection.

        Notes
        -----
        Python finalizers are not guaranteed to run promptly (or at all at process
        shutdown). Resource lifetime should be controlled through the parent
        :class:`DeviceModule` context manager whenever possible.
        """
        self.invalidate()

    def launch(self, grid_dim: dim3, block_dim: dim3):
        """
        Launch the kernel with the given grid/block configuration.

        Parameters
        ----------
        grid_dim : dim3
            Grid dimensions for the launch.
        block_dim : dim3
            Block dimensions for the launch.

        Raises
        ------
        RuntimeError
            If the function pointer is NULL, the parent module was unloaded, or the
            function has been invalidated.
        """

        if self._ptr is None or self._ptr.value is None:
            raise RuntimeError(
                "hipFunction_t is NULL. The module may have been unloaded."
            )

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
        """
        Execute the kernel under Mneme record/replay profiling.

        This method executes the kernel with the provided recorded prologue/epilogue
        state buffers and measures execution time across multiple iterations. The
        native runtime is responsible for timing, validation hooks, and any device
        synchronization required for consistent measurements.

        Parameters
        ----------
        grid_dim : dim3
            Grid dimensions for the launch.
        block_dim : dim3
            Block dimensions for the launch.
        prologue_state : MemBufferRef
            Recorded state buffer to initialize device memory / arguments prior to
            kernel execution.
        epilogue_state : MemBufferRef
            Recorded state buffer used to validate and/or capture post-state after
            kernel execution.
        shared_mem_size : int
            Dynamic shared memory size (bytes) for the launch.
        iterations : int, optional
            Number of kernel executions to perform for profiling.

        Raises
        ------
        RuntimeError
            If the parent module has been garbage collected.
        """
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
    """
    Loaded device module/object.

    A DeviceModule is constructed from a compiled object buffer (a :class:`MemBufferRef`)
    using :meth:`from_MemBuffer`. It owns the native device object handle and is
    responsible for releasing it.

    Functions obtained via :meth:`get_function` are tracked and invalidated when
    the module is disposed to prevent use-after-free.
    """

    def __init__(self, module_ptr):
        """
        Parameters
        ----------
        module_ptr
            Native module handle returned by the Mneme runtime.
        """
        super(DeviceModule, self).__init__(module_ptr)
        self._functions = weakref.WeakSet()

    def _dispose(self):
        """
        Dispose the underlying native module and invalidate dependent functions.

        This method is invoked by the ``ffi.ObjectRef`` disposal machinery and is
        responsible for:
          1) releasing the native device object handle
          2) invalidating all :class:`DeviceFunction` instances created from this module
        """
        self._capi.MnemePY_DisposeDeviceObject(self)
        for func in self._functions:
            func.invalidate()
        self._functions.clear()

    @classmethod
    def from_MemBuffer(cls, buffer: MemBufferRef):
        """
        Load a device module from an in-memory object buffer.

        Parameters
        ----------
        buffer : MemBufferRef
            Buffer containing a device object suitable for loading by the Mneme runtime.

        Returns
        -------
        DeviceModule
            A module that owns the loaded native device object.

        Raises
        ------
        TypeError
            If ``buffer`` is not a :class:`MemBufferRef`.
        """
        if not isinstance(buffer, MemBufferRef):
            raise TypeError(
                f"Expecting type of MemBufferRef instead got {type(buffer)}"
            )
        return cls(ffi.lib.MnemePY_getDeviceObject(buffer))

    def get_function(self, kernel_name: str):
        """
        Resolve a kernel function from the loaded module.

        Parameters
        ----------
        kernel_name : str
            Kernel symbol name to resolve.

        Returns
        -------
        DeviceFunction
            Function handle bound to this module.

        Notes
        -----
        The returned function is tracked by the module and will be invalidated when
        the module is disposed.
        """
        func = ffi.lib.MnemePY_getKernelFunctionFromImage(
            self,
            _encode_string(kernel_name),
        )

        dev_func = DeviceFunction(func, self, kernel_name)
        self._functions.add(dev_func)
        return dev_func


def get_device_arch():
    """
    Return the current device architecture identifier.

    Returns
    -------
    str
        Architecture string reported by the Mneme runtime (backend-defined).
    """
    return str(ffi.lib.MnemePy_getDeviceArch().decode())


def get_device_count():
    """
    Return the number of visible devices.

    Returns
    -------
    int
        Device count as reported by the Mneme runtime.
    """
    return int(ffi.lib.MnemePy_getDeviceCount())


def set_device(dev_id: int):
    """
    Set the active device for subsequent device operations.

    Parameters
    ----------
    dev_id : int
        Device index to select.
    """
    ffi.lib.MnemePy_setDevice(dev_id)
