import json
from ctypes import POINTER, c_bool, c_char_p, c_int, c_void_p
from enum import Enum
from pathlib import Path
from typing import Dict, List

from .llvm import ffi
from .mneme_types import dim3
from .proteus import jit

MnemeRecordStateRef = ffi._make_opaque_ref("MnemeRecordState")
ffi.lib.MnemePy_initializeMemState.argtypes = [c_char_p, c_char_p, c_bool]
ffi.lib.MnemePy_initializeMemState.restype = MnemeRecordStateRef

ffi.lib.MnemePy_DisposeMemState.argtypes = [MnemeRecordStateRef]

ffi.lib.MnemePy_LoadMemState.argtypes = [MnemeRecordStateRef]

ffi.lib.MnemePy_CompareMemState.argtypes = [MnemeRecordStateRef, MnemeRecordStateRef]
ffi.lib.MnemePy_CompareMemState.restype = c_bool

ffi.lib.MnemePy_ResetMemState.argtypes = [MnemeRecordStateRef]

ffi.lib.MnemePy_getNumArgs.argtypes = [MnemeRecordStateRef]
ffi.lib.MnemePy_getNumArgs.restype = c_int

ffi.lib.MnemePy_getArgs.argtypes = [MnemeRecordStateRef]
ffi.lib.MnemePy_getArgs.restype = POINTER(c_void_p)


class SnapshotType(Enum):
    PROLOGUE = 1
    EPILOGUE = 2


class MemStateRef:
    def __init__(self, fn: str, kernel_name: str, snap_type: SnapshotType):
        if not Path(fn).exists():
            raise RuntimeError(f"Expected prologue file: {fn} to exist")
        self.fn = fn
        self.kernel_name = kernel_name
        self.s_type = snap_type
        self._state = None
        self._load = False
        self._num_args = None
        self._args = None

    def _dispose(self):
        if self._state is not None:
            ffi.lib.MnemePy_DisposeMemState(self._state)

    def open(self):
        if self._state is None:
            self._state = ffi.lib.MnemePy_initializeMemState(
                c_char_p(self.kernel_name.encode("utf-8")),
                c_char_p(self.fn.encode("utf-8")),
                c_bool(self.s_type == SnapshotType.PROLOGUE),
            )

        ffi.lib.MnemePy_LoadMemState(self._state)
        self._load = True
        return self

    @property
    def args(self):
        if not self._load:
            raise RuntimeError("Cannot access arguments without loading memory state")

        if self._args == None:
            self._args = ffi.lib.MnemePy_getArgs(self._state)

        return self._args

    @property
    def num_args(self):
        if not self._load:
            raise RuntimeError("Cannot access num_args without loading memory state")

        if self._num_args == None:
            self._num_args = ffi.lib.MnemePy_getNumArgs(self._state)

        return self._num_args

    def close(self):
        self._dispose()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def reset(self):
        if self._load is False:
            raise RuntimeError("Cannot reset memory, if state is not read first.")

        ffi.lib.MnemePy_ResetMemState(self._state)

    def __eq__(self, other):
        return bool(ffi.lib.MnemePy_CompareMemState(self._state, other._state))

    def __ne__(self, other):
        return not bool(ffi.lib.MnemePy_CompareMemState(self._state, other._state))

    def __del__(self):
        self._dispose()


class RecordedExecution:
    class KernelInstance:
        def __init__(
            self,
            static_hash: str,
            dynamic_hash: str,
            kernel_name: str,
            args: List,
            shared_mem: int,
            block_dim: dim3,
            grid_dim: dim3,
            available_specializations: List[bool],
            occ: int,
            prologue_fn: str,
            epilogue_fn: str,
        ):
            self.static_hash = static_hash
            self.dynamic_hash = dynamic_hash
            self.kernel_name = kernel_name
            self.args = args
            self.shared_mem = shared_mem
            self.block_dim = block_dim
            self.grid_dim = grid_dim
            self.available_specializations = []
            for i, v in enumerate(available_specializations):
                if v:
                    self.available_specializations.append(i)
            self.occ = occ
            self.prologue = MemStateRef(prologue_fn, kernel_name, SnapshotType.PROLOGUE)
            self.epilogue = MemStateRef(epilogue_fn, kernel_name, SnapshotType.EPILOGUE)

        def __hash__(self):
            return hash(self.dynamic_hash + self.static_hash)

        def __str__(self):
            return f"Grid:{self.grid_dim}, BlockDim: {self.block_dim}, Shared Memory {self.shared_mem}"

    def __init__(
        self,
        static_hash: str,
        kernel_name: str,
        demangled_name: str,
        llvm_files: List[str],
        arg_names: List[str],
        available_specializations: List[bool],
        va_addr: str,
        va_size: int,
        kernel_instances: Dict[str, KernelInstance],
    ):
        self.static_hash = static_hash
        self.kernel_name = kernel_name
        self.demangled_name = demangled_name
        self.llvm_files = llvm_files
        self.arg_names = arg_names
        self.available_specializations = available_specializations
        self.va_addr = va_addr
        self.va_size = va_size
        self.kernel_instances = kernel_instances
        self._link_mod = None

    def __str__(self):
        return f"KernelName: {self.kernel_name} NumArgs: {len(self.arg_names)}, VASize: {self.va_size}, VAddr: {self.va_addr}"

    def __getitem__(self, key):
        return self.kernel_instances[key]

    def __setitem__(self, key, value):
        self.kernel_instances[key] = value

    def __delitem__(self, key):
        del self.kernel_instances[key]

    def __iter__(self):
        return iter(self.kernel_instances)

    def __len__(self):
        return len(self.kernel_instances)

    def __contains__(self, key):
        return key in self.kernel_instances

    def items(self):
        return self.kernel_instances.items()

    def keys(self):
        return self.kernel_instances.keys()

    def values(self):
        return self.kernel_instances.values()

    def link_llvm_modules(self, prune=True, internalize=True):
        if self._link_mod is not None:
            return self._link_mod

        self._link_mod = jit.link_llvm_modules(self.llvm_files, self.kernel_name)

        # Comment these, we will call this in the jit.cpp link
        # if internalize:
        #     jit.internalize(self._link_mod, self.kernel_name)

        #  if prune:
        #      jit.pruneIR(self._link_mod)

        return self._link_mod

    @classmethod
    def from_json(cls, fn: str):
        if not Path(fn).exists():
            raise RuntimeError("JSON file does not exist")

        with open(fn, "r") as fd:
            record_db = json.load(fd)

        instances = {}
        for dhash, inst in record_db["instances"].items():
            block_dim = dim3(
                inst["BlockDims"]["x"], inst["BlockDims"]["y"], inst["BlockDims"]["z"]
            )
            grid_dim = dim3(
                inst["GridDims"]["x"], inst["GridDims"]["y"], inst["GridDims"]["z"]
            )
            instances[dhash] = cls.KernelInstance(
                record_db["StaticHash"],
                dhash,
                record_db["KernelName"],
                inst["Args"],
                inst["SharedMem"],
                block_dim,
                grid_dim,
                record_db["Specializations"],
                inst["Occurrences"],
                inst["Prologue"],
                inst["Epilogue"],
            )

        for llvm_fn in record_db["Modules"]:
            if not Path(llvm_fn).exists():
                raise RuntimeError(f"File {llvm_fn} does not exist")

        return cls(
            record_db["StaticHash"],
            record_db["KernelName"],
            record_db["DemangledName"],
            record_db["Modules"],
            record_db["ArgNames"],
            record_db["Specializations"],
            record_db["VAddr"],
            record_db["VASize"],
            instances,
        )
