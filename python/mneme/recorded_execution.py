from pathlib import Path
import json
from enum import Enum
from typing import Dict, List
from .device import dim3
from .llvm import module
from .proteus import jit
from ctypes import c_int, c_bool, c_void_p, c_uint64


class SnapshotType(Enum):
    PROLOGUE = 1
    EPILOGUE = 2


class SnapshotFile:
    def __init__(self, fn: str, snap_type: SnapshotType):
        if not Path(fn).exists():
            raise RuntimeError(f"Expected prologue file: {fn} to exist")
        self.fn = fn
        self.s_type = snap_type
        self._loaded = False
        self._map = False


class RecordedExecution:
    class KernelInstance:
        def __init__(
            self,
            dhash: str,
            args: List,
            shared_mem: int,
            block_dim: dim3,
            grid_dim: dim3,
            occ: int,
            prologue_fn: str,
            epilogue_fn: str,
        ):
            self.dhash = dhash
            self.args = args
            self.shared_mem = shared_mem
            self.block_dim = block_dim
            self.grid_dim = grid_dim
            self.occ = occ
            self.prologue = SnapshotFile(prologue_fn, SnapshotType.PROLOGUE)
            self.epilogue = SnapshotFile(epilogue_fn, SnapshotType.EPILOGUE)

        def __hash__(self):
            return self.dhash

        def __str__(self):
            return f"Grid:{self.grid_dim}, BlockDim: {self.block_dim}, Shared Memory {self.shared_mem}"

    def __init__(
        self,
        kernel_name: str,
        demangled_name: str,
        llvm_files: List[str],
        arg_names: List[str],
        available_specializations: List[bool],
        va_addr: str,
        va_size: int,
        kernel_instances: Dict[str, KernelInstance],
    ):
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

    def link_llvm_modules(self):
        if self._link_mod is not None:
            return self._link_mod

        self._modules = []
        for ll in self.llvm_files:
            with open(ll, "rb") as fd:
                bitcode = fd.read()
            self._modules.append(module.parse_bitcode(bitcode))
        self._link_mod = jit.link_llvm_modules(self._modules)
        print(self._link_mod._ptr)
        return self._link_mod

    @classmethod
    def from_json(cls, fn: str):
        if not Path(fn).exists():
            raise RuntimeError("JSON file does not exist")

        with open(fn, "r") as fd:
            record_db = json.load(fd)

        instances = {}
        for dhash, inst in record_db["instances"].items():
            print(dhash)
            block_dim = dim3(
                inst["BlockDims"]["x"], inst["BlockDims"]["y"], inst["BlockDims"]["z"]
            )
            grid_dim = dim3(
                inst["GridDims"]["x"], inst["GridDims"]["y"], inst["GridDims"]["z"]
            )
            instances[dhash] = cls.KernelInstance(
                dhash,
                inst["Args"],
                inst["SharedMem"],
                block_dim,
                grid_dim,
                inst["Occurrences"],
                inst["Prologue"],
                inst["Epilogue"],
            )

        for llvm_fn in record_db["Modules"]:
            if not Path(llvm_fn).exists():
                raise RuntimeError(f"File {llvm_fn} does not exist")

        return cls(
            record_db["KernelName"],
            record_db["DemangledName"],
            record_db["Modules"],
            record_db["ArgNames"],
            record_db["Specializations"],
            record_db["VAddr"],
            record_db["VASize"],
            instances,
        )
