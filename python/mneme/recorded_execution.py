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


def find_non_jsonables(obj, where="$"):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return
    if isinstance(obj, Path):
        print("Non-JSON type:", where, "->", obj, type(obj))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_non_jsonables(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            find_non_jsonables(v, f"{where}[{i}]")
    else:
        # add other allowed conversions here if you plan to support them
        print("Non-JSON type:", where, "->", type(obj))

# `MemStateRef` is a Python wrapper around Mneme’s C-level memory snapshot
# representation. It loads, compares, and resets GPU memory state from a
# recorded prologue/epilogue file using the C FFI interface. Instances behave
# as context managers and expose:
#   • argument pointers stored in the snapshot
#   • the number of kernel arguments
#   • equality comparison between two memory states
#
# In short, `MemStateRef` gives high-level access to recorded GPU memory images
# used for verification during replay.
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

# `RecordedExecution` captures the full static description of a kernel as
# recorded by Mneme: argument names, specializations, LLVM IR modules, virtual
# address ranges, and all dynamic kernel instances observed at runtime.
#
# It behaves like a container mapping dynamic-hash → `KernelInstance`, and also
# provides:
#   • linking of LLVM modules into a single IR module suitable for replay
#   • serialization/deserialization to/from JSON
#
# In short, this class describes everything Mneme needs to reconstruct,
# specialize, and replay a previously recorded GPU kernel.
class RecordedExecution:
    # Represents a single dynamic instance of a recorded kernel launch: the grid and
    # block dimensions used, shared-memory size, argument values, available
    # specializations, and file paths to its prologue/epilogue memory snapshots.
    # Each instance provides:
    #   • equality/hash behavior based on dynamic+static hash
    #   • on-demand access to its memory snapshots via `MemStateRef`
    #   • conversion to a JSON-friendly dictionary
    #
    # This class is the per-launch unit of information used by Mneme during replay.
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
            specializations: List[bool],
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
            self.specializations = specializations
            for i, v in enumerate(specializations):
                if v:
                    self.available_specializations.append(i)
            self.occ = occ
            self.prologue = MemStateRef(prologue_fn, kernel_name, SnapshotType.PROLOGUE)
            self.epilogue = MemStateRef(epilogue_fn, kernel_name, SnapshotType.EPILOGUE)

        def __hash__(self):
            return hash(self.dynamic_hash + self.static_hash)

        def __str__(self):
            return f"Grid:{self.grid_dim}, BlockDim: {self.block_dim}, Shared Memory {self.shared_mem}"

        def to_dict(self):
            res = {}
            res["Args"] = self.specializations
            res["BlockDims"] = self.block_dim.to_dict()
            res["GridDims"] = self.grid_dim.to_dict()
            res["Occurrences"] = self.occ
            res["SharedMem"] = self.shared_mem
            res["Epilogue"] = self.epilogue.fn
            res["Prologue"] = self.prologue.fn

            return res

    def __init__(
        self,
        static_hash: str,
        kernel_name: str,
        demangled_name: str,
        llvm_files: List[str],
        arg_names: List[str],
        specializations: List[bool],
        va_addr: str,
        va_size: int,
        kernel_instances: Dict[str, KernelInstance],
    ):
        self.static_hash = static_hash
        self.kernel_name = kernel_name
        self.demangled_name = demangled_name
        self.llvm_files = llvm_files
        self.arg_names = arg_names
        self.specializations = specializations
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

        self._link_mod = jit.link_llvm_modules(
            self.llvm_files, self.kernel_name, prune, internalize
        )

        return self._link_mod

    def to_dict(self):
        res = {}
        res["ArgNames"] = self.arg_names
        res["BinaryBlobs"] = []
        res["DemangledName"] = self.demangled_name
        res["KernelName"] = self.kernel_name
        res["Modules"] = self.llvm_files
        res["Specializations"] = self.specializations
        res["StaticHash"] = self.static_hash
        res["VASize"] = self.va_size
        res["VAddr"] = self.va_addr
        res["instances"] = {}
        for k, v in self.items():
            res["instances"][k] = v.to_dict()
        return res

    def to_json(self, fn: str):
        find_non_jsonables(self.to_dict())
        with open(fn, "w") as fd:
            json.dump(self.to_dict(), fd, indent=2)

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
