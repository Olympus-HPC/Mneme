"""
Recorded execution database and memory snapshot bindings.

This module defines the Python-side representation of Mneme’s record/replay
artifacts:

  * **MemStateRef**: a lightweight wrapper over the native memory-snapshot object
    (prologue/epilogue) used for replay verification.
  * **RecordedExecution**: a JSON-serializable description of a recorded kernel,
    including all observed dynamic instances and the LLVM IR modules required
    for replay.

The native snapshot API is accessed via ctypes/FFI (``ffi.lib.MnemePy_*``).
Instances of :class:`MemStateRef` behave as context managers: they load the
snapshot on entry and dispose the native handle on exit.

Notes
-----
- This file is *core* to replay correctness: prologue/epilogue snapshots are used
  to verify that the replayed kernel produced the expected state.
- The JSON schema here is treated as a stable interchange format between record
  and replay tools.
"""

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
    """
    Enumeration of memory snapshot roles within a recorded execution.

    PROLOGUE
        Snapshot captured immediately before kernel execution.
    EPILOGUE
        Snapshot captured immediately after kernel execution.

    The snapshot type is used by the native snapshot loader to interpret the
    record format and to decide which parts of state are treated as inputs vs
    outputs during verification.
    """

    PROLOGUE = 1
    EPILOGUE = 2


def find_non_jsonables(obj, where="$"):
    """
    Debug helper: print paths to fields that are not JSON-serializable.

    This is used as a sanity check before writing the record database to JSON.
    It is intentionally permissive and prints to stdout rather than raising.

    Parameters
    ----------
    obj : Any
        Object graph to inspect.
    where : str
        JSONPath-like location used when printing offending fields.
    """
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


class MemStateRef:
    """
    Handle to a recorded memory snapshot (prologue/epilogue).

    A :class:`MemStateRef` wraps Mneme’s native memory snapshot representation,
    which encodes the recorded kernel argument pointers and the captured device
    memory state. During replay, these snapshots serve two purposes:

      1. **Inputs**: The prologue snapshot provides the argument pointer list and
         initial memory contents required to execute the kernel deterministically.
      2. **Verification**: The epilogue snapshot represents the expected post-kernel
         state. Replay compares a reproduced epilogue against this snapshot to
         validate correctness.

    Instances are context managers. Entering the context loads the snapshot
    into the native handle; leaving the context disposes it.

    Parameters
    ----------
    fn : str
        Path to the snapshot file on disk.
    kernel_name : str
        Kernel name associated with this snapshot (used by native layer).
    snap_type : SnapshotType
        Whether this snapshot is a prologue or epilogue capture.

    Raises
    ------
    RuntimeError
        If the snapshot file does not exist.
    """

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
        """
        Initialize and load the snapshot into the native handle.

        Returns
        -------
        MemStateRef
            Returns self for convenient chaining / context-manager usage.
        """
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
        """
        Return the kernel argument pointer array stored in the snapshot.

        Returns
        -------
        ctypes.POINTER(ctypes.c_void_p)
            Pointer to an array of argument pointers as returned by the native API.

        Raises
        ------
        RuntimeError
            If the snapshot has not been loaded via :meth:`open`.
        """
        if not self._load:
            raise RuntimeError("Cannot access arguments without loading memory state")

        if self._args == None:
            self._args = ffi.lib.MnemePy_getArgs(self._state)

        return self._args

    @property
    def num_args(self):
        """
        Return the number of kernel arguments recorded in the snapshot.

        Returns
        -------
        int
            Number of arguments.

        Raises
        ------
        RuntimeError
            If the snapshot has not been loaded via :meth:`open`.
        """
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
        """
        Reset the snapshot state in the native layer.

        This is typically used to restore the device memory state to the recorded
        baseline (e.g., before re-running a replay) without reinitializing the
        snapshot handle.

        Raises
        ------
        RuntimeError
            If the snapshot has not been loaded via :meth:`open`.
        """
        if self._load is False:
            raise RuntimeError("Cannot reset memory, if state is not read first.")

        ffi.lib.MnemePy_ResetMemState(self._state)

    def __eq__(self, other):
        """
        Compare two snapshots using the native comparison routine.

        Returns
        -------
        bool
            True if the native layer considers the states equivalent.
        """
        return bool(ffi.lib.MnemePy_CompareMemState(self._state, other._state))

    def __ne__(self, other):
        """
        Compare two snapshots using the native comparison routine.

        Returns
        -------
        bool
            True if the native layer considers the states different.
        """
        return not bool(ffi.lib.MnemePy_CompareMemState(self._state, other._state))

    def __del__(self):
        try:
            state = getattr(self, "_state", None)

            # If no state, or constructor failed, or tests used fake values → skip cleanup
            if not state or isinstance(state, str):
                return

            # Call disposer only if available and callable
            dispose = getattr(ffi.lib, "MnemePy_DisposeMemState", None)
            if callable(dispose):
                try:
                    dispose(state)
                except Exception:
                    pass

        except Exception:
            # Absolutely nothing should escape __del__
            pass


class RecordedExecution:
    """
    Description of a recorded kernel execution and its dynamic instances.

    A :class:`RecordedExecution` captures everything needed to replay and tune
    a kernel that was observed during application execution:

      - Kernel identity (static hash, name, demangled name)
      - Argument names and specialization availability
      - Virtual address space reservation information (VA base + size)
      - LLVM IR module file paths required for linking
      - A mapping of **dynamic hash → KernelInstance**, representing each observed
        launch instance (grid/block/shared-mem and snapshot paths)

    The class behaves like a mapping over kernel instances and supports JSON
    serialization via :meth:`to_json` / :meth:`from_json`.

    Parameters
    ----------
    static_hash : str
        Stable identifier for the kernel’s static code shape.
    kernel_name : str
        Mangled or runtime kernel symbol name.
    demangled_name : str
        Human-readable kernel name (if available).
    llvm_files : list[str]
        Paths to LLVM IR modules captured during recording.
    arg_names : list[str]
        Recorded kernel argument names (for display/debugging).
    specializations : list[bool]
        Per-argument specialization availability flags.
    va_addr : str
        Base virtual address (hex string) used by Mneme’s memory manager.
    va_size : int
        Virtual address space size in bytes (or recording-specific unit).
    kernel_instances : dict[str, KernelInstance]
        Mapping from dynamic hash to recorded launch instance descriptor.
    """

    class KernelInstance:
        """
        Description of one dynamic kernel launch instance.

        A kernel may be launched multiple times with different dynamic properties
        (e.g., different grid/block sizes, argument values, or observed runtime hashes).
        Each :class:`KernelInstance` stores:

          - Launch parameters (grid, block, shared memory)
          - Dynamic hash (identifies the runtime instance)
          - Available specialization indices (derived from specialization flags)
          - Snapshot file paths for prologue and epilogue

        The prologue/epilogue snapshots are exposed via :class:`MemStateRef` objects,
        which are opened on demand by the replay executor.
        """

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
            """
            Convert this instance into a JSON-friendly dictionary.

            Returns
            -------
            dict
                Serializable representation containing dims, shared memory,
                occurrence count, and snapshot file paths.
            """
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
        globals: List[str] = None,
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
        self.globals = globals or []
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
        """
        Link recorded LLVM IR modules into a single module suitable for replay.

        This is a convenience wrapper over the Proteus JIT linking layer.
        Results are cached on the first call and returned on subsequent calls.

        Parameters
        ----------
        prune : bool
            Whether to prune unused symbols/IR during linking.
        internalize : bool
            Whether to internalize symbols during linking.

        Returns
        -------
        ModuleRef
            Linked IR module produced by the JIT layer.
        """
        if self._link_mod is not None:
            return self._link_mod

        self._link_mod = jit.link_llvm_modules(
            self.llvm_files,
            self.kernel_name,
            prune,
            internalize,
            preserve_globals=self.globals,
        )

        return self._link_mod

    def to_dict(self):
        res = {}
        res["ArgNames"] = self.arg_names
        res["BinaryBlobs"] = []
        res["DemangledName"] = self.demangled_name
        res["KernelName"] = self.kernel_name
        res["Modules"] = self.llvm_files
        res["Globals"] = self.globals
        res["Specializations"] = self.specializations
        res["StaticHash"] = self.static_hash
        res["VASize"] = self.va_size
        res["VAddr"] = self.va_addr
        res["instances"] = {}
        for k, v in self.items():
            res["instances"][k] = v.to_dict()
        return res

    def to_json(self, fn: str):
        """
        Serialize this record database to a JSON file.

        Parameters
        ----------
        fn : str
            Output JSON path.
        """
        find_non_jsonables(self.to_dict())
        with open(fn, "w") as fd:
            json.dump(self.to_dict(), fd, indent=2)

    @classmethod
    def from_json(cls, fn: str):
        """
        Load a :class:`RecordedExecution` database from JSON.

        This reconstructs all :class:`KernelInstance` entries and validates that
        referenced LLVM module paths exist.

        Parameters
        ----------
        fn : str
            Path to the recorded execution JSON file.

        Returns
        -------
        RecordedExecution
            Loaded record database.

        Raises
        ------
        RuntimeError
            If the JSON file does not exist or referenced IR modules are missing.
        """
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
            record_db.get("Globals", []),
        )
