import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mneme.device import dim3
from mneme.pipeline import PipelineManager
from mneme.recorded_execution import RecordedExecution
from mneme.replay_executor import CLIExecutor


# -------------------------------------------------------------------
# Complete Fake PageManagerRef — prevents CFFI aborts
# -------------------------------------------------------------------
class FakePageManager:
    def __init__(self, *a, **k):
        self.closed = False

    def close(self):
        self.closed = True


# -------------------------------------------------------------------
# Fake device function + Fake DeviceModule that returns it
# -------------------------------------------------------------------
class FakeDeviceFunc:
    def profile(self, *a, **k):
        return 123

    @property
    def reg_usage(self):
        return 10

    @property
    def const_mem(self):
        return 0

    @property
    def local_mem(self):
        return 0


class FakeDeviceModule:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get_function(self, name):
        return FakeDeviceFunc()


# -------------------------------------------------------------------
# Fake MemBufferRef used by jit.codegen_object
# -------------------------------------------------------------------
class FakeMemBuffer:
    def get_size(self):
        return 128


# -------------------------------------------------------------------
# FULL HAPPY-PATH TEST FOR EXECUTE
# -------------------------------------------------------------------
def test_execute_happy_path(monkeypatch, tmp_path):
    class FakeProfileLib:
        def MnemePy_startProfile(self, device_id):
            # return a fake correlation ID
            return 1

        def MnemePy_stopProfile(self, correlation_id, arr, num_records):
            # return a fake correlation ID
            return [10, 1, 3, 4.5, 34]

        def MnemePy_getNumRecords(self, correlation_id):
            # no profiling records
            return 0

    class FakeProfileLib:
        def MnemePy_startProfile(self, device_id):
            return 1  # correlation id

        def MnemePy_stopProfile(self, correlation_id, arr, n):
            return [i for i in range(n)]

        def MnemePy_getNumRecords(self, correlation_id):
            return 3  # MUST BE >=3

    # Patch the CFFI library
    monkeypatch.setattr("mneme.profile.profile_lib", FakeProfileLib())

    # Patch the Python-level wrappers
    monkeypatch.setattr("mneme.profile.gpu_profile_start", lambda: 1)
    monkeypatch.setattr("mneme.profile.gpu_profile_stop", lambda cid: [10, 11, 12])

    # Patch the utils-level decorator (it calls gpu_profile_start/stop *indirectly*)
    import mneme.utils

    def fake_gpu_profile(func):
        def wrapper(*args, **kwargs):
            cid = 1
            result = func(*args, **kwargs)
            # Same measurement payload used everywhere
            setattr(args[0], "exec_time", [10, 11, 12])
            return result

        return wrapper

    monkeypatch.setattr("mneme.utils.cond_gpu_time", fake_gpu_profile)

    class FakeState:
        def __init__(self):
            self._state = True

        def close(self):
            pass  # no-op

    class FakeKernelDescr:
        kernel_name = "vec_add"
        static_hash = "ABC"
        dynamic_hash = "XYZ"
        demangled_name = "vec_add"

        grid_dim = dim3(1, 1, 1)
        block_dim = dim3(1, 1, 1)
        shared_mem = 0

        prologue = SimpleNamespace(
            open=lambda: FakeState(),
            args=[],
            num_args=0,
        )
        epilogue = SimpleNamespace(open=lambda: FakeState())

        available_specializations = []
        specializations = []
        va_addr = 0x0
        va_size = 1024

    # Fake RecordedExecution
    class FakeRecordedExec(dict):
        va_addr = "0x0"
        va_size = 1024
        llvm_files = ["dummy.bc"]
        kernel_name = "vec_add"
        demangled_name = "vec_add"
        specializations = []

        def __init__(self):
            super().__init__()
            self["kid"] = FakeKernelDescr()

        @staticmethod
        def from_json(path):
            return FakeRecordedExec()

        def link_llvm_modules(self, prune=True, internalize=False):
            return FakeModule()

    # ------------------------------------------------------------------
    # Fake IR module
    # ------------------------------------------------------------------
    class FakeModule:
        def clone(self):
            return FakeModule()

        def __repr__(self):
            return "<FakeModule>"

    # ------------------------------------------------------------------
    # Monkeypatch EVERYTHING that touches GPU/LLVM
    # ------------------------------------------------------------------
    monkeypatch.setattr(
        "mneme.recorded_execution.RecordedExecution.from_json",
        staticmethod(FakeRecordedExec.from_json),
    )

    # LLVM bitcode linking
    monkeypatch.setattr(
        "mneme.recorded_execution.RecordedExecution.link_llvm_modules",
        FakeRecordedExec().link_llvm_modules,
    )

    # Prevent PageManagerRef CFFI calls
    monkeypatch.setattr("mneme.replay_executor.PageManagerRef", FakePageManager)

    # Prevent device driver calls
    monkeypatch.setattr(
        "mneme.device.DeviceModule.from_MemBuffer",
        staticmethod(lambda mb: FakeDeviceModule()),
    )

    # prevent GPU arch queries
    monkeypatch.setattr("mneme.device.get_device_arch", lambda: "sm_80")
    monkeypatch.setattr("mneme.device.get_device_count", lambda: 1)
    monkeypatch.setattr("mneme.device.set_device", lambda dev: None)

    # Fake jit operations (IR manipulation)
    monkeypatch.setattr("mneme.proteus.jit.optimize", lambda *a, **k: None)
    monkeypatch.setattr("mneme.proteus.jit.specialize_args", lambda *a, **k: "HASH")
    monkeypatch.setattr("mneme.proteus.jit.specialize_dims", lambda *a, **k: "HASH")
    monkeypatch.setattr("mneme.proteus.jit.set_launch_bounds", lambda *a, **k: "HASH")

    # Fake codegen
    monkeypatch.setattr(
        "mneme.proteus.jit.codegen_object", lambda *a, **k: FakeMemBuffer()
    )

    # Fake transform pass
    monkeypatch.setattr(
        "mneme.transforms.transform.remove_auto_initialize", lambda ir: ir
    )

    # ------------------------------------------------------------------
    # Fake MnemeDB so execution does not try to touch any files
    # ------------------------------------------------------------------
    class FakeIRDB:
        def __init__(self, *a, **k):
            pass

        def open(self):
            return self

        def save_ir(self, ir, h):
            return "ir.bc"

        def add(self, *a, **k):
            pass

        def should_execute(self, exp):
            return True

    monkeypatch.setattr("mneme.replay_executor.MnemeDB", FakeIRDB)

    # ------------------------------------------------------------------
    # Construct CLI arguments namespace
    # ------------------------------------------------------------------
    args = SimpleNamespace(
        db="dummy.json",
        record_id="kid",
        pipeline="default<O3>",
        prune=True,
        internalize=False,
        codegen_opt=3,
        codegen_method="serial",
        iterations=3,
        device_id=0,
        increamental=False,
        specialize=False,
        dims=False,
        max_threads=False,
        min_blocks_per_sm=0,
        results_db_dir=None,
        command=None,
        func=None,
    )

    # ------------------------------------------------------------------
    # ACTUALLY RUN THE EXECUTOR
    # This should now run entirely in mocked mode with no FFI.
    # ------------------------------------------------------------------
    CLIExecutor.run(args, True)
