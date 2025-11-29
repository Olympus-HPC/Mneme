from multiprocessing import Queue
from types import SimpleNamespace

import mneme.db as db_mod
import mneme.device as dev_mod
import mneme.recorded_execution as rec_mod
import mneme.replay_executor as exec_mod
import mneme.tuner as tuner_mod
import pytest


# -------------------------------------------------------------
# Fake Worker (replaces TuneWorkerHandle)
# -------------------------------------------------------------
class FakeWorker:
    def __init__(self, idx, request_q, completed_q, *args, **kwargs):
        self.idx = idx
        self.request_q = request_q
        self.completed_q = completed_q
        self.assigned = []

    def assign(self, msg):
        self.assigned.append(msg)
        exp_id = msg["exp_id"]
        data = msg["data"]

        # Immediately emit a fake "result"
        self.completed_q.put(
            {
                "payload": "result",
                "exp_id": exp_id,
                "data": data,
                "llvm_ir": "fake.bc",
            }
        )

    def shutdown_process(self):
        pass

    def join_monitor(self):
        pass


# -------------------------------------------------------------
# Fake DB (MnemeDB)
# -------------------------------------------------------------
class FakeDB:
    _columns = []  # required class attribute
    _filename = "/tmp/fake"  # required class attribute
    _open = True  # MnemeDB.add() checks this, but we override add anyway

    def __init__(self, *a, **k):
        self.saved = []

    def open(self):
        return self

    def add(self, orig_ir, llvm_ir, exp):
        # Override CSV writing completely
        self.saved.append((orig_ir, llvm_ir, exp))

    def save_ir(self, ir, name):
        return f"{name}.bc"

    def should_execute(self, exp):
        return True

    def __len__(self):
        return 0


# -------------------------------------------------------------
# Fake RecordedExecution + Kernel Descriptor
# -------------------------------------------------------------
class FakeKernelDescr:
    kernel_name = "vec_add"
    static_hash = "AAA"
    dynamic_hash = "BBB"
    demangled_name = "vec_add"

    grid_dim = dev_mod.dim3(1, 1, 1)
    block_dim = dev_mod.dim3(1, 1, 1)
    shared_mem = 0
    available_specializations = []
    specializations = []
    va_addr = "0x0"
    va_size = 1024

    prologue = SimpleNamespace(
        open=lambda: SimpleNamespace(_state=True),
        args=[],
        num_args=0,
    )
    epilogue = SimpleNamespace(open=lambda: SimpleNamespace(_state=True))


class FakeRecordedExec(dict):
    """Minimal fake RecordedExecution with one kernel entry."""

    def __init__(self):
        super().__init__()
        self["kid"] = FakeKernelDescr()

    @staticmethod
    def from_json(path):
        return FakeRecordedExec()

    def link_llvm_modules(self, prune=True, internalize=False):
        return FakeModule()


# -------------------------------------------------------------
# Fake IR module
# -------------------------------------------------------------
class FakeModule:
    def clone(self):
        return FakeModule()

    def to_bitcode(self, fn: str):
        return 0


# -------------------------------------------------------------
# The Test
# -------------------------------------------------------------
def test_replay_tuner_random(monkeypatch):

    # ---- Patch RecordedExecution ----
    monkeypatch.setattr(
        rec_mod.RecordedExecution, "from_json", staticmethod(FakeRecordedExec.from_json)
    )
    monkeypatch.setattr(
        rec_mod.RecordedExecution,
        "link_llvm_modules",
        FakeRecordedExec().link_llvm_modules,
    )

    # ---- GPU/device patches ----
    monkeypatch.setattr(dev_mod, "get_device_count", lambda: 1)
    monkeypatch.setattr(dev_mod, "get_device_arch", lambda: "sm_80")
    monkeypatch.setattr(dev_mod, "set_device", lambda dev: None)
    monkeypatch.setattr(dev_mod, "get_max_blocks_per_sm", lambda: 0)

    # ---- Patch Worker Handle ----
    monkeypatch.setattr(
        tuner_mod,
        "TuneWorkerHandle",
        lambda *args, **kwargs: FakeWorker(args[0], args[1], args[2]),
    )

    # ---- Patch MnemeDB ----
    monkeypatch.setattr(db_mod, "MnemeDB", FakeDB)
    monkeypatch.setattr(tuner_mod, "MnemeDB", FakeDB)
    monkeypatch.setattr(exec_mod, "MnemeDB", FakeDB)

    # ---- Patch BaseExecutor.link_ir ----
    monkeypatch.setattr(exec_mod.BaseExecutor, "link_ir", lambda self: FakeModule())

    # ---- Patch Pipeline generation ----
    monkeypatch.setattr(
        tuner_mod.PipelineManager,
        "generate",
        lambda self, num, length, x, y, seed: [["p1", "p2"]],
    )
    monkeypatch.setattr(
        tuner_mod.PipelineManager, "to_string", lambda self, p: "pipeline"
    )

    # ---- Patch profiling lib getter ----
    monkeypatch.setattr(tuner_mod, "get_profile_library", lambda: "/tmp/fakeprof.so")

    # ---------------------------------------------------------
    # Construct CLI args
    # ---------------------------------------------------------
    args = SimpleNamespace(
        db="dummy.json",
        record_id="kid",
        results_db_dir="/tmp",
        prune=True,
        internalize=False,
        codegen_opt=3,
        codegen_method="serial",
        iterations=2,
        device_id=0,
        specialize=False,
        seed=0,
        average_pipeline_length=10,
        num_trials=3,
        num_workers=1,
        tuner="random",
        sampler=None,
        command=None,
        func=None,
    )

    # ---------------------------------------------------------
    # Call ReplayTuner
    # ---------------------------------------------------------
    result = tuner_mod.ReplayTuner.run(args, False)

    # ---------------------------------------------------------
    # Assert successful run
    # ---------------------------------------------------------
    assert result == 0
