import json

from mneme.mneme_types import ExperimentResult, dim3
from mneme.tuning import session as tune_session
from mneme.tuning.session import TuneOptions, TuningSession


class FakeKernel:
    kernel_name = "KERNEL"
    specializations = [True, False, True]
    shared_mem = 0
    block_dim = dim3(128, 1, 1)
    grid_dim = dim3(8, 1, 1)


class FakeRecorded:
    llvm_files = []
    demangled_name = "fake::kernel"

    def __getitem__(self, record_id):
        if record_id != "rid":
            raise KeyError(record_id)
        return FakeKernel()


class DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class FakeExecutor:
    def __init__(self, *args, **kwargs):
        self.submitted = []

    def evaluate(self, config):
        return ExperimentResult(verified=True, executed=True, exec_time=[100, 100, 100])

    def submit(self, config):
        self.submitted.append(config)
        if config.passes == "default<O2>":
            samples = [70, 70, 70]
        elif config.codegen_opt == 3:
            samples = [80, 80, 80]
        else:
            samples = [90, 90, 90]
        return DoneFuture(ExperimentResult(verified=True, executed=True, exec_time=samples))

    def shutdown(self):
        pass


def test_tuning_session_writes_core_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tune_session.RecordedExecution,
        "from_json",
        staticmethod(lambda _: FakeRecorded()),
    )
    monkeypatch.setattr(tune_session, "AsyncReplayExecutor", FakeExecutor)

    exported = {}

    def fake_export(filename, recorded, kernel, config, usage_filename=None):
        exported["filename"] = filename
        exported["config"] = config.to_dict()
        with open(filename, "w") as fd:
            json.dump({kernel.kernel_name: {"Pipeline": config.passes}}, fd)
        with open(usage_filename, "w") as fd:
            fd.write("usage")

    monkeypatch.setattr(tune_session, "export_proteus_tuned_kernel", fake_export)

    opts = TuneOptions(
        record_database="db.json",
        record_id="rid",
        preset="compiler",
        sampler="exhaustive",
        trials=4,
        iterations=3,
        results_dir=str(tmp_path),
        passes=["default<O3>", "default<O2>"],
        codegen_opt_range="2:3",
        quiet=True,
    )

    assert TuningSession(opts).run() == 0

    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["result"]["verified"] is True
    assert baseline["result"]["metric"] == 100

    trials = [
        json.loads(line)
        for line in (tmp_path / "trials.jsonl").read_text().splitlines()
    ]
    assert len(trials) == 4
    assert all(trial["status"] == "verified" for trial in trials)

    best = json.loads((tmp_path / "best.json").read_text())
    assert best["config"]["passes"] == "default<O2>"
    assert best["best_metric"] == 70
    assert (tmp_path / "best_replay.sh").exists()
    assert (tmp_path / "proteus_usage.txt").read_text() == "usage"
    assert exported["filename"] == str(tmp_path / "proteus_tuned_kernels.json")


def test_tune_cli_wires_options(monkeypatch):
    from mneme.tuning import cli as tune_cli

    captured = {}

    class FakeSession:
        def __init__(self, options):
            captured["options"] = options

        def run(self):
            return 0

    monkeypatch.setattr(tune_cli, "TuningSession", FakeSession)

    parser = tune_cli.argparse.ArgumentParser(prog="mneme tune")
    tune_cli.add_tune_args(parser)
    args = parser.parse_args(
        [
            "-rdb",
            "db.json",
            "-rid",
            "rid",
            "--preset",
            "compiler",
            "--trials",
            "3",
            "--no-proteus-output",
        ]
    )

    assert tune_cli.run_tune(args, None) == 0
    assert captured["options"].record_database == "db.json"
    assert captured["options"].record_id == "rid"
    assert captured["options"].preset == "compiler"
    assert captured["options"].trials == 3
    assert captured["options"].proteus_enabled is False
