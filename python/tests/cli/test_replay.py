from mneme.recorded_execution import RecordedExecution
from mneme.mneme_types import ExperimentConfiguration
from mneme.async_executor import AsyncReplayExecutor
from mneme.cli import main as mneme_main


def test_tune(recorded_execution, has_amd_gpu, has_nvidia_gpu):
    recorded_kernel = RecordedExecution.from_json(str(recorded_execution))
    dynamic_hash = list(recorded_kernel.kernel_instances.keys())[0]
    kernel = recorded_kernel[dynamic_hash]
    baseline_config = {
        "block": {
            "x": kernel.block_dim.x,
            "y": kernel.block_dim.y,
            "z": kernel.block_dim.z,
        },
        "grid": {
            "x": kernel.grid_dim.x,
            "y": kernel.grid_dim.y,
            "z": kernel.grid_dim.z,
        },
        "shared_mem": kernel.shared_mem,
        "specialize": True,
        "set_launch_bounds": True,
        "specialize_dims": True,
        "passes": "default<O3>",
        "codegen_method": "rtc",
    }

    if has_amd_gpu:
        baseline_config["codegen_method"] = "serial"

    executor = AsyncReplayExecutor(
        record_db=recorded_execution,
        record_id=dynamic_hash,
        iterations=5,
        results_db_dir="./",
        num_workers=1,
    )

    baseline_result = executor.evaluate(
        ExperimentConfiguration.from_dict(baseline_config)
    )
    executor.shutdown()
    assert baseline_result.verified, "Replay run was not verified"
    assert len(baseline_result.exec_time) == 7, "Did not execute 5 experiments"


def test_replay(recorded_execution, has_amd_gpu, has_nvidia_gpu):
    recorded_kernel = RecordedExecution.from_json(str(recorded_execution))
    dynamic_hash = list(recorded_kernel.kernel_instances.keys())[0]
    codegen = "rtc"
    if has_amd_gpu:
        codegen = "serial"

    result = mneme_main(
        [
            "replay",
            "-cm",
            codegen,
            "-rdb",
            str(recorded_execution),
            "-rid",
            str(dynamic_hash),
            "default<O0>",
        ]
    )

    assert result == 0
