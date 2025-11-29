import time
from unittest.mock import patch, MagicMock

from mneme.utils import cond_time, cond_gpu_time


# ---------------------------------------------------------------------------
# Helper objects
# ---------------------------------------------------------------------------

class DummyExp:
    pass

class DummySelf:
    pass


# ---------------------------------------------------------------------------
# cond_time tests
# ---------------------------------------------------------------------------

def test_cond_time_no_profile_calls_original_fn():
    """If profile=False, the decorated function behaves normally."""
    @cond_time("duration")
    def fn(self, exp):
        return 42

    exp = DummyExp()
    result = fn(DummySelf(), exp, profile=False)

    assert result == 42
    assert not hasattr(exp, "duration")


def test_cond_time_profile_sets_attribute():
    """If profile=True, execution time is measured and stored."""
    @cond_time("duration")
    def fn(self, exp):
        return "OK"

    exp = DummyExp()

    # Patch time.perf_counter for deterministic testing
    with patch("mneme.utils.time.perf_counter", side_effect=[1.0, 3.0]):
        result = fn(DummySelf(), exp, profile=True)

    assert result == "OK"
    assert exp.duration == 2.0   # end-start


# ---------------------------------------------------------------------------
# cond_gpu_time tests
# ---------------------------------------------------------------------------

def test_cond_gpu_time_no_profile_calls_original():
    """GPU timing is skipped when profile=False."""
    @cond_gpu_time("gpu_time")
    def fn(self, exp, kernel_name):
        return 7

    exp = DummyExp()

    with patch("mneme.utils.gpu_profile_start") as start, \
         patch("mneme.utils.gpu_profile_stop") as stop:

        out = fn(DummySelf(), exp, "kernel", profile=False)

    assert out == 7
    start.assert_not_called()
    stop.assert_not_called()
    assert not hasattr(exp, "gpu_time")


def test_cond_gpu_time_profile_records_gpu_measurements():
    """GPU start/stop functions are invoked and result stored on exp."""
    @cond_gpu_time("gpu_time")
    def fn(self, exp, kernel_name):
        return "DONE"

    exp = DummyExp()

    with patch("mneme.utils.gpu_profile_start", return_value="CID") as start, \
         patch("mneme.utils.gpu_profile_stop", return_value={"cycles": 123}) as stop:

        out = fn(DummySelf(), exp, "mykernel", profile=True)

    assert out == "DONE"
    start.assert_called_once_with("mykernel.kd")
    stop.assert_called_once_with("CID")
    assert exp.gpu_time == {"cycles": 123}

