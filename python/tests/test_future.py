import threading
import time

import pytest
from mneme.futures import EvalFuture


def test_future_set_and_get_result():
    f = EvalFuture(1, {"a": 10})
    assert not f.done()

    f.set_result({"ok": True})
    assert f.done()
    assert f.result() == {"ok": True}


def test_future_set_error():
    f = EvalFuture(2, {"a": 20})
    f.set_error("boom")

    assert f.done()

    with pytest.raises(RuntimeError) as exc:
        f.result()

    assert "boom" in str(exc.value)


def test_future_waits_until_result():
    f = EvalFuture(3, {})

    def producer():
        time.sleep(0.1)
        f.set_result({"value": 42})

    t = threading.Thread(target=producer)
    t.start()

    assert f.result() == {"value": 42}
    assert f.done()
    t.join()
