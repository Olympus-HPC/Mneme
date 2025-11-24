import pytest

@pytest.fixture(autouse=True)
def disable_all_dispose(monkeypatch):
    monkeypatch.setattr("mneme.llvm.buffer.MemBufferRef._dispose", lambda self: None)

