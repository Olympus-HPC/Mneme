# python/tests/test_record_cli.py

import argparse
import os
from pathlib import Path

import mneme.commands as record_mod
import pytest
from mneme.commands import Record
from mneme.commands import utils as utils_mod

# -------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------


@pytest.fixture
def record_parser():
    """Create a parser with the Record CLI wired in."""
    parser = argparse.ArgumentParser(prog="mneme record")
    Record.set_cli_args(parser)
    return parser


# -------------------------------------------------------------
# Tests
# -------------------------------------------------------------


def test_record_happy_path(tmp_path, record_parser, monkeypatch):
    """
    Test the happy path: record with valid args, capturing env and cmd
    without running HIP code.
    """

    record_dir = tmp_path / "records"
    record_dir.mkdir()

    fake_binary = "/usr/bin/true"
    fake_lib = "/tmp/libmneme_record.so"

    # Monkeypatch env-wiring helpers
    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: fake_lib)

    captured = {}

    class FakeCode:
        returncode = 0

    # Monkeypatch subprocess.run so we don't run anything real
    def fake_run(cmd, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env or {}
        return FakeCode

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--virtual-address-space-size",
            "8",
            "--per-kernel-max-recordings",
            "3",
            "--",
            fake_binary,
            "42",
        ]
    )

    Record.run(args, verbosity="DEBUG")

    # Validate command passthrough
    assert captured["cmd"] == [fake_binary, "42"]

    # Validate environment
    env = captured["env"]
    assert env["LD_PRELOAD"] == fake_lib
    assert env["MNEME_PAGE_SIZE"] == "8"
    assert env["MNEME_MAX_RECORDINGS"] == "3"
    assert env["MNEME_DATA_DIR"] == str(record_dir)
    assert env["MNEME_LOG_LEVEL"] == "DEBUG"


def test_record_requires_dash_dash(record_parser, monkeypatch):
    """Record requires '--' before the command."""
    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    args = record_parser.parse_args(
        ["--record-db-dir", "/tmp", "vec_add", "123"]  # <-- missing "--"
    )

    with pytest.raises(SystemExit):
        Record.run(args, verbosity=None)


def test_record_fails_if_dir_missing(record_parser, tmp_path, monkeypatch):
    """record-db-dir must exist."""
    missing = tmp_path / "nope"

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    args = record_parser.parse_args(
        ["--record-db-dir", str(missing), "--", "/usr/bin/true"]
    )

    with pytest.raises(FileNotFoundError):
        Record.run(args, verbosity=None)


def test_record_fails_if_not_a_directory(record_parser, tmp_path, monkeypatch):
    """record-db-dir must be a directory (not a regular file)."""
    file_path = tmp_path / "file"
    file_path.write_text("hello")

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    args = record_parser.parse_args(
        ["--record-db-dir", str(file_path), "--", "/usr/bin/true"]
    )

    with pytest.raises(NotADirectoryError):
        Record.run(args, verbosity=None)


def test_record_handles_missing_executable(record_parser, tmp_path, monkeypatch):
    """Missing executable → parser.error → SystemExit."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    def fake_run(cmd, env=None, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--",
            "/this/does/not/exist",
        ]
    )

    with pytest.raises(SystemExit):
        Record.run(args, verbosity=None)


def test_record_handles_not_executable(record_parser, tmp_path, monkeypatch):
    """Non-executable binary → parser.error → SystemExit."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    fake_path = tmp_path / "fake"
    fake_path.write_text("hi")  # not executable

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    def fake_run(cmd, env=None, **kwargs):
        raise PermissionError

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--",
            str(fake_path),
        ]
    )

    with pytest.raises(SystemExit):
        Record.run(args, verbosity=None)


def test_record_default_page_size(record_parser, tmp_path, monkeypatch):
    """MNEME_PAGE_SIZE uses default value if not provided explicitly."""
    record_dir = tmp_path / "records"
    record_dir.mkdir()

    monkeypatch.setattr(utils_mod, "get_mneme_record_library_name", lambda: "/tmp/x.so")

    captured = {}

    class FakeCode:
        returncode = 0

    def fake_run(cmd, env=None, **kwargs):
        captured["env"] = env
        return FakeCode

    monkeypatch.setattr(record_mod.subprocess, "run", fake_run)

    args = record_parser.parse_args(
        [
            "--record-db-dir",
            str(record_dir),
            "--",
            "/usr/bin/true",
        ]
    )

    Record.run(args, verbosity=None)

    assert captured["env"]["MNEME_PAGE_SIZE"] == "4"  # default from code
