# python/tests/test_cli_copy.py

import argparse
from pathlib import Path

import pytest
from mneme.commands import Copy, _copy_or_move  # adjust import if needed

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def copy_parser():
    parser = argparse.ArgumentParser(prog="mneme copy")
    Copy.set_cli_args(parser)
    return parser


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_copy_happy_path(tmp_path, copy_parser, monkeypatch):
    """
    Test that Copy.run calls _copy_or_move with move=False
    when given valid source DB(s) and valid destination dir.
    """

    src1 = tmp_path / "rec1.json"
    src2 = tmp_path / "rec2.json"
    src1.write_text("{}")
    src2.write_text("{}")

    dest = tmp_path / "dest"
    dest.mkdir()

    captured = {}

    def fake_copy_or_move(sources, d, move=False):
        captured["sources"] = sources
        captured["dest"] = d
        captured["move"] = move

    monkeypatch.setattr("mneme.commands._copy_or_move", fake_copy_or_move)

    args = copy_parser.parse_args([str(src1), str(src2), str(dest)])
    Copy.run(args, verbosity=None)

    assert captured["move"] is False
    assert captured["dest"] == dest
    assert captured["sources"] == [src1, src2]


def test_copy_requires_at_least_two_args(copy_parser):
    """
    Must have at least one src + one dest.
    """
    args = copy_parser.parse_args(["only_one_arg"])
    with pytest.raises(ValueError):
        Copy.run(args, verbosity=None)


def test_copy_fails_if_destination_missing(tmp_path, copy_parser):
    """
    Destination must exist.
    """
    src = tmp_path / "rec.json"
    src.write_text("{}")

    missing_dest = tmp_path / "nope"

    args = copy_parser.parse_args([str(src), str(missing_dest)])
    with pytest.raises(RuntimeError):
        Copy.run(args, verbosity=None)


def test_copy_fails_if_destination_not_directory(tmp_path, copy_parser):
    """
    Destination must be a directory.
    """
    src = tmp_path / "rec.json"
    src.write_text("{}")

    dest_file = tmp_path / "file_instead_of_dir"
    dest_file.write_text("X")

    args = copy_parser.parse_args([str(src), str(dest_file)])
    with pytest.raises(NotADirectoryError):
        Copy.run(args, verbosity=None)


def test_copy_multiple_sources(tmp_path, copy_parser, monkeypatch):
    """
    Copy should work with multiple sources.
    """

    src1 = tmp_path / "db1.json"
    src2 = tmp_path / "db2.json"
    src3 = tmp_path / "db3.json"
    for f in [src1, src2, src3]:
        f.write_text("{}")

    dest = tmp_path / "dest"
    dest.mkdir()

    captured = {}

    def fake_copy_or_move(sources, d, move=False):
        captured["sources"] = sources
        captured["dest"] = d
        captured["move"] = move

    monkeypatch.setattr("mneme.commands._copy_or_move", fake_copy_or_move)

    args = copy_parser.parse_args([str(src1), str(src2), str(src3), str(dest)])
    Copy.run(args, verbosity=None)

    assert captured["move"] is False
    assert captured["sources"] == [src1, src2, src3]
    assert captured["dest"] == dest
