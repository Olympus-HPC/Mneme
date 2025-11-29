# python/tests/test_cli_move.py

import argparse
from pathlib import Path

import pytest
from mneme.commands import Move, _copy_or_move  # adjust import if needed

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def move_parser():
    parser = argparse.ArgumentParser(prog="mneme move")
    Move.set_cli_args(parser)
    return parser


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_move_happy_path(tmp_path, move_parser, monkeypatch):
    """
    Test that Move.run calls _copy_or_move with move=True
    when given valid source DB(s) and valid destination dir.
    """

    # --- Setup fake source DB files ---
    src1 = tmp_path / "rec1.json"
    src2 = tmp_path / "rec2.json"
    src1.write_text("{}")
    src2.write_text("{}")

    # --- Setup destination dir ---
    dest = tmp_path / "dest"
    dest.mkdir()

    captured = {}

    def fake_copy_or_move(sources, d, move=False):
        captured["sources"] = sources
        captured["dest"] = d
        captured["move"] = move

    monkeypatch.setattr("mneme.commands._copy_or_move", fake_copy_or_move)

    args = move_parser.parse_args([str(src1), str(src2), str(dest)])
    Move.run(args, verbosity=None)

    # Validate
    assert captured["move"] is True
    assert captured["dest"] == dest
    assert captured["sources"] == [src1, src2]


def test_move_requires_at_least_two_args(move_parser):
    """
    Move requires at least one source + one destination.
    """
    # Only one argument → error
    args = move_parser.parse_args(["one.json"])
    with pytest.raises(ValueError):
        Move.run(args, verbosity=None)


def test_move_fails_if_destination_missing(tmp_path, move_parser):
    """
    Destination must exist.
    """

    src = tmp_path / "rec1.json"
    src.write_text("{}")
    missing_dest = tmp_path / "does_not_exist"

    args = move_parser.parse_args([str(src), str(missing_dest)])
    with pytest.raises(RuntimeError):
        Move.run(args, verbosity=None)


def test_move_fails_if_destination_not_directory(tmp_path, move_parser):
    """
    Destination must be a directory.
    """

    src = tmp_path / "rec1.json"
    src.write_text("{}")

    dest_file = tmp_path / "not_a_dir"
    dest_file.write_text("X")  # Regular file

    args = move_parser.parse_args([str(src), str(dest_file)])
    with pytest.raises(NotADirectoryError):
        Move.run(args, verbosity=None)


def test_move_multiple_sources(tmp_path, move_parser, monkeypatch):
    """
    Move should work with multiple sources and one dest.
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

    args = move_parser.parse_args([str(src1), str(src2), str(src3), str(dest)])

    Move.run(args, verbosity=None)

    assert captured["move"] is True
    assert captured["sources"] == [src1, src2, src3]
    assert captured["dest"] == dest
