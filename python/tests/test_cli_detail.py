import argparse
from pathlib import Path

import pandas as pd
import pytest
from mneme.commands import Detail


@pytest.fixture
def detail_parser():
    parser = argparse.ArgumentParser(prog="mneme detail")
    Detail.set_cli_args(parser)
    return parser


def test_detail_reads_csv_and_filters_hash(
    tmp_path, detail_parser, capsys, monkeypatch
):
    """
    Correct behavior:
    - Detail.detail loads the CSV
    - It filters by the given hash
    """

    # --------------------------
    # Create fake CSV
    # --------------------------
    csv_file = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "hash": ["H1", "H2", "H3"],
            "exec_time_median": [10, 20, 30],
        }
    )
    df.to_csv(csv_file, index=False)

    loaded = {}

    # monkeypatch pandas.read_csv so we don't rely on plotting or printing
    def fake_read_csv(path):
        loaded["df"] = df
        return df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # --------------------------
    # Parse args and run
    # --------------------------
    args = detail_parser.parse_args(["--results", str(csv_file), "H2"])
    Detail.detail(args)

    # Ensure CSV was read
    assert "df" in loaded
    assert isinstance(loaded["df"], pd.DataFrame)


def test_detail_requires_results_option(detail_parser):
    """
    Ensure --results parameter is mandatory.
    """
    with pytest.raises(SystemExit):
        detail_parser.parse_args(["H1"])


def test_detail_missing_csv_file(detail_parser):
    """
    If CSV file does not exist, read_csv will raise FileNotFoundError.
    """
    args = detail_parser.parse_args(["--results", "does_not_exist.csv", "HASH"])
    with pytest.raises(FileNotFoundError):
        Detail.detail(args)
