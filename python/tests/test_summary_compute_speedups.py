# python/tests/test_summary_compute_speedups.py

import numpy as np
import pandas as pd
import pytest
from mneme.commands import Summary

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_df(rows):
    """Utility to build a DataFrame with the required columns."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_summary_missing_columns_raises():
    df = pd.DataFrame(
        {
            "specialize": [False],
            "max_threads": [0],
            # missing required columns
        }
    )
    with pytest.raises(ValueError):
        Summary.compute_speedups(df)


def test_summary_baseline_not_found():
    df = make_df(
        [
            {
                "specialize": True,  # baseline requires False
                "max_threads": 0,
                "min_blocks_per_sm": 0,
                "exec_time_median": 10.0,
                "hash": "A",
                "passes": "default<O3>,globaldce",
            }
        ]
    )

    ok, result = Summary.compute_speedups(df)
    assert not ok
    assert result is None


def test_summary_happy_path_simple():
    """
    Minimal test verifying:
    - baseline selected correctly
    - speedup math correct
    - all expected keys exist
    """

    df = make_df(
        [
            # baseline row
            {
                "specialize": False,
                "max_threads": 0,
                "min_blocks_per_sm": 0,
                "exec_time_median": 100.0,
                "hash": "base",
                "passes": "default<O3>,globaldce",
            },
            # tuning candidate (LB mask)
            {
                "specialize": False,
                "max_threads": 64,
                "min_blocks_per_sm": 0,
                "exec_time_median": 50.0,
                "hash": "lb",
                "passes": "default<O3>,globaldce",
            },
            # tuned+spec candidate
            {
                "specialize": True,
                "max_threads": 64,
                "min_blocks_per_sm": 1,
                "exec_time_median": 25.0,
                "hash": "spec_tune",
                "passes": "default<O3>,globaldce",
            },
        ]
    )

    ok, result = Summary.compute_speedups(df)
    assert ok is True

    # Baseline
    assert result["Baseline"][0] == 100.0
    assert result["Baseline"][1] == "base"

    # LB = baseline_time / candidate_time
    assert result["LB"][0] == 100.0 / 50.0
    assert result["LB"][1] == "lb"

    # Spec+TuneLB
    assert result["Spec+TuneLB"][0] == 100.0 / 25.0
    assert result["Spec+TuneLB"][1] == "spec_tune"

    # Ensure all expected major categories exist
    expected = {
        "Baseline",
        "TunePipeline",
        "LB",
        "LB+TunePipeline",
        "TuneLB",
        "TuneLB+TunePipeline",
        "Spec",
        "Spec+TunePipeline",
        "Spec+LB",
        "Spec+LB+TunePipeline",
        "Spec+TuneLB",
        "Spec+TuneLB+TunePipeline",
    }
    assert expected.issubset(result.keys())


def test_summary_speedup_nan_if_no_candidate():
    """
    If no row matches a mask, the corresponding speedup should be NaN.
    """

    df = make_df(
        [
            # baseline row
            {
                "specialize": False,
                "max_threads": 0,
                "min_blocks_per_sm": 0,
                "exec_time_median": 80.0,
                "hash": "base",
                "passes": "default<O3>,globaldce",
            }
            # No LB, no Spec, no Tune rows
        ]
    )

    ok, result = Summary.compute_speedups(df)
    assert ok

    # LB doesn't exist → NaN
    assert np.isnan(result["LB"][0])
    assert result["LB"][1] is None

    # Spec also missing → NaN
    assert np.isnan(result["Spec"][0])
    assert result["Spec"][1] is None
