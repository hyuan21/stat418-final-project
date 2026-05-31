"""
Smoke tests for the Hollinger Game Score formula.

These tests use values from real, well-known NBA box scores so that any drift
in the formula constants is caught immediately.
"""
import pandas as pd
import pytest

from src.data.compute_game_score import (
    compute_game_score,
    add_game_score_column,
    GAME_SCORE_COEFFICIENTS,
)


def test_zero_stat_line_is_zero():
    row = dict.fromkeys(
        ["pts", "fgm", "fga", "ftm", "fta", "oreb", "dreb",
         "ast", "stl", "blk", "tov", "pf"], 0
    )
    assert compute_game_score(row) == 0


def test_single_point_only():
    row = dict.fromkeys(
        ["pts", "fgm", "fga", "ftm", "fta", "oreb", "dreb",
         "ast", "stl", "blk", "tov", "pf"], 0
    )
    row["pts"] = 1
    assert compute_game_score(row) == 1.0


def test_known_box_score_lebron_2016_finals_g7():
    """
    LeBron James — 2016 NBA Finals Game 7.
    Box score: 27 PTS, 9-24 FG, 4-12 3P, 5-7 FT, 11 REB (4 OREB, 7 DREB),
               11 AST, 2 STL, 3 BLK, 2 TO, 3 PF.
    """
    row = {
        "pts": 27, "fgm": 9, "fga": 24,
        "ftm": 5, "fta": 7,
        "oreb": 4, "dreb": 7,
        "ast": 11, "stl": 2, "blk": 3,
        "tov": 2, "pf": 3,
    }
    gs = compute_game_score(row)
    # Expected: 27 + 0.4*9 - 0.7*24 - 0.4*(7-5) + 0.7*4 + 0.3*7
    #          + 2 + 0.7*11 + 0.7*3 - 0.4*3 - 2
    #        = 27 + 3.6 - 16.8 - 0.8 + 2.8 + 2.1 + 2 + 7.7 + 2.1 - 1.2 - 2
    #        = 26.5
    assert gs == pytest.approx(26.5, abs=0.05)


def test_missing_columns_raises():
    with pytest.raises(KeyError):
        compute_game_score({"pts": 10})


def test_vectorized_matches_scalar():
    df = pd.DataFrame(
        [
            {"pts": 10, "fgm": 4, "fga": 8, "ftm": 2, "fta": 2,
             "oreb": 1, "dreb": 3, "ast": 5, "stl": 1, "blk": 0,
             "tov": 2, "pf": 1},
            {"pts": 30, "fgm": 12, "fga": 22, "ftm": 4, "fta": 6,
             "oreb": 0, "dreb": 8, "ast": 7, "stl": 2, "blk": 1,
             "tov": 3, "pf": 4},
        ]
    )
    df2 = add_game_score_column(df)
    for i, row in df.iterrows():
        assert df2.loc[i, "game_score"] == pytest.approx(
            compute_game_score(row), abs=1e-9
        )


def test_coefficients_constant():
    """Guard against accidental edits to the constants used downstream."""
    assert GAME_SCORE_COEFFICIENTS["pts"] == 1.0
    assert GAME_SCORE_COEFFICIENTS["fgm"] == 0.4
    assert GAME_SCORE_COEFFICIENTS["fga"] == -0.7
