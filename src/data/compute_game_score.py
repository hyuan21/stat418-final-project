"""
Hollinger Game Score computation.

Formula:
    GS = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA - FTM)
         + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
"""
from __future__ import annotations

import pandas as pd


# Coefficients of the Hollinger Game Score formula.
# Exposed as a constant so other modules (and tests) can reference them.
GAME_SCORE_COEFFICIENTS = {
    "pts": 1.0,
    "fgm": 0.4,
    "fga": -0.7,
    "fta_minus_ftm": -0.4,
    "oreb": 0.7,
    "dreb": 0.3,
    "stl": 1.0,
    "ast": 0.7,
    "blk": 0.7,
    "pf": -0.4,
    "tov": -1.0,
}


REQUIRED_BOX_SCORE_COLS = [
    "pts", "fgm", "fga", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
]


def compute_game_score(row: pd.Series | dict) -> float:
    """
    Compute Hollinger Game Score for a single player-game row.

    Parameters
    ----------
    row : pandas Series or dict
        Must contain box-score columns: pts, fgm, fga, ftm, fta,
        oreb, dreb, ast, stl, blk, tov, pf (case-insensitive).

    Returns
    -------
    float
        Hollinger Game Score.
    """
    g = {k.lower(): row[k] for k in row.keys() if k.lower() in REQUIRED_BOX_SCORE_COLS}

    missing = set(REQUIRED_BOX_SCORE_COLS) - set(g.keys())
    if missing:
        raise KeyError(f"Missing required box-score columns: {missing}")

    return (
        g["pts"]
        + 0.4 * g["fgm"]
        - 0.7 * g["fga"]
        - 0.4 * (g["fta"] - g["ftm"])
        + 0.7 * g["oreb"]
        + 0.3 * g["dreb"]
        + g["stl"]
        + 0.7 * g["ast"]
        + 0.7 * g["blk"]
        - 0.4 * g["pf"]
        - g["tov"]
    )


def add_game_score_column(df: pd.DataFrame, col_name: str = "game_score") -> pd.DataFrame:
    """
    Add a game_score column to a DataFrame of player-game rows.

    Vectorized for performance — much faster than apply(compute_game_score).
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_BOX_SCORE_COLS if c not in cols_lower]
    if missing:
        raise KeyError(f"Missing required box-score columns: {missing}")

    g = {k: df[cols_lower[k]].astype(float) for k in REQUIRED_BOX_SCORE_COLS}

    df[col_name] = (
        g["pts"]
        + 0.4 * g["fgm"]
        - 0.7 * g["fga"]
        - 0.4 * (g["fta"] - g["ftm"])
        + 0.7 * g["oreb"]
        + 0.3 * g["dreb"]
        + g["stl"]
        + 0.7 * g["ast"]
        + 0.7 * g["blk"]
        - 0.4 * g["pf"]
        - g["tov"]
    )
    return df
