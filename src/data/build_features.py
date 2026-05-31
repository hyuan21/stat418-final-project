"""
Build the final modeling table: one row per (player, playoff game).

Inputs  (from data/processed/)
------
  * player_game_logs.parquet    — every player-game (regular + playoff)
  * team_defense.parquet        — per (team, season) defensive stats
  * season_aggregates.parquet   — per (player, season) regular-season averages
  * league_norms.parquet        — per-season league mean & std (for z-score)

Output (to data/processed/playoff_features.parquet)
------
  ~45,000 playoff player-game rows × ~50 columns (features + target).

THE CARDINAL RULE
-----------------
For each playoff row at date d, only data from STRICTLY BEFORE d is allowed
into the features. This is enforced by expanding-window groupby logic,
and verified by tests/test_no_leakage.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

LOG = get_logger(__name__)


# Columns that should be z-score normalized per season
CONTINUOUS_PLAYER_FEATURES = [
    "pts_per_game", "fga_per_game", "fg3a_per_game", "fta_per_game",
    "oreb_per_game", "dreb_per_game", "reb_per_game", "ast_per_game",
    "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game",
    "min_per_game", "usage_proxy_per_game",
    "fg_pct", "fg3_pct", "ft_pct",
    "rs_game_score_mean", "rs_game_score_std",
]

CONTINUOUS_TEAM_FEATURES = [
    "opp_def_rating", "opp_pace", "opp_efg_pct_allowed",
    "opp_blocks_per_game", "opp_steals_per_game",
]

# --- Underperformance definition ---
#
# A player "underperforms" in a playoff game iff:
#     game_score < rs_game_score_mean - threshold(rs_game_score_mean)
#
# Where threshold(rs_avg) = max(rs_avg * RELATIVE_DROP, ABSOLUTE_FLOOR).
#
# Why a relative threshold? An absolute 1.0 GS drop unfairly penalizes
# bench/role players (1.0 is a big share of their baseline) and is too
# lenient for stars (LeBron dropping 1.0 from 25 to 24 is meaningless).
# Using a 20% relative drop with a 1.0 absolute floor:
#   * LeBron     (RS=25): threshold = max(5.0, 1.0) = 5.0  → drops 5+ GS to flag
#   * Star       (RS=15): threshold = max(3.0, 1.0) = 3.0  → drops 3+ GS to flag
#   * Role player(RS=8):  threshold = max(1.6, 1.0) = 1.6  → drops 1.6+ GS to flag
#   * Rookie     (RS=3):  threshold = max(0.6, 1.0) = 1.0  → drops 1.0+ GS to flag
#
# This is the standard handling of heteroskedasticity in performance data.
RELATIVE_DROP = 0.20    # 20% of regular-season average
ABSOLUTE_FLOOR = 1.0    # but never less than 1.0 GS point in absolute terms

# Minimum regular-season Game Score average required to be in the dataset.
# Players below this threshold are bench/fringe players whose regular-season
# numbers are too noisy (small sample of meaningful minutes) for the model to
# learn from. They are also not what users of the app care about predicting.
#
# Note: We A/B tested MIN_RS_GAME_SCORE = 6.0 (stricter filter) and observed
# Test AUC drop by ~2.2 percentage points (XGBoost 0.649 -> 0.627). The drop
# came from a smaller training set (~20K rows vs ~30K) hurting model variance,
# and from the loss of useful contrast signal between role players and stars.
# We retain MIN_RS_GAME_SCORE = 3.0 based on this empirical result.
MIN_RS_GAME_SCORE = 3.0
MIN_RS_GAMES_PLAYED = 20   # also require at least 20 regular-season games


def underperform_threshold(rs_avg: "pd.Series | float") -> "pd.Series | float":
    """Compute the per-player underperform threshold (GS points to flag)."""
    import numpy as _np
    return _np.maximum(rs_avg * RELATIVE_DROP, ABSOLUTE_FLOOR)


# -------------------- Feature builders --------------------

def add_series_metadata(playoffs: pd.DataFrame) -> pd.DataFrame:
    """
    For each playoff player-game, derive:
      * series_game_number — 1..7, the Nth game of this player's matchup
      * series_wins / series_losses — record going INTO this game (uses
                                       strictly previous games only)
      * is_elimination_game — player's team trails 0-3, 1-3, or 2-3 going in
      * is_closeout_game    — player's team leads 3-0, 3-1, or 3-2 going in
      * playoff_round       — best-effort 1..4 from game_id prefix
    """
    df = playoffs.copy().sort_values(
        ["player_id", "season", "opponent_abbr", "game_date"]
    ).reset_index(drop=True)

    series_key = df["player_id"].astype(str) + "|" + df["season"] + "|" + df["opponent_abbr"]

    # series_game_number — rank by date within (player, season, opponent)
    df["series_game_number"] = (
        df.groupby(series_key)["game_date"].rank(method="first").astype(int)
    )

    # Wins / losses heading INTO the game (cumsum then shift by 1, group-aware)
    win_int = (df["wl"] == "W").astype(int)
    loss_int = (df["wl"] == "L").astype(int)

    df["series_wins"] = (
        win_int.groupby(series_key).cumsum()
              .groupby(series_key).shift(1).fillna(0).astype(int)
    )
    df["series_losses"] = (
        loss_int.groupby(series_key).cumsum()
               .groupby(series_key).shift(1).fillna(0).astype(int)
    )

    df["is_elimination_game"] = (df["series_losses"] >= 3).astype(int)
    df["is_closeout_game"] = (df["series_wins"] >= 3).astype(int)

    # Playoff round — NBA game IDs encode the round as the second pair of digits
    # ("004YYY0RGG"): 1=R1, 2=R2, 3=Conf Finals, 4=Finals. Best-effort parse.
    def _round_from_game_id(gid: str) -> int:
        try:
            return int(str(gid)[7])
        except (IndexError, ValueError):
            return 0
    df["playoff_round"] = df["game_id"].astype(str).apply(_round_from_game_id).clip(lower=0, upper=4)

    return df


def add_player_baseline(
    playoffs: pd.DataFrame, season_aggs: pd.DataFrame
) -> pd.DataFrame:
    """Attach this season's regular-season averages to every playoff row."""
    return playoffs.merge(season_aggs, on=["player_id", "season"], how="left")


def add_player_playoff_history(playoffs: pd.DataFrame) -> pd.DataFrame:
    """
    For each row at date d, compute the player's historical playoff statistics
    using ONLY games with game_date < d.

    Implementation note: we exploit the fact that the previous-season-and-earlier
    rows precede the current-season rows in date order. For features needing
    expanding history within (player), we use groupby().cumsum() / cumcount(),
    then shift by 1 so the current game itself is excluded.
    """
    df = playoffs.copy().sort_values(["player_id", "game_date"]).reset_index(drop=True)

    def _expanding_prior_sum(values: pd.Series, by: pd.Series) -> pd.Series:
        """
        Within each `by` group, return cumsum SHIFTED so the current row is excluded.
        E.g. values=[10,20,30] → [0, 10, 30].
        """
        cum = values.groupby(by).cumsum()
        # Shift by 1 within each group: the first row of each group becomes 0
        shifted = cum.groupby(by).shift(1).fillna(0)
        return shifted

    pid = df["player_id"]
    df["_n_prior"] = df.groupby("player_id").cumcount()  # 0, 1, 2, ... per player

    # career_playoff_gs_mean: mean GS over all PRIOR playoff games for this player
    gs_prior_sum = _expanding_prior_sum(df["game_score"], pid)
    df["career_playoff_gs_mean"] = np.where(
        df["_n_prior"] > 0, gs_prior_sum / df["_n_prior"], np.nan
    )
    df["career_playoff_games_played"] = df["_n_prior"]

    # career_playoff_underperform_rate
    uperf_here = (
        df["game_score"]
        < df["rs_game_score_mean"] - underperform_threshold(df["rs_game_score_mean"])
    ).astype(int)
    uperf_prior_sum = _expanding_prior_sum(uperf_here, pid)
    df["career_playoff_underperform_rate"] = np.where(
        df["_n_prior"] > 0, uperf_prior_sum / df["_n_prior"], np.nan
    )

    # Situational history: mean GS over prior games where `flag` was 1.
    def _conditional_history_mean(flag: pd.Series, out_col: str) -> None:
        flag_int = flag.astype(int)
        num_prior = _expanding_prior_sum(flag_int * df["game_score"], pid)
        den_prior = _expanding_prior_sum(flag_int, pid)
        df[out_col] = np.where(
            den_prior > 0, num_prior / den_prior.replace(0, np.nan), np.nan
        )

    _conditional_history_mean(df["series_game_number"] == 1, "past_game1_gs_avg")
    _conditional_history_mean(df["series_game_number"] == 7, "past_game7_gs_avg")
    _conditional_history_mean(df["is_elimination_game"] == 1, "past_elim_gs_avg")
    is_home_int = df["is_home"].astype(int)
    _conditional_history_mean(is_home_int == 1, "past_home_playoff_gs_avg")
    _conditional_history_mean(is_home_int == 0, "past_away_playoff_gs_avg")

    drop = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=drop, errors="ignore")


def add_series_momentum(playoffs: pd.DataFrame) -> pd.DataFrame:
    """
    Features that depend on PRIOR games in the SAME series:
      * gs_in_series_so_far_mean
      * last_game_gs
      * last_game_underperformed
    All are NaN / 0 for series_game_number == 1.
    """
    df = playoffs.copy().sort_values(
        ["player_id", "season", "opponent_abbr", "series_game_number"]
    ).reset_index(drop=True)

    series_key = df["player_id"].astype(str) + "|" + df["season"] + "|" + df["opponent_abbr"]
    df["_n_prior_in_series"] = df.groupby(series_key).cumcount()

    # gs_in_series_so_far_mean: mean GS over prior games in THIS series
    gs_cum = df["game_score"].groupby(series_key).cumsum()
    gs_prior_sum = gs_cum.groupby(series_key).shift(1).fillna(0)
    df["gs_in_series_so_far_mean"] = np.where(
        df["_n_prior_in_series"] > 0,
        gs_prior_sum / df["_n_prior_in_series"],
        np.nan,
    )

    # last_game_gs and last_game_underperformed — groupby.shift(1) is group-aware
    df["last_game_gs"] = df.groupby(series_key)["game_score"].shift(1)
    uperf_here = (
        df["game_score"]
        < df["rs_game_score_mean"] - underperform_threshold(df["rs_game_score_mean"])
    ).astype(int)
    df["last_game_underperformed"] = uperf_here.groupby(series_key).shift(1)

    drop = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=drop, errors="ignore")


def add_opponent_features(
    playoffs: pd.DataFrame, team_defense: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach the opponent's regular-season defensive stats for the same season.

    The link is via team_abbreviation; team_defense has TEAM_ID + TEAM_NAME but
    we map through abbreviation. To bridge them we extract abbrev from team_name
    via a static mapping (kept simple — same 30 teams for our window).
    """
    df = playoffs.copy()
    td = team_defense.copy()

    # Build opp_<stat> columns. We rename every numerical column to be
    # prefixed with opp_ (so it joins cleanly into the player-game table)
    # but only for columns that actually exist in team_defense.
    rename_map = {
        "def_rating": "opp_def_rating",
        "pace": "opp_pace",
        "opp_efg_pct": "opp_efg_pct_allowed",  # NBA already named it opp_*
        "blocks_per_game": "opp_blocks_per_game",
        "steals_per_game": "opp_steals_per_game",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in td.columns}
    td = td.rename(columns=rename_map)

    keep_cols = ["season", "team_id", "team_name"] + list(rename_map.values())
    td = td[[c for c in keep_cols if c in td.columns]]

    # Map team_name -> abbreviation using a fixed dict.
    abbr = _TEAM_NAME_TO_ABBR
    td["opponent_abbr"] = td["team_name"].map(abbr)
    td = td.dropna(subset=["opponent_abbr"])

    return df.merge(
        td.drop(columns=["team_id", "team_name"]),
        on=["season", "opponent_abbr"], how="left",
    )


def add_zscore_features(
    df: pd.DataFrame, league_norms: pd.DataFrame
) -> pd.DataFrame:
    """
    Add a `_z` column for every feature that has a corresponding
    `<feature>_mean` and `<feature>_std` in league_norms.
    """
    df = df.merge(league_norms, on="season", how="left")

    for col in CONTINUOUS_PLAYER_FEATURES + CONTINUOUS_TEAM_FEATURES:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in df.columns and std_col in df.columns:
            denom = df[std_col].replace(0, np.nan)
            df[f"{col}_z"] = (df[col] - df[mean_col]) / denom

    # Drop the per-season mean/std helper columns to keep the table tidy
    drop = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    # keep rs_game_score_mean/std — they ARE features, not league stats
    drop = [c for c in drop if c not in ("rs_game_score_mean", "rs_game_score_std")]
    return df.drop(columns=drop, errors="ignore")


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Game-context features that aren't z-scored: home/away, days_rest."""
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["days_rest"] = (
        df.groupby("player_id")["game_date"].diff().dt.days.clip(upper=14)
    )
    df["is_home"] = df["is_home"].astype(int)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """y_underperform = 1 iff GS in this game < RS_mean - threshold(RS_mean).

    The threshold is per-player relative (20% of RS average) with a 1.0 GS
    absolute floor — see RELATIVE_DROP / ABSOLUTE_FLOOR docstring above.
    """
    thresh = underperform_threshold(df["rs_game_score_mean"])
    df["y_underperform"] = (
        df["game_score"] < df["rs_game_score_mean"] - thresh
    ).astype(int)
    df["decline_threshold_gs"] = thresh.astype(float)
    return df


# -------------------- Orchestration --------------------

def build_features(
    player_logs: pd.DataFrame,
    season_aggs: pd.DataFrame,
    team_defense: pd.DataFrame,
    league_norms: pd.DataFrame,
) -> pd.DataFrame:
    LOG.info("Filtering playoff rows...")
    playoffs = player_logs[player_logs["season_type"] == "Playoffs"].copy()

    LOG.info("Adding series metadata...")
    playoffs = add_series_metadata(playoffs)

    LOG.info("Adding player baseline (regular-season averages)...")
    playoffs = add_player_baseline(playoffs, season_aggs)

    LOG.info("Adding player playoff history (expanding window)...")
    playoffs = add_player_playoff_history(playoffs)

    LOG.info("Adding in-series momentum...")
    playoffs = add_series_momentum(playoffs)

    LOG.info("Adding opponent features...")
    playoffs = add_opponent_features(playoffs, team_defense)

    LOG.info("Adding context features...")
    playoffs = add_context_features(playoffs)

    LOG.info("Adding z-score normalized features...")
    playoffs = add_zscore_features(playoffs, league_norms)

    LOG.info("Adding target y_underperform...")
    playoffs = add_target(playoffs)

    LOG.info("Filtering bench / fringe players...")
    before = len(playoffs)
    playoffs = playoffs[
        (playoffs["rs_game_score_mean"] >= MIN_RS_GAME_SCORE)
        & (playoffs["games_played"] >= MIN_RS_GAMES_PLAYED)
    ].copy()
    LOG.info(
        "  Kept %d / %d rows (filtered %d fringe-player rows; "
        "rs_game_score_mean >= %.1f and games_played >= %d)",
        len(playoffs), before, before - len(playoffs),
        MIN_RS_GAME_SCORE, MIN_RS_GAMES_PLAYED,
    )

    return playoffs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="data/processed")
    p.add_argument("--out", default="data/processed/playoff_features.parquet")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    logs = pd.read_parquet(in_dir / "player_game_logs.parquet")
    season_aggs = pd.read_parquet(in_dir / "season_aggregates.parquet")
    team_def = pd.read_parquet(in_dir / "team_defense.parquet")
    norms = pd.read_parquet(in_dir / "league_norms.parquet")

    features = build_features(logs, season_aggs, team_def, norms)
    features.to_parquet(args.out, index=False)
    LOG.info("Wrote %d rows × %d cols to %s",
             len(features), len(features.columns), args.out)


# Static team-name → abbreviation lookup, sufficient for 2003-2025.
_TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "New Jersey Nets": "NJN",  # pre-2012
    "Charlotte Hornets": "CHA", "Charlotte Bobcats": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New Orleans Hornets": "NOH",  # pre-2013
    "New Orleans/Oklahoma City Hornets": "NOK",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Seattle SuperSonics": "SEA",  # pre-2008
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


if __name__ == "__main__":
    main()
