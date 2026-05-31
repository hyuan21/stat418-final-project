"""
The most important test in the project.

We build a synthetic playoff dataset with KNOWN history, run the feature
pipeline, then check that:

  1. `career_playoff_gs_mean` for row at date d equals the mean of game_scores
     from STRICTLY earlier dates for that player.
  2. `gs_in_series_so_far_mean` for series_game_number == 1 is NaN, and for N
     equals the mean of GS in games 1..N-1 of the same series.
  3. `last_game_gs` for series_game_number == 1 is NaN.
  4. `series_wins` going into game N equals wins in games 1..N-1.

If any of these fail, the model will silently learn from future information.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.build_features import (
    add_series_metadata,
    add_player_baseline,
    add_player_playoff_history,
    add_series_momentum,
)


def _make_synthetic_playoffs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a tiny but adversarial dataset:
      Player 1 plays a 5-game playoff series in 2022 against BOS,
      then a 5-game playoff series in 2023 against MIA.
    """
    rows = []
    # 2022 vs BOS: GS values 20, 10, 25, 15, 30
    for i, gs in enumerate([20, 10, 25, 15, 30], start=1):
        rows.append({
            "player_id": 1, "season": "2021-22", "season_type": "Playoffs",
            "game_id": f"00422{i:05d}",
            "game_date": pd.Timestamp(f"2022-04-{i*2:02d}"),
            "matchup": "MIA vs. BOS", "opponent_abbr": "BOS", "is_home": True,
            "wl": "W" if gs > 18 else "L",
            "min": 35, "pts": gs, "fgm": 5, "fga": 10, "fg_pct": 0.5,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0,
            "ftm": 0, "fta": 0, "ft_pct": 0,
            "oreb": 0, "dreb": 5, "reb": 5, "ast": 5, "stl": 0, "blk": 0,
            "tov": 0, "pf": 0, "plus_minus": 0, "game_score": gs,
        })
    # 2023 vs MIA: GS values 5, 22, 18, 28, 11
    for i, gs in enumerate([5, 22, 18, 28, 11], start=1):
        rows.append({
            "player_id": 1, "season": "2022-23", "season_type": "Playoffs",
            "game_id": f"00422{i + 10:05d}",
            "game_date": pd.Timestamp(f"2023-04-{i*2:02d}"),
            "matchup": "BOS vs. MIA", "opponent_abbr": "MIA", "is_home": True,
            "wl": "W" if gs > 18 else "L",
            "min": 35, "pts": gs, "fgm": 5, "fga": 10, "fg_pct": 0.5,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0,
            "ftm": 0, "fta": 0, "ft_pct": 0,
            "oreb": 0, "dreb": 5, "reb": 5, "ast": 5, "stl": 0, "blk": 0,
            "tov": 0, "pf": 0, "plus_minus": 0, "game_score": gs,
        })
    playoffs = pd.DataFrame(rows)

    # Player 1 had a regular-season GS mean of 18 in both 2021-22 and 2022-23.
    season_aggs = pd.DataFrame([
        {"player_id": 1, "season": "2021-22",
         "rs_game_score_mean": 18.0, "rs_game_score_std": 5.0,
         "games_played": 70, "min_per_game": 35, "pts_per_game": 20,
         "fga_per_game": 10, "fg3a_per_game": 0, "fta_per_game": 0,
         "oreb_per_game": 0, "dreb_per_game": 5, "reb_per_game": 5,
         "ast_per_game": 5, "stl_per_game": 0, "blk_per_game": 0,
         "tov_per_game": 0, "pf_per_game": 0,
         "fg_pct": 0.5, "fg3_pct": 0, "ft_pct": 0,
         "usage_proxy_per_game": 10.0},
        {"player_id": 1, "season": "2022-23",
         "rs_game_score_mean": 18.0, "rs_game_score_std": 5.0,
         "games_played": 70, "min_per_game": 35, "pts_per_game": 20,
         "fga_per_game": 10, "fg3a_per_game": 0, "fta_per_game": 0,
         "oreb_per_game": 0, "dreb_per_game": 5, "reb_per_game": 5,
         "ast_per_game": 5, "stl_per_game": 0, "blk_per_game": 0,
         "tov_per_game": 0, "pf_per_game": 0,
         "fg_pct": 0.5, "fg3_pct": 0, "ft_pct": 0,
         "usage_proxy_per_game": 10.0},
    ])
    return playoffs, season_aggs


def _run_pipeline(playoffs, season_aggs):
    df = add_series_metadata(playoffs)
    df = add_player_baseline(df, season_aggs)
    df = add_player_playoff_history(df)
    df = add_series_momentum(df)
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def test_series_game_number_within_series():
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    # First 5 rows (2022 vs BOS) should be numbered 1..5
    assert list(df.iloc[:5]["series_game_number"]) == [1, 2, 3, 4, 5]
    # Next 5 rows (2023 vs MIA) restart at 1..5
    assert list(df.iloc[5:]["series_game_number"]) == [1, 2, 3, 4, 5]


def test_career_playoff_gs_mean_uses_only_prior_games():
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    # First-ever playoff game: no history → NaN
    assert pd.isna(df.iloc[0]["career_playoff_gs_mean"])
    # 2nd ever game: mean of [20] = 20
    assert df.iloc[1]["career_playoff_gs_mean"] == pytest.approx(20.0)
    # 5th ever game (last of 2022 series): mean of [20, 10, 25, 15] = 17.5
    assert df.iloc[4]["career_playoff_gs_mean"] == pytest.approx(17.5)
    # 6th game (first of 2023 series): mean of all 5 2022 games = 20
    assert df.iloc[5]["career_playoff_gs_mean"] == pytest.approx(20.0)
    # Last game (10th overall): mean of first 9 = (20+10+25+15+30+5+22+18+28)/9
    expected = (20 + 10 + 25 + 15 + 30 + 5 + 22 + 18 + 28) / 9
    assert df.iloc[9]["career_playoff_gs_mean"] == pytest.approx(expected)


def test_gs_in_series_so_far_mean_nan_at_g1():
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    g1_rows = df[df["series_game_number"] == 1]
    assert g1_rows["gs_in_series_so_far_mean"].isna().all()


def test_gs_in_series_so_far_mean_at_game2():
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    # 2022 vs BOS series, game 2: only game 1 (GS=20) preceded
    g2_2022 = df[(df["season"] == "2021-22") & (df["series_game_number"] == 2)].iloc[0]
    assert g2_2022["gs_in_series_so_far_mean"] == pytest.approx(20.0)
    # 2022 vs BOS, game 5: mean of [20, 10, 25, 15] = 17.5
    g5_2022 = df[(df["season"] == "2021-22") & (df["series_game_number"] == 5)].iloc[0]
    assert g5_2022["gs_in_series_so_far_mean"] == pytest.approx(17.5)


def test_last_game_gs_nan_at_g1_and_correct_otherwise():
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    g1 = df[df["series_game_number"] == 1]
    assert g1["last_game_gs"].isna().all()
    # 2022 series game 2: last game GS == 20
    g2 = df[(df["season"] == "2021-22") & (df["series_game_number"] == 2)].iloc[0]
    assert g2["last_game_gs"] == pytest.approx(20.0)


def test_series_wins_into_game():
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    # In the synthetic data, "wl" depends on whether GS > 18:
    # 2022 vs BOS: GS = [20, 10, 25, 15, 30] => W L W L W
    # Going INTO each game, wins should be: 0, 1, 1, 2, 2
    series_2022 = df[df["season"] == "2021-22"].sort_values("series_game_number")
    assert list(series_2022["series_wins"]) == [0, 1, 1, 2, 2]


def test_no_negative_values():
    """series_wins and series_losses must never be negative."""
    playoffs, season_aggs = _make_synthetic_playoffs()
    df = _run_pipeline(playoffs, season_aggs)
    assert (df["series_wins"] >= 0).all()
    assert (df["series_losses"] >= 0).all()
