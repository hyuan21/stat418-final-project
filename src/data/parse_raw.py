"""
Read all cached raw JSON files from `data/raw/` and consolidate them into
clean Parquet tables under `data/processed/`.

Outputs
-------
data/processed/
  ├── player_game_logs.parquet    # one row per (player, game), reg + playoffs
  ├── team_defense.parquet        # one row per (team, season)
  ├── season_aggregates.parquet   # one row per (player, season), reg-season averages
  └── league_norms.parquet        # one row per season, league mean & std per stat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.compute_game_score import add_game_score_column
from src.utils.logging_config import get_logger

LOG = get_logger(__name__)


# Columns we keep from the PlayerGameLog endpoint, in friendly snake_case.
GAME_LOG_RENAME = {
    "Player_ID": "player_id",
    "Game_ID": "game_id",
    "GAME_DATE": "game_date",
    "MATCHUP": "matchup",
    "WL": "wl",
    "MIN": "min",
    "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct",
    "FG3M": "fg3m", "FG3A": "fg3a", "FG3_PCT": "fg3_pct",
    "FTM": "ftm", "FTA": "fta", "FT_PCT": "ft_pct",
    "OREB": "oreb", "DREB": "dreb", "REB": "reb",
    "AST": "ast", "STL": "stl", "BLK": "blk",
    "TOV": "tov", "PF": "pf", "PTS": "pts", "PLUS_MINUS": "plus_minus",
}


def _resultset_to_df(payload: dict) -> pd.DataFrame:
    """Convert a single nba_api resultSet to a DataFrame."""
    rs = payload["resultSets"][0]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def _iter_json_files(root: Path, subpath: str) -> Iterable[Path]:
    base = root / subpath
    if not base.exists():
        return []
    return sorted(base.rglob("*.json"))


def _parse_matchup(matchup: str) -> tuple[str, bool]:
    """
    Parse the MATCHUP string.

    Examples
    --------
    "LAL vs. BOS"  -> ("BOS", True)   # home
    "LAL @ BOS"    -> ("BOS", False)  # away
    """
    if " vs. " in matchup:
        own, opp = matchup.split(" vs. ")
        return opp.strip(), True
    if " @ " in matchup:
        own, opp = matchup.split(" @ ")
        return opp.strip(), False
    return "", True  # fallback for malformed strings


def build_player_game_logs(raw_dir: Path) -> pd.DataFrame:
    """Consolidate every cached player game log into a single DataFrame."""
    rows: list[pd.DataFrame] = []

    for season_type, subpath in [
        ("Regular Season", "player_game_logs/regular_season"),
        ("Playoffs", "player_game_logs/playoffs"),
    ]:
        files = list(_iter_json_files(raw_dir, subpath))
        LOG.info("Loading %d files from %s", len(files), subpath)
        for fp in files:
            try:
                with open(fp, "r") as f:
                    payload = json.load(f)
                df = _resultset_to_df(payload)
                if df.empty:
                    continue
                # Filename pattern: {season}_{player_id}.json
                stem = fp.stem  # e.g. "2023-24_1629029"
                season, _ = stem.split("_", 1)
                df["season"] = season
                df["season_type"] = season_type
                rows.append(df)
            except Exception as e:
                LOG.warning("Failed to parse %s: %s", fp, e)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns=GAME_LOG_RENAME)

    # Type coercion
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["player_id"] = df["player_id"].astype(int)

    # Parse the matchup into opp_abbr + is_home
    parsed = df["matchup"].astype(str).apply(_parse_matchup)
    df["opponent_abbr"] = parsed.apply(lambda x: x[0])
    df["is_home"] = parsed.apply(lambda x: x[1])

    # Hollinger Game Score
    df = add_game_score_column(df, col_name="game_score")

    # Final column order
    keep = [
        "player_id", "season", "season_type", "game_id", "game_date",
        "matchup", "opponent_abbr", "is_home", "wl", "min",
        "pts", "fgm", "fga", "fg_pct",
        "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct",
        "oreb", "dreb", "reb",
        "ast", "stl", "blk", "tov", "pf", "plus_minus",
        "game_score",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def build_team_defense(raw_dir: Path) -> pd.DataFrame:
    """Consolidate cached team Advanced stats per (team, season)."""
    advanced_files = list(_iter_json_files(raw_dir, "team_stats/regular_season"))
    base_files = list(_iter_json_files(raw_dir, "team_stats_base"))

    def stack(files: list[Path], cols_keep: list[str]) -> pd.DataFrame:
        out = []
        for fp in files:
            try:
                with open(fp, "r") as f:
                    payload = json.load(f)
                df = _resultset_to_df(payload)
                if df.empty:
                    continue
                df["season"] = fp.stem  # "2023-24"
                keep = [c for c in cols_keep if c in df.columns]
                out.append(df[["season"] + keep])
            except Exception as e:
                LOG.warning("Failed to parse %s: %s", fp, e)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    adv = stack(
        advanced_files,
        ["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE", "OPP_EFG_PCT"],
    )
    base = stack(
        base_files,
        ["TEAM_ID", "BLK", "STL"],
    )

    if adv.empty:
        return pd.DataFrame()

    adv = adv.rename(columns={
        "TEAM_ID": "team_id",
        "TEAM_NAME": "team_name",
        "DEF_RATING": "def_rating",
        "PACE": "pace",
        "OPP_EFG_PCT": "opp_efg_pct",
    })
    if not base.empty:
        base = base.rename(columns={
            "TEAM_ID": "team_id", "BLK": "blocks_per_game", "STL": "steals_per_game",
        })
        merged = adv.merge(base, on=["season", "team_id"], how="left")
    else:
        merged = adv
        merged["blocks_per_game"] = pd.NA
        merged["steals_per_game"] = pd.NA

    merged["team_id"] = merged["team_id"].astype(int)
    return merged


def build_season_aggregates(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Per-player per-season regular-season averages used as model features."""
    rs = player_logs[player_logs["season_type"] == "Regular Season"]
    if rs.empty:
        return pd.DataFrame()

    grouped = rs.groupby(["player_id", "season"])

    agg = grouped.agg(
        games_played=("game_id", "nunique"),
        min_per_game=("min", "mean"),
        pts_per_game=("pts", "mean"),
        fga_per_game=("fga", "mean"),
        fg3a_per_game=("fg3a", "mean"),
        fta_per_game=("fta", "mean"),
        oreb_per_game=("oreb", "mean"),
        dreb_per_game=("dreb", "mean"),
        reb_per_game=("reb", "mean"),
        ast_per_game=("ast", "mean"),
        stl_per_game=("stl", "mean"),
        blk_per_game=("blk", "mean"),
        tov_per_game=("tov", "mean"),
        pf_per_game=("pf", "mean"),
        rs_game_score_mean=("game_score", "mean"),
        rs_game_score_std=("game_score", "std"),
    ).reset_index()

    # Aggregate percentages from totals (more accurate than mean of percentages)
    totals = grouped.agg(
        fgm_total=("fgm", "sum"),
        fga_total=("fga", "sum"),
        fg3m_total=("fg3m", "sum"),
        fg3a_total=("fg3a", "sum"),
        ftm_total=("ftm", "sum"),
        fta_total=("fta", "sum"),
    ).reset_index()
    totals["fg_pct"] = totals["fgm_total"] / totals["fga_total"].replace(0, pd.NA)
    totals["fg3_pct"] = totals["fg3m_total"] / totals["fg3a_total"].replace(0, pd.NA)
    totals["ft_pct"] = totals["ftm_total"] / totals["fta_total"].replace(0, pd.NA)
    # Usage proxy
    totals["usage_proxy_per_game"] = (
        totals["fga_total"] + 0.44 * totals["fta_total"]
    ) / agg["games_played"].replace(0, pd.NA)

    agg = agg.merge(
        totals[["player_id", "season", "fg_pct", "fg3_pct", "ft_pct",
                "usage_proxy_per_game"]],
        on=["player_id", "season"],
    )

    return agg


def build_league_norms(season_aggs: pd.DataFrame) -> pd.DataFrame:
    """Per-season league mean and std for each continuous feature."""
    if season_aggs.empty:
        return pd.DataFrame()

    feature_cols = [
        c for c in season_aggs.columns
        if c not in ("player_id", "season")
        and pd.api.types.is_numeric_dtype(season_aggs[c])
    ]

    norms = season_aggs.groupby("season")[feature_cols].agg(["mean", "std"])
    norms.columns = [f"{stat}_{kind}" for stat, kind in norms.columns]
    norms = norms.reset_index()
    return norms


def build_players_meta(raw_dir: Path) -> pd.DataFrame:
    """
    Build a player_id -> name lookup table from cached CommonAllPlayers JSON.

    Aggregates across all seasons; the most recent name takes precedence
    (handles players who changed display names mid-career).
    """
    files = list(_iter_json_files(raw_dir, "players"))
    if not files:
        return pd.DataFrame()

    out = []
    for fp in sorted(files):  # sort so most-recent season comes last
        try:
            with open(fp, "r") as f:
                payload = json.load(f)
            df = _resultset_to_df(payload)
            if df.empty:
                continue
            out.append(df)
        except Exception as e:
            LOG.warning("Failed to parse %s: %s", fp, e)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={
        "PERSON_ID": "player_id",
        "DISPLAY_FIRST_LAST": "player_name",
        "TEAM_ID": "team_id",
        "TEAM_NAME": "team_name",
        "TEAM_ABBREVIATION": "team_abbreviation",
    })
    keep = [c for c in ["player_id", "player_name",
                        "team_id", "team_name", "team_abbreviation"]
            if c in df.columns]
    df = df[keep].copy()

    # Keep the most recent row per player_id (since files were sorted by season,
    # the last occurrence wins)
    df["player_id"] = df["player_id"].astype(int)
    df = df.drop_duplicates(subset=["player_id"], keep="last").reset_index(drop=True)
    return df.sort_values("player_name", kind="stable").reset_index(drop=True)


def build_teams_meta(raw_dir: Path) -> pd.DataFrame:
    """
    Build a team_id -> name lookup table from cached LeagueDashTeamStats JSON.

    Aggregates across all seasons; same dedupe-by-id strategy as players_meta.
    """
    files = list(_iter_json_files(raw_dir, "team_stats/regular_season"))
    if not files:
        return pd.DataFrame()

    out = []
    for fp in sorted(files):
        try:
            with open(fp, "r") as f:
                payload = json.load(f)
            df = _resultset_to_df(payload)
            if df.empty:
                continue
            out.append(df)
        except Exception as e:
            LOG.warning("Failed to parse %s: %s", fp, e)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={
        "TEAM_ID": "team_id",
        "TEAM_NAME": "team_name",
    })
    keep = [c for c in ["team_id", "team_name"] if c in df.columns]
    df = df[keep].copy()
    df["team_id"] = df["team_id"].astype(int)
    df = df.drop_duplicates(subset=["team_id"], keep="last").reset_index(drop=True)
    return df.sort_values("team_name", kind="stable").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Building player_game_logs ...")
    logs = build_player_game_logs(raw_dir)
    if not logs.empty:
        logs.to_parquet(out_dir / "player_game_logs.parquet", index=False)
        LOG.info("Wrote %d rows to player_game_logs.parquet", len(logs))

    LOG.info("Building team_defense ...")
    team = build_team_defense(raw_dir)
    if not team.empty:
        team.to_parquet(out_dir / "team_defense.parquet", index=False)
        LOG.info("Wrote %d rows to team_defense.parquet", len(team))

    LOG.info("Building season_aggregates ...")
    season_aggs = build_season_aggregates(logs)
    if not season_aggs.empty:
        season_aggs.to_parquet(out_dir / "season_aggregates.parquet", index=False)
        LOG.info("Wrote %d rows to season_aggregates.parquet", len(season_aggs))

    LOG.info("Building league_norms ...")
    norms = build_league_norms(season_aggs)
    if not norms.empty:
        norms.to_parquet(out_dir / "league_norms.parquet", index=False)
        LOG.info("Wrote %d rows to league_norms.parquet", len(norms))

    LOG.info("Building players_meta (id -> name lookup) ...")
    players_meta = build_players_meta(raw_dir)
    if not players_meta.empty:
        players_meta.to_parquet(out_dir / "players_meta.parquet", index=False)
        LOG.info("Wrote %d rows to players_meta.parquet", len(players_meta))

    LOG.info("Building teams_meta (id -> name lookup) ...")
    teams_meta = build_teams_meta(raw_dir)
    if not teams_meta.empty:
        teams_meta.to_parquet(out_dir / "teams_meta.parquet", index=False)
        LOG.info("Wrote %d rows to teams_meta.parquet", len(teams_meta))

    # Also copy the lookup parquet files into api/artifacts/ so the Flask
    # API container can find them at startup. This is the simplest way to
    # avoid mounting `data/processed/` into the container.
    artifacts_dir = Path("api/artifacts")
    if artifacts_dir.exists():
        for fname in ("players_meta.parquet", "teams_meta.parquet",
                      "team_defense.parquet", "league_norms.parquet",
                      "season_aggregates.parquet",
                      "playoff_features.parquet"):  # critical for inference
            src_file = out_dir / fname
            if src_file.exists():
                import shutil
                shutil.copy2(src_file, artifacts_dir / fname)
                LOG.info("Copied %s -> api/artifacts/", fname)


if __name__ == "__main__":
    main()
