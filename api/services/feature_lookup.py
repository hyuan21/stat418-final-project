"""
At inference time, build a 45-feature vector from a handful of simple
user-supplied inputs (player_id, season, opponent_team_id, ...) by looking
up the corresponding rows in the lookup parquet tables loaded at startup.

This keeps the API stateless and the request payload tiny.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class PlayerNotFoundError(LookupError):
    """Raised when the requested (player, season) has no row in the dataset."""


class FeatureLookup:
    """In-memory feature lookup service."""

    def __init__(self, artifacts_dir: str | Path):
        self.dir = Path(artifacts_dir)
        # Lazily loaded so unit tests can stub these without disk I/O
        self._playoff_features: pd.DataFrame | None = None
        self._player_features: pd.DataFrame | None = None
        self._team_defense: pd.DataFrame | None = None
        self._player_history: pd.DataFrame | None = None
        self._feature_columns: list[str] | None = None
        self._players_meta: pd.DataFrame | None = None
        self._teams_meta: pd.DataFrame | None = None
        self._league_norms: pd.DataFrame | None = None

    # --- loaders ---
    def load(self) -> None:
        def _maybe(*candidates: Path) -> pd.DataFrame:
            for p in candidates:
                if p.exists():
                    return pd.read_parquet(p)
            return pd.DataFrame()

        # ★ The KEY data source: the full modeling table.
        # It contains every player-game with all 45 features (z-scored,
        # history-rolled, leakage-safe). This is what was used to train
        # the model, so at inference time we mirror the same shape.
        self._playoff_features = _maybe(self.dir / "playoff_features.parquet")

        # Auxiliary lookups used for cold-start cases and metadata
        self._player_features = _maybe(
            self.dir / "season_aggregates.parquet",
            self.dir / "player_season_features.parquet",
        )
        self._team_defense = _maybe(self.dir / "team_defense.parquet")
        self._players_meta = _maybe(self.dir / "players_meta.parquet")
        self._teams_meta = _maybe(self.dir / "teams_meta.parquet")
        self._league_norms = _maybe(self.dir / "league_norms.parquet")

        # Legacy unused but kept for backward compatibility
        self._player_history = pd.DataFrame()

        fc_path = self.dir / "feature_columns.json"
        if fc_path.exists():
            self._feature_columns = json.loads(fc_path.read_text())
        else:
            self._feature_columns = []

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns or []

    # --- queries ---
    def list_players(
        self, limit: int = 5000, playoff_season: str | None = None,
    ) -> list[dict]:
        """
        Return players the App should show in its dropdown.

        If `playoff_season` is provided (e.g. "2024-25"), restrict to players
        who actually played in that season's playoffs. Otherwise restrict to
        players with meaningful regular-season minutes in any season.

        Result is sorted by regular-season Game Score average descending,
        so stars surface first.
        """
        if self._players_meta is None or self._players_meta.empty:
            return []

        # If a specific playoff season is requested, intersect with
        # players who appear in playoff_features for that season.
        if (
            playoff_season
            and self._playoff_features is not None
            and not self._playoff_features.empty
        ):
            playoff_ids = (
                self._playoff_features[self._playoff_features["season"] == playoff_season]
                ["player_id"].unique()
            )
            if len(playoff_ids) == 0:
                return []
            # Get RS_mean for each player in that season
            season_stats = (
                self._playoff_features[
                    self._playoff_features["season"] == playoff_season
                ][["player_id", "rs_game_score_mean"]]
                .drop_duplicates("player_id")
            )
            joined = (
                self._players_meta[self._players_meta["player_id"].isin(playoff_ids)]
                .merge(season_stats, on="player_id", how="left")
                .sort_values("rs_game_score_mean", ascending=False, na_position="last")
            )
            joined["season"] = playoff_season
            cols = [c for c in ["player_id", "player_name", "season", "rs_game_score_mean"]
                    if c in joined.columns]
            return joined[cols].head(limit).to_dict(orient="records")

        # Otherwise: most recent meaningful RS_mean per player
        if (
            self._player_features is not None
            and not self._player_features.empty
            and "rs_game_score_mean" in self._player_features.columns
        ):
            most_recent = (
                self._player_features.sort_values("season")
                .drop_duplicates("player_id", keep="last")
                [["player_id", "season", "rs_game_score_mean"]]
            )
            joined = (
                self._players_meta.merge(most_recent, on="player_id", how="inner")
                .query("rs_game_score_mean >= 3.0")
                .sort_values("rs_game_score_mean", ascending=False)
            )
            cols = [c for c in ["player_id", "player_name", "season", "rs_game_score_mean"]
                    if c in joined.columns]
            return joined[cols].head(limit).to_dict(orient="records")

        cols = [c for c in ["player_id", "player_name"]
                if c in self._players_meta.columns]
        return self._players_meta[cols].head(limit).to_dict(orient="records")

    def list_matchups(self, player_id: int, season: str) -> list[dict]:
        """
        Return the opponents this player actually faced in `season`'s playoffs,
        with how many games each series went.

        Each entry: {opponent_team_id, opponent_team_name, n_games}
        Useful for constraining the App's opponent dropdown to plausible
        choices given the (player, season).
        """
        if self._playoff_features is None or self._playoff_features.empty:
            return []

        df = self._playoff_features
        matchups = df[
            (df["player_id"] == player_id) & (df["season"] == season)
        ]
        if matchups.empty:
            return []

        out = []
        for opp_abbr, group in matchups.groupby("opponent_abbr"):
            team_id = self._abbr_to_team_id(opp_abbr)
            team_name = self.get_team_name(team_id) if team_id else opp_abbr
            out.append({
                "opponent_team_id": team_id,
                "opponent_team_name": team_name,
                "opponent_abbr": opp_abbr,
                "n_games": int(group["series_game_number"].max()),
            })
        return sorted(out, key=lambda r: -r["n_games"])

    def get_actual_result(
        self, player_id: int, season: str,
        opponent_team_id: int, series_game_number: int,
    ) -> dict | None:
        """
        Return what actually happened in a specific playoff game:
        the player's actual Game Score, whether they actually underperformed,
        their RS baseline, and the decline threshold.
        """
        if self._playoff_features is None or self._playoff_features.empty:
            return None

        opp_abbr = self._team_id_to_abbr(opponent_team_id)
        df = self._playoff_features
        row = df[
            (df["player_id"] == player_id)
            & (df["season"] == season)
            & (df["series_game_number"] == series_game_number)
        ]
        if opp_abbr and "opponent_abbr" in df.columns:
            row_with_opp = row[row["opponent_abbr"] == opp_abbr]
            if not row_with_opp.empty:
                row = row_with_opp
        if row.empty:
            return None

        r = row.iloc[0]
        rs_mean = float(r.get("rs_game_score_mean", float("nan")))
        decline_thresh = float(r.get("decline_threshold_gs", float("nan")))
        game_score = float(r.get("game_score", float("nan")))
        y = int(r.get("y_underperform", 0))
        return {
            "actual_game_score": None if pd.isna(game_score) else game_score,
            "actual_underperformed": bool(y),
            "regular_season_gs_avg": None if pd.isna(rs_mean) else rs_mean,
            "decline_threshold_gs": None if pd.isna(decline_thresh) else decline_thresh,
            "game_date": str(r.get("game_date", "")),
        }

    def _abbr_to_team_id(self, abbr: str | None) -> int | None:
        if not abbr or self._teams_meta is None or self._teams_meta.empty:
            return None
        # Try a direct abbreviation column first
        for c in ("team_abbreviation", "abbreviation"):
            if c in self._teams_meta.columns:
                row = self._teams_meta[self._teams_meta[c] == abbr]
                if not row.empty:
                    return int(row.iloc[0]["team_id"])
        # Fall back to the static name->abbr mapping (inverted)
        from src.data.build_features import _TEAM_NAME_TO_ABBR
        for name, a in _TEAM_NAME_TO_ABBR.items():
            if a == abbr:
                row = self._teams_meta[self._teams_meta.get("team_name", "") == name]
                if not row.empty:
                    return int(row.iloc[0]["team_id"])
        return None

    def list_teams(self) -> list[dict]:
        if self._teams_meta is None or self._teams_meta.empty:
            return []
        cols = [c for c in ["team_id", "team_name", "team_abbreviation"]
                if c in self._teams_meta.columns]
        return self._teams_meta[cols].drop_duplicates().to_dict(orient="records")

    def get_player_name(self, player_id: int) -> str:
        if self._players_meta is None or self._players_meta.empty:
            return f"player_{player_id}"
        row = self._players_meta.query("player_id == @player_id")
        if row.empty:
            return f"player_{player_id}"
        return str(row.iloc[0].get("player_name", f"player_{player_id}"))

    def get_team_name(self, team_id: int) -> str:
        if self._teams_meta is None or self._teams_meta.empty:
            return f"team_{team_id}"
        row = self._teams_meta.query("team_id == @team_id")
        if row.empty:
            return f"team_{team_id}"
        return str(row.iloc[0].get("team_name", f"team_{team_id}"))

    def build_feature_vector(
        self,
        player_id: int,
        season: str,
        opponent_team_id: int,
        series_game_number: int,
        is_home: bool,
        series_score: str,
        feature_overrides: dict[str, float] | None = None,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        Assemble the full feature vector for one prediction.

        Strategy:
          1. Find a representative row for (player_id, season) for the
             player's *static* features (RS z-scores, career history).
          2. Override the context fields the user actually supplied.
          3. Recompute series-dependent dynamic features (last_game_gs,
             gs_in_series_so_far_mean, last_game_underperformed) from the
             actual prior games of THIS specific (player, season, opponent)
             matchup, so they correctly reflect the user-chosen game number.
          4. Override opponent features if the user picked a different team.
        """
        # --- 1. Find a base row for static features ---
        base_row = self._find_base_row(player_id, season, opponent_team_id)
        if base_row is None:
            raise PlayerNotFoundError(
                f"No playoff data for player_id={player_id} in or before "
                f"season={season}. The model can only predict for players "
                f"with playoff history in our 2003-2025 training window."
            )

        # Start from the base row's feature values
        merged = {
            c: base_row.get(c) for c in self.feature_columns
            if c in base_row.index
        }

        # --- 2. Override context features the user actually supplied ---
        try:
            wins, losses = (int(x) for x in series_score.split("-"))
        except (ValueError, AttributeError):
            wins, losses = 0, 0

        context_overrides = {
            "series_game_number": series_game_number,
            "is_home": int(bool(is_home)),
            "series_wins": wins,
            "series_losses": losses,
            "is_elimination_game": int(losses >= 3),
            "is_closeout_game": int(wins >= 3),
        }
        for k, v in context_overrides.items():
            if k in self.feature_columns:
                merged[k] = v

        # --- 3. Recompute series-momentum features that depend on the
        #        actual game number the user picked ---
        rs_mean = base_row.get("rs_game_score_mean", np.nan)
        decline_thresh = base_row.get("decline_threshold_gs", np.nan)
        if pd.isna(decline_thresh) and pd.notna(rs_mean):
            decline_thresh = max(rs_mean * 0.20, 1.0)
        momentum = self._compute_series_momentum(
            player_id, season, opponent_team_id,
            series_game_number, rs_mean, decline_thresh,
        )
        for k, v in momentum.items():
            if k in self.feature_columns:
                merged[k] = v

        # --- 4. If user picked a different opponent than the base row,
        #        swap in that opponent's defensive stats ---
        new_opp_row = self._lookup_team(opponent_team_id, season)
        if new_opp_row:
            for k, v in new_opp_row.items():
                if k in self.feature_columns and pd.notna(v):
                    merged[k] = v

        # --- 5. Apply user-supplied feature overrides (Advanced mode) ---
        if feature_overrides:
            for k, v in feature_overrides.items():
                if k in self.feature_columns:
                    merged[k] = v

        # --- 6. Build the final ordered vector ---
        vec = pd.Series(
            {c: merged.get(c, np.nan) for c in self.feature_columns},
            dtype=float,
        )

        # --- 6. Context dict for the API response ---
        rs_avg = base_row.get("rs_game_score_mean", np.nan)
        decline_thresh = base_row.get("decline_threshold_gs", np.nan)
        ctx = {
            "player_name": self.get_player_name(player_id),
            "opponent_name": self.get_team_name(opponent_team_id),
            "season": season,
            "regular_season_gs_avg": (
                float(rs_avg) if pd.notna(rs_avg) else None
            ),
            "decline_threshold_gs": (
                float(decline_thresh) if pd.notna(decline_thresh) else None
            ),
        }
        return vec, ctx

    def _find_base_row(
        self, player_id: int, season: str, opponent_team_id: int | None = None,
    ) -> "pd.Series | None":
        """
        Return one real row from playoff_features for (player, season).

        Preference order:
          1. Any row with (player_id, season)
          2. Player's most recent row at or before `season`
        """
        if self._playoff_features is None or self._playoff_features.empty:
            return None

        df = self._playoff_features

        # Try (player, season)
        same_ps = df[(df["player_id"] == player_id) & (df["season"] == season)]
        if not same_ps.empty:
            return same_ps.iloc[0]

        # Fall back to most recent row for this player at or before `season`
        same_p = df[(df["player_id"] == player_id) & (df["season"] <= season)]
        if not same_p.empty:
            return same_p.sort_values("season").iloc[-1]

        return None

    def _compute_series_momentum(
        self,
        player_id: int,
        season: str,
        opponent_team_id: int,
        series_game_number: int,
        rs_mean: float | None,
        decline_threshold: float | None,
    ) -> dict[str, float]:
        """
        Compute series-momentum features for a hypothetical game N of a
        series, using only games 1..N-1 from the actual playoff data.

        This is what makes different series_game_number values produce
        different predictions: features like `last_game_gs` and
        `gs_in_series_so_far_mean` are recomputed from real prior games
        in that specific (player, season, opponent) matchup.
        """
        if (
            self._playoff_features is None
            or self._playoff_features.empty
            or series_game_number <= 1
        ):
            # Game 1: no prior games in the series, all momentum features NaN
            return {
                "last_game_gs": np.nan,
                "last_game_underperformed": np.nan,
                "gs_in_series_so_far_mean": np.nan,
            }

        # Find prior games in this series. We need to map opponent_team_id to
        # the abbreviation that playoff_features uses.
        opp_abbr = self._team_id_to_abbr(opponent_team_id)

        df = self._playoff_features
        prior = df[
            (df["player_id"] == player_id)
            & (df["season"] == season)
            & (df["series_game_number"] < series_game_number)
        ]
        if opp_abbr and "opponent_abbr" in df.columns:
            prior_matchup = prior[prior["opponent_abbr"] == opp_abbr]
            # Use the specific matchup if we have it; else use any series
            # in that season (fallback for matchups never actually played).
            if not prior_matchup.empty:
                prior = prior_matchup

        if prior.empty:
            return {
                "last_game_gs": np.nan,
                "last_game_underperformed": np.nan,
                "gs_in_series_so_far_mean": np.nan,
            }

        prior = prior.sort_values("series_game_number")
        last_row = prior.iloc[-1]
        last_gs = float(last_row["game_score"])
        gs_so_far_mean = float(prior["game_score"].mean())

        # Was the last game an underperform?
        if pd.notna(rs_mean) and decline_threshold is not None:
            last_under = int(last_gs < rs_mean - decline_threshold)
        else:
            last_under = 0

        return {
            "last_game_gs": last_gs,
            "last_game_underperformed": float(last_under),
            "gs_in_series_so_far_mean": gs_so_far_mean,
        }

    def _team_id_to_abbr(self, team_id: int) -> str | None:
        """Resolve team_id -> 3-letter abbreviation using teams_meta."""
        if self._teams_meta is None or self._teams_meta.empty:
            return None
        row = self._teams_meta.query("team_id == @team_id")
        if row.empty:
            return None
        # Try common abbreviation column names
        for c in ("team_abbreviation", "abbreviation"):
            if c in row.columns:
                v = row.iloc[0][c]
                if isinstance(v, str) and v:
                    return v
        # Fallback: derive from team_name via the static mapping in build_features
        from src.data.build_features import _TEAM_NAME_TO_ABBR
        name = row.iloc[0].get("team_name")
        return _TEAM_NAME_TO_ABBR.get(name)

    # --- private lookup helpers ---
    def _lookup_player(self, pid: int, season: str) -> dict:
        if self._player_features is None or self._player_features.empty:
            return {}
        row = self._player_features.query("player_id == @pid and season == @season")
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def _lookup_history(self, pid: int, season: str) -> dict:
        if self._player_history is None or self._player_history.empty:
            return {}
        # Take the latest history row STRICTLY before the requested season
        rows = self._player_history.query(
            "player_id == @pid and season < @season"
        ).sort_values("season")
        if rows.empty:
            return {}
        return rows.iloc[-1].to_dict()

    def _lookup_team(self, team_id: int, season: str) -> dict:
        if self._team_defense is None or self._team_defense.empty:
            return {}
        row = self._team_defense.query("team_id == @team_id and season == @season")
        if row.empty:
            return {}
        out = row.iloc[0].to_dict()
        # Prefix each col with opp_ if not already, so it lines up with model features
        prefixed = {}
        for k, v in out.items():
            if k in ("team_id", "team_name", "season"):
                continue
            key = k if k.startswith("opp_") else f"opp_{k}"
            prefixed[key] = v
        return prefixed
