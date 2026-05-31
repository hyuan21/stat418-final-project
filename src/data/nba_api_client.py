"""
Wrapper around `nba_api` that adds:
  * conservative rate limiting (default: 1 request every 1.2 s)
  * exponential-backoff retry on transient failures
  * on-disk JSON caching so re-runs are resumable and reproducible

Every public method returns a Python dict (the parsed JSON) and writes the
same dict to a cache file in `cache_dir`. If the cache file already exists,
the API call is skipped — this is what makes long scrapes resumable.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from nba_api.stats.endpoints import (
    PlayerGameLog,
    LeagueDashTeamStats,
    CommonAllPlayers,
)

from src.utils.logging_config import get_logger

LOG = get_logger(__name__)


class NBAClient:
    """Cached, rate-limited NBA Stats API client."""

    def __init__(
        self,
        cache_dir: str | Path = "data/raw",
        min_request_interval_seconds: float = 1.2,
        max_retries: int = 5,
        timeout: int = 60,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_request_interval_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_time = 0.0

    # ---------- internal helpers ----------

    def _throttle(self) -> None:
        """Sleep just long enough to respect the min interval between calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_path(self, *parts: str) -> Path:
        """Build a cache file path from a sequence of safe path segments."""
        return self.cache_dir.joinpath(*parts).with_suffix(".json")

    def _load_cache(self, path: Path) -> dict | None:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                LOG.warning("Corrupt cache at %s (%s); will refetch.", path, e)
                return None
        return None

    def _save_cache(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(path)  # atomic move

    def _call_with_retry(self, fn, *args, **kwargs) -> dict:
        """Call an nba_api endpoint constructor and return its parsed JSON."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle()
                endpoint = fn(*args, timeout=self.timeout, **kwargs)
                # All nba_api endpoints expose .get_dict()
                return endpoint.get_dict()
            except Exception as e:  # nba_api raises a variety of exceptions
                last_err = e
                backoff = min(2 ** attempt, 30)
                LOG.warning(
                    "Request failed (attempt %d/%d): %s. Backing off %ds.",
                    attempt, self.max_retries, e, backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"All retries exhausted: {last_err}")

    # ---------- public API ----------

    def get_all_players(self, season: str, is_only_current_season: bool = False) -> dict:
        """
        Fetch the league-wide player roster for `season`.

        Parameters
        ----------
        season : str
            Format "YYYY-YY" (e.g. "2023-24").
        is_only_current_season : bool
            Passed through to the nba_api endpoint (set False for historical scrapes).
        """
        cache = self._cache_path("players", f"all_players_{season}")
        cached = self._load_cache(cache)
        if cached is not None:
            return cached

        data = self._call_with_retry(
            CommonAllPlayers,
            season=season,
            is_only_current_season=1 if is_only_current_season else 0,
        )
        self._save_cache(cache, data)
        LOG.info("Fetched player list for %s (%d entries).",
                 season, _row_count(data))
        return data

    def get_player_game_log(
        self,
        player_id: int,
        season: str,
        season_type: str = "Regular Season",
    ) -> dict:
        """
        Fetch a single player's game-by-game box scores for a season.

        season_type : "Regular Season" or "Playoffs".
        """
        subdir = "regular_season" if season_type == "Regular Season" else "playoffs"
        cache = self._cache_path(
            "player_game_logs", subdir, f"{season}_{player_id}"
        )
        cached = self._load_cache(cache)
        if cached is not None:
            return cached

        data = self._call_with_retry(
            PlayerGameLog,
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
        )
        self._save_cache(cache, data)
        return data

    def get_team_stats(self, season: str, season_type: str = "Regular Season") -> dict:
        """
        Fetch league-wide team statistics (defensive rating, pace, etc.).
        """
        subdir = "regular_season" if season_type == "Regular Season" else "playoffs"
        cache = self._cache_path("team_stats", subdir, f"{season}")
        cached = self._load_cache(cache)
        if cached is not None:
            return cached

        # Use "Advanced" measure type to get DefRating, Pace, etc.
        data = self._call_with_retry(
            LeagueDashTeamStats,
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        self._save_cache(cache, data)
        LOG.info("Fetched team advanced stats for %s.", season)
        return data

    def get_team_stats_base(self, season: str) -> dict:
        """Fetch base team stats (used to derive opp_blocks, opp_steals per game)."""
        cache = self._cache_path("team_stats_base", f"{season}")
        cached = self._load_cache(cache)
        if cached is not None:
            return cached
        data = self._call_with_retry(
            LeagueDashTeamStats,
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        )
        self._save_cache(cache, data)
        return data


def _row_count(nba_api_dict: dict) -> int:
    """Return the number of rows in an nba_api result dict (best-effort)."""
    try:
        return len(nba_api_dict["resultSets"][0]["rowSet"])
    except (KeyError, IndexError, TypeError):
        return 0
