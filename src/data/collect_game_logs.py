"""
Batch scraper that pulls player game logs (regular season + playoffs)
and team advanced stats for a range of NBA seasons.

Usage
-----
    python -m src.data.collect_game_logs \\
        --start-season 2003-04 \\
        --end-season 2024-25 \\
        --cache-dir data/raw

The script is fully resumable — re-running it picks up where it left off
because every successful API response is cached to disk.
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

from tqdm import tqdm

from src.data.nba_api_client import NBAClient
from src.utils.logging_config import get_logger

LOG = get_logger(__name__)


def iter_seasons(start: str, end: str) -> Iterable[str]:
    """
    Yield NBA season strings from `start` to `end` inclusive.

    A season string looks like "2003-04"; the start year increases by 1 each step.
    """
    start_year = int(start.split("-")[0])
    end_year = int(end.split("-")[0])
    for year in range(start_year, end_year + 1):
        yield f"{year}-{str(year + 1)[-2:]}"


def extract_player_ids_from_roster(roster_dict: dict) -> list[int]:
    """Pull the PERSON_ID column out of a CommonAllPlayers response."""
    result_set = roster_dict["resultSets"][0]
    headers = result_set["headers"]
    rows = result_set["rowSet"]
    pid_idx = headers.index("PERSON_ID")
    # ROSTERSTATUS == 1 means the player was active in that season
    rost_idx = headers.index("ROSTERSTATUS") if "ROSTERSTATUS" in headers else None
    ids = []
    for row in rows:
        if rost_idx is not None and row[rost_idx] != 1:
            continue
        ids.append(int(row[pid_idx]))
    return ids


def run(start: str, end: str, cache_dir: str, season_types: list[str]) -> None:
    client = NBAClient(cache_dir=cache_dir)

    seasons = list(iter_seasons(start, end))
    LOG.info("Will scrape %d seasons: %s — %s",
             len(seasons), seasons[0], seasons[-1])

    for season in seasons:
        LOG.info("=== Season %s ===", season)

        # 1. Roster
        roster = client.get_all_players(season=season)
        player_ids = extract_player_ids_from_roster(roster)
        LOG.info("Found %d active players for %s.", len(player_ids), season)

        # 2. Per-player game logs
        for season_type in season_types:
            label = "regular" if season_type == "Regular Season" else "playoffs"
            for pid in tqdm(player_ids, desc=f"{season} {label}", file=sys.stdout):
                try:
                    client.get_player_game_log(
                        player_id=pid, season=season, season_type=season_type,
                    )
                except Exception as e:
                    LOG.warning("Skipping player_id=%s season=%s type=%s: %s",
                                pid, season, season_type, e)

        # 3. Team-level advanced stats
        try:
            client.get_team_stats(season=season, season_type="Regular Season")
            client.get_team_stats_base(season=season)
        except Exception as e:
            LOG.warning("Team stats failed for %s: %s", season, e)

    LOG.info("All scrapes complete.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-season", default="2003-04",
                   help='First season (format "YYYY-YY", default 2003-04)')
    p.add_argument("--end-season", default="2024-25",
                   help='Last season inclusive (default 2024-25)')
    p.add_argument("--cache-dir", default="data/raw",
                   help="Where to store raw JSON responses (default data/raw)")
    p.add_argument(
        "--season-types", nargs="+",
        default=["Regular Season", "Playoffs"],
        choices=["Regular Season", "Playoffs"],
        help="Which season types to fetch (default: both)",
    )
    args = p.parse_args()
    run(args.start_season, args.end_season, args.cache_dir, args.season_types)


if __name__ == "__main__":
    main()
