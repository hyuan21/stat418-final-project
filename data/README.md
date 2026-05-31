# Data Directory

This directory holds all data used by the project. **Raw data is not committed to git** (see `.gitignore`); only processed Parquet files small enough to ship are tracked.

## Structure

```
data/
├── raw/                              # ❌ NOT in git — recreated by scraping
│   ├── player_game_logs/
│   │   ├── regular_season/
│   │   │   └── {season}_{player_id}.json
│   │   └── playoffs/
│   │       └── {season}_{player_id}.json
│   ├── team_stats/
│   │   └── {season}_team_stats.json
│   └── players/
│       └── all_players.json
│
├── processed/                        # ✅ Built by build_features.py
│   ├── player_game_logs.parquet      # All player-game rows (regular + playoff)
│   ├── season_aggregates.parquet     # Per-player per-season stats
│   ├── team_defense.parquet          # Per-team per-season defensive stats
│   ├── league_norms.parquet          # Per-season league averages + std (for z-score)
│   └── playoff_features.parquet      # ★ Final modeling table (one row per playoff player-game)
│
└── lookup/                           # ✅ Small reference tables
    ├── players.parquet               # player_id → name, position, etc.
    └── teams.parquet                 # team_id → name, abbreviation
```

## Data Source

All data is collected from the official [NBA Stats API](https://www.nba.com/stats) via the [`nba_api`](https://github.com/swar/nba_api) Python wrapper.

**Endpoints used:**
- `PlayerGameLog` — per-player game-by-game box scores
- `LeagueDashTeamStats` — team-level defensive ratings, pace, etc.
- `CommonAllPlayers` — player roster lookup

## Coverage

| Field | Range |
|-------|-------|
| Seasons | 2003-04 through 2024-25 (22 seasons) |
| Player-game rows (regular + playoff) | ~280,000 |
| Player-game rows (playoff only — modeling set) | ~45,000-50,000 |
| Unique players | ~3,500 |
| Teams | 30 |

## How to Reproduce

```bash
# 1. Scrape (slow — ~7 hours due to rate limiting)
python -m src.data.collect_game_logs --start-season 2003-04 --end-season 2024-25

# 2. Build features
python -m src.data.build_features
```

See [`../src/data/README.md`](../src/data/README.md) for details on the pipeline.

## Schemas

### `player_game_logs.parquet`
| Column | Type | Description |
|--------|------|-------------|
| player_id | int | NBA player ID |
| player_name | str | Display name |
| game_id | str | Unique game identifier |
| game_date | date | Date of game |
| season | str | Season string (e.g., "2023-24") |
| season_type | str | "Regular Season" or "Playoffs" |
| team_id | int | Player's team |
| opponent_team_id | int | Opponent |
| matchup | str | Raw matchup string (e.g., "LAL vs. BOS") |
| is_home | bool | Whether the player was at home |
| min | float | Minutes played |
| pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus | float | Box score |
| game_score | float | Hollinger Game Score (computed) |

### `playoff_features.parquet` (modeling table)
See [`src/data/build_features.py`](../src/data/build_features.py) for the full ~45 feature columns and the binary target `y_underperform`.
