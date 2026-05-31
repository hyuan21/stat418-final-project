"""Smoke tests for helper functions in src/data/."""
import pytest

from src.data.collect_game_logs import iter_seasons
from src.data.parse_raw import _parse_matchup


def test_iter_seasons_basic():
    seasons = list(iter_seasons("2003-04", "2005-06"))
    assert seasons == ["2003-04", "2004-05", "2005-06"]


def test_iter_seasons_single_year():
    assert list(iter_seasons("2024-25", "2024-25")) == ["2024-25"]


def test_iter_seasons_22_year_span():
    seasons = list(iter_seasons("2003-04", "2024-25"))
    assert len(seasons) == 22
    assert seasons[0] == "2003-04"
    assert seasons[-1] == "2024-25"
    # Y2K rollover: 1999-00, 2009-10 — ensure two-digit padding is right
    seasons_check = list(iter_seasons("2009-10", "2010-11"))
    assert seasons_check == ["2009-10", "2010-11"]


def test_parse_matchup_home():
    opp, is_home = _parse_matchup("LAL vs. BOS")
    assert opp == "BOS"
    assert is_home is True


def test_parse_matchup_away():
    opp, is_home = _parse_matchup("LAL @ BOS")
    assert opp == "BOS"
    assert is_home is False


def test_parse_matchup_malformed():
    opp, is_home = _parse_matchup("nonsense")
    assert opp == ""
    assert is_home is True  # safe fallback
